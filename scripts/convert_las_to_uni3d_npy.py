"""Convert single-tree LAS files into Uni3D batch-extraction .npy inputs.

This script is intentionally narrow: it only prepares .npy files shaped [N, 3]
or [N, 6] for scripts/batch_extract_uni3d_embeddings.py. It does not call the
Uni3D extractor, does not require CUDA, and does not modify the main QA path.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LasArrays:
    """Raw arrays read from one LAS file."""

    xyz: np.ndarray
    rgb: np.ndarray | None
    has_rgb: bool


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert single-tree .las files to fixed-size Uni3D .npy inputs.",
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing source .las files.")
    parser.add_argument("--output-dir", required=True, help="Directory where converted .npy files are written.")
    parser.add_argument("--pattern", default="*.las", help="Glob pattern used under --input-dir.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of LAS files to convert.")
    parser.add_argument("--num-points", type=int, default=10000, help="Fixed output point count.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed.")
    parser.add_argument(
        "--include-rgb",
        action="store_true",
        help="Include LAS red/green/blue fields when all three are present.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .npy files.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        sources = discover_las_paths(input_dir=input_dir, pattern=args.pattern, limit=args.limit)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for source_las in sources:
        entry = convert_las_file(
            source_las=source_las,
            input_dir=input_dir,
            output_dir=output_dir,
            num_points=args.num_points,
            seed=args.seed,
            include_rgb=args.include_rgb,
            overwrite=args.overwrite,
        )
        entries.append(entry)
        if entry["status"] == "ok":
            print(f"OK {entry['source_las']} -> {entry['output_npy']} shape={entry['shape']}")
        else:
            print(f"FAILED {entry['source_las']}: {entry['failure_reason']}", file=sys.stderr)

    manifest_path = output_dir / "manifest.jsonl"
    write_manifest_jsonl(manifest_path, entries)
    success_count = sum(1 for entry in entries if entry["status"] == "ok")
    failure_count = len(entries) - success_count
    print(f"manifest_path: {manifest_path}")
    print(f"candidate_count: {len(entries)}")
    print(f"success_count: {success_count}")
    print(f"failure_count: {failure_count}")
    return 0 if entries and failure_count == 0 else 2


def discover_las_paths(*, input_dir: Path, pattern: str, limit: int | None) -> list[Path]:
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive when provided.")
    if not input_dir.is_dir():
        raise ValueError(f"input directory not found: {input_dir}")
    paths = sorted(path for path in input_dir.rglob(pattern) if path.is_file())
    if limit is not None:
        paths = paths[:limit]
    return paths


def convert_las_file(
    *,
    source_las: Path,
    input_dir: Path,
    output_dir: Path,
    num_points: int,
    seed: int,
    include_rgb: bool,
    overwrite: bool,
) -> dict[str, Any]:
    output_npy = output_path_for_las(source_las=source_las, input_dir=input_dir, output_dir=output_dir)
    entry: dict[str, Any] = {
        "status": "failed",
        "source_las": str(source_las),
        "output_npy": str(output_npy),
        "shape": None,
        "has_rgb": None,
        "include_rgb_requested": include_rgb,
        "num_points_before": None,
        "num_points_after": None,
        "num_points_target": num_points,
        "seed": seed,
        "failure_reason": None,
    }
    try:
        if output_npy.exists() and not overwrite:
            raise ValueError("output_exists_use_overwrite")
        arrays = read_las_arrays(source_las, include_rgb=include_rgb)
        point_count = int(arrays.xyz.shape[0])
        entry["num_points_before"] = point_count
        entry["has_rgb"] = arrays.has_rgb
        if point_count < num_points:
            raise ValueError(f"too_few_points: got {point_count}, need {num_points}")
        output_array = build_uni3d_input_array(arrays.xyz, arrays.rgb)
        sampled = sample_fixed_count(output_array, num_points=num_points, seed=seed)
        output_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_npy, sampled.astype(np.float32, copy=False))
        entry.update(
            {
                "status": "ok",
                "shape": [int(value) for value in sampled.shape],
                "num_points_after": int(sampled.shape[0]),
                "failure_reason": None,
            }
        )
    except Exception as exc:
        entry["failure_reason"] = f"{type(exc).__name__}: {exc}"
    return entry


def output_path_for_las(*, source_las: Path, input_dir: Path, output_dir: Path) -> Path:
    try:
        relative_path = source_las.relative_to(input_dir)
    except ValueError:
        relative_path = Path(source_las.name)
    return output_dir / relative_path.with_suffix(".npy")


def read_las_arrays(path: str | Path, *, include_rgb: bool) -> LasArrays:
    laspy = import_laspy()
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(str(source_path))
    las = laspy.read(source_path)
    xyz = np.column_stack(
        [
            np.asarray(las.x, dtype=np.float64),
            np.asarray(las.y, dtype=np.float64),
            np.asarray(las.z, dtype=np.float64),
        ]
    ).astype(np.float32)
    validate_xyz(xyz)

    rgb: np.ndarray | None = None
    has_rgb = False
    if include_rgb and las_has_rgb(las):
        rgb = normalize_rgb(
            np.column_stack(
                [
                    np.asarray(las.red),
                    np.asarray(las.green),
                    np.asarray(las.blue),
                ]
            )
        )
        has_rgb = True
    return LasArrays(xyz=xyz, rgb=rgb, has_rgb=has_rgb)


def import_laspy() -> Any:
    try:
        import laspy
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "laspy is required to convert .las files. Install it in the server env, "
            "for example: conda activate fa-u && pip install laspy"
        ) from exc
    return laspy


def las_has_rgb(las: Any) -> bool:
    dimension_names = set(las.point_format.dimension_names)
    return {"red", "green", "blue"}.issubset(dimension_names)


def validate_xyz(xyz: np.ndarray) -> None:
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape [N, 3], got {xyz.shape}.")
    if xyz.shape[0] == 0:
        raise ValueError("xyz contains zero points.")
    if not np.isfinite(xyz).all():
        raise ValueError("xyz contains NaN or Inf values.")


def normalize_rgb(rgb: Any) -> np.ndarray:
    array = np.asarray(rgb)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"rgb must have shape [N, 3], got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError("rgb contains NaN or Inf values.")
    scale = rgb_scale_for_array(array)
    normalized = np.asarray(array, dtype=np.float32) / np.float32(scale)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def rgb_scale_for_array(array: np.ndarray) -> float:
    if np.issubdtype(array.dtype, np.integer):
        return float(np.iinfo(array.dtype).max)
    max_value = float(np.max(array)) if array.size else 1.0
    if max_value > 255.0:
        return 65535.0
    if max_value > 1.0:
        return 255.0
    return 1.0


def build_uni3d_input_array(xyz: Any, rgb: Any | None) -> np.ndarray:
    xyz_array = np.asarray(xyz, dtype=np.float32)
    validate_xyz(xyz_array)
    if rgb is None:
        return xyz_array
    rgb_array = np.asarray(rgb, dtype=np.float32)
    if rgb_array.ndim != 2 or rgb_array.shape[1] != 3:
        raise ValueError(f"rgb must have shape [N, 3], got {rgb_array.shape}.")
    if rgb_array.shape[0] != xyz_array.shape[0]:
        raise ValueError(f"rgb point count must match xyz: got {rgb_array.shape[0]} and {xyz_array.shape[0]}.")
    if not np.isfinite(rgb_array).all():
        raise ValueError("rgb contains NaN or Inf values.")
    return np.concatenate([xyz_array, rgb_array], axis=1).astype(np.float32)


def sample_fixed_count(array: Any, *, num_points: int, seed: int) -> np.ndarray:
    if num_points <= 0:
        raise ValueError("num_points must be positive.")
    matrix = np.asarray(array, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] not in (3, 6):
        raise ValueError(f"Uni3D input array must have shape [N, 3] or [N, 6], got {matrix.shape}.")
    point_count = int(matrix.shape[0])
    if point_count < num_points:
        raise ValueError(f"too_few_points: got {point_count}, need {num_points}")
    if point_count == num_points:
        return matrix.astype(np.float32, copy=True)
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(point_count, size=num_points, replace=False))
    return matrix[indices].astype(np.float32, copy=True)


def write_manifest_jsonl(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(make_jsonable(entry), sort_keys=True))
            handle.write("\n")


def make_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
