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


@dataclass(frozen=True)
class SampledArray:
    """Fixed-size sampled array and its sampling provenance."""

    values: np.ndarray
    method: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert single-tree .las files to fixed-size Uni3D .npy inputs.",
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing source .las files.")
    parser.add_argument("--output-dir", required=True, help="Directory where converted .npy files are written.")
    parser.add_argument("--pattern", default="*.las", help="Glob pattern used under --input-dir.")
    parser.add_argument("--recursive", dest="recursive", action="store_true", default=True)
    parser.add_argument("--no-recursive", dest="recursive", action="store_false")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of LAS files to convert. 0 means all.")
    parser.add_argument("--num-points", type=int, default=10000, help="Fixed output point count.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed.")
    parser.add_argument(
        "--include-rgb",
        action="store_true",
        help="Include LAS red/green/blue fields when all three are present.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .npy files.")
    parser.add_argument(
        "--skip-existing",
        "--resume",
        dest="skip_existing",
        action="store_true",
        help="Skip existing .npy outputs and record them as skipped_existing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        sources = discover_las_paths(
            input_dir=input_dir,
            pattern=args.pattern,
            recursive=args.recursive,
            limit=normalize_limit(args.limit),
        )
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
            skip_existing=args.skip_existing,
        )
        entries.append(entry)
        if entry["status"] == "ok":
            print(f"OK {entry['source_las']} -> {entry['output_npy']} shape={entry['output_shape']}")
        elif entry["status"] == "skipped_existing":
            print(f"SKIP {entry['source_las']}: existing output {entry['output_npy']}")
        else:
            print(f"FAILED {entry['source_las']}: {entry['failure_reason']}", file=sys.stderr)

    manifest_path = output_dir / "manifest.jsonl"
    write_manifest_jsonl(manifest_path, entries)
    summary = build_summary(args=args, entries=entries, manifest_path=manifest_path)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"manifest_path: {manifest_path}")
    print(f"summary_path: {summary_path}")
    print(f"candidate_count: {summary['candidate_count']}")
    print(f"success_count: {summary['success_count']}")
    print(f"skipped_count: {summary['skipped_count']}")
    print(f"failure_count: {summary['failure_count']}")
    return 0 if entries and summary["failure_count"] == 0 else 2


def normalize_limit(limit: int | None) -> int | None:
    if limit is None or limit == 0:
        return None
    if limit < 0:
        raise ValueError("--limit must be non-negative. Use 0 or omit it for all candidates.")
    return limit


def discover_las_paths(*, input_dir: Path, pattern: str, recursive: bool, limit: int | None) -> list[Path]:
    if not input_dir.is_dir():
        raise ValueError(f"input directory not found: {input_dir}")
    globber = input_dir.rglob if recursive else input_dir.glob
    paths = sorted(path for path in globber(pattern) if path.is_file())
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
    skip_existing: bool,
) -> dict[str, Any]:
    output_npy = output_path_for_las(source_las=source_las, input_dir=input_dir, output_dir=output_dir)
    tree_id = source_las.stem
    entry: dict[str, Any] = {
        "status": "failed",
        "tree_id": tree_id,
        "source_las": str(source_las),
        "output_npy": str(output_npy),
        "output_shape": None,
        "has_rgb": None,
        "include_rgb": include_rgb,
        "input_point_count": None,
        "output_point_count": None,
        "target_point_count": num_points,
        "sampling_method": None,
        "seed": seed,
        "failure_reason": None,
    }
    try:
        if output_npy.exists() and not overwrite:
            if skip_existing:
                existing = load_existing_npy_metadata(output_npy)
                entry.update(
                    {
                        "status": "skipped_existing",
                        "output_shape": existing["output_shape"],
                        "has_rgb": existing["has_rgb"],
                        "output_point_count": existing["output_point_count"],
                        "sampling_method": "skipped_existing",
                        "failure_reason": None,
                    }
                )
                return add_legacy_manifest_aliases(entry)
            raise ValueError("output_exists_use_overwrite_or_skip_existing")
        arrays = read_las_arrays(source_las, include_rgb=include_rgb)
        point_count = int(arrays.xyz.shape[0])
        entry["input_point_count"] = point_count
        entry["has_rgb"] = arrays.has_rgb
        if point_count < num_points:
            raise ValueError(f"too_few_points: got {point_count}, need {num_points}")
        output_array = build_uni3d_input_array(arrays.xyz, arrays.rgb)
        sampled = sample_fixed_count(output_array, num_points=num_points, seed=seed)
        output_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_npy, sampled.values.astype(np.float32, copy=False))
        entry.update(
            {
                "status": "ok",
                "output_shape": [int(value) for value in sampled.values.shape],
                "output_point_count": int(sampled.values.shape[0]),
                "sampling_method": sampled.method,
                "failure_reason": None,
            }
        )
    except Exception as exc:
        entry["failure_reason"] = f"{type(exc).__name__}: {exc}"
    return add_legacy_manifest_aliases(entry)


def load_existing_npy_metadata(output_npy: Path) -> dict[str, Any]:
    array = np.load(output_npy)
    if array.ndim != 2 or array.shape[1] not in (3, 6):
        raise ValueError(f"existing output has invalid shape: {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("existing output contains NaN or Inf values")
    return {
        "output_shape": [int(value) for value in array.shape],
        "has_rgb": bool(array.shape[1] == 6),
        "output_point_count": int(array.shape[0]),
    }


def add_legacy_manifest_aliases(entry: dict[str, Any]) -> dict[str, Any]:
    entry["shape"] = entry["output_shape"]
    entry["include_rgb_requested"] = entry["include_rgb"]
    entry["num_points_before"] = entry["input_point_count"]
    entry["num_points_after"] = entry["output_point_count"]
    entry["num_points_target"] = entry["target_point_count"]
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


def sample_fixed_count(array: Any, *, num_points: int, seed: int) -> SampledArray:
    if num_points <= 0:
        raise ValueError("num_points must be positive.")
    matrix = np.asarray(array, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] not in (3, 6):
        raise ValueError(f"Uni3D input array must have shape [N, 3] or [N, 6], got {matrix.shape}.")
    point_count = int(matrix.shape[0])
    if point_count < num_points:
        raise ValueError(f"too_few_points: got {point_count}, need {num_points}")
    if point_count == num_points:
        return SampledArray(values=matrix.astype(np.float32, copy=True), method="identity")
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(point_count, size=num_points, replace=False))
    return SampledArray(
        values=matrix[indices].astype(np.float32, copy=True),
        method="deterministic_random_without_replacement",
    )


def write_manifest_jsonl(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(make_jsonable(entry), sort_keys=True))
            handle.write("\n")


def build_summary(*, args: argparse.Namespace, entries: list[dict[str, Any]], manifest_path: Path) -> dict[str, Any]:
    success_count = sum(1 for entry in entries if entry["status"] == "ok")
    skipped_count = sum(1 for entry in entries if entry["status"] == "skipped_existing")
    failure_count = sum(1 for entry in entries if entry["status"] == "failed")
    return {
        "script": "scripts/convert_las_to_uni3d_npy.py",
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "pattern": args.pattern,
        "recursive": args.recursive,
        "limit": args.limit,
        "num_points": args.num_points,
        "seed": args.seed,
        "include_rgb": args.include_rgb,
        "overwrite": args.overwrite,
        "skip_existing": args.skip_existing,
        "manifest_path": str(manifest_path),
        "candidate_count": len(entries),
        "success_count": success_count,
        "skipped_count": skipped_count,
        "failure_count": failure_count,
        "results": make_jsonable(entries),
    }


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
