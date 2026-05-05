"""Convert single-tree LAS files into Uni3D batch-extraction .npy inputs.

This script is intentionally narrow: it only prepares .npy files shaped [N, 3]
or [N, 6] for scripts/batch_extract_uni3d_embeddings.py. It does not call the
Uni3D extractor, does not require CUDA, and does not modify the main QA path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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


@dataclass(frozen=True)
class OutputArray:
    """Prepared Uni3D input array plus color provenance."""

    values: np.ndarray
    method: str
    color_source: str
    has_color_columns: bool
    natural_rgb: bool | None
    semantic_color_likely: bool
    warning: str | None


OUTPUT_MODES = ("xyz_only", "color_all")
COLOR_POLICIES = (
    "drop",
    "las_rgb_if_available_else_constant",
    "constant_all",
    "existing_semantic_if_available_else_constant",
    "site_palette",
    "species_palette",
)


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
    parser.add_argument("--output-mode", choices=OUTPUT_MODES, default=None)
    parser.add_argument(
        "--color-policy",
        choices=COLOR_POLICIES,
        default=None,
        help="How to populate color columns. Defaults to drop for xyz_only and constant_all for color_all.",
    )
    parser.add_argument(
        "--constant-color",
        nargs=3,
        type=float,
        default=[0.4, 0.4, 0.4],
        metavar=("R", "G", "B"),
        help="Constant color in [0, 1] used by color policies that need fallback color.",
    )
    parser.add_argument("--label-path", default=None, help="Optional label table for species_palette diagnostics.")
    parser.add_argument(
        "--previous-npy-dir",
        default=None,
        help="Optional existing .npy directory for existing_semantic_if_available_else_constant.",
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
        normalize_color_options(args)
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
            output_mode=args.output_mode,
            color_policy=args.color_policy,
            constant_color=normalize_constant_color(args.constant_color),
            previous_npy_dir=Path(args.previous_npy_dir) if args.previous_npy_dir else None,
            label_lookup=load_label_lookup(Path(args.label_path)) if args.label_path else {},
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


def normalize_color_options(args: argparse.Namespace) -> None:
    if args.output_mode is None:
        args.output_mode = "color_all" if args.include_rgb else "xyz_only"
    if args.color_policy is None:
        if args.include_rgb and args.output_mode == "color_all":
            args.color_policy = "las_rgb_if_available_else_constant"
        else:
            args.color_policy = "drop" if args.output_mode == "xyz_only" else "constant_all"
    if args.output_mode == "xyz_only" and args.color_policy != "drop":
        raise ValueError("--output-mode xyz_only requires --color-policy drop.")
    if args.output_mode == "color_all" and args.color_policy == "drop":
        raise ValueError("--color-policy drop is only valid with --output-mode xyz_only.")
    if args.color_policy == "species_palette" and not args.label_path:
        raise ValueError("--color-policy species_palette requires --label-path.")
    if args.color_policy == "existing_semantic_if_available_else_constant" and not args.previous_npy_dir:
        print(
            "WARNING: existing_semantic_if_available_else_constant was requested without "
            "--previous-npy-dir; all samples will use --constant-color.",
            file=sys.stderr,
        )
    normalize_constant_color(args.constant_color)


def normalize_constant_color(values: list[float] | tuple[float, float, float] | np.ndarray) -> list[float]:
    color = np.asarray(values, dtype=np.float32)
    if color.shape != (3,):
        raise ValueError(f"--constant-color must have exactly 3 values, got {values}.")
    if not np.isfinite(color).all():
        raise ValueError("--constant-color contains NaN or Inf.")
    if np.any(color < 0.0) or np.any(color > 1.0):
        raise ValueError("--constant-color values must be in [0, 1].")
    return [float(value) for value in color]


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
    output_mode: str = "xyz_only",
    color_policy: str = "drop",
    constant_color: list[float] | None = None,
    previous_npy_dir: Path | None = None,
    label_lookup: dict[str, dict[str, Any]] | None = None,
    overwrite: bool = False,
    skip_existing: bool = False,
) -> dict[str, Any]:
    output_npy = output_path_for_las(source_las=source_las, input_dir=input_dir, output_dir=output_dir)
    tree_id = source_las.stem
    site_id = site_id_from_tree_id(tree_id)
    constant_color = normalize_constant_color(constant_color or [0.4, 0.4, 0.4])
    label_lookup = label_lookup or {}
    entry: dict[str, Any] = {
        "status": "failed",
        "tree_id": tree_id,
        "site_id": site_id,
        "source_las": str(source_las),
        "output_npy": str(output_npy),
        "output_shape": None,
        "has_rgb": None,
        "include_rgb": include_rgb,
        "output_mode": output_mode,
        "color_policy": color_policy,
        "color_source": None,
        "has_las_rgb": None,
        "has_color_columns": None,
        "natural_rgb": None,
        "semantic_color_likely": None,
        "constant_color": constant_color if color_policy in color_policies_using_constant(color_policy) else None,
        "warning": warning_for_color_policy(color_policy),
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
                        "has_color_columns": existing["has_rgb"],
                        "color_source": "skipped_existing",
                        "has_las_rgb": None,
                        "natural_rgb": None,
                        "semantic_color_likely": infer_semantic_color_likely_for_existing(
                            output_mode=output_mode,
                            color_policy=color_policy,
                            has_color_columns=existing["has_rgb"],
                        ),
                        "output_point_count": existing["output_point_count"],
                        "sampling_method": "skipped_existing",
                        "failure_reason": None,
                    }
                )
                return add_legacy_manifest_aliases(entry)
            raise ValueError("output_exists_use_overwrite_or_skip_existing")
        arrays = read_las_arrays(
            source_las,
            include_rgb=include_rgb or color_policy == "las_rgb_if_available_else_constant",
        )
        point_count = int(arrays.xyz.shape[0])
        entry["input_point_count"] = point_count
        entry["has_rgb"] = arrays.has_rgb
        entry["has_las_rgb"] = arrays.has_rgb
        if point_count < num_points:
            raise ValueError(f"too_few_points: got {point_count}, need {num_points}")
        sampled = sample_for_output_mode(
            arrays=arrays,
            source_las=source_las,
            input_dir=input_dir,
            tree_id=tree_id,
            site_id=site_id,
            num_points=num_points,
            seed=seed,
            output_mode=output_mode,
            color_policy=color_policy,
            constant_color=constant_color,
            previous_npy_dir=previous_npy_dir,
            label_lookup=label_lookup,
        )
        output_npy.parent.mkdir(parents=True, exist_ok=True)
        validate_output_array(sampled.values, output_mode=output_mode)
        np.save(output_npy, sampled.values.astype(np.float32, copy=False))
        entry.update(
            {
                "status": "ok",
                "output_shape": [int(value) for value in sampled.values.shape],
                "output_point_count": int(sampled.values.shape[0]),
                "sampling_method": sampled.method,
                "color_source": sampled.color_source,
                "has_color_columns": sampled.has_color_columns,
                "natural_rgb": sampled.natural_rgb,
                "semantic_color_likely": sampled.semantic_color_likely,
                "warning": sampled.warning,
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
    has_rgb = las_has_rgb(las)
    if include_rgb and has_rgb:
        rgb = normalize_rgb(
            np.column_stack(
                [
                    np.asarray(las.red),
                    np.asarray(las.green),
                    np.asarray(las.blue),
                ]
            )
        )
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


def sample_for_output_mode(
    *,
    arrays: LasArrays,
    source_las: Path,
    input_dir: Path,
    tree_id: str,
    site_id: str,
    num_points: int,
    seed: int,
    output_mode: str,
    color_policy: str,
    constant_color: list[float],
    previous_npy_dir: Path | None,
    label_lookup: dict[str, dict[str, Any]],
) -> OutputArray:
    if output_mode == "xyz_only":
        sampled = sample_fixed_count(arrays.xyz, num_points=num_points, seed=seed)
        return OutputArray(
            values=sampled.values,
            method=sampled.method,
            color_source="dropped",
            has_color_columns=False,
            natural_rgb=False,
            semantic_color_likely=False,
            warning=None,
        )

    if output_mode != "color_all":
        raise ValueError(f"unsupported output_mode: {output_mode}")

    if color_policy == "las_rgb_if_available_else_constant":
        if arrays.rgb is not None:
            sampled = sample_fixed_count(build_uni3d_input_array(arrays.xyz, arrays.rgb), num_points=num_points, seed=seed)
            return OutputArray(
                values=sampled.values,
                method=sampled.method,
                color_source="las_rgb",
                has_color_columns=True,
                natural_rgb=None,
                semantic_color_likely=False,
                warning="LAS RGB fields are present, but this does not prove they are natural RGB.",
            )
        sampled_xyz = sample_fixed_count(arrays.xyz, num_points=num_points, seed=seed)
        return attach_color_to_sampled_xyz(
            sampled_xyz,
            color=constant_color_matrix(num_points, constant_color),
            color_source="constant_color",
            natural_rgb=False,
            semantic_color_likely=False,
            warning=None,
        )

    if color_policy == "constant_all":
        sampled_xyz = sample_fixed_count(arrays.xyz, num_points=num_points, seed=seed)
        return attach_color_to_sampled_xyz(
            sampled_xyz,
            color=constant_color_matrix(num_points, constant_color),
            color_source="constant_color",
            natural_rgb=False,
            semantic_color_likely=False,
            warning=None,
        )

    if color_policy == "existing_semantic_if_available_else_constant":
        sampled_xyz = sample_fixed_count(arrays.xyz, num_points=num_points, seed=seed)
        previous_rgb = load_previous_semantic_rgb(
            previous_npy_dir=previous_npy_dir,
            source_las=source_las,
            input_dir=input_dir,
            tree_id=tree_id,
            num_points=num_points,
        )
        if previous_rgb is not None:
            return attach_color_to_sampled_xyz(
                sampled_xyz,
                color=previous_rgb,
                color_source="existing_semantic_color",
                natural_rgb=False,
                semantic_color_likely=True,
                warning="Existing color is treated as semantic/discrete color, not natural RGB.",
            )
        return attach_color_to_sampled_xyz(
            sampled_xyz,
            color=constant_color_matrix(num_points, constant_color),
            color_source="constant_color",
            natural_rgb=False,
            semantic_color_likely=False,
            warning="No previous semantic color was found; constant color fallback was used.",
        )

    if color_policy == "site_palette":
        sampled_xyz = sample_fixed_count(arrays.xyz, num_points=num_points, seed=seed)
        return attach_color_to_sampled_xyz(
            sampled_xyz,
            color=constant_color_matrix(num_points, palette_color(site_id)),
            color_source="site_palette",
            natural_rgb=False,
            semantic_color_likely=True,
            warning="site_palette is diagnostic only and may cause site/domain leakage.",
        )

    if color_policy == "species_palette":
        sampled_xyz = sample_fixed_count(arrays.xyz, num_points=num_points, seed=seed)
        species = label_lookup.get(tree_id, {}).get("species")
        if not species:
            raise ValueError("species_palette requested but species label is missing for this tree_id")
        return attach_color_to_sampled_xyz(
            sampled_xyz,
            color=constant_color_matrix(num_points, palette_color(str(species))),
            color_source="species_palette",
            natural_rgb=False,
            semantic_color_likely=True,
            warning="species_palette encodes labels and is label leakage; do not use for formal species probes.",
        )

    raise ValueError(f"unsupported color_policy: {color_policy}")


def attach_color_to_sampled_xyz(
    sampled_xyz: SampledArray,
    *,
    color: np.ndarray,
    color_source: str,
    natural_rgb: bool | None,
    semantic_color_likely: bool,
    warning: str | None,
) -> OutputArray:
    if color.ndim != 2 or color.shape[1] != 3:
        raise ValueError(f"color must have shape [N, 3], got {color.shape}.")
    if color.shape[0] != sampled_xyz.values.shape[0]:
        raise ValueError(f"color point count must match sampled xyz: {color.shape[0]} vs {sampled_xyz.values.shape[0]}.")
    validate_color_range(color)
    return OutputArray(
        values=build_uni3d_input_array(sampled_xyz.values, color),
        method=sampled_xyz.method,
        color_source=color_source,
        has_color_columns=True,
        natural_rgb=natural_rgb,
        semantic_color_likely=semantic_color_likely,
        warning=warning,
    )


def constant_color_matrix(point_count: int, color: list[float] | tuple[float, float, float] | np.ndarray) -> np.ndarray:
    normalized = np.asarray(normalize_constant_color(color), dtype=np.float32)
    return np.repeat(normalized[None, :], point_count, axis=0).astype(np.float32)


def validate_color_range(color: np.ndarray) -> None:
    if not np.isfinite(color).all():
        raise ValueError("color contains NaN or Inf values.")
    if np.min(color) < -1e-6 or np.max(color) > 1.0 + 1e-6:
        raise ValueError("color values must be normalized to [0, 1].")


def validate_output_array(array: np.ndarray, *, output_mode: str) -> None:
    expected_dim = 3 if output_mode == "xyz_only" else 6
    if array.ndim != 2 or array.shape[1] != expected_dim:
        raise ValueError(f"{output_mode} output must have shape [N, {expected_dim}], got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError("output array contains NaN or Inf values.")
    if output_mode == "color_all":
        validate_color_range(array[:, 3:6])


def load_previous_semantic_rgb(
    *,
    previous_npy_dir: Path | None,
    source_las: Path,
    input_dir: Path,
    tree_id: str,
    num_points: int,
) -> np.ndarray | None:
    if previous_npy_dir is None:
        return None
    candidates: list[Path] = []
    try:
        candidates.append(previous_npy_dir / source_las.relative_to(input_dir).with_suffix(".npy"))
    except ValueError:
        pass
    candidates.append(previous_npy_dir / f"{tree_id}.npy")
    candidates.extend(previous_npy_dir.rglob(f"{tree_id}.npy"))
    for candidate in candidates:
        if not candidate.is_file():
            continue
        array = np.load(candidate)
        if array.ndim == 2 and array.shape == (num_points, 6):
            color = np.asarray(array[:, 3:6], dtype=np.float32)
            validate_color_range(color)
            return color
    return None


def palette_color(key: str) -> list[float]:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    # Keep colors away from exact black/white so diagnostic palettes are easy to inspect.
    return [0.15 + (digest[index] / 255.0) * 0.7 for index in range(3)]


def site_id_from_tree_id(tree_id: str) -> str:
    return tree_id.split("_", 1)[0] if "_" in tree_id else ""


def color_policies_using_constant(color_policy: str) -> set[str]:
    if color_policy in {"las_rgb_if_available_else_constant", "constant_all", "existing_semantic_if_available_else_constant"}:
        return {color_policy}
    return set()


def warning_for_color_policy(color_policy: str) -> str | None:
    if color_policy == "site_palette":
        return "site_palette is diagnostic only and may cause site/domain leakage."
    if color_policy == "species_palette":
        return "species_palette is label leakage and must not be used for formal species classification probes."
    if color_policy == "existing_semantic_if_available_else_constant":
        return "Existing color is treated as semantic/discrete color when available, not natural RGB."
    if color_policy == "las_rgb_if_available_else_constant":
        return "LAS RGB fields do not by themselves prove natural RGB."
    return None


def infer_semantic_color_likely_for_existing(*, output_mode: str, color_policy: str, has_color_columns: bool) -> bool:
    if output_mode == "xyz_only" or not has_color_columns:
        return False
    return color_policy in {"existing_semantic_if_available_else_constant", "site_palette", "species_palette"}


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


def load_label_lookup(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"label file not found: {path}")
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:
            raise RuntimeError("pandas is required to read Excel label tables for species_palette.") from exc
        rows = [dict(row) for _, row in pd.read_excel(path).iterrows()]
    else:
        raise ValueError(f"unsupported label file: {path}")
    if not rows:
        return {}
    columns = list(rows[0].keys())
    tree_column = first_existing_column(columns, ["tree_id", "sample_id", "对应的文件名", "file_name", "filename"])
    species_column = first_existing_column(columns, ["species", "树种"])
    if tree_column is None or species_column is None:
        raise ValueError("species_palette label table must contain tree_id/file_name and species columns.")
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        tree_id = normalize_tree_id(row.get(tree_column))
        if tree_id:
            output[tree_id] = {"species": str(row.get(species_column))}
    return output


def first_existing_column(columns: list[str], candidates: list[str]) -> str | None:
    lowered = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def normalize_tree_id(value: Any) -> str:
    text = str(value).strip()
    if text.lower() in {"", "none", "nan"}:
        return ""
    text = text.replace("\\", "/").split("/")[-1]
    suffix = Path(text).suffix.lower()
    if suffix in {".las", ".laz", ".npy", ".npz", ".txt", ".csv"}:
        text = Path(text).stem
    return text


def build_summary(*, args: argparse.Namespace, entries: list[dict[str, Any]], manifest_path: Path) -> dict[str, Any]:
    success_count = sum(1 for entry in entries if entry["status"] == "ok")
    skipped_count = sum(1 for entry in entries if entry["status"] == "skipped_existing")
    failure_count = sum(1 for entry in entries if entry["status"] == "failed")
    distribution_entries = [entry for entry in entries if entry["status"] in {"ok", "skipped_existing"}]
    warnings = sorted({entry.get("warning") for entry in entries if entry.get("warning")})
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
        "output_mode": args.output_mode,
        "color_policy": args.color_policy,
        "constant_color": normalize_constant_color(args.constant_color),
        "label_path": args.label_path,
        "previous_npy_dir": args.previous_npy_dir,
        "overwrite": args.overwrite,
        "skip_existing": args.skip_existing,
        "manifest_path": str(manifest_path),
        "candidate_count": len(entries),
        "success_count": success_count,
        "skipped_count": skipped_count,
        "failure_count": failure_count,
        "shape_distribution": count_values(distribution_entries, "output_shape"),
        "color_source_distribution": count_values(distribution_entries, "color_source"),
        "site_distribution": count_values(distribution_entries, "site_id"),
        "warnings": warnings,
        "results": make_jsonable(entries),
    }


def count_values(entries: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        value = entry.get(key)
        if value is None:
            value_key = "null"
        elif isinstance(value, list):
            value_key = json.dumps(value)
        else:
            value_key = str(value)
        counts[value_key] = counts.get(value_key, 0) + 1
    return counts


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
