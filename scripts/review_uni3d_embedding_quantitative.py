"""Quantitative review table for existing Uni3D embedding analyses.

This script is offline-only: it reads saved extraction summaries, similarity
outputs, candidate lists, and optional .npy inputs. It never runs Uni3D forward
and never modifies existing embeddings or baseline pipelines.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REVIEW_FIELDS = [
    "tree_id",
    "site_id",
    "source_path",
    "resolved_npy_path",
    "embedding_path",
    "candidate_type",
    "input_shape",
    "point_count",
    "rgb_source",
    "rgb_fallback",
    "centroid",
    "scale_radius",
    "xyz_min",
    "xyz_max",
    "xyz_range",
    "z_range",
    "xy_range",
    "rgb_mean",
    "rgb_std",
    "rgb_min",
    "rgb_max",
    "embedding_raw_l2_norm",
    "embedding_l2_l2_norm",
    "mean_similarity",
    "nearest_neighbor_tree_id",
    "nearest_neighbor_similarity",
]


@dataclass(frozen=True)
class NpyStats:
    resolved_npy_path: str | None
    input_shape: list[int] | None
    point_count: int | None
    xyz_min: list[float] | None
    xyz_max: list[float] | None
    xyz_range: list[float] | None
    z_range: float | None
    xy_range: list[float] | None
    rgb_mean: list[float] | None
    rgb_std: list[float] | None
    rgb_min: list[float] | None
    rgb_max: list[float] | None
    missing_reason: str | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build offline Uni3D embedding review_table.csv/json from existing artifacts.",
    )
    parser.add_argument("--extraction-summary", required=True, help="Path to embedding extraction summary.json.")
    parser.add_argument("--similarity-summary", required=True, help="Path to similarity_summary.json.")
    parser.add_argument("--nearest-neighbors", required=True, help="Path to nearest_neighbors.json.")
    parser.add_argument("--near-duplicates", required=True, help="Path to near_duplicate_pairs.csv/json.")
    parser.add_argument("--outliers", required=True, help="Path to outlier_samples.csv/json.")
    parser.add_argument("--npy-input-dir", required=True, help="Directory containing converted Uni3D .npy inputs.")
    parser.add_argument("--output-dir", required=True, help="Directory for review_table and summary outputs.")
    parser.add_argument("--top-n", type=int, default=30, help="Top candidate count for focused review tables.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extraction_summary = load_json(Path(args.extraction_summary))
    similarity_summary = load_json(Path(args.similarity_summary))
    nearest_neighbors = load_nearest_neighbors(Path(args.nearest_neighbors))
    near_pairs = load_candidate_rows(Path(args.near_duplicates))
    outliers = load_candidate_rows(Path(args.outliers))

    npy_input_dir = Path(args.npy_input_dir)
    npy_index = build_npy_index(npy_input_dir)
    extraction_rows = extraction_summary.get("results", [])
    review_rows, missing_input_records = build_review_rows(
        extraction_rows=extraction_rows,
        nearest_neighbors=nearest_neighbors,
        near_pairs=near_pairs,
        outliers=outliers,
        npy_index=npy_index,
    )

    site_summary = summarize_by_group(review_rows, "site_id")
    rgb_summary = summarize_by_group(review_rows, "rgb_source")
    top_near_duplicate_review = build_top_near_duplicate_review(near_pairs, review_rows, args.top_n)
    top_outlier_review = build_top_outlier_review(outliers, review_rows, args.top_n)

    write_csv(output_dir / "review_table.csv", review_rows, REVIEW_FIELDS)
    write_json(output_dir / "review_table.json", review_rows)
    write_csv(output_dir / "top_near_duplicate_review.csv", top_near_duplicate_review)
    write_json(output_dir / "top_near_duplicate_review.json", top_near_duplicate_review)
    write_csv(output_dir / "top_outlier_review.csv", top_outlier_review)
    write_json(output_dir / "top_outlier_review.json", top_outlier_review)

    review_summary = {
        "script": "scripts/review_uni3d_embedding_quantitative.py",
        "inputs": {
            "extraction_summary": str(Path(args.extraction_summary)),
            "similarity_summary": str(Path(args.similarity_summary)),
            "nearest_neighbors": str(Path(args.nearest_neighbors)),
            "near_duplicates": str(Path(args.near_duplicates)),
            "outliers": str(Path(args.outliers)),
            "npy_input_dir": str(npy_input_dir),
        },
        "output_dir": str(output_dir),
        "sample_count": len(review_rows),
        "candidate_counts": candidate_counts(review_rows),
        "similarity_summary_subset": {
            "sample_count": similarity_summary.get("sample_count"),
            "analyzed_sample_count": similarity_summary.get("analyzed_sample_count"),
            "embedding_dim": similarity_summary.get("embedding_dim"),
            "invalid_count": similarity_summary.get("invalid_count"),
            "pairwise_similarity": similarity_summary.get("pairwise_similarity"),
        },
        "site_summary": site_summary,
        "rgb_source_summary": rgb_summary,
        "top_near_duplicate_count": len(top_near_duplicate_review),
        "top_outlier_count": len(top_outlier_review),
        "missing_input_file_count": len(missing_input_records),
        "missing_input_file_examples": missing_input_records[:20],
        "missing_field_notes": build_missing_field_notes(review_rows, missing_input_records),
    }
    write_json(output_dir / "review_summary.json", review_summary)

    print(f"review_table={output_dir / 'review_table.csv'}")
    print(f"review_summary={output_dir / 'review_summary.json'}")
    print(f"sample_count={len(review_rows)}")
    print(f"missing_input_file_count={len(missing_input_records)}")
    return 0


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_nearest_neighbors(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    output: dict[str, dict[str, Any]] = {}
    for item in payload:
        tree_id = item.get("tree_id")
        if tree_id:
            output[str(tree_id)] = item
    return output


def load_candidate_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = load_json(path)
        if not isinstance(payload, list):
            raise ValueError(f"Expected list in {path}")
        return [dict(item) for item in payload]
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise ValueError(f"Unsupported candidate file: {path}")


def build_npy_index(npy_input_dir: Path) -> dict[str, Path]:
    if not npy_input_dir.is_dir():
        return {}
    output: dict[str, Path] = {}
    for path in sorted(npy_input_dir.rglob("*.npy")):
        output.setdefault(path.stem, path)
    return output


def build_review_rows(
    *,
    extraction_rows: list[dict[str, Any]],
    nearest_neighbors: dict[str, dict[str, Any]],
    near_pairs: list[dict[str, Any]],
    outliers: list[dict[str, Any]],
    npy_index: dict[str, Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    near_ids = collect_near_duplicate_ids(near_pairs)
    outlier_map = {str(row.get("tree_id")): row for row in outliers if row.get("tree_id")}
    missing_inputs: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []

    for row in extraction_rows:
        tree_id = str(row.get("tree_id"))
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        source_path = row.get("source_path") or metadata.get("source_path")
        npy_stats = load_npy_stats(tree_id=tree_id, source_path=source_path, npy_index=npy_index)
        if npy_stats.missing_reason:
            missing_inputs.append(
                {
                    "tree_id": tree_id,
                    "source_path": source_path,
                    "missing_reason": npy_stats.missing_reason,
                }
            )
        nn = first_neighbor(nearest_neighbors.get(tree_id))
        outlier = outlier_map.get(tree_id, {})
        review_rows.append(
            {
                "tree_id": tree_id,
                "site_id": site_id_from_tree_id(tree_id),
                "source_path": source_path,
                "resolved_npy_path": npy_stats.resolved_npy_path,
                "embedding_path": row.get("output_path") or outlier.get("embedding_path"),
                "candidate_type": candidate_type(tree_id, near_ids, outlier_map),
                "input_shape": npy_stats.input_shape or row.get("input_shape") or metadata.get("input_shape"),
                "point_count": npy_stats.point_count or metadata.get("point_count"),
                "rgb_source": metadata.get("rgb_source") or outlier.get("rgb_source"),
                "rgb_fallback": metadata.get("rgb_fallback"),
                "centroid": metadata.get("centroid"),
                "scale_radius": metadata.get("scale_radius") or outlier.get("scale_radius"),
                "xyz_min": npy_stats.xyz_min,
                "xyz_max": npy_stats.xyz_max,
                "xyz_range": npy_stats.xyz_range,
                "z_range": npy_stats.z_range,
                "xy_range": npy_stats.xy_range,
                "rgb_mean": npy_stats.rgb_mean,
                "rgb_std": npy_stats.rgb_std,
                "rgb_min": npy_stats.rgb_min,
                "rgb_max": npy_stats.rgb_max,
                "embedding_raw_l2_norm": scalar_or_json(row.get("embedding_raw_l2_norm")),
                "embedding_l2_l2_norm": scalar_or_json(row.get("embedding_l2_l2_norm")),
                "mean_similarity": to_float_or_none(outlier.get("mean_similarity_to_others")),
                "nearest_neighbor_tree_id": nn.get("tree_id") or outlier.get("nearest_neighbor_tree_id"),
                "nearest_neighbor_similarity": to_float_or_none(
                    nn.get("cosine_similarity") or outlier.get("max_neighbor_similarity")
                ),
            }
        )
    return review_rows, missing_inputs


def collect_near_duplicate_ids(near_pairs: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for row in near_pairs:
        for key in ("left_tree_id", "right_tree_id"):
            if row.get(key):
                ids.add(str(row[key]))
    return ids


def load_npy_stats(*, tree_id: str, source_path: Any, npy_index: dict[str, Path]) -> NpyStats:
    resolved = resolve_npy_path(tree_id=tree_id, source_path=source_path, npy_index=npy_index)
    if resolved is None:
        return empty_npy_stats("input_npy_not_found")
    try:
        array = np.load(resolved)
        if array.ndim != 2 or array.shape[1] not in (3, 6):
            return empty_npy_stats(f"invalid_input_shape:{array.shape}", resolved)
        if not np.isfinite(array).all():
            return empty_npy_stats("input_contains_nan_or_inf", resolved)
        xyz = array[:, :3].astype(np.float64)
        xyz_min = xyz.min(axis=0)
        xyz_max = xyz.max(axis=0)
        xyz_range = xyz_max - xyz_min
        rgb_mean = rgb_std = rgb_min = rgb_max = None
        if array.shape[1] == 6:
            rgb = array[:, 3:6].astype(np.float64)
            rgb_mean = rgb.mean(axis=0).tolist()
            rgb_std = rgb.std(axis=0).tolist()
            rgb_min = rgb.min(axis=0).tolist()
            rgb_max = rgb.max(axis=0).tolist()
        return NpyStats(
            resolved_npy_path=str(resolved),
            input_shape=[int(value) for value in array.shape],
            point_count=int(array.shape[0]),
            xyz_min=xyz_min.tolist(),
            xyz_max=xyz_max.tolist(),
            xyz_range=xyz_range.tolist(),
            z_range=float(xyz_range[2]),
            xy_range=[float(xyz_range[0]), float(xyz_range[1])],
            rgb_mean=rgb_mean,
            rgb_std=rgb_std,
            rgb_min=rgb_min,
            rgb_max=rgb_max,
            missing_reason=None,
        )
    except Exception as exc:
        return empty_npy_stats(f"{type(exc).__name__}:{exc}", resolved)


def resolve_npy_path(*, tree_id: str, source_path: Any, npy_index: dict[str, Path]) -> Path | None:
    if source_path:
        direct = Path(str(source_path))
        if direct.is_file():
            return direct
    return npy_index.get(tree_id)


def empty_npy_stats(reason: str, resolved: Path | None = None) -> NpyStats:
    return NpyStats(
        resolved_npy_path=str(resolved) if resolved is not None else None,
        input_shape=None,
        point_count=None,
        xyz_min=None,
        xyz_max=None,
        xyz_range=None,
        z_range=None,
        xy_range=None,
        rgb_mean=None,
        rgb_std=None,
        rgb_min=None,
        rgb_max=None,
        missing_reason=reason,
    )


def first_neighbor(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {}
    nearest = row.get("nearest")
    if isinstance(nearest, list) and nearest:
        return nearest[0]
    return {}


def candidate_type(tree_id: str, near_ids: set[str], outlier_map: dict[str, dict[str, Any]]) -> str:
    is_near = tree_id in near_ids
    is_outlier = tree_id in outlier_map
    if is_near and is_outlier:
        return "both"
    if is_near:
        return "near_duplicate"
    if is_outlier:
        return "outlier"
    return "normal"


def site_id_from_tree_id(tree_id: str) -> str:
    return tree_id.split("_", 1)[0]


def scalar_or_json(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def to_float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_by_group(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key))].append(row)
    summary: list[dict[str, Any]] = []
    for group, items in grouped.items():
        mean_values = [row["mean_similarity"] for row in items if row.get("mean_similarity") is not None]
        summary.append(
            {
                group_key: None if group == "None" else group,
                "sample_count": len(items),
                "mean_similarity": float(np.mean(mean_values)) if mean_values else None,
                "outlier_count": sum(1 for row in items if row.get("candidate_type") in {"outlier", "both"}),
                "near_duplicate_count": sum(
                    1 for row in items if row.get("candidate_type") in {"near_duplicate", "both"}
                ),
            }
        )
    summary.sort(key=lambda item: str(item.get(group_key)))
    return summary


def build_top_near_duplicate_review(
    near_pairs: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    top_n: int,
) -> list[dict[str, Any]]:
    by_id = {row["tree_id"]: row for row in review_rows}
    output: list[dict[str, Any]] = []
    for row in near_pairs[:top_n]:
        left_id = str(row.get("left_tree_id"))
        right_id = str(row.get("right_tree_id"))
        left = by_id.get(left_id, {})
        right = by_id.get(right_id, {})
        output.append(
            {
                "left_tree_id": left_id,
                "right_tree_id": right_id,
                "cosine_similarity": to_float_or_none(row.get("cosine_similarity")),
                "left_site_id": left.get("site_id"),
                "right_site_id": right.get("site_id"),
                "left_rgb_source": left.get("rgb_source"),
                "right_rgb_source": right.get("rgb_source"),
                "left_scale_radius": left.get("scale_radius"),
                "right_scale_radius": right.get("scale_radius"),
                "left_xyz_range": left.get("xyz_range"),
                "right_xyz_range": right.get("xyz_range"),
                "left_rgb_mean": left.get("rgb_mean"),
                "right_rgb_mean": right.get("rgb_mean"),
                "left_source_path": left.get("source_path"),
                "right_source_path": right.get("source_path"),
            }
        )
    return output


def build_top_outlier_review(
    outliers: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    top_n: int,
) -> list[dict[str, Any]]:
    by_id = {row["tree_id"]: row for row in review_rows}
    output: list[dict[str, Any]] = []
    for row in outliers[:top_n]:
        tree_id = str(row.get("tree_id"))
        review = by_id.get(tree_id, {})
        output.append(
            {
                "tree_id": tree_id,
                "site_id": review.get("site_id"),
                "mean_similarity": review.get("mean_similarity"),
                "nearest_neighbor_tree_id": review.get("nearest_neighbor_tree_id"),
                "nearest_neighbor_similarity": review.get("nearest_neighbor_similarity"),
                "rgb_source": review.get("rgb_source"),
                "scale_radius": review.get("scale_radius"),
                "xyz_range": review.get("xyz_range"),
                "z_range": review.get("z_range"),
                "xy_range": review.get("xy_range"),
                "rgb_mean": review.get("rgb_mean"),
                "source_path": review.get("source_path"),
            }
        )
    return output


def candidate_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get("candidate_type"))
        counts[value] = counts.get(value, 0) + 1
    return counts


def build_missing_field_notes(rows: list[dict[str, Any]], missing_inputs: list[dict[str, Any]]) -> list[str]:
    notes: list[str] = []
    if missing_inputs:
        notes.append(
            f"{len(missing_inputs)} samples have no readable .npy input; xyz/rgb statistics are null for them."
        )
    for field in REVIEW_FIELDS:
        missing_count = sum(1 for row in rows if row.get(field) is None)
        if missing_count:
            notes.append(f"{field}: {missing_count} null values")
    return notes


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: encode_cell(row.get(field)) for field in fieldnames})


def encode_cell(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(make_jsonable(value), ensure_ascii=False)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(make_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def make_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [make_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
