"""Compare geometry and Uni3D nearest-neighbor structures offline.

This script reads frozen Uni3D embedding caches and tabular geometry labels.
It never runs Uni3D forward, trains a model, or touches the main QA pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


GEOMETRY_FEATURES = ["dbh_cm", "height_m", "crown_ew_m", "crown_ns_m"]
METHOD_GEOMETRY = "geometry_measurement"
METHOD_UNI3D = "uni3d_xyz_only"
METHOD_FUSED = "geometry_measurement_plus_uni3d"
FUSION_K = 60.0
DEFAULT_OUTPUT_DIR = "outputs/local_reports/geometry_uni3d_nn_xyz_only_v1"


@dataclass(frozen=True)
class EmbeddingRecord:
    tree_id: str
    embedding_path: Path
    source_path: str | None
    embedding: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SampleRecord:
    tree_id: str
    site_id: str | None
    species: str | None
    embedding_path: str
    source_path: str | None
    embedding: np.ndarray
    geometry: dict[str, float]
    metadata: dict[str, Any]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Geometry-only, Uni3D-only, and fused nearest neighbors.",
    )
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--label-path", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--near-duplicate-source", default=None)
    parser.add_argument("--tool-geometry-cache", default=None)
    parser.add_argument("--embedding-key", default="embedding_l2")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_comparison(
        embedding_dir=Path(args.embedding_dir),
        label_path=Path(args.label_path),
        output_dir=Path(args.output_dir),
        top_k=args.top_k,
        near_duplicate_source=Path(args.near_duplicate_source)
        if args.near_duplicate_source
        else None,
        tool_geometry_cache=Path(args.tool_geometry_cache)
        if args.tool_geometry_cache
        else None,
        embedding_key=args.embedding_key,
    )
    print(f"matched_samples={result['aggregate_summary']['matched_sample_count']}")
    print(f"output_dir={Path(args.output_dir)}")
    return 0


def run_comparison(
    *,
    embedding_dir: Path,
    label_path: Path,
    output_dir: Path,
    top_k: int,
    near_duplicate_source: Path | None = None,
    tool_geometry_cache: Path | None = None,
    embedding_key: str = "embedding_l2",
) -> dict[str, Any]:
    if top_k < 1:
        raise ValueError("--top-k must be positive.")
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings = load_embedding_records(embedding_dir, embedding_key=embedding_key)
    labels, label_report = load_geometry_labels(label_path)
    samples, alignment_report = align_samples(embeddings, labels)
    if len(samples) < 2:
        raise ValueError("At least two matched samples are required.")

    near_duplicate_pairs = load_near_duplicate_pairs(
        source=near_duplicate_source,
        embedding_dir=embedding_dir,
    )
    tool_geometry_status = load_tool_geometry_status(tool_geometry_cache)

    matrices = compute_neighbor_matrices(samples)
    neighbor_by_method = build_all_neighbor_lists(
        samples=samples,
        matrices=matrices,
        top_k=top_k,
        near_duplicate_pairs=near_duplicate_pairs,
    )
    neighbor_rows = flatten_neighbor_rows(neighbor_by_method)
    overlaps = build_overlap_rows(neighbor_by_method, top_k=top_k)
    geometry_summary = summarize_geometry_differences(neighbor_rows)
    site_species_summary = summarize_site_species(neighbor_rows)
    near_duplicate_rows = [
        row for row in neighbor_rows if row["near_duplicate_flag"] == "true"
    ]
    manual_review_rows = build_manual_review_candidates(
        neighbor_rows=neighbor_rows,
        overlap_rows=overlaps,
        samples=samples,
        top_k=top_k,
    )
    aggregate_summary = build_aggregate_summary(
        embedding_dir=embedding_dir,
        label_path=label_path,
        top_k=top_k,
        samples=samples,
        label_report=label_report,
        alignment_report=alignment_report,
        overlap_rows=overlaps,
        near_duplicate_pairs=near_duplicate_pairs,
        tool_geometry_status=tool_geometry_status,
    )

    write_json(output_dir / "neighbor_lists.json", neighbor_by_method)
    write_rows(output_dir / "neighbor_lists.csv", neighbor_rows)
    write_rows(output_dir / "per_query_overlap.csv", overlaps)
    write_json(output_dir / "aggregate_summary.json", aggregate_summary)
    write_rows(output_dir / "geometry_difference_summary.csv", geometry_summary)
    write_rows(output_dir / "site_species_summary.csv", site_species_summary)
    write_rows(output_dir / "near_duplicate_flags.csv", near_duplicate_rows)
    write_rows(output_dir / "manual_review_candidates.csv", manual_review_rows)
    (output_dir / "experiment_record.md").write_text(
        render_experiment_record(aggregate_summary, geometry_summary, site_species_summary),
        encoding="utf-8",
    )

    return {
        "neighbor_by_method": neighbor_by_method,
        "neighbor_rows": neighbor_rows,
        "overlap_rows": overlaps,
        "aggregate_summary": aggregate_summary,
        "manual_review_rows": manual_review_rows,
    }


def load_embedding_records(embedding_dir: Path, *, embedding_key: str) -> list[EmbeddingRecord]:
    if not embedding_dir.is_dir():
        raise ValueError(f"Embedding directory not found: {embedding_dir}")
    records: list[EmbeddingRecord] = []
    for path in sorted(embedding_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            if embedding_key not in payload:
                continue
            embedding = normalize_embedding_shape(payload[embedding_key], path=path)
            tree_id = normalize_tree_id(read_scalar(payload, "tree_id") or infer_tree_id(path))
            if not tree_id:
                continue
            records.append(
                EmbeddingRecord(
                    tree_id=tree_id,
                    embedding_path=path,
                    source_path=read_scalar(payload, "source_path"),
                    embedding=l2_normalize_vector(embedding),
                    metadata=parse_metadata(read_scalar(payload, "metadata")),
                )
            )
    if not records:
        raise ValueError(f"No valid embeddings found in {embedding_dir}")
    return records


def normalize_embedding_shape(value: np.ndarray, *, path: Path) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1:
        raise ValueError(f"{path} embedding must be shaped [D] or [1, D].")
    if not np.isfinite(array).all():
        raise ValueError(f"{path} embedding contains NaN or Inf.")
    return array


def l2_normalize_vector(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm < 1e-12:
        raise ValueError("Embedding vector has near-zero norm.")
    return (value / norm).astype(np.float32)


def read_scalar(payload: Any, key: str) -> str | None:
    if key not in payload:
        return None
    value = payload[key]
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return str(value.item())
        if value.size == 1:
            return str(value.reshape(-1)[0])
    return str(value)


def parse_metadata(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def infer_tree_id(path: Path) -> str:
    stem = path.stem
    parts = stem.split("_", 1)
    return parts[1] if len(parts) == 2 and parts[0].isdigit() else stem


def normalize_tree_id(value: Any) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    if text.lower().endswith(".las") or text.lower().endswith(".npy"):
        text = Path(text).stem
    return text


def load_geometry_labels(label_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    table = read_table(label_path)
    columns = list(table[0]) if table else []
    detected = detect_label_columns(columns)
    required = ["tree_id", "dbh_cm", "height_m", "crown_ew_m", "crown_ns_m"]
    missing = [name for name in required if detected.get(name) is None]
    if missing:
        raise ValueError(f"Could not detect required label columns: {', '.join(missing)}")

    labels: dict[str, dict[str, Any]] = {}
    duplicate_ids: list[str] = []
    for row in table:
        tree_id = normalize_tree_id(row.get(detected["tree_id"] or ""))
        if not tree_id:
            continue
        parsed = {
            "tree_id": tree_id,
            "site_id": str(row.get(detected["site_id"])) if detected.get("site_id") else site_id_from_tree_id(tree_id),
            "species": stringify_optional(row.get(detected["species"])) if detected.get("species") else None,
            "geometry": {
                "dbh_cm": parse_float(row.get(detected["dbh_cm"])),
                "height_m": parse_float(row.get(detected["height_m"])),
                "crown_ew_m": parse_float(row.get(detected["crown_ew_m"])),
                "crown_ns_m": parse_float(row.get(detected["crown_ns_m"])),
            },
        }
        if any(parsed["geometry"][name] is None for name in GEOMETRY_FEATURES):
            continue
        if tree_id in labels:
            duplicate_ids.append(tree_id)
        labels[tree_id] = parsed
    return labels, {
        "label_path": str(label_path),
        "detected_columns": detected,
        "label_row_count": len(table),
        "usable_label_count": len(labels),
        "duplicate_tree_ids": sorted(set(duplicate_ids)),
    }


def read_table(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"Label table not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]
    if suffix in {".xlsx", ".xls"}:
        import pandas as pd

        frame = pd.read_excel(path)
        return [
            {str(column): make_json_safe(row[column]) for column in frame.columns}
            for _, row in frame.iterrows()
        ]
    raise ValueError(f"Unsupported label table format: {path.suffix}")


def detect_label_columns(columns: list[str]) -> dict[str, str | None]:
    return {
        "site_id": detect_column(columns, ["site_id", "\u6837\u5730\u53f7"]),
        "species": detect_column(columns, ["species", "\u6811\u79cd"]),
        "tree_id": detect_column(
            columns,
            ["tree_id", "sample_id", "file_name", "filename", "\u5bf9\u5e94\u7684\u6587\u4ef6\u540d"],
        ),
        "dbh_cm": detect_column(columns, ["dbh_cm", "dbh", "\u80f8\u5f84cm", "\u80f8\u5f84"]),
        "height_m": detect_column(columns, ["height_m", "height", "\u6811\u9ad8m", "\u6811\u9ad8"]),
        "crown_ew_m": detect_column(
            columns,
            ["crown_ew_m", "crown_ew", "\u4e1c\u897f\u51a0\u5e45m", "\u4e1c\u897f\u51a0\u5e45"],
        ),
        "crown_ns_m": detect_column(
            columns,
            ["crown_ns_m", "crown_ns", "\u5357\u5317\u51a0\u5e45m", "\u5357\u5317\u51a0\u5e45"],
        ),
    }


def detect_column(columns: list[str], candidates: list[str]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if np.isfinite(output) else None


def stringify_optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if not text or text.lower() in {"nan", "none"} else text


def site_id_from_tree_id(tree_id: str) -> str | None:
    return tree_id.split("_", 1)[0] if "_" in tree_id else None


def align_samples(
    embeddings: list[EmbeddingRecord],
    labels: dict[str, dict[str, Any]],
) -> tuple[list[SampleRecord], dict[str, Any]]:
    label_ids = set(labels)
    embedding_ids = {record.tree_id for record in embeddings}
    samples: list[SampleRecord] = []
    for record in embeddings:
        label = labels.get(record.tree_id)
        if label is None:
            continue
        samples.append(
            SampleRecord(
                tree_id=record.tree_id,
                site_id=label.get("site_id") or site_id_from_tree_id(record.tree_id),
                species=label.get("species"),
                embedding_path=str(record.embedding_path),
                source_path=record.source_path,
                embedding=record.embedding,
                geometry={name: float(label["geometry"][name]) for name in GEOMETRY_FEATURES},
                metadata=record.metadata,
            )
        )
    return samples, {
        "embedding_count": len(embeddings),
        "label_count": len(labels),
        "matched_sample_count": len(samples),
        "embedding_without_label_count": len(embedding_ids - label_ids),
        "label_without_embedding_count": len(label_ids - embedding_ids),
        "embedding_without_label_examples": sorted(embedding_ids - label_ids)[:20],
        "label_without_embedding_examples": sorted(label_ids - embedding_ids)[:20],
    }


def load_near_duplicate_pairs(
    *,
    source: Path | None,
    embedding_dir: Path,
) -> set[tuple[str, str]]:
    candidates: list[Path] = []
    if source is not None:
        candidates.append(source)
    analysis_dir = embedding_dir.parent / f"{embedding_dir.name}_analysis"
    candidates.extend(
        [
            analysis_dir / "near_duplicate_pairs.csv",
            analysis_dir / "near_duplicate_pairs.json",
            analysis_dir / "similarity_summary.json",
        ]
    )
    for path in candidates:
        if path.is_file():
            return parse_near_duplicate_file(path)
    return set()


def parse_near_duplicate_file(path: Path) -> set[tuple[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("near_duplicate_pairs", payload) if isinstance(payload, dict) else payload
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        return set()
    output: set[tuple[str, str]] = set()
    if not isinstance(rows, list):
        return output
    for row in rows:
        if not isinstance(row, dict):
            continue
        left = normalize_tree_id(row.get("left_tree_id") or row.get("query_tree_id") or "")
        right = normalize_tree_id(row.get("right_tree_id") or row.get("neighbor_tree_id") or "")
        if left and right:
            output.add(pair_key(left, right))
    return output


def pair_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))


def load_tool_geometry_status(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "status": "skipped",
            "reason": "--tool-geometry-cache was not provided.",
        }
    if not path.is_file():
        return {
            "status": "unavailable",
            "path": str(path),
            "reason": "tool geometry cache does not exist.",
        }
    return {
        "status": "available_not_used_in_phase1",
        "path": str(path),
        "reason": "cache support is reserved for the q1/q2/q3 sensitivity check.",
    }


def compute_neighbor_matrices(samples: list[SampleRecord]) -> dict[str, Any]:
    geometry = np.asarray(
        [[sample.geometry[name] for name in GEOMETRY_FEATURES] for sample in samples],
        dtype=np.float64,
    )
    mean = geometry.mean(axis=0)
    std = geometry.std(axis=0)
    std[std < 1e-12] = 1.0
    z_geometry = (geometry - mean) / std
    geometry_distance = pairwise_euclidean(z_geometry)

    embeddings = np.stack([sample.embedding for sample in samples], axis=0).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    cosine = np.clip(embeddings @ embeddings.T, -1.0, 1.0).astype(np.float32)

    geometry_ranks = rank_distance_matrix(geometry_distance)
    uni3d_ranks = rank_similarity_matrix(cosine)
    fused_score = reciprocal_rank_fusion(geometry_ranks, uni3d_ranks)
    return {
        "geometry_distance": geometry_distance,
        "uni3d_cosine": cosine,
        "geometry_ranks": geometry_ranks,
        "uni3d_ranks": uni3d_ranks,
        "fused_score": fused_score,
        "geometry_mean": dict(zip(GEOMETRY_FEATURES, mean.tolist())),
        "geometry_std": dict(zip(GEOMETRY_FEATURES, std.tolist())),
    }


def pairwise_euclidean(values: np.ndarray) -> np.ndarray:
    diff = values[:, None, :] - values[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def rank_distance_matrix(distance: np.ndarray) -> np.ndarray:
    count = distance.shape[0]
    ranks = np.full((count, count), count + 1, dtype=np.int32)
    for index in range(count):
        candidates = [candidate for candidate in range(count) if candidate != index]
        candidates.sort(key=lambda candidate: float(distance[index, candidate]))
        for rank, candidate in enumerate(candidates, start=1):
            ranks[index, candidate] = rank
    return ranks


def rank_similarity_matrix(similarity: np.ndarray) -> np.ndarray:
    count = similarity.shape[0]
    ranks = np.full((count, count), count + 1, dtype=np.int32)
    for index in range(count):
        candidates = [candidate for candidate in range(count) if candidate != index]
        candidates.sort(key=lambda candidate: float(similarity[index, candidate]), reverse=True)
        for rank, candidate in enumerate(candidates, start=1):
            ranks[index, candidate] = rank
    return ranks


def reciprocal_rank_fusion(geometry_ranks: np.ndarray, uni3d_ranks: np.ndarray) -> np.ndarray:
    score = (0.5 / (FUSION_K + geometry_ranks)) + (0.5 / (FUSION_K + uni3d_ranks))
    np.fill_diagonal(score, -np.inf)
    return score


def build_all_neighbor_lists(
    *,
    samples: list[SampleRecord],
    matrices: dict[str, Any],
    top_k: int,
    near_duplicate_pairs: set[tuple[str, str]],
) -> dict[str, Any]:
    methods = {
        METHOD_GEOMETRY: ("ascending", matrices["geometry_distance"]),
        METHOD_UNI3D: ("descending", matrices["uni3d_cosine"]),
        METHOD_FUSED: ("descending", matrices["fused_score"]),
    }
    output: dict[str, Any] = {}
    for index, sample in enumerate(samples):
        method_results: dict[str, Any] = {}
        for method, (direction, matrix) in methods.items():
            candidates = [candidate for candidate in range(len(samples)) if candidate != index]
            candidates.sort(
                key=lambda candidate: float(matrix[index, candidate]),
                reverse=direction == "descending",
            )
            method_results[method] = [
                build_neighbor_entry(
                    query=sample,
                    neighbor=samples[candidate],
                    rank=rank,
                    matrices=matrices,
                    query_index=index,
                    neighbor_index=candidate,
                    near_duplicate_pairs=near_duplicate_pairs,
                )
                for rank, candidate in enumerate(candidates[:top_k], start=1)
            ]
        output[sample.tree_id] = {
            "query_tree_id": sample.tree_id,
            "site_id": sample.site_id,
            "species": sample.species,
            "methods": method_results,
        }
    return output


def build_neighbor_entry(
    *,
    query: SampleRecord,
    neighbor: SampleRecord,
    rank: int,
    matrices: dict[str, Any],
    query_index: int,
    neighbor_index: int,
    near_duplicate_pairs: set[tuple[str, str]],
) -> dict[str, Any]:
    geometry_diffs = {
        f"{name}_diff": abs(query.geometry[name] - neighbor.geometry[name])
        for name in GEOMETRY_FEATURES
    }
    return {
        "rank": rank,
        "neighbor_tree_id": neighbor.tree_id,
        "neighbor_site_id": neighbor.site_id,
        "neighbor_species": neighbor.species,
        "same_site": query.site_id is not None and query.site_id == neighbor.site_id,
        "same_species": query.species is not None and query.species == neighbor.species,
        "geometry_distance": float(matrices["geometry_distance"][query_index, neighbor_index]),
        "uni3d_cosine": float(matrices["uni3d_cosine"][query_index, neighbor_index]),
        "fused_score": float(matrices["fused_score"][query_index, neighbor_index]),
        "geometry_rank": int(matrices["geometry_ranks"][query_index, neighbor_index]),
        "uni3d_rank": int(matrices["uni3d_ranks"][query_index, neighbor_index]),
        "near_duplicate_flag": pair_key(query.tree_id, neighbor.tree_id) in near_duplicate_pairs,
        "rgb_source": neighbor.metadata.get("rgb_source"),
        **geometry_diffs,
    }


def flatten_neighbor_rows(neighbor_by_method: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query_tree_id, payload in neighbor_by_method.items():
        for method, neighbors in payload["methods"].items():
            for item in neighbors:
                rows.append(
                    {
                        "query_tree_id": query_tree_id,
                        "query_site_id": payload.get("site_id"),
                        "query_species": payload.get("species"),
                        "method": method,
                        **csv_safe_row(item),
                    }
                )
    return rows


def build_overlap_rows(neighbor_by_method: dict[str, Any], *, top_k: int) -> list[dict[str, Any]]:
    method_pairs = [
        (METHOD_GEOMETRY, METHOD_UNI3D),
        (METHOD_GEOMETRY, METHOD_FUSED),
        (METHOD_UNI3D, METHOD_FUSED),
    ]
    rows: list[dict[str, Any]] = []
    for query_tree_id, payload in neighbor_by_method.items():
        methods = payload["methods"]
        for left, right in method_pairs:
            left_ids = {item["neighbor_tree_id"] for item in methods[left][:top_k]}
            right_ids = {item["neighbor_tree_id"] for item in methods[right][:top_k]}
            intersection = left_ids & right_ids
            union = left_ids | right_ids
            rows.append(
                {
                    "query_tree_id": query_tree_id,
                    "method_left": left,
                    "method_right": right,
                    "top_k": top_k,
                    "overlap_count": len(intersection),
                    "jaccard": len(intersection) / len(union) if union else 0.0,
                    "overlap_tree_ids": json.dumps(sorted(intersection), ensure_ascii=False),
                }
            )
    return rows


def summarize_geometry_differences(neighbor_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in sorted({row["method"] for row in neighbor_rows}):
        subset = [row for row in neighbor_rows if row["method"] == method]
        row: dict[str, Any] = {"method": method, "neighbor_count": len(subset)}
        for key in [f"{name}_diff" for name in GEOMETRY_FEATURES] + ["geometry_distance", "uni3d_cosine"]:
            values = np.asarray([float(item[key]) for item in subset if item.get(key) not in {None, ""}], dtype=float)
            row[f"{key}_mean"] = float(values.mean()) if values.size else None
            row[f"{key}_median"] = float(np.median(values)) if values.size else None
            row[f"{key}_p90"] = float(np.quantile(values, 0.9)) if values.size else None
        rows.append(row)
    return rows


def summarize_site_species(neighbor_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in sorted({row["method"] for row in neighbor_rows}):
        subset = [row for row in neighbor_rows if row["method"] == method]
        site_counter = Counter(row.get("neighbor_site_id") or "" for row in subset)
        species_counter = Counter(row.get("neighbor_species") or "" for row in subset)
        same_site = sum(1 for row in subset if row["same_site"] == "true")
        same_species = sum(1 for row in subset if row["same_species"] == "true")
        rows.append(
            {
                "method": method,
                "neighbor_count": len(subset),
                "same_site_count": same_site,
                "same_site_rate": same_site / len(subset) if subset else 0.0,
                "same_species_count": same_species,
                "same_species_rate": same_species / len(subset) if subset else 0.0,
                "neighbor_site_distribution": json.dumps(dict(site_counter), ensure_ascii=False, sort_keys=True),
                "neighbor_species_distribution": json.dumps(dict(species_counter), ensure_ascii=False, sort_keys=True),
            }
        )
    return rows


def build_manual_review_candidates(
    *,
    neighbor_rows: list[dict[str, Any]],
    overlap_rows: list[dict[str, Any]],
    samples: list[SampleRecord],
    top_k: int,
) -> list[dict[str, Any]]:
    selected: dict[tuple[str, str, str], dict[str, Any]] = {}

    def add(row: dict[str, Any], reason: str) -> None:
        key = (row["query_tree_id"], row["neighbor_tree_id"], row["method"])
        enriched = dict(row)
        enriched["review_reason"] = reason
        enriched["reviewer_label"] = ""
        enriched["reviewer_notes"] = ""
        selected.setdefault(key, enriched)

    low_overlap_ids = [
        row["query_tree_id"]
        for row in sorted(
            (
                row
                for row in overlap_rows
                if row["method_left"] == METHOD_GEOMETRY and row["method_right"] == METHOD_UNI3D
            ),
            key=lambda row: (float(row["jaccard"]), int(row["overlap_count"])),
        )[:20]
    ]
    for row in neighbor_rows:
        if row["query_tree_id"] in low_overlap_ids and int(row["rank"]) <= min(top_k, 3):
            add(row, "low_geometry_uni3d_overlap")

    for row in sorted(
        [row for row in neighbor_rows if row["method"] == METHOD_UNI3D],
        key=lambda row: (-float(row["geometry_distance"]), -float(row["uni3d_cosine"])),
    )[:20]:
        add(row, "uni3d_close_but_geometry_far")

    for row in sorted(
        [row for row in neighbor_rows if row["method"] == METHOD_GEOMETRY],
        key=lambda row: (float(row["geometry_distance"]), float(row["uni3d_cosine"])),
    )[:20]:
        add(row, "geometry_close_but_uni3d_far")

    for row in neighbor_rows:
        if row["near_duplicate_flag"] == "true":
            add(row, "near_duplicate_candidate")

    seen_sites: set[str] = set()
    for sample in samples:
        site = sample.site_id or ""
        if not site or site in seen_sites:
            continue
        seen_sites.add(site)
        site_rows = [
            row
            for row in neighbor_rows
            if row["query_tree_id"] == sample.tree_id and row["method"] == METHOD_UNI3D and int(row["rank"]) == 1
        ]
        for row in site_rows:
            add(row, "site_representative")
    return list(selected.values())


def build_aggregate_summary(
    *,
    embedding_dir: Path,
    label_path: Path,
    top_k: int,
    samples: list[SampleRecord],
    label_report: dict[str, Any],
    alignment_report: dict[str, Any],
    overlap_rows: list[dict[str, Any]],
    near_duplicate_pairs: set[tuple[str, str]],
    tool_geometry_status: dict[str, Any],
) -> dict[str, Any]:
    overlap_summary: dict[str, dict[str, float]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in overlap_rows:
        key = f"{row['method_left']}__vs__{row['method_right']}"
        grouped[key].append(row)
    for key, rows in grouped.items():
        jaccards = np.asarray([float(row["jaccard"]) for row in rows], dtype=float)
        overlaps = np.asarray([int(row["overlap_count"]) for row in rows], dtype=float)
        overlap_summary[key] = {
            "mean_jaccard": float(jaccards.mean()),
            "median_jaccard": float(np.median(jaccards)),
            "mean_overlap_count": float(overlaps.mean()),
            "median_overlap_count": float(np.median(overlaps)),
        }
    return {
        "script": "scripts/compare_geometry_uni3d_neighbors.py",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "embedding_dir": str(embedding_dir),
        "label_path": str(label_path),
        "top_k": top_k,
        "matched_sample_count": len(samples),
        "methods": [METHOD_GEOMETRY, METHOD_UNI3D, METHOD_FUSED],
        "geometry_features": GEOMETRY_FEATURES,
        "fusion": {
            "type": "reciprocal_rank_fusion",
            "k": FUSION_K,
            "geometry_weight": 0.5,
            "uni3d_weight": 0.5,
        },
        "label_report": label_report,
        "alignment_report": alignment_report,
        "near_duplicate_pair_count": len(near_duplicate_pairs),
        "overlap_summary": overlap_summary,
        "tool_geometry_status": tool_geometry_status,
        "boundary": (
            "Offline analysis only: no Uni3D forward, no model training, no LLM, "
            "and no runtime QA integration."
        ),
    }


def render_experiment_record(
    summary: dict[str, Any],
    geometry_summary: list[dict[str, Any]],
    site_species_summary: list[dict[str, Any]],
) -> str:
    lines = [
        "# Geometry vs Uni3D Nearest-Neighbor Comparison",
        "",
        f"- Created: `{summary['created_at_utc']}`",
        f"- Matched samples: `{summary['matched_sample_count']}`",
        f"- Top-k: `{summary['top_k']}`",
        f"- Embedding dir: `{summary['embedding_dir']}`",
        f"- Label path: `{summary['label_path']}`",
        "",
        "## Boundary",
        "",
        summary["boundary"],
        "",
        "## Overlap Summary",
        "",
    ]
    for key, value in summary["overlap_summary"].items():
        lines.append(
            f"- `{key}`: mean Jaccard `{value['mean_jaccard']:.4f}`, "
            f"mean overlap `{value['mean_overlap_count']:.2f}`"
        )
    lines.extend(["", "## Geometry Difference Summary", ""])
    for row in geometry_summary:
        lines.append(
            f"- `{row['method']}`: DBH diff mean `{float(row['dbh_cm_diff_mean']):.4f}`, "
            f"height diff mean `{float(row['height_m_diff_mean']):.4f}`"
        )
    lines.extend(["", "## Site / Species Summary", ""])
    for row in site_species_summary:
        lines.append(
            f"- `{row['method']}`: same-site rate `{float(row['same_site_rate']):.4f}`, "
            f"same-species rate `{float(row['same_species_rate']):.4f}`"
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "This experiment compares neighbor structure only. It does not support claims about tree health, risk, or management decisions.",
            "",
        ]
    )
    return "\n".join(lines)


def csv_safe_row(row: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, bool):
            output[key] = "true" if value else "false"
        elif isinstance(value, (dict, list)):
            output[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            output[key] = value
    return output


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(make_json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    try:
        if value != value:
            return None
    except Exception:
        pass
    return value


if __name__ == "__main__":
    raise SystemExit(main())
