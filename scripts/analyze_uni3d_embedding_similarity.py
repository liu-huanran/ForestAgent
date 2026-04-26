"""Analyze pairwise cosine similarity for offline Uni3D embedding batches."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


EXACT_DUPLICATE_THRESHOLD = 0.999999
NEAR_DUPLICATE_THRESHOLD = 0.999
ALL_NEAR_ONE_THRESHOLD = 0.99


@dataclass(frozen=True)
class EmbeddingRecord:
    tree_id: str
    source_path: str | None
    embedding_path: Path
    embedding_l2: np.ndarray


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute pairwise cosine similarity for saved Uni3D embedding .npz files.",
    )
    parser.add_argument("--embedding-dir", required=True, help="Directory containing per-tree .npz embeddings.")
    parser.add_argument("--output-dir", required=True, help="Directory for similarity analysis outputs.")
    parser.add_argument("--top-k", type=int, default=5, help="Nearest neighbors to report for each tree.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        records, invalid_records = load_embedding_records(Path(args.embedding_dir))
        analysis = analyze_embeddings(records=records, invalid_records=invalid_records, top_k=args.top_k)
        write_analysis_outputs(output_dir=Path(args.output_dir), analysis=analysis)
        print_similarity_summary(analysis)
        return 0 if analysis["summary"]["sample_count"] > 0 and not invalid_records else 2
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def load_embedding_records(embedding_dir: Path) -> tuple[list[EmbeddingRecord], list[dict[str, Any]]]:
    if not embedding_dir.is_dir():
        raise ValueError(f"embedding directory not found: {embedding_dir}")
    paths = sorted(path for path in embedding_dir.glob("*.npz") if path.is_file())
    records: list[EmbeddingRecord] = []
    invalid_records: list[dict[str, Any]] = []
    expected_shape: tuple[int, ...] | None = None

    for path in paths:
        try:
            with np.load(path, allow_pickle=False) as payload:
                if "embedding_l2" not in payload:
                    raise ValueError("missing embedding_l2")
                embedding = normalize_embedding_shape(payload["embedding_l2"], path=path)
                if not np.isfinite(embedding).all():
                    raise ValueError("embedding_l2 contains NaN or Inf")
                if expected_shape is None:
                    expected_shape = tuple(int(v) for v in embedding.shape)
                elif tuple(int(v) for v in embedding.shape) != expected_shape:
                    raise ValueError(
                        f"embedding_l2 shape mismatch: got {embedding.shape}, expected {expected_shape}"
                    )
                tree_id = read_optional_scalar(payload, "tree_id") or path.stem
                source_path = read_optional_scalar(payload, "source_path")
                records.append(
                    EmbeddingRecord(
                        tree_id=tree_id,
                        source_path=source_path,
                        embedding_path=path,
                        embedding_l2=embedding.astype(np.float32, copy=False),
                    )
                )
        except Exception as exc:
            invalid_records.append(
                {
                    "embedding_path": str(path),
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                }
            )
    return records, invalid_records


def normalize_embedding_shape(value: np.ndarray, *, path: Path) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1:
        raise ValueError(f"{path} embedding_l2 must be shaped [D] or [1, D], got {value.shape}.")
    if array.shape[0] == 0:
        raise ValueError(f"{path} embedding_l2 is empty.")
    return array


def read_optional_scalar(payload: Any, key: str) -> str | None:
    if key not in payload:
        return None
    value = payload[key]
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return str(value.item())
        if value.size == 1:
            return str(value.reshape(-1)[0])
    return str(value)


def analyze_embeddings(
    *,
    records: list[EmbeddingRecord],
    invalid_records: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    if top_k < 0:
        raise ValueError("--top-k must be non-negative.")
    if not records:
        return {
            "records": [],
            "cosine_matrix": np.zeros((0, 0), dtype=np.float32),
            "nearest_neighbors": [],
            "summary": build_empty_summary(invalid_records=invalid_records),
            "invalid_records": invalid_records,
        }

    embeddings = np.stack([record.embedding_l2 for record in records], axis=0).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("At least one embedding_l2 vector has near-zero norm.")
    unit_embeddings = embeddings / norms
    cosine_matrix = np.clip(unit_embeddings @ unit_embeddings.T, -1.0, 1.0).astype(np.float32)
    nearest_neighbors = build_nearest_neighbors(records=records, cosine_matrix=cosine_matrix, top_k=top_k)
    summary = build_summary(records=records, cosine_matrix=cosine_matrix, invalid_records=invalid_records)
    return {
        "records": records,
        "cosine_matrix": cosine_matrix,
        "nearest_neighbors": nearest_neighbors,
        "summary": summary,
        "invalid_records": invalid_records,
    }


def build_nearest_neighbors(
    *,
    records: list[EmbeddingRecord],
    cosine_matrix: np.ndarray,
    top_k: int,
) -> list[dict[str, Any]]:
    neighbors: list[dict[str, Any]] = []
    count = len(records)
    for index, record in enumerate(records):
        candidate_indices = [candidate for candidate in range(count) if candidate != index]
        candidate_indices.sort(key=lambda candidate: float(cosine_matrix[index, candidate]), reverse=True)
        top_indices = candidate_indices[:top_k]
        neighbors.append(
            {
                "tree_id": record.tree_id,
                "embedding_path": str(record.embedding_path),
                "nearest": [
                    {
                        "tree_id": records[candidate].tree_id,
                        "embedding_path": str(records[candidate].embedding_path),
                        "cosine_similarity": float(cosine_matrix[index, candidate]),
                    }
                    for candidate in top_indices
                ],
            }
        )
    return neighbors


def build_summary(
    *,
    records: list[EmbeddingRecord],
    cosine_matrix: np.ndarray,
    invalid_records: list[dict[str, Any]],
) -> dict[str, Any]:
    count = len(records)
    off_diagonal = cosine_matrix[~np.eye(count, dtype=bool)] if count > 1 else np.asarray([], dtype=np.float32)
    duplicate_pairs = collect_pairs(records, cosine_matrix, threshold=EXACT_DUPLICATE_THRESHOLD)
    near_duplicate_pairs = collect_pairs(records, cosine_matrix, threshold=NEAR_DUPLICATE_THRESHOLD)
    stats = pairwise_stats(off_diagonal)
    return {
        "sample_count": count,
        "embedding_dim": int(records[0].embedding_l2.shape[0]) if records else None,
        "invalid_count": len(invalid_records),
        "invalid_records": invalid_records,
        "pairwise_similarity": stats,
        "has_nan_or_inf": invalid_records_include_nan_or_inf(invalid_records),
        "has_exact_or_near_exact_duplicate": bool(duplicate_pairs),
        "has_near_duplicate": bool(near_duplicate_pairs),
        "duplicate_threshold": EXACT_DUPLICATE_THRESHOLD,
        "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
        "duplicate_pairs": duplicate_pairs[:20],
        "near_duplicate_pairs": near_duplicate_pairs[:20],
        "all_pairwise_similarity_near_one": bool(
            off_diagonal.size > 0 and float(np.min(off_diagonal)) >= ALL_NEAR_ONE_THRESHOLD
        ),
        "all_near_one_threshold": ALL_NEAR_ONE_THRESHOLD,
    }


def build_empty_summary(*, invalid_records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_count": 0,
        "embedding_dim": None,
        "invalid_count": len(invalid_records),
        "invalid_records": invalid_records,
        "pairwise_similarity": {"min": None, "max": None, "mean": None, "std": None},
        "has_nan_or_inf": invalid_records_include_nan_or_inf(invalid_records),
        "has_exact_or_near_exact_duplicate": False,
        "has_near_duplicate": False,
        "duplicate_threshold": EXACT_DUPLICATE_THRESHOLD,
        "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
        "duplicate_pairs": [],
        "near_duplicate_pairs": [],
        "all_pairwise_similarity_near_one": False,
        "all_near_one_threshold": ALL_NEAR_ONE_THRESHOLD,
    }


def pairwise_stats(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {"min": None, "max": None, "mean": None, "std": None}
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def invalid_records_include_nan_or_inf(invalid_records: list[dict[str, Any]]) -> bool:
    return any(
        "nan" in item.get("failure_reason", "").lower()
        or "inf" in item.get("failure_reason", "").lower()
        for item in invalid_records
    )


def collect_pairs(
    records: list[EmbeddingRecord],
    cosine_matrix: np.ndarray,
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            similarity = float(cosine_matrix[left, right])
            if similarity >= threshold:
                pairs.append(
                    {
                        "left_tree_id": records[left].tree_id,
                        "right_tree_id": records[right].tree_id,
                        "cosine_similarity": similarity,
                    }
                )
    pairs.sort(key=lambda item: item["cosine_similarity"], reverse=True)
    return pairs


def write_analysis_outputs(*, output_dir: Path, analysis: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_similarity_csv(
        output_dir / "cosine_similarity.csv",
        records=analysis["records"],
        cosine_matrix=analysis["cosine_matrix"],
    )
    (output_dir / "nearest_neighbors.json").write_text(
        json.dumps(analysis["nearest_neighbors"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "similarity_summary.json").write_text(
        json.dumps(analysis["summary"], indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_similarity_csv(path: Path, *, records: list[EmbeddingRecord], cosine_matrix: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tree_id", *[record.tree_id for record in records]])
        for index, record in enumerate(records):
            writer.writerow(
                [
                    record.tree_id,
                    *[f"{float(cosine_matrix[index, column]):.8f}" for column in range(len(records))],
                ]
            )


def print_similarity_summary(analysis: dict[str, Any]) -> None:
    summary = analysis["summary"]
    stats = summary["pairwise_similarity"]
    print(f"loaded_embeddings: {summary['sample_count']}")
    print(f"invalid_embeddings: {summary['invalid_count']}")
    print(f"embedding_dim: {summary['embedding_dim']}")
    print(
        "similarity min/mean/max: "
        f"{stats['min']} / {stats['mean']} / {stats['max']}"
    )
    if summary["near_duplicate_pairs"]:
        print("top suspicious near-duplicate pairs:")
        for pair in summary["near_duplicate_pairs"][:5]:
            print(
                f"  {pair['left_tree_id']} <-> {pair['right_tree_id']}: "
                f"{pair['cosine_similarity']:.8f}"
            )
    if summary["all_pairwise_similarity_near_one"]:
        print("WARNING: all pairwise similarities are near 1; embeddings may lack useful separation.")
    if analysis["invalid_records"]:
        print("invalid embedding files:")
        for item in analysis["invalid_records"]:
            print(f"  {item['embedding_path']}: {item['failure_reason']}")


if __name__ == "__main__":
    raise SystemExit(main())
