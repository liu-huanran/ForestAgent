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
DEFAULT_FULL_MATRIX_SAVE_THRESHOLD = 1000


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
    parser.add_argument("--max-samples", type=int, default=None, help="Analyze a deterministic sample. 0 means all.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed used with --max-samples.")
    parser.add_argument(
        "--full-matrix-threshold",
        type=int,
        default=DEFAULT_FULL_MATRIX_SAVE_THRESHOLD,
        help="Save cosine_similarity.csv by default only when analyzed samples are at or below this count.",
    )
    matrix_group = parser.add_mutually_exclusive_group()
    matrix_group.add_argument(
        "--no-full-matrix",
        action="store_true",
        help="Do not save cosine_similarity.csv.",
    )
    matrix_group.add_argument(
        "--save-full-matrix",
        action="store_true",
        help="Force saving cosine_similarity.csv even for large analyzed samples.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        records, invalid_records = load_embedding_records(Path(args.embedding_dir))
        sampled_records = sample_records(records, max_samples=normalize_max_samples(args.max_samples), seed=args.seed)
        analysis = analyze_embeddings(
            records=sampled_records,
            invalid_records=invalid_records,
            top_k=args.top_k,
            total_valid_count=len(records),
            max_samples=args.max_samples,
            seed=args.seed,
        )
        save_full_matrix = should_save_full_matrix(
            analyzed_count=len(sampled_records),
            threshold=args.full_matrix_threshold,
            no_full_matrix=args.no_full_matrix,
            force_save=args.save_full_matrix,
        )
        write_analysis_outputs(output_dir=Path(args.output_dir), analysis=analysis, save_full_matrix=save_full_matrix)
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
    total_valid_count: int | None = None,
    max_samples: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    if top_k < 0:
        raise ValueError("--top-k must be non-negative.")
    if not records:
        return {
            "records": [],
            "cosine_matrix": np.zeros((0, 0), dtype=np.float32),
            "nearest_neighbors": [],
            "summary": build_empty_summary(
                invalid_records=invalid_records,
                total_valid_count=total_valid_count or 0,
                max_samples=max_samples,
                seed=seed,
            ),
            "invalid_records": invalid_records,
        }

    embeddings = np.stack([record.embedding_l2 for record in records], axis=0).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("At least one embedding_l2 vector has near-zero norm.")
    unit_embeddings = embeddings / norms
    cosine_matrix = np.clip(unit_embeddings @ unit_embeddings.T, -1.0, 1.0).astype(np.float32)
    nearest_neighbors = build_nearest_neighbors(records=records, cosine_matrix=cosine_matrix, top_k=top_k)
    summary = build_summary(
        records=records,
        cosine_matrix=cosine_matrix,
        invalid_records=invalid_records,
        total_valid_count=total_valid_count if total_valid_count is not None else len(records),
        max_samples=max_samples,
        seed=seed,
    )
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
    total_valid_count: int,
    max_samples: int | None,
    seed: int,
) -> dict[str, Any]:
    count = len(records)
    off_diagonal = cosine_matrix[~np.eye(count, dtype=bool)] if count > 1 else np.asarray([], dtype=np.float32)
    duplicate_pairs = collect_pairs(records, cosine_matrix, threshold=EXACT_DUPLICATE_THRESHOLD)
    near_duplicate_pairs = collect_pairs(records, cosine_matrix, threshold=NEAR_DUPLICATE_THRESHOLD)
    stats = pairwise_stats(off_diagonal)
    return {
        "sample_count": total_valid_count,
        "analyzed_sample_count": count,
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
        "max_samples": max_samples,
        "seed": seed,
    }


def build_empty_summary(
    *,
    invalid_records: list[dict[str, Any]],
    total_valid_count: int,
    max_samples: int | None,
    seed: int,
) -> dict[str, Any]:
    return {
        "sample_count": total_valid_count,
        "analyzed_sample_count": 0,
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
        "max_samples": max_samples,
        "seed": seed,
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


def normalize_max_samples(max_samples: int | None) -> int | None:
    if max_samples is None or max_samples == 0:
        return None
    if max_samples < 0:
        raise ValueError("--max-samples must be non-negative. Use 0 or omit it for all embeddings.")
    return max_samples


def sample_records(records: list[EmbeddingRecord], *, max_samples: int | None, seed: int) -> list[EmbeddingRecord]:
    if max_samples is None or len(records) <= max_samples:
        return records
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(records), size=max_samples, replace=False))
    return [records[int(index)] for index in indices]


def should_save_full_matrix(
    *,
    analyzed_count: int,
    threshold: int,
    no_full_matrix: bool,
    force_save: bool,
) -> bool:
    if no_full_matrix:
        return False
    if force_save:
        return True
    return analyzed_count <= threshold


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


def write_analysis_outputs(*, output_dir: Path, analysis: dict[str, Any], save_full_matrix: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / "cosine_similarity.csv"
    if save_full_matrix:
        write_similarity_csv(
            matrix_path,
            records=analysis["records"],
            cosine_matrix=analysis["cosine_matrix"],
        )
        analysis["summary"]["cosine_similarity_csv_path"] = str(matrix_path)
    else:
        analysis["summary"]["cosine_similarity_csv_path"] = None
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
    print(f"analyzed_embeddings: {summary['analyzed_sample_count']}")
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
