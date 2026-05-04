"""Query nearest neighbors from saved Uni3D embedding .npz files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Return top-k nearest Uni3D embeddings for one sample_id.")
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = query_neighbors(
        embedding_dir=Path(args.embedding_dir),
        sample_id=args.sample_id,
        top_k=args.top_k,
    )
    text = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    return 0


def query_neighbors(*, embedding_dir: Path, sample_id: str, top_k: int) -> dict[str, Any]:
    if top_k < 1:
        raise ValueError("--top-k must be positive.")
    records = load_embeddings(embedding_dir)
    ids = [record["tree_id"] for record in records]
    if sample_id not in ids:
        raise ValueError(f"sample_id not found: {sample_id}")
    embeddings = np.stack([record["embedding_l2"] for record in records], axis=0).astype(np.float32)
    embeddings = l2_normalize(embeddings)
    index = ids.index(sample_id)
    similarities = embeddings @ embeddings[index]
    candidates = [candidate for candidate in range(len(ids)) if candidate != index]
    candidates.sort(key=lambda candidate: float(similarities[candidate]), reverse=True)
    nearest = [
        {
            "tree_id": ids[candidate],
            "cosine_similarity": float(similarities[candidate]),
            "embedding_path": str(records[candidate]["embedding_path"]),
            "source_path": records[candidate]["source_path"],
        }
        for candidate in candidates[:top_k]
    ]
    return {
        "query_tree_id": sample_id,
        "query_embedding_path": str(records[index]["embedding_path"]),
        "query_source_path": records[index]["source_path"],
        "top_k": top_k,
        "nearest": nearest,
    }


def load_embeddings(embedding_dir: Path) -> list[dict[str, Any]]:
    if not embedding_dir.is_dir():
        raise ValueError(f"Embedding directory not found: {embedding_dir}")
    records: list[dict[str, Any]] = []
    for path in sorted(embedding_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            if "embedding_l2" not in payload:
                continue
            embedding = np.asarray(payload["embedding_l2"], dtype=np.float32)
            if embedding.ndim == 2 and embedding.shape[0] == 1:
                embedding = embedding[0]
            if embedding.ndim != 1 or not np.isfinite(embedding).all():
                continue
            tree_id = read_scalar(payload, "tree_id") or infer_tree_id(path)
            records.append(
                {
                    "tree_id": tree_id,
                    "embedding_path": path,
                    "source_path": read_scalar(payload, "source_path"),
                    "embedding_l2": embedding,
                }
            )
    if not records:
        raise ValueError(f"No valid .npz embeddings found in {embedding_dir}")
    return records


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


def infer_tree_id(path: Path) -> str:
    stem = path.stem
    parts = stem.split("_", 1)
    return parts[1] if len(parts) == 2 and parts[0].isdigit() else stem


def l2_normalize(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("At least one embedding vector has near-zero norm.")
    return (values / norm).astype(np.float32)


if __name__ == "__main__":
    raise SystemExit(main())
