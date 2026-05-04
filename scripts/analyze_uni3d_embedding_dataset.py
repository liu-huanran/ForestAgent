"""First-pass offline analysis for saved Uni3D embedding datasets.

This script is intentionally independent from the main QA pipeline. It reads
saved .npz embeddings, performs dataset health checks, near-duplicate/outlier
analysis, a PCA projection, and optional lightweight label probes when a label
spreadsheet is provided.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EmbeddingRecord:
    tree_id: str
    embedding_path: Path
    source_path: str | None
    metadata: dict[str, Any]
    embedding_l2: np.ndarray


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a saved Uni3D embedding directory without touching the main system.",
    )
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--labels-path", default=None, help="Optional .xlsx/.csv label table for probe checks.")
    parser.add_argument("--top-pairs", type=int, default=30)
    parser.add_argument("--top-outliers", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.999)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    embedding_dir = Path(args.embedding_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records, invalid_records = load_embedding_records(embedding_dir)
    if not records:
        raise SystemExit("No valid embeddings were found.")

    ids = [record.tree_id for record in records]
    embeddings = np.stack([record.embedding_l2 for record in records], axis=0).astype(np.float32)
    embeddings = l2_normalize(embeddings)
    cosine = np.clip(embeddings @ embeddings.T, -1.0, 1.0)

    summary = build_dataset_summary(records=records, invalid_records=invalid_records, cosine=cosine)
    nearest = build_nearest_neighbors(ids=ids, records=records, cosine=cosine, top_k=args.top_k)
    pairs = build_near_duplicate_pairs(
        ids=ids,
        cosine=cosine,
        threshold=args.near_duplicate_threshold,
        limit=args.top_pairs,
    )
    outliers = build_outlier_rows(ids=ids, records=records, cosine=cosine, limit=args.top_outliers)
    pca = compute_pca(embeddings, n_components=2)
    labels = load_label_table(Path(args.labels_path)) if args.labels_path else None
    label_inventory = build_label_inventory(ids=ids, labels=labels)
    probe_summary: dict[str, Any] | None = None
    probe_predictions: list[dict[str, Any]] = []
    if labels is not None and label_inventory["matched_count"] > 0:
        probe_summary, probe_predictions = run_label_probe(
            ids=ids,
            embeddings=embeddings,
            cosine=cosine,
            labels=labels,
            alpha=args.ridge_alpha,
            seed=args.seed,
        )

    write_json(output_dir / "dataset_summary.json", summary)
    write_json(output_dir / "nearest_neighbors.json", nearest)
    write_json(output_dir / "near_duplicate_pairs.json", pairs)
    write_rows(output_dir / "near_duplicate_pairs.csv", pairs)
    write_json(output_dir / "outlier_samples.json", outliers)
    write_rows(output_dir / "outlier_samples.csv", outliers)
    write_pca_outputs(output_dir=output_dir, ids=ids, records=records, pca=pca, labels=labels)
    write_json(output_dir / "label_inventory.json", label_inventory)
    if probe_summary is not None:
        write_json(output_dir / "label_probe_summary.json", probe_summary)
        write_rows(output_dir / "label_probe_predictions.csv", probe_predictions)

    run_summary = {
        "embedding_dir": str(embedding_dir),
        "output_dir": str(output_dir),
        "dataset_summary_path": str(output_dir / "dataset_summary.json"),
        "near_duplicate_pairs_path": str(output_dir / "near_duplicate_pairs.csv"),
        "outlier_samples_path": str(output_dir / "outlier_samples.csv"),
        "nearest_neighbors_path": str(output_dir / "nearest_neighbors.json"),
        "pca_coordinates_path": str(output_dir / "pca_coordinates.csv"),
        "pca_svg_path": str(output_dir / "pca_embedding.svg"),
        "label_inventory_path": str(output_dir / "label_inventory.json"),
        "label_probe_summary_path": str(output_dir / "label_probe_summary.json") if probe_summary else None,
    }
    write_json(output_dir / "analysis_run_summary.json", run_summary)

    print(f"loaded_embeddings={summary['sample_count']}")
    print(f"invalid_embeddings={summary['invalid_count']}")
    print(f"embedding_dim={summary['embedding_dim']}")
    print(
        "similarity min/mean/max="
        f"{summary['pairwise_similarity']['min']} / "
        f"{summary['pairwise_similarity']['mean']} / "
        f"{summary['pairwise_similarity']['max']}"
    )
    print(f"output_dir={output_dir}")
    return 0


def load_embedding_records(embedding_dir: Path) -> tuple[list[EmbeddingRecord], list[dict[str, Any]]]:
    if not embedding_dir.is_dir():
        raise ValueError(f"Embedding directory not found: {embedding_dir}")
    records: list[EmbeddingRecord] = []
    invalid: list[dict[str, Any]] = []
    expected_dim: int | None = None
    for path in sorted(embedding_dir.glob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as payload:
                embedding = normalize_embedding_shape(payload["embedding_l2"], path=path)
                if not np.isfinite(embedding).all():
                    raise ValueError("embedding_l2 contains NaN or Inf")
                if expected_dim is None:
                    expected_dim = int(embedding.shape[0])
                elif int(embedding.shape[0]) != expected_dim:
                    raise ValueError(f"embedding dim mismatch: got {embedding.shape[0]}, expected {expected_dim}")
                tree_id = read_scalar(payload, "tree_id") or infer_tree_id(path)
                source_path = read_scalar(payload, "source_path")
                metadata = parse_metadata(read_scalar(payload, "metadata"))
                records.append(
                    EmbeddingRecord(
                        tree_id=tree_id,
                        embedding_path=path,
                        source_path=source_path,
                        metadata=metadata,
                        embedding_l2=embedding.astype(np.float32, copy=False),
                    )
                )
        except Exception as exc:
            invalid.append({"embedding_path": str(path), "failure_reason": f"{type(exc).__name__}: {exc}"})
    return records, invalid


def normalize_embedding_shape(value: np.ndarray, *, path: Path) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1:
        raise ValueError(f"{path} embedding_l2 must be shaped [D] or [1, D], got {value.shape}.")
    return array


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


def parse_metadata(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {"raw_metadata": value}
    return parsed if isinstance(parsed, dict) else {"metadata": parsed}


def infer_tree_id(path: Path) -> str:
    stem = path.stem
    parts = stem.split("_", 1)
    return parts[1] if len(parts) == 2 and parts[0].isdigit() else stem


def l2_normalize(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("At least one embedding vector has near-zero norm.")
    return (values / norm).astype(np.float32)


def build_dataset_summary(
    *,
    records: list[EmbeddingRecord],
    invalid_records: list[dict[str, Any]],
    cosine: np.ndarray,
) -> dict[str, Any]:
    count = len(records)
    off_diag = cosine[~np.eye(count, dtype=bool)] if count > 1 else np.asarray([], dtype=np.float32)
    l2_norms = [float(np.linalg.norm(record.embedding_l2)) for record in records]
    return {
        "sample_count": count,
        "invalid_count": len(invalid_records),
        "invalid_records": invalid_records,
        "embedding_dim": int(records[0].embedding_l2.shape[0]) if records else None,
        "tree_id_examples": [record.tree_id for record in records[:10]],
        "embedding_l2_norm": {
            "min": float(np.min(l2_norms)),
            "mean": float(np.mean(l2_norms)),
            "max": float(np.max(l2_norms)),
        },
        "pairwise_similarity": pairwise_stats(off_diag),
        "has_nan_or_inf": bool(invalid_records),
        "all_pairwise_similarity_near_one": bool(off_diag.size > 0 and float(np.min(off_diag)) >= 0.99),
    }


def pairwise_stats(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {"min": None, "mean": None, "max": None, "std": None}
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
    }


def build_nearest_neighbors(
    *,
    ids: list[str],
    records: list[EmbeddingRecord],
    cosine: np.ndarray,
    top_k: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, tree_id in enumerate(ids):
        candidates = [candidate for candidate in range(len(ids)) if candidate != index]
        candidates.sort(key=lambda candidate: float(cosine[index, candidate]), reverse=True)
        output.append(
            {
                "tree_id": tree_id,
                "embedding_path": str(records[index].embedding_path),
                "nearest": [
                    {
                        "tree_id": ids[candidate],
                        "embedding_path": str(records[candidate].embedding_path),
                        "cosine_similarity": float(cosine[index, candidate]),
                    }
                    for candidate in candidates[:top_k]
                ],
            }
        )
    return output


def build_near_duplicate_pairs(
    *,
    ids: list[str],
    cosine: np.ndarray,
    threshold: float,
    limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left in range(len(ids)):
        for right in range(left + 1, len(ids)):
            similarity = float(cosine[left, right])
            if similarity >= threshold:
                rows.append(
                    {
                        "left_tree_id": ids[left],
                        "right_tree_id": ids[right],
                        "cosine_similarity": similarity,
                    }
                )
    rows.sort(key=lambda row: row["cosine_similarity"], reverse=True)
    return rows[:limit]


def build_outlier_rows(
    *,
    ids: list[str],
    records: list[EmbeddingRecord],
    cosine: np.ndarray,
    limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    count = len(ids)
    for index, tree_id in enumerate(ids):
        mask = np.ones(count, dtype=bool)
        mask[index] = False
        similarities = cosine[index, mask]
        nearest_index = int(np.argmax(np.where(mask, cosine[index], -np.inf)))
        metadata = records[index].metadata
        rows.append(
            {
                "tree_id": tree_id,
                "embedding_path": str(records[index].embedding_path),
                "source_path": records[index].source_path,
                "mean_similarity_to_others": float(np.mean(similarities)),
                "median_similarity_to_others": float(np.median(similarities)),
                "max_neighbor_similarity": float(cosine[index, nearest_index]),
                "nearest_neighbor_tree_id": ids[nearest_index],
                "point_count": metadata.get("point_count"),
                "scale_radius": metadata.get("scale_radius"),
                "rgb_source": metadata.get("rgb_source"),
            }
        )
    rows.sort(key=lambda row: (row["mean_similarity_to_others"], row["max_neighbor_similarity"]))
    return rows[:limit]


def compute_pca(values: np.ndarray, n_components: int = 2) -> dict[str, Any]:
    centered = values - values.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    coordinates = centered @ components.T
    variance = singular_values**2
    total_variance = float(np.sum(variance))
    explained = (variance[:n_components] / total_variance).astype(float).tolist() if total_variance > 0 else []
    return {
        "coordinates": coordinates.astype(np.float32),
        "explained_variance_ratio": explained,
    }


def load_label_table(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"Label file not found: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        import pandas as pd

        table = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        import pandas as pd

        table = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported label file: {path}")
    if table.empty:
        return {}
    columns = list(table.columns)
    file_col = find_first_existing(columns, ["file_name", "对应的文件名", "对应文件名称"]) or columns[2]
    sample_ids = table[file_col].astype(str).str.replace(".las", "", regex=False)
    output: dict[str, dict[str, Any]] = {}
    for _, row in table.assign(sample_id=sample_ids).iterrows():
        sample_id = str(row["sample_id"])
        output[sample_id] = {str(column): to_jsonable(row[column]) for column in table.columns}
    return output


def find_first_existing(columns: list[Any], candidates: list[str]) -> Any | None:
    for candidate in candidates:
        for column in columns:
            if str(column) == candidate:
                return column
    return None


def build_label_inventory(ids: list[str], labels: dict[str, dict[str, Any]] | None) -> dict[str, Any]:
    if labels is None:
        return {"labels_available": False, "matched_count": 0, "matched_fraction": 0.0}
    matched = [sample_id for sample_id in ids if sample_id in labels]
    columns = sorted({column for row in labels.values() for column in row.keys()})
    return {
        "labels_available": True,
        "label_count": len(labels),
        "embedding_count": len(ids),
        "matched_count": len(matched),
        "matched_fraction": len(matched) / len(ids) if ids else 0.0,
        "columns": columns,
    }


def run_label_probe(
    *,
    ids: list[str],
    embeddings: np.ndarray,
    cosine: np.ndarray,
    labels: dict[str, dict[str, Any]],
    alpha: float,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    matched_indices = [index for index, sample_id in enumerate(ids) if sample_id in labels]
    matched_ids = [ids[index] for index in matched_indices]
    x = embeddings[matched_indices]
    label_rows = [labels[sample_id] for sample_id in matched_ids]
    species_key = find_first_existing(list(label_rows[0].keys()), ["树种", "species"])
    species_summary: dict[str, Any] | None = None
    predictions: list[dict[str, Any]] = []
    if species_key is not None:
        species = [str(row[species_key]) for row in label_rows]
        species_summary, species_predictions = run_knn_species_probe(
            ids=matched_ids,
            species=species,
            cosine=cosine[np.ix_(matched_indices, matched_indices)],
            k=5,
        )
        predictions.extend(species_predictions)
    numeric_results: dict[str, Any] = {}
    numeric_keys = [
        key
        for key in ["胸径cm", "树高m", "东西冠幅m", "南北冠幅m", "胸径", "树高", "东西冠幅", "南北冠幅"]
        if key in label_rows[0]
    ]
    for key in numeric_keys:
        y = np.asarray([float(row[key]) for row in label_rows], dtype=np.float64)
        if np.isfinite(y).sum() == len(y) and len(np.unique(y)) > 2:
            result, rows = run_ridge_probe(ids=matched_ids, x=x, y=y, target_name=key, alpha=alpha, seed=seed)
            numeric_results[key] = result
            predictions.extend(rows)
    return (
        {
            "matched_count": len(matched_ids),
            "species_probe": species_summary,
            "numeric_ridge_probe": numeric_results,
            "ridge_alpha": alpha,
            "seed": seed,
            "caution": "Lightweight probe only; not a formal experiment or model selection result.",
        },
        predictions,
    )


def run_knn_species_probe(
    *,
    ids: list[str],
    species: list[str],
    cosine: np.ndarray,
    k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    correct_1 = 0
    correct_k = 0
    rows: list[dict[str, Any]] = []
    for index, sample_id in enumerate(ids):
        candidates = [candidate for candidate in range(len(ids)) if candidate != index]
        candidates.sort(key=lambda candidate: float(cosine[index, candidate]), reverse=True)
        top = candidates[:k]
        top1 = species[top[0]]
        vote = majority_vote([species[candidate] for candidate in top])
        correct_1 += int(top1 == species[index])
        correct_k += int(vote == species[index])
        rows.append(
            {
                "probe_type": "species_knn",
                "target": "species",
                "tree_id": sample_id,
                "true_value": species[index],
                "prediction_top1": top1,
                "prediction_topk_vote": vote,
                "top1_neighbor": ids[top[0]],
                "top1_similarity": float(cosine[index, top[0]]),
            }
        )
    return (
        {
            "sample_count": len(ids),
            "class_counts": class_counts(species),
            "leave_one_out_top1_accuracy": correct_1 / len(ids) if ids else None,
            "leave_one_out_top5_vote_accuracy": correct_k / len(ids) if ids else None,
        },
        rows,
    )


def majority_vote(values: list[str]) -> str:
    counts = class_counts(values)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def class_counts(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def run_ridge_probe(
    *,
    ids: list[str],
    x: np.ndarray,
    y: np.ndarray,
    target_name: str,
    alpha: float,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    folds = make_folds(len(ids), fold_count=5, seed=seed)
    predictions = np.zeros_like(y)
    for test_indices in folds:
        train_mask = np.ones(len(ids), dtype=bool)
        train_mask[test_indices] = False
        x_train = x[train_mask]
        y_train = y[train_mask]
        x_test = x[test_indices]
        x_mean = x_train.mean(axis=0, keepdims=True)
        y_mean = float(y_train.mean())
        x_train_centered = x_train - x_mean
        x_test_centered = x_test - x_mean
        y_train_centered = y_train - y_mean
        kernel = x_train_centered @ x_train_centered.T
        weights = np.linalg.solve(kernel + alpha * np.eye(kernel.shape[0]), y_train_centered)
        coef = x_train_centered.T @ weights
        predictions[test_indices] = x_test_centered @ coef + y_mean
    error = predictions - y
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else None
    rows = [
        {
            "probe_type": "ridge_regression",
            "target": target_name,
            "tree_id": ids[index],
            "true_value": float(y[index]),
            "predicted_value": float(predictions[index]),
            "absolute_error": float(abs(error[index])),
        }
        for index in range(len(ids))
    ]
    return (
        {
            "sample_count": len(ids),
            "r2": r2,
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "target_min": float(np.min(y)),
            "target_max": float(np.max(y)),
        },
        rows,
    )


def make_folds(count: int, *, fold_count: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(count)
    rng.shuffle(indices)
    return [fold.astype(int) for fold in np.array_split(indices, fold_count) if fold.size > 0]


def write_pca_outputs(
    *,
    output_dir: Path,
    ids: list[str],
    records: list[EmbeddingRecord],
    pca: dict[str, Any],
    labels: dict[str, dict[str, Any]] | None,
) -> None:
    coords = np.asarray(pca["coordinates"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for index, tree_id in enumerate(ids):
        label = labels.get(tree_id, {}) if labels else {}
        species = label.get("树种") or label.get("species") or plot_group(tree_id)
        rows.append(
            {
                "tree_id": tree_id,
                "pc1": float(coords[index, 0]),
                "pc2": float(coords[index, 1]),
                "species_or_group": species,
                "embedding_path": str(records[index].embedding_path),
            }
        )
    write_rows(output_dir / "pca_coordinates.csv", rows)
    write_json(
        output_dir / "pca_summary.json",
        {"explained_variance_ratio": pca["explained_variance_ratio"]},
    )
    write_pca_svg(output_dir / "pca_embedding.svg", rows, pca["explained_variance_ratio"])


def plot_group(tree_id: str) -> str:
    return tree_id.split("_", 1)[0]


def write_pca_svg(path: Path, rows: list[dict[str, Any]], explained: list[float]) -> None:
    width = 1000
    height = 760
    margin = 70
    x_values = np.asarray([row["pc1"] for row in rows], dtype=np.float64)
    y_values = np.asarray([row["pc2"] for row in rows], dtype=np.float64)
    x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
    y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
    labels = sorted({str(row["species_or_group"]) for row in rows})
    palette = [
        "#3366cc",
        "#dc3912",
        "#ff9900",
        "#109618",
        "#990099",
        "#0099c6",
        "#dd4477",
        "#66aa00",
        "#b82e2e",
        "#316395",
    ]
    colors = {label: palette[index % len(palette)] for index, label in enumerate(labels)}

    def scale_x(value: float) -> float:
        return margin + (value - x_min) / max(x_max - x_min, 1e-12) * (width - 2 * margin)

    def scale_y(value: float) -> float:
        return height - margin - (value - y_min) / max(y_max - y_min, 1e-12) * (height - 2 * margin)

    title = "Uni3D embedding PCA"
    pc1 = explained[0] if len(explained) > 0 else 0.0
    pc2 = explained[1] if len(explained) > 1 else 0.0
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin}" y="34" font-size="24" font-family="sans-serif">{escape_xml(title)}</text>',
        f'<text x="{margin}" y="58" font-size="13" font-family="sans-serif">PC1 {pc1:.3f}, PC2 {pc2:.3f} explained variance ratio</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#333"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#333"/>',
    ]
    for row in rows:
        label = str(row["species_or_group"])
        lines.append(
            f'<circle cx="{scale_x(row["pc1"]):.2f}" cy="{scale_y(row["pc2"]):.2f}" r="4.0" '
            f'fill="{colors[label]}" opacity="0.72"><title>{escape_xml(row["tree_id"])} / {escape_xml(label)}</title></circle>'
        )
    legend_x = width - 190
    legend_y = 85
    lines.append(f'<rect x="{legend_x-12}" y="{legend_y-25}" width="170" height="{22*len(labels)+28}" fill="#fff" opacity="0.85" stroke="#ccc"/>')
    for index, label in enumerate(labels):
        y = legend_y + index * 22
        lines.append(f'<circle cx="{legend_x}" cy="{y}" r="5" fill="{colors[label]}"/>')
        lines.append(f'<text x="{legend_x+12}" y="{y+4}" font-size="12" font-family="sans-serif">{escape_xml(label)}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def escape_xml(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    try:
        if value != value:
            return None
    except Exception:
        pass
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(make_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(make_jsonable(rows))


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
