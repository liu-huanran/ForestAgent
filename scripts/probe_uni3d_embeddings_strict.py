"""Strict offline downstream probes for frozen Uni3D embeddings.

This script reads saved Uni3D embedding .npz files and a label table, then runs
lightweight train/test probes. It never runs Uni3D forward, never updates the
Uni3D backbone, and never touches the main QA pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


TASKS = ["species", "dbh", "height", "crown_ew", "crown_ns"]
FEATURE_SETS = ["uni3d", "geometry", "uni3d_plus_geometry"]
SPLITS = ["random", "stratified_random", "group", "leave_one_site_out"]
REGRESSION_TASKS = {"dbh", "height", "crown_ew", "crown_ns"}


@dataclass(frozen=True)
class EmbeddingRecord:
    tree_id: str
    embedding_path: Path
    source_path: str | None
    metadata: dict[str, Any]
    embedding: np.ndarray


@dataclass(frozen=True)
class LabelTable:
    rows_by_tree_id: dict[str, dict[str, Any]]
    columns: list[str]
    detected_columns: dict[str, str | None]
    inventory: dict[str, Any]


@dataclass(frozen=True)
class SampleRow:
    tree_id: str
    site_id: str | None
    embedding_path: str
    source_path: str | None
    metadata: dict[str, Any]
    embedding: np.ndarray
    labels: dict[str, Any]
    rgb_source: str | None
    is_near_duplicate_candidate: bool
    is_outlier_candidate: bool


@dataclass(frozen=True)
class SplitSpec:
    split_type: str
    split_id: str
    seed: int | None
    train_indices: list[int]
    test_indices: list[int]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run strict offline downstream probes on frozen Uni3D embeddings.",
    )
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--label-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--embedding-key", default="embedding_l2")
    parser.add_argument("--tree-id-column", default=None)
    parser.add_argument("--species-column", default=None)
    parser.add_argument("--dbh-column", default=None)
    parser.add_argument("--height-column", default=None)
    parser.add_argument("--crown-ew-column", default=None)
    parser.add_argument("--crown-ns-column", default=None)

    parser.add_argument("--tasks", nargs="+", default=["all"], choices=TASKS + ["all"])
    parser.add_argument("--feature-sets", nargs="+", default=["all"], choices=FEATURE_SETS + ["all"])

    parser.add_argument("--split", default="all", choices=SPLITS + ["all"])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--group-column", default=None)
    parser.add_argument("--site-id-from-tree-id", dest="site_id_from_tree_id", action="store_true", default=True)
    parser.add_argument("--no-site-id-from-tree-id", dest="site_id_from_tree_id", action="store_false")
    parser.add_argument("--site-id-delimiter", default="_")

    parser.add_argument("--near-duplicate-path", default=None)
    parser.add_argument("--outlier-path", default=None)
    parser.add_argument("--review-table-path", default=None)
    parser.add_argument("--exclude-near-duplicates", action="store_true")
    parser.add_argument("--exclude-outliers", action="store_true")
    parser.add_argument("--rgb-source", default="all", choices=["provided", "fallback_constant", "all"])
    parser.add_argument("--site-id", nargs="*", default=None)
    parser.add_argument("--min-train-samples", type=int, default=20)
    parser.add_argument("--min-test-samples", type=int, default=5)
    parser.add_argument("--min-class-count", type=int, default=2)

    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--standardize", dest="standardize", action="store_true", default=True)
    parser.add_argument("--no-standardize", dest="standardize", action="store_false")
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--save-predictions", dest="save_predictions", action="store_true", default=True)
    parser.add_argument("--no-save-predictions", dest="save_predictions", action="store_false")

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, overwrite=args.overwrite)

    tasks = expand_choice_list(args.tasks, TASKS)
    feature_sets = expand_choice_list(args.feature_sets, FEATURE_SETS)
    split_types = SPLITS if args.split == "all" else [args.split]

    embeddings, invalid_embeddings = load_embedding_records(Path(args.embedding_dir), args.embedding_key)
    labels = load_label_table(
        Path(args.label_path),
        tree_id_column=args.tree_id_column,
        species_column=args.species_column,
        dbh_column=args.dbh_column,
        height_column=args.height_column,
        crown_ew_column=args.crown_ew_column,
        crown_ns_column=args.crown_ns_column,
    )
    review_info = load_review_info(args.review_table_path)
    near_duplicate_ids = load_near_duplicate_ids(args.near_duplicate_path, review_info)
    outlier_ids = load_outlier_ids(args.outlier_path, review_info)
    aligned_rows, alignment_report = align_samples(
        embeddings=embeddings,
        invalid_embeddings=invalid_embeddings,
        labels=labels,
        tasks=tasks,
        site_id_from_tree_id=args.site_id_from_tree_id,
        site_id_delimiter=args.site_id_delimiter,
        group_column=args.group_column,
        review_info=review_info,
        near_duplicate_ids=near_duplicate_ids,
        outlier_ids=outlier_ids,
    )
    filtered_rows, filter_report = filter_samples(
        aligned_rows,
        rgb_source=args.rgb_source,
        site_ids=args.site_id,
        exclude_near_duplicates=args.exclude_near_duplicates,
        exclude_outliers=args.exclude_outliers,
    )

    write_json(output_dir / "label_inventory.json", labels.inventory)
    write_json(output_dir / "alignment_report.json", alignment_report | {"filter_report": filter_report})

    context = {
        "args": args,
        "tasks": tasks,
        "feature_sets": feature_sets,
        "split_types": split_types,
        "label_columns": labels.detected_columns,
        "filter_report": filter_report,
        "output_dir": output_dir,
    }
    results, predictions, per_class_rows = run_all_probes(filtered_rows, context)

    write_rows(output_dir / "probe_results.csv", results)
    if args.save_predictions:
        write_rows(output_dir / "per_split_predictions.csv", predictions)
    else:
        write_rows(output_dir / "per_split_predictions.csv", [])
    write_rows(output_dir / "per_class_metrics.csv", per_class_rows)

    summary = build_probe_summary(
        results=results,
        rows=filtered_rows,
        tasks=tasks,
        feature_sets=feature_sets,
        split_types=split_types,
        filter_report=filter_report,
        alignment_report=alignment_report,
    )
    write_json(output_dir / "probe_summary.json", summary)
    write_json(output_dir / "run_config.json", build_run_config(args, labels.detected_columns))

    print("offline downstream probe only: Uni3D backbone was not trained or run")
    print(f"matched_samples={alignment_report['matched_count']}")
    print(f"filtered_samples={len(filtered_rows)}")
    print(f"probe_results={output_dir / 'probe_results.csv'}")
    print(f"probe_summary={output_dir / 'probe_summary.json'}")
    return 0


def prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_outputs = [
        output_dir / "probe_results.csv",
        output_dir / "probe_summary.json",
        output_dir / "alignment_report.json",
    ]
    if not overwrite and any(path.exists() for path in existing_outputs):
        raise SystemExit(f"Output directory already contains probe results; pass --overwrite: {output_dir}")
    (output_dir / "confusion_matrices").mkdir(exist_ok=True)


def expand_choice_list(values: list[str], allowed: list[str]) -> list[str]:
    return allowed[:] if "all" in values else values


def load_embedding_records(embedding_dir: Path, embedding_key: str) -> tuple[list[EmbeddingRecord], list[dict[str, Any]]]:
    if not embedding_dir.is_dir():
        raise ValueError(f"Embedding directory not found: {embedding_dir}")
    records: list[EmbeddingRecord] = []
    invalid: list[dict[str, Any]] = []
    expected_dim: int | None = None
    for path in sorted(embedding_dir.glob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as payload:
                if embedding_key not in payload:
                    raise ValueError(f"missing embedding key: {embedding_key}")
                embedding = normalize_embedding_shape(payload[embedding_key], path=path)
                if not np.isfinite(embedding).all():
                    raise ValueError(f"{embedding_key} contains NaN or Inf")
                if expected_dim is None:
                    expected_dim = int(embedding.shape[0])
                elif int(embedding.shape[0]) != expected_dim:
                    raise ValueError(f"embedding dim mismatch: got {embedding.shape[0]}, expected {expected_dim}")
                tree_id = normalize_tree_id(read_scalar(payload, "tree_id") or infer_tree_id(path))
                records.append(
                    EmbeddingRecord(
                        tree_id=tree_id,
                        embedding_path=path,
                        source_path=read_scalar(payload, "source_path"),
                        metadata=parse_metadata(read_scalar(payload, "metadata")),
                        embedding=embedding.astype(np.float32, copy=False),
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
        raise ValueError(f"{path} embedding must be shaped [D] or [1, D], got {array.shape}")
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


def normalize_tree_id(value: Any) -> str:
    text = str(value).strip()
    if text.lower() in {"", "nan", "none"}:
        return ""
    text = text.replace("\\", "/").split("/")[-1]
    suffix = Path(text).suffix.lower()
    if suffix in {".las", ".laz", ".npy", ".npz", ".txt", ".csv"}:
        text = Path(text).stem
    return text


def load_label_table(
    label_path: Path,
    *,
    tree_id_column: str | None,
    species_column: str | None,
    dbh_column: str | None,
    height_column: str | None,
    crown_ew_column: str | None,
    crown_ns_column: str | None,
) -> LabelTable:
    if not label_path.is_file():
        raise ValueError(f"Label file not found: {label_path}")
    table = read_table(label_path)
    columns = [str(column) for column in table.columns]
    detected = detect_label_columns(
        columns,
        tree_id_column=tree_id_column,
        species_column=species_column,
        dbh_column=dbh_column,
        height_column=height_column,
        crown_ew_column=crown_ew_column,
        crown_ns_column=crown_ns_column,
    )
    tree_col = detected["tree_id"]
    if tree_col is None:
        raise ValueError("Could not detect tree_id column; pass --tree-id-column.")

    rows_by_tree_id: dict[str, dict[str, Any]] = {}
    duplicate_tree_ids: list[str] = []
    for _, row in table.iterrows():
        tree_id = normalize_tree_id(row[tree_col])
        if not tree_id:
            continue
        if tree_id in rows_by_tree_id:
            duplicate_tree_ids.append(tree_id)
        rows_by_tree_id[tree_id] = {str(column): to_jsonable(row[column]) for column in table.columns}

    inventory = build_label_inventory(table, detected, rows_by_tree_id, duplicate_tree_ids)
    return LabelTable(rows_by_tree_id=rows_by_tree_id, columns=columns, detected_columns=detected, inventory=inventory)


def read_table(path: Path) -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to read label tables (.xlsx/.csv).") from exc
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported label file: {path}")


def detect_label_columns(
    columns: list[str],
    *,
    tree_id_column: str | None,
    species_column: str | None,
    dbh_column: str | None,
    height_column: str | None,
    crown_ew_column: str | None,
    crown_ns_column: str | None,
) -> dict[str, str | None]:
    return {
        "tree_id": detect_column(
            columns,
            tree_id_column,
            ["tree_id", "sample_id", "id", "对应的文件名", "对应文件名称", "file_name", "filename"],
        ),
        "species": detect_column(columns, species_column, ["树种", "species", "taxon"]),
        "dbh": detect_column(columns, dbh_column, ["胸径cm", "胸径", "dbh", "DBH"]),
        "height": detect_column(columns, height_column, ["树高m", "树高", "height"]),
        "crown_ew": detect_column(columns, crown_ew_column, ["东西冠幅m", "东西冠幅", "crown_ew"]),
        "crown_ns": detect_column(columns, crown_ns_column, ["南北冠幅m", "南北冠幅", "crown_ns"]),
    }


def detect_column(columns: list[str], provided: str | None, candidates: list[str]) -> str | None:
    if provided:
        if provided not in columns:
            raise ValueError(f"Requested label column not found: {provided}")
        return provided
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def build_label_inventory(table: Any, detected: dict[str, str | None], rows: dict[str, dict[str, Any]], duplicates: list[str]) -> dict[str, Any]:
    column_info: dict[str, Any] = {}
    for column in table.columns:
        values = table[column]
        non_null = values.dropna()
        examples = [to_jsonable(value) for value in non_null.head(5).tolist()]
        column_info[str(column)] = {
            "non_null_count": int(non_null.shape[0]),
            "dtype": str(values.dtype),
            "examples": examples,
        }
    return {
        "label_count": int(table.shape[0]),
        "tree_id_count": len(rows),
        "columns": [str(column) for column in table.columns],
        "detected_columns": detected,
        "column_info": column_info,
        "duplicate_tree_ids": sorted(set(duplicates)),
    }


def to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    try:
        if bool(np.isnan(value)):
            return None
    except Exception:
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_review_info(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    rows = read_rows_path(Path(path))
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        tree_id = normalize_tree_id(row.get("tree_id", ""))
        if tree_id:
            output[tree_id] = row
    return output


def load_near_duplicate_ids(path: str | None, review_info: dict[str, dict[str, Any]]) -> set[str]:
    ids: set[str] = {
        tree_id for tree_id, row in review_info.items() if str(row.get("candidate_type", "")).lower() in {"near_duplicate", "both"}
    }
    if path:
        for row in read_rows_path(Path(path)):
            for key in ["left_tree_id", "right_tree_id", "tree_id"]:
                if row.get(key):
                    ids.add(normalize_tree_id(row[key]))
    return ids


def load_outlier_ids(path: str | None, review_info: dict[str, dict[str, Any]]) -> set[str]:
    ids: set[str] = {
        tree_id for tree_id, row in review_info.items() if str(row.get("candidate_type", "")).lower() in {"outlier", "both"}
    }
    if path:
        for row in read_rows_path(Path(path)):
            if row.get("tree_id"):
                ids.add(normalize_tree_id(row["tree_id"]))
    return ids


def read_rows_path(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"Candidate file not found: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ["rows", "results", "items"]:
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            raise ValueError(f"Expected list-like JSON in {path}")
        return [dict(item) for item in payload]
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise ValueError(f"Unsupported candidate file: {path}")


def align_samples(
    *,
    embeddings: list[EmbeddingRecord],
    invalid_embeddings: list[dict[str, Any]],
    labels: LabelTable,
    tasks: list[str],
    site_id_from_tree_id: bool,
    site_id_delimiter: str,
    group_column: str | None,
    review_info: dict[str, dict[str, Any]],
    near_duplicate_ids: set[str],
    outlier_ids: set[str],
) -> tuple[list[SampleRow], dict[str, Any]]:
    label_ids = set(labels.rows_by_tree_id)
    embedding_ids = {record.tree_id for record in embeddings}
    rows: list[SampleRow] = []
    for record in embeddings:
        label_row = labels.rows_by_tree_id.get(record.tree_id)
        if label_row is None:
            continue
        review_row = review_info.get(record.tree_id, {})
        rows.append(
            SampleRow(
                tree_id=record.tree_id,
                site_id=resolve_site_id(
                    tree_id=record.tree_id,
                    label_row=label_row,
                    group_column=group_column,
                    site_id_from_tree_id=site_id_from_tree_id,
                    site_id_delimiter=site_id_delimiter,
                ),
                embedding_path=str(record.embedding_path),
                source_path=record.source_path,
                metadata=record.metadata,
                embedding=record.embedding,
                labels=label_row,
                rgb_source=resolve_rgb_source(record.metadata, review_row),
                is_near_duplicate_candidate=record.tree_id in near_duplicate_ids,
                is_outlier_candidate=record.tree_id in outlier_ids,
            )
        )
    missing_counts = {
        task: count_missing_targets(rows, task, labels.detected_columns.get(task if task != "species" else "species"))
        for task in tasks
    }
    return rows, {
        "embedding_count": len(embeddings),
        "invalid_embedding_count": len(invalid_embeddings),
        "invalid_embeddings": invalid_embeddings[:50],
        "label_count": len(label_ids),
        "matched_count": len(rows),
        "unmatched_embeddings": sorted(embedding_ids - label_ids)[:200],
        "unmatched_labels": sorted(label_ids - embedding_ids)[:200],
        "missing_label_counts_per_task": missing_counts,
    }


def resolve_site_id(
    *,
    tree_id: str,
    label_row: dict[str, Any],
    group_column: str | None,
    site_id_from_tree_id: bool,
    site_id_delimiter: str,
) -> str | None:
    if group_column and group_column in label_row and label_row[group_column] is not None:
        return str(label_row[group_column])
    if site_id_from_tree_id and site_id_delimiter in tree_id:
        return tree_id.split(site_id_delimiter, 1)[0]
    if "样地号" in label_row and label_row["样地号"] is not None:
        return str(label_row["样地号"])
    return None


def resolve_rgb_source(metadata: dict[str, Any], review_row: dict[str, Any]) -> str | None:
    if review_row.get("rgb_source"):
        return str(review_row["rgb_source"])
    if metadata.get("rgb_source"):
        return str(metadata["rgb_source"])
    return None


def count_missing_targets(rows: list[SampleRow], task: str, column: str | None) -> int:
    if column is None:
        return len(rows)
    return sum(1 for row in rows if row.labels.get(column) is None)


def filter_samples(
    rows: list[SampleRow],
    *,
    rgb_source: str,
    site_ids: list[str] | None,
    exclude_near_duplicates: bool,
    exclude_outliers: bool,
) -> tuple[list[SampleRow], dict[str, Any]]:
    site_filter = set(str(site_id) for site_id in site_ids) if site_ids else None
    kept: list[SampleRow] = []
    removed = Counter()
    for row in rows:
        if exclude_near_duplicates and row.is_near_duplicate_candidate:
            removed["near_duplicate"] += 1
            continue
        if exclude_outliers and row.is_outlier_candidate:
            removed["outlier"] += 1
            continue
        if rgb_source != "all" and row.rgb_source != rgb_source:
            removed["rgb_source"] += 1
            continue
        if site_filter is not None and row.site_id not in site_filter:
            removed["site_id"] += 1
            continue
        kept.append(row)
    return kept, {
        "input_count": len(rows),
        "filtered_count": len(kept),
        "removed_counts": dict(removed),
        "rgb_source_filter": rgb_source,
        "site_id_filter": sorted(site_filter) if site_filter else None,
        "exclude_near_duplicates": exclude_near_duplicates,
        "exclude_outliers": exclude_outliers,
        "site_counts": dict(Counter(row.site_id for row in kept)),
        "rgb_source_counts": dict(Counter(row.rgb_source for row in kept)),
    }


def run_all_probes(rows: list[SampleRow], context: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    for task in context["tasks"]:
        task_rows = rows_with_task_label(rows, task, context["label_columns"].get(task if task != "species" else "species"))
        for feature_set in context["feature_sets"]:
            feature_error = validate_feature_set(task, feature_set, context["label_columns"])
            if feature_error:
                results.append(skip_result(task, feature_set, "all", "feature_validation", None, rows, feature_error, context))
                continue
            task_feature_rows = rows_with_feature_values(task_rows, task, feature_set, context["label_columns"])
            for split_type in context["split_types"]:
                if split_type == "stratified_random" and task != "species":
                    results.append(skip_result(task, feature_set, split_type, "not_applicable", None, task_feature_rows, "stratified_random is only defined for species classification", context))
                    continue
                splits, split_failure = build_splits(
                    task_feature_rows,
                    task=task,
                    species_column=context["label_columns"].get("species"),
                    split_type=split_type,
                    seeds=context["args"].seeds,
                    test_size=context["args"].test_size,
                    min_class_count=context["args"].min_class_count,
                )
                if split_failure:
                    results.append(skip_result(task, feature_set, split_type, "split", None, task_feature_rows, split_failure, context))
                    continue
                for split in splits:
                    split_results, split_predictions, split_class_rows = run_one_split(
                        task_feature_rows,
                        task=task,
                        feature_set=feature_set,
                        split=split,
                        context=context,
                    )
                    results.extend(split_results)
                    predictions.extend(split_predictions)
                    per_class_rows.extend(split_class_rows)
    return results, predictions, per_class_rows


def rows_with_task_label(rows: list[SampleRow], task: str, column: str | None) -> list[SampleRow]:
    if column is None:
        return []
    return [row for row in rows if value_available(row.labels.get(column))]


def rows_with_feature_values(rows: list[SampleRow], task: str, feature_set: str, columns: dict[str, str | None]) -> list[SampleRow]:
    if feature_set == "uni3d":
        return rows
    geometry_cols = geometry_feature_columns(task, columns)
    if not geometry_cols:
        return []
    output = []
    for row in rows:
        if all(value_available(row.labels.get(column)) for column in geometry_cols):
            output.append(row)
    return output


def value_available(value: Any) -> bool:
    if value is None:
        return False
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return str(value).strip().lower() not in {"", "nan", "none"}


def validate_feature_set(task: str, feature_set: str, columns: dict[str, str | None]) -> str | None:
    if feature_set == "uni3d":
        return None
    cols = geometry_feature_columns(task, columns)
    if not cols:
        return "No geometry feature columns remain after target leakage exclusion."
    return None


def geometry_feature_columns(task: str, columns: dict[str, str | None]) -> list[str]:
    geometry = {
        "dbh": columns.get("dbh"),
        "height": columns.get("height"),
        "crown_ew": columns.get("crown_ew"),
        "crown_ns": columns.get("crown_ns"),
    }
    if task in REGRESSION_TASKS:
        geometry[task] = None
    return [column for column in geometry.values() if column is not None]


def build_splits(
    rows: list[SampleRow],
    *,
    task: str,
    species_column: str | None,
    split_type: str,
    seeds: list[int],
    test_size: float,
    min_class_count: int,
) -> tuple[list[SplitSpec], str | None]:
    if not rows:
        return [], "No rows remain after label/feature filtering."
    if split_type == "leave_one_site_out":
        splits = make_leave_one_site_out_splits(rows)
        return splits, None if splits else "No valid site groups for leave-one-site-out."
    output: list[SplitSpec] = []
    for seed in seeds:
        if split_type == "random":
            split = make_random_split(len(rows), test_size=test_size, seed=seed)
        elif split_type == "stratified_random":
            if species_column is None:
                return [], "No species column detected for stratified_random."
            y = [str(row.labels.get(species_column)) for row in rows]
            split, failure = make_stratified_random_split(y, test_size=test_size, seed=seed, min_class_count=min_class_count)
            if failure:
                return [], failure
        elif split_type == "group":
            split = make_group_split(rows, test_size=test_size, seed=seed)
        else:
            raise ValueError(f"Unknown split type: {split_type}")
        if split is None:
            return [], f"Could not build {split_type} split."
        output.append(split)
    return output, None


def make_random_split(count: int, *, test_size: float, seed: int) -> SplitSpec | None:
    if count < 2:
        return None
    rng = np.random.default_rng(seed)
    indices = np.arange(count)
    rng.shuffle(indices)
    test_count = max(1, int(math.ceil(count * test_size)))
    test_count = min(test_count, count - 1)
    test = sorted(indices[:test_count].astype(int).tolist())
    train = sorted(indices[test_count:].astype(int).tolist())
    return SplitSpec("random", f"seed_{seed}", seed, train, test)


def make_stratified_random_split(
    labels: list[str],
    *,
    test_size: float,
    seed: int,
    min_class_count: int,
) -> tuple[SplitSpec | None, str | None]:
    class_indices: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        class_indices[label].append(index)
    too_small = [label for label, values in class_indices.items() if len(values) < min_class_count]
    if too_small:
        return None, f"Class count below --min-class-count for: {too_small[:10]}"
    rng = np.random.default_rng(seed)
    train: list[int] = []
    test: list[int] = []
    for values in class_indices.values():
        values = values[:]
        rng.shuffle(values)
        test_count = max(1, int(math.ceil(len(values) * test_size)))
        test_count = min(test_count, len(values) - 1)
        test.extend(values[:test_count])
        train.extend(values[test_count:])
    return SplitSpec("stratified_random", f"seed_{seed}", seed, sorted(train), sorted(test)), None


def make_group_split(rows: list[SampleRow], *, test_size: float, seed: int) -> SplitSpec | None:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row.site_id is not None:
            groups[row.site_id].append(index)
    if len(groups) < 2:
        return None
    rng = np.random.default_rng(seed)
    group_names = list(groups)
    rng.shuffle(group_names)
    target_test = max(1, int(math.ceil(len(rows) * test_size)))
    test_groups: list[str] = []
    test: list[int] = []
    for group in group_names:
        if not test_groups or len(test) < target_test:
            test_groups.append(group)
            test.extend(groups[group])
    train = [index for index in range(len(rows)) if index not in set(test)]
    return SplitSpec("group", f"seed_{seed}", seed, sorted(train), sorted(test))


def make_leave_one_site_out_splits(rows: list[SampleRow]) -> list[SplitSpec]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row.site_id is not None:
            groups[row.site_id].append(index)
    output = []
    for site_id in sorted(groups):
        test = sorted(groups[site_id])
        train = [index for index in range(len(rows)) if index not in set(test)]
        output.append(SplitSpec("leave_one_site_out", f"site_{site_id}", None, train, test))
    return output


def run_one_split(
    rows: list[SampleRow],
    *,
    task: str,
    feature_set: str,
    split: SplitSpec,
    context: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    args = context["args"]
    target_column = context["label_columns"]["species" if task == "species" else task]
    failure = validate_split_sizes(
        rows,
        split,
        min_train=args.min_train_samples,
        min_test=args.min_test_samples,
        task=task,
        target_column=target_column,
    )
    if failure:
        return [skip_result(task, feature_set, split.split_type, split.split_id, split.seed, rows, failure, context)], [], []

    x, y, used_rows = build_xy(rows, task, feature_set, target_column, context["label_columns"])
    train = split.train_indices
    test = split.test_indices
    x_train, x_test = x[train], x[test]
    y_train = y[train]
    y_test = y[test]
    if args.standardize:
        mean, std = fit_standardizer(x_train)
        x_train = transform_standardizer(x_train, mean, std)
        x_test = transform_standardizer(x_test, mean, std)

    if task == "species":
        y_pred = knn_classify(x_train, y_train.astype(str), x_test, k=args.knn_k)
        metrics, class_rows, confusion = classification_metrics(y_test.astype(str), y_pred)
        model_type = f"knn_k{args.knn_k}"
        write_confusion_matrix(context["output_dir"], task, feature_set, split, confusion)
    else:
        y_pred = ridge_predict(x_train, y_train.astype(float), x_test, alpha=args.ridge_alpha)
        metrics = regression_metrics(y_test.astype(float), y_pred.astype(float))
        class_rows = []
        model_type = f"ridge_alpha{args.ridge_alpha}"

    result_rows = [
        metric_result_row(
            task=task,
            target=target_column,
            feature_set=feature_set,
            split=split,
            rows=rows,
            context=context,
            model_type=model_type,
            metric_name=name,
            metric_value=value,
        )
        for name, value in metrics.items()
    ]
    prediction_rows = build_prediction_rows(
        used_rows=used_rows,
        split=split,
        task=task,
        feature_set=feature_set,
        y_true=y_test,
        y_pred=y_pred,
    )
    for row in class_rows:
        row.update(
            {
                "task": task,
                "feature_set": feature_set,
                "split_type": split.split_type,
                "split_id": split.split_id,
                "seed": split.seed,
            }
        )
    return result_rows, prediction_rows, class_rows


def validate_split_sizes(
    rows: list[SampleRow],
    split: SplitSpec,
    *,
    min_train: int,
    min_test: int,
    task: str,
    target_column: str | None,
) -> str | None:
    if len(split.train_indices) < min_train:
        return f"train_count {len(split.train_indices)} below --min-train-samples {min_train}"
    if len(split.test_indices) < min_test:
        return f"test_count {len(split.test_indices)} below --min-test-samples {min_test}"
    if task == "species":
        if target_column is None:
            return "No species target column detected."
        train_labels = [str(rows[index].labels.get(target_column)) for index in split.train_indices]
        if len(set(train_labels)) < 2:
            return "classification train split has fewer than two classes"
    return None


def build_xy(
    rows: list[SampleRow],
    task: str,
    feature_set: str,
    target_column: str | None,
    columns: dict[str, str | None],
) -> tuple[np.ndarray, np.ndarray, list[SampleRow]]:
    if target_column is None:
        raise ValueError(f"No target column detected for task {task}")
    features: list[np.ndarray] = []
    targets: list[Any] = []
    used_rows: list[SampleRow] = []
    geometry_cols = geometry_feature_columns(task, columns)
    for row in rows:
        target = row.labels.get(target_column)
        if not value_available(target):
            continue
        parts: list[np.ndarray] = []
        if feature_set in {"uni3d", "uni3d_plus_geometry"}:
            parts.append(row.embedding.astype(np.float64, copy=False))
        if feature_set in {"geometry", "uni3d_plus_geometry"}:
            values = [float(row.labels[column]) for column in geometry_cols]
            parts.append(np.asarray(values, dtype=np.float64))
        features.append(np.concatenate(parts))
        targets.append(str(target) if task == "species" else float(target))
        used_rows.append(row)
    return np.stack(features, axis=0), np.asarray(targets), used_rows


def fit_standardizer(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return mean, std


def transform_standardizer(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def knn_classify(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, *, k: int) -> np.ndarray:
    k = max(1, min(k, len(y_train)))
    predictions: list[str] = []
    for row in x_test:
        distances = np.linalg.norm(x_train - row[None, :], axis=1)
        nearest = np.argsort(distances)[:k]
        votes = Counter(y_train[nearest].tolist())
        predictions.append(sorted(votes.items(), key=lambda item: (-item[1], str(item[0])))[0][0])
    return np.asarray(predictions)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, Any]]:
    classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    confusion = {actual: {predicted: 0 for predicted in classes} for actual in classes}
    for actual, predicted in zip(y_true, y_pred, strict=True):
        confusion[str(actual)][str(predicted)] += 1
    per_class: list[dict[str, Any]] = []
    f1_values: list[float] = []
    for cls in classes:
        tp = confusion[cls][cls]
        support = sum(confusion[cls].values())
        predicted_total = sum(confusion[actual][cls] for actual in classes)
        correct = tp
        precision = safe_div(tp, predicted_total)
        recall = safe_div(tp, support)
        f1 = safe_div(2 * precision * recall, precision + recall) if precision is not None and recall is not None else None
        accuracy = safe_div(correct, support)
        if f1 is not None:
            f1_values.append(f1)
        per_class.append(
            {
                "class_name": cls,
                "support": support,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    metrics = {
        "accuracy": float(np.mean(y_true == y_pred)) if len(y_true) else float("nan"),
        "macro_f1": float(np.mean(f1_values)) if f1_values else float("nan"),
    }
    return metrics, per_class, {"classes": classes, "matrix": confusion}


def safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, *, alpha: float) -> np.ndarray:
    y_mean = float(np.mean(y_train))
    y_centered = y_train - y_mean
    gram = x_train @ x_train.T
    eye = np.eye(gram.shape[0], dtype=np.float64)
    try:
        weights = np.linalg.solve(gram + alpha * eye, y_centered)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(gram + alpha * eye) @ y_centered
    coef = x_train.T @ weights
    return (x_test @ coef + y_mean).astype(float)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_true - y_pred
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    total = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum(residual**2) / total) if total > 1e-12 else float("nan")
    return {"R2": r2, "MAE": mae, "RMSE": rmse}


def metric_result_row(
    *,
    task: str,
    target: str | None,
    feature_set: str,
    split: SplitSpec,
    rows: list[SampleRow],
    context: dict[str, Any],
    model_type: str,
    metric_name: str,
    metric_value: float,
) -> dict[str, Any]:
    return base_result_row(task, target, feature_set, split.split_type, split.split_id, split.seed, rows, context) | {
        "train_count": len(split.train_indices),
        "test_count": len(split.test_indices),
        "site_train": json.dumps(sorted({str(rows[index].site_id) for index in split.train_indices if rows[index].site_id is not None}), ensure_ascii=False),
        "site_test": json.dumps(sorted({str(rows[index].site_id) for index in split.test_indices if rows[index].site_id is not None}), ensure_ascii=False),
        "model_type": model_type,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "status": "ok",
        "failure_reason": None,
    }


def skip_result(
    task: str,
    feature_set: str,
    split_type: str,
    split_id: str,
    seed: int | None,
    rows: list[SampleRow],
    failure_reason: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    return base_result_row(task, None, feature_set, split_type, split_id, seed, rows, context) | {
        "train_count": None,
        "test_count": None,
        "site_train": None,
        "site_test": None,
        "model_type": None,
        "metric_name": None,
        "metric_value": None,
        "status": "skipped",
        "failure_reason": failure_reason,
    }


def base_result_row(
    task: str,
    target: str | None,
    feature_set: str,
    split_type: str,
    split_id: str,
    seed: int | None,
    rows: list[SampleRow],
    context: dict[str, Any],
) -> dict[str, Any]:
    args = context["args"]
    return {
        "task": task,
        "target": target,
        "feature_set": feature_set,
        "split_type": split_type,
        "split_id": split_id,
        "seed": seed,
        "train_count": None,
        "test_count": None,
        "site_train": None,
        "site_test": None,
        "rgb_filter": args.rgb_source,
        "excluded_near_duplicates": bool(args.exclude_near_duplicates),
        "excluded_outliers": bool(args.exclude_outliers),
        "model_type": None,
        "metric_name": None,
        "metric_value": None,
        "status": "skipped",
        "failure_reason": None,
    }


def build_prediction_rows(
    *,
    used_rows: list[SampleRow],
    split: SplitSpec,
    task: str,
    feature_set: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for local_index, sample_index in enumerate(split.test_indices):
        sample = used_rows[sample_index]
        rows.append(
            {
                "tree_id": sample.tree_id,
                "task": task,
                "feature_set": feature_set,
                "split_type": split.split_type,
                "split_id": split.split_id,
                "seed": split.seed,
                "y_true": to_jsonable(y_true[local_index]),
                "y_pred": to_jsonable(y_pred[local_index]),
                "site_id": sample.site_id,
                "rgb_source": sample.rgb_source,
                "is_near_duplicate_candidate": sample.is_near_duplicate_candidate,
                "is_outlier_candidate": sample.is_outlier_candidate,
            }
        )
    return rows


def write_confusion_matrix(output_dir: Path, task: str, feature_set: str, split: SplitSpec, confusion: dict[str, Any]) -> None:
    safe_name = f"{task}__{feature_set}__{split.split_type}__{split.split_id}.json".replace("/", "_")
    write_json(output_dir / "confusion_matrices" / safe_name, confusion)


def build_probe_summary(
    *,
    results: list[dict[str, Any]],
    rows: list[SampleRow],
    tasks: list[str],
    feature_sets: list[str],
    split_types: list[str],
    filter_report: dict[str, Any],
    alignment_report: dict[str, Any],
) -> dict[str, Any]:
    ok_results = [row for row in results if row.get("status") == "ok" and row.get("metric_value") is not None]
    aggregates: dict[str, Any] = {}
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in ok_results:
        key = (row["task"], row["feature_set"], row["split_type"], row["metric_name"])
        grouped[key].append(float(row["metric_value"]))
    for key, values in grouped.items():
        task, feature_set, split_type, metric_name = key
        aggregates.setdefault(task, {}).setdefault(feature_set, {}).setdefault(split_type, {})[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "count": len(values),
        }
    warnings = build_warnings(results, rows, alignment_report)
    return {
        "positioning": {
            "offline_downstream_probe": True,
            "uni3d_backbone_updated": False,
            "embeddings_are_precomputed_frozen_features": True,
            "random_split_is_loose_reference": True,
            "group_and_leave_one_site_out_are_stricter": True,
            "all_data_exploratory_results_are_not_formal": True,
        },
        "total_matched_samples": alignment_report["matched_count"],
        "filtered_samples": len(rows),
        "tasks_run": tasks,
        "feature_sets_run": feature_sets,
        "split_types_run": split_types,
        "filter_report": filter_report,
        "site_sample_counts": dict(Counter(row.site_id for row in rows)),
        "rgb_source_sample_counts": dict(Counter(row.rgb_source for row in rows)),
        "aggregate_metrics": aggregates,
        "leave_one_site_out_per_site_results": summarize_loso(results),
        "warnings": warnings,
        "skipped_count": sum(1 for row in results if row.get("status") == "skipped"),
    }


def build_warnings(results: list[dict[str, Any]], rows: list[SampleRow], alignment_report: dict[str, Any]) -> dict[str, Any]:
    warnings: dict[str, Any] = {
        "possible_site_or_domain_leakage": False,
        "possible_rgb_availability_confound": False,
        "possible_near_duplicate_leakage": False,
        "insufficient_sample_size": False,
        "class_imbalance": False,
        "label_missing": any(count > 0 for count in alignment_report["missing_label_counts_per_task"].values()),
    }
    warnings["insufficient_sample_size"] = any("below --min" in str(row.get("failure_reason")) for row in results)
    warnings["class_imbalance"] = any("Class count below" in str(row.get("failure_reason")) for row in results)
    warnings["possible_near_duplicate_leakage"] = any(row.is_near_duplicate_candidate for row in rows)
    rgb_by_site: dict[str | None, set[str | None]] = defaultdict(set)
    for row in rows:
        rgb_by_site[row.site_id].add(row.rgb_source)
    if len({row.rgb_source for row in rows}) > 1 and any(len(values) == 1 for values in rgb_by_site.values()):
        warnings["possible_rgb_availability_confound"] = True
    warnings["possible_site_or_domain_leakage"] = detect_site_leakage_pattern(results)
    return warnings


def detect_site_leakage_pattern(results: list[dict[str, Any]]) -> bool:
    by_key: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in results:
        if row.get("status") != "ok" or row.get("metric_value") is None:
            continue
        metric = row.get("metric_name")
        if metric not in {"accuracy", "R2"}:
            continue
        by_key[(row["task"], row["feature_set"], metric)][row["split_type"]].append(float(row["metric_value"]))
    for splits in by_key.values():
        if splits.get("random") and splits.get("leave_one_site_out"):
            if float(np.mean(splits["random"])) - float(np.mean(splits["leave_one_site_out"])) > 0.15:
                return True
    return False


def summarize_loso(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in results
        if row.get("split_type") == "leave_one_site_out" and row.get("status") == "ok"
    ]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(make_json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: make_json_safe(row.get(key)) for key in fieldnames} for row in rows])


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def build_run_config(args: argparse.Namespace, detected_columns: dict[str, str | None]) -> dict[str, Any]:
    return {
        "script": "scripts/probe_uni3d_embeddings_strict.py",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "positioning": "offline downstream probe over frozen precomputed Uni3D embeddings; not Uni3D training",
        "args": make_json_safe(vars(args)),
        "detected_columns": detected_columns,
    }


if __name__ == "__main__":
    raise SystemExit(main())
