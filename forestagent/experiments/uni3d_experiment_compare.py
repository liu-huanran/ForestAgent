"""Cross-experiment comparison helpers for offline Uni3D artifacts."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from forestagent.experiments.uni3d_experiment_audit import (
    audit_conversion,
    audit_extraction,
    audit_probe,
    audit_similarity,
)
from forestagent.experiments.uni3d_experiment_config import experiment_name


def compare_experiments(configs: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare multiple configured Uni3D experiment artifact directories."""

    experiments: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    probe_metric_rows: list[dict[str, Any]] = []

    for config in configs:
        name = experiment_name(config)
        conversion = audit_conversion(config)
        extraction = audit_extraction(config)
        similarity = audit_similarity(config)
        probe = audit_probe(config)
        summary = {
            "name": name,
            "conversion": _summarize_conversion(config, conversion),
            "extraction": _summarize_extraction(extraction),
            "similarity": _summarize_similarity(similarity),
            "probe": _summarize_probe(probe),
            "warnings": _collect_experiment_warnings(config, similarity, probe),
        }
        experiments.append(summary)
        table_rows.extend(_flatten_summary_rows(summary))
        probe_metric_rows.extend(_load_probe_metric_rows(config))

    aggregate_probe = _aggregate_probe_metrics(probe_metric_rows)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_count": len(configs),
        "experiments": experiments,
        "comparison_table": table_rows,
        "probe_metric_rows": probe_metric_rows,
        "aggregate_probe_metrics": aggregate_probe,
        "boundary": (
            "Offline comparison only: no Uni3D forward, no Uni3D training, no main QA integration, "
            "and no paper-level conclusion by itself."
        ),
    }


def write_comparison_outputs(comparison: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path, Path]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    json_path = target_dir / "experiment_comparison_summary.json"
    csv_path = target_dir / "experiment_comparison_table.csv"
    md_path = target_dir / "experiment_comparison_report.md"

    json_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_rows(csv_path, comparison.get("comparison_table", []))
    md_path.write_text(render_comparison_markdown(comparison), encoding="utf-8")
    return json_path, csv_path, md_path


def render_comparison_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# Uni3D Experiment Comparison",
        "",
        f"- Generated: `{comparison['generated_at']}`",
        f"- Experiments: `{comparison['experiment_count']}`",
        "",
        "## High-Level Table",
        "",
        "| Experiment | Layer | Metric | Value | Status |",
        "|---|---|---|---|---|",
    ]
    for row in comparison.get("comparison_table", []):
        lines.append(
            f"| {row.get('experiment')} | {row.get('layer')} | {row.get('metric')} | {row.get('value')} | {row.get('status', '')} |"
        )
    lines.extend(["", "## Probe Aggregates", ""])
    if not comparison.get("aggregate_probe_metrics"):
        lines.append("No probe metrics were available.")
    else:
        lines.append("| Task | Feature Set | Split | Metric | Mean | Std | N |")
        lines.append("|---|---|---|---|---:|---:|---:|")
        for row in comparison["aggregate_probe_metrics"]:
            lines.append(
                "| {task} | {feature_set} | {split_type} | {metric_name} | {mean:.6g} | {std:.6g} | {count} |".format(
                    **row
                )
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            comparison["boundary"],
            "",
            "Do not interpret these comparisons as final paper conclusions without checking data leakage, site splits, and manual artifact review.",
            "",
        ]
    )
    return "\n".join(lines)


def _summarize_conversion(config: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    conversion = config.get("conversion", {})
    summary = audit.get("summary") or {}
    return {
        "status": audit.get("status"),
        "npy_count": audit.get("npy_count"),
        "shape_distribution": audit.get("shape_distribution"),
        "failure_count": summary.get("failure_count"),
        "output_mode": conversion.get("output_mode"),
        "color_policy": conversion.get("color_policy"),
        "color_source_distribution": summary.get("color_source_distribution"),
        "site_distribution": summary.get("site_distribution"),
    }


def _summarize_extraction(audit: dict[str, Any]) -> dict[str, Any]:
    summary = audit.get("summary") or {}
    return {
        "status": audit.get("status"),
        "npz_count": audit.get("npz_count"),
        "success_count": summary.get("success_count"),
        "failure_count": summary.get("failure_count"),
        "embedding_shape_distribution": audit.get("embedding_shape_distribution"),
        "embedding_l2_norm_min": audit.get("embedding_l2_norm_min"),
        "embedding_l2_norm_max": audit.get("embedding_l2_norm_max"),
        "mock_distribution": audit.get("mock_distribution"),
    }


def _summarize_similarity(audit: dict[str, Any]) -> dict[str, Any]:
    summary = audit.get("summary") or {}
    pairwise = summary.get("pairwise_similarity") if isinstance(summary.get("pairwise_similarity"), dict) else summary
    return {
        "status": audit.get("status"),
        "sample_count": summary.get("sample_count", summary.get("analyzed_sample_count")),
        "invalid_count": summary.get("invalid_count", summary.get("invalid_embeddings")),
        "cosine_min": pairwise.get("min") if isinstance(pairwise, dict) else None,
        "cosine_mean": pairwise.get("mean") if isinstance(pairwise, dict) else None,
        "cosine_max": pairwise.get("max") if isinstance(pairwise, dict) else None,
        "cosine_std": pairwise.get("std") if isinstance(pairwise, dict) else None,
        "near_duplicate_count": audit.get("near_duplicate_count"),
        "outlier_count": audit.get("outlier_count"),
        "all_pairwise_similarity_near_one": summary.get("all_pairwise_similarity_near_one"),
    }


def _summarize_probe(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": audit.get("status"),
        "result_row_count": audit.get("result_row_count"),
        "split_types": audit.get("split_types"),
        "feature_sets": audit.get("feature_sets"),
        "tasks": audit.get("tasks"),
        "failure_row_count": audit.get("failure_row_count"),
    }


def _collect_experiment_warnings(config: dict[str, Any], similarity: dict[str, Any], probe: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    metadata = config.get("metadata", {})
    conversion = config.get("conversion", {})
    if metadata.get("mixed_rgb_availability"):
        warnings.append("mixed RGB/color availability; interpretability warning")
    if conversion.get("semantic_color_likely"):
        warnings.append("semantic/discrete color likely; not natural RGB")
    if similarity.get("summary", {}).get("all_pairwise_similarity_near_one"):
        warnings.append("all pairwise similarity near one")
    if probe.get("status") in {"WARN", "FAIL"}:
        warnings.append(f"probe audit status is {probe.get('status')}")
    return warnings


def _flatten_summary_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    name = summary["name"]
    for layer in ("conversion", "extraction", "similarity", "probe"):
        layer_summary = summary[layer]
        status = layer_summary.get("status")
        for metric, value in layer_summary.items():
            if metric == "status":
                continue
            rows.append(
                {
                    "experiment": name,
                    "layer": layer,
                    "metric": metric,
                    "value": json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value,
                    "status": status,
                }
            )
    return rows


def _load_probe_metric_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    root = Path(config["paths"]["probe_output_dir"])
    if not root.exists():
        return []
    result_files = [root / "probe_results.csv"] if (root / "probe_results.csv").exists() else sorted(root.rglob("probe_results.csv"))
    rows: list[dict[str, Any]] = []
    for result_file in result_files:
        with result_file.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                value = _to_float(row.get("metric_value"))
                if value is None:
                    continue
                rows.append(
                    {
                        "experiment": experiment_name(config),
                        "task": row.get("task"),
                        "feature_set": row.get("feature_set"),
                        "split_type": row.get("split_type"),
                        "metric_name": row.get("metric_name"),
                        "metric_value": value,
                    }
                )
    return rows


def _aggregate_probe_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (row["task"], row["feature_set"], row["split_type"], row["metric_name"])
        grouped[key].append(float(row["metric_value"]))
    output: list[dict[str, Any]] = []
    for (task, feature_set, split_type, metric_name), values in sorted(grouped.items()):
        output.append(
            {
                "task": task,
                "feature_set": feature_set,
                "split_type": split_type,
                "metric_name": metric_name,
                "mean": mean(values),
                "std": pstdev(values) if len(values) > 1 else 0.0,
                "count": len(values),
            }
        )
    return output


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

