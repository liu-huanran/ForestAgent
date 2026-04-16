"""Minimal CLI for running the fixed MVP on cataloged samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forestagent.data_catalog import DataCatalog
from forestagent.evaluation import (
    benchmark_direct_geometry_baseline,
    benchmark_treeqsm_baseline,
    compare_direct_geometry_dbh_configs,
    evaluate_scalar_tasks,
    export_evaluation_csv,
    export_evaluation_xlsx,
    run_direct_geometry_height_freeze_validation,
)
from forestagent.mvp import run_single_tree_analysis
from forestagent.sample_runner import run_task_for_sample


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""

    parser = argparse.ArgumentParser(description="ForestAgent MVP CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-samples", help="List cataloged samples.")
    list_parser.add_argument("--data-dir", default="data")
    list_parser.add_argument("--limit", type=int, default=10)

    run_parser = subparsers.add_parser("run-task", help="Run one task on one sample.")
    run_parser.add_argument("--data-dir", default="data")
    run_parser.add_argument("--task-id", required=True)
    run_parser.add_argument("--sample-id", required=True)
    run_parser.add_argument("--modality", choices=["ground", "air"], required=True)

    analyze_parser = subparsers.add_parser(
        "analyze-tree",
        help="Run the minimal q1/q2/q3 single-tree MVP on one point cloud path.",
    )
    analyze_parser.add_argument("--point-cloud", required=True)
    analyze_parser.add_argument("--task-id")
    analyze_parser.add_argument("--question")
    analyze_parser.add_argument("--config-path")
    analyze_parser.add_argument("--use-local-llm-report", action="store_true")
    analyze_parser.add_argument("--ollama-model", default="qwen3:1.7b")
    analyze_parser.add_argument("--ollama-timeout-seconds", type=float, default=60.0)

    eval_parser = subparsers.add_parser(
        "evaluate-scalar",
        help="Batch-evaluate q1_dbh/q2_height/q3_crown_width on cataloged samples.",
    )
    eval_parser.add_argument("--data-dir", default="data")
    eval_parser.add_argument("--task-id", action="append")
    eval_parser.add_argument("--sample-id", action="append")
    eval_parser.add_argument("--modality", action="append", choices=["ground", "air"])
    eval_parser.add_argument("--output-csv")
    eval_parser.add_argument("--output-xlsx")

    benchmark_parser = subparsers.add_parser(
        "benchmark-treeqsm",
        help="Run a small q1/q2 baseline benchmark with the real TreeQSM backend.",
    )
    benchmark_parser.add_argument("--data-dir", default="data")
    benchmark_parser.add_argument("--sample-id", action="append")
    benchmark_parser.add_argument("--limit", type=int, default=3)
    benchmark_parser.add_argument("--modality", choices=["ground", "air"], default="ground")
    benchmark_parser.add_argument("--treeqsm-root")
    benchmark_parser.add_argument("--matlab-path")
    benchmark_parser.add_argument("--timeout-seconds", type=int, default=900)

    direct_parser = subparsers.add_parser(
        "benchmark-direct-geometry",
        help="Run a small q1/q2 baseline benchmark with the direct geometry backend.",
    )
    direct_parser.add_argument("--data-dir", default="data")
    direct_parser.add_argument("--sample-id", action="append")
    direct_parser.add_argument("--limit", type=int, default=10)
    direct_parser.add_argument("--modality", choices=["ground", "air"], default="ground")
    direct_parser.add_argument("--config-path")

    diagnosis_parser = subparsers.add_parser(
        "diagnose-direct-geometry-dbh",
        help="Diagnose q1_dbh baseline errors and compare trunk-only filtering on small samples.",
    )
    diagnosis_parser.add_argument("--data-dir", default="data")
    diagnosis_parser.add_argument("--sample-id", action="append")
    diagnosis_parser.add_argument("--limit", type=int, default=10)
    diagnosis_parser.add_argument("--modality", choices=["ground", "air"], default="ground")
    diagnosis_parser.add_argument("--baseline-config-path")
    diagnosis_parser.add_argument("--improved-config-path")
    diagnosis_parser.add_argument(
        "--output-dir",
        default="outputs/direct_geometry_dbh_diagnosis",
    )
    diagnosis_parser.add_argument("--top-k-visualizations", type=int, default=5)

    height_parser = subparsers.add_parser(
        "freeze-direct-geometry-height",
        help="Run q2_height diagnosis-ready freeze validation with the direct geometry backend.",
    )
    height_parser.add_argument("--data-dir", default="data")
    height_parser.add_argument("--sample-id", action="append")
    height_parser.add_argument("--limit", type=int, default=50)
    height_parser.add_argument("--modality", choices=["ground", "air"], default="ground")
    height_parser.add_argument("--config-path")
    height_parser.add_argument(
        "--output-dir",
        default="outputs/direct_geometry_height_freeze_validation",
    )
    height_parser.add_argument("--top-k-errors", type=int, default=20)

    return parser


def main() -> None:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args()

    if args.command == "list-samples":
        catalog = DataCatalog.from_data_dir(args.data_dir)
        payload = {
            "catalog_summary": catalog.summary(),
            "samples": [
                {
                    "sample_id": record.sample_id,
                    "plot_id": record.plot_id,
                    "tree_number": record.tree_number,
                    "file_name": record.file_name,
                    "species": record.measured.species,
                    "available_modalities": record.available_modalities(),
                }
                for record in catalog.list_records(limit=args.limit)
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "run-task":
        payload = run_task_for_sample(
            task_id=args.task_id,
            sample_id=args.sample_id,
            modality=args.modality,
            data_dir=Path(args.data_dir),
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "analyze-tree":
        payload = run_single_tree_analysis(
            args.point_cloud,
            task_id=args.task_id,
            question=args.question,
            config_path=args.config_path,
            use_local_llm_report=args.use_local_llm_report,
            ollama_model=args.ollama_model,
            ollama_timeout_seconds=args.ollama_timeout_seconds,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "evaluate-scalar":
        result = evaluate_scalar_tasks(
            data_dir=Path(args.data_dir),
            task_ids=args.task_id,
            sample_ids=args.sample_id,
            modalities=args.modality,
        )
        exports: dict[str, str] = {}
        if args.output_csv:
            exports["csv"] = str(export_evaluation_csv(result, args.output_csv))
        if args.output_xlsx:
            exports["xlsx"] = str(export_evaluation_xlsx(result, args.output_xlsx))

        payload = {
            "record_count": len(result.records),
            "metrics": [metric.model_dump() for metric in result.metrics],
            "modality_comparison": [
                item.model_dump() for item in result.modality_comparison
            ],
            "exports": exports,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "benchmark-treeqsm":
        result = benchmark_treeqsm_baseline(
            data_dir=Path(args.data_dir),
            sample_ids=args.sample_id,
            limit=args.limit,
            modality=args.modality,
            treeqsm_root=args.treeqsm_root,
            matlab_executable=args.matlab_path,
            timeout_seconds=args.timeout_seconds,
        )
        payload = {
            "sample_ids": result.sample_ids,
            "modality": result.modality,
            "record_count": len(result.records),
            "metrics": [metric.model_dump() for metric in result.metrics],
            "failure_reason_stats": [
                item.model_dump() for item in result.failure_reason_stats
            ],
            "records": [record.model_dump() for record in result.records],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "benchmark-direct-geometry":
        result = benchmark_direct_geometry_baseline(
            data_dir=Path(args.data_dir),
            sample_ids=args.sample_id,
            limit=args.limit,
            modality=args.modality,
            config_path=args.config_path,
        )
        payload = {
            "sample_ids": result.sample_ids,
            "modality": result.modality,
            "record_count": len(result.records),
            "metrics": [metric.model_dump() for metric in result.metrics],
            "failure_reason_stats": [
                item.model_dump() for item in result.failure_reason_stats
            ],
            "records": [record.model_dump() for record in result.records],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "diagnose-direct-geometry-dbh":
        result = compare_direct_geometry_dbh_configs(
            data_dir=Path(args.data_dir),
            sample_ids=args.sample_id,
            limit=args.limit,
            modality=args.modality,
            baseline_config_path=args.baseline_config_path,
            improved_config_path=args.improved_config_path,
            output_dir=args.output_dir,
            top_k_visualizations=args.top_k_visualizations,
        )
        payload = {
            "sample_ids": result.sample_ids,
            "modality": result.modality,
            "baseline": result.baseline.model_dump(),
            "improved": result.improved.model_dump(),
            "comparison_path": result.comparison_path,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "freeze-direct-geometry-height":
        result = run_direct_geometry_height_freeze_validation(
            data_dir=Path(args.data_dir),
            sample_ids=args.sample_id,
            limit=args.limit,
            modality=args.modality,
            config_path=args.config_path,
            output_dir=args.output_dir,
            top_k_errors=args.top_k_errors,
        )
        payload = {
            "sample_ids": result.sample_ids,
            "modality": result.modality,
            "config_path": result.config_path,
            "summary": result.summary.model_dump(),
            "failure_reason_stats": [
                item.model_dump() for item in result.failure_reason_stats
            ],
            "suspicion_flag_stats": [
                item.model_dump() for item in result.suspicion_flag_stats
            ],
            "top_absolute_error_records": [
                item.model_dump() for item in result.top_absolute_error_records
            ],
            "per_sample_records_path": result.per_sample_records_path,
            "top_abs_error_path": result.top_abs_error_path,
            "summary_path": result.summary_path,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
