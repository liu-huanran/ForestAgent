"""Minimal reproducible TreeQSM failure-diagnosis script for one sample."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from forestagent.backends.treeqsm_backend import TreeQSMBackend, TreeQSMDiagnosticRun
from forestagent.data_catalog import DataCatalog
from forestagent.sample_runner import build_point_cloud_input


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a minimal TreeQSM failure diagnosis on one ground sample."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--modality", choices=["ground", "air"], default="ground")
    parser.add_argument("--treeqsm-root", required=True)
    parser.add_argument("--matlab-path", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    catalog = DataCatalog.from_data_dir(args.data_dir)
    record = catalog.get_record(args.sample_id)
    point_cloud = build_point_cloud_input(record, args.modality)
    backend = TreeQSMBackend(
        treeqsm_root=args.treeqsm_root,
        matlab_executable=args.matlab_path,
        timeout_seconds=args.timeout_seconds,
    )

    raw_run = backend.run_failure_diagnosis(point_cloud, apply_preprocess=False)
    preprocessed_run = backend.run_failure_diagnosis(point_cloud, apply_preprocess=True)
    skip_crown_run = backend.run_failure_diagnosis(
        point_cloud,
        apply_preprocess=False,
        try_skip_crown=True,
    )

    payload = {
        "sample": {
            "sample_id": record.sample_id,
            "plot_id": record.plot_id,
            "file_name": record.file_name,
            "modality": args.modality,
            "point_cloud": point_cloud.model_dump(),
        },
        "measured_reference": record.measured_summary(),
        "raw_run": _serialize_run(raw_run),
        "preprocessed_run": _serialize_run(preprocessed_run),
        "skip_crown_trial": _serialize_run(skip_crown_run),
        "judgment": _build_judgment(raw_run, preprocessed_run, skip_crown_run),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _serialize_run(run: TreeQSMDiagnosticRun) -> dict[str, object]:
    return {
        "label": run.label,
        "status": run.status,
        "runtime_seconds": run.runtime_seconds,
        "failure_message": run.failure_message,
        "preprocess_stats": asdict(run.preprocess_stats),
        "debug_info": asdict(run.debug_info) if run.debug_info is not None else None,
        "metrics": asdict(run.metrics) if run.metrics is not None else None,
    }


def _build_judgment(
    raw_run: TreeQSMDiagnosticRun,
    preprocessed_run: TreeQSMDiagnosticRun,
    skip_crown_run: TreeQSMDiagnosticRun,
) -> dict[str, object]:
    raw_debug = raw_run.debug_info
    pre_debug = preprocessed_run.debug_info

    has_intermediate_q1q2 = False
    if raw_debug is not None:
        has_intermediate_q1q2 = any(
            value is not None
            for value in [raw_debug.dbh_cyl_m, raw_debug.dbh_qsm_m, raw_debug.tree_height_m]
        )

    projected_degeneration = None
    if raw_debug is not None:
        projected_degeneration = (
            (raw_debug.unique_projected_2d_point_count or 0) < 3
            or bool(raw_debug.is_projected_collinear)
            or (raw_debug.projected_x_span == 0)
            or (raw_debug.projected_y_span == 0)
        )

    preprocessing_changed_outcome = (
        raw_run.status != preprocessed_run.status
        or raw_run.failure_message != preprocessed_run.failure_message
    )

    skip_crown_feasible = (
        skip_crown_run.status == "success"
        and skip_crown_run.metrics is not None
        and (
            skip_crown_run.metrics.tree_height_m is not None
            or skip_crown_run.metrics.dbh_cyl_m is not None
            or skip_crown_run.metrics.dbh_qsm_m is not None
        )
    )

    return {
        "raw_has_intermediate_dbh_or_height": has_intermediate_q1q2,
        "raw_projected_points_degenerate": projected_degeneration,
        "raw_last_successful_stage": raw_debug.last_successful_stage if raw_debug else None,
        "preprocessing_changed_outcome": preprocessing_changed_outcome,
        "preprocessed_crown_point_count": (
            pre_debug.crown_point_count_before_convhull if pre_debug else None
        ),
        "skip_crown_feasible": skip_crown_feasible,
        "skip_crown_status": skip_crown_run.status,
    }


if __name__ == "__main__":
    main()
