"""Diagnosis-ready freeze validation for the direct geometry q3_crown_width baseline."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from forestagent.backends.direct_geometry_backend import (
    CrownWidthInspection,
    DirectGeometryBackend,
)
from forestagent.data_catalog import DataCatalog
from forestagent.sample_runner import build_point_cloud_input

CrownWidthDiagnosisStatus = Literal[
    "success",
    "missing_sample",
    "missing_modality",
    "missing_measured",
    "task_failed",
]
CrownWidthDiagnosisModality = Literal["ground", "air"]


class CrownWidthSanityRecord(BaseModel):
    """Per-sample q3_crown_width diagnosis record with intermediate values and flags."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(min_length=1)
    modality: CrownWidthDiagnosisModality
    measured_crown_width_m: float | None = None
    pred_crown_width_m: float | None = None
    abs_error_m: float | None = None
    status: CrownWidthDiagnosisStatus
    message: str = ""
    total_point_count: int | None = None
    projected_point_count: int | None = None
    unique_xy_count: int | None = None
    removed_duplicate_count: int | None = None
    removed_outlier_count: int | None = None
    x_raw_span_m: float | None = None
    y_raw_span_m: float | None = None
    x_robust_span_m: float | None = None
    y_robust_span_m: float | None = None
    x_raw_vs_robust_gap_m: float | None = None
    y_raw_vs_robust_gap_m: float | None = None
    raw_mean_span_m: float | None = None
    robust_mean_span_m: float | None = None
    raw_vs_robust_gap_m: float | None = None
    raw_vs_robust_gap_ratio: float | None = None
    predicted_crown_width_m: float | None = None
    pca_major_span_m: float | None = None
    pca_minor_span_m: float | None = None
    pca_mean_span_m: float | None = None
    axis_vs_pca_gap_m: float | None = None
    fixed_minus_pca_mean_m: float | None = None
    fixed_to_pca_ratio: float | None = None
    raw_axis_asymmetry_ratio: float | None = None
    robust_axis_asymmetry_ratio: float | None = None
    removed_outlier_fraction: float | None = None
    x_low_edge_support_count: int | None = None
    x_high_edge_support_count: int | None = None
    y_low_edge_support_count: int | None = None
    y_high_edge_support_count: int | None = None
    x_min_edge_support_fraction: float | None = None
    y_min_edge_support_fraction: float | None = None
    edge_support_band_m: float | None = None
    sparse_projection_suspected: bool = False
    heavy_outlier_removal_suspected: bool = False
    axis_pca_gap_suspected: bool = False
    elongated_projection_suspected: bool = False
    robust_shrinkage_suspected: bool = False
    edge_sparsity_suspected: bool = False
    pca_axis_underestimate_suspected: bool = False


class CrownWidthDiagnosisSummary(BaseModel):
    """Aggregated q3_crown_width freeze validation summary."""

    model_config = ConfigDict(extra="forbid")

    total_records: int
    success_count: int
    success_rate: float
    mae: float | None = None
    rmse: float | None = None


class CrownWidthFailureReasonStat(BaseModel):
    """Aggregated count of one q3_crown_width failure reason."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    count: int = Field(ge=1)


class CrownWidthSuspicionStat(BaseModel):
    """Aggregated count/rate for one q3 suspicion flag."""

    model_config = ConfigDict(extra="forbid")

    flag_name: str = Field(min_length=1)
    count: int = Field(ge=0)
    rate_overall: float = Field(ge=0.0, le=1.0)
    rate_among_successes: float | None = Field(default=None, ge=0.0, le=1.0)


class DirectGeometryCrownWidthFreezeValidationResult(BaseModel):
    """Structured q3_crown_width freeze validation output."""

    model_config = ConfigDict(extra="forbid")

    sample_ids: list[str]
    modality: CrownWidthDiagnosisModality
    config_path: str
    records: list[CrownWidthSanityRecord]
    summary: CrownWidthDiagnosisSummary
    failure_reason_stats: list[CrownWidthFailureReasonStat]
    suspicion_flag_stats: list[CrownWidthSuspicionStat]
    top_absolute_error_records: list[CrownWidthSanityRecord]
    per_sample_records_path: str | None = None
    top_abs_error_path: str | None = None
    summary_path: str | None = None


def run_direct_geometry_crown_width_freeze_validation(
    data_dir: str | Path = "data",
    sample_ids: list[str] | None = None,
    limit: int = 50,
    modality: CrownWidthDiagnosisModality = "ground",
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    top_k_errors: int = 20,
    catalog: DataCatalog | None = None,
) -> DirectGeometryCrownWidthFreezeValidationResult:
    """Run q3_crown_width freeze validation with diagnosis-ready fields."""

    if modality not in ("ground", "air"):
        raise ValueError(f"Unsupported modality: {modality}")
    if limit <= 0:
        raise ValueError("limit must be a positive integer.")
    if top_k_errors <= 0:
        raise ValueError("top_k_errors must be a positive integer.")

    catalog = catalog or DataCatalog.from_data_dir(data_dir)
    selected_sample_ids = (
        _deduplicate_preserve_order(sample_ids)
        if sample_ids is not None
        else _select_default_sample_ids(catalog, modality, limit)
    )
    if not selected_sample_ids:
        raise ValueError("No sample_ids were selected for direct geometry q3_crown_width validation.")

    backend = DirectGeometryBackend(config_path=config_path)
    records = [
        _diagnose_one_sample(
            sample_id=sample_id,
            modality=modality,
            catalog=catalog,
            backend=backend,
        )
        for sample_id in selected_sample_ids
    ]
    summary = _build_summary(records)
    failure_reason_stats = _build_failure_reason_stats(records)
    suspicion_flag_stats = _build_suspicion_flag_stats(records)
    top_absolute_error_records = _select_top_error_records(records, top_k_errors)

    output_root = None if output_dir is None else Path(output_dir).resolve()
    per_sample_records_path = None
    top_abs_error_path = None
    summary_path = None
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)
        per_sample_records_path = str((output_root / "per_sample_records.csv").resolve())
        top_abs_error_path = str((output_root / "top_absolute_error_samples.csv").resolve())
        summary_path = str((output_root / "summary.json").resolve())
        _write_records_csv(records, Path(per_sample_records_path))
        _write_records_csv(top_absolute_error_records, Path(top_abs_error_path))
        Path(summary_path).write_text(
            json.dumps(
                {
                    "sample_ids": selected_sample_ids,
                    "modality": modality,
                    "config_path": str(backend._config_path),
                    "summary": summary.model_dump(),
                    "failure_reason_stats": [
                        item.model_dump() for item in failure_reason_stats
                    ],
                    "suspicion_flag_stats": [
                        item.model_dump() for item in suspicion_flag_stats
                    ],
                    "top_absolute_error_sample_ids": [
                        record.sample_id for record in top_absolute_error_records
                    ],
                    "per_sample_records_path": per_sample_records_path,
                    "top_abs_error_path": top_abs_error_path,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    return DirectGeometryCrownWidthFreezeValidationResult(
        sample_ids=selected_sample_ids,
        modality=modality,
        config_path=str(backend._config_path),
        records=records,
        summary=summary,
        failure_reason_stats=failure_reason_stats,
        suspicion_flag_stats=suspicion_flag_stats,
        top_absolute_error_records=top_absolute_error_records,
        per_sample_records_path=per_sample_records_path,
        top_abs_error_path=top_abs_error_path,
        summary_path=summary_path,
    )


def _diagnose_one_sample(
    *,
    sample_id: str,
    modality: CrownWidthDiagnosisModality,
    catalog: DataCatalog,
    backend: DirectGeometryBackend,
) -> CrownWidthSanityRecord:
    try:
        record = catalog.get_record(sample_id)
    except KeyError:
        return CrownWidthSanityRecord(
            sample_id=sample_id,
            modality=modality,
            status="missing_sample",
            message=f"Sample {sample_id} was not found in the catalog.",
        )

    if modality not in record.available_modalities():
        return CrownWidthSanityRecord(
            sample_id=sample_id,
            modality=modality,
            measured_crown_width_m=record.measured.crown_width_mean_m,
            status="missing_modality",
            message=f"Sample {sample_id} does not have modality {modality}.",
        )

    point_cloud = build_point_cloud_input(record, modality)
    inspection = backend.inspect_crown_width(point_cloud)
    tool_result = inspection.tool_result
    diagnostics = tool_result.extra
    measured_crown_width_m = record.measured.crown_width_mean_m
    pred_crown_width_m = float(tool_result.value) if tool_result.value is not None else None
    abs_error_m = (
        None
        if measured_crown_width_m is None or pred_crown_width_m is None
        else abs(pred_crown_width_m - measured_crown_width_m)
    )

    if tool_result.status != "success":
        return CrownWidthSanityRecord(
            sample_id=sample_id,
            modality=modality,
            measured_crown_width_m=measured_crown_width_m,
            pred_crown_width_m=None,
            abs_error_m=None,
            status="task_failed",
            message=tool_result.message,
            **_record_fields_from_diagnostics(diagnostics),
        )

    record_status: CrownWidthDiagnosisStatus = (
        "success" if measured_crown_width_m is not None else "missing_measured"
    )
    record_message = (
        tool_result.message
        if measured_crown_width_m is not None
        else "No measured reference available for q3_crown_width."
    )
    return CrownWidthSanityRecord(
        sample_id=sample_id,
        modality=modality,
        measured_crown_width_m=measured_crown_width_m,
        pred_crown_width_m=pred_crown_width_m,
        abs_error_m=abs_error_m,
        status=record_status,
        message=record_message,
        **_record_fields_from_diagnostics(diagnostics),
    )


def _record_fields_from_diagnostics(diagnostics: dict[str, object]) -> dict[str, object]:
    return {
        "total_point_count": diagnostics.get("total_point_count"),
        "projected_point_count": diagnostics.get("projected_point_count"),
        "unique_xy_count": diagnostics.get("unique_xy_count"),
        "removed_duplicate_count": diagnostics.get("removed_duplicate_count"),
        "removed_outlier_count": diagnostics.get("removed_outlier_count"),
        "x_raw_span_m": diagnostics.get("x_raw_span_m"),
        "y_raw_span_m": diagnostics.get("y_raw_span_m"),
        "x_robust_span_m": diagnostics.get("x_robust_span_m"),
        "y_robust_span_m": diagnostics.get("y_robust_span_m"),
        "x_raw_vs_robust_gap_m": diagnostics.get("x_raw_vs_robust_gap_m"),
        "y_raw_vs_robust_gap_m": diagnostics.get("y_raw_vs_robust_gap_m"),
        "raw_mean_span_m": diagnostics.get("raw_mean_span_m"),
        "robust_mean_span_m": diagnostics.get("robust_mean_span_m"),
        "raw_vs_robust_gap_m": diagnostics.get("raw_vs_robust_gap_m"),
        "raw_vs_robust_gap_ratio": diagnostics.get("raw_vs_robust_gap_ratio"),
        "predicted_crown_width_m": diagnostics.get("predicted_crown_width_m"),
        "pca_major_span_m": diagnostics.get("pca_major_span_m"),
        "pca_minor_span_m": diagnostics.get("pca_minor_span_m"),
        "pca_mean_span_m": diagnostics.get("pca_mean_span_m"),
        "axis_vs_pca_gap_m": diagnostics.get("axis_vs_pca_gap_m"),
        "fixed_minus_pca_mean_m": diagnostics.get("fixed_minus_pca_mean_m"),
        "fixed_to_pca_ratio": diagnostics.get("fixed_to_pca_ratio"),
        "raw_axis_asymmetry_ratio": diagnostics.get("raw_axis_asymmetry_ratio"),
        "robust_axis_asymmetry_ratio": diagnostics.get("robust_axis_asymmetry_ratio"),
        "removed_outlier_fraction": diagnostics.get("removed_outlier_fraction"),
        "x_low_edge_support_count": diagnostics.get("x_low_edge_support_count"),
        "x_high_edge_support_count": diagnostics.get("x_high_edge_support_count"),
        "y_low_edge_support_count": diagnostics.get("y_low_edge_support_count"),
        "y_high_edge_support_count": diagnostics.get("y_high_edge_support_count"),
        "x_min_edge_support_fraction": diagnostics.get("x_min_edge_support_fraction"),
        "y_min_edge_support_fraction": diagnostics.get("y_min_edge_support_fraction"),
        "edge_support_band_m": diagnostics.get("edge_support_band_m"),
        "sparse_projection_suspected": bool(
            diagnostics.get("sparse_projection_suspected", False)
        ),
        "heavy_outlier_removal_suspected": bool(
            diagnostics.get("heavy_outlier_removal_suspected", False)
        ),
        "axis_pca_gap_suspected": bool(diagnostics.get("axis_pca_gap_suspected", False)),
        "elongated_projection_suspected": bool(
            diagnostics.get("elongated_projection_suspected", False)
        ),
        "robust_shrinkage_suspected": bool(
            diagnostics.get("robust_shrinkage_suspected", False)
        ),
        "edge_sparsity_suspected": bool(diagnostics.get("edge_sparsity_suspected", False)),
        "pca_axis_underestimate_suspected": bool(
            diagnostics.get("pca_axis_underestimate_suspected", False)
        ),
    }


def _build_summary(records: list[CrownWidthSanityRecord]) -> CrownWidthDiagnosisSummary:
    success_records = [record for record in records if record.status == "success"]
    total_records = len(records)
    success_count = len(success_records)
    success_rate = success_count / total_records if total_records else 0.0

    if success_records:
        abs_errors = [record.abs_error_m for record in success_records if record.abs_error_m is not None]
        mae = sum(abs_errors) / len(abs_errors)
        rmse = math.sqrt(sum(error * error for error in abs_errors) / len(abs_errors))
    else:
        mae = None
        rmse = None

    return CrownWidthDiagnosisSummary(
        total_records=total_records,
        success_count=success_count,
        success_rate=success_rate,
        mae=mae,
        rmse=rmse,
    )


def _build_failure_reason_stats(
    records: list[CrownWidthSanityRecord],
) -> list[CrownWidthFailureReasonStat]:
    counts: dict[tuple[str, str], int] = {}
    for record in records:
        if record.status == "success":
            continue
        key = (record.status, record.message)
        counts[key] = counts.get(key, 0) + 1

    return [
        CrownWidthFailureReasonStat(status=status, reason=reason, count=count)
        for (status, reason), count in sorted(counts.items())
    ]


def _build_suspicion_flag_stats(
    records: list[CrownWidthSanityRecord],
) -> list[CrownWidthSuspicionStat]:
    success_records = [record for record in records if record.status == "success"]
    flag_names = [
        "sparse_projection_suspected",
        "heavy_outlier_removal_suspected",
        "axis_pca_gap_suspected",
        "elongated_projection_suspected",
        "robust_shrinkage_suspected",
        "edge_sparsity_suspected",
        "pca_axis_underestimate_suspected",
    ]
    stats: list[CrownWidthSuspicionStat] = []
    for flag_name in flag_names:
        count = sum(1 for record in records if getattr(record, flag_name))
        success_count = sum(1 for record in success_records if getattr(record, flag_name))
        stats.append(
            CrownWidthSuspicionStat(
                flag_name=flag_name,
                count=count,
                rate_overall=count / len(records) if records else 0.0,
                rate_among_successes=(
                    success_count / len(success_records) if success_records else None
                ),
            )
        )
    return stats


def _select_top_error_records(
    records: list[CrownWidthSanityRecord],
    top_k_errors: int,
) -> list[CrownWidthSanityRecord]:
    success_records = [record for record in records if record.status == "success"]
    return sorted(
        success_records,
        key=lambda record: (record.abs_error_m is None, -(record.abs_error_m or 0.0)),
    )[:top_k_errors]


def _select_default_sample_ids(
    catalog: DataCatalog,
    modality: CrownWidthDiagnosisModality,
    limit: int,
) -> list[str]:
    sample_ids: list[str] = []
    for record in catalog.list_records():
        if modality not in record.available_modalities():
            continue
        if record.measured.crown_width_mean_m is None:
            continue
        sample_ids.append(record.sample_id)
        if len(sample_ids) >= limit:
            break
    return sample_ids


def _deduplicate_preserve_order(sample_ids: list[str]) -> list[str]:
    deduplicated: list[str] = []
    seen: set[str] = set()
    for sample_id in sample_ids:
        if sample_id in seen:
            continue
        seen.add(sample_id)
        deduplicated.append(sample_id)
    return deduplicated


def _write_records_csv(records: list[CrownWidthSanityRecord], path: Path) -> None:
    rows = [record.model_dump() for record in records]
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    field_names = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)
