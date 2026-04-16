"""Diagnosis-ready freeze validation for the direct geometry q2_height baseline."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from forestagent.backends.direct_geometry_backend import (
    DirectGeometryBackend,
    HeightInspection,
)
from forestagent.data_catalog import DataCatalog
from forestagent.sample_runner import build_point_cloud_input

HeightDiagnosisStatus = Literal[
    "success",
    "missing_sample",
    "missing_modality",
    "missing_measured",
    "task_failed",
]
HeightDiagnosisModality = Literal["ground", "air"]


class HeightSanityRecord(BaseModel):
    """Per-sample q2_height diagnosis record with intermediate values and suspicion flags."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(min_length=1)
    modality: HeightDiagnosisModality
    measured_height_m: float | None = None
    pred_height_m: float | None = None
    abs_error_m: float | None = None
    status: HeightDiagnosisStatus
    message: str = ""
    ground_z: float | None = None
    top_z: float | None = None
    top_percentile: float | None = None
    top_point_count: int | None = None
    total_point_count: int | None = None
    height_value: float | None = None
    max_z: float | None = None
    top_support_window_m: float | None = None
    top_point_fraction: float | None = None
    top_support_z_span_m: float | None = None
    max_z_gap_m: float | None = None
    max_z_plateau_count: int | None = None
    max_z_plateau_ratio: float | None = None
    ground_band_point_count: int | None = None
    ground_band_iqr_m: float | None = None
    top_outlier_suspected: bool = False
    sparse_top_suspected: bool = False
    truncated_top_suspected: bool = False
    ground_reference_unstable_suspected: bool = False


class HeightDiagnosisSummary(BaseModel):
    """Aggregated q2_height freeze validation summary."""

    model_config = ConfigDict(extra="forbid")

    total_records: int
    success_count: int
    success_rate: float
    mae: float | None = None
    rmse: float | None = None


class HeightFailureReasonStat(BaseModel):
    """Aggregated count of one q2_height failure reason."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    count: int = Field(ge=1)


class HeightSuspicionStat(BaseModel):
    """Aggregated count/rate for one suspicion flag."""

    model_config = ConfigDict(extra="forbid")

    flag_name: str = Field(min_length=1)
    count: int = Field(ge=0)
    rate_overall: float = Field(ge=0.0, le=1.0)
    rate_among_successes: float | None = Field(default=None, ge=0.0, le=1.0)


class DirectGeometryHeightFreezeValidationResult(BaseModel):
    """Structured q2_height freeze validation output."""

    model_config = ConfigDict(extra="forbid")

    sample_ids: list[str]
    modality: HeightDiagnosisModality
    config_path: str
    records: list[HeightSanityRecord]
    summary: HeightDiagnosisSummary
    failure_reason_stats: list[HeightFailureReasonStat]
    suspicion_flag_stats: list[HeightSuspicionStat]
    top_absolute_error_records: list[HeightSanityRecord]
    per_sample_records_path: str | None = None
    top_abs_error_path: str | None = None
    summary_path: str | None = None


def run_direct_geometry_height_freeze_validation(
    data_dir: str | Path = "data",
    sample_ids: list[str] | None = None,
    limit: int = 50,
    modality: HeightDiagnosisModality = "ground",
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    top_k_errors: int = 20,
    catalog: DataCatalog | None = None,
) -> DirectGeometryHeightFreezeValidationResult:
    """Run q2_height freeze validation with diagnosis fields and suspicion flags."""

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
        raise ValueError("No sample_ids were selected for direct geometry q2_height validation.")

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

    return DirectGeometryHeightFreezeValidationResult(
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
    modality: HeightDiagnosisModality,
    catalog: DataCatalog,
    backend: DirectGeometryBackend,
) -> HeightSanityRecord:
    try:
        record = catalog.get_record(sample_id)
    except KeyError:
        return HeightSanityRecord(
            sample_id=sample_id,
            modality=modality,
            status="missing_sample",
            message=f"Sample {sample_id} was not found in the catalog.",
        )

    if modality not in record.available_modalities():
        return HeightSanityRecord(
            sample_id=sample_id,
            modality=modality,
            measured_height_m=record.measured.height_m,
            status="missing_modality",
            message=f"Sample {sample_id} does not have modality {modality}.",
        )

    point_cloud = build_point_cloud_input(record, modality)
    inspection = backend.inspect_height(point_cloud)
    tool_result = inspection.tool_result
    diagnostics = tool_result.extra
    measured_height_m = record.measured.height_m
    pred_height_m = float(tool_result.value) if tool_result.value is not None else None
    abs_error_m = (
        None
        if measured_height_m is None or pred_height_m is None
        else abs(pred_height_m - measured_height_m)
    )

    if tool_result.status != "success":
        return HeightSanityRecord(
            sample_id=sample_id,
            modality=modality,
            measured_height_m=measured_height_m,
            pred_height_m=None,
            abs_error_m=None,
            status="task_failed",
            message=tool_result.message,
            **_height_record_fields_from_diagnostics(diagnostics),
        )

    record_status: HeightDiagnosisStatus = (
        "success" if measured_height_m is not None else "missing_measured"
    )
    record_message = (
        tool_result.message
        if measured_height_m is not None
        else "No measured reference available for q2_height."
    )
    return HeightSanityRecord(
        sample_id=sample_id,
        modality=modality,
        measured_height_m=measured_height_m,
        pred_height_m=pred_height_m,
        abs_error_m=abs_error_m,
        status=record_status,
        message=record_message,
        **_height_record_fields_from_diagnostics(diagnostics),
    )


def _height_record_fields_from_diagnostics(diagnostics: dict[str, object]) -> dict[str, object]:
    return {
        "ground_z": diagnostics.get("ground_reference_z_m"),
        "top_z": diagnostics.get("top_reference_z_m"),
        "top_percentile": diagnostics.get("top_percentile"),
        "top_point_count": diagnostics.get("top_point_count"),
        "total_point_count": diagnostics.get("total_point_count"),
        "height_value": diagnostics.get("height_value_m"),
        "max_z": diagnostics.get("max_z_m"),
        "top_support_window_m": diagnostics.get("top_support_window_m"),
        "top_point_fraction": diagnostics.get("top_point_fraction"),
        "top_support_z_span_m": diagnostics.get("top_support_z_span_m"),
        "max_z_gap_m": diagnostics.get("max_z_gap_m"),
        "max_z_plateau_count": diagnostics.get("max_z_plateau_count"),
        "max_z_plateau_ratio": diagnostics.get("max_z_plateau_ratio"),
        "ground_band_point_count": diagnostics.get("ground_band_point_count"),
        "ground_band_iqr_m": diagnostics.get("ground_band_iqr_m"),
        "top_outlier_suspected": bool(diagnostics.get("top_outlier_suspected", False)),
        "sparse_top_suspected": bool(diagnostics.get("sparse_top_suspected", False)),
        "truncated_top_suspected": bool(diagnostics.get("truncated_top_suspected", False)),
        "ground_reference_unstable_suspected": bool(
            diagnostics.get("ground_reference_unstable_suspected", False)
        ),
    }


def _build_summary(records: list[HeightSanityRecord]) -> HeightDiagnosisSummary:
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

    return HeightDiagnosisSummary(
        total_records=total_records,
        success_count=success_count,
        success_rate=success_rate,
        mae=mae,
        rmse=rmse,
    )


def _build_failure_reason_stats(
    records: list[HeightSanityRecord],
) -> list[HeightFailureReasonStat]:
    counter: dict[tuple[str, str], int] = {}
    for record in records:
        if record.status == "success":
            continue
        key = (record.status, record.message.strip() or record.status)
        counter[key] = counter.get(key, 0) + 1

    return [
        HeightFailureReasonStat(status=status, reason=reason, count=count)
        for (status, reason), count in sorted(
            counter.items(),
            key=lambda item: (item[0][0], -item[1], item[0][1]),
        )
    ]


def _build_suspicion_flag_stats(
    records: list[HeightSanityRecord],
) -> list[HeightSuspicionStat]:
    success_records = [record for record in records if record.status == "success"]
    flag_names = [
        "top_outlier_suspected",
        "sparse_top_suspected",
        "truncated_top_suspected",
        "ground_reference_unstable_suspected",
    ]
    stats: list[HeightSuspicionStat] = []
    for flag_name in flag_names:
        count = sum(1 for record in records if getattr(record, flag_name))
        success_count = sum(1 for record in success_records if getattr(record, flag_name))
        stats.append(
            HeightSuspicionStat(
                flag_name=flag_name,
                count=count,
                rate_overall=(count / len(records)) if records else 0.0,
                rate_among_successes=(
                    success_count / len(success_records) if success_records else None
                ),
            )
        )
    return stats


def _select_top_error_records(
    records: list[HeightSanityRecord],
    top_k_errors: int,
) -> list[HeightSanityRecord]:
    successful = [record for record in records if record.status == "success"]
    return sorted(
        successful,
        key=lambda record: (-float(record.abs_error_m or 0.0), record.sample_id),
    )[:top_k_errors]


def _write_records_csv(records: list[HeightSanityRecord], output_path: Path) -> None:
    fieldnames = list(HeightSanityRecord.model_fields.keys())
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.model_dump())


def _select_default_sample_ids(
    catalog: DataCatalog,
    modality: HeightDiagnosisModality,
    limit: int,
) -> list[str]:
    selected: list[str] = []
    for record in catalog.list_records():
        if modality not in record.available_modalities():
            continue
        if record.measured.height_m is None:
            continue
        selected.append(record.sample_id)
        if len(selected) >= limit:
            break
    return selected


def _deduplicate_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered
