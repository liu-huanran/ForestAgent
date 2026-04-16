"""Batch evaluation for scalar MVP tasks."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from forestagent.backends.base_backend import BaseBackend
from forestagent.backends.mock_backend import MockBackend
from forestagent.data_catalog import DataCatalog, TreeSampleRecord
from forestagent.pipelines import FixedPipeline
from forestagent.sample_runner import build_point_cloud_input

EvaluationStatus = Literal[
    "success",
    "missing_sample",
    "missing_modality",
    "missing_measured",
    "task_failed",
]
SupportedEvalTaskId = Literal["q1_dbh", "q2_height", "q3_crown_width"]

SUPPORTED_EVAL_TASK_IDS: tuple[SupportedEvalTaskId, ...] = (
    "q1_dbh",
    "q2_height",
    "q3_crown_width",
)
SUPPORTED_MODALITIES: tuple[str, ...] = ("ground", "air")


class EvaluationRecord(BaseModel):
    """Per-sample evaluation record for one task and one modality."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(min_length=1)
    modality: str = Field(min_length=1)
    task_id: SupportedEvalTaskId
    pred_value: float | None = None
    pred_unit: str | None = None
    measured_value: float | None = None
    measured_unit: str | None = None
    abs_error: float | None = None
    status: EvaluationStatus
    message: str = ""
    bias: float | None = None
    task_name: str | None = None
    plot_id: str | None = None
    file_name: str | None = None


class EvaluationMetrics(BaseModel):
    """Aggregated metrics for one task-modality group."""

    model_config = ConfigDict(extra="forbid")

    task_id: SupportedEvalTaskId
    modality: str
    total_records: int
    success_count: int
    success_rate: float
    mae: float | None = None
    rmse: float | None = None
    mean_bias: float | None = None


class ModalityComparison(BaseModel):
    """Ground vs air metric comparison for one task."""

    model_config = ConfigDict(extra="forbid")

    task_id: SupportedEvalTaskId
    ground_total_records: int = 0
    ground_success_count: int = 0
    ground_success_rate: float | None = None
    ground_mae: float | None = None
    ground_rmse: float | None = None
    ground_mean_bias: float | None = None
    air_total_records: int = 0
    air_success_count: int = 0
    air_success_rate: float | None = None
    air_mae: float | None = None
    air_rmse: float | None = None
    air_mean_bias: float | None = None


class BatchEvaluationResult(BaseModel):
    """Complete batch evaluation output."""

    model_config = ConfigDict(extra="forbid")

    records: list[EvaluationRecord]
    metrics: list[EvaluationMetrics]
    modality_comparison: list[ModalityComparison]


def evaluate_scalar_tasks(
    data_dir: str | Path = "data",
    task_ids: list[str] | None = None,
    modalities: list[str] | None = None,
    sample_ids: list[str] | None = None,
    backend: BaseBackend | None = None,
    catalog: DataCatalog | None = None,
) -> BatchEvaluationResult:
    """Evaluate q1/q2/q3 in batch over selected samples and modalities."""

    validated_task_ids = _validate_task_ids(task_ids)
    validated_modalities = _validate_modalities(modalities)
    catalog = catalog or DataCatalog.from_data_dir(data_dir)
    backend = backend or MockBackend()
    pipeline = FixedPipeline(backend)

    requested_sample_ids = sample_ids or [record.sample_id for record in catalog.list_records()]
    requested_sample_ids = _deduplicate_preserve_order(requested_sample_ids)

    records: list[EvaluationRecord] = []
    for sample_id in requested_sample_ids:
        try:
            record = catalog.get_record(sample_id)
        except KeyError:
            records.extend(
                _build_missing_sample_records(sample_id, validated_modalities, validated_task_ids)
            )
            continue

        for modality in validated_modalities:
            records.extend(
                _evaluate_record_tasks(
                    record=record,
                    modality=modality,
                    task_ids=validated_task_ids,
                    pipeline=pipeline,
                )
            )

    metrics = _compute_metrics(records)
    comparison = _build_modality_comparison(metrics)
    return BatchEvaluationResult(
        records=records,
        metrics=metrics,
        modality_comparison=comparison,
    )


def _evaluate_record_tasks(
    record: TreeSampleRecord,
    modality: str,
    task_ids: list[SupportedEvalTaskId],
    pipeline: FixedPipeline,
) -> list[EvaluationRecord]:
    if modality not in record.available_modalities():
        return [
            _build_missing_modality_record(record, modality, task_id)
            for task_id in task_ids
        ]

    point_cloud = build_point_cloud_input(record, modality)
    evaluated_records: list[EvaluationRecord] = []
    for task_id in task_ids:
        measured_value, measured_unit = _extract_measured_value(record, task_id)
        task_result = pipeline.run(task_id, point_cloud)
        task_name = task_result.task_name

        if task_result.status != "success":
            evaluated_records.append(
                EvaluationRecord(
                    sample_id=record.sample_id,
                    modality=modality,
                    task_id=task_id,
                    pred_value=None,
                    pred_unit=None,
                    measured_value=measured_value,
                    measured_unit=measured_unit,
                    abs_error=None,
                    status="task_failed",
                    message=task_result.message,
                    bias=None,
                    task_name=task_name,
                    plot_id=record.plot_id,
                    file_name=record.file_name,
                )
            )
            continue

        pred_value = float(task_result.result.value)
        pred_unit = task_result.result.unit
        if measured_value is None:
            evaluated_records.append(
                EvaluationRecord(
                    sample_id=record.sample_id,
                    modality=modality,
                    task_id=task_id,
                    pred_value=pred_value,
                    pred_unit=pred_unit,
                    measured_value=None,
                    measured_unit=measured_unit,
                    abs_error=None,
                    status="missing_measured",
                    message=f"No measured reference available for {task_id}.",
                    bias=None,
                    task_name=task_name,
                    plot_id=record.plot_id,
                    file_name=record.file_name,
                )
            )
            continue

        bias = pred_value - measured_value
        evaluated_records.append(
            EvaluationRecord(
                sample_id=record.sample_id,
                modality=modality,
                task_id=task_id,
                pred_value=pred_value,
                pred_unit=pred_unit,
                measured_value=measured_value,
                measured_unit=measured_unit,
                abs_error=abs(bias),
                status="success",
                message=task_result.message,
                bias=bias,
                task_name=task_name,
                plot_id=record.plot_id,
                file_name=record.file_name,
            )
        )

    return evaluated_records


def _build_missing_sample_records(
    sample_id: str,
    modalities: list[str],
    task_ids: list[SupportedEvalTaskId],
) -> list[EvaluationRecord]:
    return [
        EvaluationRecord(
            sample_id=sample_id,
            modality=modality,
            task_id=task_id,
            pred_value=None,
            pred_unit=None,
            measured_value=None,
            measured_unit=_measured_unit_for_task(task_id),
            abs_error=None,
            status="missing_sample",
            message=f"Sample {sample_id} was not found in the catalog.",
        )
        for modality in modalities
        for task_id in task_ids
    ]


def _build_missing_modality_record(
    record: TreeSampleRecord, modality: str, task_id: SupportedEvalTaskId
) -> EvaluationRecord:
    measured_value, measured_unit = _extract_measured_value(record, task_id)
    return EvaluationRecord(
        sample_id=record.sample_id,
        modality=modality,
        task_id=task_id,
        pred_value=None,
        pred_unit=None,
        measured_value=measured_value,
        measured_unit=measured_unit,
        abs_error=None,
        status="missing_modality",
        message=f"Sample {record.sample_id} does not have modality {modality}.",
        bias=None,
        plot_id=record.plot_id,
        file_name=record.file_name,
    )


def _extract_measured_value(
    record: TreeSampleRecord, task_id: SupportedEvalTaskId
) -> tuple[float | None, str]:
    if task_id == "q1_dbh":
        return record.measured.dbh_cm, "cm"
    if task_id == "q2_height":
        return record.measured.height_m, "m"
    if task_id == "q3_crown_width":
        return record.measured.crown_width_mean_m, "m"
    raise ValueError(f"Unsupported evaluation task: {task_id}")


def _measured_unit_for_task(task_id: SupportedEvalTaskId) -> str:
    if task_id == "q1_dbh":
        return "cm"
    return "m"


def _compute_metrics(records: list[EvaluationRecord]) -> list[EvaluationMetrics]:
    metrics: list[EvaluationMetrics] = []
    for task_id in SUPPORTED_EVAL_TASK_IDS:
        for modality in SUPPORTED_MODALITIES:
            grouped = [
                record
                for record in records
                if record.task_id == task_id and record.modality == modality
            ]
            if not grouped:
                continue

            successful = [record for record in grouped if record.status == "success"]
            total_records = len(grouped)
            success_count = len(successful)
            success_rate = success_count / total_records if total_records else 0.0

            if successful:
                abs_errors = [record.abs_error for record in successful if record.abs_error is not None]
                biases = [record.bias for record in successful if record.bias is not None]
                mae = sum(abs_errors) / len(abs_errors)
                rmse = math.sqrt(sum(error * error for error in abs_errors) / len(abs_errors))
                mean_bias = sum(biases) / len(biases)
            else:
                mae = None
                rmse = None
                mean_bias = None

            metrics.append(
                EvaluationMetrics(
                    task_id=task_id,
                    modality=modality,
                    total_records=total_records,
                    success_count=success_count,
                    success_rate=success_rate,
                    mae=mae,
                    rmse=rmse,
                    mean_bias=mean_bias,
                )
            )

    return metrics


def _build_modality_comparison(
    metrics: list[EvaluationMetrics],
) -> list[ModalityComparison]:
    by_task_modality = {(item.task_id, item.modality): item for item in metrics}
    comparison: list[ModalityComparison] = []
    for task_id in SUPPORTED_EVAL_TASK_IDS:
        ground = by_task_modality.get((task_id, "ground"))
        air = by_task_modality.get((task_id, "air"))
        comparison.append(
            ModalityComparison(
                task_id=task_id,
                ground_total_records=ground.total_records if ground else 0,
                ground_success_count=ground.success_count if ground else 0,
                ground_success_rate=ground.success_rate if ground else None,
                ground_mae=ground.mae if ground else None,
                ground_rmse=ground.rmse if ground else None,
                ground_mean_bias=ground.mean_bias if ground else None,
                air_total_records=air.total_records if air else 0,
                air_success_count=air.success_count if air else 0,
                air_success_rate=air.success_rate if air else None,
                air_mae=air.mae if air else None,
                air_rmse=air.rmse if air else None,
                air_mean_bias=air.mean_bias if air else None,
            )
        )
    return comparison


def _validate_task_ids(task_ids: list[str] | None) -> list[SupportedEvalTaskId]:
    task_ids = task_ids or list(SUPPORTED_EVAL_TASK_IDS)
    validated = _deduplicate_preserve_order(task_ids)
    unknown = [task_id for task_id in validated if task_id not in SUPPORTED_EVAL_TASK_IDS]
    if unknown:
        raise ValueError(f"Unsupported evaluation task ids: {unknown}")
    return validated


def _validate_modalities(modalities: list[str] | None) -> list[str]:
    modalities = modalities or list(SUPPORTED_MODALITIES)
    validated = _deduplicate_preserve_order(modalities)
    unknown = [modality for modality in validated if modality not in SUPPORTED_MODALITIES]
    if unknown:
        raise ValueError(f"Unsupported modalities: {unknown}")
    return validated


def _deduplicate_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result
