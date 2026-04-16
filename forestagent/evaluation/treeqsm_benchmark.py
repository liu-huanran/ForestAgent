"""Small-sample TreeQSM baseline benchmark for q1/q2."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from forestagent.backends.base_backend import BaseBackend
from forestagent.backends.treeqsm_backend import TreeQSMBackend
from forestagent.data_catalog import DataCatalog
from forestagent.evaluation.batch_evaluator import (
    EvaluationMetrics,
    EvaluationRecord,
    evaluate_scalar_tasks,
)

BenchmarkModality = Literal["ground", "air"]
_SUPPORTED_BENCHMARK_TASK_IDS = ("q1_dbh", "q2_height")


class TreeQSMFailureReasonStat(BaseModel):
    """Aggregated count of one benchmark failure reason."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    status: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    count: int = Field(ge=1)


class TreeQSMBenchmarkResult(BaseModel):
    """Structured output of the minimal TreeQSM benchmark entry."""

    model_config = ConfigDict(extra="forbid")

    sample_ids: list[str]
    modality: BenchmarkModality
    records: list[EvaluationRecord]
    metrics: list[EvaluationMetrics]
    failure_reason_stats: list[TreeQSMFailureReasonStat]


def benchmark_treeqsm_baseline(
    data_dir: str | Path = "data",
    sample_ids: list[str] | None = None,
    limit: int = 3,
    modality: BenchmarkModality = "ground",
    treeqsm_root: str | Path | None = None,
    matlab_executable: str | Path | None = None,
    timeout_seconds: int = 900,
    backend: BaseBackend | None = None,
    catalog: DataCatalog | None = None,
) -> TreeQSMBenchmarkResult:
    """Run a small q1/q2 benchmark with the real TreeQSM backend."""

    if modality not in ("ground", "air"):
        raise ValueError(f"Unsupported modality: {modality}")
    if limit <= 0:
        raise ValueError("limit must be a positive integer.")

    catalog = catalog or DataCatalog.from_data_dir(data_dir)
    selected_sample_ids = (
        _deduplicate_preserve_order(sample_ids)
        if sample_ids is not None
        else _select_default_sample_ids(catalog, modality, limit)
    )
    if not selected_sample_ids:
        raise ValueError("No sample_ids were selected for the TreeQSM benchmark.")

    backend = backend or TreeQSMBackend(
        treeqsm_root=treeqsm_root,
        matlab_executable=matlab_executable,
        timeout_seconds=timeout_seconds,
    )
    evaluation_result = evaluate_scalar_tasks(
        data_dir=data_dir,
        task_ids=list(_SUPPORTED_BENCHMARK_TASK_IDS),
        modalities=[modality],
        sample_ids=selected_sample_ids,
        backend=backend,
        catalog=catalog,
    )
    metrics = [
        metric
        for metric in evaluation_result.metrics
        if metric.modality == modality and metric.task_id in _SUPPORTED_BENCHMARK_TASK_IDS
    ]
    return TreeQSMBenchmarkResult(
        sample_ids=selected_sample_ids,
        modality=modality,
        records=evaluation_result.records,
        metrics=metrics,
        failure_reason_stats=_build_failure_reason_stats(evaluation_result.records),
    )


def _select_default_sample_ids(
    catalog: DataCatalog,
    modality: BenchmarkModality,
    limit: int,
) -> list[str]:
    selected: list[str] = []
    for record in catalog.list_records():
        if modality not in record.available_modalities():
            continue
        if record.measured.dbh_cm is None or record.measured.height_m is None:
            continue
        selected.append(record.sample_id)
        if len(selected) >= limit:
            break
    return selected


def _build_failure_reason_stats(
    records: list[EvaluationRecord],
) -> list[TreeQSMFailureReasonStat]:
    counter: Counter[tuple[str, str, str]] = Counter()
    for record in records:
        if record.status == "success":
            continue
        reason = record.message.strip() or record.status
        counter[(record.task_id, record.status, reason)] += 1

    return [
        TreeQSMFailureReasonStat(
            task_id=task_id,
            status=status,
            reason=reason,
            count=count,
        )
        for (task_id, status, reason), count in sorted(
            counter.items(),
            key=lambda item: (item[0][0], item[0][1], -item[1], item[0][2]),
        )
    ]


def _deduplicate_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered
