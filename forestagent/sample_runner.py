"""Helpers for running fixed tasks against cataloged real samples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from forestagent.backends.base_backend import BaseBackend
from forestagent.backends.mock_backend import MockBackend
from forestagent.data_catalog import DataCatalog, TreeSampleRecord
from forestagent.pipelines import FixedPipeline
from forestagent.renderers import render_task_answer
from forestagent.schemas import PointCloudInput


def build_point_cloud_input(record: TreeSampleRecord, modality: str) -> PointCloudInput:
    """Build PointCloudInput from one catalog record and a modality."""

    point_cloud_path = Path(record.point_cloud_path_for(modality))
    suffix = point_cloud_path.suffix.lower().lstrip(".")
    if not suffix:
        raise ValueError(f"Point cloud file has no suffix: {point_cloud_path}")
    return PointCloudInput(path=str(point_cloud_path), format=suffix)


def run_task_for_sample(
    task_id: str,
    sample_id: str,
    modality: str,
    data_dir: str | Path = "data",
    backend: BaseBackend | None = None,
    catalog: DataCatalog | None = None,
) -> dict[str, Any]:
    """Run one fixed task for one real sample and return a structured demo payload."""

    catalog = catalog or DataCatalog.from_data_dir(data_dir)
    record = catalog.get_record(sample_id)
    point_cloud_input = build_point_cloud_input(record, modality)
    pipeline = FixedPipeline(backend or MockBackend())
    task_result = pipeline.run(task_id, point_cloud_input)

    return {
        "sample": {
            "sample_id": record.sample_id,
            "plot_id": record.plot_id,
            "tree_number": record.tree_number,
            "file_name": record.file_name,
            "modality": modality,
            "point_cloud": point_cloud_input.model_dump(),
            "available_modalities": record.available_modalities(),
        },
        "measured_reference": record.measured_summary(),
        "task_result": task_result.model_dump(),
        "answer_text": render_task_answer(task_result),
    }

