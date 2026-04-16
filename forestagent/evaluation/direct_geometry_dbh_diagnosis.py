"""Diagnosis and comparison utilities for the direct geometry DBH baseline."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from forestagent.backends.direct_geometry_backend import DBHInspection, DirectGeometryBackend
from forestagent.data_catalog import DataCatalog
from forestagent.sample_runner import build_point_cloud_input

DiagnosisStatus = Literal[
    "success",
    "missing_sample",
    "missing_modality",
    "missing_measured",
    "task_failed",
]
DiagnosisModality = Literal["ground", "air"]

_DEFAULT_BASELINE_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "direct_geometry.yaml"
_DEFAULT_IMPROVED_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "direct_geometry_trunk_only.yaml"
)


class DBHSanityRecord(BaseModel):
    """Per-sample DBH diagnosis record with intermediate geometry details."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(min_length=1)
    modality: DiagnosisModality
    measured_dbh_cm: float | None = None
    pred_dbh_cm: float | None = None
    abs_error_cm: float | None = None
    status: DiagnosisStatus
    message: str = ""
    ground_z: float | None = None
    breast_height_z: float | None = None
    slice_point_count: int | None = None
    unique_xy_count: int | None = None
    outlier_removed_count: int | None = None
    fit_rmse: float | None = None
    fitted_radius_m: float | None = None
    fitted_center_x: float | None = None
    fitted_center_y: float | None = None
    selected_cluster_point_count: int | None = None
    cluster_candidate_count: int | None = None
    secondary_review_candidate_count: int | None = None
    secondary_review_selected: bool = False
    secondary_review_reason: str | None = None
    selected_component_index: int | None = None
    primary_component_index: int | None = None
    selected_cluster_spread_x_m: float | None = None
    selected_cluster_spread_y_m: float | None = None
    merged_cluster_suspected: bool = False
    merged_cluster_reasons: str = ""
    radial_trim_removed_count: int | None = None
    trunk_only_filtering_enabled: bool = False


class DBHDiagnosisSummary(BaseModel):
    """Aggregated DBH diagnosis summary for one config/run."""

    model_config = ConfigDict(extra="forbid")

    total_records: int
    success_count: int
    success_rate: float
    mae: float | None = None
    rmse: float | None = None


class DirectGeometryDBHDiagnosisResult(BaseModel):
    """One diagnosis run for a specific direct geometry config."""

    model_config = ConfigDict(extra="forbid")

    sample_ids: list[str]
    modality: DiagnosisModality
    config_path: str
    records: list[DBHSanityRecord]
    summary: DBHDiagnosisSummary
    sanity_table_path: str | None = None
    visualization_paths: list[str] = Field(default_factory=list)


class DirectGeometryDBHComparisonResult(BaseModel):
    """Before/after comparison between baseline and trunk-only DBH runs."""

    model_config = ConfigDict(extra="forbid")

    sample_ids: list[str]
    modality: DiagnosisModality
    baseline: DirectGeometryDBHDiagnosisResult
    improved: DirectGeometryDBHDiagnosisResult
    comparison_path: str | None = None


def compare_direct_geometry_dbh_configs(
    data_dir: str | Path = "data",
    sample_ids: list[str] | None = None,
    limit: int = 10,
    modality: DiagnosisModality = "ground",
    baseline_config_path: str | Path | None = None,
    improved_config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    top_k_visualizations: int = 5,
    catalog: DataCatalog | None = None,
) -> DirectGeometryDBHComparisonResult:
    """Run baseline and improved direct-geometry DBH diagnosis on the same samples."""

    catalog = catalog or DataCatalog.from_data_dir(data_dir)
    selected_sample_ids = (
        _deduplicate_preserve_order(sample_ids)
        if sample_ids is not None
        else _select_default_sample_ids(catalog, modality, limit)
    )
    if not selected_sample_ids:
        raise ValueError("No sample_ids were selected for direct geometry DBH diagnosis.")

    output_root = None if output_dir is None else Path(output_dir).resolve()
    baseline_output_dir = None if output_root is None else output_root / "baseline"
    improved_output_dir = None if output_root is None else output_root / "improved"

    baseline = run_direct_geometry_dbh_diagnosis(
        data_dir=data_dir,
        sample_ids=selected_sample_ids,
        modality=modality,
        config_path=baseline_config_path or _DEFAULT_BASELINE_CONFIG_PATH,
        output_dir=baseline_output_dir,
        top_k_visualizations=top_k_visualizations,
        catalog=catalog,
    )
    improved = run_direct_geometry_dbh_diagnosis(
        data_dir=data_dir,
        sample_ids=selected_sample_ids,
        modality=modality,
        config_path=improved_config_path or _DEFAULT_IMPROVED_CONFIG_PATH,
        output_dir=improved_output_dir,
        top_k_visualizations=top_k_visualizations,
        catalog=catalog,
    )

    comparison_path = None
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)
        comparison_path = str((output_root / "comparison_summary.json").resolve())
        Path(comparison_path).write_text(
            json.dumps(
                {
                    "sample_ids": selected_sample_ids,
                    "modality": modality,
                    "baseline": baseline.model_dump(),
                    "improved": improved.model_dump(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    return DirectGeometryDBHComparisonResult(
        sample_ids=selected_sample_ids,
        modality=modality,
        baseline=baseline,
        improved=improved,
        comparison_path=comparison_path,
    )


def run_direct_geometry_dbh_diagnosis(
    data_dir: str | Path = "data",
    sample_ids: list[str] | None = None,
    limit: int = 10,
    modality: DiagnosisModality = "ground",
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    top_k_visualizations: int = 5,
    catalog: DataCatalog | None = None,
) -> DirectGeometryDBHDiagnosisResult:
    """Run the direct geometry DBH diagnosis for one config on selected samples."""

    if modality not in ("ground", "air"):
        raise ValueError(f"Unsupported modality: {modality}")
    if limit <= 0:
        raise ValueError("limit must be a positive integer.")
    if top_k_visualizations < 0:
        raise ValueError("top_k_visualizations must be zero or positive.")

    catalog = catalog or DataCatalog.from_data_dir(data_dir)
    selected_sample_ids = (
        _deduplicate_preserve_order(sample_ids)
        if sample_ids is not None
        else _select_default_sample_ids(catalog, modality, limit)
    )
    if not selected_sample_ids:
        raise ValueError("No sample_ids were selected for direct geometry DBH diagnosis.")

    backend = DirectGeometryBackend(config_path=config_path)
    records: list[DBHSanityRecord] = []
    inspections_by_sample: dict[str, DBHInspection] = {}

    for sample_id in selected_sample_ids:
        records.append(
            _diagnose_one_sample(
                sample_id=sample_id,
                modality=modality,
                catalog=catalog,
                backend=backend,
                inspections_by_sample=inspections_by_sample,
            )
        )

    output_path = None if output_dir is None else Path(output_dir).resolve()
    sanity_table_path = None
    visualization_paths: list[str] = []
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
        sanity_table_path = str((output_path / "dbh_sanity_table.csv").resolve())
        _write_sanity_table_csv(records, Path(sanity_table_path))
        visualization_paths = _write_top_error_visualizations(
            records=records,
            inspections_by_sample=inspections_by_sample,
            output_dir=output_path / "slice_visualizations",
            top_k=top_k_visualizations,
        )

    return DirectGeometryDBHDiagnosisResult(
        sample_ids=selected_sample_ids,
        modality=modality,
        config_path=str(backend._config_path),
        records=records,
        summary=_build_summary(records),
        sanity_table_path=sanity_table_path,
        visualization_paths=visualization_paths,
    )


def _diagnose_one_sample(
    *,
    sample_id: str,
    modality: DiagnosisModality,
    catalog: DataCatalog,
    backend: DirectGeometryBackend,
    inspections_by_sample: dict[str, DBHInspection],
) -> DBHSanityRecord:
    try:
        record = catalog.get_record(sample_id)
    except KeyError:
        return DBHSanityRecord(
            sample_id=sample_id,
            modality=modality,
            status="missing_sample",
            message=f"Sample {sample_id} was not found in the catalog.",
        )

    if modality not in record.available_modalities():
        return DBHSanityRecord(
            sample_id=sample_id,
            modality=modality,
            measured_dbh_cm=record.measured.dbh_cm,
            status="missing_modality",
            message=f"Sample {sample_id} does not have modality {modality}.",
        )

    if record.measured.dbh_cm is None:
        return DBHSanityRecord(
            sample_id=sample_id,
            modality=modality,
            status="missing_measured",
            message=f"No measured DBH reference available for {sample_id}.",
        )

    inspection = backend.inspect_dbh(build_point_cloud_input(record, modality))
    inspections_by_sample[sample_id] = inspection
    result = inspection.tool_result
    diagnostics = result.extra
    pred_dbh_cm = None if result.status != "success" else float(result.value)
    abs_error_cm = None if pred_dbh_cm is None else abs(pred_dbh_cm - record.measured.dbh_cm)
    status: DiagnosisStatus = "success" if result.status == "success" else "task_failed"

    return DBHSanityRecord(
        sample_id=sample_id,
        modality=modality,
        measured_dbh_cm=record.measured.dbh_cm,
        pred_dbh_cm=pred_dbh_cm,
        abs_error_cm=abs_error_cm,
        status=status,
        message=result.message,
        ground_z=_optional_float(diagnostics.get("ground_reference_z_m")),
        breast_height_z=_optional_float(diagnostics.get("slice_target_z_m")),
        slice_point_count=_optional_int(diagnostics.get("raw_slice_point_count")),
        unique_xy_count=_optional_int(diagnostics.get("unique_slice_point_count")),
        outlier_removed_count=_optional_int(diagnostics.get("removed_outlier_count")),
        fit_rmse=_optional_float(diagnostics.get("fit_rmse_m")),
        fitted_radius_m=_optional_float(diagnostics.get("fitted_radius_m")),
        fitted_center_x=_optional_float(diagnostics.get("circle_center_x_m")),
        fitted_center_y=_optional_float(diagnostics.get("circle_center_y_m")),
        selected_cluster_point_count=_optional_int(diagnostics.get("selected_cluster_point_count")),
        cluster_candidate_count=_optional_int(diagnostics.get("cluster_candidate_count")),
        secondary_review_candidate_count=_optional_int(
            diagnostics.get("secondary_review_candidate_count")
        ),
        secondary_review_selected=bool(diagnostics.get("secondary_review_selected", False)),
        secondary_review_reason=_optional_str(diagnostics.get("secondary_review_reason")),
        selected_component_index=_optional_int(diagnostics.get("selected_component_index")),
        primary_component_index=_optional_int(diagnostics.get("primary_component_index")),
        selected_cluster_spread_x_m=_optional_float(diagnostics.get("selected_cluster_spread_x_m")),
        selected_cluster_spread_y_m=_optional_float(diagnostics.get("selected_cluster_spread_y_m")),
        merged_cluster_suspected=bool(diagnostics.get("merged_cluster_suspected", False)),
        merged_cluster_reasons=";".join(_string_list(diagnostics.get("merged_cluster_reasons"))),
        radial_trim_removed_count=_optional_int(diagnostics.get("radial_trim_removed_count")),
        trunk_only_filtering_enabled=bool(diagnostics.get("trunk_only_filtering_enabled", False)),
    )


def _build_summary(records: list[DBHSanityRecord]) -> DBHDiagnosisSummary:
    successful = [record for record in records if record.status == "success"]
    total_records = len(records)
    success_count = len(successful)
    success_rate = success_count / total_records if total_records else 0.0
    if successful:
        abs_errors = [record.abs_error_cm for record in successful if record.abs_error_cm is not None]
        mae = sum(abs_errors) / len(abs_errors)
        rmse = math.sqrt(sum(error * error for error in abs_errors) / len(abs_errors))
    else:
        mae = None
        rmse = None
    return DBHDiagnosisSummary(
        total_records=total_records,
        success_count=success_count,
        success_rate=success_rate,
        mae=mae,
        rmse=rmse,
    )


def _write_sanity_table_csv(records: list[DBHSanityRecord], output_path: Path) -> None:
    field_names = [
        "sample_id",
        "modality",
        "measured_dbh_cm",
        "pred_dbh_cm",
        "abs_error_cm",
        "status",
        "message",
        "ground_z",
        "breast_height_z",
        "slice_point_count",
        "unique_xy_count",
        "outlier_removed_count",
        "fit_rmse",
        "fitted_radius_m",
        "fitted_center_x",
        "fitted_center_y",
        "selected_cluster_point_count",
        "cluster_candidate_count",
        "secondary_review_candidate_count",
        "secondary_review_selected",
        "secondary_review_reason",
        "selected_component_index",
        "primary_component_index",
        "selected_cluster_spread_x_m",
        "selected_cluster_spread_y_m",
        "merged_cluster_suspected",
        "merged_cluster_reasons",
        "radial_trim_removed_count",
        "trunk_only_filtering_enabled",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for record in records:
            writer.writerow(record.model_dump())


def _write_top_error_visualizations(
    *,
    records: list[DBHSanityRecord],
    inspections_by_sample: dict[str, DBHInspection],
    output_dir: Path,
    top_k: int,
) -> list[str]:
    if top_k == 0:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(
        (
            record
            for record in records
            if record.status == "success" and record.abs_error_cm is not None
        ),
        key=lambda item: item.abs_error_cm,
        reverse=True,
    )[:top_k]
    output_paths: list[str] = []
    for record in ranked:
        inspection = inspections_by_sample.get(record.sample_id)
        if inspection is None:
            continue
        output_path = output_dir / f"{record.sample_id}_dbh_slice.svg"
        _write_slice_svg(record=record, inspection=inspection, output_path=output_path)
        output_paths.append(str(output_path.resolve()))
    return output_paths


def _write_slice_svg(
    *,
    record: DBHSanityRecord,
    inspection: DBHInspection,
    output_path: Path,
) -> None:
    width = 1200
    height = 340
    panel_width = 270
    panel_height = 220
    margin_x = 20
    margin_y = 60
    gap_x = 20
    panels = [
        ("Raw Slice", inspection.raw_slice_xy, None, None),
        ("Deduplicated", inspection.deduplicated_xy, None, None),
        ("Filtered", inspection.filtered_xy, None, None),
        ("Fitted Circle", inspection.filtered_xy, inspection.fitted_center_xy, inspection.fitted_radius_m),
    ]

    bounds = _combined_bounds(
        inspection.raw_slice_xy,
        inspection.deduplicated_xy,
        inspection.filtered_xy,
        inspection.fitted_center_xy,
        inspection.fitted_radius_m,
    )
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;font-size:14px;} .small{font-size:12px;} .title{font-size:16px;font-weight:bold;}</style>',
        f'<text x="{margin_x}" y="24" class="title">DBH Slice Diagnosis: {record.sample_id}</text>',
        f'<text x="{margin_x}" y="44" class="small">measured={_fmt(record.measured_dbh_cm)} cm, pred={_fmt(record.pred_dbh_cm)} cm, abs_error={_fmt(record.abs_error_cm)} cm</text>',
    ]

    for index, (title, points, center_xy, radius_m) in enumerate(panels):
        origin_x = margin_x + index * (panel_width + gap_x)
        origin_y = margin_y
        svg_parts.append(
            f'<rect x="{origin_x}" y="{origin_y}" width="{panel_width}" height="{panel_height}" fill="white" stroke="#bbbbbb" />'
        )
        svg_parts.append(
            f'<text x="{origin_x + 8}" y="{origin_y - 12}" class="small">{title}</text>'
        )
        if points.shape[0] == 0:
            svg_parts.append(
                f'<text x="{origin_x + 10}" y="{origin_y + 24}" class="small">No points</text>'
            )
            continue
        for x, y in points:
            panel_x, panel_y = _project_to_panel(
                x=x,
                y=y,
                bounds=bounds,
                origin_x=origin_x,
                origin_y=origin_y,
                width=panel_width,
                height=panel_height,
            )
            svg_parts.append(
                f'<circle cx="{panel_x:.2f}" cy="{panel_y:.2f}" r="1.8" fill="#1f77b4" fill-opacity="0.7" />'
            )
        if center_xy is not None and radius_m is not None:
            center_x, center_y = _project_to_panel(
                x=float(center_xy[0]),
                y=float(center_xy[1]),
                bounds=bounds,
                origin_x=origin_x,
                origin_y=origin_y,
                width=panel_width,
                height=panel_height,
            )
            scale = _panel_scale(bounds, panel_width, panel_height)
            svg_parts.append(
                f'<circle cx="{center_x:.2f}" cy="{center_y:.2f}" r="{radius_m * scale:.2f}" fill="none" stroke="#d62728" stroke-width="2" />'
            )
            svg_parts.append(
                f'<circle cx="{center_x:.2f}" cy="{center_y:.2f}" r="2.5" fill="#d62728" />'
            )

    svg_parts.append("</svg>")
    output_path.write_text("\n".join(svg_parts), encoding="utf-8")


def _combined_bounds(
    raw_slice_xy,
    deduplicated_xy,
    filtered_xy,
    fitted_center_xy,
    fitted_radius_m,
) -> tuple[float, float, float, float]:
    point_sets = [points for points in (raw_slice_xy, deduplicated_xy, filtered_xy) if points.shape[0] > 0]
    if not point_sets:
        return (0.0, 1.0, 0.0, 1.0)
    combined = point_sets[0] if len(point_sets) == 1 else np.vstack(point_sets)
    min_x = float(combined[:, 0].min())
    max_x = float(combined[:, 0].max())
    min_y = float(combined[:, 1].min())
    max_y = float(combined[:, 1].max())
    if fitted_center_xy is not None and fitted_radius_m is not None:
        min_x = min(min_x, float(fitted_center_xy[0] - fitted_radius_m))
        max_x = max(max_x, float(fitted_center_xy[0] + fitted_radius_m))
        min_y = min(min_y, float(fitted_center_xy[1] - fitted_radius_m))
        max_y = max(max_y, float(fitted_center_xy[1] + fitted_radius_m))
    if max_x - min_x < 1e-9:
        max_x = min_x + 1.0
    if max_y - min_y < 1e-9:
        max_y = min_y + 1.0
    return (min_x, max_x, min_y, max_y)


def _panel_scale(bounds: tuple[float, float, float, float], width: float, height: float) -> float:
    min_x, max_x, min_y, max_y = bounds
    data_span = max(max_x - min_x, max_y - min_y)
    drawable = min(width, height) - 20.0
    return drawable / data_span


def _project_to_panel(
    *,
    x: float,
    y: float,
    bounds: tuple[float, float, float, float],
    origin_x: float,
    origin_y: float,
    width: float,
    height: float,
) -> tuple[float, float]:
    min_x, max_x, min_y, max_y = bounds
    span = max(max_x - min_x, max_y - min_y)
    pad_x = (span - (max_x - min_x)) / 2.0
    pad_y = (span - (max_y - min_y)) / 2.0
    scale = _panel_scale(bounds, width, height)
    panel_x = origin_x + 10.0 + (x - min_x + pad_x) * scale
    panel_y = origin_y + height - 10.0 - (y - min_y + pad_y) * scale
    return panel_x, panel_y


def _select_default_sample_ids(
    catalog: DataCatalog,
    modality: DiagnosisModality,
    limit: int,
) -> list[str]:
    selected: list[str] = []
    for record in catalog.list_records():
        if modality not in record.available_modalities():
            continue
        if record.measured.dbh_cm is None:
            continue
        selected.append(record.sample_id)
        if len(selected) >= limit:
            break
    return selected


def _deduplicate_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.3f}"
