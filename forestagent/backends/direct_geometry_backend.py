"""Direct geometry backend for q1_dbh, q2_height, and q3_crown_width baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from forestagent.backends.base_backend import BaseBackend
from forestagent.config_loader import load_yaml_mapping
from forestagent.schemas import PointCloudInput, ToolResult

_SUPPORTED_POINT_CLOUD_FORMATS = frozenset({"las", "laz"})
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "direct_geometry.yaml"


class DirectGeometryBackendError(RuntimeError):
    """Raised when direct geometry estimation cannot produce a valid metric."""


@dataclass(frozen=True)
class TrunkFilteringConfig:
    enabled: bool
    xy_cluster_radius_m: float
    min_cluster_points: int
    min_angular_coverage: float
    cluster_selection_metric: str
    radial_trim_mad_threshold: float | None
    secondary_review_enabled: bool
    secondary_review_min_angular_coverage: float
    secondary_review_max_fit_rmse_m: float
    secondary_review_max_radius_ratio_vs_primary: float
    secondary_review_min_score_ratio_vs_primary: float
    suspicion_large_point_count_threshold: int
    suspicion_spread_to_diameter_ratio_threshold: float
    suspicion_large_candidate_radius_m: float
    suspicion_min_large_candidate_count: int


@dataclass(frozen=True)
class DBHConfig:
    breast_height_m: float
    slice_thickness_m: float
    min_slice_points: int
    circle_fitting_method: str
    radial_outlier_mad_threshold: float
    min_radius_m: float
    max_radius_m: float
    trunk_filtering: TrunkFilteringConfig


@dataclass(frozen=True)
class HeightConfig:
    top_percentile: float
    min_point_count: int
    top_support_window_m: float
    top_outlier_gap_threshold_m: float
    sparse_top_min_point_count: int
    sparse_top_min_fraction: float
    truncated_top_span_threshold_m: float
    truncated_top_plateau_tolerance_m: float
    truncated_top_plateau_ratio_threshold: float
    ground_stability_band_m: float
    ground_stability_max_iqr_m: float


@dataclass(frozen=True)
class CrownWidthConfig:
    robust_low_percentile: float
    robust_high_percentile: float
    min_projected_points: int
    radial_outlier_mad_threshold: float
    sparse_projection_min_unique_points: int
    heavy_outlier_removal_fraction_threshold: float
    axis_pca_gap_threshold_m: float
    elongation_ratio_threshold: float
    robust_shrinkage_ratio_threshold: float
    pca_axis_underestimate_threshold_m: float
    edge_support_band_m: float
    edge_support_min_point_count: int
    edge_support_min_fraction: float


@dataclass(frozen=True)
class DirectGeometryConfig:
    ground_reference_percentile: float
    dbh: DBHConfig
    height: HeightConfig
    crown_width: CrownWidthConfig


@dataclass(frozen=True)
class TrunkClusterCandidate:
    component_index: int
    points_xy: np.ndarray
    center_xy: np.ndarray
    radius_m: float
    fit_rmse_m: float
    angular_coverage: float
    selection_score: float
    point_count: int
    spread_x_m: float
    spread_y_m: float


@dataclass(frozen=True)
class DBHInspection:
    tool_result: ToolResult
    raw_slice_xy: np.ndarray
    deduplicated_xy: np.ndarray
    filtered_xy: np.ndarray
    fitted_center_xy: np.ndarray | None
    fitted_radius_m: float | None
    fit_rmse_m: float | None


@dataclass(frozen=True)
class HeightInspection:
    tool_result: ToolResult
    ground_z_m: float | None
    top_z_m: float | None
    max_z_m: float | None
    height_m: float | None
    top_point_count: int | None
    total_point_count: int | None


@dataclass(frozen=True)
class CrownWidthInspection:
    tool_result: ToolResult
    projected_xy: np.ndarray
    unique_xy: np.ndarray
    filtered_xy: np.ndarray


class DirectGeometryBackend(BaseBackend):
    """Local geometry baseline backend for q1_dbh, q2_height, and q3_crown_width."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        resolved_config_path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG_PATH
        self._config_path = resolved_config_path.resolve()
        self._config = self._load_config(self._config_path)
        self._reset_trunk_selection_state()

    def inspect_dbh(self, point_cloud: PointCloudInput) -> DBHInspection:
        raw_slice_xy = np.empty((0, 2), dtype=np.float64)
        deduplicated_xy = np.empty((0, 2), dtype=np.float64)
        filtered_xy = np.empty((0, 2), dtype=np.float64)
        diagnostics: dict[str, object] = {
            "backend": "direct_geometry",
            "input_format": point_cloud.format,
            "config_path": str(self._config_path),
            "trunk_only_filtering_enabled": self._config.dbh.trunk_filtering.enabled,
            "circle_fitting_method": self._config.dbh.circle_fitting_method,
            "cluster_selection_metric": self._config.dbh.trunk_filtering.cluster_selection_metric,
        }
        fitted_center_xy: np.ndarray | None = None
        fitted_radius_m: float | None = None
        fit_rmse_m: float | None = None

        try:
            points = self._load_point_cloud_matrix(point_cloud)
            ground_z = self._estimate_ground_reference(points[:, 2])
            target_z = ground_z + self._config.dbh.breast_height_m
            raw_slice_xy = self._slice_breast_height(points, target_z)
            deduplicated_xy = np.unique(raw_slice_xy, axis=0)
            filtered_xy, diagnostics = self._prepare_dbh_slice(
                deduplicated_xy=deduplicated_xy,
                diagnostics={
                    **diagnostics,
                    "ground_reference_z_m": ground_z,
                    "slice_target_z_m": target_z,
                    "raw_slice_point_count": int(raw_slice_xy.shape[0]),
                    "unique_slice_point_count": int(deduplicated_xy.shape[0]),
                    "removed_duplicate_count": int(raw_slice_xy.shape[0] - deduplicated_xy.shape[0]),
                },
            )

            center_xy, radius_m, fit_rmse_m = self._fit_circle(
                filtered_xy,
                method=self._config.dbh.circle_fitting_method,
            )
            if radius_m < self._config.dbh.min_radius_m or radius_m > self._config.dbh.max_radius_m:
                raise DirectGeometryBackendError(
                    f"Estimated DBH radius {radius_m:.4f} m is outside the configured range."
                )

            fitted_center_xy = center_xy
            fitted_radius_m = radius_m
            diagnostics = {
                **diagnostics,
                "filtered_slice_point_count": int(filtered_xy.shape[0]),
                "fitted_radius_m": radius_m,
                "circle_center_x_m": float(center_xy[0]),
                "circle_center_y_m": float(center_xy[1]),
                "fit_rmse_m": fit_rmse_m,
            }
            tool_result = ToolResult(
                tool_name="estimate_dbh",
                status="success",
                value=radius_m * 200.0,
                unit="cm",
                confidence=None,
                extra=diagnostics,
                message="Direct geometry DBH estimated from breast-height slice circle fitting.",
            )
        except DirectGeometryBackendError as exc:
            tool_result = self._build_failed_result(
                "estimate_dbh",
                point_cloud,
                str(exc),
                diagnostics=diagnostics,
            )

        return DBHInspection(
            tool_result=tool_result,
            raw_slice_xy=raw_slice_xy,
            deduplicated_xy=deduplicated_xy,
            filtered_xy=filtered_xy,
            fitted_center_xy=fitted_center_xy,
            fitted_radius_m=fitted_radius_m,
            fit_rmse_m=fit_rmse_m,
        )

    def estimate_dbh(self, point_cloud: PointCloudInput) -> ToolResult:
        return self.inspect_dbh(point_cloud).tool_result

    def inspect_height(self, point_cloud: PointCloudInput) -> HeightInspection:
        diagnostics: dict[str, object] = {
            "backend": "direct_geometry",
            "input_format": point_cloud.format,
            "config_path": str(self._config_path),
        }
        ground_z: float | None = None
        top_z: float | None = None
        max_z: float | None = None
        height_m: float | None = None
        top_point_count: int | None = None
        total_point_count: int | None = None

        try:
            points = self._load_point_cloud_matrix(point_cloud)
            total_point_count = int(points.shape[0])
            diagnostics["total_point_count"] = total_point_count
            if points.shape[0] < self._config.height.min_point_count:
                raise DirectGeometryBackendError(
                    "Point cloud does not contain enough points for height estimation."
                )

            z_values = points[:, 2]
            ground_seed_z = self._estimate_ground_reference(z_values)
            ground_support_mask = (
                z_values <= (ground_seed_z + self._config.height.ground_stability_band_m)
            )
            ground_support_values = z_values[ground_support_mask]
            if ground_support_values.shape[0] == 0:
                raise DirectGeometryBackendError(
                    "Ground support region is empty for height estimation."
                )

            ground_z = float(np.min(ground_support_values))
            ground_seed_gap_m = ground_seed_z - ground_z
            ground_support_point_count = int(ground_support_values.shape[0])
            ground_support_span_m = float(np.max(ground_support_values) - ground_z)
            if ground_support_values.shape[0] >= 4:
                ground_band_iqr_m = float(
                    np.percentile(ground_support_values, 75.0)
                    - np.percentile(ground_support_values, 25.0)
                )
            else:
                ground_band_iqr_m = 0.0

            top_seed_z = float(np.percentile(z_values, self._config.height.top_percentile))
            top_support_mask = z_values >= (top_seed_z - self._config.height.top_support_window_m)
            top_support_values = z_values[top_support_mask]
            if top_support_values.shape[0] == 0:
                raise DirectGeometryBackendError(
                    "Top support region is empty for height estimation."
                )

            top_z = float(np.max(top_support_values))
            max_z = float(np.max(z_values))
            height_m = top_z - ground_z
            if not np.isfinite(height_m) or height_m <= 0:
                raise DirectGeometryBackendError("Estimated height is not a positive finite value.")

            top_point_count = int(top_support_values.shape[0])
            top_point_fraction = top_point_count / max(total_point_count, 1)
            top_support_z_min = float(np.min(top_support_values))
            top_support_z_span = float(np.max(top_support_values) - top_support_z_min)
            max_z_gap = max_z - top_seed_z
            max_z_plateau_mask = z_values >= (
                top_z - self._config.height.truncated_top_plateau_tolerance_m
            )
            max_z_plateau_count = int(np.count_nonzero(max_z_plateau_mask))
            max_z_plateau_ratio = max_z_plateau_count / max(top_point_count, 1)

            top_outlier_suspected = max_z_gap >= self._config.height.top_outlier_gap_threshold_m
            sparse_top_suspected = (
                top_point_count < self._config.height.sparse_top_min_point_count
                or top_point_fraction < self._config.height.sparse_top_min_fraction
            )
            truncated_top_suspected = (
                top_support_z_span <= self._config.height.truncated_top_span_threshold_m
                and max_z_plateau_ratio
                >= self._config.height.truncated_top_plateau_ratio_threshold
                and top_point_count >= self._config.height.sparse_top_min_point_count
            )
            ground_reference_unstable_suspected = (
                ground_seed_gap_m >= self._config.height.ground_stability_max_iqr_m
                and ground_support_point_count < self._config.height.min_point_count
            )

            diagnostics = {
                **diagnostics,
                "ground_reference_z_m": ground_z,
                "ground_seed_z_m": ground_seed_z,
                "ground_seed_gap_m": ground_seed_gap_m,
                "ground_reference_percentile": self._config.ground_reference_percentile,
                "ground_band_point_count": ground_support_point_count,
                "ground_band_iqr_m": ground_band_iqr_m,
                "ground_support_span_m": ground_support_span_m,
                "top_reference_z_m": top_z,
                "top_seed_z_m": top_seed_z,
                "top_percentile": self._config.height.top_percentile,
                "top_support_window_m": self._config.height.top_support_window_m,
                "top_point_count": top_point_count,
                "top_point_fraction": top_point_fraction,
                "top_support_z_min_m": top_support_z_min,
                "top_support_z_span_m": top_support_z_span,
                "max_z_m": max_z,
                "max_z_gap_m": max_z_gap,
                "max_z_plateau_count": max_z_plateau_count,
                "max_z_plateau_ratio": max_z_plateau_ratio,
                "height_value_m": height_m,
                "top_outlier_suspected": top_outlier_suspected,
                "sparse_top_suspected": sparse_top_suspected,
                "truncated_top_suspected": truncated_top_suspected,
                "ground_reference_unstable_suspected": ground_reference_unstable_suspected,
            }
            tool_result = ToolResult(
                tool_name="estimate_height",
                status="success",
                value=height_m,
                unit="m",
                confidence=None,
                extra=diagnostics,
                message="Direct geometry height estimated from top support minus ground support reference.",
            )
        except DirectGeometryBackendError as exc:
            tool_result = self._build_failed_result(
                "estimate_height",
                point_cloud,
                str(exc),
                diagnostics=diagnostics,
            )

        return HeightInspection(
            tool_result=tool_result,
            ground_z_m=ground_z,
            top_z_m=top_z,
            max_z_m=max_z,
            height_m=height_m,
            top_point_count=top_point_count,
            total_point_count=total_point_count,
        )

    def estimate_height(self, point_cloud: PointCloudInput) -> ToolResult:
        return self.inspect_height(point_cloud).tool_result

    def inspect_crown_width(self, point_cloud: PointCloudInput) -> CrownWidthInspection:
        diagnostics: dict[str, object] = {
            "backend": "direct_geometry",
            "input_format": point_cloud.format,
            "config_path": str(self._config_path),
            "estimation_method": "fixed_axis_robust_span_mean",
        }
        projected_xy = np.empty((0, 2), dtype=np.float64)
        unique_xy = np.empty((0, 2), dtype=np.float64)
        filtered_xy = np.empty((0, 2), dtype=np.float64)

        try:
            points = self._load_point_cloud_matrix(point_cloud)
            projected_xy = points[:, :2]
            total_point_count = int(points.shape[0])
            projected_point_count = int(projected_xy.shape[0])
            unique_xy = np.unique(projected_xy, axis=0)
            removed_duplicate_count = int(projected_xy.shape[0] - unique_xy.shape[0])

            raw_x_span_m = float(projected_xy[:, 0].max() - projected_xy[:, 0].min())
            raw_y_span_m = float(projected_xy[:, 1].max() - projected_xy[:, 1].min())

            filtered_xy, removed_outlier_count = self._filter_radial_outliers(
                unique_xy,
                threshold=self._config.crown_width.radial_outlier_mad_threshold,
            )
            if filtered_xy.shape[0] < self._config.crown_width.min_projected_points:
                raise DirectGeometryBackendError(
                    "Projected crown footprint does not contain enough filtered points."
                )

            x_robust_span_m = self._robust_span(
                filtered_xy[:, 0],
                low_percentile=self._config.crown_width.robust_low_percentile,
                high_percentile=self._config.crown_width.robust_high_percentile,
            )
            y_robust_span_m = self._robust_span(
                filtered_xy[:, 1],
                low_percentile=self._config.crown_width.robust_low_percentile,
                high_percentile=self._config.crown_width.robust_high_percentile,
            )
            predicted_crown_width_m = (x_robust_span_m + y_robust_span_m) / 2.0
            if not np.isfinite(predicted_crown_width_m) or predicted_crown_width_m <= 0:
                raise DirectGeometryBackendError(
                    "Estimated crown width is not a positive finite value."
                )

            pca_major_span_m, pca_minor_span_m = self._pca_robust_spans(
                filtered_xy,
                low_percentile=self._config.crown_width.robust_low_percentile,
                high_percentile=self._config.crown_width.robust_high_percentile,
            )
            pca_mean_span_m = (pca_major_span_m + pca_minor_span_m) / 2.0
            axis_vs_pca_gap_m = abs(predicted_crown_width_m - pca_mean_span_m)
            fixed_minus_pca_mean_m = predicted_crown_width_m - pca_mean_span_m

            unique_xy_count = int(unique_xy.shape[0])
            removed_outlier_fraction = removed_outlier_count / max(unique_xy_count, 1)
            raw_mean_span_m = (raw_x_span_m + raw_y_span_m) / 2.0
            robust_mean_span_m = predicted_crown_width_m
            raw_vs_robust_gap_m = raw_mean_span_m - robust_mean_span_m
            raw_vs_robust_gap_ratio = raw_vs_robust_gap_m / max(raw_mean_span_m, 1e-9)
            fixed_to_pca_ratio = predicted_crown_width_m / max(pca_mean_span_m, 1e-9)
            axis_ratio = max(x_robust_span_m, y_robust_span_m) / max(
                min(x_robust_span_m, y_robust_span_m),
                1e-9,
            )
            raw_axis_ratio = max(raw_x_span_m, raw_y_span_m) / max(
                min(raw_x_span_m, raw_y_span_m),
                1e-9,
            )
            x_raw_vs_robust_gap_m = raw_x_span_m - x_robust_span_m
            y_raw_vs_robust_gap_m = raw_y_span_m - y_robust_span_m

            min_x = float(unique_xy[:, 0].min())
            max_x = float(unique_xy[:, 0].max())
            min_y = float(unique_xy[:, 1].min())
            max_y = float(unique_xy[:, 1].max())
            edge_band_m = self._config.crown_width.edge_support_band_m
            x_low_edge_support_count = int(np.count_nonzero(unique_xy[:, 0] <= (min_x + edge_band_m)))
            x_high_edge_support_count = int(np.count_nonzero(unique_xy[:, 0] >= (max_x - edge_band_m)))
            y_low_edge_support_count = int(np.count_nonzero(unique_xy[:, 1] <= (min_y + edge_band_m)))
            y_high_edge_support_count = int(np.count_nonzero(unique_xy[:, 1] >= (max_y - edge_band_m)))
            x_min_edge_support_fraction = min(x_low_edge_support_count, x_high_edge_support_count) / max(
                unique_xy_count,
                1,
            )
            y_min_edge_support_fraction = min(y_low_edge_support_count, y_high_edge_support_count) / max(
                unique_xy_count,
                1,
            )

            sparse_projection_suspected = (
                unique_xy_count < self._config.crown_width.sparse_projection_min_unique_points
            )
            heavy_outlier_removal_suspected = (
                removed_outlier_fraction
                >= self._config.crown_width.heavy_outlier_removal_fraction_threshold
            )
            axis_pca_gap_suspected = (
                axis_vs_pca_gap_m >= self._config.crown_width.axis_pca_gap_threshold_m
            )
            elongated_projection_suspected = (
                axis_ratio >= self._config.crown_width.elongation_ratio_threshold
            )
            robust_shrinkage_suspected = (
                raw_vs_robust_gap_ratio >= self._config.crown_width.robust_shrinkage_ratio_threshold
            )
            pca_axis_underestimate_suspected = (
                (pca_mean_span_m - predicted_crown_width_m)
                >= self._config.crown_width.pca_axis_underestimate_threshold_m
            )
            edge_sparsity_suspected = (
                (
                    min(
                        x_low_edge_support_count,
                        x_high_edge_support_count,
                        y_low_edge_support_count,
                        y_high_edge_support_count,
                    )
                    < self._config.crown_width.edge_support_min_point_count
                )
                or (
                    min(x_min_edge_support_fraction, y_min_edge_support_fraction)
                    < self._config.crown_width.edge_support_min_fraction
                )
            )

            diagnostics = {
                **diagnostics,
                "total_point_count": total_point_count,
                "projected_point_count": projected_point_count,
                "unique_xy_count": unique_xy_count,
                "removed_duplicate_count": removed_duplicate_count,
                "removed_outlier_count": removed_outlier_count,
                "x_raw_span_m": raw_x_span_m,
                "y_raw_span_m": raw_y_span_m,
                "x_robust_span_m": x_robust_span_m,
                "y_robust_span_m": y_robust_span_m,
                "x_raw_vs_robust_gap_m": x_raw_vs_robust_gap_m,
                "y_raw_vs_robust_gap_m": y_raw_vs_robust_gap_m,
                "raw_mean_span_m": raw_mean_span_m,
                "robust_mean_span_m": robust_mean_span_m,
                "raw_vs_robust_gap_m": raw_vs_robust_gap_m,
                "raw_vs_robust_gap_ratio": raw_vs_robust_gap_ratio,
                "predicted_crown_width_m": predicted_crown_width_m,
                "pca_major_span_m": pca_major_span_m,
                "pca_minor_span_m": pca_minor_span_m,
                "pca_mean_span_m": pca_mean_span_m,
                "axis_vs_pca_gap_m": axis_vs_pca_gap_m,
                "fixed_minus_pca_mean_m": fixed_minus_pca_mean_m,
                "fixed_to_pca_ratio": fixed_to_pca_ratio,
                "robust_low_percentile": self._config.crown_width.robust_low_percentile,
                "robust_high_percentile": self._config.crown_width.robust_high_percentile,
                "raw_axis_asymmetry_ratio": raw_axis_ratio,
                "robust_axis_asymmetry_ratio": axis_ratio,
                "sparse_projection_suspected": sparse_projection_suspected,
                "heavy_outlier_removal_suspected": heavy_outlier_removal_suspected,
                "axis_pca_gap_suspected": axis_pca_gap_suspected,
                "elongated_projection_suspected": elongated_projection_suspected,
                "projection_axis_ratio": axis_ratio,
                "removed_outlier_fraction": removed_outlier_fraction,
                "x_low_edge_support_count": x_low_edge_support_count,
                "x_high_edge_support_count": x_high_edge_support_count,
                "y_low_edge_support_count": y_low_edge_support_count,
                "y_high_edge_support_count": y_high_edge_support_count,
                "x_min_edge_support_fraction": x_min_edge_support_fraction,
                "y_min_edge_support_fraction": y_min_edge_support_fraction,
                "edge_support_band_m": edge_band_m,
                "robust_shrinkage_suspected": robust_shrinkage_suspected,
                "edge_sparsity_suspected": edge_sparsity_suspected,
                "pca_axis_underestimate_suspected": pca_axis_underestimate_suspected,
            }
            tool_result = ToolResult(
                tool_name="estimate_crown_width",
                status="success",
                value=predicted_crown_width_m,
                unit="m",
                confidence=None,
                extra=diagnostics,
                message="Direct geometry crown width estimated from fixed-axis robust XY spans.",
            )
        except DirectGeometryBackendError as exc:
            tool_result = self._build_failed_result(
                "estimate_crown_width",
                point_cloud,
                str(exc),
                diagnostics=diagnostics,
            )

        return CrownWidthInspection(
            tool_result=tool_result,
            projected_xy=projected_xy,
            unique_xy=unique_xy,
            filtered_xy=filtered_xy,
        )

    def estimate_crown_width(self, point_cloud: PointCloudInput) -> ToolResult:
        return self.inspect_crown_width(point_cloud).tool_result

    def estimate_tilt(self, point_cloud: PointCloudInput) -> ToolResult:
        return self._build_unsupported_result(
            "estimate_tilt",
            point_cloud,
            "Unsupported by direct geometry q1/q2/q3 baseline round.",
        )

    def assess_quality(self, point_cloud: PointCloudInput) -> ToolResult:
        return self._build_unsupported_result(
            "assess_quality",
            point_cloud,
            "Unsupported by direct geometry q1/q2/q3 baseline round.",
        )

    @classmethod
    def _load_config(cls, config_path: Path) -> DirectGeometryConfig:
        payload = load_yaml_mapping(config_path)
        dbh_payload = payload.get("dbh")
        height_payload = payload.get("height")
        crown_width_payload = payload.get("crown_width")
        if not isinstance(dbh_payload, dict) or not isinstance(height_payload, dict):
            raise ValueError("direct_geometry config must define dbh and height mappings.")
        if crown_width_payload is None:
            crown_width_payload = {}
        if not isinstance(crown_width_payload, dict):
            raise ValueError("direct_geometry config crown_width section must be a mapping.")

        trunk_payload = dbh_payload.get("trunk_filtering")
        if trunk_payload is None:
            trunk_payload = {}
        if not isinstance(trunk_payload, dict):
            raise ValueError("dbh.trunk_filtering must be a mapping when provided.")
        secondary_review_payload = trunk_payload.get("secondary_review")
        if secondary_review_payload is None:
            secondary_review_payload = {}
        if not isinstance(secondary_review_payload, dict):
            raise ValueError("dbh.trunk_filtering.secondary_review must be a mapping when provided.")
        suspicion_payload = trunk_payload.get("suspicion")
        if suspicion_payload is None:
            suspicion_payload = {}
        if not isinstance(suspicion_payload, dict):
            raise ValueError("dbh.trunk_filtering.suspicion must be a mapping when provided.")

        config = DirectGeometryConfig(
            ground_reference_percentile=float(payload["ground_reference_percentile"]),
            dbh=DBHConfig(
                breast_height_m=float(dbh_payload["breast_height_m"]),
                slice_thickness_m=float(dbh_payload["slice_thickness_m"]),
                min_slice_points=int(dbh_payload["min_slice_points"]),
                circle_fitting_method=str(dbh_payload["circle_fitting_method"]).lower(),
                radial_outlier_mad_threshold=float(dbh_payload["radial_outlier_mad_threshold"]),
                min_radius_m=float(dbh_payload["min_radius_m"]),
                max_radius_m=float(dbh_payload["max_radius_m"]),
                trunk_filtering=TrunkFilteringConfig(
                    enabled=bool(trunk_payload.get("enabled", False)),
                    xy_cluster_radius_m=float(trunk_payload.get("xy_cluster_radius_m", 0.025)),
                    min_cluster_points=int(trunk_payload.get("min_cluster_points", 20)),
                    min_angular_coverage=float(trunk_payload.get("min_angular_coverage", 0.7)),
                    cluster_selection_metric=str(
                        trunk_payload.get("cluster_selection_metric", "coverage_over_radius")
                    ).lower(),
                    radial_trim_mad_threshold=(
                        None
                        if trunk_payload.get("radial_trim_mad_threshold") is None
                        else float(trunk_payload["radial_trim_mad_threshold"])
                    ),
                    secondary_review_enabled=bool(secondary_review_payload.get("enabled", False)),
                    secondary_review_min_angular_coverage=float(
                        secondary_review_payload.get("min_angular_coverage", 0.55)
                    ),
                    secondary_review_max_fit_rmse_m=float(
                        secondary_review_payload.get("max_fit_rmse_m", 0.03)
                    ),
                    secondary_review_max_radius_ratio_vs_primary=float(
                        secondary_review_payload.get("max_radius_ratio_vs_primary", 0.5)
                    ),
                    secondary_review_min_score_ratio_vs_primary=float(
                        secondary_review_payload.get("min_score_ratio_vs_primary", 2.0)
                    ),
                    suspicion_large_point_count_threshold=int(
                        suspicion_payload.get("large_point_count_threshold", 250)
                    ),
                    suspicion_spread_to_diameter_ratio_threshold=float(
                        suspicion_payload.get("spread_to_diameter_ratio_threshold", 1.35)
                    ),
                    suspicion_large_candidate_radius_m=float(
                        suspicion_payload.get("large_candidate_radius_m", 0.1)
                    ),
                    suspicion_min_large_candidate_count=int(
                        suspicion_payload.get("min_large_candidate_count", 2)
                    ),
                ),
            ),
            height=HeightConfig(
                top_percentile=float(height_payload["top_percentile"]),
                min_point_count=int(height_payload["min_point_count"]),
                top_support_window_m=float(height_payload.get("top_support_window_m", 0.25)),
                top_outlier_gap_threshold_m=float(
                    height_payload.get("top_outlier_gap_threshold_m", 0.75)
                ),
                sparse_top_min_point_count=int(
                    height_payload.get("sparse_top_min_point_count", 12)
                ),
                sparse_top_min_fraction=float(height_payload.get("sparse_top_min_fraction", 0.002)),
                truncated_top_span_threshold_m=float(
                    height_payload.get("truncated_top_span_threshold_m", 0.05)
                ),
                truncated_top_plateau_tolerance_m=float(
                    height_payload.get("truncated_top_plateau_tolerance_m", 0.01)
                ),
                truncated_top_plateau_ratio_threshold=float(
                    height_payload.get("truncated_top_plateau_ratio_threshold", 0.6)
                ),
                ground_stability_band_m=float(
                    height_payload.get("ground_stability_band_m", 0.2)
                ),
                ground_stability_max_iqr_m=float(
                    height_payload.get("ground_stability_max_iqr_m", 0.08)
                ),
            ),
            crown_width=CrownWidthConfig(
                robust_low_percentile=float(crown_width_payload.get("robust_low_percentile", 2.0)),
                robust_high_percentile=float(crown_width_payload.get("robust_high_percentile", 98.0)),
                min_projected_points=int(crown_width_payload.get("min_projected_points", 50)),
                radial_outlier_mad_threshold=float(
                    crown_width_payload.get("radial_outlier_mad_threshold", 4.0)
                ),
                sparse_projection_min_unique_points=int(
                    crown_width_payload.get("sparse_projection_min_unique_points", 30)
                ),
                heavy_outlier_removal_fraction_threshold=float(
                    crown_width_payload.get("heavy_outlier_removal_fraction_threshold", 0.15)
                ),
                axis_pca_gap_threshold_m=float(
                    crown_width_payload.get("axis_pca_gap_threshold_m", 0.75)
                ),
                elongation_ratio_threshold=float(
                    crown_width_payload.get("elongation_ratio_threshold", 2.5)
                ),
                robust_shrinkage_ratio_threshold=float(
                    crown_width_payload.get("robust_shrinkage_ratio_threshold", 0.35)
                ),
                pca_axis_underestimate_threshold_m=float(
                    crown_width_payload.get("pca_axis_underestimate_threshold_m", 0.1)
                ),
                edge_support_band_m=float(
                    crown_width_payload.get("edge_support_band_m", 0.05)
                ),
                edge_support_min_point_count=int(
                    crown_width_payload.get("edge_support_min_point_count", 20)
                ),
                edge_support_min_fraction=float(
                    crown_width_payload.get("edge_support_min_fraction", 0.005)
                ),
            ),
        )
        cls._validate_config(config)
        return config

    @staticmethod
    def _validate_config(config: DirectGeometryConfig) -> None:
        if not 0.0 <= config.ground_reference_percentile <= 100.0:
            raise ValueError("ground_reference_percentile must be between 0 and 100.")
        if config.dbh.breast_height_m <= 0:
            raise ValueError("breast_height_m must be positive.")
        if config.dbh.slice_thickness_m <= 0:
            raise ValueError("slice_thickness_m must be positive.")
        if config.dbh.min_slice_points < 3:
            raise ValueError("min_slice_points must be at least 3.")
        if config.dbh.circle_fitting_method != "kasa":
            raise ValueError("Only the 'kasa' circle fitting method is currently supported.")
        if config.dbh.radial_outlier_mad_threshold <= 0:
            raise ValueError("radial_outlier_mad_threshold must be positive.")
        if config.dbh.min_radius_m <= 0 or config.dbh.max_radius_m <= config.dbh.min_radius_m:
            raise ValueError("DBH radius bounds must be positive and ordered.")
        trunk = config.dbh.trunk_filtering
        if trunk.xy_cluster_radius_m <= 0:
            raise ValueError("xy_cluster_radius_m must be positive.")
        if trunk.min_cluster_points < 3:
            raise ValueError("min_cluster_points must be at least 3.")
        if not 0.0 <= trunk.min_angular_coverage <= 1.0:
            raise ValueError("min_angular_coverage must be between 0 and 1.")
        if trunk.cluster_selection_metric != "coverage_over_radius":
            raise ValueError(
                "Only the 'coverage_over_radius' trunk cluster selection metric is supported."
            )
        if trunk.radial_trim_mad_threshold is not None and trunk.radial_trim_mad_threshold <= 0:
            raise ValueError("radial_trim_mad_threshold must be positive when provided.")
        if not 0.0 <= trunk.secondary_review_min_angular_coverage <= trunk.min_angular_coverage:
            raise ValueError(
                "secondary_review_min_angular_coverage must be between 0 and min_angular_coverage."
            )
        if trunk.secondary_review_max_fit_rmse_m <= 0:
            raise ValueError("secondary_review_max_fit_rmse_m must be positive.")
        if trunk.secondary_review_max_radius_ratio_vs_primary <= 0:
            raise ValueError("secondary_review_max_radius_ratio_vs_primary must be positive.")
        if trunk.secondary_review_min_score_ratio_vs_primary <= 0:
            raise ValueError("secondary_review_min_score_ratio_vs_primary must be positive.")
        if trunk.suspicion_large_point_count_threshold < trunk.min_cluster_points:
            raise ValueError(
                "suspicion_large_point_count_threshold must be at least min_cluster_points."
            )
        if trunk.suspicion_spread_to_diameter_ratio_threshold <= 0:
            raise ValueError("suspicion_spread_to_diameter_ratio_threshold must be positive.")
        if trunk.suspicion_large_candidate_radius_m <= 0:
            raise ValueError("suspicion_large_candidate_radius_m must be positive.")
        if trunk.suspicion_min_large_candidate_count < 2:
            raise ValueError("suspicion_min_large_candidate_count must be at least 2.")
        if not 0.0 <= config.height.top_percentile <= 100.0:
            raise ValueError("top_percentile must be between 0 and 100.")
        if config.height.min_point_count < 3:
            raise ValueError("min_point_count must be at least 3.")
        if config.height.top_support_window_m <= 0:
            raise ValueError("top_support_window_m must be positive.")
        if config.height.top_outlier_gap_threshold_m <= 0:
            raise ValueError("top_outlier_gap_threshold_m must be positive.")
        if config.height.sparse_top_min_point_count < 1:
            raise ValueError("sparse_top_min_point_count must be at least 1.")
        if not 0.0 <= config.height.sparse_top_min_fraction <= 1.0:
            raise ValueError("sparse_top_min_fraction must be between 0 and 1.")
        if config.height.truncated_top_span_threshold_m <= 0:
            raise ValueError("truncated_top_span_threshold_m must be positive.")
        if config.height.truncated_top_plateau_tolerance_m <= 0:
            raise ValueError("truncated_top_plateau_tolerance_m must be positive.")
        if not 0.0 <= config.height.truncated_top_plateau_ratio_threshold <= 1.0:
            raise ValueError("truncated_top_plateau_ratio_threshold must be between 0 and 1.")
        if config.height.ground_stability_band_m <= 0:
            raise ValueError("ground_stability_band_m must be positive.")
        if config.height.ground_stability_max_iqr_m <= 0:
            raise ValueError("ground_stability_max_iqr_m must be positive.")
        if not 0.0 <= config.crown_width.robust_low_percentile < config.crown_width.robust_high_percentile <= 100.0:
            raise ValueError("crown_width robust percentiles must be ordered within [0, 100].")
        if config.crown_width.min_projected_points < 3:
            raise ValueError("crown_width min_projected_points must be at least 3.")
        if config.crown_width.radial_outlier_mad_threshold <= 0:
            raise ValueError("crown_width radial_outlier_mad_threshold must be positive.")
        if config.crown_width.sparse_projection_min_unique_points < 1:
            raise ValueError("crown_width sparse_projection_min_unique_points must be at least 1.")
        if not 0.0 <= config.crown_width.heavy_outlier_removal_fraction_threshold <= 1.0:
            raise ValueError(
                "crown_width heavy_outlier_removal_fraction_threshold must be between 0 and 1."
            )
        if config.crown_width.axis_pca_gap_threshold_m <= 0:
            raise ValueError("crown_width axis_pca_gap_threshold_m must be positive.")
        if config.crown_width.elongation_ratio_threshold <= 1.0:
            raise ValueError("crown_width elongation_ratio_threshold must be greater than 1.")
        if not 0.0 <= config.crown_width.robust_shrinkage_ratio_threshold <= 1.0:
            raise ValueError("crown_width robust_shrinkage_ratio_threshold must be between 0 and 1.")
        if config.crown_width.pca_axis_underestimate_threshold_m <= 0:
            raise ValueError("crown_width pca_axis_underestimate_threshold_m must be positive.")
        if config.crown_width.edge_support_band_m <= 0:
            raise ValueError("crown_width edge_support_band_m must be positive.")
        if config.crown_width.edge_support_min_point_count < 1:
            raise ValueError("crown_width edge_support_min_point_count must be at least 1.")
        if not 0.0 <= config.crown_width.edge_support_min_fraction <= 1.0:
            raise ValueError("crown_width edge_support_min_fraction must be between 0 and 1.")

    @staticmethod
    def _load_point_cloud_matrix(point_cloud: PointCloudInput) -> np.ndarray:
        point_cloud_path = Path(point_cloud.path)
        if point_cloud.format.lower() not in _SUPPORTED_POINT_CLOUD_FORMATS:
            raise DirectGeometryBackendError(
                f"Direct geometry backend only supports LAS/LAZ, got format={point_cloud.format!r}."
            )
        if not point_cloud_path.exists():
            raise DirectGeometryBackendError(f"Point cloud file was not found: {point_cloud_path}")

        try:
            import laspy
        except ModuleNotFoundError as exc:
            raise DirectGeometryBackendError(
                "laspy is required for DirectGeometryBackend point cloud loading."
            ) from exc

        las = laspy.read(point_cloud_path)
        points = np.column_stack((las.x, las.y, las.z)).astype(np.float64, copy=False)
        finite_mask = np.all(np.isfinite(points), axis=1)
        finite_points = points[finite_mask]
        if finite_points.shape[0] == 0:
            raise DirectGeometryBackendError("Point cloud does not contain any finite XYZ points.")
        return finite_points

    def _estimate_ground_reference(self, z_values: np.ndarray) -> float:
        return float(np.percentile(z_values, self._config.ground_reference_percentile))

    def _slice_breast_height(self, points: np.ndarray, target_z: float) -> np.ndarray:
        slice_mask = np.abs(points[:, 2] - target_z) <= (self._config.dbh.slice_thickness_m / 2.0)
        return points[slice_mask, :2]

    @staticmethod
    def _robust_span(
        values: np.ndarray,
        *,
        low_percentile: float,
        high_percentile: float,
    ) -> float:
        low_value = float(np.percentile(values, low_percentile))
        high_value = float(np.percentile(values, high_percentile))
        return high_value - low_value

    @classmethod
    def _pca_robust_spans(
        cls,
        xy_points: np.ndarray,
        *,
        low_percentile: float,
        high_percentile: float,
    ) -> tuple[float, float]:
        centered = xy_points - np.mean(xy_points, axis=0, keepdims=True)
        if np.linalg.matrix_rank(centered) < 2:
            raise DirectGeometryBackendError(
                "Projected crown footprint is degenerate after XY centering."
            )
        covariance = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        sorted_vectors = eigenvectors[:, order]
        projected = centered @ sorted_vectors
        major_span = cls._robust_span(
            projected[:, 0],
            low_percentile=low_percentile,
            high_percentile=high_percentile,
        )
        minor_span = cls._robust_span(
            projected[:, 1],
            low_percentile=low_percentile,
            high_percentile=high_percentile,
        )
        return float(major_span), float(minor_span)

    def _prepare_dbh_slice(
        self,
        *,
        deduplicated_xy: np.ndarray,
        diagnostics: dict[str, object],
    ) -> tuple[np.ndarray, dict[str, object]]:
        filtered_xy, outlier_removed_count = self._filter_radial_outliers(
            deduplicated_xy,
            threshold=self._config.dbh.radial_outlier_mad_threshold,
        )
        diagnostics = {
            **diagnostics,
            "post_outlier_filter_point_count": int(filtered_xy.shape[0]),
            "removed_outlier_count": outlier_removed_count,
            "cluster_candidate_count": 0,
            "secondary_review_candidate_count": 0,
            "secondary_review_selected": False,
            "secondary_review_reason": None,
            "selected_cluster_point_count": None,
            "selected_cluster_score": None,
            "selected_cluster_angular_coverage": None,
            "selected_cluster_fit_rmse_m": None,
            "selected_cluster_radius_m": None,
            "selected_cluster_center_x_m": None,
            "selected_cluster_center_y_m": None,
            "selected_cluster_spread_x_m": None,
            "selected_cluster_spread_y_m": None,
            "selected_component_index": None,
            "primary_component_index": None,
            "merged_cluster_suspected": False,
            "merged_cluster_reasons": [],
            "radial_trim_removed_count": 0,
        }

        if filtered_xy.shape[0] < self._config.dbh.min_slice_points:
            raise DirectGeometryBackendError(
                "Breast-height slice does not contain enough points for circle fitting."
            )

        final_xy = filtered_xy
        trunk = self._config.dbh.trunk_filtering
        if trunk.enabled:
            candidate = self._select_trunk_cluster(filtered_xy)
            if candidate is None:
                raise DirectGeometryBackendError(
                    "No trunk-like XY cluster was found in the breast-height slice."
                )
            final_xy = candidate.points_xy
            diagnostics = {
                **diagnostics,
                "cluster_candidate_count": self._last_trunk_candidate_count,
                "secondary_review_candidate_count": self._last_secondary_review_candidate_count,
                "secondary_review_selected": self._last_secondary_review_selected,
                "secondary_review_reason": self._last_secondary_review_reason,
                "selected_cluster_point_count": int(candidate.points_xy.shape[0]),
                "selected_cluster_score": candidate.selection_score,
                "selected_cluster_angular_coverage": candidate.angular_coverage,
                "selected_cluster_fit_rmse_m": candidate.fit_rmse_m,
                "selected_cluster_radius_m": candidate.radius_m,
                "selected_cluster_center_x_m": float(candidate.center_xy[0]),
                "selected_cluster_center_y_m": float(candidate.center_xy[1]),
                "selected_cluster_spread_x_m": candidate.spread_x_m,
                "selected_cluster_spread_y_m": candidate.spread_y_m,
                "selected_component_index": self._last_selected_component_index,
                "primary_component_index": self._last_primary_component_index,
                "merged_cluster_suspected": self._last_merged_cluster_suspected,
                "merged_cluster_reasons": list(self._last_merged_cluster_reasons),
            }
            final_xy, radial_trim_removed_count = self._radial_trim(
                final_xy,
                center_xy=candidate.center_xy,
                radius_m=candidate.radius_m,
                threshold=trunk.radial_trim_mad_threshold,
            )
            diagnostics["radial_trim_removed_count"] = radial_trim_removed_count

        if final_xy.shape[0] < self._config.dbh.min_slice_points:
            raise DirectGeometryBackendError(
                "Breast-height slice does not contain enough filtered points for circle fitting."
            )

        centered_points = final_xy - np.mean(final_xy, axis=0, keepdims=True)
        if np.linalg.matrix_rank(centered_points) < 2:
            raise DirectGeometryBackendError("Breast-height slice is degenerate after XY projection.")

        return final_xy, diagnostics

    def _select_trunk_cluster(self, xy_points: np.ndarray) -> TrunkClusterCandidate | None:
        self._reset_trunk_selection_state()
        tree = self._build_xy_tree(xy_points)
        trunk = self._config.dbh.trunk_filtering
        primary_candidates: list[TrunkClusterCandidate] = []
        secondary_candidates: list[TrunkClusterCandidate] = []
        best_candidate: TrunkClusterCandidate | None = None
        for component_index, indices in enumerate(
            self._connected_components(
                tree=tree,
                xy_points=xy_points,
                radius_m=trunk.xy_cluster_radius_m,
            )
        ):
            cluster_points = xy_points[indices]
            if cluster_points.shape[0] < trunk.min_cluster_points:
                continue
            spread_x_m = float(cluster_points[:, 0].max() - cluster_points[:, 0].min())
            spread_y_m = float(cluster_points[:, 1].max() - cluster_points[:, 1].min())
            centered_points = cluster_points - np.mean(cluster_points, axis=0, keepdims=True)
            if np.linalg.matrix_rank(centered_points) < 2:
                continue
            try:
                center_xy, radius_m, fit_rmse_m = self._fit_circle(
                    cluster_points,
                    method=self._config.dbh.circle_fitting_method,
                )
            except DirectGeometryBackendError:
                continue
            if radius_m < self._config.dbh.min_radius_m or radius_m > self._config.dbh.max_radius_m:
                continue
            angular_coverage = self._angular_coverage(cluster_points, center_xy)
            selection_score = self._cluster_selection_score(
                radius_m=radius_m,
                fit_rmse_m=fit_rmse_m,
                angular_coverage=angular_coverage,
            )
            candidate = TrunkClusterCandidate(
                component_index=component_index,
                points_xy=cluster_points,
                center_xy=center_xy,
                radius_m=radius_m,
                fit_rmse_m=fit_rmse_m,
                angular_coverage=angular_coverage,
                selection_score=selection_score,
                point_count=int(cluster_points.shape[0]),
                spread_x_m=spread_x_m,
                spread_y_m=spread_y_m,
            )
            if angular_coverage >= trunk.min_angular_coverage:
                primary_candidates.append(candidate)
                if best_candidate is None or candidate.selection_score > best_candidate.selection_score:
                    best_candidate = candidate
                continue
            if not trunk.secondary_review_enabled:
                continue
            if angular_coverage < trunk.secondary_review_min_angular_coverage:
                continue
            if fit_rmse_m > trunk.secondary_review_max_fit_rmse_m:
                continue
            secondary_candidates.append(candidate)

        self._last_trunk_candidate_count = len(primary_candidates)
        self._last_primary_component_index = (
            None if best_candidate is None else best_candidate.component_index
        )
        self._last_secondary_review_candidate_count = 0
        self._last_secondary_review_selected = False
        self._last_secondary_review_reason = None

        self._update_merged_cluster_suspicion(
            primary_candidates=primary_candidates,
            primary_candidate=best_candidate,
        )
        if best_candidate is None:
            self._last_selected_component_index = None
            return None

        eligible_secondary_candidates: list[TrunkClusterCandidate] = []
        for secondary_candidate in secondary_candidates:
            if (
                secondary_candidate.radius_m
                > best_candidate.radius_m * trunk.secondary_review_max_radius_ratio_vs_primary
            ):
                continue
            if (
                secondary_candidate.selection_score
                < best_candidate.selection_score * trunk.secondary_review_min_score_ratio_vs_primary
            ):
                continue
            eligible_secondary_candidates.append(secondary_candidate)

        self._last_secondary_review_candidate_count = len(eligible_secondary_candidates)
        if eligible_secondary_candidates:
            best_secondary_candidate = max(
                eligible_secondary_candidates,
                key=lambda candidate: candidate.selection_score,
            )
            best_candidate = best_secondary_candidate
            self._last_secondary_review_selected = True
            self._last_secondary_review_reason = (
                "Selected a smaller, low-RMSE secondary-review cluster with slightly lower "
                "angular coverage than the primary candidate."
            )

        self._last_selected_component_index = best_candidate.component_index
        return best_candidate

    def _update_merged_cluster_suspicion(
        self,
        *,
        primary_candidates: list[TrunkClusterCandidate],
        primary_candidate: TrunkClusterCandidate | None,
    ) -> None:
        self._last_merged_cluster_suspected = False
        self._last_merged_cluster_reasons = []
        if primary_candidate is None:
            return

        trunk = self._config.dbh.trunk_filtering
        reasons: list[str] = []
        if primary_candidate.point_count >= trunk.suspicion_large_point_count_threshold:
            reasons.append("selected_primary_candidate_point_count_unusually_large")

        fitted_diameter_m = max(primary_candidate.radius_m * 2.0, 1e-9)
        spread_ratio = max(primary_candidate.spread_x_m, primary_candidate.spread_y_m) / fitted_diameter_m
        if spread_ratio >= trunk.suspicion_spread_to_diameter_ratio_threshold:
            reasons.append("selected_primary_candidate_spread_large_relative_to_fitted_diameter")

        large_primary_candidates = [
            candidate
            for candidate in primary_candidates
            if candidate.radius_m >= trunk.suspicion_large_candidate_radius_m
        ]
        if len(large_primary_candidates) >= trunk.suspicion_min_large_candidate_count:
            reasons.append("multiple_plausible_primary_candidates_all_large")

        self._last_merged_cluster_suspected = bool(reasons)
        self._last_merged_cluster_reasons = reasons

    def _reset_trunk_selection_state(self) -> None:
        self._last_trunk_candidate_count = 0
        self._last_secondary_review_candidate_count = 0
        self._last_secondary_review_selected = False
        self._last_secondary_review_reason: str | None = None
        self._last_selected_component_index: int | None = None
        self._last_primary_component_index: int | None = None
        self._last_merged_cluster_suspected = False
        self._last_merged_cluster_reasons: list[str] = []

    def _cluster_selection_score(
        self,
        *,
        radius_m: float,
        fit_rmse_m: float,
        angular_coverage: float,
    ) -> float:
        return angular_coverage / ((1.0 + fit_rmse_m * 10.0) * radius_m)

    @staticmethod
    def _build_xy_tree(xy_points: np.ndarray):
        try:
            from scipy.spatial import cKDTree
        except ModuleNotFoundError as exc:
            raise DirectGeometryBackendError(
                "scipy is required when trunk-only filtering is enabled."
            ) from exc
        return cKDTree(xy_points)

    @staticmethod
    def _connected_components(tree, xy_points: np.ndarray, radius_m: float) -> list[np.ndarray]:
        visited = np.zeros(xy_points.shape[0], dtype=bool)
        components: list[np.ndarray] = []
        for start_index in range(xy_points.shape[0]):
            if visited[start_index]:
                continue
            stack = [start_index]
            visited[start_index] = True
            component_indices: list[int] = []
            while stack:
                current_index = stack.pop()
                component_indices.append(current_index)
                for neighbor_index in tree.query_ball_point(xy_points[current_index], radius_m):
                    if not visited[neighbor_index]:
                        visited[neighbor_index] = True
                        stack.append(int(neighbor_index))
            components.append(np.asarray(component_indices, dtype=np.int64))
        return components

    @staticmethod
    def _angular_coverage(xy_points: np.ndarray, center_xy: np.ndarray, bins: int = 24) -> float:
        angles = np.arctan2(xy_points[:, 1] - center_xy[1], xy_points[:, 0] - center_xy[0])
        normalized = (angles + np.pi) / (2.0 * np.pi)
        counts, _ = np.histogram(normalized, bins=bins, range=(0.0, 1.0))
        return float(np.count_nonzero(counts) / bins)

    @staticmethod
    def _radial_trim(
        xy_points: np.ndarray,
        *,
        center_xy: np.ndarray,
        radius_m: float,
        threshold: float | None,
    ) -> tuple[np.ndarray, int]:
        if threshold is None:
            return xy_points, 0
        radial_distances = np.linalg.norm(xy_points - center_xy, axis=1)
        residuals = np.abs(radial_distances - radius_m)
        residual_median = float(np.median(residuals))
        residual_mad = float(np.median(np.abs(residuals - residual_median)))
        if residual_mad <= 1e-9:
            return xy_points, 0
        keep_mask = residuals <= (residual_median + threshold * residual_mad)
        trimmed = xy_points[keep_mask]
        return trimmed, int(xy_points.shape[0] - trimmed.shape[0])

    @staticmethod
    def _filter_radial_outliers(
        xy_points: np.ndarray,
        *,
        threshold: float,
    ) -> tuple[np.ndarray, int]:
        if xy_points.shape[0] == 0:
            return xy_points, 0

        median_xy = np.median(xy_points, axis=0)
        radial_distances = np.linalg.norm(xy_points - median_xy, axis=1)
        radial_median = float(np.median(radial_distances))
        radial_mad = float(np.median(np.abs(radial_distances - radial_median)))
        if radial_mad <= 1e-9:
            return xy_points, 0

        keep_mask = np.abs(radial_distances - radial_median) <= (threshold * radial_mad)
        filtered_points = xy_points[keep_mask]
        removed_count = int(xy_points.shape[0] - filtered_points.shape[0])
        return filtered_points, removed_count

    @staticmethod
    def _fit_circle(
        xy_points: np.ndarray,
        *,
        method: str,
    ) -> tuple[np.ndarray, float, float]:
        if method != "kasa":
            raise DirectGeometryBackendError(
                f"Unsupported circle fitting method for direct geometry: {method}"
            )
        if xy_points.shape[0] < 3:
            raise DirectGeometryBackendError("Circle fitting requires at least 3 XY points.")

        xy_mean = np.mean(xy_points, axis=0)
        centered = xy_points - xy_mean
        design = np.column_stack(
            (
                2.0 * centered[:, 0],
                2.0 * centered[:, 1],
                np.ones(centered.shape[0]),
            )
        )
        rhs = centered[:, 0] ** 2 + centered[:, 1] ** 2
        parameters, _, rank, _ = np.linalg.lstsq(design, rhs, rcond=None)
        if rank < 3:
            raise DirectGeometryBackendError("Circle fitting design matrix is rank deficient.")

        center_local = parameters[:2]
        radius_squared = float(parameters[2] + np.dot(center_local, center_local))
        if not np.isfinite(radius_squared) or radius_squared <= 0:
            raise DirectGeometryBackendError("Circle fitting produced a non-positive radius.")

        center_xy = center_local + xy_mean
        radius_m = float(np.sqrt(radius_squared))
        radial_distances = np.linalg.norm(xy_points - center_xy, axis=1)
        fit_rmse_m = float(np.sqrt(np.mean((radial_distances - radius_m) ** 2)))
        return center_xy, radius_m, fit_rmse_m

    def _build_failed_result(
        self,
        tool_name: str,
        point_cloud: PointCloudInput,
        message: str,
        diagnostics: dict[str, object] | None = None,
    ) -> ToolResult:
        extra = {
            "backend": "direct_geometry",
            "input_format": point_cloud.format,
            "config_path": str(self._config_path),
        }
        if diagnostics:
            extra.update(diagnostics)
        return ToolResult(
            tool_name=tool_name,
            status="failed",
            value=None,
            unit=None,
            confidence=None,
            extra=extra,
            message=message,
        )

    def _build_unsupported_result(
        self,
        tool_name: str,
        point_cloud: PointCloudInput,
        message: str,
    ) -> ToolResult:
        return ToolResult(
            tool_name=tool_name,
            status="failed",
            value=None,
            unit=None,
            confidence=None,
            extra={
                "backend": "direct_geometry",
                "input_format": point_cloud.format,
                "unsupported": True,
                "config_path": str(self._config_path),
            },
            message=message,
        )
