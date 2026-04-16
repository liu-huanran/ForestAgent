"""Sanity checks for the direct geometry DBH and height algorithms."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import laspy
import numpy as np

from forestagent.backends.direct_geometry_backend import DirectGeometryBackend
from forestagent.schemas import PointCloudInput


class DirectGeometrySanityTests(unittest.TestCase):
    def test_estimate_dbh_recovers_synthetic_cylinder_radius(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            point_cloud_path = Path(temp_dir) / "dbh_sanity.las"
            expected_radius_m = 0.18
            self._write_tree_point_cloud(
                point_cloud_path,
                radius_m=expected_radius_m,
                height_m=9.0,
                include_noise=False,
            )
            backend = DirectGeometryBackend()

            result = backend.estimate_dbh(PointCloudInput(path=str(point_cloud_path), format="las"))

            self.assertEqual(result.status, "success")
            self.assertAlmostEqual(result.value, expected_radius_m * 200.0, delta=1.0)

    def test_estimate_height_recovers_synthetic_tree_height(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            point_cloud_path = Path(temp_dir) / "height_sanity.las"
            expected_height_m = 11.5
            self._write_tree_point_cloud(
                point_cloud_path,
                radius_m=0.12,
                height_m=expected_height_m,
                include_noise=False,
            )
            backend = DirectGeometryBackend()

            result = backend.estimate_height(
                PointCloudInput(path=str(point_cloud_path), format="las")
            )

            self.assertEqual(result.status, "success")
            self.assertAlmostEqual(result.value, expected_height_m, delta=0.05)
            self.assertFalse(result.extra["ground_reference_unstable_suspected"])

    def test_estimate_height_flags_top_outlier_suspicion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            point_cloud_path = Path(temp_dir) / "height_outlier.las"
            base_points = self._build_tree_points(radius_m=0.12, height_m=10.0)
            outliers = np.asarray(
                [
                    [0.0, 0.0, 12.4],
                    [0.05, 0.02, 12.7],
                    [-0.04, -0.03, 12.9],
                ],
                dtype=np.float64,
            )
            self._write_points(path=point_cloud_path, points=np.vstack((base_points, outliers)))
            backend = DirectGeometryBackend()

            result = backend.estimate_height(
                PointCloudInput(path=str(point_cloud_path), format="las")
            )

            self.assertEqual(result.status, "success")
            self.assertTrue(result.extra["top_outlier_suspected"])
            self.assertGreater(result.extra["max_z_gap_m"], 0.75)

    def test_estimate_crown_width_recovers_synthetic_elliptical_crown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            point_cloud_path = Path(temp_dir) / "crown_width_sanity.las"
            crown_radius_x_m = 2.2
            crown_radius_y_m = 1.4
            expected_crown_width_m = ((2.0 * crown_radius_x_m) + (2.0 * crown_radius_y_m)) / 2.0
            self._write_crown_tree_point_cloud(
                point_cloud_path,
                trunk_radius_m=0.08,
                trunk_height_m=6.5,
                crown_radius_x_m=crown_radius_x_m,
                crown_radius_y_m=crown_radius_y_m,
            )
            backend = DirectGeometryBackend()

            result = backend.estimate_crown_width(
                PointCloudInput(path=str(point_cloud_path), format="las")
            )

            self.assertEqual(result.status, "success")
            self.assertAlmostEqual(result.value, expected_crown_width_m, delta=0.2)
            self.assertFalse(result.extra["sparse_projection_suspected"])
            self.assertFalse(result.extra["axis_pca_gap_suspected"])

    def test_estimate_crown_width_handles_sparse_outliers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            point_cloud_path = Path(temp_dir) / "crown_width_outlier.las"
            config_path = Path(temp_dir) / "crown_config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "ground_reference_percentile: 1.0",
                        "dbh:",
                        "  breast_height_m: 1.3",
                        "  slice_thickness_m: 0.12",
                        "  min_slice_points: 24",
                        '  circle_fitting_method: "kasa"',
                        "  radial_outlier_mad_threshold: 3.5",
                        "  min_radius_m: 0.01",
                        "  max_radius_m: 1.0",
                        "  trunk_filtering:",
                        "    enabled: false",
                        "    xy_cluster_radius_m: 0.025",
                        "    min_cluster_points: 20",
                        "    min_angular_coverage: 0.7",
                        '    cluster_selection_metric: "coverage_over_radius"',
                        "    radial_trim_mad_threshold: null",
                        "height:",
                        "  top_percentile: 99.5",
                        "  min_point_count: 50",
                        "crown_width:",
                        "  robust_low_percentile: 2.0",
                        "  robust_high_percentile: 98.0",
                        "  min_projected_points: 50",
                        "  radial_outlier_mad_threshold: 4.0",
                        "  sparse_projection_min_unique_points: 30",
                        "  heavy_outlier_removal_fraction_threshold: 0.005",
                        "  axis_pca_gap_threshold_m: 0.75",
                        "  elongation_ratio_threshold: 2.5",
                    ]
                ),
                encoding="utf-8",
            )
            self._write_crown_tree_point_cloud(
                point_cloud_path,
                trunk_radius_m=0.08,
                trunk_height_m=6.5,
                crown_radius_x_m=1.8,
                crown_radius_y_m=1.2,
                extra_points=np.asarray(
                    [
                        [8.0, 8.0, 5.0],
                        [-8.0, 7.5, 5.2],
                        [7.5, -8.0, 5.4],
                        [-7.5, -7.0, 5.6],
                    ],
                    dtype=np.float64,
                ),
            )
            backend = DirectGeometryBackend(config_path=config_path)

            result = backend.estimate_crown_width(
                PointCloudInput(path=str(point_cloud_path), format="las")
            )

            self.assertEqual(result.status, "success")
            self.assertTrue(result.extra["heavy_outlier_removal_suspected"])
            self.assertGreater(result.extra["removed_outlier_count"], 0)

    def test_estimate_crown_width_flags_shrinkage_and_edge_sparsity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            point_cloud_path = Path(temp_dir) / "crown_width_shrinkage.las"
            config_path = Path(temp_dir) / "crown_diag.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "ground_reference_percentile: 1.0",
                        "dbh:",
                        "  breast_height_m: 1.3",
                        "  slice_thickness_m: 0.12",
                        "  min_slice_points: 24",
                        '  circle_fitting_method: "kasa"',
                        "  radial_outlier_mad_threshold: 3.5",
                        "  min_radius_m: 0.01",
                        "  max_radius_m: 1.0",
                        "  trunk_filtering:",
                        "    enabled: false",
                        "    xy_cluster_radius_m: 0.025",
                        "    min_cluster_points: 20",
                        "    min_angular_coverage: 0.7",
                        '    cluster_selection_metric: "coverage_over_radius"',
                        "    radial_trim_mad_threshold: null",
                        "height:",
                        "  top_percentile: 99.5",
                        "  min_point_count: 50",
                        "crown_width:",
                        "  robust_low_percentile: 2.0",
                        "  robust_high_percentile: 98.0",
                        "  min_projected_points: 50",
                        "  radial_outlier_mad_threshold: 10.0",
                        "  sparse_projection_min_unique_points: 30",
                        "  heavy_outlier_removal_fraction_threshold: 0.15",
                        "  axis_pca_gap_threshold_m: 0.75",
                        "  elongation_ratio_threshold: 2.5",
                        "  robust_shrinkage_ratio_threshold: 0.2",
                        "  pca_axis_underestimate_threshold_m: 0.1",
                        "  edge_support_band_m: 0.05",
                        "  edge_support_min_point_count: 10",
                        "  edge_support_min_fraction: 0.01",
                    ]
                ),
                encoding="utf-8",
            )
            self._write_partial_crown_tree_point_cloud(
                point_cloud_path,
                core_radius_x_m=1.0,
                core_radius_y_m=1.0,
                sparse_edge_radius_x_m=1.9,
                sparse_edge_radius_y_m=1.6,
            )
            backend = DirectGeometryBackend(config_path=config_path)

            result = backend.estimate_crown_width(
                PointCloudInput(path=str(point_cloud_path), format="las")
            )

            self.assertEqual(result.status, "success")
            self.assertTrue(result.extra["robust_shrinkage_suspected"])
            self.assertTrue(result.extra["edge_sparsity_suspected"])
            self.assertGreater(result.extra["raw_vs_robust_gap_ratio"], 0.2)

    def test_estimate_crown_width_wider_robust_percentiles_reduce_partial_crown_underestimate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            point_cloud_path = Path(temp_dir) / "crown_width_percentile_tuning.las"
            baseline_config_path = Path(temp_dir) / "baseline_crown.yaml"
            tuned_config_path = Path(temp_dir) / "tuned_crown.yaml"
            expected_crown_width_m = ((2.0 * 1.9) + (2.0 * 1.6)) / 2.0

            self._write_partial_crown_tree_point_cloud(
                point_cloud_path,
                core_radius_x_m=1.0,
                core_radius_y_m=1.0,
                sparse_edge_radius_x_m=1.9,
                sparse_edge_radius_y_m=1.6,
            )
            baseline_config_path.write_text(
                "\n".join(
                    [
                        "ground_reference_percentile: 1.0",
                        "dbh:",
                        "  breast_height_m: 1.3",
                        "  slice_thickness_m: 0.12",
                        "  min_slice_points: 24",
                        '  circle_fitting_method: "kasa"',
                        "  radial_outlier_mad_threshold: 3.5",
                        "  min_radius_m: 0.01",
                        "  max_radius_m: 1.0",
                        "  trunk_filtering:",
                        "    enabled: false",
                        "    xy_cluster_radius_m: 0.025",
                        "    min_cluster_points: 20",
                        "    min_angular_coverage: 0.7",
                        '    cluster_selection_metric: "coverage_over_radius"',
                        "    radial_trim_mad_threshold: null",
                        "height:",
                        "  top_percentile: 99.5",
                        "  min_point_count: 50",
                        "crown_width:",
                        "  robust_low_percentile: 2.0",
                        "  robust_high_percentile: 98.0",
                        "  min_projected_points: 50",
                        "  radial_outlier_mad_threshold: 10.0",
                        "  sparse_projection_min_unique_points: 30",
                        "  heavy_outlier_removal_fraction_threshold: 0.15",
                        "  axis_pca_gap_threshold_m: 0.75",
                        "  elongation_ratio_threshold: 2.5",
                        "  robust_shrinkage_ratio_threshold: 0.35",
                        "  pca_axis_underestimate_threshold_m: 0.1",
                        "  edge_support_band_m: 0.05",
                        "  edge_support_min_point_count: 20",
                        "  edge_support_min_fraction: 0.005",
                    ]
                ),
                encoding="utf-8",
            )
            tuned_config_path.write_text(
                baseline_config_path.read_text(encoding="utf-8")
                .replace("  robust_low_percentile: 2.0", "  robust_low_percentile: 0.5")
                .replace("  robust_high_percentile: 98.0", "  robust_high_percentile: 99.5"),
                encoding="utf-8",
            )

            point_cloud = PointCloudInput(path=str(point_cloud_path), format="las")
            baseline_backend = DirectGeometryBackend(config_path=baseline_config_path)
            tuned_backend = DirectGeometryBackend(config_path=tuned_config_path)

            baseline_result = baseline_backend.estimate_crown_width(point_cloud)
            tuned_result = tuned_backend.estimate_crown_width(point_cloud)

            self.assertEqual(baseline_result.status, "success")
            self.assertEqual(tuned_result.status, "success")
            self.assertGreater(float(tuned_result.value), float(baseline_result.value))
            self.assertLess(
                abs(float(tuned_result.value) - expected_crown_width_m),
                abs(float(baseline_result.value) - expected_crown_width_m),
            )

    def test_estimate_dbh_fails_for_degenerate_slice(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            point_cloud_path = Path(temp_dir) / "degenerate_slice.las"
            z_levels = np.linspace(0.0, 5.0, 120)
            x_values = np.linspace(-0.2, 0.2, 120)
            points = np.column_stack((x_values, np.zeros_like(x_values), z_levels))
            self._write_points(path=point_cloud_path, points=points)
            backend = DirectGeometryBackend()

            result = backend.estimate_dbh(PointCloudInput(path=str(point_cloud_path), format="las"))

            self.assertEqual(result.status, "failed")
            self.assertIn("slice", result.message.lower())

    def test_trunk_only_filtering_reduces_contaminated_slice_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            point_cloud_path = Path(temp_dir) / "contaminated_slice.las"
            baseline_config_path = Path(temp_dir) / "baseline.yaml"
            improved_config_path = Path(temp_dir) / "improved.yaml"
            expected_dbh_cm = 12.0
            self._write_contaminated_tree_point_cloud(
                point_cloud_path,
                trunk_radius_m=0.06,
                branch_radius_m=0.18,
                height_m=9.0,
            )
            baseline_config_path.write_text(
                "\n".join(
                    [
                        "ground_reference_percentile: 1.0",
                        "dbh:",
                        "  breast_height_m: 1.3",
                        "  slice_thickness_m: 0.12",
                        "  min_slice_points: 24",
                        '  circle_fitting_method: "kasa"',
                        "  radial_outlier_mad_threshold: 3.5",
                        "  min_radius_m: 0.01",
                        "  max_radius_m: 1.0",
                        "  trunk_filtering:",
                        "    enabled: false",
                        "    xy_cluster_radius_m: 0.025",
                        "    min_cluster_points: 20",
                        "    min_angular_coverage: 0.7",
                        '    cluster_selection_metric: "coverage_over_radius"',
                        "    radial_trim_mad_threshold: null",
                        "height:",
                        "  top_percentile: 99.5",
                        "  min_point_count: 50",
                    ]
                ),
                encoding="utf-8",
            )
            improved_config_path.write_text(
                "\n".join(
                    [
                        "ground_reference_percentile: 1.0",
                        "dbh:",
                        "  breast_height_m: 1.3",
                        "  slice_thickness_m: 0.08",
                        "  min_slice_points: 24",
                        '  circle_fitting_method: "kasa"',
                        "  radial_outlier_mad_threshold: 3.0",
                        "  min_radius_m: 0.01",
                        "  max_radius_m: 1.0",
                        "  trunk_filtering:",
                        "    enabled: true",
                        "    xy_cluster_radius_m: 0.025",
                        "    min_cluster_points: 20",
                        "    min_angular_coverage: 0.7",
                        '    cluster_selection_metric: "coverage_over_radius"',
                        "    radial_trim_mad_threshold: null",
                        "height:",
                        "  top_percentile: 99.5",
                        "  min_point_count: 50",
                    ]
                ),
                encoding="utf-8",
            )
            point_cloud = PointCloudInput(path=str(point_cloud_path), format="las")
            baseline_backend = DirectGeometryBackend(config_path=baseline_config_path)
            improved_backend = DirectGeometryBackend(config_path=improved_config_path)

            baseline_result = baseline_backend.estimate_dbh(point_cloud)
            improved_result = improved_backend.estimate_dbh(point_cloud)

            self.assertEqual(baseline_result.status, "success")
            self.assertEqual(improved_result.status, "success")
            self.assertGreater(
                abs(float(baseline_result.value) - expected_dbh_cm),
                10.0,
            )
            self.assertLess(
                abs(float(improved_result.value) - expected_dbh_cm),
                1.5,
            )

    def test_secondary_review_recovers_small_incomplete_trunk_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            point_cloud_path = Path(temp_dir) / "secondary_review_slice.las"
            improved_config_path = Path(temp_dir) / "improved.yaml"
            reviewed_config_path = Path(temp_dir) / "reviewed.yaml"
            expected_dbh_cm = 8.0
            self._write_secondary_review_tree_point_cloud(
                point_cloud_path,
                trunk_radius_m=0.04,
                branch_radius_m=0.18,
                height_m=8.0,
            )
            improved_config_path.write_text(
                "\n".join(
                    [
                        "ground_reference_percentile: 0.0",
                        "dbh:",
                        "  breast_height_m: 1.3",
                        "  slice_thickness_m: 0.08",
                        "  min_slice_points: 24",
                        '  circle_fitting_method: "kasa"',
                        "  radial_outlier_mad_threshold: 100.0",
                        "  min_radius_m: 0.01",
                        "  max_radius_m: 1.0",
                        "  trunk_filtering:",
                        "    enabled: true",
                        "    xy_cluster_radius_m: 0.025",
                        "    min_cluster_points: 20",
                        "    min_angular_coverage: 0.7",
                        '    cluster_selection_metric: "coverage_over_radius"',
                        "    radial_trim_mad_threshold: null",
                        "height:",
                        "  top_percentile: 99.5",
                        "  min_point_count: 50",
                    ]
                ),
                encoding="utf-8",
            )
            reviewed_config_path.write_text(
                "\n".join(
                    [
                        "ground_reference_percentile: 0.0",
                        "dbh:",
                        "  breast_height_m: 1.3",
                        "  slice_thickness_m: 0.08",
                        "  min_slice_points: 24",
                        '  circle_fitting_method: "kasa"',
                        "  radial_outlier_mad_threshold: 100.0",
                        "  min_radius_m: 0.01",
                        "  max_radius_m: 1.0",
                        "  trunk_filtering:",
                        "    enabled: true",
                        "    xy_cluster_radius_m: 0.025",
                        "    min_cluster_points: 20",
                        "    min_angular_coverage: 0.7",
                        '    cluster_selection_metric: "coverage_over_radius"',
                        "    radial_trim_mad_threshold: null",
                        "    secondary_review:",
                        "      enabled: true",
                        "      min_angular_coverage: 0.55",
                        "      max_fit_rmse_m: 0.03",
                        "      max_radius_ratio_vs_primary: 0.4",
                        "      min_score_ratio_vs_primary: 2.0",
                        "    suspicion:",
                        "      large_point_count_threshold: 160",
                        "      spread_to_diameter_ratio_threshold: 1.35",
                        "      large_candidate_radius_m: 0.1",
                        "      min_large_candidate_count: 2",
                        "height:",
                        "  top_percentile: 99.5",
                        "  min_point_count: 50",
                    ]
                ),
                encoding="utf-8",
            )
            point_cloud = PointCloudInput(path=str(point_cloud_path), format="las")
            improved_backend = DirectGeometryBackend(config_path=improved_config_path)
            reviewed_backend = DirectGeometryBackend(config_path=reviewed_config_path)

            improved_result = improved_backend.estimate_dbh(point_cloud)
            reviewed_result = reviewed_backend.estimate_dbh(point_cloud)

            self.assertEqual(improved_result.status, "success")
            self.assertEqual(reviewed_result.status, "success")
            self.assertGreater(abs(float(improved_result.value) - expected_dbh_cm), 10.0)
            self.assertLess(abs(float(reviewed_result.value) - expected_dbh_cm), 2.0)
            self.assertTrue(reviewed_result.extra["secondary_review_selected"])
            self.assertTrue(reviewed_result.extra["merged_cluster_suspected"])
            self.assertIn("selected_primary_candidate_point_count_unusually_large", reviewed_result.extra["merged_cluster_reasons"])

    @classmethod
    def _write_tree_point_cloud(
        cls,
        path: Path,
        *,
        radius_m: float,
        height_m: float,
        include_noise: bool,
    ) -> None:
        angles = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
        z_levels = np.linspace(0.0, height_m, 220)
        points: list[list[float]] = []
        for z in z_levels:
            for angle in angles:
                points.append([radius_m * np.cos(angle), radius_m * np.sin(angle), z])
        if include_noise:
            rng = np.random.default_rng(7)
            for _ in range(150):
                points.append(
                    [
                        float(rng.normal(0.0, radius_m * 1.8)),
                        float(rng.normal(0.0, radius_m * 1.8)),
                        float(rng.uniform(0.0, height_m)),
                    ]
                )

        cls._write_points(path=path, points=np.asarray(points, dtype=np.float64))

    @staticmethod
    def _build_tree_points(*, radius_m: float, height_m: float) -> np.ndarray:
        angles = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
        z_levels = np.linspace(0.0, height_m, 220)
        points: list[list[float]] = []
        for z in z_levels:
            for angle in angles:
                points.append([radius_m * np.cos(angle), radius_m * np.sin(angle), z])
        return np.asarray(points, dtype=np.float64)

    @classmethod
    def _write_crown_tree_point_cloud(
        cls,
        path: Path,
        *,
        trunk_radius_m: float,
        trunk_height_m: float,
        crown_radius_x_m: float,
        crown_radius_y_m: float,
        extra_points: np.ndarray | None = None,
    ) -> None:
        trunk_points = cls._build_tree_points(radius_m=trunk_radius_m, height_m=trunk_height_m)
        crown_angles = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
        crown_levels = np.linspace(trunk_height_m * 0.55, trunk_height_m, 36)
        crown_points: list[list[float]] = []
        for z in crown_levels:
            for angle in crown_angles:
                crown_points.append(
                    [
                        crown_radius_x_m * np.cos(angle),
                        crown_radius_y_m * np.sin(angle),
                        z,
                    ]
                )
        point_array = np.vstack((trunk_points, np.asarray(crown_points, dtype=np.float64)))
        if extra_points is not None:
            point_array = np.vstack((point_array, extra_points))
        cls._write_points(path=path, points=point_array)

    @classmethod
    def _write_partial_crown_tree_point_cloud(
        cls,
        path: Path,
        *,
        core_radius_x_m: float,
        core_radius_y_m: float,
        sparse_edge_radius_x_m: float,
        sparse_edge_radius_y_m: float,
    ) -> None:
        trunk_points = cls._build_tree_points(radius_m=0.08, height_m=6.0)
        dense_angles = np.linspace(0.0, 2.0 * np.pi, 220, endpoint=False)
        sparse_angles = np.deg2rad(np.asarray([0, 12, 168, 180, 192, 348], dtype=np.float64))
        crown_levels = np.linspace(3.6, 6.0, 26)
        points: list[list[float]] = []
        for z in crown_levels:
            for angle in dense_angles:
                points.append(
                    [
                        core_radius_x_m * np.cos(angle),
                        core_radius_y_m * np.sin(angle),
                        z,
                    ]
                )
            for angle in sparse_angles:
                points.append(
                    [
                        sparse_edge_radius_x_m * np.cos(angle),
                        sparse_edge_radius_y_m * np.sin(angle),
                        z,
                    ]
                )
        point_array = np.vstack((trunk_points, np.asarray(points, dtype=np.float64)))
        cls._write_points(path=path, points=point_array)

    @classmethod
    def _write_contaminated_tree_point_cloud(
        cls,
        path: Path,
        *,
        trunk_radius_m: float,
        branch_radius_m: float,
        height_m: float,
    ) -> None:
        trunk_angles = np.linspace(0.0, 2.0 * np.pi, 90, endpoint=False)
        z_levels = np.linspace(0.0, height_m, 220)
        points: list[list[float]] = []
        for z in z_levels:
            for angle in trunk_angles:
                points.append([trunk_radius_m * np.cos(angle), trunk_radius_m * np.sin(angle), z])

        branch_angles = np.linspace(0.0, 2.0 * np.pi, 160, endpoint=False)
        branch_z_levels = np.linspace(1.26, 1.34, 36)
        for z in branch_z_levels:
            for angle in branch_angles:
                points.append(
                    [
                        branch_radius_m * np.cos(angle) + 0.12,
                        branch_radius_m * np.sin(angle) + 0.06,
                        z,
                    ]
                )

        cls._write_points(path=path, points=np.asarray(points, dtype=np.float64))

    @classmethod
    def _write_secondary_review_tree_point_cloud(
        cls,
        path: Path,
        *,
        trunk_radius_m: float,
        branch_radius_m: float,
        height_m: float,
    ) -> None:
        z_levels = np.linspace(0.0, height_m, 220)
        points: list[list[float]] = []

        full_angles = np.linspace(0.0, 2.0 * np.pi, 90, endpoint=False)
        partial_angles = np.linspace(-0.10 * np.pi, 1.05 * np.pi, 48, endpoint=False)
        for z in z_levels:
            active_angles = partial_angles if abs(z - 1.3) <= 0.04 else full_angles
            for angle in active_angles:
                points.append([trunk_radius_m * np.cos(angle), trunk_radius_m * np.sin(angle), z])

        branch_angles = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
        branch_z_levels = np.linspace(1.26, 1.34, 36)
        for z in branch_z_levels:
            for angle in branch_angles:
                points.append(
                    [
                        branch_radius_m * np.cos(angle) + 0.26,
                        branch_radius_m * np.sin(angle) + 0.16,
                        z,
                    ]
                )

        cls._write_points(path=path, points=np.asarray(points, dtype=np.float64))

    @staticmethod
    def _write_points(path: Path, points: np.ndarray) -> None:
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.scales = np.array([0.001, 0.001, 0.001])
        header.offsets = np.array([0.0, 0.0, 0.0])
        las = laspy.LasData(header)
        las.x = points[:, 0]
        las.y = points[:, 1]
        las.z = points[:, 2]
        las.write(path)


if __name__ == "__main__":
    unittest.main()
