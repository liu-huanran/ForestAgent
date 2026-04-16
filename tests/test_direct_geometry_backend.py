"""Tests for the direct geometry backend tool contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import laspy
import numpy as np

from forestagent.backends.direct_geometry_backend import DirectGeometryBackend
from forestagent.schemas import PointCloudInput
from forestagent.tools.assess_quality import assess_quality
from forestagent.tools.estimate_crown_width import estimate_crown_width
from forestagent.tools.estimate_dbh import estimate_dbh
from forestagent.tools.estimate_height import estimate_height
from forestagent.tools.estimate_tilt import estimate_tilt


class DirectGeometryBackendContractTests(unittest.TestCase):
    def test_tools_return_normalized_results_for_direct_geometry_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            point_cloud_path = Path(temp_dir) / "synthetic_tree.las"
            self._write_tree_point_cloud(point_cloud_path, radius_m=0.15, height_m=8.0)
            point_cloud = PointCloudInput(path=str(point_cloud_path), format="las")
            backend = DirectGeometryBackend()

            dbh_result = estimate_dbh(point_cloud, backend)
            height_result = estimate_height(point_cloud, backend)
            crown_width_result = estimate_crown_width(point_cloud, backend)

            self.assertEqual(dbh_result.status, "success")
            self.assertEqual(dbh_result.unit, "cm")
            self.assertEqual(dbh_result.tool_name, "estimate_dbh")
            self.assertIn("circle_fitting_method", dbh_result.extra)
            self.assertTrue(dbh_result.extra.get("trunk_only_filtering_enabled", False))
            self.assertFalse(dbh_result.extra.get("secondary_review_selected", False))
            self.assertIn("merged_cluster_suspected", dbh_result.extra)

            self.assertEqual(height_result.status, "success")
            self.assertEqual(height_result.unit, "m")
            self.assertEqual(height_result.tool_name, "estimate_height")
            self.assertIn("top_percentile", height_result.extra)
            self.assertIn("ground_reference_z_m", height_result.extra)
            self.assertIn("ground_seed_z_m", height_result.extra)
            self.assertIn("top_reference_z_m", height_result.extra)
            self.assertIn("top_seed_z_m", height_result.extra)
            self.assertIn("top_point_count", height_result.extra)
            self.assertIn("total_point_count", height_result.extra)
            self.assertIn("height_value_m", height_result.extra)
            self.assertIn("top_outlier_suspected", height_result.extra)
            self.assertIn("sparse_top_suspected", height_result.extra)
            self.assertIn("truncated_top_suspected", height_result.extra)
            self.assertIn("ground_reference_unstable_suspected", height_result.extra)
            self.assertFalse(height_result.extra["top_outlier_suspected"])
            self.assertFalse(height_result.extra["sparse_top_suspected"])
            self.assertFalse(height_result.extra["ground_reference_unstable_suspected"])

            self.assertEqual(crown_width_result.status, "success")
            self.assertEqual(crown_width_result.unit, "m")
            self.assertEqual(crown_width_result.tool_name, "estimate_crown_width")
            self.assertIn("total_point_count", crown_width_result.extra)
            self.assertIn("projected_point_count", crown_width_result.extra)
            self.assertIn("unique_xy_count", crown_width_result.extra)
            self.assertIn("removed_duplicate_count", crown_width_result.extra)
            self.assertIn("removed_outlier_count", crown_width_result.extra)
            self.assertIn("x_raw_span_m", crown_width_result.extra)
            self.assertIn("y_raw_span_m", crown_width_result.extra)
            self.assertIn("x_robust_span_m", crown_width_result.extra)
            self.assertIn("y_robust_span_m", crown_width_result.extra)
            self.assertIn("raw_mean_span_m", crown_width_result.extra)
            self.assertIn("raw_vs_robust_gap_m", crown_width_result.extra)
            self.assertIn("raw_vs_robust_gap_ratio", crown_width_result.extra)
            self.assertIn("predicted_crown_width_m", crown_width_result.extra)
            self.assertIn("pca_major_span_m", crown_width_result.extra)
            self.assertIn("pca_minor_span_m", crown_width_result.extra)
            self.assertIn("pca_mean_span_m", crown_width_result.extra)
            self.assertIn("axis_vs_pca_gap_m", crown_width_result.extra)
            self.assertIn("fixed_minus_pca_mean_m", crown_width_result.extra)
            self.assertIn("fixed_to_pca_ratio", crown_width_result.extra)
            self.assertIn("raw_axis_asymmetry_ratio", crown_width_result.extra)
            self.assertIn("robust_axis_asymmetry_ratio", crown_width_result.extra)
            self.assertIn("removed_outlier_fraction", crown_width_result.extra)
            self.assertIn("x_low_edge_support_count", crown_width_result.extra)
            self.assertIn("x_high_edge_support_count", crown_width_result.extra)
            self.assertIn("y_low_edge_support_count", crown_width_result.extra)
            self.assertIn("y_high_edge_support_count", crown_width_result.extra)
            self.assertIn("edge_support_band_m", crown_width_result.extra)
            self.assertIn("sparse_projection_suspected", crown_width_result.extra)
            self.assertIn("heavy_outlier_removal_suspected", crown_width_result.extra)
            self.assertIn("axis_pca_gap_suspected", crown_width_result.extra)
            self.assertIn("elongated_projection_suspected", crown_width_result.extra)
            self.assertIn("robust_shrinkage_suspected", crown_width_result.extra)
            self.assertIn("edge_sparsity_suspected", crown_width_result.extra)
            self.assertIn("pca_axis_underestimate_suspected", crown_width_result.extra)
            self.assertFalse(crown_width_result.extra["sparse_projection_suspected"])

    def test_direct_geometry_backend_reports_unsupported_tools(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            point_cloud_path = Path(temp_dir) / "synthetic_tree.las"
            self._write_tree_point_cloud(point_cloud_path, radius_m=0.15, height_m=8.0)
            point_cloud = PointCloudInput(path=str(point_cloud_path), format="las")
            backend = DirectGeometryBackend()

            for result in [
                estimate_tilt(point_cloud, backend),
                assess_quality(point_cloud, backend),
            ]:
                self.assertEqual(result.status, "failed")
                self.assertTrue(result.extra["unsupported"])
                self.assertEqual(result.extra["backend"], "direct_geometry")

    @staticmethod
    def _write_tree_point_cloud(path: Path, *, radius_m: float, height_m: float) -> None:
        angles = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
        z_levels = np.linspace(0.0, height_m, 160)
        points: list[list[float]] = []
        for z in z_levels:
            for angle in angles:
                points.append([radius_m * np.cos(angle), radius_m * np.sin(angle), z])

        point_array = np.asarray(points, dtype=np.float64)
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.scales = np.array([0.001, 0.001, 0.001])
        header.offsets = np.array([0.0, 0.0, 0.0])
        las = laspy.LasData(header)
        las.x = point_array[:, 0]
        las.y = point_array[:, 1]
        las.z = point_array[:, 2]
        las.write(path)


if __name__ == "__main__":
    unittest.main()
