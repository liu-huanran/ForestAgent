"""Smoke tests for direct geometry DBH diagnosis and comparison exports."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import laspy
import numpy as np
from openpyxl import Workbook

from forestagent.data_catalog import DataCatalog
from forestagent.evaluation import compare_direct_geometry_dbh_configs


class DirectGeometryDBHDiagnosisTests(unittest.TestCase):
    def test_compare_direct_geometry_dbh_configs_exports_tables_and_visuals(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            output_dir = Path(temp_dir) / "outputs"
            data_dir.mkdir(parents=True, exist_ok=True)
            self._create_dataset(data_dir)

            baseline_config_path = Path(temp_dir) / "baseline.yaml"
            improved_config_path = Path(temp_dir) / "improved.yaml"
            baseline_config_path.write_text(self._baseline_config_text(), encoding="utf-8")
            improved_config_path.write_text(self._improved_config_text(), encoding="utf-8")

            result = compare_direct_geometry_dbh_configs(
                data_dir=data_dir,
                sample_ids=["150_64"],
                modality="ground",
                baseline_config_path=baseline_config_path,
                improved_config_path=improved_config_path,
                output_dir=output_dir,
                top_k_visualizations=1,
            )

            self.assertEqual(result.sample_ids, ["150_64"])
            self.assertEqual(result.baseline.summary.total_records, 1)
            self.assertEqual(result.improved.summary.total_records, 1)
            self.assertTrue(Path(result.baseline.sanity_table_path).exists())
            self.assertTrue(Path(result.improved.sanity_table_path).exists())
            self.assertTrue(Path(result.comparison_path).exists())
            self.assertEqual(len(result.baseline.visualization_paths), 1)
            self.assertEqual(len(result.improved.visualization_paths), 1)
            self.assertTrue(Path(result.baseline.visualization_paths[0]).exists())
            self.assertTrue(Path(result.improved.visualization_paths[0]).exists())
            self.assertGreater(result.baseline.summary.mae, result.improved.summary.mae)
            self.assertIn("secondary_review_selected", result.improved.records[0].model_dump())
            self.assertIn("merged_cluster_suspected", result.improved.records[0].model_dump())

    @classmethod
    def _create_dataset(cls, data_dir: Path) -> None:
        ground_dir = data_dir / "TLS" / "150"
        air_dir = data_dir / "ULS" / "150"
        ground_dir.mkdir(parents=True, exist_ok=True)
        air_dir.mkdir(parents=True, exist_ok=True)
        point_cloud_path = ground_dir / "150_64.las"
        cls._write_contaminated_tree_point_cloud(
            point_cloud_path,
            trunk_radius_m=0.06,
            branch_radius_m=0.18,
            height_m=9.0,
        )
        cls._write_contaminated_tree_point_cloud(
            air_dir / "150_64.las",
            trunk_radius_m=0.06,
            branch_radius_m=0.18,
            height_m=9.0,
        )

        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Sheet1"
        sheet.append(DataCatalog._EXPECTED_HEADERS)
        sheet.append([150, "species", "150_64.las", 1, 1.0, 2.0, 3.0, 12.0, 9.0, 2.0, 2.0])
        workbook.save(data_dir / "reference.xlsx")
        workbook.close()

    @staticmethod
    def _write_contaminated_tree_point_cloud(
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

        point_array = np.asarray(points, dtype=np.float64)
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.scales = np.array([0.001, 0.001, 0.001])
        header.offsets = np.array([0.0, 0.0, 0.0])
        las = laspy.LasData(header)
        las.x = point_array[:, 0]
        las.y = point_array[:, 1]
        las.z = point_array[:, 2]
        las.write(path)

    @staticmethod
    def _baseline_config_text() -> str:
        return "\n".join(
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
        )

    @staticmethod
    def _improved_config_text() -> str:
        return "\n".join(
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
        )


if __name__ == "__main__":
    unittest.main()
