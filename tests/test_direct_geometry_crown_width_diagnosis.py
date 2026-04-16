"""Smoke tests for direct geometry q3_crown_width freeze validation exports."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import laspy
import numpy as np
from openpyxl import Workbook

from forestagent.data_catalog import DataCatalog
from forestagent.evaluation import run_direct_geometry_crown_width_freeze_validation


class DirectGeometryCrownWidthDiagnosisTests(unittest.TestCase):
    def test_run_direct_geometry_crown_width_freeze_validation_exports_records_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            output_dir = Path(temp_dir) / "outputs"
            data_dir.mkdir(parents=True, exist_ok=True)
            self._create_dataset(data_dir)

            result = run_direct_geometry_crown_width_freeze_validation(
                data_dir=data_dir,
                sample_ids=["150_01", "150_02"],
                modality="ground",
                output_dir=output_dir,
                top_k_errors=2,
            )

            self.assertEqual(result.sample_ids, ["150_01", "150_02"])
            self.assertEqual(result.summary.total_records, 2)
            self.assertEqual(result.summary.success_count, 2)
            self.assertTrue(Path(result.per_sample_records_path).exists())
            self.assertTrue(Path(result.top_abs_error_path).exists())
            self.assertTrue(Path(result.summary_path).exists())
            self.assertEqual(len(result.top_absolute_error_records), 2)

            first_record = next(record for record in result.records if record.sample_id == "150_01")
            self.assertIsNotNone(first_record.projected_point_count)
            self.assertIsNotNone(first_record.unique_xy_count)
            self.assertIsNotNone(first_record.predicted_crown_width_m)
            self.assertIsNotNone(first_record.raw_vs_robust_gap_m)
            self.assertIsNotNone(first_record.pca_mean_span_m)
            self.assertIsNotNone(first_record.fixed_minus_pca_mean_m)
            self.assertFalse(first_record.sparse_projection_suspected)

            shrinkage_stat = next(
                stat
                for stat in result.suspicion_flag_stats
                if stat.flag_name == "robust_shrinkage_suspected"
            )
            self.assertGreaterEqual(shrinkage_stat.count, 0)

    @classmethod
    def _create_dataset(cls, data_dir: Path) -> None:
        ground_dir = data_dir / "TLS" / "150"
        air_dir = data_dir / "ULS" / "150"
        ground_dir.mkdir(parents=True, exist_ok=True)
        air_dir.mkdir(parents=True, exist_ok=True)
        cls._write_crown_tree_point_cloud(
            ground_dir / "150_01.las",
            crown_radius_x_m=1.8,
            crown_radius_y_m=1.2,
        )
        cls._write_crown_tree_point_cloud(
            ground_dir / "150_02.las",
            crown_radius_x_m=1.5,
            crown_radius_y_m=1.0,
            extra_points=np.asarray([[7.0, 7.0, 5.0], [-7.0, -7.0, 5.2]], dtype=np.float64),
        )
        cls._write_crown_tree_point_cloud(air_dir / "150_01.las", crown_radius_x_m=1.8, crown_radius_y_m=1.2)
        cls._write_crown_tree_point_cloud(air_dir / "150_02.las", crown_radius_x_m=1.5, crown_radius_y_m=1.0)

        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Sheet1"
        sheet.append(DataCatalog._EXPECTED_HEADERS)
        sheet.append([150, "species_a", "150_01.las", 1, 1.0, 2.0, 3.0, 10.0, 11.0, 3.6, 2.4])
        sheet.append([150, "species_b", "150_02.las", 2, 1.0, 2.0, 3.0, 12.0, 9.5, 3.0, 2.0])
        workbook.save(data_dir / "reference.xlsx")
        workbook.close()

    @staticmethod
    def _write_crown_tree_point_cloud(
        path: Path,
        *,
        crown_radius_x_m: float,
        crown_radius_y_m: float,
        extra_points: np.ndarray | None = None,
    ) -> None:
        trunk_angles = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)
        trunk_levels = np.linspace(0.0, 6.0, 120)
        points: list[list[float]] = []
        for z in trunk_levels:
            for angle in trunk_angles:
                points.append([0.1 * np.cos(angle), 0.1 * np.sin(angle), z])

        crown_angles = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
        crown_levels = np.linspace(3.5, 6.2, 32)
        for z in crown_levels:
            for angle in crown_angles:
                points.append(
                    [
                        crown_radius_x_m * np.cos(angle),
                        crown_radius_y_m * np.sin(angle),
                        z,
                    ]
                )

        point_array = np.asarray(points, dtype=np.float64)
        if extra_points is not None:
            point_array = np.vstack((point_array, extra_points))

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
