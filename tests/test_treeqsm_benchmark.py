"""Tests for the small-sample TreeQSM baseline benchmark entry."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook

from forestagent.backends.treeqsm_backend import (
    TreeQSMBackend,
    TreeQSMBackendError,
    TreeQSMMetrics,
    TreeQSMRunResult,
)
from forestagent.data_catalog import DataCatalog
from forestagent.evaluation import benchmark_treeqsm_baseline
from forestagent.schemas import PointCloudInput


class MappingTreeQSMBackend(TreeQSMBackend):
    """Backend stub that returns per-sample TreeQSM metrics from a mapping."""

    def __init__(
        self,
        metrics_by_sample: dict[str, TreeQSMMetrics],
        error_by_sample: dict[str, str] | None = None,
    ) -> None:
        super().__init__(treeqsm_root="D:/fake/treeqsm", matlab_executable="C:/fake/matlab.exe")
        self._metrics_by_sample = dict(metrics_by_sample)
        self._error_by_sample = dict(error_by_sample or {})

    def _compute_treeqsm_run_result(self, point_cloud: PointCloudInput) -> TreeQSMRunResult:
        sample_id = Path(point_cloud.path).stem
        if sample_id in self._error_by_sample:
            raise TreeQSMBackendError(self._error_by_sample[sample_id])
        if sample_id not in self._metrics_by_sample:
            raise TreeQSMBackendError(f"No stub metrics configured for {sample_id}.")
        return TreeQSMRunResult(
            metrics=self._metrics_by_sample[sample_id],
            runtime_seconds=2.0,
        )


class TreeQSMBenchmarkTests(unittest.TestCase):
    def test_benchmark_treeqsm_baseline_computes_metrics_and_failure_stats(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            self._create_sample_dataset(
                data_dir,
                rows=[
                    (150, "树种A", "150_01.las", 1, 1.0, 2.0, 3.0, 10.0, 20.0, 2.0, 4.0),
                    (151, "树种B", "151_02.las", 2, 4.0, 5.0, 6.0, 20.0, 30.0, 3.0, 5.0),
                ],
                ground_files=["150_01.las", "151_02.las"],
                air_files=["150_01.las", "151_02.las"],
            )
            backend = MappingTreeQSMBackend(
                metrics_by_sample={
                    "150_01": TreeQSMMetrics(dbh_cyl_m=0.12, dbh_qsm_m=0.118, tree_height_m=18.0),
                    "151_02": TreeQSMMetrics(dbh_cyl_m=0.21, dbh_qsm_m=0.205, tree_height_m=None),
                }
            )

            result = benchmark_treeqsm_baseline(
                data_dir=data_dir,
                sample_ids=["150_01", "151_02"],
                modality="ground",
                backend=backend,
            )

            self.assertEqual(result.sample_ids, ["150_01", "151_02"])
            self.assertEqual(len(result.records), 4)

            q1_metrics = next(metric for metric in result.metrics if metric.task_id == "q1_dbh")
            self.assertEqual(q1_metrics.total_records, 2)
            self.assertEqual(q1_metrics.success_count, 2)
            self.assertAlmostEqual(q1_metrics.success_rate, 1.0, places=6)
            self.assertAlmostEqual(q1_metrics.mae, 1.5, places=6)

            q2_metrics = next(
                metric for metric in result.metrics if metric.task_id == "q2_height"
            )
            self.assertEqual(q2_metrics.total_records, 2)
            self.assertEqual(q2_metrics.success_count, 1)
            self.assertAlmostEqual(q2_metrics.success_rate, 0.5, places=6)
            self.assertAlmostEqual(q2_metrics.mae, 2.0, places=6)
            self.assertAlmostEqual(q2_metrics.rmse, 2.0, places=6)

            failure_stat = next(
                item for item in result.failure_reason_stats if item.task_id == "q2_height"
            )
            self.assertEqual(failure_stat.status, "task_failed")
            self.assertEqual(failure_stat.count, 1)
            self.assertIn("TreeQSM output missing TreeHeight.", failure_stat.reason)

    def test_benchmark_treeqsm_baseline_selects_default_samples_with_measured_q1_q2(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            self._create_sample_dataset(
                data_dir,
                rows=[
                    (150, "树种A", "150_01.las", 1, 1.0, 2.0, 3.0, 10.0, 20.0, 2.0, 4.0),
                    (151, "树种B", "151_02.las", 2, 4.0, 5.0, 6.0, None, 30.0, 3.0, 5.0),
                    (152, "树种C", "152_03.las", 3, 7.0, 8.0, 9.0, 14.0, 22.0, 3.0, 5.0),
                ],
                ground_files=["150_01.las", "151_02.las", "152_03.las"],
                air_files=["150_01.las", "151_02.las", "152_03.las"],
            )
            backend = MappingTreeQSMBackend(
                metrics_by_sample={
                    "150_01": TreeQSMMetrics(dbh_cyl_m=0.10, dbh_qsm_m=0.10, tree_height_m=20.0),
                    "152_03": TreeQSMMetrics(dbh_cyl_m=0.14, dbh_qsm_m=0.14, tree_height_m=22.0),
                }
            )

            result = benchmark_treeqsm_baseline(
                data_dir=data_dir,
                limit=2,
                modality="ground",
                backend=backend,
            )

            self.assertEqual(result.sample_ids, ["150_01", "152_03"])
            self.assertEqual(len(result.records), 4)

    @staticmethod
    def _create_sample_dataset(
        data_dir: Path,
        rows: list[tuple[object, ...]],
        ground_files: list[str],
        air_files: list[str],
    ) -> None:
        ground_dir = data_dir / "ground_points"
        air_dir = data_dir / "air_points"
        for row in rows:
            plot_id = str(row[0])
            (ground_dir / plot_id).mkdir(parents=True, exist_ok=True)
            (air_dir / plot_id).mkdir(parents=True, exist_ok=True)

        for file_name in ground_files:
            plot_id = file_name.split("_", 1)[0]
            (ground_dir / plot_id / file_name).write_text("", encoding="utf-8")
        for file_name in air_files:
            plot_id = file_name.split("_", 1)[0]
            (air_dir / plot_id / file_name).write_text("", encoding="utf-8")

        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Sheet1"
        sheet.append(DataCatalog._EXPECTED_HEADERS)
        for row in rows:
            sheet.append(list(row))
        workbook.save(data_dir / "reference.xlsx")
        workbook.close()


if __name__ == "__main__":
    unittest.main()
