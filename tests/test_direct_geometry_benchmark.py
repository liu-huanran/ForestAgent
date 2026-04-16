"""Smoke tests for the direct geometry q1/q2 benchmark entry."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook

from forestagent.backends.base_backend import BaseBackend
from forestagent.data_catalog import DataCatalog
from forestagent.evaluation import benchmark_direct_geometry_baseline
from forestagent.schemas import PointCloudInput, ToolResult


class MappingDirectBenchmarkBackend(BaseBackend):
    """Simple backend stub for direct geometry benchmark smoke tests."""

    def __init__(self, results_by_sample: dict[str, dict[str, ToolResult]]) -> None:
        self._results_by_sample = results_by_sample

    def estimate_dbh(self, point_cloud: PointCloudInput) -> ToolResult:
        return self._results_by_sample[Path(point_cloud.path).stem]["estimate_dbh"]

    def estimate_height(self, point_cloud: PointCloudInput) -> ToolResult:
        return self._results_by_sample[Path(point_cloud.path).stem]["estimate_height"]

    def estimate_crown_width(self, point_cloud: PointCloudInput) -> ToolResult:
        return ToolResult(
            tool_name="estimate_crown_width",
            status="failed",
            value=None,
            unit=None,
            confidence=None,
            extra={"backend": "stub", "unsupported": True},
            message="unsupported",
        )

    def estimate_tilt(self, point_cloud: PointCloudInput) -> ToolResult:
        return self.estimate_crown_width(point_cloud)

    def assess_quality(self, point_cloud: PointCloudInput) -> ToolResult:
        return self.estimate_crown_width(point_cloud)


class DirectGeometryBenchmarkTests(unittest.TestCase):
    def test_benchmark_direct_geometry_baseline_returns_records_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            self._create_sample_dataset(
                data_dir,
                rows=[
                    (150, "species_a", "150_01.las", 1, 1.0, 2.0, 3.0, 10.0, 8.0, 2.0, 4.0),
                    (151, "species_b", "151_02.las", 2, 4.0, 5.0, 6.0, 20.0, 12.0, 3.0, 5.0),
                ],
            )
            backend = MappingDirectBenchmarkBackend(
                results_by_sample={
                    "150_01": {
                        "estimate_dbh": ToolResult(
                            tool_name="estimate_dbh",
                            status="success",
                            value=11.5,
                            unit="cm",
                            confidence=None,
                            extra={"backend": "stub"},
                            message="dbh ok",
                        ),
                        "estimate_height": ToolResult(
                            tool_name="estimate_height",
                            status="success",
                            value=7.5,
                            unit="m",
                            confidence=None,
                            extra={"backend": "stub"},
                            message="height ok",
                        ),
                    },
                    "151_02": {
                        "estimate_dbh": ToolResult(
                            tool_name="estimate_dbh",
                            status="failed",
                            value=None,
                            unit=None,
                            confidence=None,
                            extra={"backend": "stub"},
                            message="dbh failed",
                        ),
                        "estimate_height": ToolResult(
                            tool_name="estimate_height",
                            status="success",
                            value=12.5,
                            unit="m",
                            confidence=None,
                            extra={"backend": "stub"},
                            message="height ok",
                        ),
                    },
                }
            )

            result = benchmark_direct_geometry_baseline(
                data_dir=data_dir,
                sample_ids=["150_01", "151_02"],
                modality="ground",
                backend=backend,
            )

            self.assertEqual(result.sample_ids, ["150_01", "151_02"])
            self.assertEqual(len(result.records), 4)
            self.assertTrue(all(record.modality == "ground" for record in result.records))

            q1_metrics = next(metric for metric in result.metrics if metric.task_id == "q1_dbh")
            self.assertEqual(q1_metrics.success_count, 1)
            self.assertAlmostEqual(q1_metrics.success_rate, 0.5, places=6)

            q2_metrics = next(metric for metric in result.metrics if metric.task_id == "q2_height")
            self.assertEqual(q2_metrics.success_count, 2)
            self.assertAlmostEqual(q2_metrics.success_rate, 1.0, places=6)

            failure_stat = next(
                item for item in result.failure_reason_stats if item.task_id == "q1_dbh"
            )
            self.assertEqual(failure_stat.count, 1)
            self.assertIn("dbh failed", failure_stat.reason)

    @staticmethod
    def _create_sample_dataset(data_dir: Path, rows: list[tuple[object, ...]]) -> None:
        ground_dir = data_dir / "TLS"
        air_dir = data_dir / "ULS"
        for row in rows:
            plot_id = str(row[0])
            (ground_dir / plot_id).mkdir(parents=True, exist_ok=True)
            (air_dir / plot_id).mkdir(parents=True, exist_ok=True)
            (ground_dir / plot_id / str(row[2])).write_text("", encoding="utf-8")
            (air_dir / plot_id / str(row[2])).write_text("", encoding="utf-8")

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
