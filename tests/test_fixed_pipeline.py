"""Tests for the fixed task pipeline."""

from __future__ import annotations

import unittest

from forestagent.backends.mock_backend import MockBackend
from forestagent.pipelines import FixedPipeline
from forestagent.schemas import PointCloudInput


class FixedPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.point_cloud = PointCloudInput(path="data/mock_tree.laz", format="laz")

    def test_q1_dbh_returns_scalar_task_result(self) -> None:
        pipeline = FixedPipeline(MockBackend())

        result = pipeline.run("q1_dbh", self.point_cloud)

        self.assertEqual(result.task_id, "q1_dbh")
        self.assertEqual(result.status, "success")
        self.assertEqual(result.output_type, "scalar")
        self.assertEqual(result.result.unit, "cm")
        self.assertEqual(result.result.value, 28.4)

    def test_q4_tilt_returns_label_with_value_result(self) -> None:
        pipeline = FixedPipeline(MockBackend())

        result = pipeline.run("q4_tilt", self.point_cloud)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.output_type, "label_with_value")
        self.assertEqual(result.result.label, "无明显倾斜")
        self.assertEqual(result.result.value, 9.5)

    def test_q8_brief_report_returns_template_text(self) -> None:
        pipeline = FixedPipeline(MockBackend())

        result = pipeline.run("q8_brief_report", self.point_cloud)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.output_type, "report_text")
        self.assertIn("胸径约 28.40 cm", result.result.text)
        self.assertIn("无明显倾斜", result.result.text)
        self.assertIn("中等", result.result.text)
        self.assertIn("低", result.result.text)
        self.assertIn("derived_summaries", result.extra)

    def test_q6_failure_propagates_partial_results(self) -> None:
        backend = MockBackend(failures={"estimate_height": "Height tool failed in mock."})
        pipeline = FixedPipeline(backend)

        result = pipeline.run("q6_form", self.point_cloud)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.output_type, "json_summary")
        self.assertIsNone(result.result)
        self.assertIn("Height tool failed in mock.", result.message)
        self.assertEqual(
            list(result.extra["partial_results"].keys()),
            ["estimate_dbh", "estimate_height"],
        )
        self.assertEqual(
            result.extra["partial_results"]["estimate_dbh"]["status"], "success"
        )
        self.assertEqual(
            result.extra["partial_results"]["estimate_height"]["status"], "failed"
        )
        self.assertEqual(result.extra["failed_tools"], ["estimate_height"])

    def test_q7_failure_propagates_last_tool_failure(self) -> None:
        backend = MockBackend(failures={"assess_quality": "Quality tool failed in mock."})
        pipeline = FixedPipeline(backend)

        result = pipeline.run("q7_fall_risk", self.point_cloud)

        self.assertEqual(result.status, "failed")
        self.assertEqual(
            list(result.extra["partial_results"].keys()),
            ["estimate_tilt", "estimate_dbh", "estimate_height", "assess_quality"],
        )
        self.assertEqual(result.extra["failed_tools"], ["assess_quality"])
        self.assertIn("Quality tool failed in mock.", result.message)

    def test_q8_stops_on_failed_tool_and_returns_failed_task(self) -> None:
        backend = MockBackend(
            failures={"estimate_crown_width": "Crown width tool failed in mock."}
        )
        pipeline = FixedPipeline(backend)

        result = pipeline.run("q8_brief_report", self.point_cloud)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.output_type, "report_text")
        self.assertEqual(
            list(result.extra["partial_results"].keys()),
            ["estimate_dbh", "estimate_height", "estimate_crown_width"],
        )
        self.assertEqual(result.extra["failed_tools"], ["estimate_crown_width"])

    def test_unknown_task_id_raises_key_error(self) -> None:
        pipeline = FixedPipeline(MockBackend())

        with self.assertRaises(KeyError):
            pipeline.run("unknown_task", self.point_cloud)


if __name__ == "__main__":
    unittest.main()

