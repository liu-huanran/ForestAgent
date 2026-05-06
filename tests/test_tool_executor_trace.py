"""Tests for v2 mock executor and trace records."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from forestagent.agent.executor import MockToolExecutor
from forestagent.agent.planner import DeterministicPlanner


class ToolExecutorTraceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = DeterministicPlanner()
        self.executor = MockToolExecutor()

    def test_mock_q123_execution_generates_measurement_evidence(self) -> None:
        plan = self.planner.plan(
            "\u7ed9\u6211\u4e00\u4e2a\u8fd9\u68f5\u6811\u7684\u7efc\u5408\u62a5\u544a"
        )

        result = self.executor.execute(plan, {"point_cloud_path": "mock://tree"})

        self.assertEqual(
            [tool_result.status for tool_result in result.tool_results],
            ["ok", "ok", "ok", "skipped"],
        )
        self.assertEqual(len(result.evidence_packet.ok_measurements()), 3)
        self.assertEqual(
            result.evidence_packet.get_item("measurement.dbh").source_tool,
            "q1_dbh",
        )
        self.assertEqual(result.trace.selected_tools, plan.selected_tools)

    def test_mock_failure_is_recorded_as_failed_tool_and_evidence(self) -> None:
        plan = self.planner.plan("\u8fd9\u68f5\u6811\u7684\u80f8\u5f84\u662f\u591a\u5c11\uff1f")

        result = self.executor.execute(
            plan,
            {
                "point_cloud_path": "mock://tree",
                "mock_failures": {"q1_dbh": "mock DBH failure"},
            },
        )

        self.assertEqual(result.tool_results[0].status, "failed")
        self.assertIn("mock DBH failure", result.trace.errors)
        unavailable = result.evidence_packet.get_item("unavailable.q1_dbh")
        self.assertIsNotNone(unavailable)
        self.assertEqual(unavailable.status, "failed")
        self.assertIsNone(unavailable.value)

    def test_missing_required_input_returns_unavailable(self) -> None:
        plan = self.planner.plan("\u5e2e\u6211\u6d4b\u4e00\u4e0b\u6811\u9ad8")

        result = self.executor.execute(plan, {})

        self.assertEqual(result.tool_results[0].status, "unavailable")
        self.assertIn("point_cloud_path", result.tool_results[0].error_message)
        unavailable = result.evidence_packet.get_item("unavailable.q2_height")
        self.assertEqual(unavailable.status, "unavailable")

    def test_offline_cache_tool_remains_unavailable_stub(self) -> None:
        plan = self.planner.plan(
            "\u627e\u548c\u8fd9\u68f5\u6811\u76f8\u4f3c\u7684\u6811"
        )

        result = self.executor.execute(
            plan,
            {"tree_id": "tree_001", "embedding_cache_dir": "mock://cache"},
        )

        self.assertEqual(result.tool_results[0].tool_name, "similar_tree_retrieval")
        self.assertEqual(result.tool_results[0].status, "unavailable")
        self.assertTrue(result.tool_results[0].metadata["offline_only"])
        self.assertIn("Uni3D forward", result.tool_results[0].error_message)

    def test_trace_can_save_json(self) -> None:
        plan = self.planner.plan("\u51a0\u5e45\u5927\u6982\u662f\u591a\u5c11\uff1f")
        result = self.executor.execute(plan, {"point_cloud_path": "mock://tree"})

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = result.trace.save_json(Path(temp_dir) / "trace.json")
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["request_id"], plan.request_id)
        self.assertEqual(payload["resolved_intent"], "q3_crown_width")
        self.assertEqual(payload["config_version"], "forestagent-v2-phase1-2")


if __name__ == "__main__":
    unittest.main()
