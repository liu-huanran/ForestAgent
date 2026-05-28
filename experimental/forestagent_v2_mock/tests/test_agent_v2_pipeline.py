"""Tests for the ForestAgent v2 mock pipeline."""

from __future__ import annotations

import json
import unittest

from forestagent.agent.v2_pipeline import AgentV2Pipeline, run_agent_v2_mock


class AgentV2PipelineTests(unittest.TestCase):
    def test_comprehensive_report_minimum_loop_succeeds(self) -> None:
        result = run_agent_v2_mock(
            "\u7ed9\u6211\u4e00\u4e2a\u8fd9\u68f5\u6811\u7684\u7efc\u5408\u62a5\u544a"
        )

        self.assertEqual(result.planner_result.status, "planned")
        self.assertEqual(result.planner_result.resolved_intent, "tree_report_q123")
        self.assertEqual(result.structured_summary["status"], "success")
        self.assertEqual(len(result.tool_results), 4)
        self.assertEqual(len(result.evidence_packet.ok_measurements()), 3)
        self.assertIn("23.40 cm", result.report_text)
        self.assertIn("12.50 m", result.report_text)
        self.assertIn("4.20 m", result.report_text)
        self.assertEqual(result.trace.report_text, result.report_text)

    def test_unsupported_question_does_not_call_measurement_tools(self) -> None:
        result = run_agent_v2_mock("\u8fd9\u68f5\u6811\u662f\u4e0d\u662f\u5371\u9669\u6728\uff1f")

        self.assertEqual(result.planner_result.status, "unsupported")
        self.assertEqual(result.tool_results, [])
        self.assertEqual(result.evidence_packet.items, [])
        self.assertEqual(result.structured_summary["status"], "unsupported")
        self.assertEqual(result.structured_summary["measurements"], {})
        self.assertIn("\u65e0\u6cd5\u56de\u7b54", result.report_text)

    def test_pipeline_accepts_mock_measurement_overrides(self) -> None:
        result = AgentV2Pipeline().run(
            "\u5e2e\u6211\u6d4b\u4e00\u4e0b\u6811\u9ad8",
            input_context={"mock_values": {"q2_height": 18.75}},
        )

        self.assertEqual(result.structured_summary["measurements"]["height_m"]["value"], 18.75)
        self.assertIn("18.75 m", result.report_text)

    def test_trace_is_json_serializable_without_large_artifacts(self) -> None:
        result = run_agent_v2_mock("\u51a0\u5e45\u5927\u6982\u662f\u591a\u5c11\uff1f")

        payload = json.loads(result.trace.model_dump_json())

        self.assertEqual(payload["resolved_intent"], "q3_crown_width")
        self.assertIn("evidence_packet", payload)
        self.assertNotIn("point_cloud_bytes", json.dumps(payload))
        self.assertNotIn("embedding_vector", json.dumps(payload))


if __name__ == "__main__":
    unittest.main()
