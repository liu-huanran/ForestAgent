"""Tests for the ForestAgent v2 response builder."""

from __future__ import annotations

import unittest

from forestagent.agent.executor import MockToolExecutor
from forestagent.agent.planner import DeterministicPlanner
from forestagent.agent.response_builder import ResponseBuilder


class ResponseBuilderV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = DeterministicPlanner()
        self.executor = MockToolExecutor()
        self.builder = ResponseBuilder()

    def test_report_text_uses_evidence_value_not_default_claim(self) -> None:
        plan = self.planner.plan("\u8fd9\u68f5\u6811\u7684\u80f8\u5f84\u662f\u591a\u5c11\uff1f")
        execution = self.executor.execute(
            plan,
            {"point_cloud_path": "mock://tree", "mock_values": {"q1_dbh": 31.2}},
        )

        response = self.builder.build(
            user_question=plan.user_question,
            planner_result=plan,
            evidence_packet=execution.evidence_packet,
            tool_results=execution.tool_results,
        )

        self.assertIn("31.20 cm", response.report_text)
        self.assertNotIn("23.40 cm", response.report_text)
        self.assertEqual(response.structured_summary["measurements"]["dbh_cm"]["value"], 31.2)

    def test_missing_dbh_evidence_does_not_fabricate_value(self) -> None:
        plan = self.planner.plan("\u8fd9\u68f5\u6811\u7684\u80f8\u5f84\u662f\u591a\u5c11\uff1f")
        execution = self.executor.execute(plan, {})

        response = self.builder.build(
            user_question=plan.user_question,
            planner_result=plan,
            evidence_packet=execution.evidence_packet,
            tool_results=execution.tool_results,
        )

        self.assertIn("\u672a\u80fd\u8ba1\u7b97", response.report_text)
        self.assertNotIn("23.40 cm", response.report_text)
        self.assertEqual(response.structured_summary["status"], "failed")

    def test_unsupported_question_gets_explicit_refusal(self) -> None:
        plan = self.planner.plan("\u8fd9\u68f5\u6811\u5065\u5eb7\u5417\uff1f")
        execution = self.executor.execute(plan, {"point_cloud_path": "mock://tree"})

        response = self.builder.build(
            user_question=plan.user_question,
            planner_result=plan,
            evidence_packet=execution.evidence_packet,
            tool_results=execution.tool_results,
        )

        self.assertEqual(response.structured_summary["status"], "unsupported")
        self.assertIn("\u65e0\u6cd5\u56de\u7b54", response.report_text)
        self.assertEqual(response.structured_summary["measurements"], {})

    def test_comprehensive_report_can_show_partial_failure(self) -> None:
        plan = self.planner.plan(
            "\u7ed9\u6211\u4e00\u4e2a\u8fd9\u68f5\u6811\u7684\u7efc\u5408\u62a5\u544a"
        )
        execution = self.executor.execute(
            plan,
            {
                "point_cloud_path": "mock://tree",
                "mock_failures": {"q2_height": "mock height failure"},
            },
        )

        response = self.builder.build(
            user_question=plan.user_question,
            planner_result=plan,
            evidence_packet=execution.evidence_packet,
            tool_results=execution.tool_results,
        )

        self.assertEqual(response.structured_summary["status"], "partial")
        self.assertIn("23.40 cm", response.report_text)
        self.assertIn("\u6811\u9ad8\u672a\u80fd\u8ba1\u7b97", response.report_text)
        self.assertIn("4.20 m", response.report_text)
        self.assertNotIn("12.50 m", response.report_text)


if __name__ == "__main__":
    unittest.main()
