"""Tests for the deterministic ForestAgent v2 planner."""

from __future__ import annotations

import unittest

from forestagent.agent.planner import DeterministicPlanner


class AgentPlannerV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = DeterministicPlanner()

    def test_routes_dbh_question_to_q1(self) -> None:
        result = self.planner.plan("\u8fd9\u68f5\u6811\u7684\u80f8\u5f84\u662f\u591a\u5c11\uff1f")

        self.assertEqual(result.status, "planned")
        self.assertEqual(result.resolved_intent, "q1_dbh")
        self.assertEqual(result.selected_tools, ["q1_dbh"])
        self.assertIn("Rule route", result.planner_reason)

    def test_routes_height_question_to_q2(self) -> None:
        result = self.planner.plan("\u5e2e\u6211\u6d4b\u4e00\u4e0b\u6811\u9ad8")

        self.assertEqual(result.status, "planned")
        self.assertEqual(result.resolved_intent, "q2_height")
        self.assertEqual(result.selected_tools, ["q2_height"])

    def test_routes_crown_width_question_to_q3(self) -> None:
        result = self.planner.plan("\u51a0\u5e45\u5927\u6982\u662f\u591a\u5c11\uff1f")

        self.assertEqual(result.status, "planned")
        self.assertEqual(result.resolved_intent, "q3_crown_width")
        self.assertEqual(result.selected_tools, ["q3_crown_width"])

    def test_routes_comprehensive_report_to_q123_report(self) -> None:
        result = self.planner.plan(
            "\u7ed9\u6211\u4e00\u4e2a\u8fd9\u68f5\u6811\u7684\u7efc\u5408\u62a5\u544a"
        )

        self.assertEqual(result.status, "planned")
        self.assertEqual(result.resolved_intent, "tree_report_q123")
        self.assertEqual(
            result.selected_tools,
            ["q1_dbh", "q2_height", "q3_crown_width", "tree_report_q123"],
        )

    def test_similar_tree_query_is_unavailable_without_cache(self) -> None:
        result = self.planner.plan(
            "\u627e\u548c\u8fd9\u68f5\u6811\u76f8\u4f3c\u7684\u6811"
        )

        self.assertEqual(result.status, "unavailable")
        self.assertEqual(result.resolved_intent, "similar_tree_retrieval")
        self.assertEqual(result.selected_tools, ["similar_tree_retrieval"])
        self.assertEqual(result.unsupported_reason, "embedding_cache_not_configured")

    def test_unsupported_questions_are_explicitly_rejected(self) -> None:
        cases = {
            "\u8fd9\u68f5\u6811\u5065\u5eb7\u5417\uff1f": "health_status_without_evidence",
            "\u8fd9\u68f5\u6811\u662f\u4e0d\u662f\u5371\u9669\u6728\uff1f": (
                "danger_tree_without_evidence"
            ),
            "\u5b83\u662f\u4e0d\u662f\u6709\u75c5\u866b\u5bb3\uff1f": (
                "disease_or_pest_without_evidence"
            ),
            "\u8fd9\u68f5\u6811\u957f\u5f97\u597d\u5417\uff1f": (
                "subjective_growth_quality_without_evidence"
            ),
            "\u8bf7\u5224\u65ad\u8fd9\u68f5\u6811\u662f\u5426\u9700\u8981\u780d\u4f10": (
                "cutting_decision_without_evidence"
            ),
            "q4 can you assess tilt?": "q4_to_q8_out_of_scope",
        }

        for question, expected_reason in cases.items():
            with self.subTest(question=question):
                result = self.planner.plan(question)
                self.assertEqual(result.status, "unsupported")
                self.assertIsNone(result.resolved_intent)
                self.assertEqual(result.selected_tools, [])
                self.assertEqual(result.execution_plan, [])
                self.assertEqual(result.unsupported_reason, expected_reason)


if __name__ == "__main__":
    unittest.main()
