"""Tests for the small single-tree MVP demo script."""

from __future__ import annotations

import unittest

from forestagent.backends.mock_backend import MockBackend
from forestagent.demo_single_tree_mvp import build_demo_cases, run_demo_cases


class DemoSingleTreeMvpTests(unittest.TestCase):
    def test_build_demo_cases_without_failure_demo_returns_four_cases(self) -> None:
        cases = build_demo_cases(include_failure_demo=False)

        self.assertEqual(len(cases), 4)
        self.assertEqual(cases[0]["label"], "q1_dbh")
        self.assertEqual(cases[-1]["label"], "tree_report_q123")

    def test_build_demo_cases_with_failure_demo_adds_ambiguous_case(self) -> None:
        cases = build_demo_cases(include_failure_demo=True)

        self.assertEqual(len(cases), 5)
        self.assertEqual(cases[-1]["label"], "ambiguous_question")

    def test_run_demo_cases_uses_existing_single_tree_api(self) -> None:
        payload = run_demo_cases(
            "demo_tree.las",
            include_failure_demo=True,
            backend=MockBackend(),
        )

        self.assertEqual(payload["point_cloud"], "demo_tree.las")
        self.assertEqual(payload["case_count"], 5)
        self.assertEqual(payload["cases"][0]["response"]["status"], "success")
        self.assertEqual(payload["cases"][-1]["response"]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
