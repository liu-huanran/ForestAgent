"""Tests for template-based answer rendering."""

from __future__ import annotations

import unittest

from forestagent.renderers import render_task_answer
from forestagent.schemas import (
    LabelWithValuePayload,
    ReportTextPayload,
    ScalarPayload,
    TaskResult,
)


class AnswerRendererTests(unittest.TestCase):
    def test_render_scalar_task_answer(self) -> None:
        task_result = TaskResult(
            task_id="q1_dbh",
            task_name="胸径是多少",
            status="success",
            output_type="scalar",
            result=ScalarPayload(value=28.4, unit="cm", confidence=0.87),
            message="ok",
        )

        text = render_task_answer(task_result)

        self.assertIn("胸径是多少", text)
        self.assertIn("28.40 cm", text)
        self.assertIn("0.87", text)

    def test_render_label_with_value_task_answer(self) -> None:
        task_result = TaskResult(
            task_id="q4_tilt",
            task_name="是否明显倾斜",
            status="success",
            output_type="label_with_value",
            result=LabelWithValuePayload(
                label="无明显倾斜",
                value=9.5,
                unit="deg",
                confidence=0.78,
            ),
            message="ok",
        )

        text = render_task_answer(task_result)

        self.assertIn("无明显倾斜", text)
        self.assertIn("9.50 deg", text)

    def test_render_report_text_task_answer(self) -> None:
        task_result = TaskResult(
            task_id="q8_brief_report",
            task_name="生成简短单木报告",
            status="success",
            output_type="report_text",
            result=ReportTextPayload(text="模板报告正文"),
            message="ok",
        )

        text = render_task_answer(task_result)

        self.assertEqual(text, "模板报告正文")

    def test_render_failed_task_answer(self) -> None:
        task_result = TaskResult(
            task_id="q6_form",
            task_name="偏细高还是矮壮",
            status="failed",
            output_type="json_summary",
            result=None,
            message="estimate_height failed",
        )

        text = render_task_answer(task_result)

        self.assertIn("任务失败", text)
        self.assertIn("estimate_height failed", text)


if __name__ == "__main__":
    unittest.main()

