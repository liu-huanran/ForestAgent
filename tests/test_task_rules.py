"""Tests for task-level rule evaluation."""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from forestagent.logic.task_rules import (
    TaskRuleError,
    evaluate_q4_tilt,
    evaluate_q5_quality,
    evaluate_q6_form,
    evaluate_q7_fall_risk,
    load_task_rules,
)
from forestagent.schemas import ToolResult


def make_success_tool_result(
    tool_name: str,
    value: object,
    unit: str | None,
    confidence: float | None,
) -> ToolResult:
    return ToolResult(
        tool_name=tool_name,
        status="success",
        value=value,
        unit=unit,
        confidence=confidence,
        extra={"backend": "mock"},
        message="ok",
    )


class TaskRuleTests(unittest.TestCase):
    def test_load_task_rules_reads_default_config(self) -> None:
        rules = load_task_rules()

        self.assertEqual(rules.thresholds.q4_tilt.obvious_tilt_deg, 12.0)
        self.assertEqual(rules.labels.q6_form.slender.label, "偏细高")
        self.assertEqual(rules.labels.q7_fall_risk.uncertain.label, "不确定")

    def test_q4_tilt_uses_thresholds_from_custom_yaml(self) -> None:
        custom_rules = self._load_custom_rules(
            """
            thresholds:
              q4_tilt:
                obvious_tilt_deg: 5.0
              q5_quality:
                high_confidence_min: 0.85
                medium_confidence_min: 0.60
              q6_form:
                slender_ratio_tall_min: 70.0
                slender_ratio_stout_max: 45.0
              q7_fall_risk:
                high_tilt_deg_min: 15.0
                moderate_tilt_deg_min: 8.0
                high_slender_ratio_min: 75.0
                moderate_slender_ratio_min: 60.0
                small_dbh_cm_max: 20.0
                low_quality_confidence_max: 0.60
            labels:
              q4_tilt:
                obvious_tilt:
                  label: "Custom obvious"
                  message: "Custom obvious tilt."
                no_obvious_tilt:
                  label: "Custom stable"
                  message: "Custom stable tilt."
              q5_quality:
                high:
                  label: "High"
                  message: "High quality."
                medium:
                  label: "Medium"
                  message: "Medium quality."
                low:
                  label: "Low"
                  message: "Low quality."
              q6_form:
                slender:
                  label: "Slender"
                  message: "Slender form."
                stout:
                  label: "Stout"
                  message: "Stout form."
                balanced:
                  label: "Balanced"
                  message: "Balanced form."
              q7_fall_risk:
                high:
                  label: "High"
                  message: "High risk."
                medium:
                  label: "Medium"
                  message: "Medium risk."
                low:
                  label: "Low"
                  message: "Low risk."
                uncertain:
                  label: "Uncertain"
                  message: "Uncertain risk."
            """
        )

        conclusion = evaluate_q4_tilt(
            make_success_tool_result("estimate_tilt", value=6.0, unit="deg", confidence=0.8),
            rules=custom_rules,
        )

        self.assertEqual(conclusion.output_type, "label_with_value")
        self.assertEqual(conclusion.result.label, "Custom obvious")
        self.assertEqual(conclusion.message, "Custom obvious tilt.")
        self.assertEqual(conclusion.extra["obvious_tilt_deg_threshold"], 5.0)

    def test_q5_quality_maps_confidence_to_label(self) -> None:
        high = evaluate_q5_quality(
            make_success_tool_result("assess_quality", value="raw", unit=None, confidence=0.90)
        )
        medium = evaluate_q5_quality(
            make_success_tool_result("assess_quality", value="raw", unit=None, confidence=0.70)
        )
        low = evaluate_q5_quality(
            make_success_tool_result("assess_quality", value="raw", unit=None, confidence=0.30)
        )

        self.assertEqual(high.result.label, "高")
        self.assertEqual(medium.result.label, "中")
        self.assertEqual(low.result.label, "低")

    def test_q6_form_returns_slender_summary(self) -> None:
        conclusion = evaluate_q6_form(
            make_success_tool_result("estimate_dbh", value=20.0, unit="cm", confidence=0.9),
            make_success_tool_result("estimate_height", value=16.0, unit="m", confidence=0.9),
        )

        self.assertEqual(conclusion.output_type, "json_summary")
        self.assertEqual(conclusion.result.summary["label"], "偏细高")
        self.assertEqual(conclusion.result.summary["slenderness_ratio"], 80.0)

    def test_q6_form_returns_stout_summary(self) -> None:
        conclusion = evaluate_q6_form(
            make_success_tool_result("estimate_dbh", value=40.0, unit="cm", confidence=0.9),
            make_success_tool_result("estimate_height", value=12.0, unit="m", confidence=0.9),
        )

        self.assertEqual(conclusion.result.summary["label"], "矮壮")
        self.assertEqual(conclusion.result.summary["slenderness_ratio"], 30.0)

    def test_q7_fall_risk_returns_uncertain_for_low_quality(self) -> None:
        conclusion = evaluate_q7_fall_risk(
            make_success_tool_result("estimate_tilt", value=14.0, unit="deg", confidence=0.8),
            make_success_tool_result("estimate_dbh", value=18.0, unit="cm", confidence=0.9),
            make_success_tool_result("estimate_height", value=16.0, unit="m", confidence=0.9),
            make_success_tool_result("assess_quality", value="raw", unit=None, confidence=0.55),
        )

        self.assertEqual(conclusion.result.summary["label"], "不确定")
        self.assertTrue(conclusion.result.summary["signals"]["low_quality"])

    def test_q7_fall_risk_returns_high_when_geometry_signals_are_strong(self) -> None:
        conclusion = evaluate_q7_fall_risk(
            make_success_tool_result("estimate_tilt", value=16.0, unit="deg", confidence=0.8),
            make_success_tool_result("estimate_dbh", value=20.0, unit="cm", confidence=0.9),
            make_success_tool_result("estimate_height", value=16.0, unit="m", confidence=0.9),
            make_success_tool_result("assess_quality", value="raw", unit=None, confidence=0.85),
        )

        self.assertEqual(conclusion.result.summary["label"], "高")
        self.assertTrue(conclusion.result.summary["signals"]["tilt_high"])

    def test_task_rules_reject_failed_tool_results(self) -> None:
        failed_tilt = ToolResult(
            tool_name="estimate_tilt",
            status="failed",
            value=None,
            unit=None,
            confidence=None,
            extra={"backend": "mock"},
            message="tilt failed",
        )

        with self.assertRaises(TaskRuleError):
            evaluate_q4_tilt(failed_tilt)

    def _load_custom_rules(self, yaml_text: str):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", encoding="utf-8", delete=False
        ) as handle:
            handle.write(textwrap.dedent(yaml_text).strip() + "\n")
            temp_path = Path(handle.name)

        try:
            return load_task_rules(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()

