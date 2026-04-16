"""Template-based answer rendering for the first MVP."""

from __future__ import annotations

from collections.abc import Mapping

from forestagent.logic import TaskRuleConclusion
from forestagent.schemas import TaskResult, ToolResult


def render_task_answer(task_result: TaskResult) -> str:
    """Render a short human-readable answer from a structured task result."""

    if task_result.status != "success":
        return f"{task_result.task_name}：任务失败。原因：{task_result.message}"

    result = task_result.result
    if task_result.output_type == "scalar":
        return (
            f"{task_result.task_name}：{result.value:.2f} {result.unit}"
            f"（置信度 {format_confidence(result.confidence)}）。"
        )
    if task_result.output_type == "label_with_value":
        return (
            f"{task_result.task_name}：{result.label}，数值 {result.value:.2f} {result.unit}"
            f"（置信度 {format_confidence(result.confidence)}）。"
        )
    if task_result.output_type == "quality_label":
        return (
            f"{task_result.task_name}：{result.label}"
            f"（置信度 {format_confidence(result.confidence)}）。"
        )
    if task_result.output_type == "json_summary":
        label = result.summary.get("label", "未命名结论")
        message = result.summary.get("message", task_result.message)
        return f"{task_result.task_name}：{label}。{message}"
    if task_result.output_type == "report_text":
        return result.text
    raise ValueError(f"Unsupported output type: {task_result.output_type}")


def render_brief_report(
    task_name: str,
    tool_results: Mapping[str, ToolResult],
    derived_conclusions: Mapping[str, TaskRuleConclusion],
) -> str:
    """Render a short single-tree report from tool outputs and derived summaries."""

    dbh = tool_results["estimate_dbh"]
    height = tool_results["estimate_height"]
    crown_width = tool_results["estimate_crown_width"]
    tilt = tool_results["estimate_tilt"]
    quality = tool_results["assess_quality"]

    tilt_conclusion = derived_conclusions["q4_tilt"]
    quality_conclusion = derived_conclusions["q5_quality"]
    form_conclusion = derived_conclusions["q6_form"]
    risk_conclusion = derived_conclusions["q7_fall_risk"]

    return (
        f"{task_name}：胸径约 {dbh.value:.2f} {dbh.unit}，树高约 {height.value:.2f} {height.unit}，"
        f"冠幅约 {crown_width.value:.2f} {crown_width.unit}。"
        f"倾斜判断为“{tilt_conclusion.result.label}”"
        f"（{tilt.value:.2f} {tilt.unit}，置信度 {format_confidence(tilt.confidence)}）；"
        f"点云质量为“{quality_conclusion.result.label}”"
        f"（原始质量值 {quality.value}，置信度 {format_confidence(quality.confidence)}）。"
        f"形态判断为“{form_conclusion.result.summary['label']}”；"
        f"初步倒伏风险为“{risk_conclusion.result.summary['label']}”。"
    )


def format_confidence(confidence: float | None) -> str:
    """Format confidence in a compact and stable way."""

    if confidence is None:
        return "N/A"
    return f"{confidence:.2f}"

