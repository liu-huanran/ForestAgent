"""Minimal q1/q2/q3 single-tree MVP entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from forestagent.backends.base_backend import BaseBackend
from forestagent.backends.direct_geometry_backend import DirectGeometryBackend
from forestagent.renderers import render_minimal_tree_report
from forestagent.schemas import PointCloudInput, ToolStatus, ToolResult
from forestagent.tools import estimate_crown_width, estimate_dbh, estimate_height
from forestagent.verbalizers import OllamaVerbalizerError, verbalize_json_summary_with_ollama

SupportedIntent = Literal["q1_dbh", "q2_height", "q3_crown_width", "tree_report_q123"]

_SUPPORTED_TASK_IDS = frozenset({"q1_dbh", "q2_height", "q3_crown_width", "tree_report_q123"})
_SUPPORTED_TASK_HINT = "支持的任务包括：q1_dbh、q2_height、q3_crown_width、tree_report_q123。"
_QUESTION_KEYWORDS: dict[SupportedIntent, tuple[str, ...]] = {
    "q1_dbh": ("胸径", "dbh", "diameter at breast height"),
    "q2_height": ("树高", "高度", "height"),
    "q3_crown_width": ("冠幅", "冠宽", "crown width"),
    "tree_report_q123": ("报告", "总结", "汇总", "summary", "report"),
}
_INTENT_TOOLS: dict[SupportedIntent, list[str]] = {
    "q1_dbh": ["estimate_dbh"],
    "q2_height": ["estimate_height"],
    "q3_crown_width": ["estimate_crown_width"],
    "tree_report_q123": ["estimate_dbh", "estimate_height", "estimate_crown_width"],
}
_TOOL_FUNCTIONS = {
    "estimate_dbh": estimate_dbh,
    "estimate_height": estimate_height,
    "estimate_crown_width": estimate_crown_width,
}
_SUMMARY_KEY_BY_TOOL = {
    "estimate_dbh": "dbh_cm",
    "estimate_height": "height_m",
    "estimate_crown_width": "crown_width_m",
}


class SingleTreeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str | None = None
    question: str | None = None
    resolved_intent: str | None = None


class SingleTreeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    format: str | None = None


class SingleTreeMetricValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float | None = None
    unit: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    status: ToolStatus
    message: str = ""


class SingleTreeJsonSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dbh_cm: SingleTreeMetricValue | None = None
    height_m: SingleTreeMetricValue | None = None
    crown_width_m: SingleTreeMetricValue | None = None


class SingleTreeAnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: SingleTreeRequest
    input: SingleTreeInput
    tool_results: dict[str, dict[str, Any]]
    json_summary: SingleTreeJsonSummary
    report_text: str
    status: ToolStatus
    message: str = ""


def run_single_tree_analysis(
    point_cloud: str | PointCloudInput,
    *,
    task_id: str | None = None,
    question: str | None = None,
    backend: BaseBackend | None = None,
    config_path: str | Path | None = None,
    use_local_llm_report: bool = False,
    ollama_model: str = "qwen3:1.7b",
    ollama_base_url: str = "http://localhost:11434/api",
    ollama_timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Run the minimal q1/q2/q3-only single-tree MVP."""

    normalized_input = _normalize_point_cloud_input(point_cloud)
    request = SingleTreeRequest(task_id=task_id, question=question, resolved_intent=None)
    empty_summary = SingleTreeJsonSummary()

    if isinstance(normalized_input, str):
        return _build_failed_response(
            request=request,
            point_cloud_input=SingleTreeInput(path=str(point_cloud), format=None),
            tool_results={},
            summary=empty_summary,
            message=normalized_input,
            use_local_llm_report=use_local_llm_report,
        )

    request.resolved_intent = _resolve_intent(task_id=task_id, question=question)
    if request.resolved_intent is None:
        return _build_failed_response(
            request=request,
            point_cloud_input=SingleTreeInput(
                path=normalized_input.path,
                format=normalized_input.format,
            ),
            tool_results={},
            summary=empty_summary,
            message=_resolve_failure_message(task_id=task_id, question=question),
            use_local_llm_report=use_local_llm_report,
        )

    resolved_intent = request.resolved_intent
    active_backend = backend or DirectGeometryBackend(config_path=config_path)
    tool_results: dict[str, ToolResult] = {}
    summary = SingleTreeJsonSummary()

    for tool_name in _INTENT_TOOLS[resolved_intent]:
        tool_result = _TOOL_FUNCTIONS[tool_name](normalized_input, active_backend)
        tool_results[tool_name] = tool_result
        setattr(summary, _SUMMARY_KEY_BY_TOOL[tool_name], _metric_from_tool_result(tool_result))
        if tool_result.status != "success":
            return _build_failed_response(
                request=request,
                point_cloud_input=SingleTreeInput(
                    path=normalized_input.path,
                    format=normalized_input.format,
                ),
                tool_results={name: result.model_dump() for name, result in tool_results.items()},
                summary=summary,
                message=tool_result.message,
                use_local_llm_report=use_local_llm_report,
            )

    template_report_text = render_minimal_tree_report(
        resolved_intent=resolved_intent,
        json_summary=summary.model_dump(),
        status="success",
        message="",
    )
    success_message = _build_success_message(resolved_intent, tool_results)
    report_text_llm: str | None = None
    if use_local_llm_report:
        try:
            report_text_llm = verbalize_json_summary_with_ollama(
                summary.model_dump(),
                model=ollama_model,
                base_url=ollama_base_url,
                timeout_seconds=ollama_timeout_seconds,
            )
        except OllamaVerbalizerError as exc:
            success_message = (
                f"{success_message} "
                f"Local Ollama verbalizer fallback happened: {exc}"
            )
    response = SingleTreeAnalysisResponse(
        request=request,
        input=SingleTreeInput(path=normalized_input.path, format=normalized_input.format),
        tool_results={name: result.model_dump() for name, result in tool_results.items()},
        json_summary=summary,
        report_text=template_report_text,
        status="success",
        message=success_message,
    )
    payload = response.model_dump()
    if use_local_llm_report:
        payload["report_text_template"] = template_report_text
        if report_text_llm is not None:
            payload["report_text_llm"] = report_text_llm
    return payload


def _normalize_point_cloud_input(point_cloud: str | PointCloudInput) -> PointCloudInput | str:
    if isinstance(point_cloud, PointCloudInput):
        return point_cloud
    if not isinstance(point_cloud, str):
        return "点云输入必须是文件路径字符串或 PointCloudInput。"

    point_cloud_path = Path(point_cloud)
    suffix = point_cloud_path.suffix.lower().lstrip(".")
    if not suffix:
        return f"点云文件缺少后缀，无法推断 format：{point_cloud_path}"
    return PointCloudInput(path=str(point_cloud_path), format=suffix)


def _resolve_intent(task_id: str | None, question: str | None) -> SupportedIntent | None:
    if task_id is not None:
        if task_id in _SUPPORTED_TASK_IDS:
            return task_id  # type: ignore[return-value]
        return None
    if question is None or not question.strip():
        return None

    normalized = question.strip().lower()
    report_keywords = _QUESTION_KEYWORDS["tree_report_q123"]
    if any(keyword in normalized for keyword in report_keywords):
        return "tree_report_q123"

    matched_intents = [
        intent
        for intent in ("q1_dbh", "q2_height", "q3_crown_width")
        if any(keyword in normalized for keyword in _QUESTION_KEYWORDS[intent])
    ]
    if len(matched_intents) == 1:
        return matched_intents[0]  # type: ignore[return-value]
    return None


def _resolve_failure_message(task_id: str | None, question: str | None) -> str:
    if task_id is not None and task_id not in _SUPPORTED_TASK_IDS:
        return f"不支持的 task_id：{task_id}。{_SUPPORTED_TASK_HINT}"
    if task_id is None and (question is None or not question.strip()):
        return f"至少需要提供 task_id 或 question。{_SUPPORTED_TASK_HINT}"
    if question is not None and question.strip():
        return f"无法从问题中唯一解析任务意图：{question}。{_SUPPORTED_TASK_HINT}"
    return _SUPPORTED_TASK_HINT


def _metric_from_tool_result(tool_result: ToolResult) -> SingleTreeMetricValue:
    numeric_value = (
        float(tool_result.value)
        if isinstance(tool_result.value, (int, float))
        else None
    )
    return SingleTreeMetricValue(
        value=numeric_value,
        unit=tool_result.unit,
        confidence=tool_result.confidence,
        status=tool_result.status,
        message=tool_result.message,
    )


def _build_success_message(
    resolved_intent: SupportedIntent,
    tool_results: dict[str, ToolResult],
) -> str:
    if resolved_intent == "tree_report_q123":
        return "q1/q2/q3 单木分析完成，简短报告已生成。"
    first_result = next(iter(tool_results.values()))
    return first_result.message


def _build_failed_response(
    *,
    request: SingleTreeRequest,
    point_cloud_input: SingleTreeInput,
    tool_results: dict[str, dict[str, Any]],
    summary: SingleTreeJsonSummary,
    message: str,
    use_local_llm_report: bool,
) -> dict[str, Any]:
    template_report_text = render_minimal_tree_report(
        resolved_intent=request.resolved_intent,
        json_summary=summary.model_dump(),
        status="failed",
        message=message,
    )
    response = SingleTreeAnalysisResponse(
        request=request,
        input=point_cloud_input,
        tool_results=tool_results,
        json_summary=summary,
        report_text=template_report_text,
        status="failed",
        message=message,
    )
    payload = response.model_dump()
    if use_local_llm_report:
        payload["report_text_template"] = template_report_text
    return payload
