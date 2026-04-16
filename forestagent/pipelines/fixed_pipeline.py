"""Fixed task pipeline for the first ForestAgent MVP."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from forestagent.backends.base_backend import BaseBackend
from forestagent.config_loader import load_yaml_mapping
from forestagent.logic import (
    TaskRuleError,
    evaluate_q4_tilt,
    evaluate_q5_quality,
    evaluate_q6_form,
    evaluate_q7_fall_risk,
)
from forestagent.renderers import render_brief_report
from forestagent.schemas import (
    PointCloudInput,
    ReportTextPayload,
    ScalarPayload,
    TaskResult,
    ToolResult,
)
from forestagent.tools import (
    assess_quality,
    estimate_crown_width,
    estimate_dbh,
    estimate_height,
    estimate_tilt,
)

_DEFAULT_TASK_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "task_registry.yaml"
)


class TaskRegistryEntry(BaseModel):
    """Task registry entry loaded from config."""

    model_config = ConfigDict(extra="forbid")

    task_name: str = Field(min_length=1)
    output_type: str = Field(min_length=1)
    required_tools: list[str]
    description: str = ""


class FixedPipeline:
    """Fixed task pipeline that never delegates tool selection to the LLM."""

    _TOOL_FUNCTIONS = {
        "estimate_dbh": estimate_dbh,
        "estimate_height": estimate_height,
        "estimate_crown_width": estimate_crown_width,
        "estimate_tilt": estimate_tilt,
        "assess_quality": assess_quality,
    }

    def __init__(
        self,
        backend: BaseBackend,
        task_registry_path: str | Path | None = None,
    ) -> None:
        self._backend = backend
        self._task_registry = self._load_task_registry(
            task_registry_path or _DEFAULT_TASK_REGISTRY_PATH
        )

    def run(self, task_id: str, point_cloud: PointCloudInput) -> TaskResult:
        """Run one fixed task against a single-tree point cloud input."""

        if task_id not in self._task_registry:
            raise KeyError(f"Unknown task_id: {task_id}")

        task_entry = self._task_registry[task_id]
        tool_results = self._execute_tools(task_entry.required_tools, point_cloud)
        failed_result = next(
            (result for result in tool_results.values() if result.status != "success"),
            None,
        )
        if failed_result is not None:
            return self._build_failed_task_result(
                task_id=task_id,
                task_name=task_entry.task_name,
                output_type=task_entry.output_type,
                tool_results=tool_results,
                message=failed_result.message,
            )

        try:
            if task_id == "q1_dbh":
                return self._build_scalar_task_result(
                    task_id, task_entry, tool_results["estimate_dbh"]
                )
            if task_id == "q2_height":
                return self._build_scalar_task_result(
                    task_id, task_entry, tool_results["estimate_height"]
                )
            if task_id == "q3_crown_width":
                return self._build_scalar_task_result(
                    task_id, task_entry, tool_results["estimate_crown_width"]
                )
            if task_id == "q4_tilt":
                conclusion = evaluate_q4_tilt(tool_results["estimate_tilt"])
                return self._build_task_result_from_conclusion(task_id, task_entry, conclusion)
            if task_id == "q5_quality":
                conclusion = evaluate_q5_quality(tool_results["assess_quality"])
                return self._build_task_result_from_conclusion(task_id, task_entry, conclusion)
            if task_id == "q6_form":
                conclusion = evaluate_q6_form(
                    tool_results["estimate_dbh"],
                    tool_results["estimate_height"],
                )
                return self._build_task_result_from_conclusion(task_id, task_entry, conclusion)
            if task_id == "q7_fall_risk":
                conclusion = evaluate_q7_fall_risk(
                    tool_results["estimate_tilt"],
                    tool_results["estimate_dbh"],
                    tool_results["estimate_height"],
                    tool_results["assess_quality"],
                )
                return self._build_task_result_from_conclusion(task_id, task_entry, conclusion)
            if task_id == "q8_brief_report":
                return self._build_report_task_result(task_id, task_entry, tool_results)
        except TaskRuleError as exc:
            return self._build_failed_task_result(
                task_id=task_id,
                task_name=task_entry.task_name,
                output_type=task_entry.output_type,
                tool_results=tool_results,
                message=f"{task_id} rule evaluation failed: {exc}",
            )

        raise NotImplementedError(f"Task {task_id} is not implemented in FixedPipeline.")

    @classmethod
    def _load_task_registry(cls, path: str | Path) -> dict[str, TaskRegistryEntry]:
        raw_registry = load_yaml_mapping(path)
        return {
            task_id: TaskRegistryEntry.model_validate(entry)
            for task_id, entry in raw_registry.items()
        }

    def _execute_tools(
        self, required_tools: list[str], point_cloud: PointCloudInput
    ) -> dict[str, ToolResult]:
        tool_results: dict[str, ToolResult] = {}
        for tool_name in required_tools:
            if tool_name not in self._TOOL_FUNCTIONS:
                raise KeyError(f"Unknown tool name in registry: {tool_name}")
            tool_results[tool_name] = self._TOOL_FUNCTIONS[tool_name](
                point_cloud, self._backend
            )
            if tool_results[tool_name].status != "success":
                break
        return tool_results

    @staticmethod
    def _build_scalar_task_result(
        task_id: str,
        task_entry: TaskRegistryEntry,
        tool_result: ToolResult,
    ) -> TaskResult:
        if tool_result.value is None or not isinstance(tool_result.value, (int, float)):
            raise TaskRuleError(f"{tool_result.tool_name} must provide a numeric scalar value.")
        if tool_result.unit is None:
            raise TaskRuleError(f"{tool_result.tool_name} must provide a scalar unit.")

        return TaskResult(
            task_id=task_id,
            task_name=task_entry.task_name,
            status="success",
            output_type=task_entry.output_type,
            result=ScalarPayload(
                value=float(tool_result.value),
                unit=tool_result.unit,
                confidence=tool_result.confidence,
            ),
            extra={"source_tool": tool_result.tool_name},
            message=tool_result.message,
        )

    @staticmethod
    def _build_task_result_from_conclusion(
        task_id: str,
        task_entry: TaskRegistryEntry,
        conclusion,
    ) -> TaskResult:
        return TaskResult(
            task_id=task_id,
            task_name=task_entry.task_name,
            status="success",
            output_type=task_entry.output_type,
            result=conclusion.result,
            extra=conclusion.extra,
            message=conclusion.message,
        )

    def _build_report_task_result(
        self,
        task_id: str,
        task_entry: TaskRegistryEntry,
        tool_results: dict[str, ToolResult],
    ) -> TaskResult:
        derived = {
            "q4_tilt": evaluate_q4_tilt(tool_results["estimate_tilt"]),
            "q5_quality": evaluate_q5_quality(tool_results["assess_quality"]),
            "q6_form": evaluate_q6_form(
                tool_results["estimate_dbh"],
                tool_results["estimate_height"],
            ),
            "q7_fall_risk": evaluate_q7_fall_risk(
                tool_results["estimate_tilt"],
                tool_results["estimate_dbh"],
                tool_results["estimate_height"],
                tool_results["assess_quality"],
            ),
        }
        report_text = render_brief_report(task_entry.task_name, tool_results, derived)

        return TaskResult(
            task_id=task_id,
            task_name=task_entry.task_name,
            status="success",
            output_type=task_entry.output_type,
            result=ReportTextPayload(text=report_text),
            extra={"derived_summaries": self._serialize_conclusions(derived)},
            message="模板化单木报告已生成。",
        )

    @staticmethod
    def _serialize_conclusions(conclusions: dict[str, object]) -> dict[str, dict[str, object]]:
        serialized: dict[str, dict[str, object]] = {}
        for task_id, conclusion in conclusions.items():
            serialized[task_id] = {
                "output_type": conclusion.output_type,
                "result": conclusion.result.model_dump(),
                "message": conclusion.message,
                "extra": conclusion.extra,
            }
        return serialized

    @staticmethod
    def _build_failed_task_result(
        task_id: str,
        task_name: str,
        output_type: str,
        tool_results: dict[str, ToolResult],
        message: str,
    ) -> TaskResult:
        failed_tools = [
            tool_name
            for tool_name, tool_result in tool_results.items()
            if tool_result.status != "success"
        ]
        return TaskResult(
            task_id=task_id,
            task_name=task_name,
            status="failed",
            output_type=output_type,
            result=None,
            extra={
                "partial_results": {
                    tool_name: tool_result.model_dump()
                    for tool_name, tool_result in tool_results.items()
                },
                "failed_tools": failed_tools,
            },
            message=message,
        )
