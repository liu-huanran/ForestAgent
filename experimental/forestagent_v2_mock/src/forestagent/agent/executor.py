"""Mock executor for the ForestAgent v2 phase-1/2 skeleton."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from forestagent.agent.planner import PlannerResult
from forestagent.evidence.builder import evidence_items_from_tool_result
from forestagent.evidence.packet import EvidencePacket
from forestagent.tools.base import ToolExecutionContext
from forestagent.tools.registry import ToolRegistry, build_default_tool_registry
from forestagent.tools.schema import ToolOutput, ToolResult
from forestagent.tools.trace import TraceRecord

_DEFAULT_MOCK_MEASUREMENTS = {
    "q1_dbh": {
        "value": 23.4,
        "unit": "cm",
        "evidence_id": "measurement.dbh",
        "explanation": "DBH estimated by v2 mock q1 geometry tool.",
    },
    "q2_height": {
        "value": 12.5,
        "unit": "m",
        "evidence_id": "measurement.height",
        "explanation": "Height estimated by v2 mock q2 geometry tool.",
    },
    "q3_crown_width": {
        "value": 4.2,
        "unit": "m",
        "evidence_id": "measurement.crown_width",
        "explanation": "Crown width estimated by v2 mock q3 geometry tool.",
    },
}


class ExecutionResult(BaseModel):
    """Output from the v2 executor."""

    model_config = ConfigDict(extra="forbid")

    tool_results: list[ToolResult] = Field(default_factory=list)
    evidence_packet: EvidencePacket
    trace: TraceRecord


class MockToolExecutor:
    """Deterministic mock executor used by the v2 minimum loop."""

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry or build_default_tool_registry()

    def execute(
        self,
        planner_result: PlannerResult,
        input_context: ToolExecutionContext | dict[str, Any] | None = None,
    ) -> ExecutionResult:
        context = self._normalize_context(input_context)
        evidence_packet = EvidencePacket(request_id=planner_result.request_id)
        tool_results: list[ToolResult] = []
        warnings: list[str] = []
        errors: list[str] = []

        for step in planner_result.execution_plan:
            started = perf_counter()
            try:
                metadata = self.registry.get_tool(step.tool_name)
                result = self._run_step(step.tool_name, metadata.required_inputs, context)
            except Exception as exc:
                result = ToolResult(
                    tool_name=step.tool_name,
                    status="failed",
                    error_message=f"v2 executor failed before tool completion: {exc}",
                    metadata={"executor": "MockToolExecutor"},
                    runtime_ms=(perf_counter() - started) * 1000.0,
                )
            else:
                result.runtime_ms = (perf_counter() - started) * 1000.0

            tool_results.append(result)
            warnings.extend(result.warnings)
            if result.status in {"failed", "unavailable"} and result.error_message:
                errors.append(result.error_message)
            for item in evidence_items_from_tool_result(result):
                evidence_packet.add_item(item)

        trace = TraceRecord(
            request_id=planner_result.request_id,
            user_question=planner_result.user_question,
            resolved_intent=planner_result.resolved_intent,
            planner_reason=planner_result.planner_reason,
            selected_tools=planner_result.selected_tools,
            execution_plan=[step.model_dump() for step in planner_result.execution_plan],
            tool_results=[result.model_dump() for result in tool_results],
            evidence_packet=evidence_packet.model_dump(),
            warnings=warnings,
            errors=errors,
        )
        return ExecutionResult(
            tool_results=tool_results,
            evidence_packet=evidence_packet,
            trace=trace,
        )

    def _run_step(
        self,
        tool_name: str,
        required_inputs: list[str],
        context: ToolExecutionContext,
    ) -> ToolResult:
        if tool_name == "tree_report_q123":
            return ToolResult(
                tool_name=tool_name,
                status="skipped",
                warnings=["tree_report_q123 is composed by the v2 response builder."],
                metadata={"handled_by": "response_builder"},
            )

        missing_inputs = [
            name for name in required_inputs if getattr(context, name, None) in {None, ""}
        ]
        if missing_inputs:
            return ToolResult(
                tool_name=tool_name,
                status="unavailable",
                error_message=(
                    f"Required input(s) missing for {tool_name}: "
                    f"{', '.join(missing_inputs)}."
                ),
                metadata={"missing_inputs": missing_inputs},
            )

        if tool_name in {"uni3d_embedding_lookup", "similar_tree_retrieval", "embedding_outlier_lookup"}:
            return ToolResult(
                tool_name=tool_name,
                status="unavailable",
                error_message=(
                    f"{tool_name} is registered as an offline cache tool, but Phase 1/2 "
                    "only provides an unavailable stub and never runs Uni3D forward."
                ),
                metadata={"offline_only": True, "phase": "v2_phase1_2"},
            )

        if tool_name in context.mock_failures:
            return ToolResult(
                tool_name=tool_name,
                status="failed",
                error_message=context.mock_failures[tool_name],
                metadata={"mock": True},
            )

        if tool_name in _DEFAULT_MOCK_MEASUREMENTS:
            output_payload = dict(_DEFAULT_MOCK_MEASUREMENTS[tool_name])
            override = context.mock_values.get(tool_name)
            if isinstance(override, dict):
                output_payload.update(override)
            elif isinstance(override, (int, float)):
                output_payload["value"] = float(override)
            return ToolResult(
                tool_name=tool_name,
                status="ok",
                output=ToolOutput(
                    value=output_payload["value"],
                    unit=output_payload["unit"],
                    evidence_id=output_payload["evidence_id"],
                    evidence_type="measurement",
                    explanation=output_payload["explanation"],
                    metadata={"mock": True},
                ),
                metadata={"mock": True},
            )

        return ToolResult(
            tool_name=tool_name,
            status="unavailable",
            error_message=f"No mock implementation is registered for {tool_name}.",
            metadata={"mock": True},
        )

    @staticmethod
    def _normalize_context(
        input_context: ToolExecutionContext | dict[str, Any] | None,
    ) -> ToolExecutionContext:
        if isinstance(input_context, ToolExecutionContext):
            return input_context
        return ToolExecutionContext.model_validate(input_context or {})
