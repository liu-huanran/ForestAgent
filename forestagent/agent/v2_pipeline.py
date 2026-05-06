"""Minimum ForestAgent v2 mock pipeline."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from forestagent.agent.executor import ExecutionResult, MockToolExecutor
from forestagent.agent.planner import DeterministicPlanner, PlannerResult
from forestagent.agent.response_builder import ResponseBuilder
from forestagent.evidence.packet import EvidencePacket
from forestagent.tools.registry import ToolRegistry, build_default_tool_registry
from forestagent.tools.schema import ToolResult
from forestagent.tools.trace import TraceRecord


class AgentV2PipelineResult(BaseModel):
    """Full output from the v2 mock pipeline."""

    model_config = ConfigDict(extra="forbid")

    planner_result: PlannerResult
    tool_results: list[ToolResult] = Field(default_factory=list)
    evidence_packet: EvidencePacket
    structured_summary: dict[str, Any]
    report_text: str
    trace: TraceRecord


class AgentV2Pipeline:
    """Orchestrates deterministic planning, mock execution, and response building."""

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry or build_default_tool_registry()
        self.planner = DeterministicPlanner(self.registry)
        self.executor = MockToolExecutor(self.registry)
        self.response_builder = ResponseBuilder()

    def run(
        self,
        user_question: str,
        input_context: dict[str, Any] | None = None,
    ) -> AgentV2PipelineResult:
        context = {"point_cloud_path": "mock://single-tree"}
        if input_context:
            context.update(input_context)

        planner_result = self.planner.plan(user_question)
        execution_result: ExecutionResult = self.executor.execute(planner_result, context)
        response = self.response_builder.build(
            user_question=user_question,
            planner_result=planner_result,
            evidence_packet=execution_result.evidence_packet,
            tool_results=execution_result.tool_results,
        )
        trace = execution_result.trace.model_copy(
            update={
                "structured_summary": response.structured_summary,
                "report_text": response.report_text,
                "warnings": list(
                    dict.fromkeys(execution_result.trace.warnings + response.warnings)
                ),
            }
        )
        return AgentV2PipelineResult(
            planner_result=planner_result,
            tool_results=execution_result.tool_results,
            evidence_packet=execution_result.evidence_packet,
            structured_summary=response.structured_summary,
            report_text=response.report_text,
            trace=trace,
        )


def run_agent_v2_mock(
    user_question: str,
    input_context: dict[str, Any] | None = None,
) -> AgentV2PipelineResult:
    """Run the v2 minimum loop with deterministic mock tools."""

    return AgentV2Pipeline().run(user_question, input_context=input_context)
