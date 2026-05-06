"""Deterministic ForestAgent v2 planner."""

from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from forestagent.tools.registry import ToolRegistry, build_default_tool_registry

PlannerStatus = Literal["planned", "unsupported", "unavailable"]


class ExecutionStep(BaseModel):
    """One planned tool invocation."""

    model_config = ConfigDict(extra="forbid")

    step_id: int
    tool_name: str
    intent_name: str
    required_inputs: list[str] = Field(default_factory=list)


class PlannerResult(BaseModel):
    """Output from the deterministic v2 planner."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(default_factory=lambda: f"request-{uuid4().hex}")
    user_question: str
    status: PlannerStatus
    resolved_intent: str | None = None
    selected_tools: list[str] = Field(default_factory=list)
    execution_plan: list[ExecutionStep] = Field(default_factory=list)
    planner_reason: str = ""
    unsupported_reason: str | None = None


class DeterministicPlanner:
    """Keyword-based planner for the v2 mock skeleton."""

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry or build_default_tool_registry()

    def plan(self, user_question: str) -> PlannerResult:
        normalized = user_question.strip().lower()
        if not normalized:
            return PlannerResult(
                user_question=user_question,
                status="unsupported",
                resolved_intent=None,
                planner_reason="Rule route: empty question cannot map to a supported task.",
                unsupported_reason="empty_question",
            )

        unsupported_reason = self._unsupported_reason(normalized)
        if unsupported_reason is not None:
            return PlannerResult(
                user_question=user_question,
                status="unsupported",
                resolved_intent=None,
                selected_tools=[],
                execution_plan=[],
                planner_reason=(
                    "Rule route: unsupported keyword category matched: "
                    f"{unsupported_reason}."
                ),
                unsupported_reason=unsupported_reason,
            )

        if self._contains(
            normalized,
            ("\u76f8\u4f3c", "similar", "nearest neighbor", "\u8fd1\u90bb"),
        ):
            return self._build_tool_plan(
                user_question=user_question,
                status="unavailable",
                resolved_intent="similar_tree_retrieval",
                tool_names=["similar_tree_retrieval"],
                reason=(
                    "Rule route: similar-tree retrieval requested, but Phase 1/2 "
                    "does not enable offline cache execution."
                ),
                unsupported_reason="embedding_cache_not_configured",
            )

        if self._contains(
            normalized,
            (
                "\u5f02\u5e38\u6837\u672c",
                "outlier",
                "\u79bb\u7fa4",
                "\u5f02\u5e38\u70b9",
            ),
        ):
            return self._build_tool_plan(
                user_question=user_question,
                status="unavailable",
                resolved_intent="embedding_outlier_lookup",
                tool_names=["embedding_outlier_lookup"],
                reason=(
                    "Rule route: embedding outlier lookup requested, but Phase 1/2 "
                    "does not enable offline analysis-cache execution."
                ),
                unsupported_reason="analysis_cache_not_configured",
            )

        if self._contains(
            normalized,
            (
                "\u7efc\u5408\u62a5\u544a",
                "\u6574\u4f53\u60c5\u51b5",
                "\u62a5\u544a",
                "\u603b\u7ed3",
                "\u6c47\u603b",
                "tree report",
                "summary",
                "report",
            ),
        ):
            return self._build_tool_plan(
                user_question=user_question,
                status="planned",
                resolved_intent="tree_report_q123",
                tool_names=["q1_dbh", "q2_height", "q3_crown_width", "tree_report_q123"],
                reason=(
                    "Rule route: comprehensive report uses only q1/q2/q3 tools "
                    "and report composition over evidence."
                ),
            )

        matched = []
        if self._contains(
            normalized,
            ("\u80f8\u5f84", "dbh", "diameter at breast height"),
        ):
            matched.append(
                (
                    "q1_dbh",
                    "q1_dbh",
                    "Rule route: DBH keyword matched; select frozen q1 geometry tool.",
                )
            )
        if self._contains(normalized, ("\u6811\u9ad8", "\u9ad8\u5ea6", "height")):
            matched.append(
                (
                    "q2_height",
                    "q2_height",
                    "Rule route: height keyword matched; select frozen q2 geometry tool.",
                )
            )
        if self._contains(
            normalized,
            ("\u51a0\u5e45", "\u51a0\u5bbd", "crown width", "crown"),
        ):
            matched.append(
                (
                    "q3_crown_width",
                    "q3_crown_width",
                    (
                        "Rule route: crown-width keyword matched; select frozen "
                        "q3 geometry tool."
                    ),
                )
            )

        if len(matched) == 1:
            intent, tool_name, reason = matched[0]
            return self._build_tool_plan(
                user_question=user_question,
                status="planned",
                resolved_intent=intent,
                tool_names=[tool_name],
                reason=reason,
            )
        if len(matched) > 1:
            return PlannerResult(
                user_question=user_question,
                status="unsupported",
                planner_reason=(
                    "Rule route: multiple measurement intents matched. Ask for "
                    "a comprehensive report or one specific metric."
                ),
                unsupported_reason="ambiguous_measurement_intent",
            )

        return PlannerResult(
            user_question=user_question,
            status="unsupported",
            planner_reason=(
                "Rule route: no supported q1/q2/q3/report/retrieval keyword matched."
            ),
            unsupported_reason="no_supported_intent",
        )

    def _build_tool_plan(
        self,
        *,
        user_question: str,
        status: PlannerStatus,
        resolved_intent: str,
        tool_names: list[str],
        reason: str,
        unsupported_reason: str | None = None,
    ) -> PlannerResult:
        steps: list[ExecutionStep] = []
        for index, tool_name in enumerate(tool_names, start=1):
            metadata = self.registry.get_tool(tool_name)
            steps.append(
                ExecutionStep(
                    step_id=index,
                    tool_name=tool_name,
                    intent_name=resolved_intent,
                    required_inputs=metadata.required_inputs,
                )
            )
        return PlannerResult(
            user_question=user_question,
            status=status,
            resolved_intent=resolved_intent,
            selected_tools=tool_names,
            execution_plan=steps,
            planner_reason=reason,
            unsupported_reason=unsupported_reason,
        )

    @staticmethod
    def _contains(text: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword in text for keyword in keywords)

    @staticmethod
    def _unsupported_reason(text: str) -> str | None:
        unsupported_keywords = {
            "\u5065\u5eb7": "health_status_without_evidence",
            "\u5371\u9669\u6728": "danger_tree_without_evidence",
            "\u5371\u9669": "danger_tree_without_evidence",
            "\u75c5\u866b\u5bb3": "disease_or_pest_without_evidence",
            "\u866b\u5bb3": "disease_or_pest_without_evidence",
            "\u75c5\u5bb3": "disease_or_pest_without_evidence",
            "\u957f\u52bf": "growth_quality_without_evidence",
            "\u957f\u5f97\u597d": "subjective_growth_quality_without_evidence",
            "\u597d\u4e0d\u597d": "subjective_growth_quality_without_evidence",
            "\u780d\u4f10": "cutting_decision_without_evidence",
            "\u780d\u6389": "cutting_decision_without_evidence",
            "\u9700\u8981\u780d": "cutting_decision_without_evidence",
            "q4": "q4_to_q8_out_of_scope",
            "q5": "q4_to_q8_out_of_scope",
            "q6": "q4_to_q8_out_of_scope",
            "q7": "q4_to_q8_out_of_scope",
            "q8": "q4_to_q8_out_of_scope",
            "\u503e\u659c": "q4_to_q8_out_of_scope",
            "tilt": "q4_to_q8_out_of_scope",
            "\u70b9\u4e91\u8d28\u91cf": "q4_to_q8_out_of_scope",
            "\u8d28\u91cf": "q4_to_q8_out_of_scope",
            "\u5012\u4f0f": "q4_to_q8_out_of_scope",
            "\u98ce\u9669": "q4_to_q8_out_of_scope",
        }
        for keyword, reason in unsupported_keywords.items():
            if keyword in text:
                return reason
        return None
