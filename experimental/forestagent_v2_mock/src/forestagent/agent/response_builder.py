"""Evidence-only response builder for ForestAgent v2."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from forestagent.agent.planner import PlannerResult
from forestagent.evidence.packet import EvidenceItem, EvidencePacket
from forestagent.tools.schema import ToolResult


class ResponseBuildResult(BaseModel):
    """Structured and textual response built from evidence only."""

    model_config = ConfigDict(extra="forbid")

    structured_summary: dict[str, Any]
    report_text: str
    warnings: list[str] = Field(default_factory=list)


class ResponseBuilder:
    """Template response builder constrained by EvidencePacket."""

    def build(
        self,
        *,
        user_question: str,
        planner_result: PlannerResult,
        evidence_packet: EvidencePacket,
        tool_results: list[ToolResult],
    ) -> ResponseBuildResult:
        del user_question
        tool_status = {result.tool_name: result.status for result in tool_results}
        warnings = [
            item.explanation
            for item in evidence_packet.items
            if item.status in {"failed", "unavailable"}
        ]

        if planner_result.status == "unsupported":
            summary = {
                "intent": planner_result.resolved_intent,
                "status": "unsupported",
                "measurements": {},
                "retrievals": [],
                "warnings": warnings,
                "unsupported": {"reason": planner_result.unsupported_reason},
                "evidence_refs": {},
                "tool_status": tool_status,
            }
            return ResponseBuildResult(
                structured_summary=summary,
                report_text=(
                    "\u5f53\u524d\u65e0\u6cd5\u56de\u7b54\u8be5\u95ee\u9898\uff1a"
                    "\u7f3a\u5c11\u53d7\u652f\u6301\u7684\u5de5\u5177\u8bc1\u636e\u3002"
                    "\u672c\u9636\u6bb5\u53ea\u652f\u6301\u80f8\u5f84\u3001\u6811\u9ad8\u3001"
                    "\u51a0\u5e45\u548c q1/q2/q3 \u7efc\u5408\u62a5\u544a\u3002"
                ),
                warnings=warnings,
            )

        if planner_result.status == "unavailable":
            summary = {
                "intent": planner_result.resolved_intent,
                "status": "unavailable",
                "measurements": {},
                "retrievals": [],
                "warnings": warnings,
                "unsupported": {"reason": planner_result.unsupported_reason},
                "evidence_refs": {},
                "tool_status": tool_status,
            }
            return ResponseBuildResult(
                structured_summary=summary,
                report_text=(
                    "\u5f53\u524d\u65e0\u6cd5\u5b8c\u6210\u8be5\u79bb\u7ebf\u67e5\u8be2\uff1a"
                    "\u6240\u9700\u7f13\u5b58\u5de5\u5177\u5728 v2 Phase 1/2 "
                    "\u4e2d\u672a\u542f\u7528\u3002\u672c\u7248\u4e0d\u4f1a\u5b9e\u65f6"
                    "\u8fd0\u884c Uni3D\uff0c\u4e5f\u4e0d\u4f1a\u57fa\u4e8e\u7f3a\u5931"
                    "\u7f13\u5b58\u751f\u6210\u7ed3\u8bba\u3002"
                ),
                warnings=warnings,
            )

        measurements = self._build_measurements(evidence_packet)
        evidence_refs = {
            key: value["evidence_id"]
            for key, value in measurements.items()
            if isinstance(value, dict) and value.get("evidence_id")
        }
        status = self._summary_status(planner_result, measurements, warnings)
        summary = {
            "intent": planner_result.resolved_intent,
            "status": status,
            "measurements": measurements,
            "retrievals": [],
            "warnings": warnings,
            "unsupported": None,
            "evidence_refs": evidence_refs,
            "tool_status": tool_status,
        }
        return ResponseBuildResult(
            structured_summary=summary,
            report_text=self._build_report_text(planner_result, evidence_packet),
            warnings=warnings,
        )

    @staticmethod
    def _build_measurements(evidence_packet: EvidencePacket) -> dict[str, dict[str, Any]]:
        mapping = {
            "measurement.dbh": "dbh_cm",
            "measurement.height": "height_m",
            "measurement.crown_width": "crown_width_m",
        }
        measurements: dict[str, dict[str, Any]] = {}
        for evidence_id, summary_key in mapping.items():
            item = evidence_packet.get_item(evidence_id)
            if item is not None:
                measurements[summary_key] = {
                    "value": item.value,
                    "unit": item.unit,
                    "status": item.status,
                    "evidence_id": item.evidence_id,
                    "source_tool": item.source_tool,
                }
        return measurements

    @staticmethod
    def _summary_status(
        planner_result: PlannerResult,
        measurements: dict[str, dict[str, Any]],
        warnings: list[str],
    ) -> str:
        if warnings and measurements:
            return "partial"
        if warnings and not measurements:
            return "failed"
        if planner_result.resolved_intent == "tree_report_q123":
            expected = {"dbh_cm", "height_m", "crown_width_m"}
            if expected.issubset(measurements):
                return "success"
            return "partial" if measurements else "failed"
        return "success" if measurements else "failed"

    def _build_report_text(
        self,
        planner_result: PlannerResult,
        evidence_packet: EvidencePacket,
    ) -> str:
        intent = planner_result.resolved_intent
        if intent == "q1_dbh":
            return self._single_measurement_text(
                evidence_packet.get_item("measurement.dbh"),
                label="\u80f8\u5f84",
            )
        if intent == "q2_height":
            return self._single_measurement_text(
                evidence_packet.get_item("measurement.height"),
                label="\u6811\u9ad8",
            )
        if intent == "q3_crown_width":
            return self._single_measurement_text(
                evidence_packet.get_item("measurement.crown_width"),
                label="\u51a0\u5e45",
            )
        if intent == "tree_report_q123":
            clauses = [
                self._measurement_clause(evidence_packet.get_item("measurement.dbh"), "\u80f8\u5f84"),
                self._measurement_clause(evidence_packet.get_item("measurement.height"), "\u6811\u9ad8"),
                self._measurement_clause(
                    evidence_packet.get_item("measurement.crown_width"),
                    "\u51a0\u5e45",
                ),
            ]
            return (
                "\u8be5\u5355\u6728" + "\uff0c".join(clauses) + "\u3002"
                "\u4ee5\u4e0a\u7ed3\u679c\u4ec5\u6765\u81ea Evidence Packet "
                "\u4e2d\u7684 q1/q2/q3 \u5de5\u5177\u8bc1\u636e\u3002"
            )
        return (
            "\u5f53\u524d\u8bc1\u636e\u4e0d\u8db3\uff0c"
            "\u65e0\u6cd5\u751f\u6210\u53d7\u652f\u6301\u7684\u62a5\u544a\u3002"
        )

    def _single_measurement_text(self, item: EvidenceItem | None, *, label: str) -> str:
        if item is None or item.status != "ok":
            return (
                f"{label}\u672a\u80fd\u8ba1\u7b97\uff0c"
                "\u5f53\u524d\u8bc1\u636e\u4e0d\u8db3\u3002"
            )
        return f"\u8be5\u5355\u6728\u4f30\u8ba1{label}\u4e3a {self._format_value(item)}\u3002"

    def _measurement_clause(self, item: EvidenceItem | None, label: str) -> str:
        if item is None or item.status != "ok":
            return f"{label}\u672a\u80fd\u8ba1\u7b97"
        return f"{label}\u4e3a {self._format_value(item)}"

    @staticmethod
    def _format_value(item: EvidenceItem) -> str:
        if isinstance(item.value, (int, float)):
            value_text = f"{float(item.value):.2f}"
        else:
            value_text = str(item.value)
        return f"{value_text} {item.unit}".strip()
