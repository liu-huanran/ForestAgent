"""Helpers for converting v2 tool results into evidence items."""

from __future__ import annotations

from forestagent.evidence.packet import EvidenceItem
from forestagent.tools.schema import ToolResult

_DEFAULT_EVIDENCE_BY_TOOL = {
    "q1_dbh": ("measurement.dbh", "measurement", "DBH estimated by v2 mock q1 tool."),
    "q2_height": (
        "measurement.height",
        "measurement",
        "Height estimated by v2 mock q2 tool.",
    ),
    "q3_crown_width": (
        "measurement.crown_width",
        "measurement",
        "Crown width estimated by v2 mock q3 tool.",
    ),
}


def evidence_items_from_tool_result(tool_result: ToolResult) -> list[EvidenceItem]:
    """Convert one tool result to zero or more evidence items."""

    if tool_result.evidence_items:
        return [EvidenceItem.model_validate(item) for item in tool_result.evidence_items]

    if tool_result.status == "skipped":
        return []

    if tool_result.status in {"failed", "unavailable"}:
        return [
            EvidenceItem(
                evidence_id=f"unavailable.{tool_result.tool_name}",
                source_tool=tool_result.tool_name,
                evidence_type="unavailable",
                value=None,
                unit=None,
                status="unavailable" if tool_result.status == "unavailable" else "failed",
                explanation=tool_result.error_message or "Tool did not produce evidence.",
                source_metadata=tool_result.metadata,
            )
        ]

    if tool_result.output is None:
        return []

    evidence_id, evidence_type, explanation = _DEFAULT_EVIDENCE_BY_TOOL.get(
        tool_result.tool_name,
        (
            tool_result.output.evidence_id or f"measurement.{tool_result.tool_name}",
            tool_result.output.evidence_type or "measurement",
            tool_result.output.explanation or f"Evidence from {tool_result.tool_name}.",
        ),
    )
    return [
        EvidenceItem(
            evidence_id=tool_result.output.evidence_id or evidence_id,
            source_tool=tool_result.tool_name,
            evidence_type=evidence_type,  # type: ignore[arg-type]
            value=tool_result.output.value,
            unit=tool_result.output.unit,
            status="ok",
            explanation=tool_result.output.explanation or explanation,
            source_metadata={**tool_result.metadata, **tool_result.output.metadata},
        )
    ]
