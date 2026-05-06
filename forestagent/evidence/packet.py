"""Evidence packet models for ForestAgent v2."""

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

EvidenceType = Literal[
    "measurement",
    "retrieval",
    "classification",
    "regression",
    "warning",
    "unavailable",
]
EvidenceStatus = Literal["ok", "failed", "unavailable"]


class EvidenceItem(BaseModel):
    """A single auditable fact or failure produced by a tool."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1)
    source_tool: str = Field(min_length=1)
    evidence_type: EvidenceType
    value: Any | None = None
    unit: str | None = None
    status: EvidenceStatus
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    quality_flag: str | None = None
    explanation: str = ""
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    trace_ref: str | None = None

    @model_validator(mode="after")
    def validate_evidence_shape(self) -> "EvidenceItem":
        if self.status in {"failed", "unavailable"}:
            if self.value is not None or self.unit is not None:
                raise ValueError("failed/unavailable evidence must not carry values.")
            if not self.explanation:
                raise ValueError("failed/unavailable evidence must include an explanation.")
        return self


class EvidencePacket(BaseModel):
    """Serializable collection of evidence items."""

    model_config = ConfigDict(extra="forbid")

    packet_id: str = Field(default_factory=lambda: f"evidence-{uuid4().hex}")
    request_id: str | None = None
    items: list[EvidenceItem] = Field(default_factory=list)

    def add_item(self, item: EvidenceItem) -> None:
        self.items.append(item)

    def get_item(self, evidence_id: str) -> EvidenceItem | None:
        for item in self.items:
            if item.evidence_id == evidence_id:
                return item
        return None

    def by_type(self, evidence_type: EvidenceType) -> list[EvidenceItem]:
        return [item for item in self.items if item.evidence_type == evidence_type]

    def by_source_tool(self, source_tool: str) -> list[EvidenceItem]:
        return [item for item in self.items if item.source_tool == source_tool]

    def ok_measurements(self) -> list[EvidenceItem]:
        return [
            item
            for item in self.items
            if item.evidence_type == "measurement" and item.status == "ok"
        ]
