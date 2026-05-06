"""V2 tool schema definitions for controlled tool calling."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ToolResultStatus = Literal["ok", "failed", "unavailable", "skipped"]
ToolFamily = Literal[
    "geometry",
    "representation",
    "retrieval",
    "audit",
    "report_composition",
]


class ToolInput(BaseModel):
    """Normalized input passed to a v2 tool implementation."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(min_length=1)
    context: dict[str, Any] = Field(default_factory=dict)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolOutput(BaseModel):
    """Small structured output from a v2 tool.

    This model is intentionally light. Large artifacts such as point clouds and
    full embedding vectors should stay outside this structure.
    """

    model_config = ConfigDict(extra="forbid")

    value: Any | None = None
    unit: str | None = None
    evidence_id: str | None = None
    evidence_type: str | None = None
    explanation: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Unified v2 tool result contract."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(min_length=1)
    status: ToolResultStatus
    output: ToolOutput | None = None
    error_message: str | None = None
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    runtime_ms: float | None = Field(default=None, ge=0.0)
    evidence_items: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_result_shape(self) -> "ToolResult":
        if self.status in {"failed", "unavailable"} and not self.error_message:
            raise ValueError("failed/unavailable tool results must include error_message.")
        if self.status in {"failed", "unavailable"} and self.output is not None:
            if self.output.value is not None or self.output.unit is not None:
                raise ValueError("failed/unavailable tool results must not carry values.")
        return self
