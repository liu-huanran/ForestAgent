"""Schema definitions for tool-level outputs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ToolStatus = Literal["success", "failed"]


class ToolResult(BaseModel):
    """Unified tool output contract for the MVP."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(min_length=1)
    status: ToolStatus
    value: Any | None = None
    unit: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    extra: dict[str, Any] = Field(default_factory=dict)
    message: str = ""

    @model_validator(mode="after")
    def validate_failure_shape(self) -> "ToolResult":
        if self.status == "failed":
            if self.value is not None or self.unit is not None or self.confidence is not None:
                raise ValueError(
                    "Failed tool results must set value, unit, and confidence to None."
                )
            if not self.message:
                raise ValueError("Failed tool results must include a message.")
        return self

