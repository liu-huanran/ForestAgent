"""Schema definitions for task-level outputs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .tool_result import ToolStatus

TaskOutputType = Literal[
    "scalar",
    "label_with_value",
    "quality_label",
    "json_summary",
    "report_text",
]


class ScalarPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float
    unit: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class LabelWithValuePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    value: float
    unit: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class QualityLabelPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class JsonSummaryPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: dict[str, Any]


class ReportTextPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str


TaskPayload = (
    ScalarPayload
    | LabelWithValuePayload
    | QualityLabelPayload
    | JsonSummaryPayload
    | ReportTextPayload
)

_PAYLOAD_BY_OUTPUT_TYPE: dict[TaskOutputType, type[BaseModel]] = {
    "scalar": ScalarPayload,
    "label_with_value": LabelWithValuePayload,
    "quality_label": QualityLabelPayload,
    "json_summary": JsonSummaryPayload,
    "report_text": ReportTextPayload,
}


class TaskResult(BaseModel):
    """Unified task output contract for the MVP."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    task_name: str = Field(min_length=1)
    status: ToolStatus
    output_type: TaskOutputType
    result: TaskPayload | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    message: str = ""

    @model_validator(mode="after")
    def validate_result_shape(self) -> "TaskResult":
        expected_payload_type = _PAYLOAD_BY_OUTPUT_TYPE[self.output_type]

        if self.status == "success":
            if self.result is None:
                raise ValueError("Successful task results must include a result payload.")
            if not isinstance(self.result, expected_payload_type):
                raise TypeError(
                    f"Result payload does not match output_type={self.output_type!r}."
                )
        else:
            if self.result is not None:
                raise ValueError("Failed task results must set result to None.")
            if not self.message:
                raise ValueError("Failed task results must include a message.")

        return self

