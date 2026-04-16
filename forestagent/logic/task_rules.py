"""Task-level rule evaluation for the fixed MVP tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from forestagent.config_loader import load_yaml_mapping
from forestagent.schemas import (
    JsonSummaryPayload,
    LabelWithValuePayload,
    QualityLabelPayload,
    TaskOutputType,
    TaskPayload,
    ToolResult,
)

_DEFAULT_RULES_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "task_rules.yaml"
)


class TaskRuleError(ValueError):
    """Raised when task-level rule evaluation cannot proceed."""


@dataclass(frozen=True)
class TaskRuleConclusion:
    """Task-level structured conclusion produced from tool outputs."""

    output_type: TaskOutputType
    result: TaskPayload
    message: str
    extra: dict[str, Any] = field(default_factory=dict)


class LabelMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1)
    message: str = Field(min_length=1)


class Q4TiltThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    obvious_tilt_deg: float = Field(gt=0.0)


class Q5QualityThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    high_confidence_min: float = Field(ge=0.0, le=1.0)
    medium_confidence_min: float = Field(ge=0.0, le=1.0)


class Q6FormThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slender_ratio_tall_min: float = Field(gt=0.0)
    slender_ratio_stout_max: float = Field(gt=0.0)


class Q7FallRiskThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    high_tilt_deg_min: float = Field(gt=0.0)
    moderate_tilt_deg_min: float = Field(gt=0.0)
    high_slender_ratio_min: float = Field(gt=0.0)
    moderate_slender_ratio_min: float = Field(gt=0.0)
    small_dbh_cm_max: float = Field(gt=0.0)
    low_quality_confidence_max: float = Field(ge=0.0, le=1.0)


class ThresholdConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    q4_tilt: Q4TiltThresholds
    q5_quality: Q5QualityThresholds
    q6_form: Q6FormThresholds
    q7_fall_risk: Q7FallRiskThresholds


class Q4TiltLabels(BaseModel):
    model_config = ConfigDict(extra="forbid")

    obvious_tilt: LabelMessage
    no_obvious_tilt: LabelMessage


class Q5QualityLabels(BaseModel):
    model_config = ConfigDict(extra="forbid")

    high: LabelMessage
    medium: LabelMessage
    low: LabelMessage


class Q6FormLabels(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slender: LabelMessage
    stout: LabelMessage
    balanced: LabelMessage


class Q7FallRiskLabels(BaseModel):
    model_config = ConfigDict(extra="forbid")

    high: LabelMessage
    medium: LabelMessage
    low: LabelMessage
    uncertain: LabelMessage


class LabelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    q4_tilt: Q4TiltLabels
    q5_quality: Q5QualityLabels
    q6_form: Q6FormLabels
    q7_fall_risk: Q7FallRiskLabels


class TaskRulesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thresholds: ThresholdConfig
    labels: LabelConfig


def load_task_rules(config_path: str | Path | None = None) -> TaskRulesConfig:
    """Load task rule thresholds and label/message definitions from YAML."""

    resolved_path = Path(config_path or _DEFAULT_RULES_PATH).resolve()
    raw_config = load_yaml_mapping(resolved_path)
    return TaskRulesConfig.model_validate(raw_config)


def evaluate_q4_tilt(
    tilt_result: ToolResult, rules: TaskRulesConfig | None = None
) -> TaskRuleConclusion:
    """Convert the tilt tool output into a task-level labeled conclusion."""

    rules = rules or load_task_rules()
    angle_deg = _require_numeric_tool_result(tilt_result, "estimate_tilt")
    if tilt_result.unit is None:
        raise TaskRuleError("estimate_tilt must provide a unit.")

    threshold = rules.thresholds.q4_tilt.obvious_tilt_deg
    label_config = (
        rules.labels.q4_tilt.obvious_tilt
        if angle_deg >= threshold
        else rules.labels.q4_tilt.no_obvious_tilt
    )

    return TaskRuleConclusion(
        output_type="label_with_value",
        result=LabelWithValuePayload(
            label=label_config.label,
            value=round(angle_deg, 2),
            unit=tilt_result.unit,
            confidence=tilt_result.confidence,
        ),
        message=label_config.message,
        extra={"obvious_tilt_deg_threshold": threshold},
    )


def evaluate_q5_quality(
    quality_result: ToolResult, rules: TaskRulesConfig | None = None
) -> TaskRuleConclusion:
    """Convert the quality tool output into a task-level quality label."""

    rules = rules or load_task_rules()
    _require_success_tool_result(quality_result, "assess_quality")

    if quality_result.confidence is None:
        raise TaskRuleError("assess_quality must provide confidence for rule evaluation.")

    thresholds = rules.thresholds.q5_quality
    if quality_result.confidence >= thresholds.high_confidence_min:
        label_config = rules.labels.q5_quality.high
    elif quality_result.confidence >= thresholds.medium_confidence_min:
        label_config = rules.labels.q5_quality.medium
    else:
        label_config = rules.labels.q5_quality.low

    return TaskRuleConclusion(
        output_type="quality_label",
        result=QualityLabelPayload(
            label=label_config.label,
            confidence=quality_result.confidence,
        ),
        message=label_config.message,
        extra={
            "backend_quality_value": quality_result.value,
            "high_confidence_min": thresholds.high_confidence_min,
            "medium_confidence_min": thresholds.medium_confidence_min,
        },
    )


def evaluate_q6_form(
    dbh_result: ToolResult,
    height_result: ToolResult,
    rules: TaskRulesConfig | None = None,
) -> TaskRuleConclusion:
    """Combine DBH and height into a form summary."""

    rules = rules or load_task_rules()
    dbh_cm = _require_numeric_tool_result(dbh_result, "estimate_dbh")
    height_m = _require_numeric_tool_result(height_result, "estimate_height")
    slenderness_ratio = _compute_slenderness_ratio(height_m, dbh_cm)

    thresholds = rules.thresholds.q6_form
    if slenderness_ratio >= thresholds.slender_ratio_tall_min:
        label_config = rules.labels.q6_form.slender
    elif slenderness_ratio <= thresholds.slender_ratio_stout_max:
        label_config = rules.labels.q6_form.stout
    else:
        label_config = rules.labels.q6_form.balanced

    return TaskRuleConclusion(
        output_type="json_summary",
        result=JsonSummaryPayload(
            summary={
                "label": label_config.label,
                "message": label_config.message,
                "dbh_cm": round(dbh_cm, 2),
                "height_m": round(height_m, 2),
                "slenderness_ratio": round(slenderness_ratio, 2),
            }
        ),
        message=label_config.message,
        extra={
            "slender_ratio_tall_min": thresholds.slender_ratio_tall_min,
            "slender_ratio_stout_max": thresholds.slender_ratio_stout_max,
        },
    )


def evaluate_q7_fall_risk(
    tilt_result: ToolResult,
    dbh_result: ToolResult,
    height_result: ToolResult,
    quality_result: ToolResult,
    rules: TaskRulesConfig | None = None,
) -> TaskRuleConclusion:
    """Combine geometry and quality signals into a fall-risk summary."""

    rules = rules or load_task_rules()
    tilt_deg = _require_numeric_tool_result(tilt_result, "estimate_tilt")
    dbh_cm = _require_numeric_tool_result(dbh_result, "estimate_dbh")
    height_m = _require_numeric_tool_result(height_result, "estimate_height")
    _require_success_tool_result(quality_result, "assess_quality")

    if quality_result.confidence is None:
        raise TaskRuleError("assess_quality must provide confidence for risk evaluation.")

    slenderness_ratio = _compute_slenderness_ratio(height_m, dbh_cm)
    thresholds = rules.thresholds.q7_fall_risk

    tilt_high = tilt_deg >= thresholds.high_tilt_deg_min
    tilt_moderate = tilt_deg >= thresholds.moderate_tilt_deg_min
    slender_high = slenderness_ratio >= thresholds.high_slender_ratio_min
    slender_moderate = slenderness_ratio >= thresholds.moderate_slender_ratio_min
    small_dbh = dbh_cm <= thresholds.small_dbh_cm_max
    low_quality = quality_result.confidence <= thresholds.low_quality_confidence_max

    if low_quality:
        risk_key = "uncertain"
    elif (tilt_high and (slender_moderate or small_dbh)) or (
        tilt_moderate and slender_high
    ):
        risk_key = "high"
    elif tilt_high or (tilt_moderate and slender_moderate) or (
        slender_high and small_dbh
    ):
        risk_key = "medium"
    else:
        risk_key = "low"

    label_config = getattr(rules.labels.q7_fall_risk, risk_key)

    return TaskRuleConclusion(
        output_type="json_summary",
        result=JsonSummaryPayload(
            summary={
                "label": label_config.label,
                "message": label_config.message,
                "tilt_deg": round(tilt_deg, 2),
                "dbh_cm": round(dbh_cm, 2),
                "height_m": round(height_m, 2),
                "slenderness_ratio": round(slenderness_ratio, 2),
                "quality_confidence": quality_result.confidence,
                "signals": {
                    "tilt_high": tilt_high,
                    "tilt_moderate": tilt_moderate,
                    "slender_high": slender_high,
                    "slender_moderate": slender_moderate,
                    "small_dbh": small_dbh,
                    "low_quality": low_quality,
                },
            }
        ),
        message=label_config.message,
        extra={
            "high_tilt_deg_min": thresholds.high_tilt_deg_min,
            "moderate_tilt_deg_min": thresholds.moderate_tilt_deg_min,
            "high_slender_ratio_min": thresholds.high_slender_ratio_min,
            "moderate_slender_ratio_min": thresholds.moderate_slender_ratio_min,
            "small_dbh_cm_max": thresholds.small_dbh_cm_max,
            "low_quality_confidence_max": thresholds.low_quality_confidence_max,
        },
    )


def _require_success_tool_result(tool_result: ToolResult, expected_tool_name: str) -> None:
    if tool_result.tool_name != expected_tool_name:
        raise TaskRuleError(
            f"Expected tool result {expected_tool_name}, got {tool_result.tool_name}."
        )
    if tool_result.status != "success":
        raise TaskRuleError(
            f"{expected_tool_name} must succeed before task-level rule evaluation."
        )


def _require_numeric_tool_result(tool_result: ToolResult, expected_tool_name: str) -> float:
    _require_success_tool_result(tool_result, expected_tool_name)
    if tool_result.value is None or isinstance(tool_result.value, bool):
        raise TaskRuleError(f"{expected_tool_name} must provide a numeric value.")
    if not isinstance(tool_result.value, (int, float)):
        raise TaskRuleError(f"{expected_tool_name} must provide a numeric value.")
    return float(tool_result.value)


def _compute_slenderness_ratio(height_m: float, dbh_cm: float) -> float:
    if dbh_cm <= 0:
        raise TaskRuleError("estimate_dbh must be greater than 0 for rule evaluation.")
    if height_m <= 0:
        raise TaskRuleError("estimate_height must be greater than 0 for rule evaluation.")
    return (height_m * 100.0) / dbh_cm
