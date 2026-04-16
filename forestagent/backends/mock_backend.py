"""Deterministic mock backend for the first ForestAgent MVP round."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from forestagent.backends.base_backend import BaseBackend
from forestagent.schemas import PointCloudInput, ToolResult

KnownToolName = str
ResultOverride = ToolResult | Mapping[str, Any]
FailureSpec = ToolResult | Mapping[str, Any] | str


class MockBackend(BaseBackend):
    """Mock backend with deterministic defaults and controllable failures."""

    _SUPPORTED_TOOLS = frozenset(
        {
            "estimate_dbh",
            "estimate_height",
            "estimate_crown_width",
            "estimate_tilt",
            "assess_quality",
        }
    )

    _DEFAULT_SUCCESS_RESULTS: dict[KnownToolName, dict[str, Any]] = {
        "estimate_dbh": {
            "tool_name": "estimate_dbh",
            "status": "success",
            "value": 28.4,
            "unit": "cm",
            "confidence": 0.87,
            "extra": {"backend": "mock"},
            "message": "Mock DBH estimate generated successfully.",
        },
        "estimate_height": {
            "tool_name": "estimate_height",
            "status": "success",
            "value": 16.2,
            "unit": "m",
            "confidence": 0.90,
            "extra": {"backend": "mock"},
            "message": "Mock tree height estimate generated successfully.",
        },
        "estimate_crown_width": {
            "tool_name": "estimate_crown_width",
            "status": "success",
            "value": 7.4,
            "unit": "m",
            "confidence": 0.82,
            "extra": {"backend": "mock"},
            "message": "Mock crown width estimate generated successfully.",
        },
        "estimate_tilt": {
            "tool_name": "estimate_tilt",
            "status": "success",
            "value": 9.5,
            "unit": "deg",
            "confidence": 0.78,
            "extra": {"backend": "mock"},
            "message": "Mock tilt estimate generated successfully.",
        },
        "assess_quality": {
            "tool_name": "assess_quality",
            "status": "success",
            "value": "medium",
            "unit": None,
            "confidence": 0.75,
            "extra": {"backend": "mock"},
            "message": "Mock point cloud quality assessment generated successfully.",
        },
    }

    def __init__(
        self,
        overrides: Mapping[KnownToolName, ResultOverride] | None = None,
        failures: Mapping[KnownToolName, FailureSpec] | None = None,
    ) -> None:
        self._overrides = dict(overrides or {})
        self._failures = dict(failures or {})
        self._validate_tool_names(self._overrides, "overrides")
        self._validate_tool_names(self._failures, "failures")

    def estimate_dbh(self, point_cloud: PointCloudInput) -> ToolResult:
        return self._resolve_result("estimate_dbh", point_cloud)

    def estimate_height(self, point_cloud: PointCloudInput) -> ToolResult:
        return self._resolve_result("estimate_height", point_cloud)

    def estimate_crown_width(self, point_cloud: PointCloudInput) -> ToolResult:
        return self._resolve_result("estimate_crown_width", point_cloud)

    def estimate_tilt(self, point_cloud: PointCloudInput) -> ToolResult:
        return self._resolve_result("estimate_tilt", point_cloud)

    def assess_quality(self, point_cloud: PointCloudInput) -> ToolResult:
        return self._resolve_result("assess_quality", point_cloud)

    @classmethod
    def _validate_tool_names(
        cls, specs: Mapping[KnownToolName, Any], spec_name: str
    ) -> None:
        unknown_tools = sorted(set(specs) - cls._SUPPORTED_TOOLS)
        if unknown_tools:
            names = ", ".join(unknown_tools)
            raise ValueError(f"Unknown tool names in {spec_name}: {names}")

    def _resolve_result(
        self, tool_name: KnownToolName, point_cloud: PointCloudInput
    ) -> ToolResult:
        if tool_name in self._failures:
            return self._build_failed_result(tool_name, self._failures[tool_name], point_cloud)

        default_payload = deepcopy(self._DEFAULT_SUCCESS_RESULTS[tool_name])
        default_payload["extra"] = {
            **default_payload["extra"],
            "input_format": point_cloud.format,
        }

        override = self._overrides.get(tool_name)
        if override is None:
            return ToolResult.model_validate(default_payload)

        if isinstance(override, ToolResult):
            return override

        if not isinstance(override, Mapping):
            raise TypeError(f"Override for {tool_name} must be a mapping or ToolResult.")

        merged_payload = dict(default_payload)
        for key, value in override.items():
            if key == "extra" and isinstance(value, Mapping):
                merged_payload["extra"] = {**merged_payload["extra"], **dict(value)}
            else:
                merged_payload[key] = value

        merged_payload["tool_name"] = tool_name
        if merged_payload.get("status") != "success":
            raise ValueError(
                f"Override for {tool_name} must keep status='success'. "
                "Use failures to inject failed results."
            )

        return ToolResult.model_validate(merged_payload)

    @staticmethod
    def _build_failed_result(
        tool_name: KnownToolName, spec: FailureSpec, point_cloud: PointCloudInput
    ) -> ToolResult:
        default_message = f"Mock failure injected for {tool_name}."
        payload: dict[str, Any] = {
            "tool_name": tool_name,
            "status": "failed",
            "value": None,
            "unit": None,
            "confidence": None,
            "extra": {
                "backend": "mock",
                "input_format": point_cloud.format,
            },
            "message": default_message,
        }

        if isinstance(spec, ToolResult):
            if spec.status != "failed":
                raise ValueError(
                    f"Failure override for {tool_name} must have status='failed'."
                )
            spec_payload = spec.model_dump()
            payload["extra"] = {**payload["extra"], **spec_payload.get("extra", {})}
            payload["message"] = spec_payload.get("message") or default_message
        elif isinstance(spec, str):
            payload["message"] = spec
        elif isinstance(spec, Mapping):
            payload["message"] = str(spec.get("message", default_message))
            extra = spec.get("extra")
            if extra is not None:
                if not isinstance(extra, Mapping):
                    raise TypeError(
                        f"Failure extra for {tool_name} must be a mapping if provided."
                    )
                payload["extra"] = {**payload["extra"], **dict(extra)}
        else:
            raise TypeError(
                f"Failure spec for {tool_name} must be a string, mapping, or ToolResult."
            )

        return ToolResult.model_validate(payload)
