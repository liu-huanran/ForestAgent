"""Shared helpers for tool wrappers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from forestagent.backends.base_backend import BaseBackend
from forestagent.schemas import PointCloudInput, ToolResult


def run_backend_tool(
    tool_name: str,
    point_cloud: PointCloudInput,
    backend: BaseBackend,
    backend_call: Callable[[PointCloudInput], Any],
) -> ToolResult:
    """Invoke a backend method and normalize the output into ToolResult."""

    try:
        raw_result = backend_call(point_cloud)
    except Exception as exc:
        return ToolResult(
            tool_name=tool_name,
            status="failed",
            value=None,
            unit=None,
            confidence=None,
            extra={
                "backend_class": backend.__class__.__name__,
                "input_format": point_cloud.format,
            },
            message=f"{tool_name} backend call failed: {exc}",
        )

    try:
        return normalize_tool_result(tool_name, raw_result)
    except Exception as exc:
        return ToolResult(
            tool_name=tool_name,
            status="failed",
            value=None,
            unit=None,
            confidence=None,
            extra={
                "backend_class": backend.__class__.__name__,
                "input_format": point_cloud.format,
            },
            message=f"{tool_name} returned an invalid backend payload: {exc}",
        )


def normalize_tool_result(tool_name: str, raw_result: Any) -> ToolResult:
    """Normalize a backend payload to the unified ToolResult contract."""

    if isinstance(raw_result, ToolResult):
        payload = raw_result.model_dump()
    elif isinstance(raw_result, Mapping):
        payload = dict(raw_result)
    else:
        raise TypeError("Backend payload must be a ToolResult or mapping.")

    extra = payload.get("extra", {})
    if extra is None:
        extra = {}
    if not isinstance(extra, Mapping):
        raise TypeError("ToolResult extra must be a mapping if provided.")

    payload["tool_name"] = tool_name
    payload["extra"] = dict(extra)
    return ToolResult.model_validate(payload)

