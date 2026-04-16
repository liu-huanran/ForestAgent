"""Tool wrapper for crown width estimation."""

from __future__ import annotations

from forestagent.backends.base_backend import BaseBackend
from forestagent.schemas import PointCloudInput, ToolResult
from forestagent.tools._helpers import run_backend_tool


def estimate_crown_width(point_cloud: PointCloudInput, backend: BaseBackend) -> ToolResult:
    """Call the backend crown width estimator and return a normalized ToolResult."""

    return run_backend_tool(
        "estimate_crown_width", point_cloud, backend, backend.estimate_crown_width
    )
