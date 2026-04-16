"""Tool wrapper for trunk tilt estimation."""

from __future__ import annotations

from forestagent.backends.base_backend import BaseBackend
from forestagent.schemas import PointCloudInput, ToolResult
from forestagent.tools._helpers import run_backend_tool


def estimate_tilt(point_cloud: PointCloudInput, backend: BaseBackend) -> ToolResult:
    """Call the backend tilt estimator and return a normalized ToolResult."""

    return run_backend_tool("estimate_tilt", point_cloud, backend, backend.estimate_tilt)
