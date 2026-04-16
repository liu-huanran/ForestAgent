"""Tool wrapper for DBH estimation."""

from __future__ import annotations

from forestagent.backends.base_backend import BaseBackend
from forestagent.schemas import PointCloudInput, ToolResult
from forestagent.tools._helpers import run_backend_tool


def estimate_dbh(point_cloud: PointCloudInput, backend: BaseBackend) -> ToolResult:
    """Call the backend DBH estimator and return a normalized ToolResult."""

    return run_backend_tool("estimate_dbh", point_cloud, backend, backend.estimate_dbh)
