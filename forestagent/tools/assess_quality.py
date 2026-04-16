"""Tool wrapper for point cloud quality assessment."""

from __future__ import annotations

from forestagent.backends.base_backend import BaseBackend
from forestagent.schemas import PointCloudInput, ToolResult
from forestagent.tools._helpers import run_backend_tool


def assess_quality(point_cloud: PointCloudInput, backend: BaseBackend) -> ToolResult:
    """Call the backend quality assessor and return a normalized ToolResult."""

    return run_backend_tool(
        "assess_quality", point_cloud, backend, backend.assess_quality
    )
