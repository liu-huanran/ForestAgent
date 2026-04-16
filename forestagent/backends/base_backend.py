"""Abstract backend contract for geometry estimation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from forestagent.schemas import PointCloudInput, ToolResult


class BaseBackend(ABC):
    """Minimal backend interface exposed to the first MVP pipeline."""

    @abstractmethod
    def estimate_dbh(self, point_cloud: PointCloudInput) -> ToolResult:
        """Estimate DBH."""

    @abstractmethod
    def estimate_height(self, point_cloud: PointCloudInput) -> ToolResult:
        """Estimate tree height."""

    @abstractmethod
    def estimate_crown_width(self, point_cloud: PointCloudInput) -> ToolResult:
        """Estimate crown width."""

    @abstractmethod
    def estimate_tilt(self, point_cloud: PointCloudInput) -> ToolResult:
        """Estimate trunk tilt angle."""

    @abstractmethod
    def assess_quality(self, point_cloud: PointCloudInput) -> ToolResult:
        """Assess point cloud quality."""
