"""Input models used across the ForestAgent MVP."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PointCloudInput(BaseModel):
    """Single-tree point cloud input.

    The `format` field is intentionally extensible. Known formats currently include
    `las`, `laz`, `ply`, and `xyz`.
    """

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    format: str = Field(min_length=1)

