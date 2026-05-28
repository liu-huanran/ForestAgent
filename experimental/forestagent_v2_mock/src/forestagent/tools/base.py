"""Base protocols for v2 tool implementations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from forestagent.tools.schema import ToolInput, ToolResult

ToolCallable = Callable[[ToolInput], ToolResult]


class ToolExecutionContext(BaseModel):
    """Execution context shared by v2 tools."""

    model_config = ConfigDict(extra="forbid")

    point_cloud_path: str | None = None
    tree_id: str | None = None
    embedding_cache_dir: str | None = None
    analysis_dir: str | None = None
    mock_values: dict[str, Any] = Field(default_factory=dict)
    mock_failures: dict[str, str] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)

    def as_tool_context(self) -> dict[str, Any]:
        return self.model_dump()


class V2Tool(Protocol):
    """Protocol for a callable v2 tool adapter."""

    def __call__(self, tool_input: ToolInput) -> ToolResult:
        """Run a tool with normalized input."""
