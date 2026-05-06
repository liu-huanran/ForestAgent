"""Trace schema for ForestAgent v2 controlled tool calls."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class TraceRecord(BaseModel):
    """Serializable v2 execution trace.

    Trace records keep references and summaries only. They must not store raw
    point clouds or full embedding vectors.
    """

    model_config = ConfigDict(extra="forbid")

    trace_id: str = Field(default_factory=lambda: f"trace-{uuid4().hex}")
    timestamp: str = Field(default_factory=_utc_timestamp)
    request_id: str = Field(default_factory=lambda: f"request-{uuid4().hex}")
    user_question: str = ""
    resolved_intent: str | None = None
    planner_reason: str = ""
    selected_tools: list[str] = Field(default_factory=list)
    execution_plan: list[dict[str, Any]] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    evidence_packet: dict[str, Any] = Field(default_factory=dict)
    structured_summary: dict[str, Any] = Field(default_factory=dict)
    report_text: str = ""
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    config_version: str = "forestagent-v2-phase1-2"

    def save_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.model_dump(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return output_path
