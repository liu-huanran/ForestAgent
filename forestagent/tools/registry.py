"""V2 tool registry for controlled tool discovery."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from forestagent.tools.schema import ToolFamily


class ToolMetadata(BaseModel):
    """Static metadata for a v2 tool.

    The registry describes tools only; it does not execute them.
    """

    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(min_length=1)
    tool_version: str = Field(min_length=1)
    tool_family: ToolFamily
    intent_names: list[str] = Field(default_factory=list)
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    required_inputs: list[str] = Field(default_factory=list)
    optional_inputs: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    deterministic: bool = True
    requires_gpu: bool = False
    requires_checkpoint: bool = False
    is_frozen_baseline: bool = False
    allowed_in_mvp: bool = False
    allowed_in_v2: bool = False
    source_note: str = ""
    risk_note: str = ""
    owner_note: str = ""


class ToolRegistry:
    """In-memory registry for v2 tool metadata."""

    def __init__(self, tools: list[ToolMetadata] | None = None) -> None:
        self._tools: dict[str, ToolMetadata] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: ToolMetadata) -> None:
        if tool.tool_name in self._tools:
            raise ValueError(f"Tool is already registered: {tool.tool_name}")
        self._tools[tool.tool_name] = tool

    def list_tools(
        self,
        *,
        allowed_in_mvp: bool | None = None,
        allowed_in_v2: bool | None = None,
        family: str | None = None,
    ) -> list[ToolMetadata]:
        tools = list(self._tools.values())
        if allowed_in_mvp is not None:
            tools = [tool for tool in tools if tool.allowed_in_mvp is allowed_in_mvp]
        if allowed_in_v2 is not None:
            tools = [tool for tool in tools if tool.allowed_in_v2 is allowed_in_v2]
        if family is not None:
            tools = [tool for tool in tools if tool.tool_family == family]
        return sorted(tools, key=lambda tool: tool.tool_name)

    def get_tool(self, tool_name: str) -> ToolMetadata:
        try:
            return self._tools[tool_name]
        except KeyError as exc:
            raise KeyError(f"Tool is not registered: {tool_name}") from exc

    def find_by_intent(
        self,
        intent_name: str,
        *,
        allowed_in_mvp: bool | None = None,
        allowed_in_v2: bool | None = None,
    ) -> list[ToolMetadata]:
        tools = [
            tool for tool in self._tools.values() if intent_name in tool.intent_names
        ]
        if allowed_in_mvp is not None:
            tools = [tool for tool in tools if tool.allowed_in_mvp is allowed_in_mvp]
        if allowed_in_v2 is not None:
            tools = [tool for tool in tools if tool.allowed_in_v2 is allowed_in_v2]
        return sorted(tools, key=lambda tool: tool.tool_name)


def build_default_tool_registry() -> ToolRegistry:
    """Build the phase-1/2 v2 registry.

    Uni3D-related tools are registered for discovery only. They are not allowed
    in MVP or default v2 execution in this phase.
    """

    return ToolRegistry(
        [
            ToolMetadata(
                tool_name="q1_dbh",
                tool_version="v1.1",
                tool_family="geometry",
                intent_names=["q1_dbh"],
                description="Estimate DBH using the frozen q1 direct geometry baseline.",
                input_schema={"point_cloud_path": "str"},
                output_schema={"value": "float", "unit": "cm"},
                required_inputs=["point_cloud_path"],
                failure_modes=["missing_point_cloud", "geometry_backend_failed"],
                deterministic=True,
                is_frozen_baseline=True,
                allowed_in_mvp=True,
                allowed_in_v2=True,
                source_note="Thin v2 wrapper over the existing q1 geometry baseline.",
                risk_note="Do not change the core DBH algorithm through this registry.",
                owner_note="ForestAgent geometry baseline.",
            ),
            ToolMetadata(
                tool_name="q2_height",
                tool_version="v1.1",
                tool_family="geometry",
                intent_names=["q2_height"],
                description="Estimate tree height using the frozen q2 geometry baseline.",
                input_schema={"point_cloud_path": "str"},
                output_schema={"value": "float", "unit": "m"},
                required_inputs=["point_cloud_path"],
                failure_modes=["missing_point_cloud", "geometry_backend_failed"],
                deterministic=True,
                is_frozen_baseline=True,
                allowed_in_mvp=True,
                allowed_in_v2=True,
                source_note="Thin v2 wrapper over the existing q2 geometry baseline.",
                risk_note="Do not change the core height algorithm through this registry.",
                owner_note="ForestAgent geometry baseline.",
            ),
            ToolMetadata(
                tool_name="q3_crown_width",
                tool_version="v1.1",
                tool_family="geometry",
                intent_names=["q3_crown_width"],
                description="Estimate crown width using the frozen q3 geometry baseline.",
                input_schema={"point_cloud_path": "str"},
                output_schema={"value": "float", "unit": "m"},
                required_inputs=["point_cloud_path"],
                failure_modes=["missing_point_cloud", "geometry_backend_failed"],
                deterministic=True,
                is_frozen_baseline=True,
                allowed_in_mvp=True,
                allowed_in_v2=True,
                source_note="Thin v2 wrapper over the existing q3 geometry baseline.",
                risk_note="Do not change the core crown-width algorithm through this registry.",
                owner_note="ForestAgent geometry baseline.",
            ),
            ToolMetadata(
                tool_name="tree_report_q123",
                tool_version="v2.phase1",
                tool_family="report_composition",
                intent_names=["tree_report_q123"],
                description="Compose a q1/q2/q3 report from existing evidence.",
                input_schema={"evidence_packet": "EvidencePacket"},
                output_schema={"report_text": "str"},
                required_inputs=[],
                failure_modes=["missing_evidence"],
                deterministic=True,
                is_frozen_baseline=False,
                allowed_in_mvp=True,
                allowed_in_v2=True,
                source_note="Report composition only; does not create measurements.",
                risk_note="Must not introduce numbers outside the evidence packet.",
                owner_note="ForestAgent v2 response layer.",
            ),
            ToolMetadata(
                tool_name="uni3d_embedding_lookup",
                tool_version="v2.phase1.stub",
                tool_family="representation",
                intent_names=["uni3d_embedding_lookup"],
                description="Look up an existing cached Uni3D embedding summary.",
                input_schema={"tree_id": "str", "embedding_cache_dir": "str"},
                output_schema={"tree_id": "str", "embedding_path": "str"},
                required_inputs=["tree_id", "embedding_cache_dir"],
                failure_modes=["cache_not_configured", "tree_id_not_found"],
                deterministic=True,
                requires_gpu=False,
                requires_checkpoint=False,
                allowed_in_mvp=False,
                allowed_in_v2=False,
                source_note="Offline cache-only stub in phase 1/2.",
                risk_note="Must not run Uni3D forward or expose full vectors in reports.",
                owner_note="ForestAgent Uni3D offline branch.",
            ),
            ToolMetadata(
                tool_name="similar_tree_retrieval",
                tool_version="v2.phase1.stub",
                tool_family="retrieval",
                intent_names=["similar_tree_retrieval"],
                description="Retrieve similar trees from an existing embedding cache.",
                input_schema={"tree_id": "str", "embedding_cache_dir": "str"},
                output_schema={"neighbors": "list"},
                required_inputs=["tree_id", "embedding_cache_dir"],
                failure_modes=["cache_not_configured", "tree_id_not_found"],
                deterministic=True,
                requires_gpu=False,
                requires_checkpoint=False,
                allowed_in_mvp=False,
                allowed_in_v2=False,
                source_note="Offline retrieval stub in phase 1/2.",
                risk_note="Retrieval is not evidence for health, safety, or species claims.",
                owner_note="ForestAgent Uni3D offline branch.",
            ),
            ToolMetadata(
                tool_name="embedding_outlier_lookup",
                tool_version="v2.phase1.stub",
                tool_family="audit",
                intent_names=["embedding_outlier_lookup"],
                description="Look up existing embedding outlier analysis for a tree.",
                input_schema={"tree_id": "str", "analysis_dir": "str"},
                output_schema={"outlier_summary": "dict"},
                required_inputs=["tree_id", "analysis_dir"],
                failure_modes=["analysis_cache_not_configured", "tree_id_not_found"],
                deterministic=True,
                requires_gpu=False,
                requires_checkpoint=False,
                allowed_in_mvp=False,
                allowed_in_v2=False,
                source_note="Offline analysis-cache stub in phase 1/2.",
                risk_note="Outlier status must not be converted into health or risk claims.",
                owner_note="ForestAgent Uni3D offline branch.",
            ),
        ]
    )
