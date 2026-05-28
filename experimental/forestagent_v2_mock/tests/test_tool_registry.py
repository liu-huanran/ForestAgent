"""Tests for the ForestAgent v2 tool registry."""

from __future__ import annotations

import unittest

from forestagent.tools.registry import ToolMetadata, ToolRegistry, build_default_tool_registry


class ToolRegistryTests(unittest.TestCase):
    def test_default_registry_contains_phase_one_two_tools(self) -> None:
        registry = build_default_tool_registry()
        tool_names = {tool.tool_name for tool in registry.list_tools()}

        self.assertIn("q1_dbh", tool_names)
        self.assertIn("q2_height", tool_names)
        self.assertIn("q3_crown_width", tool_names)
        self.assertIn("tree_report_q123", tool_names)
        self.assertIn("uni3d_embedding_lookup", tool_names)
        self.assertIn("similar_tree_retrieval", tool_names)
        self.assertIn("embedding_outlier_lookup", tool_names)

    def test_q123_metadata_is_frozen_and_allowed_in_v2(self) -> None:
        registry = build_default_tool_registry()

        for tool_name in ("q1_dbh", "q2_height", "q3_crown_width"):
            metadata = registry.get_tool(tool_name)
            self.assertEqual(metadata.tool_family, "geometry")
            self.assertTrue(metadata.is_frozen_baseline)
            self.assertTrue(metadata.allowed_in_mvp)
            self.assertTrue(metadata.allowed_in_v2)
            self.assertEqual(metadata.required_inputs, ["point_cloud_path"])

    def test_uni3d_related_tools_are_metadata_only_by_default(self) -> None:
        registry = build_default_tool_registry()

        for tool_name in (
            "uni3d_embedding_lookup",
            "similar_tree_retrieval",
            "embedding_outlier_lookup",
        ):
            metadata = registry.get_tool(tool_name)
            self.assertFalse(metadata.requires_gpu)
            self.assertFalse(metadata.requires_checkpoint)
            self.assertFalse(metadata.allowed_in_mvp)
            self.assertFalse(metadata.allowed_in_v2)

    def test_registry_can_filter_and_find_by_intent(self) -> None:
        registry = build_default_tool_registry()

        mvp_tools = {tool.tool_name for tool in registry.list_tools(allowed_in_mvp=True)}
        self.assertEqual(
            mvp_tools,
            {"q1_dbh", "q2_height", "q3_crown_width", "tree_report_q123"},
        )

        q1_candidates = registry.find_by_intent("q1_dbh", allowed_in_v2=True)
        self.assertEqual([tool.tool_name for tool in q1_candidates], ["q1_dbh"])

    def test_missing_tool_and_duplicate_registration_fail_clearly(self) -> None:
        registry = build_default_tool_registry()

        with self.assertRaisesRegex(KeyError, "not registered: missing_tool"):
            registry.get_tool("missing_tool")

        duplicate = registry.get_tool("q1_dbh")
        with self.assertRaisesRegex(ValueError, "already registered: q1_dbh"):
            registry.register(duplicate)

    def test_custom_registry_accepts_valid_metadata(self) -> None:
        registry = ToolRegistry()
        registry.register(
            ToolMetadata(
                tool_name="custom_audit",
                tool_version="v0",
                tool_family="audit",
                intent_names=["custom_audit"],
            )
        )

        self.assertEqual(registry.get_tool("custom_audit").tool_family, "audit")


if __name__ == "__main__":
    unittest.main()
