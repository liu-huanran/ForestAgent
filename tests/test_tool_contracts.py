"""Contract tests for ForestAgent tool wrappers."""

from __future__ import annotations

import unittest

from forestagent.backends.base_backend import BaseBackend
from forestagent.backends.mock_backend import MockBackend
from forestagent.schemas import PointCloudInput, ToolResult
from forestagent.tools import (
    assess_quality,
    estimate_crown_width,
    estimate_dbh,
    estimate_height,
    estimate_tilt,
)


class ExplodingBackend(BaseBackend):
    """Backend used to verify tool-level exception handling."""

    def _explode(self, tool_name: str, _: PointCloudInput) -> ToolResult:
        raise RuntimeError(f"{tool_name} exploded")

    def estimate_dbh(self, point_cloud: PointCloudInput) -> ToolResult:
        return self._explode("estimate_dbh", point_cloud)

    def estimate_height(self, point_cloud: PointCloudInput) -> ToolResult:
        return self._explode("estimate_height", point_cloud)

    def estimate_crown_width(self, point_cloud: PointCloudInput) -> ToolResult:
        return self._explode("estimate_crown_width", point_cloud)

    def estimate_tilt(self, point_cloud: PointCloudInput) -> ToolResult:
        return self._explode("estimate_tilt", point_cloud)

    def assess_quality(self, point_cloud: PointCloudInput) -> ToolResult:
        return self._explode("assess_quality", point_cloud)


class ToolContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.point_cloud = PointCloudInput(path="data/mock_tree.laz", format="laz")
        self.tool_cases = [
            (estimate_dbh, "estimate_dbh", "cm"),
            (estimate_height, "estimate_height", "m"),
            (estimate_crown_width, "estimate_crown_width", "m"),
            (estimate_tilt, "estimate_tilt", "deg"),
            (assess_quality, "assess_quality", None),
        ]

    def test_tools_return_normalized_tool_results(self) -> None:
        backend = MockBackend()

        for tool_func, tool_name, expected_unit in self.tool_cases:
            with self.subTest(tool_name=tool_name):
                result = tool_func(self.point_cloud, backend)
                self.assertIsInstance(result, ToolResult)
                self.assertEqual(result.tool_name, tool_name)
                self.assertEqual(result.status, "success")
                self.assertEqual(result.unit, expected_unit)
                self.assertIn("backend", result.extra)
                self.assertEqual(result.extra["input_format"], "laz")

    def test_tools_preserve_backend_failures(self) -> None:
        backend = MockBackend(
            failures={
                "estimate_dbh": "DBH unavailable",
                "estimate_height": "Height unavailable",
                "estimate_crown_width": "Crown width unavailable",
                "estimate_tilt": "Tilt unavailable",
                "assess_quality": "Quality unavailable",
            }
        )

        for tool_func, tool_name, _ in self.tool_cases:
            with self.subTest(tool_name=tool_name):
                result = tool_func(self.point_cloud, backend)
                self.assertEqual(result.tool_name, tool_name)
                self.assertEqual(result.status, "failed")
                self.assertIsNone(result.value)
                self.assertIsNone(result.unit)
                self.assertIsNone(result.confidence)
                self.assertIn("unavailable", result.message.lower())

    def test_tools_convert_backend_exceptions_to_failed_results(self) -> None:
        backend = ExplodingBackend()

        for tool_func, tool_name, _ in self.tool_cases:
            with self.subTest(tool_name=tool_name):
                result = tool_func(self.point_cloud, backend)
                self.assertEqual(result.tool_name, tool_name)
                self.assertEqual(result.status, "failed")
                self.assertIsNone(result.value)
                self.assertIsNone(result.unit)
                self.assertIsNone(result.confidence)
                self.assertIn("backend call failed", result.message)
                self.assertEqual(result.extra["backend_class"], "ExplodingBackend")


if __name__ == "__main__":
    unittest.main()

