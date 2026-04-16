"""Tests for the minimal q1/q2/q3 single-tree MVP entrypoint."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import laspy
import numpy as np

from forestagent.backends.mock_backend import MockBackend
from forestagent.mvp import run_single_tree_analysis
from forestagent.verbalizers import OllamaVerbalizerError


class SingleTreeAnalysisTests(unittest.TestCase):
    def test_task_id_q1_runs_only_dbh_tool(self) -> None:
        result = run_single_tree_analysis(
            "demo_tree.las",
            task_id="q1_dbh",
            backend=MockBackend(),
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["request"]["resolved_intent"], "q1_dbh")
        self.assertEqual(list(result["tool_results"]), ["estimate_dbh"])
        self.assertEqual(result["json_summary"]["dbh_cm"]["status"], "success")
        self.assertIsNone(result["json_summary"]["height_m"])
        self.assertIsNone(result["json_summary"]["crown_width_m"])
        self.assertIn("胸径", result["report_text"])
        self.assertNotIn("report_text_template", result)
        self.assertNotIn("report_text_llm", result)

    def test_question_routing_maps_to_q2_height(self) -> None:
        result = run_single_tree_analysis(
            "demo_tree.laz",
            question="这棵树的树高是多少？",
            backend=MockBackend(),
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["input"]["format"], "laz")
        self.assertEqual(result["request"]["resolved_intent"], "q2_height")
        self.assertEqual(list(result["tool_results"]), ["estimate_height"])
        self.assertEqual(result["json_summary"]["height_m"]["status"], "success")
        self.assertIn("树高", result["report_text"])

    def test_report_question_runs_three_fixed_tools(self) -> None:
        result = run_single_tree_analysis(
            "demo_tree.las",
            question="给我一个简短单木报告",
            backend=MockBackend(),
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["request"]["resolved_intent"], "tree_report_q123")
        self.assertEqual(
            list(result["tool_results"]),
            ["estimate_dbh", "estimate_height", "estimate_crown_width"],
        )
        self.assertIn("胸径", result["report_text"])
        self.assertIn("树高", result["report_text"])
        self.assertIn("冠幅", result["report_text"])
        self.assertIn("未进行倾斜", result["report_text"])

    def test_task_id_priority_over_question(self) -> None:
        result = run_single_tree_analysis(
            "demo_tree.las",
            task_id="q1_dbh",
            question="这棵树的树高是多少？",
            backend=MockBackend(),
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["request"]["resolved_intent"], "q1_dbh")
        self.assertEqual(list(result["tool_results"]), ["estimate_dbh"])

    def test_ambiguous_question_returns_failed(self) -> None:
        result = run_single_tree_analysis(
            "demo_tree.las",
            question="这棵树的胸径和树高是多少？",
            backend=MockBackend(),
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["tool_results"], {})
        self.assertIsNone(result["request"]["resolved_intent"])
        self.assertIn("无法从问题中唯一解析任务意图", result["message"])
        self.assertIn("未完成", result["report_text"])

    def test_unknown_task_id_returns_failed(self) -> None:
        result = run_single_tree_analysis(
            "demo_tree.las",
            task_id="q9_unknown",
            backend=MockBackend(),
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("不支持的 task_id", result["message"])
        self.assertEqual(result["tool_results"], {})

    def test_report_failure_propagates_and_keeps_partial_results(self) -> None:
        backend = MockBackend(failures={"estimate_height": "Height failed."})
        result = run_single_tree_analysis(
            "demo_tree.las",
            task_id="tree_report_q123",
            backend=backend,
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(list(result["tool_results"]), ["estimate_dbh", "estimate_height"])
        self.assertEqual(result["json_summary"]["dbh_cm"]["status"], "success")
        self.assertEqual(result["json_summary"]["height_m"]["status"], "failed")
        self.assertIsNone(result["json_summary"]["crown_width_m"])
        self.assertIn("Height failed", result["message"])

    def test_default_direct_geometry_backend_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            point_cloud_path = Path(temp_dir) / "synthetic_tree.las"
            self._write_demo_tree_point_cloud(point_cloud_path)

            result = run_single_tree_analysis(
                str(point_cloud_path),
                question="给我一个简短单木报告",
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["request"]["resolved_intent"], "tree_report_q123")
        self.assertEqual(
            list(result["tool_results"]),
            ["estimate_dbh", "estimate_height", "estimate_crown_width"],
        )
        self.assertTrue(
            result["tool_results"]["estimate_dbh"]["extra"]["trunk_only_filtering_enabled"]
        )
        self.assertIn("direct geometry baseline", result["report_text"])

    def test_local_llm_report_success_adds_optional_fields(self) -> None:
        with patch(
            "forestagent.mvp.single_tree_analysis.verbalize_json_summary_with_ollama",
            return_value="这棵树估计胸径为 12.00 cm。",
        ) as verbalizer:
            result = run_single_tree_analysis(
                "demo_tree.las",
                task_id="q1_dbh",
                backend=MockBackend(),
                use_local_llm_report=True,
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["report_text"], result["report_text_template"])
        self.assertEqual(result["report_text_llm"], "这棵树估计胸径为 12.00 cm。")
        self.assertEqual(verbalizer.call_args.kwargs["timeout_seconds"], 60.0)

    def test_local_llm_report_custom_timeout_is_forwarded(self) -> None:
        with patch(
            "forestagent.mvp.single_tree_analysis.verbalize_json_summary_with_ollama",
            return_value="这棵树估计胸径为 12.00 cm。",
        ) as verbalizer:
            run_single_tree_analysis(
                "demo_tree.las",
                task_id="q1_dbh",
                backend=MockBackend(),
                use_local_llm_report=True,
                ollama_timeout_seconds=120.0,
            )

        self.assertEqual(verbalizer.call_args.kwargs["timeout_seconds"], 120.0)

    def test_local_llm_report_failure_falls_back_to_template(self) -> None:
        with patch(
            "forestagent.mvp.single_tree_analysis.verbalize_json_summary_with_ollama",
            side_effect=OllamaVerbalizerError("Ollama request timed out."),
        ):
            result = run_single_tree_analysis(
                "demo_tree.las",
                task_id="q1_dbh",
                backend=MockBackend(),
                use_local_llm_report=True,
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["report_text"], result["report_text_template"])
        self.assertNotIn("report_text_llm", result)
        self.assertIn("Local Ollama verbalizer fallback happened", result["message"])
        self.assertIn("timed out", result["message"])

    def test_without_local_llm_flag_behavior_stays_unchanged(self) -> None:
        with patch(
            "forestagent.mvp.single_tree_analysis.verbalize_json_summary_with_ollama",
        ) as verbalizer:
            result = run_single_tree_analysis(
                "demo_tree.las",
                task_id="q1_dbh",
                backend=MockBackend(),
            )

        self.assertEqual(result["status"], "success")
        self.assertNotIn("report_text_template", result)
        self.assertNotIn("report_text_llm", result)
        verbalizer.assert_not_called()

    @staticmethod
    def _write_demo_tree_point_cloud(path: Path) -> None:
        trunk_height_m = 8.0
        trunk_radius_m = 0.12
        crown_radius_x_m = 2.0
        crown_radius_y_m = 1.5

        trunk_z = np.linspace(0.0, trunk_height_m, 180)
        trunk_angles = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
        trunk_x = (trunk_radius_m * np.cos(trunk_angles))[None, :].repeat(trunk_z.size, axis=0)
        trunk_y = (trunk_radius_m * np.sin(trunk_angles))[None, :].repeat(trunk_z.size, axis=0)
        trunk_points = np.column_stack(
            (
                trunk_x.reshape(-1),
                trunk_y.reshape(-1),
                np.repeat(trunk_z, trunk_angles.size),
            )
        )

        crown_levels = np.linspace(5.2, 8.0, 36)
        crown_angles = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
        crown_points: list[np.ndarray] = []
        for level in crown_levels:
            scale = max(0.25, 1.0 - ((level - 6.6) / 2.1) ** 2)
            x = crown_radius_x_m * scale * np.cos(crown_angles)
            y = crown_radius_y_m * scale * np.sin(crown_angles)
            z = np.full_like(x, level)
            crown_points.append(np.column_stack((x, y, z)))
        points = np.vstack([trunk_points, *crown_points]).astype(np.float64)

        las = laspy.create(file_version="1.4", point_format=6)
        las.x = points[:, 0]
        las.y = points[:, 1]
        las.z = points[:, 2]
        las.write(path)


if __name__ == "__main__":
    unittest.main()
