"""Smoke test for the frozen Baseline V0 CLI path."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import laspy
import numpy as np

from forestagent import BASELINE_TAG
from forestagent.mvp import run_single_tree_analysis


class BaselineV0SmokeTests(unittest.TestCase):
    def test_cli_analyze_tree_q123_report_smoke(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]

        with tempfile.TemporaryDirectory() as temp_dir:
            point_cloud_path = Path(temp_dir) / "synthetic_tree.las"
            self._write_demo_tree_point_cloud(point_cloud_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "forestagent.cli",
                    "analyze-tree",
                    "--point-cloud",
                    str(point_cloud_path),
                    "--task-id",
                    "tree_report_q123",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=True,
            )

        payload = json.loads(completed.stdout)

        self.assertEqual(BASELINE_TAG, "baseline-v0")
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["request"]["resolved_intent"], "tree_report_q123")
        self.assertEqual(payload["input"]["format"], "las")
        self.assertEqual(
            list(payload["tool_results"]),
            ["estimate_dbh", "estimate_height", "estimate_crown_width"],
        )
        self.assertEqual(
            sorted(payload["json_summary"].keys()),
            ["crown_width_m", "dbh_cm", "height_m"],
        )
        self.assertIn("direct geometry baseline", payload["report_text"])
        self.assertNotIn("report_text_llm", payload)
        self.assertNotIn("report_text_template", payload)

    def test_optional_ollama_verbalizer_mode_is_still_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            point_cloud_path = Path(temp_dir) / "synthetic_tree.las"
            self._write_demo_tree_point_cloud(point_cloud_path)

            with patch(
                "forestagent.mvp.single_tree_analysis.verbalize_json_summary_with_ollama",
                return_value="这是本地 Ollama verbalizer 生成的示例报告。",
            ):
                payload = run_single_tree_analysis(
                    str(point_cloud_path),
                    task_id="tree_report_q123",
                    use_local_llm_report=True,
                )

        self.assertEqual(BASELINE_TAG, "baseline-v0")
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["request"]["resolved_intent"], "tree_report_q123")
        self.assertEqual(payload["report_text"], payload["report_text_template"])
        self.assertEqual(
            payload["report_text_llm"],
            "这是本地 Ollama verbalizer 生成的示例报告。",
        )
        self.assertEqual(
            list(payload["tool_results"]),
            ["estimate_dbh", "estimate_height", "estimate_crown_width"],
        )

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
