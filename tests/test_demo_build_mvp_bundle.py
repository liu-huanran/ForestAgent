"""Tests for the single-tree MVP demo bundle generator."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from forestagent.backends.mock_backend import MockBackend
from forestagent.demo_build_mvp_bundle import build_demo_bundle


class DemoBuildMvpBundleTests(unittest.TestCase):
    def test_build_demo_bundle_writes_manifest_and_case_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "demo_bundle"
            manifest = build_demo_bundle(
                output_dir=output_dir,
                backend=MockBackend(),
                sample_specs=[
                    {
                        "sample_id": "demo_001",
                        "point_cloud": "demo_tree.las",
                        "reason": "unit test sample",
                    }
                ],
                cases=[
                    {"case_id": "q1_dbh", "task_id": "q1_dbh", "question": None},
                    {
                        "case_id": "unsupported_question",
                        "task_id": None,
                        "question": "这棵树有倒伏风险吗？",
                    },
                ],
            )

            self.assertEqual(manifest["sample_count"], 1)
            self.assertEqual(manifest["case_count_per_sample"], 2)
            self.assertTrue((output_dir / "manifest.json").exists())
            self.assertTrue((output_dir / "demo_001" / "q1_dbh.json").exists())
            self.assertTrue((output_dir / "demo_001" / "unsupported_question.json").exists())
            self.assertEqual(manifest["samples"][0]["cases"][0]["status"], "success")
            self.assertEqual(manifest["samples"][0]["cases"][1]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
