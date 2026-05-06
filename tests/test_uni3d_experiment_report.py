"""Tests for offline Uni3D experiment Markdown report helpers."""

from __future__ import annotations

import unittest

from forestagent.experiments.uni3d_report import generate_experiment_report, generate_memory_snippet


class Uni3DExperimentReportTests(unittest.TestCase):
    def test_report_and_memory_snippet_are_generated(self) -> None:
        config = _fake_config()
        audit = {"overall_status": "PASS", "checks": {"extraction": {"status": "PASS", "reason": "ok", "path": "/tmp/e"}}}
        comparison = {"experiment_count": 3}

        report = generate_experiment_report(config, audit=audit, comparison=comparison)
        snippet = generate_memory_snippet(config, audit=audit, comparison=comparison)

        self.assertIn("# 实验记录｜fake_report", report)
        self.assertIn("不能说 Uni3D 已经接入主系统", report)
        self.assertIn("offline experiment framework", snippet)
        self.assertIn("PASS", snippet)


def _fake_config() -> dict:
    return {
        "experiment": {
            "name": "fake_report",
            "description": "Fake report config.",
            "stage": "planned",
            "version": "v1",
            "created_by": "test",
        },
        "paths": {
            "raw_las_dir": "/tmp/las",
            "npy_input_dir": "/tmp/npy",
            "embedding_output_dir": "/tmp/embedding",
            "similarity_output_dir": "/tmp/similarity",
            "probe_output_dir": "/tmp/probe",
            "label_path": "data/parameters.xlsx",
            "checkpoint_path": "/tmp/model.pt",
            "uni3d_repo_path": "/tmp/Uni3D",
        },
        "conversion": {"enabled": True, "output_mode": "xyz_only", "color_policy": "drop"},
        "extractor": {"pc_model": "eva02", "embed_dim": 1024, "num_group": 512, "group_size": 64},
        "analysis": {"similarity": {"top_k": 5}, "probe": {"splits": ["random"], "rgb_source": "all"}},
        "metadata": {},
    }


if __name__ == "__main__":
    unittest.main()

