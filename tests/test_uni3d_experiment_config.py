"""Tests for offline Uni3D experiment config and planning helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from forestagent.experiments.uni3d_experiment_config import (
    ConfigError,
    build_experiment_plan,
    load_experiment_config,
)


class Uni3DExperimentConfigTests(unittest.TestCase):
    def test_loads_sample_yaml_config(self) -> None:
        config = load_experiment_config("configs/uni3d_experiments/all_tls_xyz_only_v1.yaml")

        self.assertEqual(config["experiment"]["name"], "all_tls_xyz_only_v1")
        self.assertEqual(config["conversion"]["output_mode"], "xyz_only")

    def test_missing_required_field_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.json"
            path.write_text(json.dumps({"experiment": {"name": "bad"}}), encoding="utf-8")

            with self.assertRaisesRegex(ConfigError, "Missing required mapping"):
                load_experiment_config(path)

    def test_plan_generation_includes_offline_commands(self) -> None:
        config = _fake_config()

        plan = build_experiment_plan(config)
        commands = [item["command"] for item in plan["commands"]]

        self.assertEqual([item["stage"] for item in plan["commands"]], ["convert", "extract", "similarity", "probe", "probe"])
        self.assertTrue(any("convert_las_to_uni3d_npy.py" in command for command in commands))
        self.assertTrue(any("batch_extract_uni3d_embeddings.py" in command for command in commands))
        self.assertTrue(any("analyze_uni3d_embedding_similarity.py" in command for command in commands))
        self.assertTrue(any("probe_uni3d_embeddings_strict.py" in command for command in commands))
        self.assertTrue(plan["dry_run"])

    def test_species_palette_warning_is_explicit(self) -> None:
        config = _fake_config()
        config["conversion"]["color_policy"] = "species_palette"

        plan = build_experiment_plan(config)

        self.assertTrue(any("label leakage" in warning for warning in plan["warnings"]))


def _fake_config() -> dict:
    return {
        "experiment": {"name": "fake_xyz", "stage": "planned", "version": "v1"},
        "paths": {
            "raw_las_dir": "/tmp/raw_las",
            "npy_input_dir": "/tmp/npy",
            "embedding_output_dir": "/tmp/embeddings",
            "similarity_output_dir": "/tmp/similarity",
            "probe_output_dir": "/tmp/probe",
            "label_path": "data/parameters.xlsx",
            "checkpoint_path": "/tmp/checkpoint.pt",
            "uni3d_repo_path": "/tmp/Uni3D",
        },
        "conversion": {
            "enabled": True,
            "output_mode": "xyz_only",
            "color_policy": "drop",
            "num_points": 10000,
            "seed": 42,
            "pattern": "*.las",
            "recursive": True,
            "skip_existing": True,
        },
        "extractor": {
            "pc_model": "eva02_base_patch14_448",
            "pc_feat_dim": 768,
            "embed_dim": 1024,
            "pc_encoder_dim": 512,
            "num_group": 512,
            "group_size": 64,
            "device": "cuda",
            "pattern": "*.npy",
            "recursive": True,
            "skip_existing": True,
        },
        "analysis": {
            "similarity": {"top_k": 5, "save_full_matrix": False, "no_full_matrix": True},
            "probe": {"tasks": ["all"], "feature_sets": ["all"], "splits": ["random", "leave_one_site_out"], "seeds": [0]},
        },
        "metadata": {"expected_sample_count": 2, "current_status": "planned"},
    }


if __name__ == "__main__":
    unittest.main()

