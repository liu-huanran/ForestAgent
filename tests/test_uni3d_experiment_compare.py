"""Tests for offline Uni3D experiment comparison helpers."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from forestagent.experiments.uni3d_experiment_compare import compare_experiments


class Uni3DExperimentCompareTests(unittest.TestCase):
    def test_compare_aggregates_probe_metrics_from_fake_experiments(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_a = _fake_config(root / "a", "exp_a")
            config_b = _fake_config(root / "b", "exp_b")
            _write_probe(config_a["paths"]["probe_output_dir"], metric_value=0.25)
            _write_probe(config_b["paths"]["probe_output_dir"], metric_value=0.75)

            comparison = compare_experiments([config_a, config_b])

            self.assertEqual(comparison["experiment_count"], 2)
            aggregates = comparison["aggregate_probe_metrics"]
            self.assertEqual(len(aggregates), 1)
            self.assertEqual(aggregates[0]["metric_name"], "accuracy")
            self.assertAlmostEqual(aggregates[0]["mean"], 0.5)


def _fake_config(root: Path, name: str) -> dict:
    return {
        "experiment": {"name": name, "stage": "probed", "version": "v1"},
        "paths": {
            "raw_las_dir": str(root / "las"),
            "npy_input_dir": str(root / "npy"),
            "embedding_output_dir": str(root / "embeddings"),
            "similarity_output_dir": str(root / "similarity"),
            "probe_output_dir": str(root / "probe"),
            "label_path": str(root / "parameters.xlsx"),
            "checkpoint_path": str(root / "model.pt"),
            "uni3d_repo_path": str(root / "Uni3D"),
        },
        "conversion": {"enabled": False, "output_mode": "legacy_mixed", "color_policy": "mixed_historical", "num_points": 10000, "seed": 42, "pattern": "*.las"},
        "extractor": {
            "pc_model": "eva02_base_patch14_448",
            "pc_feat_dim": 768,
            "embed_dim": 1024,
            "pc_encoder_dim": 512,
            "num_group": 512,
            "group_size": 64,
            "device": "cuda",
        },
        "analysis": {"similarity": {}, "probe": {}},
        "metadata": {},
    }


def _write_probe(path_text: str, metric_value: float) -> None:
    path = Path(path_text)
    path.mkdir(parents=True)
    with (path / "probe_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["task", "feature_set", "split_type", "metric_name", "metric_value", "status"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "task": "species",
                "feature_set": "uni3d",
                "split_type": "random",
                "metric_name": "accuracy",
                "metric_value": metric_value,
                "status": "ok",
            }
        )
    (path / "probe_summary.json").write_text(json.dumps({"warnings": []}), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()

