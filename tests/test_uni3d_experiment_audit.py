"""Tests for read-only Uni3D experiment artifact audit."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from forestagent.experiments.uni3d_experiment_audit import audit_experiment


class Uni3DExperimentAuditTests(unittest.TestCase):
    def test_audit_reads_fake_artifacts_and_marks_missing_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = _fake_config(root)
            _write_conversion_artifacts(root / "npy")
            _write_extraction_artifacts(root / "embeddings")
            _write_similarity_artifacts(root / "similarity")

            report = audit_experiment(config)

            self.assertEqual(report["checks"]["conversion"]["status"], "PASS")
            self.assertEqual(report["checks"]["extraction"]["status"], "PASS")
            self.assertEqual(report["checks"]["similarity"]["status"], "PASS")
            self.assertEqual(report["checks"]["probe"]["status"], "NOT_AVAILABLE")
            self.assertEqual(report["overall_status"], "WARN")

    def test_audit_detects_point_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = _fake_config(root)
            npy_dir = root / "npy"
            npy_dir.mkdir()
            np.save(npy_dir / "tree.npy", np.zeros((5, 3), dtype=np.float32))
            (npy_dir / "manifest.jsonl").write_text("", encoding="utf-8")
            (npy_dir / "summary.json").write_text(json.dumps({"failure_count": 0}), encoding="utf-8")

            report = audit_experiment(config, stage="conversion")

            self.assertEqual(report["checks"]["conversion"]["status"], "WARN")
            self.assertEqual(report["checks"]["conversion"]["point_count_mismatch"], 1)


def _fake_config(root: Path) -> dict:
    return {
        "experiment": {"name": "fake_audit", "stage": "embedded", "version": "v1"},
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
        "conversion": {"enabled": True, "output_mode": "xyz_only", "color_policy": "drop", "num_points": 10, "seed": 42, "pattern": "*.las"},
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


def _write_conversion_artifacts(path: Path) -> None:
    path.mkdir()
    np.save(path / "tree.npy", np.zeros((10, 3), dtype=np.float32))
    (path / "manifest.jsonl").write_text(json.dumps({"tree_id": "tree", "status": "success"}) + "\n", encoding="utf-8")
    (path / "summary.json").write_text(json.dumps({"success_count": 1, "failure_count": 0}), encoding="utf-8")


def _write_extraction_artifacts(path: Path) -> None:
    path.mkdir()
    metadata = json.dumps({"mock": False, "pc_model": "eva02_base_patch14_448", "embed_dim": 1024})
    np.savez(path / "001_tree.npz", embedding_l2=np.ones((1, 1024), dtype=np.float32), metadata=metadata)
    (path / "summary.json").write_text(json.dumps({"success_count": 1, "failure_count": 0}), encoding="utf-8")


def _write_similarity_artifacts(path: Path) -> None:
    path.mkdir()
    (path / "similarity_summary.json").write_text(
        json.dumps({"sample_count": 1, "invalid_count": 0, "all_pairwise_similarity_near_one": False}),
        encoding="utf-8",
    )
    (path / "nearest_neighbors.json").write_text("[]", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()

