"""Tests for offline Uni3D embedding analysis/query helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.analyze_uni3d_embedding_dataset import main as analysis_main
from scripts.query_uni3d_embedding_neighbors import query_neighbors


class Uni3DEmbeddingAnalysisToolTests(unittest.TestCase):
    def test_query_neighbors_returns_top_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            embedding_dir = Path(temp_dir)
            self._write_embedding(embedding_dir / "001_a.npz", "a", [1.0, 0.0, 0.0])
            self._write_embedding(embedding_dir / "002_b.npz", "b", [0.9, 0.1, 0.0])
            self._write_embedding(embedding_dir / "003_c.npz", "c", [0.0, 1.0, 0.0])

            result = query_neighbors(embedding_dir=embedding_dir, sample_id="a", top_k=2)

            self.assertEqual(result["query_tree_id"], "a")
            self.assertEqual(result["nearest"][0]["tree_id"], "b")
            self.assertGreater(result["nearest"][0]["cosine_similarity"], result["nearest"][1]["cosine_similarity"])

    def test_analysis_script_writes_health_outputs_without_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            embedding_dir = root / "embeddings"
            output_dir = root / "analysis"
            embedding_dir.mkdir()
            self._write_embedding(embedding_dir / "001_a.npz", "a", [1.0, 0.0, 0.0])
            self._write_embedding(embedding_dir / "002_b.npz", "b", [0.99, 0.01, 0.0])
            self._write_embedding(embedding_dir / "003_c.npz", "c", [0.0, 1.0, 0.0])

            exit_code = analysis_main(
                [
                    "--embedding-dir",
                    str(embedding_dir),
                    "--output-dir",
                    str(output_dir),
                    "--near-duplicate-threshold",
                    "0.99",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "dataset_summary.json").is_file())
            self.assertTrue((output_dir / "near_duplicate_pairs.csv").is_file())
            self.assertTrue((output_dir / "outlier_samples.csv").is_file())
            self.assertTrue((output_dir / "nearest_neighbors.json").is_file())
            self.assertTrue((output_dir / "pca_coordinates.csv").is_file())
            self.assertTrue((output_dir / "pca_embedding.svg").is_file())
            summary = json.loads((output_dir / "dataset_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["sample_count"], 3)
            pairs = json.loads((output_dir / "near_duplicate_pairs.json").read_text(encoding="utf-8"))
            self.assertEqual(pairs[0]["left_tree_id"], "a")
            self.assertEqual(pairs[0]["right_tree_id"], "b")

    @staticmethod
    def _write_embedding(path: Path, tree_id: str, values: list[float]) -> None:
        embedding = np.asarray(values, dtype=np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        np.savez_compressed(
            path,
            embedding_l2=embedding[None, :],
            embedding_raw=embedding[None, :],
            tree_id=np.asarray(tree_id),
            source_path=np.asarray(f"/fake/{tree_id}.npy"),
            metadata=np.asarray("{}"),
        )


if __name__ == "__main__":
    unittest.main()
