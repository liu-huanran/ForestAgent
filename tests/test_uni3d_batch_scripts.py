"""Lightweight tests for Uni3D batch/similarity scripts.

These tests use only fake .npy inputs and fake embedding .npz outputs. They do
not instantiate the real Uni3DFeatureExtractor, do not require a checkpoint, and
do not require CUDA or pointnet2_ops.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.analyze_uni3d_embedding_similarity import main as similarity_main
from scripts.batch_extract_uni3d_embeddings import (
    discover_input_paths,
    load_npy_point_cloud,
    main as batch_main,
    normalize_limit,
)


class Uni3DBatchScriptTests(unittest.TestCase):
    def test_load_npy_point_cloud_accepts_xyz(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tree_xyz.npy"
            xyz = np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
            np.save(path, xyz)

            loaded = load_npy_point_cloud(path)

            self.assertEqual(loaded.tree_id, "tree_xyz")
            self.assertEqual(loaded.input_shape, (2, 3))
            self.assertTrue(np.allclose(loaded.xyz, xyz))
            self.assertIsNone(loaded.rgb)

    def test_load_npy_point_cloud_accepts_xyzrgb(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tree_xyzrgb.npy"
            xyzrgb = np.asarray(
                [
                    [0.0, 1.0, 2.0, 0.1, 0.2, 0.3],
                    [3.0, 4.0, 5.0, 0.4, 0.5, 0.6],
                ],
                dtype=np.float32,
            )
            np.save(path, xyzrgb)

            loaded = load_npy_point_cloud(path)

            self.assertEqual(loaded.input_shape, (2, 6))
            self.assertTrue(np.allclose(loaded.xyz, xyzrgb[:, :3]))
            self.assertIsNotNone(loaded.rgb)
            self.assertTrue(np.allclose(loaded.rgb, xyzrgb[:, 3:6]))

    def test_load_npy_point_cloud_rejects_invalid_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.npy"
            np.save(path, np.zeros((3, 4), dtype=np.float32))

            with self.assertRaisesRegex(ValueError, r"shape \[N, 3\] or \[N, 6\]"):
                load_npy_point_cloud(path)

    def test_discover_input_paths_supports_full_limit_and_recursive_search(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nested = root / "nested"
            nested.mkdir()
            np.save(root / "a.npy", np.zeros((2, 3), dtype=np.float32))
            np.save(nested / "b.npy", np.zeros((2, 3), dtype=np.float32))

            all_paths = discover_input_paths(
                input_dir=root,
                manifest_path=None,
                pattern="*.npy",
                recursive=True,
                limit=normalize_limit(None),
            )
            zero_paths = discover_input_paths(
                input_dir=root,
                manifest_path=None,
                pattern="*.npy",
                recursive=True,
                limit=normalize_limit(0),
            )
            top_level_only = discover_input_paths(
                input_dir=root,
                manifest_path=None,
                pattern="*.npy",
                recursive=False,
                limit=normalize_limit(None),
            )

            self.assertEqual(len(all_paths), 2)
            self.assertEqual(len(zero_paths), 2)
            self.assertEqual(len(top_level_only), 1)

    def test_batch_skip_existing_does_not_require_checkpoint_or_forward(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "inputs"
            output_dir = root / "outputs"
            input_dir.mkdir()
            output_dir.mkdir()
            np.save(input_dir / "tree.npy", np.zeros((4, 3), dtype=np.float32))
            self._write_embedding(output_dir / "001_tree.npz", "tree", [1.0, 0.0, 0.0])

            exit_code = batch_main(
                [
                    "--input-dir",
                    str(input_dir),
                    "--output-dir",
                    str(output_dir),
                    "--checkpoint-path",
                    str(root / "missing.pt"),
                    "--uni3d-repo-path",
                    str(root / "missing_uni3d"),
                    "--num-group",
                    "4",
                    "--group-size",
                    "2",
                    "--limit",
                    "0",
                    "--skip-existing",
                ]
            )

            self.assertEqual(exit_code, 0)
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["skipped_count"], 1)
            self.assertEqual(summary["failure_count"], 0)

    def test_batch_invalid_shape_is_recorded_without_forward(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "inputs"
            output_dir = root / "outputs"
            input_dir.mkdir()
            np.save(input_dir / "bad.npy", np.zeros((3, 4), dtype=np.float32))

            exit_code = batch_main(
                [
                    "--input-dir",
                    str(input_dir),
                    "--output-dir",
                    str(output_dir),
                    "--checkpoint-path",
                    str(root / "missing.pt"),
                    "--uni3d-repo-path",
                    str(root / "missing_uni3d"),
                    "--limit",
                    "0",
                ]
            )

            self.assertEqual(exit_code, 2)
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["success_count"], 0)
            self.assertEqual(summary["failure_count"], 1)
            self.assertIn("shape [N, 3] or [N, 6]", summary["results"][0]["failure_reason"])

    def test_similarity_analysis_reads_fake_embeddings_and_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            embedding_dir = root / "embeddings"
            output_dir = root / "analysis"
            embedding_dir.mkdir()
            self._write_embedding(embedding_dir / "tree_a.npz", "tree_a", [1.0, 0.0, 0.0])
            self._write_embedding(embedding_dir / "tree_b.npz", "tree_b", [0.0, 1.0, 0.0])
            self._write_embedding(embedding_dir / "tree_c.npz", "tree_c", [1.0, 1.0, 0.0])

            exit_code = similarity_main(
                [
                    "--embedding-dir",
                    str(embedding_dir),
                    "--output-dir",
                    str(output_dir),
                    "--top-k",
                    "2",
                ]
            )

            self.assertEqual(exit_code, 0)
            summary_path = output_dir / "similarity_summary.json"
            neighbors_path = output_dir / "nearest_neighbors.json"
            matrix_path = output_dir / "cosine_similarity.csv"
            self.assertTrue(summary_path.is_file())
            self.assertTrue(neighbors_path.is_file())
            self.assertTrue(matrix_path.is_file())

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["sample_count"], 3)
            self.assertEqual(summary["embedding_dim"], 3)
            self.assertEqual(summary["invalid_count"], 0)
            self.assertFalse(summary["all_pairwise_similarity_near_one"])
            self.assertAlmostEqual(summary["pairwise_similarity"]["min"], 0.0, places=6)
            self.assertGreater(summary["pairwise_similarity"]["max"], 0.7)

            neighbors = json.loads(neighbors_path.read_text(encoding="utf-8"))
            self.assertEqual(len(neighbors), 3)
            self.assertEqual(len(neighbors[0]["nearest"]), 2)

    def test_similarity_analysis_large_sample_can_skip_full_matrix_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            embedding_dir = root / "embeddings"
            output_dir = root / "analysis"
            embedding_dir.mkdir()
            for index in range(6):
                values = [float(index + 1), 1.0, 0.5]
                self._write_embedding(embedding_dir / f"tree_{index}.npz", f"tree_{index}", values)

            exit_code = similarity_main(
                [
                    "--embedding-dir",
                    str(embedding_dir),
                    "--output-dir",
                    str(output_dir),
                    "--top-k",
                    "2",
                    "--full-matrix-threshold",
                    "3",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertFalse((output_dir / "cosine_similarity.csv").exists())
            summary = json.loads((output_dir / "similarity_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["sample_count"], 6)
            self.assertEqual(summary["analyzed_sample_count"], 6)
            self.assertIsNone(summary["cosine_similarity_csv_path"])

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
