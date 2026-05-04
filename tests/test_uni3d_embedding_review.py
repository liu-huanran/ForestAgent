"""Tests for offline Uni3D embedding quantitative review script."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.review_uni3d_embedding_quantitative import main as review_main


class Uni3DEmbeddingReviewTests(unittest.TestCase):
    def test_review_script_writes_table_and_summary_with_npy_stats(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            npy_dir = root / "npy"
            output_dir = root / "review"
            npy_dir.mkdir()
            np.save(
                npy_dir / "160_01.npy",
                np.asarray(
                    [
                        [0.0, 0.0, 0.0, 0.1, 0.2, 0.3],
                        [1.0, 2.0, 3.0, 0.4, 0.5, 0.6],
                    ],
                    dtype=np.float32,
                ),
            )
            np.save(
                npy_dir / "160_02.npy",
                np.asarray([[0.0, 0.0, 1.0], [2.0, 2.0, 5.0]], dtype=np.float32),
            )

            extraction_summary = root / "summary.json"
            nearest = root / "nearest_neighbors.json"
            near = root / "near_duplicate_pairs.json"
            outliers = root / "outlier_samples.json"
            similarity = root / "similarity_summary.json"
            self._write_json(
                extraction_summary,
                {
                    "results": [
                        self._summary_row("160_01", [2, 6], "provided"),
                        self._summary_row("160_02", [2, 3], "fallback_constant"),
                    ]
                },
            )
            self._write_json(
                nearest,
                [
                    {
                        "tree_id": "160_01",
                        "nearest": [{"tree_id": "160_02", "cosine_similarity": 0.99}],
                    },
                    {
                        "tree_id": "160_02",
                        "nearest": [{"tree_id": "160_01", "cosine_similarity": 0.99}],
                    },
                ],
            )
            self._write_json(
                near,
                [{"left_tree_id": "160_01", "right_tree_id": "160_02", "cosine_similarity": 0.99}],
            )
            self._write_json(
                outliers,
                [{"tree_id": "160_02", "mean_similarity_to_others": 0.2, "max_neighbor_similarity": 0.99}],
            )
            self._write_json(
                similarity,
                {"sample_count": 2, "embedding_dim": 3, "invalid_count": 0},
            )

            exit_code = review_main(
                [
                    "--extraction-summary",
                    str(extraction_summary),
                    "--similarity-summary",
                    str(similarity),
                    "--nearest-neighbors",
                    str(nearest),
                    "--near-duplicates",
                    str(near),
                    "--outliers",
                    str(outliers),
                    "--npy-input-dir",
                    str(npy_dir),
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            review = json.loads((output_dir / "review_table.json").read_text(encoding="utf-8"))
            self.assertEqual(len(review), 2)
            first = next(row for row in review if row["tree_id"] == "160_01")
            second = next(row for row in review if row["tree_id"] == "160_02")
            self.assertEqual(first["candidate_type"], "near_duplicate")
            self.assertEqual(second["candidate_type"], "both")
            self.assertEqual(first["xyz_range"], [1.0, 2.0, 3.0])
            self.assertIsNotNone(first["rgb_mean"])
            self.assertEqual(second["z_range"], 4.0)
            summary = json.loads((output_dir / "review_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["missing_input_file_count"], 0)
            self.assertEqual(summary["candidate_counts"]["both"], 1)
            self.assertTrue((output_dir / "top_near_duplicate_review.csv").is_file())
            self.assertTrue((output_dir / "top_outlier_review.csv").is_file())

    @staticmethod
    def _summary_row(tree_id: str, input_shape: list[int], rgb_source: str) -> dict[str, object]:
        return {
            "tree_id": tree_id,
            "source_path": f"/server/path/{tree_id}.npy",
            "output_path": f"/server/out/{tree_id}.npz",
            "input_shape": input_shape,
            "embedding_raw_l2_norm": [10.0],
            "embedding_l2_l2_norm": [1.0],
            "metadata": {
                "input_shape": input_shape,
                "point_count": input_shape[0],
                "rgb_source": rgb_source,
                "rgb_fallback": None if rgb_source == "provided" else 0.4,
                "centroid": [0.0, 0.0, 0.0],
                "scale_radius": 1.0,
            },
        }

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
