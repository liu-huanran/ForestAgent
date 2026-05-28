"""Tests for Geometry vs Uni3D nearest-neighbor comparison."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.compare_geometry_uni3d_neighbors import (
    METHOD_GEOMETRY,
    METHOD_FUSED,
    METHOD_UNI3D,
    run_comparison,
)


class GeometryUni3DNeighborComparisonTests(unittest.TestCase):
    def test_comparison_writes_expected_outputs_and_rankings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            embedding_dir = root / "embeddings"
            output_dir = root / "reports"
            embedding_dir.mkdir()

            self._write_embedding(embedding_dir / "001_a.npz", "a", [1.0, 0.0, 0.0])
            self._write_embedding(embedding_dir / "002_b.npz", "b", [0.0, 1.0, 0.0])
            self._write_embedding(embedding_dir / "003_c.npz", "c", [0.99, 0.01, 0.0])
            self._write_embedding(embedding_dir / "004_d.npz", "d", [0.2, 0.8, 0.0])
            self._write_embedding(embedding_dir / "005_missing_label.npz", "missing_label", [0.3, 0.7, 0.0])

            label_path = root / "labels.csv"
            self._write_labels(label_path)
            near_path = root / "near_duplicate_pairs.json"
            near_path.write_text(
                json.dumps([{"left_tree_id": "a", "right_tree_id": "c"}]),
                encoding="utf-8",
            )

            result = run_comparison(
                embedding_dir=embedding_dir,
                label_path=label_path,
                output_dir=output_dir,
                top_k=2,
                near_duplicate_source=near_path,
            )

            self.assertEqual(result["aggregate_summary"]["matched_sample_count"], 4)
            self.assertEqual(
                result["aggregate_summary"]["alignment_report"]["embedding_without_label_count"],
                1,
            )
            self.assertEqual(
                result["aggregate_summary"]["alignment_report"]["label_without_embedding_count"],
                1,
            )
            self.assertEqual(
                result["neighbor_by_method"]["a"]["methods"][METHOD_GEOMETRY][0]["neighbor_tree_id"],
                "b",
            )
            self.assertEqual(
                result["neighbor_by_method"]["a"]["methods"][METHOD_UNI3D][0]["neighbor_tree_id"],
                "c",
            )
            self.assertIn(METHOD_FUSED, result["neighbor_by_method"]["a"]["methods"])

            neighbor_rows = self._read_csv(output_dir / "neighbor_lists.csv")
            a_uni3d_c = next(
                row
                for row in neighbor_rows
                if row["query_tree_id"] == "a"
                and row["method"] == METHOD_UNI3D
                and row["neighbor_tree_id"] == "c"
            )
            self.assertEqual(a_uni3d_c["near_duplicate_flag"], "true")
            self.assertGreater(float(a_uni3d_c["geometry_distance"]), 0.0)

            overlap_rows = self._read_csv(output_dir / "per_query_overlap.csv")
            a_overlap = next(
                row
                for row in overlap_rows
                if row["query_tree_id"] == "a"
                and row["method_left"] == METHOD_GEOMETRY
                and row["method_right"] == METHOD_UNI3D
            )
            self.assertEqual(int(a_overlap["top_k"]), 2)
            self.assertIn("jaccard", a_overlap)

            for name in (
                "neighbor_lists.json",
                "aggregate_summary.json",
                "geometry_difference_summary.csv",
                "site_species_summary.csv",
                "near_duplicate_flags.csv",
                "manual_review_candidates.csv",
                "experiment_record.md",
            ):
                self.assertTrue((output_dir / name).is_file(), name)

    def test_script_does_not_import_torch(self) -> None:
        script = Path("scripts/compare_geometry_uni3d_neighbors.py").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("import torch", script)
        self.assertNotIn("Uni3DFeatureExtractor", script)

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
            metadata=np.asarray(json.dumps({"rgb_source": "fallback_constant"})),
        )

    @staticmethod
    def _write_labels(path: Path) -> None:
        rows = [
            {
                "tree_id": "a",
                "site_id": "site1",
                "species": "oak",
                "dbh_cm": "10.0",
                "height_m": "5.0",
                "crown_ew_m": "1.0",
                "crown_ns_m": "1.0",
            },
            {
                "tree_id": "b",
                "site_id": "site1",
                "species": "oak",
                "dbh_cm": "10.1",
                "height_m": "5.0",
                "crown_ew_m": "1.0",
                "crown_ns_m": "1.0",
            },
            {
                "tree_id": "c",
                "site_id": "site2",
                "species": "pine",
                "dbh_cm": "30.0",
                "height_m": "20.0",
                "crown_ew_m": "5.0",
                "crown_ns_m": "5.0",
            },
            {
                "tree_id": "d",
                "site_id": "site2",
                "species": "pine",
                "dbh_cm": "12.0",
                "height_m": "6.0",
                "crown_ew_m": "1.5",
                "crown_ns_m": "1.2",
            },
            {
                "tree_id": "missing_embedding",
                "site_id": "site3",
                "species": "birch",
                "dbh_cm": "11.0",
                "height_m": "7.0",
                "crown_ew_m": "2.0",
                "crown_ns_m": "2.0",
            },
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
