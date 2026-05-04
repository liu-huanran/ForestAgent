"""Tests for strict offline Uni3D downstream probe utilities.

These tests use fake embeddings and fake labels only. They do not run Uni3D
forward, do not instantiate Uni3DFeatureExtractor, and do not require a GPU.
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.probe_uni3d_embeddings_strict import (
    align_samples,
    detect_label_columns,
    filter_samples,
    fit_standardizer,
    geometry_feature_columns,
    load_embedding_records,
    load_label_table,
    main as strict_probe_main,
    make_leave_one_site_out_splits,
    make_random_split,
    transform_standardizer,
)


class Uni3DStrictProbeTests(unittest.TestCase):
    def test_load_fake_embeddings_and_align_with_label_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            embedding_dir = root / "embeddings"
            embedding_dir.mkdir()
            self._write_embedding(embedding_dir / "001_150_1.npz", "150_1", [1.0, 0.0, 0.0])
            label_path = self._write_labels(root / "labels.csv", ["150_1"])

            embeddings, invalid = load_embedding_records(embedding_dir, "embedding_l2")
            labels = load_label_table(
                label_path,
                tree_id_column=None,
                species_column=None,
                dbh_column=None,
                height_column=None,
                crown_ew_column=None,
                crown_ns_column=None,
            )
            rows, report = align_samples(
                embeddings=embeddings,
                invalid_embeddings=invalid,
                labels=labels,
                tasks=["species", "dbh"],
                site_id_from_tree_id=True,
                site_id_delimiter="_",
                group_column=None,
                review_info={},
                near_duplicate_ids=set(),
                outlier_ids=set(),
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(report["matched_count"], 1)
            self.assertEqual(rows[0].tree_id, "150_1")
            self.assertEqual(rows[0].site_id, "150")

    def test_detect_label_columns_for_current_parameter_names(self) -> None:
        columns = ["样地号", "树种", "对应的文件名", "胸径cm", "树高m", "东西冠幅m", "南北冠幅m"]
        detected = detect_label_columns(
            columns,
            tree_id_column=None,
            species_column=None,
            dbh_column=None,
            height_column=None,
            crown_ew_column=None,
            crown_ns_column=None,
        )

        self.assertEqual(detected["tree_id"], "对应的文件名")
        self.assertEqual(detected["species"], "树种")
        self.assertEqual(detected["dbh"], "胸径cm")
        self.assertEqual(detected["height"], "树高m")
        self.assertEqual(detected["crown_ew"], "东西冠幅m")
        self.assertEqual(detected["crown_ns"], "南北冠幅m")

    def test_random_split_has_no_train_test_overlap(self) -> None:
        split = make_random_split(30, test_size=0.2, seed=7)
        self.assertIsNotNone(split)
        assert split is not None

        self.assertFalse(set(split.train_indices) & set(split.test_indices))
        self.assertEqual(len(split.train_indices) + len(split.test_indices), 30)

    def test_leave_one_site_out_holds_out_one_site(self) -> None:
        rows = [
            self._sample_row("150_1"),
            self._sample_row("150_2"),
            self._sample_row("160_1"),
            self._sample_row("160_2"),
        ]

        splits = make_leave_one_site_out_splits(rows)

        self.assertEqual(len(splits), 2)
        for split in splits:
            train_sites = {rows[index].site_id for index in split.train_indices}
            test_sites = {rows[index].site_id for index in split.test_indices}
            self.assertFalse(train_sites & test_sites)

    def test_standardizer_is_fit_on_train_only(self) -> None:
        train = np.asarray([[0.0, 10.0], [2.0, 14.0]], dtype=np.float64)
        test = np.asarray([[100.0, 200.0]], dtype=np.float64)

        mean, std = fit_standardizer(train)
        train_scaled = transform_standardizer(train, mean, std)
        test_scaled = transform_standardizer(test, mean, std)

        self.assertTrue(np.allclose(train_scaled.mean(axis=0), [0.0, 0.0]))
        self.assertFalse(np.allclose(test_scaled, [[0.0, 0.0]]))

    def test_geometry_features_exclude_regression_target(self) -> None:
        columns = {
            "dbh": "胸径cm",
            "height": "树高m",
            "crown_ew": "东西冠幅m",
            "crown_ns": "南北冠幅m",
        }

        dbh_features = geometry_feature_columns("dbh", columns)
        species_features = geometry_feature_columns("species", columns)

        self.assertNotIn("胸径cm", dbh_features)
        self.assertIn("树高m", dbh_features)
        self.assertIn("胸径cm", species_features)

    def test_near_duplicate_and_outlier_filters_apply(self) -> None:
        rows = [
            self._sample_row("150_1", near=True),
            self._sample_row("150_2", outlier=True),
            self._sample_row("150_3"),
        ]

        filtered, report = filter_samples(
            rows,
            rgb_source="all",
            site_ids=None,
            exclude_near_duplicates=True,
            exclude_outliers=True,
        )

        self.assertEqual([row.tree_id for row in filtered], ["150_3"])
        self.assertEqual(report["removed_counts"]["near_duplicate"], 1)
        self.assertEqual(report["removed_counts"]["outlier"], 1)

    def test_main_outputs_classification_and_regression_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            embedding_dir, label_path = self._write_probe_fixture(root, count=40)
            output_dir = root / "probe_out"

            exit_code = strict_probe_main(
                [
                    "--embedding-dir",
                    str(embedding_dir),
                    "--label-path",
                    str(label_path),
                    "--output-dir",
                    str(output_dir),
                    "--tasks",
                    "species",
                    "dbh",
                    "--feature-sets",
                    "uni3d",
                    "geometry",
                    "uni3d_plus_geometry",
                    "--split",
                    "random",
                    "--seeds",
                    "0",
                    "--min-train-samples",
                    "10",
                    "--min-test-samples",
                    "3",
                ]
            )

            self.assertEqual(exit_code, 0)
            result_rows = self._read_csv(output_dir / "probe_results.csv")
            metric_names = {row["metric_name"] for row in result_rows if row["status"] == "ok"}
            self.assertIn("accuracy", metric_names)
            self.assertIn("macro_f1", metric_names)
            self.assertIn("R2", metric_names)
            self.assertIn("MAE", metric_names)
            self.assertTrue((output_dir / "per_split_predictions.csv").is_file())
            self.assertTrue(any((output_dir / "confusion_matrices").glob("*.json")))

    def test_main_leave_one_site_out_outputs_group_safe_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            embedding_dir, label_path = self._write_probe_fixture(root, count=40)
            output_dir = root / "probe_loso"

            exit_code = strict_probe_main(
                [
                    "--embedding-dir",
                    str(embedding_dir),
                    "--label-path",
                    str(label_path),
                    "--output-dir",
                    str(output_dir),
                    "--tasks",
                    "dbh",
                    "--feature-sets",
                    "uni3d",
                    "--split",
                    "leave_one_site_out",
                    "--min-train-samples",
                    "10",
                    "--min-test-samples",
                    "3",
                ]
            )

            self.assertEqual(exit_code, 0)
            results = self._read_csv(output_dir / "probe_results.csv")
            ok = [row for row in results if row["status"] == "ok"]
            self.assertTrue(ok)
            for row in ok:
                train_sites = set(json.loads(row["site_train"]))
                test_sites = set(json.loads(row["site_test"]))
                self.assertFalse(train_sites & test_sites)

    def test_main_records_skip_when_sample_size_insufficient(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            embedding_dir, label_path = self._write_probe_fixture(root, count=12)
            output_dir = root / "probe_skip"

            exit_code = strict_probe_main(
                [
                    "--embedding-dir",
                    str(embedding_dir),
                    "--label-path",
                    str(label_path),
                    "--output-dir",
                    str(output_dir),
                    "--tasks",
                    "dbh",
                    "--feature-sets",
                    "uni3d",
                    "--split",
                    "random",
                    "--seeds",
                    "0",
                    "--min-train-samples",
                    "100",
                ]
            )

            self.assertEqual(exit_code, 0)
            rows = self._read_csv(output_dir / "probe_results.csv")
            self.assertEqual(rows[0]["status"], "skipped")
            self.assertIn("below --min-train-samples", rows[0]["failure_reason"])

    def _write_probe_fixture(self, root: Path, *, count: int) -> tuple[Path, Path]:
        embedding_dir = root / "embeddings"
        embedding_dir.mkdir()
        label_rows = []
        rng = np.random.default_rng(123)
        for index in range(count):
            site = "150" if index < count // 2 else "160"
            tree_id = f"{site}_{index:03d}"
            species = "银杏" if index % 2 == 0 else "国槐"
            vector = rng.normal(size=8).astype(np.float32)
            if species == "银杏":
                vector[0] += 2.0
            else:
                vector[1] += 2.0
            self._write_embedding(embedding_dir / f"{index + 1:03d}_{tree_id}.npz", tree_id, vector.tolist())
            label_rows.append(
                {
                    "对应的文件名": f"{tree_id}.las",
                    "树种": species,
                    "胸径cm": 10.0 + index,
                    "树高m": 5.0 + index * 0.5,
                    "东西冠幅m": 1.0 + index * 0.1,
                    "南北冠幅m": 1.2 + index * 0.1,
                }
            )
        label_path = root / "labels.csv"
        with label_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(label_rows[0].keys()))
            writer.writeheader()
            writer.writerows(label_rows)
        return embedding_dir, label_path

    def _write_labels(self, path: Path, tree_ids: list[str]) -> Path:
        rows = [
            {
                "对应的文件名": f"{tree_id}.las",
                "树种": "银杏",
                "胸径cm": 10.0,
                "树高m": 5.0,
                "东西冠幅m": 1.0,
                "南北冠幅m": 1.2,
            }
            for tree_id in tree_ids
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return path

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
            metadata=np.asarray(json.dumps({"rgb_source": "provided"})),
        )

    @staticmethod
    def _sample_row(tree_id: str, *, near: bool = False, outlier: bool = False):
        from scripts.probe_uni3d_embeddings_strict import SampleRow

        return SampleRow(
            tree_id=tree_id,
            site_id=tree_id.split("_", 1)[0],
            embedding_path=f"/fake/{tree_id}.npz",
            source_path=f"/fake/{tree_id}.npy",
            metadata={"rgb_source": "provided"},
            embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            labels={
                "树种": "银杏",
                "胸径cm": 10.0,
                "树高m": 5.0,
                "东西冠幅m": 1.0,
                "南北冠幅m": 1.2,
            },
            rgb_source="provided",
            is_near_duplicate_candidate=near,
            is_outlier_candidate=outlier,
        )

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]


if __name__ == "__main__":
    unittest.main()
