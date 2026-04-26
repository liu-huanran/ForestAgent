"""Tests for the LAS-to-Uni3D-NPY conversion helpers.

These tests intentionally avoid real Uni3D forward, checkpoints, CUDA, and
pointnet2_ops. They also do not require laspy; the real LAS reader is validated
on the server when laspy and source .las files are available.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.convert_las_to_uni3d_npy import (
    build_uni3d_input_array,
    convert_las_file,
    discover_las_paths,
    normalize_rgb,
    normalize_limit,
    sample_fixed_count,
    write_manifest_jsonl,
)


class ConvertLasToUni3DNpyTests(unittest.TestCase):
    def test_sample_fixed_count_is_deterministic(self) -> None:
        array = np.arange(30, dtype=np.float32).reshape(10, 3)

        first = sample_fixed_count(array, num_points=5, seed=42)
        second = sample_fixed_count(array, num_points=5, seed=42)

        self.assertEqual(first.values.shape, (5, 3))
        self.assertTrue(np.array_equal(first.values, second.values))
        self.assertEqual(first.values.dtype, np.float32)
        self.assertEqual(first.method, "deterministic_random_without_replacement")

    def test_sample_fixed_count_rejects_too_few_points(self) -> None:
        array = np.zeros((4, 3), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "too_few_points"):
            sample_fixed_count(array, num_points=5, seed=42)

    def test_limit_none_or_zero_means_all_candidates_and_recursive_finds_subdirs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nested = root / "nested"
            nested.mkdir()
            (root / "a.las").write_bytes(b"fake")
            (nested / "b.las").write_bytes(b"fake")

            all_paths = discover_las_paths(
                input_dir=root,
                pattern="*.las",
                recursive=True,
                limit=normalize_limit(None),
            )
            zero_paths = discover_las_paths(
                input_dir=root,
                pattern="*.las",
                recursive=True,
                limit=normalize_limit(0),
            )
            top_level_only = discover_las_paths(
                input_dir=root,
                pattern="*.las",
                recursive=False,
                limit=normalize_limit(None),
            )

            self.assertEqual(len(all_paths), 2)
            self.assertEqual(len(zero_paths), 2)
            self.assertEqual(len(top_level_only), 1)

    def test_convert_las_file_skip_existing_does_not_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "las"
            output_dir = root / "npy"
            input_dir.mkdir()
            output_dir.mkdir()
            source_las = input_dir / "tree_001.las"
            output_npy = output_dir / "tree_001.npy"
            source_las.write_bytes(b"not read during skip")
            existing = np.ones((5, 3), dtype=np.float32)
            np.save(output_npy, existing)

            entry = convert_las_file(
                source_las=source_las,
                input_dir=input_dir,
                output_dir=output_dir,
                num_points=5,
                seed=42,
                include_rgb=False,
                overwrite=False,
                skip_existing=True,
            )

            self.assertEqual(entry["status"], "skipped_existing")
            self.assertEqual(entry["output_shape"], [5, 3])
            self.assertTrue(np.array_equal(np.load(output_npy), existing))

    def test_normalize_rgb_uint16_to_unit_range(self) -> None:
        rgb = np.asarray([[0, 32768, 65535]], dtype=np.uint16)

        normalized = normalize_rgb(rgb)

        self.assertEqual(normalized.dtype, np.float32)
        self.assertTrue(np.all(normalized >= 0.0))
        self.assertTrue(np.all(normalized <= 1.0))
        self.assertTrue(np.allclose(normalized[0, [0, 2]], [0.0, 1.0]))

    def test_build_uni3d_input_array_outputs_xyz_or_xyzrgb(self) -> None:
        xyz = np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
        rgb = np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)

        xyz_only = build_uni3d_input_array(xyz, None)
        xyzrgb = build_uni3d_input_array(xyz, rgb)

        self.assertEqual(xyz_only.shape, (2, 3))
        self.assertEqual(xyzrgb.shape, (2, 6))
        self.assertTrue(np.allclose(xyzrgb[:, :3], xyz))
        self.assertTrue(np.allclose(xyzrgb[:, 3:], rgb))

    def test_build_uni3d_input_array_rejects_rgb_count_mismatch(self) -> None:
        xyz = np.zeros((2, 3), dtype=np.float32)
        rgb = np.zeros((1, 3), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "rgb point count"):
            build_uni3d_input_array(xyz, rgb)

    def test_write_manifest_jsonl_records_success_and_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifest.jsonl"
            entries = [
                {
                    "status": "ok",
                    "source_las": "/source/a.las",
                    "output_npy": "/out/a.npy",
                    "shape": [10000, 3],
                    "has_rgb": False,
                    "num_points_before": 12345,
                    "num_points_after": 10000,
                    "seed": 42,
                    "failure_reason": None,
                },
                {
                    "status": "failed",
                    "source_las": "/source/b.las",
                    "output_npy": "/out/b.npy",
                    "shape": None,
                    "has_rgb": None,
                    "num_points_before": 100,
                    "num_points_after": None,
                    "seed": 42,
                    "failure_reason": "too_few_points",
                },
            ]

            write_manifest_jsonl(manifest_path, entries)

            lines = manifest_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            parsed = [json.loads(line) for line in lines]
            self.assertEqual(parsed[0]["status"], "ok")
            self.assertEqual(parsed[1]["status"], "failed")
            self.assertEqual(parsed[1]["failure_reason"], "too_few_points")


if __name__ == "__main__":
    unittest.main()
