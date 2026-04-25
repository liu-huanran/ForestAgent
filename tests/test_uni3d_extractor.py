"""Tests for the standalone Uni3D extractor adapter."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from forestagent.adapters.uni3d_extractor import (
    MockUni3DExtractor,
    Uni3DCheckpointError,
    Uni3DExtractorConfig,
    Uni3DFeatureExtractor,
    Uni3DInputError,
    preprocess_point_cloud,
    select_checkpoint_state_dict,
)


class Uni3DExtractorAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Uni3DExtractorConfig(num_group=4, group_size=2, embed_dim=16)
        self.xyz = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )

    def test_preprocess_builds_batched_xyz_rgb_feature_with_rgb_fallback(self) -> None:
        preprocessed = preprocess_point_cloud(self.xyz, config=self.config)

        self.assertEqual(preprocessed.feature.shape, (1, 5, 6))
        self.assertEqual(preprocessed.metadata["rgb_source"], "fallback_constant")
        self.assertTrue(np.allclose(preprocessed.rgb, np.ones((5, 3), dtype=np.float32) * 0.4))
        self.assertTrue(np.allclose(preprocessed.xyz.mean(axis=0), np.zeros(3), atol=1e-6))
        self.assertTrue(
            np.isclose(float(np.max(np.linalg.norm(preprocessed.xyz, axis=1))), 1.0, atol=1e-6)
        )

    def test_preprocess_can_scale_uint8_like_rgb(self) -> None:
        rgb = np.asarray([[255.0, 128.0, 0.0]] * 5, dtype=np.float32)
        config = Uni3DExtractorConfig(num_group=4, group_size=2, rgb_scale=255.0)

        preprocessed = preprocess_point_cloud(self.xyz, rgb, config=config)

        self.assertEqual(preprocessed.metadata["rgb_source"], "provided")
        self.assertTrue(np.allclose(preprocessed.rgb[0], [1.0, 128.0 / 255.0, 0.0]))

    def test_preprocess_rejects_too_few_points_for_grouping(self) -> None:
        with self.assertRaises(Uni3DInputError):
            preprocess_point_cloud(self.xyz[:3], config=self.config)

    def test_preprocess_rejects_degenerate_point_cloud(self) -> None:
        xyz = np.ones((5, 3), dtype=np.float32)

        with self.assertRaises(Uni3DInputError):
            preprocess_point_cloud(xyz, config=self.config)

    def test_select_checkpoint_state_dict_prefers_module_and_strips_ddp_prefix(self) -> None:
        selected = select_checkpoint_state_dict(
            {
                "module": {
                    "module.point_encoder.weight": "weight",
                    "module.point_encoder.bias": "bias",
                },
                "state_dict": {"ignored": "value"},
            }
        )

        self.assertEqual(
            selected,
            {
                "point_encoder.weight": "weight",
                "point_encoder.bias": "bias",
            },
        )

    def test_select_checkpoint_state_dict_supports_state_dict_and_bare_layouts(self) -> None:
        from_state_dict = select_checkpoint_state_dict({"state_dict": {"a": 1}})
        bare = select_checkpoint_state_dict({"b": 2})

        self.assertEqual(from_state_dict, {"a": 1})
        self.assertEqual(bare, {"b": 2})

    def test_mock_extractor_returns_deterministic_test_only_embedding(self) -> None:
        extractor = MockUni3DExtractor(self.config)

        first = extractor.extract(self.xyz)
        second = extractor.extract(self.xyz)

        self.assertEqual(first.embedding_raw.shape, (1, 16))
        self.assertEqual(first.embedding_l2.shape, (1, 16))
        self.assertTrue(first.metadata["mock"])
        self.assertEqual(first.metadata["mock_reason"], "adapter_interface_test_only")
        self.assertTrue(np.allclose(first.embedding_raw, second.embedding_raw))
        self.assertTrue(np.isclose(float(np.linalg.norm(first.embedding_l2, axis=1)[0]), 1.0, atol=1e-6))

    def test_real_extractor_requires_checkpoint_and_does_not_return_mock_embedding(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_checkpoint = Path(temp_dir) / "missing_model.pt"
            extractor = Uni3DFeatureExtractor(
                Uni3DExtractorConfig(
                    checkpoint_path=missing_checkpoint,
                    num_group=4,
                    group_size=2,
                    embed_dim=16,
                )
            )

            with self.assertRaises(Uni3DCheckpointError):
                extractor.extract(self.xyz)


if __name__ == "__main__":
    unittest.main()
