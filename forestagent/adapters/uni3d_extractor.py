"""Offline adapter for using Uni3D as a frozen feature extractor.

This module intentionally does not vendor Uni3D or import its heavy dependencies at
module import time. The real extractor path imports torch and the official Uni3D
code lazily, so mock tests can run on machines without CUDA or pointnet2_ops.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping
import sys

import numpy as np


class Uni3DExtractorError(RuntimeError):
    """Base error for Uni3D extractor failures."""


class Uni3DDependencyError(Uni3DExtractorError):
    """Raised when the official Uni3D runtime dependencies are unavailable."""


class Uni3DCheckpointError(Uni3DExtractorError):
    """Raised when a checkpoint is missing or has an unsupported shape."""


class Uni3DInputError(Uni3DExtractorError):
    """Raised when point-cloud input cannot be normalized for Uni3D."""


@dataclass(frozen=True)
class Uni3DExtractorConfig:
    """Configuration needed to recreate the Uni3D model structure."""

    checkpoint_path: str | Path | None = None
    uni3d_repo_path: str | Path | None = None
    model_builder: str = "create_uni3d"
    pc_model: str = "eva02_base_patch14_448"
    pc_feat_dim: int = 768
    embed_dim: int = 1024
    pc_encoder_dim: int = 512
    num_group: int = 512
    group_size: int = 64
    patch_dropout: float = 0.0
    drop_path_rate: float = 0.0
    pretrained_pc: str = ""
    device: str = "cpu"
    rgb_fallback: float = 0.4
    rgb_scale: float | None = None
    normalize_xyz: bool = True
    strict_checkpoint: bool = True

    @property
    def min_point_count(self) -> int:
        """Minimum point count needed by FPS/KNN grouping."""

        return max(self.num_group, self.group_size)


@dataclass(frozen=True)
class PreprocessedPointCloud:
    """Point cloud after the fixed Uni3D adapter preprocessing step."""

    xyz: np.ndarray
    rgb: np.ndarray
    feature: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Uni3DExtractionResult:
    """Output returned by real and mock Uni3D extractor adapters."""

    embedding_raw: np.ndarray
    embedding_l2: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def preprocess_point_cloud(
    xyz: Any,
    rgb: Any | None = None,
    *,
    config: Uni3DExtractorConfig | None = None,
) -> PreprocessedPointCloud:
    """Validate and preprocess a single point cloud for Uni3D.

    The official Uni3D evaluation path feeds the model with batched
    ``xyz + rgb`` features. This function prepares exactly that adapter boundary:
    ``xyz`` and ``rgb`` are returned separately, while ``feature`` is shaped as
    ``[1, N, 6]`` for ``Uni3D.encode_pc``.
    """

    cfg = config or Uni3DExtractorConfig()
    xyz_array = _as_float32_matrix(xyz, name="xyz")
    if xyz_array.shape[0] < cfg.min_point_count:
        raise Uni3DInputError(
            "Uni3D input has too few points for the configured grouping: "
            f"got {xyz_array.shape[0]}, need at least {cfg.min_point_count} "
            f"(num_group={cfg.num_group}, group_size={cfg.group_size})."
        )

    if rgb is None:
        rgb_array = np.ones_like(xyz_array, dtype=np.float32) * np.float32(cfg.rgb_fallback)
        rgb_source = "fallback_constant"
    else:
        rgb_array = _as_float32_matrix(rgb, name="rgb")
        if rgb_array.shape[0] != xyz_array.shape[0]:
            raise Uni3DInputError(
                f"rgb must have the same point count as xyz: got {rgb_array.shape[0]} "
                f"and {xyz_array.shape[0]}."
            )
        if cfg.rgb_scale is not None:
            if cfg.rgb_scale <= 0:
                raise Uni3DInputError("rgb_scale must be positive when provided.")
            rgb_array = rgb_array / np.float32(cfg.rgb_scale)
        rgb_source = "provided"

    centroid: list[float] | None = None
    scale_radius: float | None = None
    normalized_xyz = xyz_array
    if cfg.normalize_xyz:
        centroid_array = normalized_xyz.mean(axis=0)
        centered = normalized_xyz - centroid_array
        radius = float(np.max(np.linalg.norm(centered, axis=1)))
        if radius < 1e-6:
            raise Uni3DInputError("Cannot normalize a degenerate point cloud with near-zero radius.")
        normalized_xyz = centered / np.float32(radius)
        centroid = centroid_array.astype(np.float64).tolist()
        scale_radius = radius

    feature = np.concatenate([normalized_xyz, rgb_array], axis=1)[None, :, :].astype(np.float32)
    return PreprocessedPointCloud(
        xyz=normalized_xyz.astype(np.float32),
        rgb=rgb_array.astype(np.float32),
        feature=feature,
        metadata={
            "point_count": int(xyz_array.shape[0]),
            "feature_shape": tuple(int(v) for v in feature.shape),
            "rgb_source": rgb_source,
            "rgb_fallback": cfg.rgb_fallback if rgb is None else None,
            "rgb_scale": cfg.rgb_scale,
            "normalize_xyz": cfg.normalize_xyz,
            "centroid": centroid,
            "scale_radius": scale_radius,
        },
    )


def select_checkpoint_state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    """Select a model state_dict from known Uni3D/PyTorch checkpoint layouts."""

    if "module" in checkpoint and isinstance(checkpoint["module"], Mapping):
        state_dict = checkpoint["module"]
    elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], Mapping):
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    return strip_module_prefix(state_dict)


def strip_module_prefix(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Remove a DistributedDataParallel ``module.`` prefix when present."""

    if not state_dict:
        raise Uni3DCheckpointError("Checkpoint state_dict is empty.")
    first_key = next(iter(state_dict.keys()))
    if isinstance(first_key, str) and first_key.startswith("module."):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return dict(state_dict)


class Uni3DFeatureExtractor:
    """Real Uni3D feature extractor adapter.

    This class never fabricates embeddings. If official Uni3D dependencies,
    checkpoint files, or runtime execution are unavailable, it raises a clear
    error instead of falling back to mock output.
    """

    def __init__(self, config: Uni3DExtractorConfig) -> None:
        self.config = config
        self._model: Any | None = None

    def extract(self, xyz: Any, rgb: Any | None = None) -> Uni3DExtractionResult:
        """Run the real Uni3D ``encode_pc`` path and return raw/L2 embeddings."""

        preprocessed = preprocess_point_cloud(xyz, rgb, config=self.config)
        model = self._ensure_model_loaded()
        torch = self._import_torch()
        feature = torch.as_tensor(
            preprocessed.feature,
            dtype=torch.float32,
            device=self.config.device,
        )
        try:
            with torch.no_grad():
                embedding = model.encode_pc(feature)
        except Exception as exc:  # pragma: no cover - requires official Uni3D runtime
            raise Uni3DExtractorError(
                "Real Uni3D forward failed. This adapter does not return mock embeddings "
                "from the real extractor path."
            ) from exc

        raw = embedding.detach().cpu().numpy().astype(np.float32)
        l2 = _l2_normalize(raw)
        return Uni3DExtractionResult(
            embedding_raw=raw,
            embedding_l2=l2,
            metadata={
                **preprocessed.metadata,
                **self._model_metadata(),
                "mock": False,
                "embedding_shape": tuple(int(v) for v in raw.shape),
                "embedding_l2_normalized": True,
            },
        )

    def load_model(self) -> Any:
        """Build the official Uni3D model, load checkpoint weights, and enter eval mode."""

        if self.config.checkpoint_path is None:
            raise Uni3DCheckpointError("checkpoint_path is required for the real Uni3D extractor.")
        checkpoint_path = Path(self.config.checkpoint_path)
        if not checkpoint_path.is_file():
            raise Uni3DCheckpointError(f"Uni3D checkpoint not found: {checkpoint_path}")

        torch = self._import_torch()
        with _optional_sys_path(self.config.uni3d_repo_path):
            try:
                import models.uni3d as uni3d_models  # type: ignore[import-not-found]
            except ModuleNotFoundError as exc:
                raise Uni3DDependencyError(
                    "Could not import official Uni3D modules. Set uni3d_repo_path to a "
                    "local checkout of baaivision/Uni3D and install its runtime dependencies."
                ) from exc

        args = self._build_official_args()
        try:
            builder = getattr(uni3d_models, self.config.model_builder)
        except AttributeError as exc:
            raise Uni3DDependencyError(
                f"Official Uni3D module has no builder named {self.config.model_builder!r}."
            ) from exc

        try:
            model = builder(args=args)
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if not isinstance(checkpoint, Mapping):
                raise Uni3DCheckpointError("Checkpoint must load to a mapping-like object.")
            state_dict = select_checkpoint_state_dict(checkpoint)
            model.load_state_dict(state_dict, strict=self.config.strict_checkpoint)
            model.to(self.config.device)
            model.eval()
        except Uni3DExtractorError:
            raise
        except Exception as exc:  # pragma: no cover - requires official Uni3D runtime
            raise Uni3DExtractorError(
                "Failed to build or load the real Uni3D model. Check that config values "
                "match the checkpoint scale and official dependencies are installed."
            ) from exc
        self._model = model
        return model

    def _ensure_model_loaded(self) -> Any:
        if self._model is None:
            return self.load_model()
        return self._model

    def _build_official_args(self) -> SimpleNamespace:
        return SimpleNamespace(
            pc_model=self.config.pc_model,
            pretrained_pc=self.config.pretrained_pc,
            drop_path_rate=self.config.drop_path_rate,
            pc_feat_dim=self.config.pc_feat_dim,
            embed_dim=self.config.embed_dim,
            group_size=self.config.group_size,
            num_group=self.config.num_group,
            pc_encoder_dim=self.config.pc_encoder_dim,
            patch_dropout=self.config.patch_dropout,
        )

    def _model_metadata(self) -> dict[str, Any]:
        return {
            "model_builder": self.config.model_builder,
            "pc_model": self.config.pc_model,
            "pc_feat_dim": self.config.pc_feat_dim,
            "embed_dim": self.config.embed_dim,
            "pc_encoder_dim": self.config.pc_encoder_dim,
            "num_group": self.config.num_group,
            "group_size": self.config.group_size,
            "patch_dropout": self.config.patch_dropout,
            "checkpoint_path": str(self.config.checkpoint_path)
            if self.config.checkpoint_path is not None
            else None,
        }

    @staticmethod
    def _import_torch() -> Any:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise Uni3DDependencyError(
                "PyTorch is required for the real Uni3D extractor path. "
                "Install the official Uni3D runtime dependencies before running real inference."
            ) from exc
        return torch


class MockUni3DExtractor:
    """Deterministic mock adapter for tests that must not use real Uni3D output."""

    def __init__(self, config: Uni3DExtractorConfig | None = None) -> None:
        self.config = config or Uni3DExtractorConfig(num_group=4, group_size=2, embed_dim=16)

    def extract(self, xyz: Any, rgb: Any | None = None) -> Uni3DExtractionResult:
        preprocessed = preprocess_point_cloud(xyz, rgb, config=self.config)
        stats = np.concatenate(
            [
                preprocessed.xyz.mean(axis=0),
                preprocessed.xyz.std(axis=0),
                preprocessed.rgb.mean(axis=0),
                preprocessed.rgb.std(axis=0),
            ]
        ).astype(np.float32)
        repeats = int(np.ceil(self.config.embed_dim / stats.shape[0]))
        raw = np.tile(stats, repeats)[: self.config.embed_dim][None, :].astype(np.float32)
        l2 = _l2_normalize(raw)
        return Uni3DExtractionResult(
            embedding_raw=raw,
            embedding_l2=l2,
            metadata={
                **preprocessed.metadata,
                "mock": True,
                "mock_reason": "adapter_interface_test_only",
                "embedding_shape": tuple(int(v) for v in raw.shape),
                "embed_dim": self.config.embed_dim,
            },
        )


def _as_float32_matrix(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3:
        raise Uni3DInputError(f"{name} must have shape [N, 3], got {array.shape}.")
    if array.shape[0] == 0:
        raise Uni3DInputError(f"{name} must contain at least one point.")
    if not np.isfinite(array).all():
        raise Uni3DInputError(f"{name} contains NaN or infinite values.")
    return array


def _l2_normalize(embedding: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(embedding, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise Uni3DExtractorError("Cannot L2-normalize a zero Uni3D embedding.")
    return (embedding / norm).astype(np.float32)


@contextmanager
def _optional_sys_path(path: str | Path | None) -> Iterator[None]:
    if path is None:
        yield
        return
    repo_path = str(Path(path))
    sys.path.insert(0, repo_path)
    try:
        yield
    finally:
        try:
            sys.path.remove(repo_path)
        except ValueError:
            pass
