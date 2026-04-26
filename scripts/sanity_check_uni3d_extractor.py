"""Run a real Uni3D extractor sanity check on a GPU server.

This script intentionally uses only Uni3DFeatureExtractor, never the mock
extractor. It does not save embeddings; the first version only prints checks
needed to confirm that the real Uni3D forward path runs end to end.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from forestagent.adapters.uni3d_extractor import (  # noqa: E402
    Uni3DCheckpointError,
    Uni3DDependencyError,
    Uni3DExtractorConfig,
    Uni3DExtractorError,
    Uni3DFeatureExtractor,
    Uni3DInputError,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a real Uni3DFeatureExtractor forward sanity check.",
    )
    parser.add_argument("--xyz-path", required=True, help="Path to a .npy file shaped [N, 3].")
    parser.add_argument(
        "--rgb-path",
        default=None,
        help="Optional path to a .npy RGB file shaped [N, 3]. If omitted, adapter fallback RGB is used.",
    )
    parser.add_argument("--checkpoint-path", required=True, help="Path to the Uni3D checkpoint .pt file.")
    parser.add_argument("--uni3d-repo-path", required=True, help="Path to a local checkout of baaivision/Uni3D.")
    parser.add_argument("--device", default="cuda", help="Torch device for real forward, e.g. cuda or cuda:0.")
    parser.add_argument("--model-builder", default="create_uni3d")
    parser.add_argument("--pc-model", default="eva02_base_patch14_448")
    parser.add_argument("--pc-feat-dim", type=int, default=768)
    parser.add_argument("--embed-dim", type=int, default=1024)
    parser.add_argument("--pc-encoder-dim", type=int, default=512)
    parser.add_argument("--num-group", type=int, default=512)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--patch-dropout", type=float, default=0.0)
    parser.add_argument("--drop-path-rate", type=float, default=0.0)
    parser.add_argument("--pretrained-pc", default="")
    parser.add_argument("--rgb-scale", type=float, default=None)
    parser.add_argument("--rgb-fallback", type=float, default=0.4)
    parser.add_argument(
        "--no-normalize-xyz",
        action="store_true",
        help="Disable adapter xyz centering and unit-radius normalization.",
    )
    parser.add_argument(
        "--allow-nonstrict-checkpoint",
        action="store_true",
        help="Load checkpoint with strict=False. Use only for diagnosis.",
    )
    parser.add_argument("--consistency-atol", type=float, default=1e-5)
    parser.add_argument("--consistency-rtol", type=float, default=1e-5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        xyz = load_npy_matrix(args.xyz_path, name="xyz")
        rgb = load_npy_matrix(args.rgb_path, name="rgb") if args.rgb_path else None
        config = Uni3DExtractorConfig(
            checkpoint_path=args.checkpoint_path,
            uni3d_repo_path=args.uni3d_repo_path,
            model_builder=args.model_builder,
            pc_model=args.pc_model,
            pc_feat_dim=args.pc_feat_dim,
            embed_dim=args.embed_dim,
            pc_encoder_dim=args.pc_encoder_dim,
            num_group=args.num_group,
            group_size=args.group_size,
            patch_dropout=args.patch_dropout,
            drop_path_rate=args.drop_path_rate,
            pretrained_pc=args.pretrained_pc,
            device=args.device,
            rgb_fallback=args.rgb_fallback,
            rgb_scale=args.rgb_scale,
            normalize_xyz=not args.no_normalize_xyz,
            strict_checkpoint=not args.allow_nonstrict_checkpoint,
        )
        extractor = Uni3DFeatureExtractor(config)
        print_section("Input")
        print(f"xyz_path: {Path(args.xyz_path)}")
        print(f"xyz_shape: {xyz.shape}")
        print(f"rgb_path: {Path(args.rgb_path) if args.rgb_path else None}")
        print(f"rgb_shape: {rgb.shape if rgb is not None else None}")
        print(f"device: {args.device}")

        print_section("Forward Pass 1")
        first = extractor.extract(xyz, rgb)
        print_result_checks(first.embedding_raw, first.embedding_l2, first.metadata)

        print_section("Forward Pass 2")
        second = extractor.extract(xyz, rgb)
        print_result_checks(second.embedding_raw, second.embedding_l2, second.metadata)

        print_section("Repeatability")
        raw_close = np.allclose(
            first.embedding_raw,
            second.embedding_raw,
            atol=args.consistency_atol,
            rtol=args.consistency_rtol,
        )
        l2_close = np.allclose(
            first.embedding_l2,
            second.embedding_l2,
            atol=args.consistency_atol,
            rtol=args.consistency_rtol,
        )
        raw_max_abs_diff = float(np.max(np.abs(first.embedding_raw - second.embedding_raw)))
        l2_max_abs_diff = float(np.max(np.abs(first.embedding_l2 - second.embedding_l2)))
        print(f"raw_allclose: {raw_close}")
        print(f"raw_max_abs_diff: {raw_max_abs_diff:.8g}")
        print(f"l2_allclose: {l2_close}")
        print(f"l2_max_abs_diff: {l2_max_abs_diff:.8g}")
        if not raw_close or not l2_close:
            print(
                "ERROR: repeated extraction was not consistent. Check patch_dropout, eval mode, "
                "CUDA determinism, and input preprocessing.",
                file=sys.stderr,
            )
            return 2
        return 0
    except (Uni3DDependencyError, Uni3DCheckpointError, Uni3DInputError, Uni3DExtractorError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"ERROR: input file not found: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"ERROR: invalid input array: {exc}", file=sys.stderr)
        return 1


def load_npy_matrix(path: str | Path, *, name: str) -> np.ndarray:
    array_path = Path(path)
    if not array_path.is_file():
        raise FileNotFoundError(str(array_path))
    array = np.load(array_path)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must be a .npy array with shape [N, 3], got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf values.")
    return array.astype(np.float32, copy=False)


def print_result_checks(raw: np.ndarray, l2: np.ndarray, metadata: dict[str, Any]) -> None:
    raw_norm = np.linalg.norm(raw, axis=-1)
    l2_norm = np.linalg.norm(l2, axis=-1)
    raw_finite = bool(np.isfinite(raw).all())
    l2_finite = bool(np.isfinite(l2).all())
    print(f"embedding_raw_shape: {raw.shape}")
    print(f"embedding_l2_shape: {l2.shape}")
    print(f"embedding_raw_finite: {raw_finite}")
    print(f"embedding_l2_finite: {l2_finite}")
    print(f"embedding_raw_l2_norm: {format_array(raw_norm)}")
    print(f"embedding_l2_l2_norm: {format_array(l2_norm)}")
    print("metadata:")
    print(json.dumps(make_jsonable(metadata), indent=2, sort_keys=True))
    if not raw_finite or not l2_finite:
        raise Uni3DExtractorError("Embedding contains NaN or Inf values.")


def format_array(array: np.ndarray) -> str:
    return np.array2string(array.astype(np.float64), precision=8, separator=", ")


def make_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def print_section(title: str) -> None:
    print(f"\n=== {title} ===")


if __name__ == "__main__":
    raise SystemExit(main())
