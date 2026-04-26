"""Batch extract real Uni3D embeddings for small offline sanity checks.

This script intentionally uses only Uni3DFeatureExtractor. It never uses the
mock extractor, never registers a main-system tool, and keeps execution
single-process so GPU failures are easy to diagnose on the server.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import re
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


@dataclass(frozen=True)
class LoadedPointCloud:
    """A point cloud loaded from a single .npy file before adapter preprocessing."""

    source_path: Path
    tree_id: str
    input_shape: tuple[int, ...]
    xyz: np.ndarray
    rgb: np.ndarray | None
    yz_swap: bool


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch extract real Uni3D embeddings from .npy single-tree point clouds.",
    )
    parser.add_argument("--input-dir", default=None, help="Directory to scan when no manifest is provided.")
    parser.add_argument("--manifest-path", default=None, help="Optional manifest listing point-cloud paths.")
    parser.add_argument("--output-dir", required=True, help="Directory for per-tree .npz files and summary.json.")
    parser.add_argument("--checkpoint-path", required=True, help="Path to the official Uni3D checkpoint.")
    parser.add_argument("--uni3d-repo-path", required=True, help="Path to a local checkout of baaivision/Uni3D.")
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda, cuda:0, or cpu for diagnosis.")
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
    parser.add_argument(
        "--yz-swap",
        action="store_true",
        help="Swap y/z columns after loading xyz. Use only when the dataset convention requires it.",
    )
    parser.add_argument("--recursive", dest="recursive", action="store_true", default=True)
    parser.add_argument("--no-recursive", dest="recursive", action="store_false")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of files to process. 0 means all.")
    parser.add_argument("--pattern", default="*.npy", help="Glob pattern used with --input-dir.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing per-tree output files.")
    parser.add_argument(
        "--skip-existing",
        "--resume",
        dest="skip_existing",
        action="store_true",
        help="Skip existing .npz outputs and record them as skipped_existing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        input_paths = discover_input_paths(
            input_dir=Path(args.input_dir) if args.input_dir else None,
            manifest_path=Path(args.manifest_path) if args.manifest_path else None,
            pattern=args.pattern,
            recursive=args.recursive,
            limit=normalize_limit(args.limit),
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = build_extractor_config(args)
    extractor = Uni3DFeatureExtractor(config)
    results: list[dict[str, Any]] = []

    total = len(input_paths)
    for index, source_path in enumerate(input_paths, start=1):
        tree_id = infer_tree_id(source_path)
        output_path = output_dir / f"{index:03d}_{safe_filename(tree_id)}.npz"
        entry_base = {
            "tree_id": tree_id,
            "source_path": str(source_path),
            "output_path": str(output_path),
        }
        if output_path.exists() and not args.overwrite:
            if args.skip_existing:
                try:
                    existing = load_existing_embedding_metadata(output_path)
                    results.append(
                        {
                            **entry_base,
                            **existing,
                            "status": "skipped_existing",
                            "failure_reason": None,
                        }
                    )
                    print(f"[{index}/{total}] {tree_id} skipped_existing")
                except ValueError as exc:
                    results.append(
                        {
                            **entry_base,
                            "status": "failed",
                            "failure_reason": f"ValueError: {exc}",
                        }
                    )
                    print(f"[{index}/{total}] {tree_id} failed: {exc}", file=sys.stderr)
                continue
            results.append(
                {
                    **entry_base,
                    "status": "failed",
                    "failure_reason": "output_exists_use_overwrite_or_skip_existing",
                }
            )
            print(f"[{index}/{total}] {tree_id} failed: output exists", file=sys.stderr)
            continue

        try:
            point_cloud = load_npy_point_cloud(source_path, yz_swap=args.yz_swap)
            extraction = extractor.extract(point_cloud.xyz, point_cloud.rgb)
            metadata = {
                **extraction.metadata,
                "tree_id": tree_id,
                "source_path": str(source_path),
                "input_shape": point_cloud.input_shape,
                "yz_swap": point_cloud.yz_swap,
            }
            save_embedding_npz(
                output_path=output_path,
                tree_id=tree_id,
                source_path=source_path,
                embedding_raw=extraction.embedding_raw,
                embedding_l2=extraction.embedding_l2,
                metadata=metadata,
            )
            raw_norm = np.linalg.norm(extraction.embedding_raw, axis=-1)
            l2_norm = np.linalg.norm(extraction.embedding_l2, axis=-1)
            results.append(
                {
                    **entry_base,
                    "status": "ok",
                    "input_shape": point_cloud.input_shape,
                    "xyz_shape": tuple(int(v) for v in point_cloud.xyz.shape),
                    "rgb_shape": tuple(int(v) for v in point_cloud.rgb.shape)
                    if point_cloud.rgb is not None
                    else None,
                    "embedding_raw_shape": tuple(int(v) for v in extraction.embedding_raw.shape),
                    "embedding_l2_shape": tuple(int(v) for v in extraction.embedding_l2.shape),
                    "embedding_raw_finite": bool(np.isfinite(extraction.embedding_raw).all()),
                    "embedding_l2_finite": bool(np.isfinite(extraction.embedding_l2).all()),
                    "embedding_raw_l2_norm": raw_norm.astype(float).tolist(),
                    "embedding_l2_l2_norm": l2_norm.astype(float).tolist(),
                    "metadata": make_jsonable(metadata),
                }
            )
            print(f"[{index}/{total}] {tree_id} ok")
        except (
            ValueError,
            FileNotFoundError,
            Uni3DDependencyError,
            Uni3DCheckpointError,
            Uni3DInputError,
            Uni3DExtractorError,
        ) as exc:
            results.append(
                {
                    **entry_base,
                    "status": "failed",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[{index}/{total}] {tree_id} failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    summary = build_summary(args=args, config=config, input_paths=input_paths, results=results)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print_batch_summary(summary_path=summary_path, summary=summary)
    return 0 if summary["candidate_count"] > 0 and summary["failure_count"] == 0 else 2


def build_extractor_config(args: argparse.Namespace) -> Uni3DExtractorConfig:
    return Uni3DExtractorConfig(
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


def discover_input_paths(
    *,
    input_dir: Path | None,
    manifest_path: Path | None,
    pattern: str,
    recursive: bool,
    limit: int | None,
) -> list[Path]:
    if manifest_path is not None:
        paths = read_manifest_paths(manifest_path=manifest_path, input_dir=input_dir)
    else:
        if input_dir is None:
            raise ValueError("--input-dir is required when --manifest-path is not provided.")
        if not input_dir.is_dir():
            raise ValueError(f"input directory not found: {input_dir}")
        globber = input_dir.rglob if recursive else input_dir.glob
        paths = sorted(path for path in globber(pattern) if path.is_file())
    return paths[:limit] if limit is not None else paths


def normalize_limit(limit: int | None) -> int | None:
    if limit is None or limit == 0:
        return None
    if limit < 0:
        raise ValueError("--limit must be non-negative. Use 0 or omit it for all candidates.")
    return limit


def read_manifest_paths(*, manifest_path: Path, input_dir: Path | None) -> list[Path]:
    if not manifest_path.is_file():
        raise ValueError(f"manifest file not found: {manifest_path}")
    suffix = manifest_path.suffix.lower()
    raw_paths: list[str] = []
    if suffix == ".json":
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("JSON manifest must be a list of paths or objects with a path field.")
        raw_paths = [_extract_manifest_path(item) for item in payload]
    elif suffix == ".jsonl":
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            raw_paths.append(_extract_manifest_path(json.loads(stripped)))
    elif suffix == ".csv":
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                raw_paths.append(
                    row.get("path") or row.get("source_path") or row.get("output_npy") or next(iter(row.values()))
                )
    else:
        raw_paths = [
            line.strip()
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    return [resolve_manifest_entry(path, manifest_path=manifest_path, input_dir=input_dir) for path in raw_paths]


def _extract_manifest_path(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        value = item.get("path") or item.get("source_path") or item.get("output_npy")
        if isinstance(value, str):
            return value
    raise ValueError("Manifest entries must be strings or objects with path/source_path.")


def resolve_manifest_entry(path_text: str, *, manifest_path: Path, input_dir: Path | None) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    bases = [base for base in (input_dir, manifest_path.parent) if base is not None]
    for base in bases:
        candidate = base / path
        if candidate.exists():
            return candidate
    return (input_dir or manifest_path.parent) / path


def load_npy_point_cloud(path: str | Path, *, yz_swap: bool = False) -> LoadedPointCloud:
    source_path = Path(path)
    if source_path.suffix.lower() != ".npy":
        raise ValueError(f"Only .npy point-cloud files are supported in this first batch script: {source_path}")
    if not source_path.is_file():
        raise FileNotFoundError(str(source_path))
    array = np.load(source_path)
    if array.ndim != 2 or array.shape[1] not in (3, 6):
        raise ValueError(f"{source_path} must have shape [N, 3] or [N, 6], got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{source_path} contains NaN or Inf values.")
    xyz = array[:, :3].astype(np.float32, copy=True)
    rgb = array[:, 3:6].astype(np.float32, copy=True) if array.shape[1] == 6 else None
    if yz_swap:
        xyz = xyz[:, [0, 2, 1]]
    return LoadedPointCloud(
        source_path=source_path,
        tree_id=infer_tree_id(source_path),
        input_shape=tuple(int(v) for v in array.shape),
        xyz=xyz,
        rgb=rgb,
        yz_swap=yz_swap,
    )


def save_embedding_npz(
    *,
    output_path: Path,
    tree_id: str,
    source_path: Path,
    embedding_raw: np.ndarray,
    embedding_l2: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embedding_raw=embedding_raw.astype(np.float32, copy=False),
        embedding_l2=embedding_l2.astype(np.float32, copy=False),
        tree_id=np.asarray(tree_id),
        source_path=np.asarray(str(source_path)),
        metadata=np.asarray(json.dumps(make_jsonable(metadata), sort_keys=True)),
    )


def load_existing_embedding_metadata(output_path: Path) -> dict[str, Any]:
    try:
        with np.load(output_path, allow_pickle=False) as payload:
            if "embedding_raw" not in payload or "embedding_l2" not in payload:
                raise ValueError("existing output missing embedding_raw or embedding_l2")
            raw = np.asarray(payload["embedding_raw"], dtype=np.float32)
            l2 = np.asarray(payload["embedding_l2"], dtype=np.float32)
    except Exception as exc:
        raise ValueError(f"could not read existing embedding output: {exc}") from exc
    if raw.ndim == 0 or l2.ndim == 0:
        raise ValueError("existing embedding arrays must not be scalar")
    return {
        "embedding_raw_shape": tuple(int(v) for v in raw.shape),
        "embedding_l2_shape": tuple(int(v) for v in l2.shape),
        "embedding_raw_finite": bool(np.isfinite(raw).all()),
        "embedding_l2_finite": bool(np.isfinite(l2).all()),
        "embedding_raw_l2_norm": np.linalg.norm(raw, axis=-1).astype(float).tolist(),
        "embedding_l2_l2_norm": np.linalg.norm(l2, axis=-1).astype(float).tolist(),
        "metadata": {"skipped_existing": True},
    }


def build_summary(
    *,
    args: argparse.Namespace,
    config: Uni3DExtractorConfig,
    input_paths: list[Path],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    success_count = sum(1 for result in results if result.get("status") == "ok")
    skipped_count = sum(1 for result in results if result.get("status") == "skipped_existing")
    failure_count = sum(1 for result in results if result.get("status") == "failed")
    return {
        "script": "scripts/batch_extract_uni3d_embeddings.py",
        "mock": False,
        "input_dir": args.input_dir,
        "manifest_path": args.manifest_path,
        "pattern": args.pattern,
        "recursive": args.recursive,
        "limit": args.limit,
        "candidate_count": len(input_paths),
        "success_count": success_count,
        "skipped_count": skipped_count,
        "failure_count": failure_count,
        "config": make_jsonable(config.__dict__),
        "results": make_jsonable(results),
    }


def print_batch_summary(*, summary_path: Path, summary: dict[str, Any]) -> None:
    print(f"summary_path: {summary_path}")
    print(f"candidate_count: {summary['candidate_count']}")
    print(f"success_count: {summary['success_count']}")
    print(f"skipped_count: {summary['skipped_count']}")
    print(f"failure_count: {summary['failure_count']}")
    for result in summary["results"]:
        status = result.get("status")
        tree_id = result.get("tree_id")
        if status == "ok":
            print(
                f"OK {tree_id}: embedding_l2_shape={result.get('embedding_l2_shape')} "
                f"l2_norm={result.get('embedding_l2_l2_norm')}"
            )
        elif status == "skipped_existing":
            print(
                f"SKIP {tree_id}: embedding_l2_shape={result.get('embedding_l2_shape')} "
                f"l2_norm={result.get('embedding_l2_l2_norm')}"
            )
        else:
            print(f"FAILED {tree_id}: {result.get('failure_reason')}")


def infer_tree_id(path: str | Path) -> str:
    return Path(path).stem


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "tree"


def make_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
