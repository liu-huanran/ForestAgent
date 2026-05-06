"""Read-only audits for offline Uni3D experiment artifacts."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from forestagent.experiments.uni3d_experiment_config import experiment_name


STATUS_PASS = "PASS"
STATUS_WARN = "WARN"
STATUS_FAIL = "FAIL"
STATUS_NOT_AVAILABLE = "NOT_AVAILABLE"


def audit_experiment(config: dict[str, Any], *, stage: str = "all") -> dict[str, Any]:
    """Audit all available artifacts for an experiment without mutating them."""

    if stage not in {"all", "conversion", "extraction", "similarity", "probe"}:
        raise ValueError(f"Unsupported audit stage: {stage}")
    report: dict[str, Any] = {
        "experiment_name": experiment_name(config),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "checks": {},
    }
    if stage in {"all", "conversion"}:
        report["checks"]["conversion"] = audit_conversion(config)
    if stage in {"all", "extraction"}:
        report["checks"]["extraction"] = audit_extraction(config)
    if stage in {"all", "similarity"}:
        report["checks"]["similarity"] = audit_similarity(config)
    if stage in {"all", "probe"}:
        report["checks"]["probe"] = audit_probe(config)
    report["overall_status"] = combine_statuses(
        check.get("status", STATUS_NOT_AVAILABLE) for check in report["checks"].values()
    )
    return report


def audit_conversion(config: dict[str, Any]) -> dict[str, Any]:
    input_dir = Path(config["paths"]["npy_input_dir"])
    check = _base_check("conversion", input_dir)
    if not input_dir.exists():
        check.update(status=STATUS_NOT_AVAILABLE, reason="npy_input_dir does not exist")
        return check

    npy_paths = sorted(input_dir.rglob("*.npy"))
    shape_counts: Counter[str] = Counter()
    dtype_counts: Counter[str] = Counter()
    point_count_mismatch = 0
    nonfinite_count = 0
    expected_points = int(config.get("conversion", {}).get("num_points", 10000))
    sample_errors: list[dict[str, str]] = []

    for path in npy_paths:
        try:
            array = np.load(path, mmap_mode="r")
            shape_counts[_shape_key(array.shape)] += 1
            dtype_counts[str(array.dtype)] += 1
            if len(array.shape) != 2 or array.shape[0] != expected_points:
                point_count_mismatch += 1
            if not np.isfinite(np.asarray(array)).all():
                nonfinite_count += 1
        except Exception as exc:  # pragma: no cover - defensive audit path
            sample_errors.append({"path": str(path), "error": str(exc)})

    summary = _read_json(input_dir / "summary.json")
    manifest_path = input_dir / "manifest.jsonl"
    status = STATUS_PASS
    reasons: list[str] = []
    if not npy_paths:
        status = STATUS_NOT_AVAILABLE
        reasons.append("no .npy files found")
    if point_count_mismatch:
        status = STATUS_WARN
        reasons.append(f"{point_count_mismatch} .npy files do not match expected point count {expected_points}")
    if nonfinite_count or sample_errors:
        status = STATUS_FAIL
        reasons.append("non-finite or unreadable .npy inputs detected")
    if summary is None:
        status = _downgrade(status, STATUS_WARN)
        reasons.append("summary.json missing")
    if not manifest_path.exists():
        status = _downgrade(status, STATUS_WARN)
        reasons.append("manifest.jsonl missing")

    check.update(
        status=status,
        reason="; ".join(reasons) if reasons else "conversion artifacts look readable",
        npy_count=len(npy_paths),
        shape_distribution=dict(shape_counts),
        dtype_distribution=dict(dtype_counts),
        expected_point_count=expected_points,
        point_count_mismatch=point_count_mismatch,
        nonfinite_count=nonfinite_count,
        manifest_exists=manifest_path.exists(),
        summary_exists=summary is not None,
        summary=summary,
        sample_errors=sample_errors[:20],
    )
    return check


def audit_extraction(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["paths"]["embedding_output_dir"])
    check = _base_check("extraction", output_dir)
    if not output_dir.exists():
        check.update(status=STATUS_NOT_AVAILABLE, reason="embedding_output_dir does not exist")
        return check

    npz_paths = sorted(output_dir.rglob("*.npz"))
    summary = _read_json(output_dir / "summary.json")
    shape_counts: Counter[str] = Counter()
    l2_min: float | None = None
    l2_max: float | None = None
    nonfinite_count = 0
    metadata_missing_count = 0
    mock_values: Counter[str] = Counter()
    metadata_mismatches: list[str] = []
    extractor = config.get("extractor", {})
    expected_checkpoint = config.get("paths", {}).get("checkpoint_path")

    for path in npz_paths:
        try:
            with np.load(path, allow_pickle=True) as data:
                embedding = _read_embedding_array(data)
                if embedding is not None:
                    shape_counts[_shape_key(embedding.shape)] += 1
                    if not np.isfinite(embedding).all():
                        nonfinite_count += 1
                l2 = data["embedding_l2"] if "embedding_l2" in data else embedding
                if l2 is not None:
                    norm = float(np.linalg.norm(np.asarray(l2).reshape(-1)))
                    l2_min = norm if l2_min is None else min(l2_min, norm)
                    l2_max = norm if l2_max is None else max(l2_max, norm)
                metadata = _load_npz_metadata(data)
                if metadata is not None:
                    mock_values[str(metadata.get("mock"))] += 1
                    if extractor.get("pc_model") and metadata.get("pc_model") not in (None, extractor.get("pc_model")):
                        metadata_mismatches.append(f"{path.name}: pc_model={metadata.get('pc_model')}")
                    if extractor.get("embed_dim") and metadata.get("embed_dim") not in (None, extractor.get("embed_dim")):
                        metadata_mismatches.append(f"{path.name}: embed_dim={metadata.get('embed_dim')}")
                    if expected_checkpoint and metadata.get("checkpoint_path") not in (None, expected_checkpoint):
                        metadata_mismatches.append(f"{path.name}: checkpoint_path={metadata.get('checkpoint_path')}")
                else:
                    metadata_missing_count += 1
        except Exception as exc:  # pragma: no cover - defensive audit path
            metadata_mismatches.append(f"{path.name}: unreadable npz ({exc})")

    status = STATUS_PASS
    reasons: list[str] = []
    if not npz_paths:
        status = STATUS_NOT_AVAILABLE
        reasons.append("no .npz embeddings found")
    if nonfinite_count:
        status = STATUS_FAIL
        reasons.append("non-finite embeddings detected")
    if summary is None:
        status = _downgrade(status, STATUS_WARN)
        reasons.append("summary.json missing")
    if metadata_missing_count:
        status = _downgrade(status, STATUS_WARN)
        reasons.append("some .npz files do not expose metadata")
    if metadata_mismatches:
        status = _downgrade(status, STATUS_WARN)
        reasons.append("metadata mismatches or unreadable metadata detected")
    if mock_values and any(value.lower() == "true" for value in mock_values):
        status = STATUS_FAIL
        reasons.append("mock embeddings detected in extraction artifacts")

    check.update(
        status=status,
        reason="; ".join(reasons) if reasons else "embedding artifacts look readable",
        npz_count=len(npz_paths),
        summary_exists=summary is not None,
        summary=summary,
        embedding_shape_distribution=dict(shape_counts),
        embedding_l2_norm_min=l2_min,
        embedding_l2_norm_max=l2_max,
        nonfinite_count=nonfinite_count,
        metadata_missing_count=metadata_missing_count,
        mock_distribution=dict(mock_values),
        metadata_mismatches=metadata_mismatches[:50],
    )
    return check


def audit_similarity(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["paths"]["similarity_output_dir"])
    check = _base_check("similarity", output_dir)
    if not output_dir.exists():
        check.update(status=STATUS_NOT_AVAILABLE, reason="similarity_output_dir does not exist")
        return check

    summary = _read_json(output_dir / "similarity_summary.json")
    nearest_exists = (output_dir / "nearest_neighbors.json").exists()
    full_matrix_exists = (output_dir / "cosine_similarity.csv").exists()
    near_duplicate_count = _count_rows(output_dir / "near_duplicate_pairs.csv")
    outlier_count = _count_rows(output_dir / "outlier_samples.csv")

    status = STATUS_PASS
    reasons: list[str] = []
    if summary is None:
        status = STATUS_NOT_AVAILABLE
        reasons.append("similarity_summary.json missing")
    else:
        if summary.get("invalid_count", summary.get("invalid_embeddings", 0)):
            status = STATUS_FAIL
            reasons.append("invalid embeddings reported")
        if summary.get("all_pairwise_similarity_near_one"):
            status = STATUS_FAIL
            reasons.append("all pairwise similarities are near one")
    if not nearest_exists:
        status = _downgrade(status, STATUS_WARN)
        reasons.append("nearest_neighbors.json missing")

    check.update(
        status=status,
        reason="; ".join(reasons) if reasons else "similarity artifacts look readable",
        summary_exists=summary is not None,
        summary=summary,
        nearest_neighbors_exists=nearest_exists,
        cosine_similarity_csv_exists=full_matrix_exists,
        near_duplicate_count=near_duplicate_count,
        outlier_count=outlier_count,
    )
    return check


def audit_probe(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["paths"]["probe_output_dir"])
    check = _base_check("probe", output_dir)
    if not output_dir.exists():
        check.update(status=STATUS_NOT_AVAILABLE, reason="probe_output_dir does not exist")
        return check

    probe_dirs = _find_probe_dirs(output_dir)
    if not probe_dirs:
        check.update(status=STATUS_NOT_AVAILABLE, reason="no probe_results.csv found")
        return check

    result_rows: list[dict[str, str]] = []
    for probe_dir in probe_dirs:
        result_rows.extend(_read_csv_rows(probe_dir / "probe_results.csv"))

    split_types = sorted({row.get("split_type", "") for row in result_rows if row.get("split_type")})
    feature_sets = sorted({row.get("feature_set", "") for row in result_rows if row.get("feature_set")})
    tasks = sorted({row.get("task", "") for row in result_rows if row.get("task")})
    exploratory_rows = [row for row in result_rows if row.get("exploratory_only") in {"true", "True", "1"}]
    failure_rows = [row for row in result_rows if row.get("status") not in {"ok", "success", "PASS", ""}]

    status = STATUS_PASS
    reasons: list[str] = []
    if "random" not in split_types:
        status = _downgrade(status, STATUS_WARN)
        reasons.append("random split not found")
    if "leave_one_site_out" not in split_types:
        status = _downgrade(status, STATUS_WARN)
        reasons.append("leave-one-site-out split not found")
    for feature_set in ("uni3d", "geometry", "uni3d_plus_geometry"):
        if feature_set not in feature_sets:
            status = _downgrade(status, STATUS_WARN)
            reasons.append(f"feature_set {feature_set} not found")
    if exploratory_rows:
        status = _downgrade(status, STATUS_WARN)
        reasons.append("exploratory_only rows found")

    check.update(
        status=status,
        reason="; ".join(reasons) if reasons else "probe artifacts look readable",
        probe_dirs=[str(path) for path in probe_dirs],
        result_row_count=len(result_rows),
        split_types=split_types,
        feature_sets=feature_sets,
        tasks=tasks,
        failure_row_count=len(failure_rows),
        exploratory_row_count=len(exploratory_rows),
        label_inventory_exists=any((path / "label_inventory.json").exists() for path in probe_dirs),
        alignment_report_exists=any((path / "alignment_report.json").exists() for path in probe_dirs),
        per_split_predictions_exists=any((path / "per_split_predictions.csv").exists() for path in probe_dirs),
    )
    return check


def write_audit_outputs(report: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    json_path = target_dir / "experiment_audit_report.json"
    md_path = target_dir / "experiment_audit_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_audit_markdown(report), encoding="utf-8")
    return json_path, md_path


def render_audit_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Uni3D Experiment Audit | {report['experiment_name']}",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Overall status: `{report['overall_status']}`",
        "",
        "## Checks",
    ]
    for name, check in report.get("checks", {}).items():
        lines.extend(
            [
                f"### {name}",
                f"- Status: `{check.get('status')}`",
                f"- Reason: {check.get('reason', '')}",
                f"- Path: `{check.get('path', '')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Boundary",
            "",
            "This audit is read-only. It does not run Uni3D forward, train Uni3D, or modify main QA/baseline code.",
            "",
        ]
    )
    return "\n".join(lines)


def combine_statuses(statuses: Any) -> str:
    status_list = list(statuses)
    if not status_list:
        return STATUS_NOT_AVAILABLE
    if STATUS_FAIL in status_list:
        return STATUS_FAIL
    if STATUS_WARN in status_list:
        return STATUS_WARN
    if all(status == STATUS_NOT_AVAILABLE for status in status_list):
        return STATUS_NOT_AVAILABLE
    if STATUS_NOT_AVAILABLE in status_list:
        return STATUS_WARN
    return STATUS_PASS


def _base_check(stage: str, path: Path) -> dict[str, Any]:
    return {"stage": stage, "path": str(path), "status": STATUS_NOT_AVAILABLE, "reason": ""}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _count_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    return len(_read_csv_rows(path))


def _shape_key(shape: tuple[int, ...]) -> str:
    return "x".join(str(value) for value in shape)


def _read_embedding_array(data: Any) -> np.ndarray | None:
    for key in ("embedding_l2", "embedding_raw"):
        if key in data:
            return np.asarray(data[key])
    return None


def _load_npz_metadata(data: Any) -> dict[str, Any] | None:
    if "metadata" not in data:
        return None
    value = data["metadata"]
    try:
        if value.shape == ():
            raw = value.item()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            if isinstance(raw, str):
                return json.loads(raw)
            if isinstance(raw, dict):
                return raw
    except Exception:
        return None
    return None


def _downgrade(current: str, candidate: str) -> str:
    order = {STATUS_PASS: 0, STATUS_WARN: 1, STATUS_FAIL: 2, STATUS_NOT_AVAILABLE: 1}
    return candidate if order[candidate] > order.get(current, 0) else current


def _find_probe_dirs(root: Path) -> list[Path]:
    if (root / "probe_results.csv").exists():
        return [root]
    return sorted(path.parent for path in root.rglob("probe_results.csv"))


def _finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except Exception:
        return None
    return result if math.isfinite(result) else None
