"""Build a small demo bundle for the single-tree MVP v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from forestagent.backends.base_backend import BaseBackend
from forestagent.mvp import run_single_tree_analysis

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "outputs" / "single_tree_mvp_v1_demo_bundle_20260415"
_CURATED_SAMPLE_SPECS = (
    {
        "sample_id": "150_59",
        "point_cloud": _REPO_ROOT / "data" / "TLS" / "150" / "150_59.las",
        "reason": "Cross-task demo sample with very small q1/q2/q3 errors.",
    },
    {
        "sample_id": "150_24",
        "point_cloud": _REPO_ROOT / "data" / "TLS" / "150" / "150_24.las",
        "reason": "Stable demo sample with balanced q1/q2/q3 performance.",
    },
    {
        "sample_id": "151_17",
        "point_cloud": _REPO_ROOT / "data" / "TLS" / "151" / "151_17.las",
        "reason": "Third reference sample for report and unsupported-intent demo.",
    },
)
_FIXED_DEMO_CASES = (
    {"case_id": "q1_dbh", "task_id": "q1_dbh", "question": None},
    {"case_id": "q2_height", "task_id": "q2_height", "question": None},
    {"case_id": "q3_crown_width", "task_id": "q3_crown_width", "question": None},
    {"case_id": "tree_report_q123", "task_id": None, "question": "给我一个简短单木报告"},
    {"case_id": "unsupported_question", "task_id": None, "question": "这棵树有倒伏风险吗？"},
)


def build_demo_bundle(
    *,
    output_dir: str | Path | None = None,
    config_path: str | Path | None = None,
    backend: BaseBackend | None = None,
    sample_specs: list[dict[str, Any]] | None = None,
    cases: list[dict[str, str | None]] | None = None,
) -> dict[str, Any]:
    """Build a fixed single-tree MVP demo bundle and return its manifest."""

    target_dir = Path(output_dir) if output_dir is not None else _DEFAULT_OUTPUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    manifest_samples: list[dict[str, Any]] = []
    effective_samples = list(sample_specs) if sample_specs is not None else list(_CURATED_SAMPLE_SPECS)
    effective_cases = list(cases) if cases is not None else list(_FIXED_DEMO_CASES)

    for sample in effective_samples:
        sample_id = str(sample["sample_id"])
        point_cloud = str(sample["point_cloud"])
        sample_dir = target_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        case_entries: list[dict[str, Any]] = []
        for case in effective_cases:
            response = run_single_tree_analysis(
                point_cloud,
                task_id=case.get("task_id"),
                question=case.get("question"),
                backend=backend,
                config_path=config_path,
            )
            case_id = str(case["case_id"])
            case_path = sample_dir / f"{case_id}.json"
            case_path.write_text(
                json.dumps(response, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            case_entries.append(
                {
                    "case_id": case_id,
                    "task_id": case.get("task_id"),
                    "question": case.get("question"),
                    "status": response["status"],
                    "path": str(case_path),
                }
            )

        manifest_samples.append(
            {
                "sample_id": sample_id,
                "point_cloud": point_cloud,
                "reason": sample.get("reason", ""),
                "cases": case_entries,
            }
        )

    manifest = {
        "bundle_name": "single_tree_mvp_v1_demo_bundle",
        "output_dir": str(target_dir),
        "config_path": None if config_path is None else str(Path(config_path)),
        "sample_count": len(manifest_samples),
        "case_count_per_sample": len(effective_cases),
        "samples": manifest_samples,
    }
    manifest_path = target_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """Build the parser for the demo bundle generator."""

    parser = argparse.ArgumentParser(description="Build the single-tree MVP v1 demo bundle")
    parser.add_argument("--output-dir", default=str(_DEFAULT_OUTPUT_DIR))
    parser.add_argument("--config-path")
    return parser


def main() -> None:
    """CLI entrypoint for the demo bundle generator."""

    args = build_parser().parse_args()
    manifest = build_demo_bundle(
        output_dir=args.output_dir,
        config_path=args.config_path,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
