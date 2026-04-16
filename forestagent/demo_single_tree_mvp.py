"""Very small demo script for the single-tree MVP v1."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from typing import Any

from forestagent.backends.base_backend import BaseBackend
from forestagent.mvp import run_single_tree_analysis


def build_demo_cases(include_failure_demo: bool = False) -> list[dict[str, str | None]]:
    """Return a fixed set of small demo cases for the MVP."""

    cases: list[dict[str, str | None]] = [
        {"label": "q1_dbh", "task_id": "q1_dbh", "question": None},
        {"label": "q2_height", "task_id": None, "question": "这棵树的树高是多少？"},
        {"label": "q3_crown_width", "task_id": "q3_crown_width", "question": None},
        {"label": "tree_report_q123", "task_id": None, "question": "给我一个简短单木报告"},
    ]
    if include_failure_demo:
        cases.append(
            {
                "label": "ambiguous_question",
                "task_id": None,
                "question": "这棵树的胸径和树高是多少？",
            }
        )
    return cases


def run_demo_cases(
    point_cloud: str,
    *,
    config_path: str | None = None,
    include_failure_demo: bool = False,
    backend: BaseBackend | None = None,
    cases: Iterable[dict[str, str | None]] | None = None,
) -> dict[str, Any]:
    """Run a fixed set of MVP demo cases on one point cloud."""

    demo_cases = list(cases) if cases is not None else build_demo_cases(include_failure_demo)
    results: list[dict[str, Any]] = []
    for case in demo_cases:
        response = run_single_tree_analysis(
            point_cloud,
            task_id=case.get("task_id"),
            question=case.get("question"),
            backend=backend,
            config_path=config_path,
        )
        results.append(
            {
                "label": case.get("label"),
                "task_id": case.get("task_id"),
                "question": case.get("question"),
                "response": response,
            }
        )

    return {
        "point_cloud": point_cloud,
        "case_count": len(results),
        "cases": results,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the small demo script."""

    parser = argparse.ArgumentParser(description="Single-tree MVP demo runner")
    parser.add_argument("--point-cloud", required=True)
    parser.add_argument("--config-path")
    parser.add_argument("--include-failure-demo", action="store_true")
    return parser


def main() -> None:
    """Run the small MVP demo script."""

    args = build_parser().parse_args()
    payload = run_demo_cases(
        args.point_cloud,
        config_path=args.config_path,
        include_failure_demo=args.include_failure_demo,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
