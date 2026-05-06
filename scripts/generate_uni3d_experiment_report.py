"""Generate Chinese Markdown records for offline Uni3D experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from forestagent.experiments.uni3d_experiment_config import experiment_name, load_experiment_config  # noqa: E402
from forestagent.experiments.uni3d_report import generate_experiment_report, generate_memory_snippet  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an offline Uni3D experiment Markdown report.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--audit-json", default=None)
    parser.add_argument("--comparison-json", default=None)
    parser.add_argument(
        "--output-md",
        default=None,
        help="Report path. Defaults to outputs/local_reports/experiment_reports/<experiment>_report.md.",
    )
    parser.add_argument("--memory-snippet-output", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_experiment_config(args.config)
    audit = _read_json(args.audit_json)
    comparison = _read_json(args.comparison_json)

    output_md = Path(args.output_md) if args.output_md else Path("outputs/local_reports/experiment_reports") / f"{experiment_name(config)}_report.md"
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(generate_experiment_report(config, audit=audit, comparison=comparison), encoding="utf-8")
    print(f"Wrote experiment report: {output_md}")

    if args.memory_snippet_output:
        snippet_path = Path(args.memory_snippet_output)
        snippet_path.parent.mkdir(parents=True, exist_ok=True)
        snippet_path.write_text(generate_memory_snippet(config, audit=audit, comparison=comparison), encoding="utf-8")
        print(f"Wrote PROJECT_MEMORY snippet: {snippet_path}")
    return 0


def _read_json(path: str | None) -> dict | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

