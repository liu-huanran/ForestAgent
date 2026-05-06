"""Compare multiple offline Uni3D experiment artifact sets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from forestagent.experiments.uni3d_experiment_compare import compare_experiments, write_comparison_outputs  # noqa: E402
from forestagent.experiments.uni3d_experiment_config import load_experiment_config  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare configured offline Uni3D experiments.")
    parser.add_argument("--configs", nargs="+", required=True, help="Experiment YAML/JSON configs.")
    parser.add_argument(
        "--output-dir",
        default="outputs/local_reports/experiment_comparisons",
        help="Directory for comparison outputs.",
    )
    parser.add_argument("--print", dest="print_report", action="store_true", help="Print a short comparison summary.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configs = [load_experiment_config(path) for path in args.configs]
    comparison = compare_experiments(configs)
    json_path, csv_path, md_path = write_comparison_outputs(comparison, args.output_dir)
    if args.print_report:
        print(f"Compared experiments: {comparison['experiment_count']}")
        for experiment in comparison["experiments"]:
            print(f"- {experiment['name']}: extraction={experiment['extraction'].get('status')}, probe={experiment['probe'].get('status')}")
    print(f"Wrote comparison JSON: {json_path}")
    print(f"Wrote comparison CSV: {csv_path}")
    print(f"Wrote comparison Markdown: {md_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

