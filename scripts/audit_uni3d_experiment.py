"""Read-only audit for configured offline Uni3D experiment artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from forestagent.experiments.uni3d_experiment_audit import audit_experiment, write_audit_outputs  # noqa: E402
from forestagent.experiments.uni3d_experiment_config import experiment_name, load_experiment_config  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit an offline Uni3D experiment without mutating artifacts.")
    parser.add_argument("--config", required=True, help="Experiment YAML/JSON config.")
    parser.add_argument("--stage", default="all", choices=["all", "conversion", "extraction", "similarity", "probe"])
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for audit report outputs. Defaults to outputs/local_reports/experiment_audits/<experiment>.",
    )
    parser.add_argument("--print", dest="print_report", action="store_true", help="Print a short audit summary.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_experiment_config(args.config)
    report = audit_experiment(config, stage=args.stage)
    output_dir = args.output_dir or str(Path("outputs/local_reports/experiment_audits") / experiment_name(config))
    json_path, md_path = write_audit_outputs(report, output_dir)
    if args.print_report:
        print(f"Experiment: {report['experiment_name']}")
        print(f"Overall status: {report['overall_status']}")
        for name, check in report.get("checks", {}).items():
            print(f"- {name}: {check.get('status')} ({check.get('reason')})")
    print(f"Wrote audit JSON: {json_path}")
    print(f"Wrote audit Markdown: {md_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

