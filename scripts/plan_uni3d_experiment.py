"""Generate non-executing server command plans for offline Uni3D experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from forestagent.experiments.uni3d_experiment_config import (  # noqa: E402
    build_experiment_plan,
    load_experiment_config,
    write_plan_outputs,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan an offline Uni3D experiment without running it.")
    parser.add_argument("--config", required=True, help="Experiment YAML/JSON config.")
    parser.add_argument(
        "--output-plan",
        default="outputs/local_reports/experiment_plans",
        help="Directory for generated .sh and .json plans.",
    )
    parser.add_argument("--stage", default="all", choices=["all", "convert", "extract", "similarity", "probe"])
    parser.add_argument("--shell", default="bash", choices=["bash"])
    parser.add_argument("--dry-run", action="store_true", default=True, help="Always true; commands are not executed.")
    parser.add_argument("--validate-paths", action="store_true", help="Warn if declared paths are absent locally.")
    parser.add_argument("--print", dest="print_plan", action="store_true", help="Print commands to the terminal.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_experiment_config(args.config)
    plan = build_experiment_plan(
        config,
        stage=args.stage,
        shell=args.shell,
        validate_paths=args.validate_paths,
    )
    json_path, shell_path = write_plan_outputs(plan, args.output_plan)
    if args.print_plan:
        for warning in plan.get("warnings", []):
            print(f"WARNING: {warning}")
        for command in plan["commands"]:
            print(f"\n[{command['stage']}] {command['name']}")
            print(command["command"])
    print(f"Wrote plan JSON: {json_path}")
    print(f"Wrote shell plan: {shell_path}")
    print("No commands were executed. Run the shell plan manually on the GPU server when ready.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

