"""Config loading and command planning for offline Uni3D experiments.

This module deliberately avoids importing torch, the Uni3D repo, or the real
extractor. It only builds reproducible command plans that the user may run
manually on the GPU server.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when an experiment config is incomplete or inconsistent."""


@dataclass(frozen=True)
class PlannedCommand:
    stage: str
    name: str
    command: str
    output_dir: str | None = None


REQUIRED_TOP_LEVEL = ("experiment", "paths", "extractor", "analysis", "metadata")
REQUIRED_EXPERIMENT = ("name", "stage", "version")
REQUIRED_PATHS = (
    "npy_input_dir",
    "embedding_output_dir",
    "similarity_output_dir",
    "probe_output_dir",
    "label_path",
    "checkpoint_path",
    "uni3d_repo_path",
)
REQUIRED_EXTRACTOR = (
    "pc_model",
    "pc_feat_dim",
    "embed_dim",
    "pc_encoder_dim",
    "num_group",
    "group_size",
    "device",
)


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML or JSON experiment config."""

    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Experiment config does not exist: {config_path}")

    text = config_path.read_text(encoding="utf-8")
    suffix = config_path.suffix.lower()
    if suffix == ".json":
        config = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ConfigError(
                "PyYAML is required to read YAML configs. Install pyyaml or use a JSON config."
            ) from exc
        config = yaml.safe_load(text)
    else:
        raise ConfigError(f"Unsupported config format {suffix!r}; use .yaml, .yml, or .json")

    if not isinstance(config, dict):
        raise ConfigError(f"Experiment config must contain a mapping: {config_path}")
    validate_experiment_config(config)
    return config


def validate_experiment_config(config: dict[str, Any]) -> None:
    """Validate the stable fields used by the offline experiment tools."""

    for key in REQUIRED_TOP_LEVEL:
        _require_mapping(config, key)
    for key in REQUIRED_EXPERIMENT:
        _require_value(config["experiment"], f"experiment.{key}")
    for key in REQUIRED_PATHS:
        _require_value(config["paths"], f"paths.{key}")
    for key in REQUIRED_EXTRACTOR:
        _require_value(config["extractor"], f"extractor.{key}")

    conversion = config.get("conversion", {})
    if conversion.get("enabled", False):
        for key in ("raw_las_dir",):
            _require_value(config["paths"], f"paths.{key}")
        for key in ("output_mode", "color_policy", "num_points", "seed", "pattern"):
            _require_value(conversion, f"conversion.{key}")

    similarity = config["analysis"].get("similarity", {})
    if similarity.get("save_full_matrix") and similarity.get("no_full_matrix"):
        raise ConfigError("analysis.similarity cannot set both save_full_matrix and no_full_matrix")


def experiment_name(config: dict[str, Any]) -> str:
    return str(config["experiment"]["name"])


def build_experiment_plan(
    config: dict[str, Any],
    *,
    stage: str = "all",
    shell: str = "bash",
    validate_paths: bool = False,
) -> dict[str, Any]:
    """Build a non-executing command plan from a Uni3D experiment config."""

    if shell != "bash":
        raise ConfigError("Only bash command plan generation is currently supported.")
    if stage not in {"all", "convert", "extract", "similarity", "probe"}:
        raise ConfigError(f"Unsupported stage: {stage}")

    validate_experiment_config(config)
    warnings = collect_config_warnings(config)
    if validate_paths:
        warnings.extend(_validate_declared_paths(config))

    commands: list[PlannedCommand] = []
    if stage in {"all", "convert"} and config.get("conversion", {}).get("enabled", False):
        commands.append(_build_convert_command(config))
    if stage in {"all", "extract"}:
        commands.append(_build_extract_command(config))
    if stage in {"all", "similarity"}:
        commands.append(_build_similarity_command(config))
    if stage in {"all", "probe"}:
        commands.extend(_build_probe_commands(config))

    return {
        "experiment": config["experiment"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "dry_run": True,
        "warnings": warnings,
        "commands": [command.__dict__ for command in commands],
    }


def write_plan_outputs(plan: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    """Write JSON and shell-script plan files."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    name = str(plan["experiment"]["name"])
    json_path = target_dir / f"{name}_plan.json"
    shell_path = target_dir / f"{name}_commands.sh"

    json_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    shell_path.write_text(render_shell_plan(plan), encoding="utf-8")
    return json_path, shell_path


def render_shell_plan(plan: dict[str, Any]) -> str:
    name = str(plan["experiment"]["name"])
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"# Uni3D offline experiment command plan: {name}",
        f"# Generated at: {plan['generated_at']}",
        "# This file is a command plan only. It was generated without running Uni3D forward.",
        "# Run long GPU jobs manually on the server, preferably inside tmux.",
        "# Do not commit data/, outputs/, checkpoints/, weights/, *.npy, or *.npz artifacts.",
        "",
    ]
    for warning in plan.get("warnings", []):
        lines.append(f"# WARNING: {warning}")
    if plan.get("warnings"):
        lines.append("")
    for command in plan.get("commands", []):
        lines.extend(
            [
                f"echo '[{command['stage']}] {command['name']}'",
                command["command"],
                "",
            ]
        )
    return "\n".join(lines)


def collect_config_warnings(config: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    conversion = config.get("conversion", {})
    metadata = config.get("metadata", {})
    color_policy = conversion.get("color_policy")
    output_mode = conversion.get("output_mode")

    if color_policy == "species_palette":
        warnings.append(
            "species_palette encodes labels as colors; this is label leakage and must not be used for formal species probes."
        )
    if color_policy == "site_palette":
        warnings.append(
            "site_palette can encode site/domain identity; use only for diagnostics, not formal generalization claims."
        )
    if output_mode == "legacy_mixed" or metadata.get("mixed_rgb_availability"):
        warnings.append(
            "Legacy mixed RGB availability is present; interpret similarity/probe results cautiously."
        )
    if conversion.get("semantic_color_likely"):
        warnings.append("Color channels are marked semantic/discrete; do not describe them as natural RGB.")
    return warnings


def _build_convert_command(config: dict[str, Any]) -> PlannedCommand:
    paths = config["paths"]
    conversion = config["conversion"]
    args = [
        "python",
        "scripts/convert_las_to_uni3d_npy.py",
        "--input-dir",
        paths["raw_las_dir"],
        "--output-dir",
        paths["npy_input_dir"],
        "--pattern",
        conversion.get("pattern", "*.las"),
        "--num-points",
        str(conversion.get("num_points", 10000)),
        "--seed",
        str(conversion.get("seed", 42)),
        "--output-mode",
        conversion["output_mode"],
        "--color-policy",
        conversion["color_policy"],
    ]
    if conversion.get("recursive", True):
        args.append("--recursive")
    if conversion.get("skip_existing", False):
        args.append("--skip-existing")
    if conversion.get("overwrite", False):
        args.append("--overwrite")
    if conversion.get("constant_color") is not None:
        args.append("--constant-color")
        args.extend(str(value) for value in conversion["constant_color"])
    if conversion.get("previous_npy_dir"):
        args.extend(["--previous-npy-dir", conversion["previous_npy_dir"]])
    if conversion.get("label_path"):
        args.extend(["--label-path", conversion["label_path"]])
    elif config["paths"].get("label_path"):
        args.extend(["--label-path", config["paths"]["label_path"]])
    limit = _normalize_limit(conversion.get("limit"))
    if limit is not None:
        args.extend(["--limit", str(limit)])
    return PlannedCommand(
        stage="convert",
        name="LAS to Uni3D NPY",
        command=_quote_command(args),
        output_dir=str(paths["npy_input_dir"]),
    )


def _build_extract_command(config: dict[str, Any]) -> PlannedCommand:
    paths = config["paths"]
    extractor = config["extractor"]
    args = [
        "python",
        "scripts/batch_extract_uni3d_embeddings.py",
        "--input-dir",
        paths["npy_input_dir"],
        "--output-dir",
        paths["embedding_output_dir"],
        "--checkpoint-path",
        paths["checkpoint_path"],
        "--uni3d-repo-path",
        paths["uni3d_repo_path"],
        "--device",
        extractor.get("device", "cuda"),
        "--pc-model",
        extractor["pc_model"],
        "--pc-feat-dim",
        str(extractor["pc_feat_dim"]),
        "--embed-dim",
        str(extractor["embed_dim"]),
        "--pc-encoder-dim",
        str(extractor["pc_encoder_dim"]),
        "--num-group",
        str(extractor["num_group"]),
        "--group-size",
        str(extractor["group_size"]),
        "--pattern",
        extractor.get("pattern", "*.npy"),
    ]
    if extractor.get("recursive", True):
        args.append("--recursive")
    if extractor.get("skip_existing", False):
        args.append("--skip-existing")
    if extractor.get("overwrite", False):
        args.append("--overwrite")
    if extractor.get("yz_swap", False):
        args.append("--yz-swap")
    limit = _normalize_limit(extractor.get("limit"))
    if limit is not None:
        args.extend(["--limit", str(limit)])
    return PlannedCommand(
        stage="extract",
        name="Batch Uni3D embedding extraction",
        command=_quote_command(args),
        output_dir=str(paths["embedding_output_dir"]),
    )


def _build_similarity_command(config: dict[str, Any]) -> PlannedCommand:
    paths = config["paths"]
    similarity = config["analysis"].get("similarity", {})
    args = [
        "python",
        "scripts/analyze_uni3d_embedding_similarity.py",
        "--embedding-dir",
        paths["embedding_output_dir"],
        "--output-dir",
        paths["similarity_output_dir"],
        "--top-k",
        str(similarity.get("top_k", 5)),
    ]
    max_samples = _normalize_limit(similarity.get("max_samples"))
    if max_samples is not None:
        args.extend(["--max-samples", str(max_samples)])
    if similarity.get("seed") is not None:
        args.extend(["--seed", str(similarity["seed"])])
    if similarity.get("save_full_matrix"):
        args.append("--save-full-matrix")
    if similarity.get("no_full_matrix"):
        args.append("--no-full-matrix")
    return PlannedCommand(
        stage="similarity",
        name="Embedding similarity analysis",
        command=_quote_command(args),
        output_dir=str(paths["similarity_output_dir"]),
    )


def _build_probe_commands(config: dict[str, Any]) -> list[PlannedCommand]:
    paths = config["paths"]
    probe = config["analysis"].get("probe", {})
    splits = probe.get("splits", ["random", "leave_one_site_out"])
    if splits == ["all"] or splits == "all":
        splits = ["all"]
    if isinstance(splits, str):
        splits = [splits]

    commands: list[PlannedCommand] = []
    for split in splits:
        output_dir = paths["probe_output_dir"]
        if len(splits) > 1:
            output_dir = str(Path(output_dir) / str(split))
        args = [
            "python",
            "scripts/probe_uni3d_embeddings_strict.py",
            "--embedding-dir",
            paths["embedding_output_dir"],
            "--label-path",
            paths["label_path"],
            "--output-dir",
            output_dir,
            "--tasks",
            *[str(value) for value in probe.get("tasks", ["all"])],
            "--feature-sets",
            *[str(value) for value in probe.get("feature_sets", ["all"])],
            "--split",
            str(split),
            "--seeds",
            *[str(value) for value in probe.get("seeds", [0, 1, 2, 3, 4])],
        ]
        if probe.get("near_duplicate_path"):
            args.extend(["--near-duplicate-path", probe["near_duplicate_path"]])
        if probe.get("outlier_path"):
            args.extend(["--outlier-path", probe["outlier_path"]])
        if probe.get("review_table_path"):
            args.extend(["--review-table-path", probe["review_table_path"]])
        if probe.get("exclude_near_duplicates"):
            args.append("--exclude-near-duplicates")
        if probe.get("exclude_outliers"):
            args.append("--exclude-outliers")
        if probe.get("rgb_source") and probe.get("rgb_source") != "all":
            args.extend(["--rgb-source", str(probe["rgb_source"])])
        if probe.get("overwrite", False):
            args.append("--overwrite")
        commands.append(
            PlannedCommand(
                stage="probe",
                name=f"Strict downstream probe ({split})",
                command=_quote_command(args),
                output_dir=str(output_dir),
            )
        )
    return commands


def _require_mapping(config: dict[str, Any], key: str) -> None:
    if key not in config or not isinstance(config[key], dict):
        raise ConfigError(f"Missing required mapping: {key}")


def _require_value(mapping: dict[str, Any], dotted_key: str) -> None:
    key = dotted_key.split(".")[-1]
    if key not in mapping or mapping[key] in (None, ""):
        raise ConfigError(f"Missing required config value: {dotted_key}")


def _quote_command(args: list[str]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in args)


def _normalize_limit(value: Any) -> int | None:
    if value in (None, "", "none", "None"):
        return None
    value_int = int(value)
    if value_int == 0:
        return None
    return value_int


def _validate_declared_paths(config: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    paths = config["paths"]
    for key in ("raw_las_dir", "npy_input_dir", "embedding_output_dir", "similarity_output_dir", "probe_output_dir"):
        value = paths.get(key)
        if value and not Path(value).exists():
            warnings.append(f"Declared path does not currently exist on this machine: paths.{key}={value}")
    for key in ("label_path", "checkpoint_path", "uni3d_repo_path"):
        value = paths.get(key)
        if value and not Path(value).exists():
            warnings.append(f"Declared dependency path is not visible on this machine: paths.{key}={value}")
    return warnings

