"""Minimal real TreeQSM backend for q1/q2 baseline validation."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from forestagent.backends.base_backend import BaseBackend
from forestagent.schemas import PointCloudInput, ToolResult

_SUPPORTED_POINT_CLOUD_FORMATS = frozenset({"las", "laz"})


class TreeQSMBackendError(RuntimeError):
    """Raised when the minimal TreeQSM backend cannot produce valid outputs."""


@dataclass(frozen=True)
class TreeQSMMetrics:
    """Minimal set of TreeQSM metrics needed for q1/q2 baseline validation."""

    dbh_cyl_m: float | None = None
    dbh_qsm_m: float | None = None
    tree_height_m: float | None = None


@dataclass(frozen=True)
class TreeQSMRunResult:
    """One cached TreeQSM run output for a single point cloud."""

    metrics: TreeQSMMetrics
    runtime_seconds: float


@dataclass(frozen=True)
class TreeQSMRunResolution:
    """Resolved q1/q2 run outcome, including optional degradation metadata."""

    run_result: TreeQSMRunResult
    degraded: bool = False
    degradation_reason: str | None = None
    raw_failure_stage: str | None = None


@dataclass(frozen=True)
class TreeQSMPreprocessStats:
    """Point-cloud preprocessing statistics for one diagnostic run."""

    input_point_count: int
    preprocessed_point_count: int
    removed_nonfinite_count: int
    removed_duplicate_count: int
    removed_outlier_count: int
    has_nan_or_inf_before: bool
    has_nan_or_inf_after: bool


@dataclass(frozen=True)
class TreeQSMDebugInfo:
    """Debug information collected around crown_measures before convhull."""

    crown_point_count_before_convhull: int | None = None
    unique_projected_2d_point_count: int | None = None
    duplicate_projected_2d_point_count: int | None = None
    has_nan_or_inf_in_crown_points: bool | None = None
    projected_rank: int | None = None
    is_projected_collinear: bool | None = None
    projected_x_span: float | None = None
    projected_y_span: float | None = None
    dbh_cyl_m: float | None = None
    dbh_qsm_m: float | None = None
    tree_height_m: float | None = None
    last_successful_stage: str | None = None
    convhull_identifier: str | None = None
    convhull_message: str | None = None
    skip_crown_applied: bool | None = None
    skip_crown_reason: str | None = None


@dataclass(frozen=True)
class TreeQSMDiagnosticRun:
    """Structured output of one TreeQSM failure-diagnosis attempt."""

    label: str
    status: str
    preprocess_stats: TreeQSMPreprocessStats
    debug_info: TreeQSMDebugInfo | None
    metrics: TreeQSMMetrics | None
    failure_message: str | None
    runtime_seconds: float | None


class TreeQSMBackend(BaseBackend):
    """Minimal TreeQSM backend that currently supports only q1/q2 metrics."""

    _SKIP_CROWN_ERROR_IDENTIFIER = "MATLAB:convhull:EmptyConvhull2DErrId"
    _SKIP_CROWN_FAILURE_STAGES = frozenset(
        {
            "tree_data_layer_spread_before_convhull",
            "tree_data_before_crown_convhull",
        }
    )

    def __init__(
        self,
        treeqsm_root: str | Path | None = None,
        matlab_executable: str | Path | None = None,
        timeout_seconds: int = 900,
        allow_skip_crown_for_q1_q2: bool = False,
    ) -> None:
        self._treeqsm_root = str(treeqsm_root) if treeqsm_root is not None else None
        self._matlab_executable = (
            str(matlab_executable) if matlab_executable is not None else None
        )
        self._timeout_seconds = timeout_seconds
        self._allow_skip_crown_for_q1_q2 = allow_skip_crown_for_q1_q2
        self._run_cache: dict[tuple[str, int], TreeQSMRunResult] = {}
        self._q1_q2_resolution_cache: dict[tuple[str, int], TreeQSMRunResolution] = {}

    def estimate_dbh(self, point_cloud: PointCloudInput) -> ToolResult:
        try:
            resolution = self._resolve_q1_q2_run(point_cloud)
            run_result = resolution.run_result
            metric_value_m, source_metric = self._select_dbh_metric(run_result.metrics)
        except TreeQSMBackendError as exc:
            return self._build_failed_result("estimate_dbh", point_cloud, str(exc))

        extra = {
            "backend": "treeqsm",
            "input_format": point_cloud.format,
            "runtime_seconds": round(run_result.runtime_seconds, 6),
            "source_metric": source_metric,
            "degraded": resolution.degraded,
        }
        if resolution.degraded:
            extra.update(
                {
                    "degradation_reason": resolution.degradation_reason,
                    "raw_failure_stage": resolution.raw_failure_stage,
                    "used_metrics": [source_metric],
                }
            )

        return ToolResult(
            tool_name="estimate_dbh",
            status="success",
            value=metric_value_m * 100.0,
            unit="cm",
            confidence=None,
            extra=extra,
            message=(
                f"TreeQSM DBH estimate loaded from {source_metric}."
                if not resolution.degraded
                else f"TreeQSM DBH estimate loaded from {source_metric} via experimental skip-crown degradation."
            ),
        )

    def estimate_height(self, point_cloud: PointCloudInput) -> ToolResult:
        try:
            resolution = self._resolve_q1_q2_run(point_cloud)
            run_result = resolution.run_result
            if run_result.metrics.tree_height_m is None:
                raise TreeQSMBackendError("TreeQSM output missing TreeHeight.")
        except TreeQSMBackendError as exc:
            return self._build_failed_result("estimate_height", point_cloud, str(exc))

        extra = {
            "backend": "treeqsm",
            "input_format": point_cloud.format,
            "runtime_seconds": round(run_result.runtime_seconds, 6),
            "source_metric": "TreeHeight",
            "degraded": resolution.degraded,
        }
        if resolution.degraded:
            extra.update(
                {
                    "degradation_reason": resolution.degradation_reason,
                    "raw_failure_stage": resolution.raw_failure_stage,
                    "used_metrics": ["TreeHeight"],
                }
            )

        return ToolResult(
            tool_name="estimate_height",
            status="success",
            value=run_result.metrics.tree_height_m,
            unit="m",
            confidence=None,
            extra=extra,
            message=(
                "TreeQSM tree height estimate loaded from TreeHeight."
                if not resolution.degraded
                else "TreeQSM tree height estimate loaded from TreeHeight via experimental skip-crown degradation."
            ),
        )

    def estimate_crown_width(self, point_cloud: PointCloudInput) -> ToolResult:
        return self._build_unsupported_result(
            "estimate_crown_width",
            point_cloud,
            "Unsupported by minimal TreeQSM backend baseline round.",
        )

    def estimate_tilt(self, point_cloud: PointCloudInput) -> ToolResult:
        return self._build_unsupported_result(
            "estimate_tilt",
            point_cloud,
            "Unsupported by minimal TreeQSM backend baseline round.",
        )

    def assess_quality(self, point_cloud: PointCloudInput) -> ToolResult:
        return self._build_unsupported_result(
            "assess_quality",
            point_cloud,
            "Unsupported by minimal TreeQSM backend baseline round.",
        )

    def run_failure_diagnosis(
        self,
        point_cloud: PointCloudInput,
        *,
        apply_preprocess: bool = False,
        try_skip_crown: bool = False,
    ) -> TreeQSMDiagnosticRun:
        """Run one instrumented TreeQSM diagnosis pass for a single point cloud."""

        self._validate_point_cloud_input(point_cloud)
        raw_points = self._load_point_cloud_matrix(point_cloud)
        preprocess_stats = self._build_identity_preprocess_stats(raw_points)
        points_for_run = raw_points
        label_parts = ["preprocessed" if apply_preprocess else "raw"]
        if try_skip_crown:
            label_parts.append("skip_crown")

        if apply_preprocess:
            points_for_run, preprocess_stats = self._preprocess_points_for_diagnosis(
                raw_points
            )

        if points_for_run.size == 0:
            return TreeQSMDiagnosticRun(
                label="_".join(label_parts),
                status="failed",
                preprocess_stats=preprocess_stats,
                debug_info=None,
                metrics=None,
                failure_message="No points remain after preprocessing.",
                runtime_seconds=None,
            )

        try:
            run_result, debug_info = self._run_instrumented_treeqsm_job(
                points=points_for_run,
                sample_name=Path(point_cloud.path).stem,
                try_skip_crown=try_skip_crown,
            )
        except TreeQSMBackendError as exc:
            debug_info = getattr(exc, "debug_info", None)
            runtime_seconds = getattr(exc, "runtime_seconds", None)
            return TreeQSMDiagnosticRun(
                label="_".join(label_parts),
                status="failed",
                preprocess_stats=preprocess_stats,
                debug_info=debug_info,
                metrics=None,
                failure_message=str(exc),
                runtime_seconds=runtime_seconds,
            )

        return TreeQSMDiagnosticRun(
            label="_".join(label_parts),
            status="success",
            preprocess_stats=preprocess_stats,
            debug_info=debug_info,
            metrics=run_result.metrics,
            failure_message=None,
            runtime_seconds=run_result.runtime_seconds,
        )

    def _get_treeqsm_run_result(self, point_cloud: PointCloudInput) -> TreeQSMRunResult:
        self._validate_point_cloud_input(point_cloud)
        cache_key = self._build_cache_key(point_cloud)
        cached = self._run_cache.get(cache_key)
        if cached is not None:
            return cached

        run_result = self._compute_treeqsm_run_result(point_cloud)
        self._run_cache[cache_key] = run_result
        return run_result

    def _resolve_q1_q2_run(self, point_cloud: PointCloudInput) -> TreeQSMRunResolution:
        self._validate_point_cloud_input(point_cloud)
        cache_key = self._build_cache_key(point_cloud)
        cached = self._q1_q2_resolution_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            resolution = TreeQSMRunResolution(run_result=self._get_treeqsm_run_result(point_cloud))
        except TreeQSMBackendError as raw_error:
            if not self._allow_skip_crown_for_q1_q2:
                raise
            resolution = self._attempt_skip_crown_recovery(point_cloud, raw_error)

        self._q1_q2_resolution_cache[cache_key] = resolution
        return resolution

    def _attempt_skip_crown_recovery(
        self,
        point_cloud: PointCloudInput,
        raw_error: TreeQSMBackendError,
    ) -> TreeQSMRunResolution:
        raw_error_message = str(raw_error)
        if (
            self._SKIP_CROWN_ERROR_IDENTIFIER not in raw_error_message
            or "crown_measures" not in raw_error_message
        ):
            raise raw_error

        diagnostic_run = self.run_failure_diagnosis(point_cloud, apply_preprocess=False)
        if not self._should_allow_skip_crown_recovery(diagnostic_run):
            raise raw_error

        skip_crown_run = self.run_failure_diagnosis(
            point_cloud,
            apply_preprocess=False,
            try_skip_crown=True,
        )
        if skip_crown_run.status != "success" or skip_crown_run.metrics is None:
            skip_failure_message = skip_crown_run.failure_message or "unknown skip-crown failure"
            raise TreeQSMBackendError(
                f"{raw_error_message} Skip-crown degradation attempt failed: {skip_failure_message}"
            )

        debug_info = diagnostic_run.debug_info
        source_metrics: list[str] = []
        if debug_info is not None:
            if debug_info.dbh_cyl_m is not None:
                source_metrics.append("DBHcyl")
            if debug_info.dbh_qsm_m is not None:
                source_metrics.append("DBHqsm")
            if debug_info.tree_height_m is not None:
                source_metrics.append("TreeHeight")

        return TreeQSMRunResolution(
            run_result=TreeQSMRunResult(
                metrics=skip_crown_run.metrics,
                runtime_seconds=skip_crown_run.runtime_seconds or 0.0,
            ),
            degraded=True,
            degradation_reason=(
                "Experimental skip-crown recovery after "
                f"{self._SKIP_CROWN_ERROR_IDENTIFIER} in crown_measures; "
                f"available intermediate metrics: {', '.join(source_metrics) or 'unknown'}."
            ),
            raw_failure_stage=debug_info.last_successful_stage if debug_info is not None else None,
        )

    @classmethod
    def _should_allow_skip_crown_recovery(
        cls,
        diagnostic_run: TreeQSMDiagnosticRun,
    ) -> bool:
        if diagnostic_run.status != "failed":
            return False
        debug_info = diagnostic_run.debug_info
        if debug_info is None:
            return False
        if debug_info.convhull_identifier != cls._SKIP_CROWN_ERROR_IDENTIFIER:
            return False
        if debug_info.last_successful_stage not in cls._SKIP_CROWN_FAILURE_STAGES:
            return False
        if debug_info.tree_height_m is None:
            return False
        return debug_info.dbh_cyl_m is not None or debug_info.dbh_qsm_m is not None

    @staticmethod
    def _build_identity_preprocess_stats(points: np.ndarray) -> TreeQSMPreprocessStats:
        has_nan_or_inf = bool(np.any(~np.isfinite(points)))
        return TreeQSMPreprocessStats(
            input_point_count=int(points.shape[0]),
            preprocessed_point_count=int(points.shape[0]),
            removed_nonfinite_count=0,
            removed_duplicate_count=0,
            removed_outlier_count=0,
            has_nan_or_inf_before=has_nan_or_inf,
            has_nan_or_inf_after=has_nan_or_inf,
        )

    @classmethod
    def _preprocess_points_for_diagnosis(
        cls, points: np.ndarray
    ) -> tuple[np.ndarray, TreeQSMPreprocessStats]:
        finite_mask = np.all(np.isfinite(points), axis=1)
        finite_points = points[finite_mask]
        removed_nonfinite_count = int(points.shape[0] - finite_points.shape[0])

        unique_points = np.unique(finite_points, axis=0)
        removed_duplicate_count = int(finite_points.shape[0] - unique_points.shape[0])

        filtered_points, removed_outlier_count = cls._remove_obvious_outliers(unique_points)
        preprocess_stats = TreeQSMPreprocessStats(
            input_point_count=int(points.shape[0]),
            preprocessed_point_count=int(filtered_points.shape[0]),
            removed_nonfinite_count=removed_nonfinite_count,
            removed_duplicate_count=removed_duplicate_count,
            removed_outlier_count=removed_outlier_count,
            has_nan_or_inf_before=bool(np.any(~np.isfinite(points))),
            has_nan_or_inf_after=bool(np.any(~np.isfinite(filtered_points))),
        )
        return filtered_points, preprocess_stats

    @staticmethod
    def _remove_obvious_outliers(points: np.ndarray) -> tuple[np.ndarray, int]:
        if points.size == 0:
            return points, 0

        median = np.median(points, axis=0)
        absolute_deviation = np.abs(points - median)
        mad = np.median(absolute_deviation, axis=0)
        robust_scale = 1.4826 * mad
        std_scale = np.std(points, axis=0)
        robust_scale = np.where(robust_scale < 1e-9, std_scale, robust_scale)
        robust_scale = np.where(robust_scale < 1e-9, 1.0, robust_scale)

        robust_z = absolute_deviation / robust_scale
        keep_mask = np.all(robust_z <= 6.0, axis=1)
        filtered_points = points[keep_mask]
        removed_outlier_count = int(points.shape[0] - filtered_points.shape[0])
        return filtered_points, removed_outlier_count

    @staticmethod
    def _build_cache_key(point_cloud: PointCloudInput) -> tuple[str, int]:
        point_cloud_path = Path(point_cloud.path).resolve()
        mtime_ns = point_cloud_path.stat().st_mtime_ns if point_cloud_path.exists() else 0
        return str(point_cloud_path), mtime_ns

    def _compute_treeqsm_run_result(self, point_cloud: PointCloudInput) -> TreeQSMRunResult:
        treeqsm_root = self._resolve_treeqsm_root()
        matlab_executable = self._resolve_matlab_executable()
        points = self._load_point_cloud_matrix(point_cloud)

        with tempfile.TemporaryDirectory(prefix="forestagent_treeqsm_") as temp_dir:
            working_dir = Path(temp_dir)
            (working_dir / "results").mkdir(parents=True, exist_ok=True)
            input_mat_path = working_dir / "forestagent_points.mat"
            output_mat_path = working_dir / "forestagent_qsm_output.mat"
            script_path = working_dir / "run_treeqsm_job.m"

            self._save_points_mat(input_mat_path, points)
            script_path.write_text(
                self._build_matlab_script(
                    treeqsm_root=treeqsm_root,
                    input_mat_path=input_mat_path,
                    output_mat_path=output_mat_path,
                    sample_name=Path(point_cloud.path).stem,
                ),
                encoding="utf-8",
            )

            command = [
                str(matlab_executable),
                "-batch",
                f"run('{self._matlab_string(script_path)}')",
            ]
            start_time = time.monotonic()
            try:
                completed = subprocess.run(
                    command,
                    cwd=working_dir,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise TreeQSMBackendError(
                    f"TreeQSM MATLAB run timed out after {self._timeout_seconds} seconds."
                ) from exc
            runtime_seconds = time.monotonic() - start_time

            if completed.returncode != 0:
                diagnostic = self._extract_process_diagnostic(completed)
                raise TreeQSMBackendError(
                    f"TreeQSM MATLAB run failed with exit code {completed.returncode}. "
                    f"{diagnostic}"
                )
            if not output_mat_path.exists():
                raise TreeQSMBackendError("TreeQSM did not produce the expected output MAT file.")

            metrics = self._load_metrics_from_output_mat(output_mat_path)
            return TreeQSMRunResult(metrics=metrics, runtime_seconds=runtime_seconds)

    def _run_instrumented_treeqsm_job(
        self,
        *,
        points: np.ndarray,
        sample_name: str,
        try_skip_crown: bool,
    ) -> tuple[TreeQSMRunResult, TreeQSMDebugInfo | None]:
        treeqsm_root = self._resolve_treeqsm_root()
        matlab_executable = self._resolve_matlab_executable()

        with tempfile.TemporaryDirectory(prefix="forestagent_treeqsm_debug_") as temp_dir:
            working_dir = Path(temp_dir)
            (working_dir / "results").mkdir(parents=True, exist_ok=True)
            input_mat_path = working_dir / "forestagent_points.mat"
            output_mat_path = working_dir / "forestagent_qsm_output.mat"
            debug_mat_path = working_dir / "forestagent_debug_info.mat"
            script_path = working_dir / "run_treeqsm_debug_job.m"
            override_dir = working_dir / "debug_override"
            override_dir.mkdir(parents=True, exist_ok=True)

            self._save_points_mat(input_mat_path, points)
            self._write_instrumented_tree_data_override(
                treeqsm_root=treeqsm_root,
                override_dir=override_dir,
                try_skip_crown=try_skip_crown,
            )
            script_path.write_text(
                self._build_debug_matlab_script(
                    treeqsm_root=treeqsm_root,
                    override_dir=override_dir,
                    input_mat_path=input_mat_path,
                    output_mat_path=output_mat_path,
                    sample_name=sample_name,
                ),
                encoding="utf-8",
            )

            command = [
                str(matlab_executable),
                "-batch",
                f"run('{self._matlab_string(script_path)}')",
            ]
            environment = dict(os.environ)
            environment["FORESTAGENT_TREEQSM_DEBUG_INFO"] = str(debug_mat_path)

            start_time = time.monotonic()
            try:
                completed = subprocess.run(
                    command,
                    cwd=working_dir,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout_seconds,
                    check=False,
                    env=environment,
                )
            except subprocess.TimeoutExpired as exc:
                timeout_error = TreeQSMBackendError(
                    f"TreeQSM MATLAB run timed out after {self._timeout_seconds} seconds."
                )
                timeout_error.runtime_seconds = time.monotonic() - start_time
                timeout_error.debug_info = self._load_debug_info_from_mat(debug_mat_path)
                raise timeout_error from exc

            runtime_seconds = time.monotonic() - start_time
            debug_info = self._load_debug_info_from_mat(debug_mat_path)
            if completed.returncode != 0:
                diagnostic = self._extract_process_diagnostic(completed)
                process_error = TreeQSMBackendError(
                    f"TreeQSM MATLAB run failed with exit code {completed.returncode}. "
                    f"{diagnostic}"
                )
                process_error.runtime_seconds = runtime_seconds
                process_error.debug_info = debug_info
                raise process_error
            if not output_mat_path.exists():
                missing_output_error = TreeQSMBackendError(
                    "TreeQSM did not produce the expected output MAT file."
                )
                missing_output_error.runtime_seconds = runtime_seconds
                missing_output_error.debug_info = debug_info
                raise missing_output_error

            metrics = self._load_metrics_from_output_mat(output_mat_path)
            return TreeQSMRunResult(metrics=metrics, runtime_seconds=runtime_seconds), debug_info

    @classmethod
    def _write_instrumented_tree_data_override(
        cls,
        *,
        treeqsm_root: Path,
        override_dir: Path,
        try_skip_crown: bool,
    ) -> None:
        original_tree_data_path = treeqsm_root / "src" / "main_steps" / "tree_data.m"
        original_source = original_tree_data_path.read_text(encoding="utf-8")
        instrumented_source = cls._build_instrumented_tree_data_source(
            original_source=original_source,
            try_skip_crown=try_skip_crown,
        )
        (override_dir / "tree_data.m").write_text(instrumented_source, encoding="utf-8")

    @classmethod
    def _build_instrumented_tree_data_source(
        cls,
        *,
        original_source: str,
        try_skip_crown: bool,
    ) -> str:
        source = original_source

        crown_call = "[treedata,spreads] = crown_measures(treedata,cylinder,branch);"
        if try_skip_crown:
            skip_wrapper = """try
  [treedata,spreads] = crown_measures(treedata,cylinder,branch);
catch ME
  DebugInfoPath = getenv('FORESTAGENT_TREEQSM_DEBUG_INFO');
  if ~isempty(DebugInfoPath)
    skip_crown_info = struct();
    skip_crown_info.applied = true;
    skip_crown_info.reason = ME.message;
    skip_crown_info.last_successful_stage = 'tree_data_skip_crown_recovery';
    if exist(DebugInfoPath,'file')
      save(DebugInfoPath,'skip_crown_info','-append');
    else
      save(DebugInfoPath,'skip_crown_info');
    end
  end
  if treedata.TreeHeight > 10
    m = 20;
  elseif treedata.TreeHeight > 2
    m = 10;
  else
    m = 5;
  end
  spreads = zeros(m,18);
  treedata.CrownDiamAve = 0;
  treedata.CrownDiamMax = 0;
  treedata.CrownAreaConv = 0;
  treedata.CrownAreaAlpha = 0;
  treedata.CrownBaseHeight = treedata.TreeHeight;
  treedata.CrownLength = 0;
  treedata.CrownRatio = 0;
  treedata.CrownVolumeConv = 0;
  treedata.CrownVolumeAlpha = 0;
end"""
            if crown_call not in source:
                raise TreeQSMBackendError("Could not instrument tree_data.m crown call.")
            source = source.replace(crown_call, skip_wrapper, 1)

        layer_convhull_snippet = """  I = P(:,3) >= bot+(j-1)*Hei/m & P(:,3) < bot+j*Hei/m;
  X = unique(P(I,:),'rows');
  if size(X,1) > 5
    [K,A] = convhull(X(:,1),X(:,2));"""
        instrumented_layer_convhull = """  I = P(:,3) >= bot+(j-1)*Hei/m & P(:,3) < bot+j*Hei/m;
  X = unique(P(I,:),'rows');
  if size(X,1) > 5
    X2D = X(:,1:2);
    X2DUnique = unique(X2D,'rows');
    DebugInfoPath = getenv('FORESTAGENT_TREEQSM_DEBUG_INFO');
    if ~isempty(DebugInfoPath)
      debug_info = struct();
      debug_info.crown_point_count_before_convhull = size(X,1);
      debug_info.unique_projected_2d_point_count = size(X2DUnique,1);
      debug_info.duplicate_projected_2d_point_count = size(X2D,1)-size(X2DUnique,1);
      debug_info.has_nan_or_inf_in_crown_points = any(~isfinite(X(:)));
      debug_info.dbh_cyl = double(treedata.DBHcyl);
      debug_info.dbh_qsm = double(treedata.DBHqsm);
      debug_info.tree_height = double(treedata.TreeHeight);
      debug_info.last_successful_stage = 'tree_data_layer_spread_before_convhull';
      if size(X2DUnique,1) > 0
        debug_info.projected_x_span = double(max(X2DUnique(:,1))-min(X2DUnique(:,1)));
        debug_info.projected_y_span = double(max(X2DUnique(:,2))-min(X2DUnique(:,2)));
      else
        debug_info.projected_x_span = 0;
        debug_info.projected_y_span = 0;
      end
      if size(X2DUnique,1) >= 2
        Xc = double(X2DUnique)-mean(double(X2DUnique),1);
        debug_info.projected_rank = rank(Xc);
      else
        debug_info.projected_rank = 0;
      end
      debug_info.is_projected_collinear = debug_info.projected_rank < 2;
      save(DebugInfoPath,'debug_info');
    end
    try
      [K,A] = convhull(X(:,1),X(:,2));
    catch ME
      if ~isempty(DebugInfoPath)
        debug_info.convhull_identifier = ME.identifier;
        debug_info.convhull_message = ME.message;
        save(DebugInfoPath,'debug_info');
      end
      rethrow(ME)
    end"""
        if layer_convhull_snippet not in source:
            raise TreeQSMBackendError("Could not instrument tree_data.m layer convhull block.")
        source = source.replace(layer_convhull_snippet, instrumented_layer_convhull, 1)

        convhull_snippet = "X = unique(P(:,1:2),'rows');\n[K,A] = convhull(X(:,1),X(:,2));"
        instrumented_convhull = """X2DAll = P(:,1:2);
X = unique(X2DAll,'rows');
DebugInfoPath = getenv('FORESTAGENT_TREEQSM_DEBUG_INFO');
if ~isempty(DebugInfoPath)
  debug_info = struct();
  debug_info.crown_point_count_before_convhull = size(P,1);
  debug_info.unique_projected_2d_point_count = size(X,1);
  debug_info.duplicate_projected_2d_point_count = size(X2DAll,1)-size(X,1);
  debug_info.has_nan_or_inf_in_crown_points = any(~isfinite(P(:)));
  debug_info.dbh_cyl = double(treedata.DBHcyl);
  debug_info.dbh_qsm = double(treedata.DBHqsm);
  debug_info.tree_height = double(treedata.TreeHeight);
  debug_info.last_successful_stage = 'tree_data_before_crown_convhull';
  if size(X,1) > 0
    debug_info.projected_x_span = double(max(X(:,1))-min(X(:,1)));
    debug_info.projected_y_span = double(max(X(:,2))-min(X(:,2)));
  else
    debug_info.projected_x_span = 0;
    debug_info.projected_y_span = 0;
  end
  if size(X,1) >= 2
    Xc = double(X)-mean(double(X),1);
    debug_info.projected_rank = rank(Xc);
  else
    debug_info.projected_rank = 0;
  end
  debug_info.is_projected_collinear = debug_info.projected_rank < 2;
  save(DebugInfoPath,'debug_info');
end
try
  [K,A] = convhull(X(:,1),X(:,2));
catch ME
  if ~isempty(DebugInfoPath)
    debug_info.convhull_identifier = ME.identifier;
    debug_info.convhull_message = ME.message;
    save(DebugInfoPath,'debug_info');
  end
  rethrow(ME)
end"""
        if convhull_snippet not in source:
            raise TreeQSMBackendError("Could not instrument tree_data.m convhull block.")
        source = source.replace(convhull_snippet, instrumented_convhull, 1)
        return source

    @classmethod
    def _build_debug_matlab_script(
        cls,
        *,
        treeqsm_root: Path,
        override_dir: Path,
        input_mat_path: Path,
        output_mat_path: Path,
        sample_name: str,
    ) -> str:
        src_path = treeqsm_root / "src"
        lines = [
            "try",
            f"  addpath(genpath('{cls._matlab_string(src_path)}'));",
            f"  addpath(genpath('{cls._matlab_string(override_dir)}'));",
            f"  load('{cls._matlab_string(input_mat_path)}', 'P');",
            "  inputs = define_input(P,1,1,1);",
            f"  inputs.name = '{cls._matlab_string(sample_name)}';",
            "  inputs.tree = 1;",
            "  inputs.model = 1;",
            "  inputs.savemat = 0;",
            "  inputs.savetxt = 0;",
            "  inputs.plot = 0;",
            "  inputs.disp = 0;",
            "  inputs.Dist = 0;",
            "  inputs.Tria = 0;",
            "  inputs.OnlyTree = 1;",
            "  QSM = treeqsm(P, inputs);",
            f"  save('{cls._matlab_string(output_mat_path)}', 'QSM', '-v7');",
            "catch ME",
            "  disp('FORESTAGENT_DEBUG_BEGIN');",
            "  disp(ME.identifier);",
            "  disp(ME.message);",
            "  disp(getReport(ME, 'extended', 'hyperlinks', 'off'));",
            "  disp('FORESTAGENT_DEBUG_END');",
            "  rethrow(ME);",
            "end",
        ]
        return "\n".join(lines) + "\n"

    def _resolve_treeqsm_root(self) -> Path:
        candidate = self._treeqsm_root or os.environ.get("TREEQSM_ROOT")
        if not candidate:
            raise TreeQSMBackendError(
                "TreeQSM root is not configured. Pass treeqsm_root or set TREEQSM_ROOT."
            )

        root_path = Path(candidate).resolve()
        if not (root_path / "src" / "treeqsm.m").exists():
            raise TreeQSMBackendError(
                f"TreeQSM entrypoint was not found under {root_path}."
            )
        return root_path

    def _resolve_matlab_executable(self) -> Path:
        candidate = self._matlab_executable or os.environ.get("MATLAB_EXE")
        if not candidate:
            candidate = shutil.which("matlab")
        if not candidate:
            raise TreeQSMBackendError(
                "MATLAB executable is not configured. Pass matlab_executable or set MATLAB_EXE."
            )

        matlab_path = Path(candidate)
        if matlab_path.is_dir():
            matlab_path = matlab_path / "matlab.exe"
        matlab_path = matlab_path.resolve()
        if not matlab_path.exists():
            raise TreeQSMBackendError(f"MATLAB executable was not found: {matlab_path}")
        return matlab_path

    @staticmethod
    def _validate_point_cloud_input(point_cloud: PointCloudInput) -> None:
        point_cloud_path = Path(point_cloud.path)
        if point_cloud.format.lower() not in _SUPPORTED_POINT_CLOUD_FORMATS:
            raise TreeQSMBackendError(
                f"TreeQSM backend only supports LAS/LAZ, got format={point_cloud.format!r}."
            )
        if not point_cloud_path.exists():
            raise TreeQSMBackendError(f"Point cloud file was not found: {point_cloud_path}")

    @staticmethod
    def _load_point_cloud_matrix(point_cloud: PointCloudInput) -> np.ndarray:
        try:
            import laspy
        except ModuleNotFoundError as exc:
            raise TreeQSMBackendError(
                "laspy is required for TreeQSMBackend point cloud loading."
            ) from exc

        las = laspy.read(point_cloud.path)
        points = np.column_stack((las.x, las.y, las.z)).astype(np.float64, copy=False)
        if points.size == 0:
            raise TreeQSMBackendError("Point cloud is empty and cannot be sent to TreeQSM.")
        return points

    @staticmethod
    def _save_points_mat(input_mat_path: Path, points: np.ndarray) -> None:
        try:
            from scipy.io import savemat
        except ModuleNotFoundError as exc:
            raise TreeQSMBackendError(
                "scipy is required for TreeQSMBackend MAT file generation."
            ) from exc

        savemat(input_mat_path, {"P": points}, do_compression=False)

    @classmethod
    def _load_metrics_from_output_mat(cls, output_mat_path: Path) -> TreeQSMMetrics:
        try:
            from scipy.io import loadmat
        except ModuleNotFoundError as exc:
            raise TreeQSMBackendError(
                "scipy is required for TreeQSMBackend MAT file parsing."
            ) from exc

        raw_output = loadmat(output_mat_path, squeeze_me=True, struct_as_record=False)
        qsm = raw_output.get("QSM")
        if qsm is None:
            qsm = raw_output.get("qsm")
        if qsm is None:
            raise TreeQSMBackendError("TreeQSM output MAT file does not contain QSM.")

        if isinstance(qsm, np.ndarray):
            if qsm.size == 0:
                raise TreeQSMBackendError("TreeQSM output MAT file contains an empty QSM.")
            qsm = qsm.flat[0]

        treedata = getattr(qsm, "treedata", None)
        if treedata is None:
            raise TreeQSMBackendError("TreeQSM output QSM is missing treedata.")

        return TreeQSMMetrics(
            dbh_cyl_m=cls._extract_optional_float(treedata, "DBHcyl"),
            dbh_qsm_m=cls._extract_optional_float(treedata, "DBHqsm"),
            tree_height_m=cls._extract_optional_float(treedata, "TreeHeight"),
        )

    @classmethod
    def _load_debug_info_from_mat(cls, debug_mat_path: Path) -> TreeQSMDebugInfo | None:
        if not debug_mat_path.exists():
            return None

        try:
            from scipy.io import loadmat
        except ModuleNotFoundError as exc:
            raise TreeQSMBackendError(
                "scipy is required for TreeQSM debug MAT file parsing."
            ) from exc

        raw_output = loadmat(debug_mat_path, squeeze_me=True, struct_as_record=False)
        debug_info = raw_output.get("debug_info")
        skip_crown_info = raw_output.get("skip_crown_info")
        if debug_info is None and skip_crown_info is None:
            return None

        return TreeQSMDebugInfo(
            crown_point_count_before_convhull=cls._extract_optional_int(
                debug_info, "crown_point_count_before_convhull"
            ),
            unique_projected_2d_point_count=cls._extract_optional_int(
                debug_info, "unique_projected_2d_point_count"
            ),
            duplicate_projected_2d_point_count=cls._extract_optional_int(
                debug_info, "duplicate_projected_2d_point_count"
            ),
            has_nan_or_inf_in_crown_points=cls._extract_optional_bool(
                debug_info, "has_nan_or_inf_in_crown_points"
            ),
            projected_rank=cls._extract_optional_int(debug_info, "projected_rank"),
            is_projected_collinear=cls._extract_optional_bool(
                debug_info, "is_projected_collinear"
            ),
            projected_x_span=cls._extract_optional_float(debug_info, "projected_x_span"),
            projected_y_span=cls._extract_optional_float(debug_info, "projected_y_span"),
            dbh_cyl_m=cls._extract_optional_float(debug_info, "dbh_cyl"),
            dbh_qsm_m=cls._extract_optional_float(debug_info, "dbh_qsm"),
            tree_height_m=cls._extract_optional_float(debug_info, "tree_height"),
            last_successful_stage=cls._extract_optional_string(
                skip_crown_info, "last_successful_stage"
            )
            or cls._extract_optional_string(debug_info, "last_successful_stage"),
            convhull_identifier=cls._extract_optional_string(
                debug_info, "convhull_identifier"
            ),
            convhull_message=cls._extract_optional_string(debug_info, "convhull_message"),
            skip_crown_applied=cls._extract_optional_bool(skip_crown_info, "applied"),
            skip_crown_reason=cls._extract_optional_string(skip_crown_info, "reason"),
        )

    @staticmethod
    def _extract_optional_float(obj: Any, field_name: str) -> float | None:
        value = TreeQSMBackend._extract_optional_value(obj, field_name)
        if value is None:
            return None
        return float(value)

    @staticmethod
    def _extract_optional_int(obj: Any, field_name: str) -> int | None:
        value = TreeQSMBackend._extract_optional_value(obj, field_name)
        if value is None:
            return None
        return int(value)

    @staticmethod
    def _extract_optional_bool(obj: Any, field_name: str) -> bool | None:
        value = TreeQSMBackend._extract_optional_value(obj, field_name)
        if value is None:
            return None
        return bool(value)

    @staticmethod
    def _extract_optional_string(obj: Any, field_name: str) -> str | None:
        value = TreeQSMBackend._extract_optional_value(obj, field_name)
        if value is None:
            return None
        return str(value)

    @staticmethod
    def _extract_optional_value(obj: Any, field_name: str) -> Any | None:
        if obj is None:
            return None
        value = getattr(obj, field_name, None)
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return None
            value = value.flat[0]
        return value

    @staticmethod
    def _select_dbh_metric(metrics: TreeQSMMetrics) -> tuple[float, str]:
        if metrics.dbh_cyl_m is not None:
            return metrics.dbh_cyl_m, "DBHcyl"
        if metrics.dbh_qsm_m is not None:
            return metrics.dbh_qsm_m, "DBHqsm"
        raise TreeQSMBackendError("TreeQSM output missing DBHcyl and DBHqsm.")

    @classmethod
    def _build_matlab_script(
        cls,
        treeqsm_root: Path,
        input_mat_path: Path,
        output_mat_path: Path,
        sample_name: str,
    ) -> str:
        src_path = treeqsm_root / "src"
        lines = [
            "try",
            f"  addpath(genpath('{cls._matlab_string(src_path)}'));",
            f"  load('{cls._matlab_string(input_mat_path)}', 'P');",
            "  inputs = define_input(P,1,1,1);",
            f"  inputs.name = '{cls._matlab_string(sample_name)}';",
            "  inputs.tree = 1;",
            "  inputs.model = 1;",
            "  inputs.savemat = 0;",
            "  inputs.savetxt = 0;",
            "  inputs.plot = 0;",
            "  inputs.disp = 0;",
            "  inputs.Dist = 0;",
            "  inputs.Tria = 0;",
            "  inputs.OnlyTree = 1;",
            "  QSM = treeqsm(P, inputs);",
            f"  save('{cls._matlab_string(output_mat_path)}', 'QSM', '-v7');",
            "catch ME",
            "  disp('FORESTAGENT_DEBUG_BEGIN');",
            "  disp(ME.identifier);",
            "  disp(ME.message);",
            "  disp(getReport(ME, 'extended', 'hyperlinks', 'off'));",
            "  disp('FORESTAGENT_DEBUG_END');",
            "  rethrow(ME);",
            "end",
        ]
        return "\n".join(lines) + "\n"

    @staticmethod
    def _matlab_string(path_like: str | Path) -> str:
        return str(path_like).replace("\\", "/").replace("'", "''")

    @staticmethod
    def _extract_process_diagnostic(completed: subprocess.CompletedProcess[str]) -> str:
        combined = "\n".join(
            chunk.strip() for chunk in [completed.stdout, completed.stderr] if chunk and chunk.strip()
        )
        if not combined:
            return "MATLAB returned no diagnostic output."
        lines = [line.strip() for line in combined.splitlines() if line.strip()]
        if "FORESTAGENT_DEBUG_BEGIN" in lines and "FORESTAGENT_DEBUG_END" in lines:
            start = lines.index("FORESTAGENT_DEBUG_BEGIN") + 1
            end = lines.index("FORESTAGENT_DEBUG_END")
            debug_lines = [line for line in lines[start:end] if line]
            if debug_lines:
                return "\n".join(debug_lines[:12])

        filtered_lines = [line for line in lines if not line.startswith("ERROR: MATLAB error")]
        if filtered_lines:
            return "\n".join(filtered_lines[-8:])
        return lines[-1]

    def _build_failed_result(
        self,
        tool_name: str,
        point_cloud: PointCloudInput,
        message: str,
    ) -> ToolResult:
        return ToolResult(
            tool_name=tool_name,
            status="failed",
            value=None,
            unit=None,
            confidence=None,
            extra={"backend": "treeqsm", "input_format": point_cloud.format},
            message=message,
        )

    def _build_unsupported_result(
        self,
        tool_name: str,
        point_cloud: PointCloudInput,
        message: str,
    ) -> ToolResult:
        return ToolResult(
            tool_name=tool_name,
            status="failed",
            value=None,
            unit=None,
            confidence=None,
            extra={
                "backend": "treeqsm",
                "input_format": point_cloud.format,
                "unsupported": True,
            },
            message=message,
        )
