"""Tests for the minimal TreeQSM backend contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import subprocess

from scipy.io import savemat

from forestagent.backends.treeqsm_backend import (
    TreeQSMBackend,
    TreeQSMBackendError,
    TreeQSMDebugInfo,
    TreeQSMDiagnosticRun,
    TreeQSMMetrics,
    TreeQSMPreprocessStats,
    TreeQSMRunResult,
)
from forestagent.schemas import PointCloudInput


class StubTreeQSMBackend(TreeQSMBackend):
    """TreeQSM backend stub that bypasses external execution for tests."""

    def __init__(
        self,
        run_result: TreeQSMRunResult | None = None,
        error_message: str | None = None,
    ) -> None:
        super().__init__(treeqsm_root="D:/fake/treeqsm", matlab_executable="C:/fake/matlab.exe")
        self._stub_run_result = run_result
        self._error_message = error_message
        self.compute_calls = 0

    def _compute_treeqsm_run_result(self, point_cloud: PointCloudInput) -> TreeQSMRunResult:
        self.compute_calls += 1
        if self._error_message is not None:
            raise TreeQSMBackendError(self._error_message)
        if self._stub_run_result is None:
            raise AssertionError("StubTreeQSMBackend requires a run_result in this test.")
        return self._stub_run_result

    @staticmethod
    def _validate_point_cloud_input(point_cloud: PointCloudInput) -> None:
        return None


class DegradingTreeQSMBackend(TreeQSMBackend):
    """Backend stub for exercising the experimental skip-crown recovery path."""

    def __init__(
        self,
        *,
        allow_skip_crown_for_q1_q2: bool,
        raw_error_message: str,
        raw_diagnostic_run: TreeQSMDiagnosticRun,
        skip_crown_run: TreeQSMDiagnosticRun | None = None,
    ) -> None:
        super().__init__(
            treeqsm_root="D:/fake/treeqsm",
            matlab_executable="C:/fake/matlab.exe",
            allow_skip_crown_for_q1_q2=allow_skip_crown_for_q1_q2,
        )
        self._raw_error_message = raw_error_message
        self._raw_diagnostic_run = raw_diagnostic_run
        self._skip_crown_run = skip_crown_run
        self.diagnosis_calls: list[tuple[bool, bool]] = []

    def _compute_treeqsm_run_result(self, point_cloud: PointCloudInput) -> TreeQSMRunResult:
        raise TreeQSMBackendError(self._raw_error_message)

    def run_failure_diagnosis(
        self,
        point_cloud: PointCloudInput,
        *,
        apply_preprocess: bool = False,
        try_skip_crown: bool = False,
    ) -> TreeQSMDiagnosticRun:
        self.diagnosis_calls.append((apply_preprocess, try_skip_crown))
        if try_skip_crown:
            if self._skip_crown_run is None:
                raise AssertionError("skip_crown_run must be configured when skip-crown is attempted.")
            return self._skip_crown_run
        return self._raw_diagnostic_run

    @staticmethod
    def _validate_point_cloud_input(point_cloud: PointCloudInput) -> None:
        return None


class TreeQSMBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.point_cloud = PointCloudInput(path="D:/fake/sample_01.las", format="las")

    def test_estimate_dbh_prefers_dbhcyl_and_converts_to_cm(self) -> None:
        backend = StubTreeQSMBackend(
            run_result=TreeQSMRunResult(
                metrics=TreeQSMMetrics(dbh_cyl_m=0.31, dbh_qsm_m=0.29, tree_height_m=15.2),
                runtime_seconds=3.2,
            )
        )

        result = backend.estimate_dbh(self.point_cloud)

        self.assertEqual(result.status, "success")
        self.assertAlmostEqual(result.value, 31.0, places=6)
        self.assertEqual(result.unit, "cm")
        self.assertEqual(result.extra["source_metric"], "DBHcyl")

    def test_estimate_dbh_falls_back_to_dbhqsm(self) -> None:
        backend = StubTreeQSMBackend(
            run_result=TreeQSMRunResult(
                metrics=TreeQSMMetrics(dbh_cyl_m=None, dbh_qsm_m=0.284, tree_height_m=15.2),
                runtime_seconds=1.0,
            )
        )

        result = backend.estimate_dbh(self.point_cloud)

        self.assertEqual(result.status, "success")
        self.assertAlmostEqual(result.value, 28.4, places=6)
        self.assertEqual(result.extra["source_metric"], "DBHqsm")

    def test_estimate_height_returns_tree_height(self) -> None:
        backend = StubTreeQSMBackend(
            run_result=TreeQSMRunResult(
                metrics=TreeQSMMetrics(dbh_cyl_m=0.31, dbh_qsm_m=0.29, tree_height_m=18.6),
                runtime_seconds=1.8,
            )
        )

        result = backend.estimate_height(self.point_cloud)

        self.assertEqual(result.status, "success")
        self.assertAlmostEqual(result.value, 18.6, places=6)
        self.assertEqual(result.unit, "m")

    def test_unsupported_tools_return_explicit_failed_results(self) -> None:
        backend = StubTreeQSMBackend(
            run_result=TreeQSMRunResult(metrics=TreeQSMMetrics(), runtime_seconds=1.0)
        )

        crown = backend.estimate_crown_width(self.point_cloud)
        tilt = backend.estimate_tilt(self.point_cloud)
        quality = backend.assess_quality(self.point_cloud)

        for result in [crown, tilt, quality]:
            self.assertEqual(result.status, "failed")
            self.assertTrue(result.extra["unsupported"])
            self.assertIn("Unsupported", result.message)

    def test_backend_errors_become_failed_tool_results(self) -> None:
        backend = StubTreeQSMBackend(error_message="MATLAB execution failed.")

        result = backend.estimate_height(self.point_cloud)

        self.assertEqual(result.status, "failed")
        self.assertIn("MATLAB execution failed.", result.message)

    def test_treeqsm_run_result_is_cached_per_point_cloud(self) -> None:
        backend = StubTreeQSMBackend(
            run_result=TreeQSMRunResult(
                metrics=TreeQSMMetrics(dbh_cyl_m=0.31, dbh_qsm_m=0.29, tree_height_m=18.6),
                runtime_seconds=2.1,
            )
        )

        dbh = backend.estimate_dbh(self.point_cloud)
        height = backend.estimate_height(self.point_cloud)

        self.assertEqual(dbh.status, "success")
        self.assertEqual(height.status, "success")
        self.assertEqual(backend.compute_calls, 1)

    def test_load_metrics_from_output_mat_extracts_expected_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mat_path = Path(temp_dir) / "treeqsm_output.mat"
            savemat(
                mat_path,
                {
                    "QSM": {
                        "treedata": {
                            "DBHcyl": 0.317,
                            "DBHqsm": 0.313,
                            "TreeHeight": 25.68,
                        }
                    }
                },
            )

            metrics = TreeQSMBackend._load_metrics_from_output_mat(mat_path)

            self.assertAlmostEqual(metrics.dbh_cyl_m, 0.317, places=6)
            self.assertAlmostEqual(metrics.dbh_qsm_m, 0.313, places=6)
            self.assertAlmostEqual(metrics.tree_height_m, 25.68, places=6)

    def test_extract_process_diagnostic_prefers_debug_block(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["matlab", "-batch", "run('dummy')"],
            returncode=1,
            stdout=(
                "FORESTAGENT_DEBUG_BEGIN\n"
                "MATLAB:convhull:EmptyConvhull2DErrId\n"
                "convhull failed\n"
                "stack trace line\n"
                "FORESTAGENT_DEBUG_END\n"
            ),
            stderr="ERROR: MATLAB error Exit Status: 0x00000001",
        )

        diagnostic = TreeQSMBackend._extract_process_diagnostic(completed)

        self.assertIn("MATLAB:convhull:EmptyConvhull2DErrId", diagnostic)
        self.assertIn("convhull failed", diagnostic)
        self.assertNotIn("Exit Status", diagnostic)

    def test_skip_crown_is_disabled_by_default(self) -> None:
        raw_run = self._build_failed_diagnostic_run(
            identifier="MATLAB:convhull:EmptyConvhull2DErrId",
            stage="tree_data_layer_spread_before_convhull",
            include_metrics=True,
        )
        skip_run = self._build_success_diagnostic_run()
        backend = DegradingTreeQSMBackend(
            allow_skip_crown_for_q1_q2=False,
            raw_error_message=(
                "TreeQSM MATLAB run failed with exit code 1. "
                "MATLAB:convhull:EmptyConvhull2DErrId "
                "tree_data>crown_measures"
            ),
            raw_diagnostic_run=raw_run,
            skip_crown_run=skip_run,
        )

        result = backend.estimate_dbh(self.point_cloud)

        self.assertEqual(result.status, "failed")
        self.assertEqual(backend.diagnosis_calls, [])

    def test_skip_crown_can_recover_q1_and_q2_when_enabled(self) -> None:
        raw_run = self._build_failed_diagnostic_run(
            identifier="MATLAB:convhull:EmptyConvhull2DErrId",
            stage="tree_data_layer_spread_before_convhull",
            include_metrics=True,
        )
        skip_run = self._build_success_diagnostic_run()
        backend = DegradingTreeQSMBackend(
            allow_skip_crown_for_q1_q2=True,
            raw_error_message=(
                "TreeQSM MATLAB run failed with exit code 1. "
                "MATLAB:convhull:EmptyConvhull2DErrId "
                "tree_data>crown_measures"
            ),
            raw_diagnostic_run=raw_run,
            skip_crown_run=skip_run,
        )

        dbh = backend.estimate_dbh(self.point_cloud)
        height = backend.estimate_height(self.point_cloud)

        self.assertEqual(dbh.status, "success")
        self.assertEqual(height.status, "success")
        self.assertTrue(dbh.extra["degraded"])
        self.assertTrue(height.extra["degraded"])
        self.assertEqual(dbh.extra["raw_failure_stage"], "tree_data_layer_spread_before_convhull")
        self.assertEqual(height.extra["raw_failure_stage"], "tree_data_layer_spread_before_convhull")
        self.assertEqual(dbh.extra["used_metrics"], ["DBHcyl"])
        self.assertEqual(height.extra["used_metrics"], ["TreeHeight"])
        self.assertEqual(backend.diagnosis_calls, [(False, False), (False, True)])

    def test_skip_crown_does_not_change_q3_behavior(self) -> None:
        raw_run = self._build_failed_diagnostic_run(
            identifier="MATLAB:convhull:EmptyConvhull2DErrId",
            stage="tree_data_layer_spread_before_convhull",
            include_metrics=True,
        )
        skip_run = self._build_success_diagnostic_run()
        backend = DegradingTreeQSMBackend(
            allow_skip_crown_for_q1_q2=True,
            raw_error_message=(
                "TreeQSM MATLAB run failed with exit code 1. "
                "MATLAB:convhull:EmptyConvhull2DErrId "
                "tree_data>crown_measures"
            ),
            raw_diagnostic_run=raw_run,
            skip_crown_run=skip_run,
        )

        result = backend.estimate_crown_width(self.point_cloud)

        self.assertEqual(result.status, "failed")
        self.assertTrue(result.extra["unsupported"])
        self.assertNotIn("degraded", result.extra)

    def test_skip_crown_does_not_trigger_when_conditions_are_not_met(self) -> None:
        raw_run = self._build_failed_diagnostic_run(
            identifier="MATLAB:convhull:EmptyConvhull2DErrId",
            stage="tree_data_segmentation_before_convex_hull",
            include_metrics=True,
        )
        skip_run = self._build_success_diagnostic_run()
        backend = DegradingTreeQSMBackend(
            allow_skip_crown_for_q1_q2=True,
            raw_error_message=(
                "TreeQSM MATLAB run failed with exit code 1. "
                "MATLAB:convhull:EmptyConvhull2DErrId "
                "tree_data>crown_measures"
            ),
            raw_diagnostic_run=raw_run,
            skip_crown_run=skip_run,
        )

        result = backend.estimate_height(self.point_cloud)

        self.assertEqual(result.status, "failed")
        self.assertEqual(backend.diagnosis_calls, [(False, False)])

    @staticmethod
    def _build_failed_diagnostic_run(
        *,
        identifier: str,
        stage: str,
        include_metrics: bool,
    ) -> TreeQSMDiagnosticRun:
        metrics = {
            "dbh_cyl_m": 0.31 if include_metrics else None,
            "dbh_qsm_m": 0.29 if include_metrics else None,
            "tree_height_m": 15.2 if include_metrics else None,
        }
        return TreeQSMDiagnosticRun(
            label="raw",
            status="failed",
            preprocess_stats=TreeQSMPreprocessStats(
                input_point_count=100,
                preprocessed_point_count=100,
                removed_nonfinite_count=0,
                removed_duplicate_count=0,
                removed_outlier_count=0,
                has_nan_or_inf_before=False,
                has_nan_or_inf_after=False,
            ),
            debug_info=TreeQSMDebugInfo(
                crown_point_count_before_convhull=25,
                unique_projected_2d_point_count=4,
                duplicate_projected_2d_point_count=21,
                has_nan_or_inf_in_crown_points=False,
                projected_rank=1,
                is_projected_collinear=True,
                projected_x_span=0.5,
                projected_y_span=0.0,
                dbh_cyl_m=metrics["dbh_cyl_m"],
                dbh_qsm_m=metrics["dbh_qsm_m"],
                tree_height_m=metrics["tree_height_m"],
                last_successful_stage=stage,
                convhull_identifier=identifier,
                convhull_message="convhull failed",
            ),
            metrics=None,
            failure_message="raw TreeQSM failure",
            runtime_seconds=12.0,
        )

    @staticmethod
    def _build_success_diagnostic_run() -> TreeQSMDiagnosticRun:
        return TreeQSMDiagnosticRun(
            label="raw_skip_crown",
            status="success",
            preprocess_stats=TreeQSMPreprocessStats(
                input_point_count=100,
                preprocessed_point_count=100,
                removed_nonfinite_count=0,
                removed_duplicate_count=0,
                removed_outlier_count=0,
                has_nan_or_inf_before=False,
                has_nan_or_inf_after=False,
            ),
            debug_info=TreeQSMDebugInfo(
                last_successful_stage="tree_data_skip_crown_recovery",
                skip_crown_applied=True,
                skip_crown_reason="convhull failed",
            ),
            metrics=TreeQSMMetrics(dbh_cyl_m=0.31, dbh_qsm_m=0.29, tree_height_m=15.2),
            failure_message=None,
            runtime_seconds=5.0,
        )


if __name__ == "__main__":
    unittest.main()
