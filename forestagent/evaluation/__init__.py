"""Evaluation utilities for scalar MVP tasks."""

from .batch_evaluator import (
    BatchEvaluationResult,
    EvaluationMetrics,
    EvaluationRecord,
    evaluate_scalar_tasks,
)
from .exporters import export_evaluation_csv, export_evaluation_xlsx
from .direct_geometry_benchmark import (
    DirectGeometryBenchmarkResult,
    DirectGeometryFailureReasonStat,
    benchmark_direct_geometry_baseline,
)
from .direct_geometry_dbh_diagnosis import (
    DBHDiagnosisSummary,
    DBHSanityRecord,
    DirectGeometryDBHComparisonResult,
    DirectGeometryDBHDiagnosisResult,
    compare_direct_geometry_dbh_configs,
    run_direct_geometry_dbh_diagnosis,
)
from .direct_geometry_height_diagnosis import (
    DirectGeometryHeightFreezeValidationResult,
    HeightDiagnosisSummary,
    HeightFailureReasonStat,
    HeightSanityRecord,
    HeightSuspicionStat,
    run_direct_geometry_height_freeze_validation,
)
from .direct_geometry_crown_width_diagnosis import (
    CrownWidthDiagnosisSummary,
    CrownWidthFailureReasonStat,
    CrownWidthSanityRecord,
    CrownWidthSuspicionStat,
    DirectGeometryCrownWidthFreezeValidationResult,
    run_direct_geometry_crown_width_freeze_validation,
)
from .treeqsm_benchmark import (
    TreeQSMBenchmarkResult,
    TreeQSMFailureReasonStat,
    benchmark_treeqsm_baseline,
)

__all__ = [
    "BatchEvaluationResult",
    "EvaluationMetrics",
    "EvaluationRecord",
    "evaluate_scalar_tasks",
    "export_evaluation_csv",
    "export_evaluation_xlsx",
    "DirectGeometryBenchmarkResult",
    "DirectGeometryFailureReasonStat",
    "benchmark_direct_geometry_baseline",
    "DBHDiagnosisSummary",
    "DBHSanityRecord",
    "DirectGeometryDBHComparisonResult",
    "DirectGeometryDBHDiagnosisResult",
    "compare_direct_geometry_dbh_configs",
    "run_direct_geometry_dbh_diagnosis",
    "DirectGeometryHeightFreezeValidationResult",
    "HeightDiagnosisSummary",
    "HeightFailureReasonStat",
    "HeightSanityRecord",
    "HeightSuspicionStat",
    "run_direct_geometry_height_freeze_validation",
    "CrownWidthDiagnosisSummary",
    "CrownWidthFailureReasonStat",
    "CrownWidthSanityRecord",
    "CrownWidthSuspicionStat",
    "DirectGeometryCrownWidthFreezeValidationResult",
    "run_direct_geometry_crown_width_freeze_validation",
    "TreeQSMBenchmarkResult",
    "TreeQSMFailureReasonStat",
    "benchmark_treeqsm_baseline",
]
