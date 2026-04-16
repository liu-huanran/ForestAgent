"""Focused tests for TreeQSM failure-diagnosis helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from forestagent.backends.treeqsm_backend import TreeQSMBackend


class TreeQSMFailureDiagnosisTests(unittest.TestCase):
    def test_preprocess_points_removes_nonfinite_duplicates_and_outliers(self) -> None:
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [np.nan, 0.0, 1.0],
                [2.0, 2.0, 2.0],
                [1000.0, 1000.0, 1000.0],
            ],
            dtype=np.float64,
        )

        filtered_points, stats = TreeQSMBackend._preprocess_points_for_diagnosis(points)

        self.assertEqual(stats.input_point_count, 6)
        self.assertEqual(stats.removed_nonfinite_count, 1)
        self.assertEqual(stats.removed_duplicate_count, 1)
        self.assertEqual(stats.removed_outlier_count, 1)
        self.assertEqual(stats.preprocessed_point_count, 3)
        self.assertFalse(stats.has_nan_or_inf_after)
        self.assertEqual(filtered_points.shape, (3, 3))

    def test_load_debug_info_from_mat_reads_crown_and_skip_crown_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            debug_mat_path = Path(temp_dir) / "debug_info.mat"
            savemat(
                debug_mat_path,
                {
                    "debug_info": {
                        "crown_point_count_before_convhull": 18,
                        "unique_projected_2d_point_count": 2,
                        "duplicate_projected_2d_point_count": 6,
                        "has_nan_or_inf_in_crown_points": 0,
                        "projected_rank": 1,
                        "is_projected_collinear": 1,
                        "projected_x_span": 0.23,
                        "projected_y_span": 0.0,
                        "dbh_cyl": 0.31,
                        "dbh_qsm": 0.29,
                        "tree_height": 15.8,
                        "last_successful_stage": "tree_data_before_crown_convhull",
                        "convhull_identifier": "MATLAB:convhull:EmptyConvhull2DErrId",
                        "convhull_message": "convhull failed",
                    },
                    "skip_crown_info": {
                        "applied": 1,
                        "reason": "skip crown fallback",
                        "last_successful_stage": "tree_data_skip_crown_recovery",
                    },
                },
                long_field_names=True,
            )

            debug_info = TreeQSMBackend._load_debug_info_from_mat(debug_mat_path)

            self.assertIsNotNone(debug_info)
            assert debug_info is not None
            self.assertEqual(debug_info.crown_point_count_before_convhull, 18)
            self.assertEqual(debug_info.unique_projected_2d_point_count, 2)
            self.assertTrue(debug_info.is_projected_collinear)
            self.assertAlmostEqual(debug_info.dbh_cyl_m, 0.31, places=6)
            self.assertAlmostEqual(debug_info.tree_height_m, 15.8, places=6)
            self.assertEqual(debug_info.last_successful_stage, "tree_data_skip_crown_recovery")
            self.assertTrue(debug_info.skip_crown_applied)
            self.assertEqual(debug_info.skip_crown_reason, "skip crown fallback")

    def test_debug_matlab_script_puts_override_ahead_of_src(self) -> None:
        script = TreeQSMBackend._build_debug_matlab_script(
            treeqsm_root=Path("D:/Programming/LLM/TreeQSM"),
            override_dir=Path("D:/temp/override"),
            input_mat_path=Path("D:/temp/input.mat"),
            output_mat_path=Path("D:/temp/output.mat"),
            sample_name="sample_01",
        )

        src_line = "  addpath(genpath('D:/Programming/LLM/TreeQSM/src'));"
        override_line = "  addpath(genpath('D:/temp/override'));"
        self.assertIn(src_line, script)
        self.assertIn(override_line, script)
        self.assertLess(script.index(src_line), script.index(override_line))

    def test_instrumented_tree_data_source_includes_layer_and_crown_debug_hooks(self) -> None:
        original_source = """  I = P(:,3) >= bot+(j-1)*Hei/m & P(:,3) < bot+j*Hei/m;
  X = unique(P(I,:),'rows');
  if size(X,1) > 5
    [K,A] = convhull(X(:,1),X(:,2));
  end

%% Crown measures,Vertical profile and spreads
[treedata,spreads] = crown_measures(treedata,cylinder,branch);

X = unique(P(:,1:2),'rows');
[K,A] = convhull(X(:,1),X(:,2));
"""
        instrumented = TreeQSMBackend._build_instrumented_tree_data_source(
            original_source=original_source,
            try_skip_crown=True,
        )

        self.assertIn("tree_data_layer_spread_before_convhull", instrumented)
        self.assertIn("tree_data_before_crown_convhull", instrumented)
        self.assertIn("tree_data_skip_crown_recovery", instrumented)


if __name__ == "__main__":
    unittest.main()
