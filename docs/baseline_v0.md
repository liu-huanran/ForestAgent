# Baseline V0

## 1. Definition

Baseline V0 is the current pre-feature, question-driven single-tree baseline that already runs end to end in this repository.

Its frozen goal is:

- accept one single-tree point cloud path plus `task_id` or a limited `question`
- run a fixed deterministic q1/q2/q3 process
- return a structured JSON payload and a report

Baseline V0 includes two already-existing report modes:

- default template report mode
- optional local Ollama verbalizer mode enabled by `--use-local-llm-report`

The important boundary is that Ollama is included only as a local post-processing verbalizer. It is not used for task routing, tool selection, metric estimation, or rule evaluation.

Baseline V0 still does **not** try to improve the algorithm, extend task coverage, or introduce agent-style / free-form LLM reasoning.

## 2. Frozen Scope

Baseline V0 covers only these four intents on the current question-driven main path:

- `q1_dbh`
- `q2_height`
- `q3_crown_width`
- `tree_report_q123`

The report intent is still q1/q2/q3-only. It does not include q4/q5/q6/q7 reasoning.

## 3. Actual Main Entry

The frozen main entry is:

```powershell
python -m forestagent.cli analyze-tree --point-cloud <path-to-tree.las> --task-id tree_report_q123
```

This route is frozen as the official Baseline V0 path because it is the current code path that really matches:

- question or task request
- fixed process
- final answer

## 4. Actual Execution Path

The current Baseline V0 execution path is:

1. `forestagent/cli.py`
2. `analyze-tree` subcommand
3. `forestagent/mvp/single_tree_analysis.py`
4. fixed intent resolution from `task_id` or a limited keyword-based `question`
5. `forestagent/backends/direct_geometry_backend.py`
6. `forestagent/tools/estimate_dbh.py`
7. `forestagent/tools/estimate_height.py`
8. `forestagent/tools/estimate_crown_width.py`
9. `forestagent/renderers/minimal_tree_report.py`
10. template report rendering
11. optional local Ollama verbalization when `--use-local-llm-report` is enabled
12. JSON response printed by CLI

Important boundary:

- `forestagent/pipelines/fixed_pipeline.py` exists in the repo, but it is **not** the current question-driven Baseline V0 main path.
- `forestagent/verbalizers/ollama_verbalizer.py` is now treated as part of Baseline V0, but only as an opt-in local verbalizer layer.

## 5. Required Files On The Main Path

The baseline core files are:

- `forestagent/cli.py`
- `forestagent/mvp/single_tree_analysis.py`
- `forestagent/backends/direct_geometry_backend.py`
- `forestagent/tools/estimate_dbh.py`
- `forestagent/tools/estimate_height.py`
- `forestagent/tools/estimate_crown_width.py`
- `forestagent/tools/_helpers.py`
- `forestagent/renderers/minimal_tree_report.py`
- `forestagent/verbalizers/ollama_verbalizer.py`
- `forestagent/schemas/io_models.py`
- `forestagent/schemas/tool_result.py`
- `configs/direct_geometry.yaml`

The package-level baseline tag is defined in:

- `forestagent/__init__.py`

## 6. Environment And Dependencies

The question-driven Baseline V0 path needs a Python environment that can import:

- `numpy`
- `pydantic`
- `laspy`

Notes:

- `openpyxl` is used by catalog / dataset utilities, but it is not required for the `analyze-tree` baseline path itself.
- The direct geometry baseline currently supports `las` / `laz` input only.
- If you want the Ollama verbalizer mode, you also need a local Ollama service and a local model available to that service.

## 7. Input Format

Baseline V0 expects one single-tree point cloud input:

- `--point-cloud <path>`

And one of the following request forms:

- explicit `--task-id q1_dbh|q2_height|q3_crown_width|tree_report_q123`
- limited `--question ...` keyword mapping

Current `question` routing is rule-based, not free-form reasoning:

- `dbh` / `胸径` -> `q1_dbh`
- `height` / `树高` / `高度` -> `q2_height`
- `crown width` / `冠幅` / `冠宽` -> `q3_crown_width`
- `report` / `summary` / `报告` / `总结` -> `tree_report_q123`

If a question is ambiguous or unsupported, the run fails explicitly.

Optional report verbalization flags on the same baseline path:

- `--use-local-llm-report`
- `--ollama-model`
- `--ollama-timeout-seconds`

## 8. Output Format

Baseline V0 prints one JSON object to stdout. The top-level structure is:

- `request`
- `input`
- `tool_results`
- `json_summary`
- `report_text`
- `status`
- `message`

Important shape notes:

- `tool_results` contains only the tools actually executed in this request.
- `json_summary` always uses the fixed keys `dbh_cm`, `height_m`, `crown_width_m`.
- metrics that were not executed stay `null`.
- failed runs return `status = "failed"` and keep the failure message explicit.
- this main-path output is **not** a `TaskResult`; it is the `SingleTreeAnalysisResponse` JSON defined by `forestagent/mvp/single_tree_analysis.py`.

When `--use-local-llm-report` is enabled:

- `report_text` still keeps the template report text
- `report_text_template` is added explicitly
- `report_text_llm` is added only when the local Ollama call succeeds
- if the local Ollama call fails, the run still returns `status = "success"` and falls back to the template report while appending the fallback reason to `message`

## 9. Recommended Baseline Commands

Run the official q1/q2/q3-only report:

```powershell
python -m forestagent.cli analyze-tree --point-cloud <path-to-tree.las> --task-id tree_report_q123
```

Run one scalar task explicitly:

```powershell
python -m forestagent.cli analyze-tree --point-cloud <path-to-tree.las> --task-id q1_dbh
python -m forestagent.cli analyze-tree --point-cloud <path-to-tree.las> --task-id q2_height
python -m forestagent.cli analyze-tree --point-cloud <path-to-tree.las> --task-id q3_crown_width
```

Run with question routing:

```powershell
python -m forestagent.cli analyze-tree --point-cloud <path-to-tree.las> --question "给我一个简短单木报告"
```

Use the frozen config explicitly:

```powershell
python -m forestagent.cli analyze-tree --point-cloud <path-to-tree.las> --task-id tree_report_q123 --config-path configs/direct_geometry.yaml
```

Run the same baseline path with local Ollama verbalization:

```powershell
python -m forestagent.cli analyze-tree --point-cloud <path-to-tree.las> --task-id tree_report_q123 --use-local-llm-report --ollama-model qwen3:1.7b
```

## 10. Minimal Validation Path

The minimal smoke test for Baseline V0 is:

```powershell
python -m unittest tests.test_baseline_v0_smoke
```

This smoke test:

- generates a temporary synthetic single-tree `.las`
- runs the official CLI entrypoint
- verifies that the end-to-end q1/q2/q3 report path still succeeds
- verifies that the Baseline V0 local Ollama verbalizer path is still wired

## 11. Minimal Example

Baseline V0 does not add a tracked binary sample point cloud file.

Instead, the minimal example is defined as:

- valid input: a synthetic single-tree `.las` generated at runtime by the smoke test
- successful run: the `analyze-tree` CLI command above
- representative output: `docs/examples/baseline_v0_success.json`

The example output is illustrative. Numeric values are tied to the synthetic tree geometry, while local absolute paths may vary across machines.

## 12. Out Of Scope

The following are explicitly out of scope for Baseline V0:

- q4 / q5 / q6 / q7 / q8 as baseline target scope
- Uni3D features
- any learned feature
- any LLM-based task routing, tool selection, metric estimation, or free-form reasoning
- agent planning logic
- backend replacement or architecture redesign
- TreeQSM integration as the frozen default path
- batch evaluation, diagnosis, benchmark, or long-tail analysis as the official runtime path

## 13. Known Limitations

- only `las` / `laz` single-tree point clouds are supported on the baseline path
- question routing is limited keyword mapping, not robust natural-language understanding
- the report is template-based and does not perform free-form reasoning
- q4/q5-related judgments are not part of this baseline path
- confidence is currently not populated by the direct geometry q1/q2/q3 path
- the local Ollama verbalizer is optional and requires a separately running local Ollama service
- even in Ollama mode, `report_text` is not replaced; the verbalized text appears in `report_text_llm` when available

## 14. Core vs Optional Code Boundary

Core Baseline V0 files:

- `forestagent/cli.py`
- `forestagent/mvp/single_tree_analysis.py`
- `forestagent/backends/direct_geometry_backend.py`
- `forestagent/tools/estimate_dbh.py`
- `forestagent/tools/estimate_height.py`
- `forestagent/tools/estimate_crown_width.py`
- `forestagent/tools/_helpers.py`
- `forestagent/renderers/minimal_tree_report.py`
- `forestagent/verbalizers/ollama_verbalizer.py`
- `forestagent/schemas/io_models.py`
- `forestagent/schemas/tool_result.py`
- `configs/direct_geometry.yaml`

Present in the repo but not part of the frozen Baseline V0 main path:

- `forestagent/pipelines/fixed_pipeline.py`
- `forestagent/sample_runner.py`
- `forestagent/data_catalog.py`
- `forestagent/evaluation/*`
- `forestagent/demo_single_tree_mvp.py`
- `forestagent/demo_build_mvp_bundle.py`
- `forestagent/backends/treeqsm_backend.py`
- `forestagent/backends/mock_backend.py`
- `forestagent/logic/task_rules.py`
- `forestagent/renderers/answer_renderer.py`
- `configs/task_registry.yaml`
- `configs/task_rules.yaml`

Those files remain in the repository, but they are not the frozen official runtime definition for Baseline V0.
