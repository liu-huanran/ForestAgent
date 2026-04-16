Codex should explain before implementing when the requested change touches pipeline, task rules, schemas, or backend boundaries.

# AGENTS.md

## Project Overview
ForestAgent is a research prototype for **tree-level point cloud interpretation with geometry-enhanced reasoning**.

Current phase: **MVP fixed pipeline**
Out of scope for the current phase:
- real LLM reasoning
- agent-style dynamic planning
- training / finetuning
- direct third-party backend exposure to the LLM

The current goal is to build a stable pipeline:

`task registry -> tools -> task rules -> fixed pipeline -> template renderer`

---

## Current Architecture Boundaries

### Schemas
- `ToolResult` is the only valid tool output contract.
- `TaskResult` is the only valid task output contract.
- Do not invent ad-hoc dict formats if a schema already exists.

### Tools
- Tools only do:
  1. call a backend
  2. normalize backend output into `ToolResult`
- Tools must **not** contain task-level reasoning or task-specific thresholds.

### Task rules
- Q4 / Q5 / Q6 / Q7 decision logic must live in `forestagent/logic/task_rules.py`
- Thresholds and label texts must come from `configs/task_rules.yaml`
- Do not hardcode research thresholds in pipeline or renderer

### Pipeline
- `fixed_pipeline.py` only does fixed orchestration
- It may:
  - read task registry
  - call required tools in fixed order
  - collect results
  - call task rules when required
  - construct `TaskResult`
- It must **not**:
  - implement task-specific thresholds directly
  - call real LLM APIs
  - expose third-party backend internals

### Renderer
- Renderer is currently **template-based only**
- It only converts structured results into human-readable text
- It must not estimate physical values or perform free-form reasoning

### Backends
- Third-party algorithms such as TreeQSM are treated as **backends**
- LLM-facing tool names must remain project-local names such as:
  - `estimate_dbh`
  - `estimate_height`
  - `estimate_crown_width`
  - `estimate_tilt`
  - `assess_quality`
- Do not expose raw TreeQSM commands, raw files, or backend-specific jargon to higher layers unless explicitly required

---

## Frozen MVP Tasks

Current frozen task set:
- Q1: DBH estimation
- Q2: Height estimation
- Q3: Crown width estimation
- Q4: Tilt state judgment
- Q5: Point cloud quality judgment
- Q6: Form judgment
- Q7: Preliminary fall-risk judgment
- Q8: Brief tree report

Use `task_id` as the stable key.
Use `task_name` only as display text.

---

## Failure Handling Rules
Failure propagation must be explicit and consistent.

If any required tool fails for a task:
- task `status = "failed"`
- task `result = null`
- successful tool outputs go into `extra.partial_results`
- failure reasons must be summarized in `message`

Do not silently swallow tool failures.

---

## Data and Evaluation Rules
- Prefer structured outputs over free text
- Preserve measured references when available
- Do not mix prediction fields with measured-reference fields
- Keep ground and air modality explicit

For evaluation:
- Q1 / Q2 / Q3 are the primary first-stage benchmark tasks
- Do not redesign evaluation output formats without strong reason

---

## Coding Rules
- Keep functions small and single-purpose
- Add type hints
- Prefer explicit names over clever abstractions
- Do not introduce unnecessary frameworks
- Do not expand scope beyond the requested phase
- Avoid hidden side effects
- Prefer configuration-driven behavior where thresholds or mappings may change later

---

## Testing Rules
When implementing a new layer, add or update focused tests.

Minimum expectations:
- schema contract tests
- tool contract tests
- task rule tests
- fixed pipeline tests
- failure propagation tests

Do not mark work as complete without tests or a clear explanation of why tests are deferred.

---

## When Making Changes
Before coding, first explain:
1. which files will be changed
2. what each file is responsible for
3. what invariants must remain true
4. likely failure points
5. what tests will be added or updated

After coding, summarize:
1. what changed
2. why it changed
3. what was tested
4. what remains for the next step

---

## Current Priorities
Prefer this implementation order unless the user explicitly changes it:
1. schemas
2. configs
3. mock backend
4. tools
5. task rules
6. fixed pipeline
7. template renderer
8. evaluation layer
9. real backend integration
10. advanced planning / agent features

---

## Do Not Do Yet
Unless explicitly requested, do not:
- connect a real LLM SDK
- add agent planning logic
- add training code
- tightly couple pipeline to TreeQSM
- move task rules into tools or pipeline
- replace mock behavior with speculative heuristics without clear note

---

## Default Review Mindset
This repository is a **research prototype**, not a general product codebase.

Prioritize:
- clarity
- explicit contracts
- reproducibility
- easy inspection
- easy replacement of backends
- easy benchmarking