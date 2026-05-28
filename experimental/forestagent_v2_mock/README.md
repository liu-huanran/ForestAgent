# ForestAgent v2 Mock Skeleton

This directory archives an experimental ForestAgent v2 tool-calling skeleton.

It is intentionally outside the formal `forestagent/` package because it is not
the current mainline implementation.

## Status

- Experimental design reference only.
- Not used by Baseline V0.
- Not imported by the default CLI.
- Not part of the current paper mainline.
- Not connected to real q1/q2/q3 adapters.
- Not connected to Uni3D forward, checkpoints, or embedding caches.
- Not connected to any LLM planner or free-form reasoning.

## Why It Was Moved Here

The mock skeleton overlaps with mechanisms that already exist in the current
MVP:

- keyword/fixed intent routing
- fixed q1/q2/q3 tool calls
- structured `json_summary`
- template `report_text`
- bounded optional local verbalization

Keeping the mock skeleton under `forestagent/agent` made it look like a formal
runtime path. It is now archived here so the repository mainline remains clear.

## Current Mainline

The active ForestAgent mainline remains:

1. Baseline V0 q1/q2/q3 geometry tools
2. `json_summary` plus template `report_text`
3. Uni3D offline embedding experiments and audit/comparison tooling

## Optional Experimental Test Command

These tests are not part of the mainline test suite. To inspect the archived
mock skeleton later, run them from this directory:

```powershell
cd experimental\forestagent_v2_mock
$env:PYTHONPATH = "src"
python -m unittest discover tests
```

Do not treat passing experimental tests as evidence that v2 is integrated into
the ForestAgent runtime.
