# AGENTS.md

This file contains stable instructions and collaboration rules for Codex and future agents.
It should change infrequently.

## Important instruction for future agents

Do not treat future plans as implemented facts.
Always distinguish between:
- confirmed current behavior
- experimental code
- intended future direction

If this file conflicts with the current verified code path, report the conflict explicitly and prefer the verified code path.

---

## 1. Project Identity

**Project name:** ForestAgent

**Project type:** Research prototype for single-tree point cloud understanding

**Core motivation:**
Large models can hallucinate when interpreting 3D point clouds, especially on fine-grained physical details.
This project aims to reduce hallucination by grounding system outputs in **geometry-based tools** and later, in a controlled way, augmenting the system with **learned 3D representations**.

The project is **not** intended to let an LLM freely infer physical measurements from raw point clouds.

---

## 2. High-Level Project Goal

Build a controlled point-cloud understanding system for **single-tree analysis**.

The long-term intended structure is:

1. **Geometry / physical tools** provide measurable, grounded signals
2. **Learned 3D features** may later provide semantic representation
3. **Structured intermediate outputs** remain the source of truth
4. Any natural-language answer layer must be controlled and must not invent physical facts

The system should remain interpretable, modular, and easy to audit.

---

## 3. Stable Project Constraints

### 3.1 Grounding constraint
LLM-like components must not invent:
- physical measurements
- factual geometry
- unsupported judgments

### 3.2 Two-layer output philosophy
The system should preserve a distinction between:
1. **structured intermediate outputs** (source of truth)
2. **human-readable report / answer text** (derived layer)

The structured layer is primary.

### 3.3 Minimal-scope principle
At each stage, prefer:
- a smaller closed loop
- fewer moving parts
- explicit boundaries
- clear verification

Avoid:
- doing multiple uncertain expansions at once
- combining feature extraction, new tasks, and LLM generation in one step

### 3.4 Stability over ambition
When there is a choice between:
- preserving a verified working path
- or making it look more sophisticated

prefer preserving the verified working path.

---

## 4. Current Stable Task Scope

The currently stable core task scope is centered on **single-tree tasks**, especially:

- **q1**: DBH
- **q2**: Height
- **q3**: Crown Width

Unless explicitly requested, do not silently expand the stable supported scope beyond this.

---

## 5. Explicit Non-Goals Right Now

The following are **not** default goals:

- do not turn the system into an open-ended agent
- do not add free LLM reasoning
- do not let a model invent physical values
- do not use learned features to directly replace q1/q2/q3 geometry measurements
- do not broadly redesign the architecture
- do not expand scope just to make the demo look bigger
- do not prioritize elegance over preserving the current working system

In particular, **q4 / q5 are not the default main implementation priority**.
They may be discussed at the level of:
- task definition
- feasibility
- data support
- evaluation possibility

But they should not become the main engineering focus before the learned representation branch is clarified.

---

## 6. Rules for Baseline V0 Tasks

If the task concerns **Baseline V0**:

- preserve current behavior
- prefer audit over refactor
- identify the true entrypoint from code
- do not add new abilities
- do not introduce learned 3D features
- do not introduce LLM logic
- do not expand to q4/q5
- treat the currently working fixed path as the baseline, even if it looks simple

When uncertain, choose the smaller, safer change.

---

## 7. Rules for Uni3D / Learned Feature Tasks

If the task concerns **Uni3D** or learned 3D representation:

- first do repository/code reconnaissance
- identify actual builder / checkpoint / forward path
- do not directly integrate into the main QA pipeline first
- first target a standalone extractor
- first support only **global embedding**
- defer token-level, fine-grained, or LLM-coupled usage
- do not use Uni3D to replace geometry measurement logic for q1/q2/q3

The preferred early-stage target is:

`point cloud -> preprocess -> Uni3D -> global embedding -> cache`

---

## 8. Sanity-Check Policy for Learned Features

Before broader integration, learned features should first pass limited sanity checks:

1. **repeatability**
   - same input should give stable embedding

2. **light perturbation robustness**
   - minor sampling/noise changes should not completely destroy similarity

3. **nearest-neighbor plausibility**
   - similar embeddings should look reasonably similar at a human-inspection level

Do not jump directly from “extractor exists” to “LLM should use it”.

---

## 9. Collaboration Rules for Codex / Future Agents

### 9.1 General rule
Do not assume future plans are already implemented.
Always distinguish between:
- current verified baseline
- experimental branches
- future design ideas

### 9.2 Required working style
Codex or future agents should:

1. audit first
2. summarize findings before major implementation
3. keep changes small and local
4. explicitly label uncertainty
5. avoid speculative redesign

### 9.3 Memory maintenance rule
At the end of any task that changes the verified project state, update:
- `PROJECT_MEMORY.md` → Confirmed Current State
- `PROJECT_MEMORY.md` → Known Uncertainties
- `PROJECT_MEMORY.md` → Immediate Next Actions

Do not rewrite the whole file.
Do not change stable rules in `AGENTS.md` unless explicitly asked.
If the code and memory file conflict, report the conflict and prefer the verified code path.

---

## 10. Preferred Human-AI Collaboration Style

The preferred assistance style for this project is:

- plain and concrete explanations
- minimal but sufficient scope
- no unnecessary abstraction
- explicit statement of assumptions
- explicit statement of uncertainty
- reviewer-like criticism when needed

After each meaningful experiment or implementation stage, it is preferred to produce:

1. **experiment record**
   - what was changed
   - what was run
   - what happened

2. **understanding record**
   - what was learned
   - what became clearer
   - what remains uncertain
   - what should happen next

---

## 11. Practical Decision Rules

### Rule 1
If the choice is between:
- smaller, verifiable progress
- larger but blurrier progress

choose smaller, verifiable progress.

### Rule 2
If the choice is between:
- preserving baseline behavior
- making the code look more advanced

preserve baseline behavior.

### Rule 3
If the choice is between:
- introducing learned features in a controlled way
- expanding more fixed tasks

at the current stage, prioritize learned representation setup, not task sprawl.

### Rule 4
If the choice is between:
- reading everything first
- doing minimal reading plus controlled implementation

choose minimal reading plus controlled implementation.
