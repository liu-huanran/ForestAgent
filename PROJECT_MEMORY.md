# PROJECT MEMORY

This file is the living project memory for this repository.
It records the current verified project state, confirmed decisions, open uncertainties, and immediate next actions.

When conflicts arise between this file and the current codebase, treat the current verified working code as ground truth, report the conflict explicitly, and then update this file.

---

## 1. Confirmed Current State

### 1.0 Latest verified project status

As of the latest verified project state:

- **q1 / q2 / q3 baseline v1.1 is frozen**
- **single-tree MVP v1 is connected to a local Ollama verbalizer**
- the next development stage moves to a **Uni3D extractor / feature branch**

The local Ollama verbalizer is a controlled wording layer.
It must not be treated as the source of physical measurements or free-form reasoning.

The next Uni3D branch should start as an independent feature-extraction branch, not as a direct integration into the main QA path.

### 1.1 Current confirmed baseline state
The current system has already run through the full single-tree MVP process:

`question -> fixed process -> answer`

The q1 / q2 / q3 baseline v1.1 is now frozen as the current geometry-grounded reference path.

- the current baseline path is **fixed / deterministic**
- it does **not** currently rely on learned 3D features
- physical measurements still come from structured geometry/tool outputs
- the local Ollama verbalizer is only a verbalization layer over structured results

This is extremely important:
**Baseline v1.1 is the frozen q1/q2/q3 reference path, even though single-tree MVP v1 now has a local verbalizer.**
Do not silently reinterpret the verbalizer as a reasoning or measurement engine.

### 1.2 Current stable problem scope
The currently stable scope is centered on **single-tree tasks**, especially:

- **q1**: DBH
- **q2**: Height
- **q3**: Crown Width

These are currently the main baseline-supported tasks.

### 1.3 Current baseline nature
The current baseline should be understood as:

- a working end-to-end system
- deterministic routing / fixed logic
- geometry-centered
- structured enough to be frozen as a baseline
- suitable for later comparison against feature-augmented or LLM-augmented variants

### 1.4 Important correction to older memory
Older memory treated Ollama-style verbalization as only a future idea.
The verified current state is now:

- single-tree MVP v1 has a local Ollama verbalizer connected
- q1 / q2 / q3 baseline v1.1 remains frozen and geometry-grounded
- the verbalizer must stay downstream of structured results

Baseline v1.1 should currently be treated as:
- fixed for q1 / q2 / q3
- deterministic for measurement logic
- pre-Uni3D / pre-learned-feature
- paired with a controlled local verbalization layer in single-tree MVP v1

---

## 2. What Baseline v1.1 Means

Baseline v1.1 is the current frozen q1 / q2 / q3 reference system.

Its purpose is:

- to preserve the currently working path
- to provide a reproducible comparison point
- to prevent future changes from blurring what the original system actually did

Baseline v1.1 should answer these questions clearly:

1. What is the actual entrypoint?
2. What files are on the real working path?
3. What inputs does it take?
4. What outputs does it produce?
5. What tasks does it currently support?
6. What is explicitly **out of scope**?

---

## 3. Current Main Development Direction

### 3.1 Immediate main direction
The current main direction is:

1. **Keep q1 / q2 / q3 baseline v1.1 frozen**
2. Keep **single-tree MVP v1 with local Ollama verbalizer** as the current MVP state
3. Move next development to **Stage 2: Uni3D as an independent feature extractor**

### 3.2 Why Uni3D is next
The project currently has a functioning fixed pipeline, but it still lacks a real **learned semantic representation layer**.

The next important step is **not** to immediately expand more tasks or add free-form LLM reasoning.

The next important step is to introduce a learned 3D representation in a controlled way:

`point cloud -> preprocess -> Uni3D -> global embedding -> cache -> downstream use`

### 3.3 What Uni3D should mean at this stage
At this stage, Uni3D should be treated as:

- a **frozen representation module**
- an **independent feature extractor**
- not part of free-form generation
- not a replacement for geometry tools
- not a justification to change q1/q2/q3 measurement logic

The target is to first obtain a stable, reusable global embedding pipeline.

### 3.4 Current paper / research positioning

The project should not be framed merely as a system prototype.

The intended research framing is:

**a geometry-grounded and Uni3D-representation-assisted framework for reliable single-tree point cloud question answering.**

This means future writing should emphasize:

- geometry-based tools as the grounded source of physical facts
- Uni3D embeddings as a learned representation layer that assists retrieval, diagnosis, or controlled downstream analysis
- reliability and hallucination reduction as the central motivation
- single-tree point cloud question answering as the target task setting
- a controlled framework rather than an open-ended agent or unconstrained LLM system

---

## 4. Stage 2 Plan: Uni3D

### 4.1 Correct interpretation of Stage 2
Stage 2 is **not** “learn all LLM knowledge first”.

Stage 2 is:

**turn Uni3D into a frozen, independent, cached feature extractor**

### 4.2 Minimal technical target
The minimal target is:

- input: single-tree point cloud
- preprocessing: fixed and documented
- model: Uni3D loaded in eval mode
- output: global embedding
- storage: cacheable feature artifact
- usage: independent of the main question-answer pipeline

### 4.3 What must be clarified before full implementation
Before implementing the extractor, the following must be confirmed from the official Uni3D repo:

1. where the model is built
2. how checkpoints are loaded
3. what the real inference path is
4. what input preprocessing is expected
5. which output tensor is the best candidate for **global embedding**
6. whether the embedding is normalized or needs post-processing

### 4.4 Recommended order
The current recommended order is:

1. minimal reading of Uni3D paper + repo
2. technical reconnaissance of official implementation
3. confirm builder / checkpoint / forward / output
4. implement independent extractor
5. run sanity checks
6. only then decide downstream use

### 4.5 Verified Uni3D extractor sanity check

As of 2026-04-26, the Uni3D extractor passed a real GPU forward sanity check on the lab server.

Verified details:

- conda env: `fa-u`
- checkpoint: `BAAI/Uni3D modelzoo/uni3d-b/model.pt`
- `models.uni3d` import: ok
- `pointnet2_ops` import: ok
- ForestAgent import: ok
- `mock=false`
- `model_builder=create_uni3d`
- input shape: `(10000, 3)`
- feature shape: `[1, 10000, 6]`
- embedding shape: `[1, 1024]`
- `embedding_l2_normalized=true`
- repeatability passed: `raw_max_abs_diff=0`, `l2_max_abs_diff=0`

Boundary:

- This verifies real Uni3D forward execution only.
- It does not yet validate whether Uni3D features improve q1 / q2 / q3 measurement or report quality.
- The current run used RGB `fallback_constant`, not real RGB.

### 4.6 Uni3D small-batch sanity tooling

As of 2026-04-26, the repository now includes offline tooling for the next Uni3D extractor sanity step:

- `scripts/batch_extract_uni3d_embeddings.py`
- `scripts/analyze_uni3d_embedding_similarity.py`
- `scripts/convert_las_to_uni3d_npy.py`
- `docs/uni3d_extractor_sanity_check.md`
- `tests/test_uni3d_batch_scripts.py`
- `tests/test_convert_las_to_uni3d_npy.py`

Confirmed locally:

- script CLI/help paths are available
- `.npy` input loading for `[N, 3]` xyz and `[N, 6]` xyzrgb is covered by tests
- invalid input shape rejection is covered by tests
- similarity analysis over fake `.npz` embeddings is covered by tests
- LAS-to-NPY helper logic for deterministic sampling, RGB normalization, `[N, 3]` / `[N, 6]` shape construction, and manifest writing is covered by tests
- no local test uses mock results as real Uni3D output
- no local test requires checkpoint, GPU, CUDA, `pointnet2_ops`, or real Uni3D forward

Boundary:

- real LAS reading with `laspy` over server `.las` files is not yet verified locally
- real small-batch Uni3D forward over 5-10 trees is not yet verified
- embedding cosine similarity over real Uni3D outputs is not yet verified
- this tooling remains independent of the main QA system and q1 / q2 / q3 geometry baseline

### 4.7 Uni3D full TLS processing readiness

As of 2026-04-26, the Uni3D offline tooling has been strengthened for safe full TLS runs:

- LAS conversion supports recursive scanning, full-run `--limit 0`, `--skip-existing` / `--resume`, `manifest.jsonl`, and `summary.json`
- batch embedding extraction supports recursive scanning, full-run `--limit 0`, `--skip-existing` / `--resume`, per-sample progress, and failure continuation
- similarity analysis supports `--max-samples`, `--seed`, `--no-full-matrix`, and `--save-full-matrix`
- `docs/uni3d_full_tls_embedding_run.md` records the recommended full TLS server workflow

User-reported server result before this readiness step:

- 10-tree real GPU sanity check passed
- `success_count=10`
- `failure_count=0`
- `embedding_dim=1024`
- `invalid_count=0`
- pairwise cosine similarity min / mean / max = `0.6335 / 0.8661 / 0.9688`

Boundary:

- this is still engineering readiness, not a formal experiment conclusion
- full TLS conversion and full TLS Uni3D forward still need to be run on the server
- the main QA system, agent tools, and q1 / q2 / q3 geometry baseline remain unchanged

### 4.8 Uni3D full TLS first embedding analysis

As of 2026-04-26, the full TLS Uni3D embedding set has been analyzed offline in a first-pass data health check.

User-reported full TLS similarity result:

- `loaded_embeddings=430`
- `analyzed_embeddings=430`
- `invalid_embeddings=0`
- `embedding_dim=1024`
- pairwise cosine similarity min / mean / max = `0.0939965 / 0.5662398 / 0.9999282`

Local analysis artifacts:

- `outputs/all_tls_001_embedding_analysis_round1/`
- `scripts/analyze_uni3d_embedding_dataset.py`
- `scripts/query_uni3d_embedding_neighbors.py`

Confirmed locally:

- saved embedding format is `.npz` with `embedding_raw`, `embedding_l2`, `tree_id`, `source_path`, and `metadata`
- `data/parameters.xlsx` covers all 430 embedding IDs
- first-pass PCA, near-duplicate, outlier, nearest-neighbor, and lightweight label-probe outputs were generated

Boundary:

- this is data/feature health analysis only
- it does not integrate Uni3D into the main QA system
- it does not modify q1 / q2 / q3 baseline behavior
- lightweight probes are not formal experiment results

### 4.9 Uni3D quantitative review table

As of 2026-05-04, the repository includes an offline quantitative review script for existing Uni3D embedding artifacts:

- `scripts/review_uni3d_embedding_quantitative.py`
- `tests/test_uni3d_embedding_review.py`

The script reads existing extraction summaries, similarity summaries, nearest-neighbor outputs, near-duplicate candidates, outlier candidates, and converted `.npy` inputs when available.

Generated local review artifacts:

- `outputs/all_tls_001_embedding_quant_review_round1/review_table.csv`
- `outputs/all_tls_001_embedding_quant_review_round1/review_table.json`
- `outputs/all_tls_001_embedding_quant_review_round1/review_summary.json`

Boundary:

- the script does not run Uni3D forward
- the script does not modify existing embeddings
- the script does not integrate with the main QA system
- local review output has null `xyz/rgb` input statistics because the full server `.npy` input directory is not present in the local workspace
- rerun on the server with the actual `uni3d_all_inputs_rgb` directory to fill raw input statistics

---

## 5. Known Uncertainties

The following are currently not fully settled:

1. exact q4/q5 definition and feasibility
2. whether Uni3D features improve q1/q2/q3 measurement or report quality
3. exact preprocessing details needed to match official inference behavior
4. which downstream task should first consume Uni3D features after extraction
5. how the local Ollama verbalizer should be documented and bounded relative to structured outputs
6. whether real Uni3D embeddings over 5-10 single-tree samples show useful separation in pairwise cosine similarity
7. whether LAS-derived `.npy` inputs preserve the desired coordinate/RGB conventions for Uni3D sanity checks
8. full TLS conversion success/failure rate and the distribution of point-count failures
9. full TLS Uni3D embedding extraction success/failure rate and similarity distribution
10. whether near-duplicate pairs reflect true duplicate-like trees, acquisition overlap, or model over-smoothing
11. whether `rgb_source=fallback_constant` groups should be analyzed separately from real RGB groups
12. whether quantitative review input statistics confirm that near-duplicates/outliers are caused by raw point-cloud statistics, RGB availability, site effects, or embedding-space behavior

These uncertainties should not be hidden.
They should be tracked and resolved one by one.

---

## 6. Immediate Next Actions

The current immediate next actions are:

1. preserve frozen q1 / q2 / q3 baseline v1.1
2. preserve single-tree MVP v1 local Ollama verbalizer boundaries
3. create a Uni3D extractor / feature branch from `main`
4. inspect Uni3D official repo
5. identify:
   - model builder
   - checkpoint loading
   - inference path
   - output tensor for global embedding
6. preserve the standalone Uni3D feature extractor boundary
7. run full TLS `.las` to Uni3D `.npy` conversion on the server with `--skip-existing`
8. inspect conversion `summary.json` and point-count failures
9. run full TLS Uni3D embedding extraction on the GPU server with `--skip-existing`
10. inspect extraction `summary.json`, finite checks, and L2 norms
11. run sampled similarity analysis before deciding whether to save any full NxN matrix
12. manually review top near-duplicate and outlier samples in the point clouds
13. decide whether to split future analyses by RGB availability / plot / species
14. rerun quantitative review on the server with the real full `.npy` input directory to fill xyz/rgb statistics
15. only then decide how to use the features downstream

---

## 7.  Stage Roadmap

### Current overall stage

The project has completed problem scoping and is currently at:

**Stage 1 complete for q1 / q2 / q3 baseline v1.1 -> entering Stage 2 (Uni3D independent feature extractor)**

This means:

- q1 / q2 / q3 baseline v1.1 is frozen
- single-tree MVP v1 is connected to a local Ollama verbalizer
- the next major step is not immediate q4/q5 expansion or free-form LLM integration
- the next major step is to introduce a learned 3D representation branch in a controlled way

------

### Stage 0 — Problem Scoping and Boundary Definition

**Goal:**
Define a research problem that is small enough to execute, clear enough to explain, and structured enough to evaluate.

**What this stage should accomplish:**

- restrict the task to single-tree point cloud understanding
- identify the current stable core tasks (q1/q2/q3)
- clarify what is in scope and what is out of scope
- reject open-ended LLM-only reasoning as the current main path
- establish the principle of geometry grounding first

**Expected outputs:**

- a clear problem statement
- a reduced and workable project scope
- explicit non-goals
- an initial research narrative

**Current status:**
This stage is considered largely complete.

------

### Stage 1 - Baseline v1.1 Freeze

**Goal:**
Freeze the currently working fixed pipeline into a reproducible and clearly documented baseline.

**What this stage should accomplish:**

- identify the real baseline entrypoint and actual working path
- define what baseline v1.1 does and does not include
- preserve the current behavior as the reference system
- provide a clean way to run and verify the baseline
- document the baseline clearly for future comparison

**Expected outputs:**

- Baseline v1.1 documentation
- clear entrypoint / run path
- minimal runnable example or smoke test
- stable description of supported tasks and outputs

**Current status:**
This stage is complete for q1 / q2 / q3 baseline v1.1.
Single-tree MVP v1 now includes a local Ollama verbalizer as a downstream wording layer.

------

### Stage 2 — Uni3D as an Independent Feature Extractor

**Goal:**
Turn Uni3D into a frozen, standalone, cacheable feature extractor before integrating it into the main system.

**What this stage should accomplish:**

- inspect the official Uni3D repo
- identify builder / checkpoint loading / inference path / output tensor
- confirm which output should be used as global embedding
- implement a standalone extractor
- add cache support
- run minimal sanity checks

**Important constraints:**

- do not integrate Uni3D directly into the main QA pipeline yet
- do not replace q1/q2/q3 geometry tools
- do not immediately connect Uni3D to LLM generation
- do not expand q4/q5 in parallel

**Expected outputs:**

- Uni3D reconnaissance report
- standalone feature extractor
- cache mechanism
- sanity check results

**Current status:**
This is the next main stage.

------

### Stage 3 — Minimal Method Integration

**Goal:**
Integrate geometry grounding and Uni3D representation into a minimal method version of the system.

**What this stage should accomplish:**

- define the role of Uni3D features inside the method
- define how geometry signals and learned features complement each other
- produce a minimal integrated method pipeline
- clarify the system’s structured intermediate representation

**Expected outputs:**

- method structure diagram
- minimal integrated pipeline
- explicit module-role definition

**Exit condition:**
The method must be clearly explainable: what is added beyond the baseline, why it is added, and what role it plays.

------

### Stage 4 — Experiments and Ablations

**Goal:**
Demonstrate that the proposed method is useful, not just implemented.

**What this stage should accomplish:**

- compare baseline v1.1 vs method-enhanced version
- run ablations on geometry / Uni3D / combined settings
- evaluate stability and difficult cases
- analyze failure cases

**Expected outputs:**

- main result tables
- ablation tables
- stability observations
- failure case analysis

**Exit condition:**
It should be possible to answer:

- what improved
- why it improved
- where it did not improve
- what the method boundary is

------

### Stage 5 — Paper Writing and Presentation

**Goal:**
Turn the implemented system and experimental findings into a coherent small paper.

**What this stage should accomplish:**

- write the problem, method, experiments, and conclusions clearly
- produce figures, tables, and method diagrams
- refine contribution statements
- prepare for report / defense / presentation

**Expected outputs:**

- paper draft
- figure/table set
- presentation outline
- contribution summary

------

### Current stage focus

The current focus should remain:

1. preserve q1 / q2 / q3 baseline v1.1
2. preserve single-tree MVP v1 verbalizer boundaries
3. inspect the official Uni3D implementation
4. confirm the real feature extraction path
5. implement a standalone Uni3D extractor
6. run minimal sanity checks

------

### Things that are explicitly not the current focus

- not full q4/q5 expansion
- not open-ended LLM reasoning
- not large-scale architecture redesign
- not feature-to-text generation directly
- not learning “all LLM knowledge” before implementation

------

### Current success criterion

At the current project stage, success means:

- q1 / q2 / q3 baseline v1.1 is frozen and explainable
- single-tree MVP v1 local Ollama verbalizer is bounded as a verbalization layer
- Uni3D official implementation path is understood at the interface level
- a standalone global feature extractor can be built next with confidence

## 8. Short Status Summary

If a future agent needs a short summary, use this:

- ForestAgent is a single-tree point cloud research prototype.
- q1/q2/q3 baseline v1.1 is frozen.
- single-tree MVP v1 is connected to a local Ollama verbalizer.
- The Uni3D extractor has passed a real GPU forward sanity check on the lab server.
- The current main QA system is still not using Uni3D features by default.
- q1/q2/q3 remain the currently stable core tasks.
- The next main step is a Uni3D extractor / feature branch, not q4/q5 expansion.
- Uni3D should first be used only for global feature extraction + cache + sanity checks.
- Preserve geometry grounding and do not let any language layer invent physical facts.

---

## 9. Server Profile v0.1

This profile records the current target server environment for future Uni3D / feature-extraction work.

### 9.1 Machine

- Hostname: `teamlu-System-Product-Name`
- User: `team-lu`
- Home: `/home/team-lu`
- Remote access: ToDesk desktop
- SSH: unavailable for now

### 9.2 OS

- Ubuntu 22.04.5 LTS
- Kernel: `6.8.0-64-generic`

### 9.3 Hardware

- RAM: 125 GiB
- Disk: 1.9 TB total, 1.6 TB available on `/`
- GPU: 2 x NVIDIA GeForce RTX 3090, 24 GB each

### 9.4 NVIDIA / CUDA

- NVIDIA Driver: `535.230.02`
- `nvidia-smi` CUDA Version: 12.2
- `nvcc` CUDA Toolkit: 11.8, V11.8.89

### 9.5 Python / Conda

- Conda path: `/home/team-lu/anaconda3/bin/conda`
- Conda version: 24.11.3
- Current env: `base`
- base Python: 3.12.2
- ForestAgent / Uni3D server env: `fa-u`
- Use: `conda activate fa-u`
- Recommendation: do not use `base` for ForestAgent server work; use `fa-u`

### 9.6 Server workspace

- Personal project root: `/home/team-lu/harrison_workspace`
- Code: `~/harrison_workspace/projects`
- Data: `~/harrison_workspace/data/forestagent`
- Checkpoints: `~/harrison_workspace/checkpoints`
- Runs: `~/harrison_workspace/runs/forestagent`
- Logs: `~/harrison_workspace/logs/forestagent`
- Outputs: `~/harrison_workspace/outputs/forestagent`
- Use `tmux` for long-running jobs
- Use `CUDA_VISIBLE_DEVICES=1` when targeting GPU 1
- Back up code, configs, logs, metrics, and best checkpoints
- Avoid downloading everything from the server; pull only the artifacts needed for review or recovery

### 9.7 Codex-server workflow

- Codex and the server are not directly coupled initially
- Use GitHub/Gitee as the bridge
- Codex edits code and pushes commits / PRs
- Server pulls code under `~/harrison_workspace/projects/ForestAgent`
- Server runs GPU jobs with `conda activate fa-u`, `tmux`, and `CUDA_VISIBLE_DEVICES=1`
- Do not let Codex modify shared server environments or run long GPU jobs without manual supervision

---

## 10. Update Template For Future Tasks

When a task changes the verified project state, update only the relevant parts of this file.
Prefer incremental edits.

Recommended update checklist:

1. **Confirmed Current State**
   - what is newly verified?
   - what is still not implemented?

2. **Current Main Development Direction**
   - did the main next step change?

3. **Known Uncertainties**
   - which uncertainty was resolved?
   - which new uncertainty appeared?

4. **Immediate Next Actions**
   - what should happen next, based on the latest verified state?
