# Geometry vs Uni3D Nearest-Neighbor Comparison

This experiment checks whether frozen `xyz_only_v1` Uni3D embeddings produce a
nearest-neighbor structure that differs from simple geometry measurements.

## Boundary

- Offline analysis only.
- Not used by Baseline V0.
- Not part of the runtime question-answering pipeline.
- Does not run Uni3D forward.
- Does not train a model.
- Does not call an LLM.
- Does not modify q1/q2/q3 baseline algorithms.
- Does not support claims about health, danger, pest damage, cutting decisions,
  or management quality.

## Main Run

```powershell
python scripts/compare_geometry_uni3d_neighbors.py `
  --embedding-dir outputs/all_tls_xyz_only_v1 `
  --label-path data/parameters.xlsx `
  --output-dir outputs/local_reports/geometry_uni3d_nn_xyz_only_v1 `
  --top-k 10
```

The output directory is under `outputs/` and should not be committed.

## Compared Neighbor Definitions

- `geometry_measurement`: z-scored DBH, height, east-west crown width, and
  north-south crown width from `data/parameters.xlsx`.
- `uni3d_xyz_only`: cosine similarity over saved `embedding_l2` vectors.
- `geometry_measurement_plus_uni3d`: reciprocal-rank fusion of the two rankings.

The first pass uses measured geometry as the main geometry reference. A future
sensitivity check can pass a q1/q2/q3 cache through `--tool-geometry-cache`, but
that is deliberately not required for the main run.

## Key Outputs

- `neighbor_lists.csv/json`: per-tree Top-k neighbors for all methods.
- `per_query_overlap.csv`: overlap and Jaccard between methods.
- `geometry_difference_summary.csv`: geometry differences among retrieved neighbors.
- `site_species_summary.csv`: same-site and same-species rates.
- `near_duplicate_flags.csv`: retrieved pairs that hit known near-duplicate pairs.
- `manual_review_candidates.csv`: focused cases for human point-cloud review.
- `experiment_record.md`: concise experiment record.

## Interpretation

Useful signs:

- Uni3D neighbors are not just the same as geometry neighbors.
- Uni3D-only neighbors are visually or structurally plausible after manual review.
- Differences are not explained only by site leakage, species labels, or near duplicates.

Weak signs:

- Uni3D-only Top-k almost duplicates geometry-only Top-k.
- Uni3D-only Top-k is dominated by site or near-duplicate artifacts.
- Geometry + Uni3D fusion does not produce interpretable new cases.
