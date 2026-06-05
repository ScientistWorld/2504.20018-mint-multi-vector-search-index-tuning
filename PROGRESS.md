# Progress

## What Works

- Implemented MINT's HNSW-based multi-column index construction, relevant-extended-k query planning, and beam-search configuration search.
- Implemented PerColumn and PerQuery baselines.
- Implemented evaluator that reads saved method artifacts and writes `scoring/scores.json`.
- Lightweight local smoke runs complete on 500 and 2000 rows.
- Completed the scheduled 8000-row CPU reproduction job and wrote final scores.
- Extended the evaluator so one run can score multiple saved workload artifacts.
- Updated query planning to estimate recall from the ANN backend's actual HNSW returned order, aligning planned recall with measured execution recall.
- After the two-workload job failed because strict ANN-order seed feasibility found no seed configuration, restored the paper-style rank estimator and added a measured HNSW recall repair pass that only increases extended-k values for executed plans.

## Results

Local 2000-row smoke result on `naive_small`:

- PerColumn weighted cost: 25223.04
- MINT weighted cost: 11190.24
- MINT speedup vs PerColumn: 2.25x
- Mean and minimum recall@50: 0.90
- MINT storage indexes: 3

Fresh 8000-row reproduction result on `naive_small` from scheduled job `50adc4a0-617`:

- PerColumn weighted cost: 50473.95 raw, 1.000 normalized
- PerQuery weighted cost: 7188.74 raw, 0.142 normalized
- MINT weighted cost: 16643.23 raw, 0.330 normalized
- MINT speedup vs PerColumn: 3.03x
- MINT mean and minimum recall@50: 0.90
- MINT storage indexes: 3, matching the storage budget

This supports the paper's core claim on the scaled workload: tuning multi-vector indexes and query plans can substantially reduce query cost relative to independent per-column indexes while meeting recall and storage constraints. `scoring/scores.json` reports normalized weighted cost to match the paper reference scale.

## Remaining

- Optional extension: revisit `bicomplex_small` with more efficient seed generation and progress logging if pushing beyond `core_claim`.
- Optional extension: add a DiskANN-like backend or larger public embedding dataset for closer coverage of the paper's full experiments.

## Issues

- `hnswlib` could not build on the login host because Python headers were unavailable. The implementation now uses FAISS CPU HNSW, which installs from wheels and preserves the graph ANN backend required by the method.
- FAISS is installed in the reproducible container overlay, not the login-node Python environment. Local direct Python smoke commands that import `faiss` fail outside the container; scheduled jobs use the correct environment.
- `apptainer` is not available in the interactive login shell, so container smoke checks are submitted through the scheduler.
- Job `c7043bb2-054` failed with `RuntimeError: no feasible seed configuration found`; the cause was making beam-search seeds depend on actual ANN ordering rather than the estimator used by MINT's query planner.
- Job `993b2c8e-4a6` failed with the same seed-feasibility error after estimator restoration. The remaining issue was the seed set itself: one- and two-index single-query seeds were too narrow to cover every query at the recall threshold. The beam search now starts from PerColumn/PerQuery when budget-feasible and additional compact coverage seeds.
- Job `9eec7a82-e11` was cancelled after 20 minutes with no new output. It had written `checkpoints/mint_results.json`, so the hang occurred in the added `bicomplex_small` extension rather than the core workload.
- Job `50adc4a0-617` completed successfully after narrowing standardized scoring back to `naive_small`; all validation checks passed.

## Deviations from Paper

- The paper used million-row public embedding datasets and a real News embedding workload. This reproduction uses a deterministic scaled semi-synthetic workload because downloading and building the full million-row HNSW/DiskANN benchmark is unnecessary for the minimum core claim and would be expensive for iterative reproduction.
- The paper evaluated HNSW and DiskANN C++ backends. This reproduction uses FAISS CPU HNSW for the core experiment to keep the environment portable while retaining graph-based ANN indexes.
