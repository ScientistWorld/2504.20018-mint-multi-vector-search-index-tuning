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

Final 8000-row reproduction result on `naive_small`:

- PerColumn weighted cost: 50562.84 raw, 1.000 normalized
- PerQuery weighted cost: 7188.74 raw, 0.142 normalized
- MINT weighted cost: 16643.23 raw, 0.329 normalized
- MINT speedup vs PerColumn: 3.04x
- MINT mean and minimum recall@50: 0.90
- MINT storage indexes: 3, matching the storage budget

This supports the paper's core claim on the scaled workload: tuning multi-vector indexes and query plans can substantially reduce query cost relative to independent per-column indexes while meeting recall and storage constraints. `scoring/scores.json` reports normalized weighted cost to match the paper reference scale.

## Remaining

- Submitted extension: run `bicomplex_small`, a five-column/eight-query workload, alongside `naive_small` to test whether the core claim holds on an additional setting.
- Optional extension: add a DiskANN-like backend or larger public embedding dataset for closer coverage of the paper's full experiments.

## Issues

- `hnswlib` could not build on the login host because Python headers were unavailable. The implementation now uses FAISS CPU HNSW, which installs from wheels and preserves the graph ANN backend required by the method.
- FAISS is installed in the reproducible container overlay, not the login-node Python environment. Local direct Python smoke commands that import `faiss` fail outside the container; scheduled jobs use the correct environment.
- `apptainer` is not available in the interactive login shell, so container smoke checks are submitted through the scheduler.
- Job `c7043bb2-054` failed with `RuntimeError: no feasible seed configuration found`; the cause was making beam-search seeds depend on actual ANN ordering rather than the estimator used by MINT's query planner.

## Deviations from Paper

- The paper used million-row public embedding datasets and a real News embedding workload. This reproduction uses a deterministic scaled semi-synthetic workload because downloading and building the full million-row HNSW/DiskANN benchmark is unnecessary for the minimum core claim and would be expensive for iterative reproduction.
- The paper evaluated HNSW and DiskANN C++ backends. This reproduction uses FAISS CPU HNSW for the core experiment to keep the environment portable while retaining graph-based ANN indexes.
