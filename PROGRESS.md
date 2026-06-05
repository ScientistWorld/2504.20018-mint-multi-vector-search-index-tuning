# Progress

## What Works

- Implemented MINT's HNSW-based multi-column index construction, relevant-extended-k query planning, and beam-search configuration search.
- Implemented PerColumn and PerQuery baselines.
- Implemented evaluator that reads saved method artifacts and writes `scoring/scores.json`.
- Lightweight local smoke runs complete on 500 and 2000 rows.
- Completed the scheduled 8000-row CPU reproduction job and wrote final scores.

## Results

Local 2000-row smoke result on `naive_small`:

- PerColumn weighted cost: 25223.04
- MINT weighted cost: 11190.24
- MINT speedup vs PerColumn: 2.25x
- Mean and minimum recall@50: 0.90
- MINT storage indexes: 3

Final 8000-row reproduction result on `naive_small`:

- PerColumn weighted cost: 50562.84
- PerQuery weighted cost: 7188.74
- MINT weighted cost: 16643.23
- MINT speedup vs PerColumn: 3.04x
- MINT mean and minimum recall@50: 0.90
- MINT storage indexes: 3, matching the storage budget

This supports the paper's core claim on the scaled workload: tuning multi-vector indexes and query plans can substantially reduce query cost relative to independent per-column indexes while meeting recall and storage constraints.

## Remaining

- Optional extension: add a second workload with different query mix or dimensionality to reach `core_claim_plus`.
- Optional extension: add a DiskANN-like backend or larger public embedding dataset for closer coverage of the paper's full experiments.

## Issues

- `hnswlib` could not build on the login host because Python headers were unavailable. The implementation now uses FAISS CPU HNSW, which installs from wheels and preserves the graph ANN backend required by the method.

## Deviations from Paper

- The paper used million-row public embedding datasets and a real News embedding workload. This reproduction uses a deterministic scaled semi-synthetic workload because downloading and building the full million-row HNSW/DiskANN benchmark is unnecessary for the minimum core claim and would be expensive for iterative reproduction.
- The paper evaluated HNSW and DiskANN C++ backends. This reproduction uses FAISS CPU HNSW for the core experiment to keep the environment portable while retaining graph-based ANN indexes.
