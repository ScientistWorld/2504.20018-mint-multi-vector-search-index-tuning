# Progress

## What Works

- Implemented MINT's HNSW-based multi-column index construction, relevant-extended-k query planning, and beam-search configuration search.
- Implemented PerColumn and PerQuery baselines.
- Implemented evaluator that reads saved method artifacts and writes `scoring/scores.json`.
- Lightweight local smoke runs complete on 500 and 2000 rows.

## Results

Local 2000-row smoke result on `naive_small`:

- PerColumn weighted cost: 25223.04
- MINT weighted cost: 11190.24
- MINT speedup vs PerColumn: 2.25x
- Mean and minimum recall@50: 0.90
- MINT storage indexes: 3

The official 8000-row result is being submitted as a CPU job.

## Remaining

- Run the scheduled 8000-row reproduction job.
- Inspect `scoring/scores.json` after the job and update milestone if the result supports the core claim.

## Issues

- `hnswlib` could not build on the login host because Python headers were unavailable. The implementation now uses FAISS CPU HNSW, which installs from wheels and preserves the graph ANN backend required by the method.

## Deviations from Paper

- The paper used million-row public embedding datasets and a real News embedding workload. This reproduction uses a deterministic scaled semi-synthetic workload because downloading and building the full million-row HNSW/DiskANN benchmark is unnecessary for the minimum core claim and would be expensive for iterative reproduction.
- The paper evaluated HNSW and DiskANN C++ backends. This reproduction uses FAISS CPU HNSW for the core experiment to keep the environment portable while retaining graph-based ANN indexes.
