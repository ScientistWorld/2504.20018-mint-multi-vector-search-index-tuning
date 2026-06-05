# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-05 03:07] - none
- Started reproduction work.
- Read the paper and identified MINT's core components: HNSW/DiskANN candidate indexes, relevant-extended-k query planning, cost/recall constraints, and beam-search configuration tuning.

### [2026-06-05 03:07] - method_runs
- Implemented the paper's MINT computation with FAISS HNSW multi-column indexes, relevant-extended-k query planning, and beam-search configuration search.
- Implemented PerColumn and PerQuery baselines plus method-independent evaluation.
- Verified lightweight local smoke runs complete end to end.

### [2026-06-05 03:10] - core_claim
- Completed the 8000-row `naive_small` reproduction job successfully.
- MINT reached mean/min recall@50 of 0.90 with 3 stored indexes, satisfying the recall and storage constraints.
- MINT reduced weighted cost from PerColumn's 50562.84 to 16643.23, a 3.04x speedup/cost reduction.

## Stop Justification
- Core claim reproduced on the scaled HNSW multi-vector workload. Stopping at `core_claim`; larger public datasets and DiskANN-style backend coverage remain possible extensions.
