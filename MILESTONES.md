# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-05 03:16] - method_runs
- Implemented MINT's HNSW multi-column index construction, relevant extended-k query planning, and beam-search configuration tuning.
- End-to-end method and evaluator executed successfully on compact smoke workloads.

### [2026-06-05 03:20] - core_claim
- Completed an 8000-row scheduled CPU job for `naive_small`.
- MINT met recall@50 and storage constraints with three indexes.
- MINT achieved a 3.04x weighted-cost speedup over the PerColumn baseline, supporting the paper's central constrained index-tuning claim on a scaled workload.
