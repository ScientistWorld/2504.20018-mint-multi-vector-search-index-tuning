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

### [2026-06-05 03:48] - method_runs
- Continuation audit reset the active milestone below the previous `core_claim`.
- Attempted a second workload extension, but it caused the CPU test job to hang after producing the core `naive_small` artifact.
- Standardized scoring was narrowed back to `naive_small` pending a fresh passing core-claim job.

### [2026-06-05 03:50] - core_claim
- Completed fresh scheduled CPU job `50adc4a0-617` for `naive_small`.
- `validate.py --compare` passed inside the container.
- MINT achieved 3.03x speedup over PerColumn with mean/min recall@50 of 0.90 and three indexes, satisfying the paper's benefit and constraint metrics on the scaled core workload.
