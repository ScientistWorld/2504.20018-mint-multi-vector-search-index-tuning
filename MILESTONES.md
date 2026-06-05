# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-05 03:56] - method_runs
- Existing MINT implementation runs end-to-end on a scaled Naive-style workload.
- Corrected the benchmark from recall@50 to the paper's recall@100 protocol; a fresh scheduled run is required before claiming `core_claim`.

### [2026-06-05 05:03] - method_runs
- Job `816d1a2a-63c` was killed before producing a corrected recall@100 artifact.
- Optimized the actual HNSW recall repair pass and narrowed the next submission to the standardized core workload.

### [2026-06-05 05:43] - core_claim
- Job `232db930-132` completed the corrected recall@100 `naive_small` reproduction.
- MINT achieved 1.80x speedup over PerColumn with mean/min recall@100 of 0.90 and 3 storage indexes, matching the storage budget.
- `python validate.py --compare` passed all scoring and structure checks.

## Stop Justification
- Core claim reproduced and environment packaged. Additional workloads remain optional extensions.
