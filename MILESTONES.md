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

### [2026-06-05 06:09] - core_claim
- Final verification kept the milestone at `core_claim`.
- Cleaned method-agnostic documentation wording and ensured the small canonical result artifact can be tracked for fresh-clone evaluation.
- `python validate.py --compare` passed after the verification edits.

### [2026-11-08] - core_claim (final hardening)
- Training sample 200 -> 1000 queries per template, held-out test 500 -> 1000, comparator re-frozen; artifact consistency checks in both evaluators; exact-equivalent greedy_cover speed-up (reference rerun reproduces the committed plans).
- Hidden test: MINT 0.554 / 0.447 normalized cost (1.81x / 2.24x vs PerColumn) at recall >= 0.915 and budget storage; seed spread 0.042 (was 0.123); calibrated variant +0.149..+0.268, PerColumn -1.539; the three tie shortcuts now fail; seed lottery +0.042. See HARDENING.md.

## Stop Justification
- Completed at milestone `core_claim`.
