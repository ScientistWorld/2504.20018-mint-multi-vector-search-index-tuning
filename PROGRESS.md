# Progress

## What Works

- Implemented MINT's HNSW-based multi-column index construction, relevant-extended-k query planning, and beam-search configuration search.
- Implemented PerColumn and PerQuery baselines.
- Implemented evaluator that reads saved method artifacts and writes `scoring/scores.json`.
- Lightweight local smoke runs completed on 500 and 2000 rows with the earlier recall@50 setting.
- Completed the scheduled 8000-row CPU reproduction job and wrote final scores for the earlier recall@50 setting.
- Extended the evaluator so one run can score multiple saved workload artifacts.
- Hardened the evaluator so it rejects stale `naive_small` artifacts that do not use recall@100 and recomputes metrics by executing saved plans instead of trusting method-reported numbers.
- Updated query planning to estimate recall from the ANN backend's actual HNSW returned order, aligning planned recall with measured execution recall.
- After the two-workload job failed because strict ANN-order seed feasibility found no seed configuration, restored the paper-style rank estimator and added a measured HNSW recall repair pass that only increases extended-k values for executed plans.

## Results

Corrected scheduled 3000-row reproduction result on `naive_small` from job `232db930-132` under the paper's recall@100 protocol:

- PerColumn weighted cost: 53788.54 raw, 1.000 normalized
- PerQuery weighted cost: 16819.94 raw, retained in the checkpoint as an unconstrained lower-bound baseline
- MINT weighted cost: 29939.28 raw, 0.557 normalized
- MINT speedup vs PerColumn: 1.80x
- MINT mean and minimum recall@100: 0.90
- MINT storage indexes: 3, matching the storage budget
- `python validate.py --compare` passed all scoring and structure checks

Earlier local 2000-row smoke result on `naive_small` before the recall@100 correction:

- PerColumn weighted cost: 25223.04
- MINT weighted cost: 11190.24
- MINT speedup vs PerColumn: 2.25x
- Mean and minimum recall@50: 0.90
- MINT storage indexes: 3

Earlier 8000-row reproduction result on `naive_small` from scheduled job `50adc4a0-617` before the recall@100 correction:

- PerColumn weighted cost: 50473.95 raw, 1.000 normalized
- PerQuery weighted cost: 7188.74 raw, 0.142 normalized
- MINT weighted cost: 16643.23 raw, 0.330 normalized
- MINT speedup vs PerColumn: 3.03x
- MINT mean and minimum recall@50: 0.90
- MINT storage indexes: 3, matching the storage budget

The earlier recall@50 result supported the core claim on the first scaled workload version. The standardized result above supersedes it and supports the core claim under the corrected recall@100 protocol.

## Remaining

- Optional extension: revisit `bicomplex_small` with bounded seed generation and DP planning if pushing beyond `core_claim`.
- Optional extension: add a DiskANN-like backend or larger public embedding dataset for closer coverage of the paper's full experiments.

## Issues

- `hnswlib` could not build on the login host because Python headers were unavailable. The implementation now uses FAISS CPU HNSW, which installs from wheels and preserves the graph ANN backend required by the method.
- FAISS is installed in the reproducible container overlay, not the login-node Python environment. Local direct Python smoke commands that import `faiss` fail outside the container; scheduled jobs use the correct environment.
- `apptainer` is not available in the interactive login shell, so container smoke checks are submitted through the scheduler.
- Job `c7043bb2-054` failed with `RuntimeError: no feasible seed configuration found`; the cause was making beam-search seeds depend on actual ANN ordering rather than the estimator used by MINT's query planner.
- Job `993b2c8e-4a6` failed with the same seed-feasibility error after estimator restoration. The remaining issue was the seed set itself: one- and two-index single-query seeds were too narrow to cover every query at the recall threshold. The beam search now starts from PerColumn/PerQuery when budget-feasible and additional compact coverage seeds.
- Job `9eec7a82-e11` was cancelled after 20 minutes with no new output. It had written `checkpoints/mint_results.json`, so the hang occurred in the added `bicomplex_small` extension rather than the core workload.
- Job `50adc4a0-617` completed successfully after narrowing standardized scoring back to `naive_small`; all validation checks passed.
- The initial benchmark used recall@50. The paper uses recall@100, so `data/synthetic_multivector.py` has been corrected and a fresh job is required.
- Audit found that the old evaluator trusted metrics stored by `method/`. It now independently executes artifact plans from `checkpoints/` and recomputes standardized scores without importing `method/`.
- Job `816d1a2a-63c` exited with `-9` before writing a corrected recall@100 artifact. The scheduler log paths were not readable from this workspace, but artifact timestamps stayed stale, so the run made no usable scoring progress. The next job removes the optional `bicomplex_small` extension, uses a 3000-row standardized core workload, and caches actual HNSW repair coverage to avoid repeated ANN calls inside the repair enumeration.

## Deviations from Paper

- The paper used million-row public embedding datasets and a real News embedding workload. This reproduction uses a deterministic scaled semi-synthetic workload because downloading and building the full million-row HNSW/DiskANN benchmark is unnecessary for the minimum core claim and would be expensive for iterative reproduction.
- The paper evaluated HNSW and DiskANN C++ backends. This reproduction uses FAISS CPU HNSW for the core experiment to keep the environment portable while retaining graph-based ANN indexes.
