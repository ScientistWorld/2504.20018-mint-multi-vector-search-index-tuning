# Overview

The paper proposes MINT, a multi-vector search index tuner. Instead of creating only one index per vector column or one full index per query, MINT can recommend shared multi-column indexes and query plans that retrieve fewer candidates while maintaining recall and storage constraints.

This reproduction packages two compact semi-synthetic workloads shaped after the paper's Naive and BiComplex workloads, a FAISS-HNSW implementation of MINT's query planner and beam-search configuration tuner, the PerColumn and PerQuery baselines, and an evaluator that executes submitted per-template plans on held-out queries and scores cost, recall, storage and speedup, failing any run that breaks the recall floor or the storage budget.

---

# Problem Definition

The task is multi-vector search index tuning. A database row represents one item and contains several vector-valued columns. A query names a subset of columns (its *template*) and provides one query vector per named column. The query result is the top-k rows by the sum of cosine similarities over the named columns.

A workload is a set of query templates with probabilities. Given the database and the workload, choose

- a configuration: a set of ANN indexes, each built over the concatenation of one or more columns, and
- a plan per query template: which of the configuration's indexes to scan and how many items (`ek`) to retrieve from each; the union of retrieved items is re-ranked with the exact score,

so that the probability-weighted query cost is minimal while

- every template meets a recall@k floor (mean recall over that template's queries, measured against the exact full-scan top-k), and
- the number of distinct indexes used by the plans does not exceed the storage budget.

Plans are per template, so they must serve *unseen* queries of that template: the visible data provides a sample of training queries per template, and scoring uses held-out queries drawn from the same per-template distribution.

Two deterministic semi-synthetic workloads are provided (`data/synthetic_multivector.py`):

| workload | rows | columns (dims) | templates | k | recall floor | storage budget |
|---|---:|---|---:|---:|---:|---:|
| `naive_small` | 3,000 | 3 (32, 40, 48) | 4: {1}, {1,2}, {2,3}, {1,2,3} | 100 | 0.90 | 3 |
| `bicomplex_small` | 3,000 | 5 (16, 20, 24, 28, 32) | 8 subsets with >= 2 columns | 100 | 0.90 | 5 |

Each template has 200 visible training queries (`make_workload(name).train_queries`).

Cost is a machine-independent latency proxy: for each query, the sum over scanned indexes of `index dimension x ek`, plus `query dimension x number of distinct retrieved items` (re-ranking).

---

# Method

MINT tunes indexes for multi-vector search workloads. It considers candidate indexes over column subsets and asks a query planner ("what-if") how each query template would use a proposed configuration.

The reproduced method (`method/mint.py`) implements the paper's computational steps:

- graph-based HNSW ANN indexes (FAISS `IndexHNSWFlat`) on single-column and multi-column vector concatenations;
- exact ground truth by full scan for the training queries of each template;
- relevant extended-k (ek) values: the rank of every ground-truth item in each candidate index; Algorithm 1 enumerates the relevant ek of all but one index and derives the minimal last ek meeting the recall target (plans over <= 3 indexes); Algorithm 2 (dynamic programming over the power set of k' = 5 sampled ground-truth items, several samples) for larger plans;
- the simplified cost estimator: per-index scan cost `dim * (80 + 1.25 * ek)` plus re-ranking cost `query dim * sum(ek)` (duplicates counted, as in the paper);
- Algorithm 3 beam search over configurations under the storage and recall constraints (candidate indexes per template with at least |q| - di columns, di = 2; seeds of <= se = 2 per-template candidates plus covering seeds; beam width 5; stop when the improvement is <= im = 5%), with plan caching;
- PerColumn and PerQuery baselines planned by the same planner.

Scaled-setup choices (not algorithm changes):

- Plans are per template and must generalize to unseen queries, so planning pools the ground truth of the template's 200 training queries (the relevant-ek grid is thinned to at most 100-256 values per index, as the paper bounds it by k).
- The recall estimator is the training-sample recall minus a fixed generalization allowance of 0.03 (plans target 0.93 on the training queries for a 0.90 floor).
- Chosen plans are checked against the actual HNSW results on the training queries and ek of already-used indexes is raised where the measured recall falls short.
- The paper used HNSW and DiskANN C++ libraries on 1M-row public embeddings; this reproduction uses FAISS HNSW on 3,000-row synthetic workloads.

Optional flags (off for the reference): `--rerank-measured K` re-ranks the K best configurations by cost measured on the training queries; `--trim-deterministic` drops the allowance for single-index plans whose recall is identical on all training queries; `--beam-width/--seed-limit/--improvement-threshold/--max-iters/--local-search` widen the search; `--recall-margin` and `--train-seed` change the allowance and the training sample.

---

# Evaluation Protocol

Evaluation reads saved result artifacts (`checkpoints/*_results.json`), executes their plans and recomputes every number. It never imports `method/` and never trusts numbers, flags or declarations written in an artifact (reported costs, recalls, configuration lists, row counts, thresholds and budgets are ignored).

## Artifact format

One JSON file per workload:

```json
{
  "workload": "naive_small",
  "results": {
    "<method name>": {
      "plans": {
        "q1": {"indexes": ["x1"], "ek": [93]},
        "q2": {"indexes": ["x1,2", "x2"], "ek": [120, 0]}
      }
    }
  }
}
```

- `plans` needs an entry for every template of the workload (`q1`..`q4` for `naive_small`, `q1`..`q8` for `bicomplex_small`).
- `"x1,3"` names an index over the concatenation of columns 1 and 3 (1-based). Every index in a plan must use only that template's columns, appear once, and have an integer `ek` in `[0, 3000]`. `ek = 0` means the index is not scanned.
- The names `per_column` and `per_query` are reserved for the evaluator's own baselines; artifact entries with these names are ignored.
- A method present in some artifacts but missing a workload is scored as failed on that workload.

## Execution

For each template, every evaluation query is executed: each listed index with `ek > 0` is searched for its top-`ek` items with a fixed FAISS `IndexHNSWFlat` engine (inner product on L2-normalized concatenated columns, M=16, efConstruction=120, efSearch=300, single-threaded), the union is formed, and recall@100 against the exact top-100 and the cost are computed. Storage is the number of distinct indexes with `ek > 0` across all template plans.

## Feasibility gate

A run is **failed** on a workload if any plan is invalid, any template's mean recall@100 is below 0.90, or storage exceeds the budget. A failed run gets fixed worst-case values on every metric (cost at least 100x PerColumn, speedup 0, recall 0, maximal violations and candidate counts), so it ranks below every feasible run.

## Slices

- `scripts/evaluate_train.sh` -> `scoring/scores_train.json`: the visible training queries (in-sample for any method that plans on them).
- `scripts/evaluate_test.sh` -> `scoring/scores_test.json`: held-out queries of the same templates (500 per template), generated only inside the evaluator.
- `scripts/evaluate.sh` -> `scoring/scores.json`: same as the train slice.

All three score every `checkpoints/*_results.json` when called without arguments.

## Comparator and metrics

`per_column` (one single-column index per column, planned once on the training queries and frozen in `eval/*/baselines.json`) is the normalization baseline on the same queries. Per workload:

- `weighted_cost`: probability-weighted mean query cost / PerColumn's (lower is better; primary)
- `speedup_vs_per_column`: PerColumn cost / submission cost
- `mean_recall_at_k`, `min_recall_at_k`: mean and worst template mean recall@100
- `recall_floor_violation`, `storage_budget_violation`: 0 for every feasible run
- `storage_indexes`: distinct indexes used
- `mean_candidate_multiplier_at_k`, `max_candidate_multiplier_at_k`: mean / largest template mean distinct-candidate count divided by k

The ranking combines metrics with the coefficients in `scoring/reference.json` (both workloads weigh 0.5); cost and speedup dominate, recall above the floor, storage and candidate counts act as tie-breakers. A one-index-per-template layout (`per_query`) is reported under `_raw` for reference only: it exceeds the storage budget.

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 9m | **GPU**: 0.0h
- **Jobs**: 1 total (1 completed, 0 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 2: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 9m | **GPU**: 0.0h
- **Jobs**: 5 total (2 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 3: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 14m | **GPU**: 0.0h
- **Jobs**: 7 total (3 completed, 3 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 4: gpt-5.5 (designer_stage_1)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 6m | **GPU**: 0.0h
- **Jobs**: 7 total (3 completed, 3 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 5: gpt-5.5 (designer_stage_2)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
- **Jobs**: 7 total (3 completed, 3 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 6: gpt-5.5 (designer_stage_3)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 3m | **GPU**: 0.0h
- **Jobs**: 9 total (5 completed, 3 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 7: gpt-5.5 (packager)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 2m | **GPU**: 0.0h
- **Jobs**: 9 total (5 completed, 3 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 8: gpt-5.5 (verifier)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 5m | **GPU**: 0.0h
- **Jobs**: 9 total (5 completed, 3 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 9: gpt-5.5 (verifier)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 30m | **GPU**: 0.0h
- **Jobs**: 9 total (5 completed, 3 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 10: gpt-6.1-sol (redteam)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 10m | **GPU**: 0.0h
- **Jobs**: 9 total (5 completed, 3 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 38m | **GPU**: 0.0h
- **Jobs**: 12 total (8 completed, 3 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 10: gpt-6.1-sol (redteam)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 9m | **GPU**: 0.0h
- **Jobs**: 12 total (8 completed, 3 failed)

<details>
<summary>Progress Log</summary>

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

</details>


---

# Reproduction Milestones

**Current: core_claim**

## Stop Justification
- Completed at milestone `core_claim`.
