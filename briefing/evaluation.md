# Evaluation Protocol

Evaluation reads saved result artifacts (`checkpoints/*_results.json`), executes their plans and recomputes every number. It never imports `method/` and never trusts numbers, flags or verdicts written in an artifact: reported costs, recalls, storage counts and diagnostics are ignored, and declarations can only make a run stricter (see *Artifact consistency*).

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
- The names `per_column` and `per_query` are reserved for the evaluator's own baselines; artifact entries with these names are never scored (but see below).
- A method present in some artifacts but missing a workload is scored as failed on that workload.

## Artifact consistency

An artifact is one submission and must not contradict the fixed protocol. Every method in an artifact fails on that workload if

- the artifact declares any of `n_rows`, `top_k`, `recall_threshold`, `storage_budget` with a value other than the protocol's (3000, 100, 0.90, and 3 or 5); leaving them out is fine;
- an entry under a reserved name (`per_column`, `per_query`) is present but is not a valid plan set (same plan rules as above).

A method may declare `configuration` (the list of indexes it builds). If it does, the list must hold valid, distinct index names and include every index its plans scan with `ek > 0`; otherwise the method fails.

## Execution

For each template, every evaluation query is executed: each listed index with `ek > 0` is searched for its top-`ek` items with a fixed FAISS `IndexHNSWFlat` engine (inner product on L2-normalized concatenated columns, M=16, efConstruction=120, efSearch=300, single-threaded), the union is formed, and recall@100 against the exact top-100 and the cost are computed. Storage is the number of distinct indexes with `ek > 0` across all template plans, together with any further indexes in the declared `configuration`.

## Feasibility gate

A run is **failed** on a workload if any plan is invalid, the artifact is inconsistent (above), any template's mean recall@100 is below 0.90, or storage exceeds the budget. A failed run gets fixed worst-case values on every metric (cost at least 100x PerColumn, speedup 0, recall 0, maximal violations and candidate counts), so it ranks below every feasible run.

## Slices

- `scripts/evaluate_train.sh` -> `scoring/scores_train.json`: the visible training queries (1000 per template; in-sample for any method that plans on them).
- `scripts/evaluate_test.sh` -> `scoring/scores_test.json`: held-out queries of the same templates (1000 per template), generated only inside the evaluator.
- `scripts/evaluate.sh` -> `scoring/scores.json`: same as the train slice.

All three score every `checkpoints/*_results.json` when called without arguments.

## Comparator and metrics

`per_column` (one single-column index per column, planned once on the 1000 training queries per template and frozen in `eval/*/baselines.json`) is the normalization baseline on the same queries. Per workload:

- `weighted_cost`: probability-weighted mean query cost / PerColumn's (lower is better; primary)
- `speedup_vs_per_column`: PerColumn cost / submission cost
- `mean_recall_at_k`, `min_recall_at_k`: mean and worst template mean recall@100
- `recall_floor_violation`, `storage_budget_violation`: 0 for every feasible run
- `storage_indexes`: distinct indexes used
- `mean_candidate_multiplier_at_k`, `max_candidate_multiplier_at_k`: mean / largest template mean distinct-candidate count divided by k

The ranking combines metrics with the coefficients in `scoring/reference.json` (both workloads weigh 0.5); cost and speedup dominate, recall above the floor, storage and candidate counts act as tie-breakers. A one-index-per-template layout (`per_query`) is reported under `_raw` for reference only: it exceeds the storage budget.
