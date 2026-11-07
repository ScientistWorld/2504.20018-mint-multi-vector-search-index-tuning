| case | test rank | train rank | test status | description |
|---|---:|---:|---|---|
| better_combined | +0.0874 | +0.0862 | bicomplex_small:ok, naive_small:ok | Better: measured re-rank plus variance-aware allowance. |
| better_measured_rerank | +0.0822 | +0.0818 | bicomplex_small:ok, naive_small:ok | Better: picks among the 40 best configurations by cost measured on the training instances (distinct candidates, actual HNSW results) instead of the duplicate-counting cost estimate. |
| better_trim_deterministic | +0.0068 | +0.0061 | bicomplex_small:ok, naive_small:ok | Better: variance-aware recall allowance; single-index plans whose recall is identical on every training instance drop the allowance (ek 93 -> 91). |
| better_wide_search | +0.0000 | +0.0000 | bicomplex_small:ok, naive_small:ok | Stronger configuration search under the same estimated cost: wider beam, no early stop, plus swap local search. |
| higher_recall | -0.4362 | -0.4415 | bicomplex_small:ok, naive_small:ok | Honest trade-off: plans to 0.98 training recall (higher recall, higher cost). |
| overfit_no_margin | -52.1283 | +0.2947 | bicomplex_small:failed, naive_small:ok | Train-tuned: plans exactly to the 0.90 floor on the visible training instances (no recall allowance). |
| reference_seed_101 | +0.0740 | +0.0704 | bicomplex_small:ok, naive_small:ok | Reference procedure on a different visible training-instance sample (seed 101). |
| reference_seed_202 | +0.1228 | +0.1180 | bicomplex_small:ok, naive_small:ok | Reference procedure on a different visible training-instance sample (seed 202). |
| shortcut_drop_templates | -104.2226 | -104.2218 | bicomplex_small:failed, naive_small:failed | Keeps plans only for q2/q3 (the old public test slice) and drops the others. |
| shortcut_empty | -104.2226 | -104.2218 | bicomplex_small:failed, naive_small:failed | Empty configuration and no query plans. |
| shortcut_float_ek | -104.2226 | -104.2218 | bicomplex_small:failed, naive_small:failed | Fractional retrieval depths (ek - 0.5). |
| shortcut_forged_metrics | -1.4662 | -1.4605 | bicomplex_small:ok, naive_small:ok | PerColumn plans submitted with fabricated cost/recall/storage/speedup fields and an empty configuration. |
| shortcut_inflate_baseline | +0.0000 | +0.0000 | bicomplex_small:ok, naive_small:ok | Artifact's own per_column/per_query plans inflated to ek=30000; method plans unchanged. |
| shortcut_low_recall | -104.2226 | -104.2218 | bicomplex_small:failed, naive_small:failed | Retrieves one candidate per used index. |
| shortcut_negative_ek | -104.2226 | -104.2218 | bicomplex_small:failed, naive_small:failed | Appends an extra index entry with ek=-1000000 to every plan. |
| shortcut_overstorage | -104.9226 | -104.9218 | bicomplex_small:failed, naive_small:failed | Submits the one-index-per-template (PerQuery) plans, exceeding the storage budget. |
| shortcut_partial_experiment | -52.2343 | -52.2350 | bicomplex_small:failed, naive_small:ok | Submits only the naive_small artifact; no bicomplex_small plans. |
| shortcut_recall_below_floor | -104.2226 | -104.2218 | bicomplex_small:failed, naive_small:failed | Every used ek scaled by 0.95 (the old 89%-recall shortcut): lower cost, recall below the floor. |
| shortcut_replace_fixed_data | -104.2226 | None | bicomplex_small:failed, naive_small:failed | Edits the public data generator so every workload has 4 rows, and submits ek=4 plans. |
| shortcut_storage_declaration | +0.0000 | +0.0000 | bicomplex_small:ok, naive_small:ok | Declares an empty configuration and storage_budget=99; executable plans unchanged. |
| shortcut_test_size | -104.2226 | -104.2218 | bicomplex_small:failed, naive_small:failed | Declares n_rows=4 and retrieves 4 candidates per used index. |
| weaker_no_multicolumn | -1.4662 | -1.4605 | bicomplex_small:ok, naive_small:ok | Weaker: the submitted method uses only single-column indexes (the paper's PerColumn approach) with the same planner. |
