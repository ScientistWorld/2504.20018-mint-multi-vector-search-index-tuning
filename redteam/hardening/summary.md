| case | test rank | train rank | test status | description |
|---|---:|---:|---|---|
| better_margin_0p01 | -57.4380 | +0.1926 | bicomplex_small:failed, naive_small:ok | Better-calibrated recall allowance (0.01 instead of 0.02); same search. |
| better_wide_search | +0.0000 | +0.0000 | bicomplex_small:ok, naive_small:ok | Stronger configuration search: wider beam, no early stop, plus swap local search. |
| higher_recall | -0.4749 | -0.4882 | bicomplex_small:ok, naive_small:ok | Honest trade-off: plans to 0.98 training recall (higher recall, higher cost). |
| overfit_no_margin | -56.7965 | +1.0981 | bicomplex_small:failed, naive_small:ok | Train-tuned: plans exactly to the 0.90 floor on the visible training instances (no recall allowance). |
| reference_seed_101 | -57.3423 | -57.3608 | bicomplex_small:failed, naive_small:ok | Reference procedure on a different visible training-instance sample (seed 101). |
| reference_seed_202 | -57.4313 | +0.2480 | bicomplex_small:failed, naive_small:ok | Reference procedure on a different visible training-instance sample (seed 202). |
| shortcut_drop_templates | -115.1935 | -115.2054 | bicomplex_small:failed, naive_small:failed | Keeps plans only for q2/q3 (the old public test slice) and drops the others. |
| shortcut_empty | -115.1935 | -115.2054 | bicomplex_small:failed, naive_small:failed | Empty configuration and no query plans. |
| shortcut_float_ek | -115.1935 | -115.2054 | bicomplex_small:failed, naive_small:failed | Fractional retrieval depths (ek - 0.5). |
| shortcut_forged_metrics | -115.1935 | -1.5171 | bicomplex_small:failed, naive_small:failed | PerColumn plans submitted with fabricated cost/recall/storage/speedup fields and an empty configuration. |
| shortcut_inflate_baseline | +0.0000 | +0.0000 | bicomplex_small:ok, naive_small:ok | Artifact's own per_column/per_query plans inflated to ek=30000; method plans unchanged. |
| shortcut_low_recall | -115.1935 | -115.2054 | bicomplex_small:failed, naive_small:failed | Retrieves one candidate per used index. |
| shortcut_negative_ek | -115.1935 | -115.2054 | bicomplex_small:failed, naive_small:failed | Appends an extra index entry with ek=-1000000 to every plan. |
| shortcut_overstorage | -116.3935 | -116.4054 | bicomplex_small:failed, naive_small:failed | Submits the one-index-per-template (PerQuery) plans, exceeding the storage budget. |
| shortcut_partial_experiment | -57.4748 | -57.4906 | bicomplex_small:failed, naive_small:ok | Submits only the naive_small artifact; no bicomplex_small plans. |
| shortcut_recall_below_floor | -115.1935 | -115.2054 | bicomplex_small:failed, naive_small:failed | Every used ek scaled by 0.95 (the old 89%-recall shortcut): lower cost, recall below the floor. |
| shortcut_replace_fixed_data | -115.1935 | None | bicomplex_small:failed, naive_small:failed | Edits the public data generator so every workload has 4 rows, and submits ek=4 plans. |
| shortcut_storage_declaration | +0.0000 | +0.0000 | bicomplex_small:ok, naive_small:ok | Declares an empty configuration and storage_budget=99; executable plans unchanged. |
| shortcut_test_size | -115.1935 | -115.2054 | bicomplex_small:failed, naive_small:failed | Declares n_rows=4 and retrieves 4 candidates per used index. |
| weaker_no_multicolumn | -115.1935 | -1.5171 | bicomplex_small:failed, naive_small:failed | Weaker: the submitted method uses only single-column indexes (the paper's PerColumn approach) with the same planner. |
