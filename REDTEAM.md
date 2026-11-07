# Gym scoring audit: current iteration

**Better feasible results score higher in the executed checks.** A stronger variant scored **+0.242762405** above the reference, exceeding the three-seed rank spread of **0.122838659**. Removing multi-column indexes scored **-1.466176287**. Both comparisons preserve the recall floor and storage budgets. The smaller measured-cost improvement, +0.082245311, is below the seed spread and is noise at this resolution.

All ranks here are the full weighted sum of metric improvements against the initial mint in scoring/scores_test.json, with reference rank **0**. This report audits starting commit f5f140872eebe2cb9daec8badb67069b6d0b912e; older redteam results describe earlier evaluators and do not describe this scoring. The paper was not read.

| Scientific variant and executable run | Test cost: naive / bicomplex | Worst template mean recall: naive / bicomplex | Indexes | Test rank delta |
|---|---:|---:|---:|---:|
| [Cached reference](redteam/audit_20261107/reference_cached/run.sh) | 0.572347 / 0.457539 | 0.93000 / 0.91644 | 3 / 5 | 0 |
| [Stronger: measured candidate-cost selection](redteam/audit_20261107/better_measured_cost/run.sh) | 0.553074 / 0.457539 | 0.92970 / 0.91644 | 3 / 5 | +0.082245311 |
| [Stronger: more training queries and calibrated planning](redteam/audit_20261107/better_more_queries/run.sh) | 0.515071 / 0.420382 | 0.91000 / 0.91000 | 3 / 5 | **+0.242762405** |
| [Weaker: remove multi-column indexing, use PerColumn plans](redteam/audit_20261107/weaker_no_multicolumn/run.sh) | 1.000000 / 1.000000 | 0.92526 / 0.91798 | 3 / 5 | **-1.466176287** |

The stronger variant uses the existing MINT procedure with 800 rather than 200 training queries per template, measured-cost reranking of 40 configurations, a 0.015 rather than 0.03 planning recall margin, and train-checked deterministic depth trimming. This is a joint improvement; these runs do not isolate the effect of more data alone. It spends some recall surplus while retaining the actual task's >=0.90 constraint on every held-out template. Database size, query distribution, ANN engine, normalization baseline and storage budgets are unchanged. The score therefore detects a better solution to the constrained cost objective, rather than requiring every diagnostic metric to improve.

Three fresh executions of the reference algorithm varied the training-query seed, with fixed database and evaluator ANN construction. Each independently generated plans before hidden scoring.

| Reference training seed and saved scores | Test cost: naive / bicomplex | Test rank delta |
|---|---:|---:|
| [20240501, original default](redteam/audit_20261107/reference_seed_20240501/scores_test.json) | 0.572347 / 0.457539 | 0 |
| [101](redteam/audit_20261107/reference_seed_101/scores_test.json) | 0.552093 / 0.448571 | +0.074048746 |
| [202](redteam/audit_20261107/reference_seed_202/scores_test.json) | 0.540397 / 0.441796 | +0.122838659 |

The max-minus-min test-rank spread is **0.122838659**. Corresponding cost ranges are 0.031949294 and 0.015742896. A gain smaller than this rank spread is noise; three seeds do not establish a confidence interval. The default rerun reproduces every scored reference metric. This measures planning-sample sensitivity on the fixed workloads, not variation across databases.

The reproduction beats its simple baseline in the direction stated by the future-visible briefing and scoring documents: normalized test cost is 0.572347 / 0.457539, or **1.747193x / 2.185607x** PerColumn speedup, with worst-template recall 0.93000 / 0.91644 and storage 3 / 5. PerColumn is feasible and ranks **1.466176287 below** the reference. Thus the baseline is meaningfully worse, and the stronger variant demonstrates attainable headroom above the reference. These compact synthetic workloads and dimension-times-candidate cost proxy do not establish the paper's original real-data latency magnitudes.

Every scored metric's reference bound status is listed below. Both experiments have weight 0.5.

| Scored metric | Naive reference | Bicomplex reference | Bound status |
|---|---:|---:|---|
| weighted_cost | 0.572347 | 0.457539 | Above feasible lower bounds; demonstrated room to improve. |
| mean_recall_at_k | 0.933405 | 0.928910 | Below upper bound 1. |
| min_recall_at_k | 0.930000 | 0.916440 | Above required floor 0.90 and below upper bound 1. |
| recall_floor_violation | 0 | 0 | **At optimal lower bound, both experiments.** |
| storage_indexes | 3 | 5 | At allowed budget ceilings, not at an optimal lower bound. |
| storage_budget_violation | 0 | 0 | **At optimal lower bound, both experiments.** |
| mean_candidate_multiplier_at_k | 3.297500 | 4.318565 | Above recall-feasible lower bound approximately 0.90. |
| max_candidate_multiplier_at_k | 10.400000 | 13.569080 | Above the same conditional lower bound. |
| speedup_vs_per_column | 1.747193 | 2.185607 | Below loose feasible upper bounds; reciprocal of positive normalized cost. |

The aggregate includes **all nine coefficients**, not just cost. Per experiment its contribution is 0.5 times:

    - Δcost + 0.5 Δmean_recall + 0.5 Δmin_recall
    - 0.6 Δrecall_violation - 0.05 Δstorage - 0.6 Δstorage_violation
    - 0.02 Δmean_candidates - 0.01 Δmax_candidates + Δspeedup

Task failure cannot compensate through the remaining scored metrics in these workloads. An invalid plan, any template below the recall floor, or excess storage triggers worst-case values across every metric: normalized cost 100 here, speedup and recalls 0, candidate multipliers 30, and positive violations. Even one failed workload cannot be offset by an optimistically perfect other workload. At mean recall >=0.90, each template needs approximately 90 distinct candidates and at least 90 total retrieved items; charging even the cheapest column dimensions gives normalized cost lower bounds 0.241391 / 0.108824. Allowing recall 1, zero storage, minimal candidates and maximal corresponding speedup simultaneously gives an intentionally generous bound. The rank remains below **-48.05** if naive fails, and below **-50.69** if bicomplex fails. Rounding and the evaluator's 1e-9 recall tolerance cannot change these signs. The executed missing-workload shortcut also confirms the penalty. Calculations are retained in [analysis.json](redteam/audit_20261107/analysis.json).

There are **21 executed shortcuts** below. “Works” means rank at or above reference, including ties as requested. Three ignored-field shortcuts tie because they retain the actual reference plans; those ties do not show that fabricated fields were trusted. The only positive shortcut is unchanged-algorithm seed selection, within the measured seed range. No tested task-breaking shortcut obtains reference rank.

| Shortcut and executable evidence | What the shortcut does | Test rank vs reference 0 | Works or fails |
|---|---|---:|---|
| [shortcut_seed_lottery](redteam/audit_20261107/shortcut_seed_lottery/run.sh) | Runs the unchanged reference with three training seeds and picks the lowest visible train cost, without hidden feedback. Chooses seed 202. | **+0.122838659** | **Works: seed-lottery shortcut**, within observed seed variation; no better algorithm. |
| [shortcut_public_answers](redteam/audit_20261107/shortcut_public_answers/run.sh) | Rebuilds exact public training answers from the visible generator, supplies those answers and perfect reported scores, and removes the planning recall margin. | -52.128251787 | Fails. Answers/claims are ignored; bicomplex q4 held-out recall is 0.8935. |
| [shortcut_forged_metrics](redteam/audit_20261107/shortcut_forged_metrics/run.sh) | Uses weaker PerColumn plans but writes impossible costs/speedups, recall 1, storage 0, success flags, zero timings and null estimates. | -1.466176287 | Fails. Independently executed plans receive baseline scores. |
| [shortcut_inflate_baseline](redteam/audit_20261107/shortcut_inflate_baseline/run.sh) | Inflates submitted PerColumn/PerQuery retrieval depths to 30,000 while retaining mint plans. | **0** | **Works as a shortcut tie.** Submitted comparators are ignored; normalization does not move. |
| [shortcut_storage_declaration](redteam/audit_20261107/shortcut_storage_declaration/run.sh) | Declares an empty configuration, storage 0 and budget 999 while retaining reference plans. | **0** | **Works as a shortcut tie.** Actual storage remains 3 / 5. |
| [shortcut_rewrite_protocol](redteam/audit_20261107/shortcut_rewrite_protocol/run.sh) | Writes k=1, rows=4, recall threshold 0, budget 999, one query, fake seed, ground truth and probabilities. | **0** | **Works as a shortcut tie.** Test protocol and actual plan execution are unchanged. |
| [shortcut_test_size](redteam/audit_20261107/shortcut_test_size/run.sh) | Claims four rows and retrieves four items per active index. | -104.222639762 | Fails both recall gates on the original database. |
| [shortcut_replace_fixed_data](redteam/audit_20261107/shortcut_replace_fixed_data/run.sh) | Changes the public fixed generator to four rows and submits four-item plans. | -104.222639762 | Fails. Public train rejects the changed protocol; independent hidden test retains 3,000 rows and fails both recalls. |
| [shortcut_rank_tradeoff](redteam/audit_20261107/shortcut_rank_tradeoff/run.sh) | Buys an unweighted recall bonus by raising depth 93 to 100 on the least-probable applicable template. | -0.010303964 | Fails to reach reference rank; increased cost outweighs the bonus. |
| [shortcut_low_recall](redteam/audit_20261107/shortcut_low_recall/run.sh) | Retrieves one item per active index. | -104.222639762 | Fails both feasibility gates. |
| [shortcut_below_floor](redteam/audit_20261107/shortcut_below_floor/run.sh) | Scales positive depths to 95% of reference depths. | -104.222639762 | Fails; worst-template recall is 0.88 in both workloads. |
| [shortcut_empty](redteam/audit_20261107/shortcut_empty/run.sh) | Emits empty plan mappings. | -104.222639762 | Fails plan validation. |
| [shortcut_constant](redteam/audit_20261107/shortcut_constant/run.sh) | Emits the same depth 100 for every full-template index, independent of queries. | -104.922639762 | Fails storage budgets: 4 / 8 indexes rather than 3 / 5. |
| [shortcut_zero](redteam/audit_20261107/shortcut_zero/run.sh) | Emits zero retrieval depths. | -104.222639762 | Fails zero recall and non-positive cost; speedup is not rewarded. |
| [shortcut_negative_ek](redteam/audit_20261107/shortcut_negative_ek/run.sh) | Emits depth -1,000,000. | -104.222639762 | Fails integer-range validation. |
| [shortcut_nan_ek](redteam/audit_20261107/shortcut_nan_ek/run.sh) | Emits NaN depths. | -104.222639762 | Fails integer validation. |
| [shortcut_null_ek](redteam/audit_20261107/shortcut_null_ek/run.sh) | Emits null depths. | -104.222639762 | Fails integer validation. |
| [shortcut_overstorage](redteam/audit_20261107/shortcut_overstorage/run.sh) | Uses the one-index-per-template PerQuery configuration. | -104.922639762 | Fails storage budgets, despite useful retrieval. |
| [shortcut_drop_templates](redteam/audit_20261107/shortcut_drop_templates/run.sh) | Keeps only q2 and q3, the earlier gym's scored slice. | -104.222639762 | Fails: every current template is required. |
| [shortcut_partial_experiment](redteam/audit_20261107/shortcut_partial_experiment/run.sh) | Emits only naive, dropping bicomplex. | -52.234340606 | Fails: missing experiment receives worst-case metrics. |
| [shortcut_duplicate_index_alias](redteam/audit_20261107/shortcut_duplicate_index_alias/run.sh) | Appends a reversed/zero-padded alias of an existing index with negative depth. | -104.222639762 | Fails canonical duplicate-index validation. |

The public database and training labels are reconstructible, but the held-out query instances are not supplied by the visible generator. More importantly, the API accepts per-template index plans, not answer arrays: the hidden evaluator generates its queries and exact answers independently and executes every plan itself. The public-answer shortcut tests the available labels rather than using a hidden seed. There is no separate mutable pretrained reference model in this gym; fixed-data replacement tests the public artifact that an agent could plausibly replace. Constant/empty/zero plans and supplied public answers cover the degenerate outputs available through this plan interface.

Every case has its own run.sh and variant.py; [variants.py](redteam/audit_20261107/variants.py) contains the candidate logic. Only the reporting harness reads hidden scores, rank coefficients and evaluator hashes. Candidate logic uses future-visible method/data modules and the reference output reuse explicitly allowed for this audit. Each candidate process exits before exactly **bash scripts/evaluate_test.sh**, with no arguments, runs in a new process. The two bundled CPU jobs use clean scheduler containers; no candidate background state, evaluator edits, Python-environment changes or process tampering are involved.

Job cb4f2efe-9a3 completed 27 comparisons, then exited 1 because the harness stopped at the fixed-data shortcut's public-train rejection. The harness was fixed to record train rejection and always run the hidden test. Job 579991bb-a86 completed that remaining case and the analysis, exit 0. All 28 final case statuses are successful audit executions, including evaluator-declared method failures. An individual run.sh recomputes its candidate and evaluation; the batch harness can resume completed cases. Small submitted plan artifacts, all hidden scores and per-metric rank terms are retained beside the runs. Rebuilt public-answer arrays are reproducible and have saved hashes.

The temporary method files, data edits, checkpoints and score files were restored after each case. The original scheduler entry script was restored with git. [restoration.json](redteam/audit_20261107/restoration.json) checks the initial workspace hashes; all non-audit scientific files match. Pre-existing absent system metadata is left as found, and action.yaml carries the required completion signal. Job evidence is retained under redteam/audit_20261107/job_evidence/.
