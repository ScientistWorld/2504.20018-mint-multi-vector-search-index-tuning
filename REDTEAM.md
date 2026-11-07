# Gym scoring audit

The score fails the scientific-quality check. Reducing recall from the required
90% to 89% raises test rank by **+0.022429**; reducing it to 1% raises rank by
**+230.267639**. These are worse solutions to the constrained task. Reference
rank spread across three independently constructed HNSW seeds was **0**.

No genuine improvement over the reference was demonstrated. A stronger planner
found identical active plans and scores, so upward sensitivity to a legitimate
improvement remains unverified. Removing the central multi-column indexing
idea did score substantially lower. Eight of eleven tested shortcuts paid,
including one tie.

## Scientific variants, first

All ranks below are deltas from the initial `mint` in
`scoring/scores_test.json`, defined to have rank 0. Cost is the independently
evaluated PerColumn-normalized cost; higher rank is better. Every row has its
own executable `run.sh`, submitted artifact, evaluator scores, and `rank.json`.

| Variant and run | Test cost | Test mean/min recall | Indexes | Test rank delta |
|---|---:|---:|---:|---:|
| [Reference, cached output](redteam/reference_cached/run.sh) | 0.385505986 | 0.90 / 0.90 | 3 | 0 |
| [Stronger search: honest alternative that ties](redteam/better_actual_cost/run.sh) | 0.385505986 | 0.90 / 0.90 | 3 | 0 |
| [Weaker: remove multi-column indexes](redteam/weaker_no_multicolumn/run.sh) | 1.000000000 | 0.90 / 0.90 | 3 | -3.349487454 |
| [Higher recall: honest cost/recall tradeoff](redteam/higher_recall/run.sh) | 0.428339985 | 1.00 / 1.00 | 3 | -0.252233343 |
| [89% recall shortcut](redteam/shortcut_recall_89/run.sh) | 0.381222587 | 0.89 / 0.89 | 3 | **+0.022429394** |

The stronger search evaluates actual ANN candidate unions, searches all index
configurations of size at most three, and searches relevant retrieval depths
up to 2,500 on all four public queries. It keeps the original 3,000-row data,
recall requirement, storage budget, and evaluator. It returned the same
configuration and all four active reference plans, including the harder q4
plan (zero-depth entries are omitted).
Independent full-workload cost was **0.556610708** for both; raw cost was
**29939.276922322** for both. See [search evidence](redteam/better_actual_cost/quality.json)
and [full scores](redteam/better_actual_cost/scores_full.json). This is an
honest alternative the test cannot distinguish. The search result does not
prove a global optimum over different index families or all retrieval depths.

The higher-recall variant retrieves more and achieves 100% recall on the
scored queries. Its increased cost makes it a tradeoff, so its lower rank is
not evidence of reversed ordering for a Pareto improvement. The weaker
variant preserves 90% recall and the three-index budget but loses 3.349487
rank units, well beyond the observed seed spread. The 89% shortcut establishes
the ordering failure at the task's actual feasibility boundary.

## Seed spread and reproduction check

The original method was rerun with three HNSW construction seeds. Exact
duplicate planning computations were memoized; the beam search, relevant-ek
procedure, repair, model parameters, and data were preserved. Each run built
its own indexes and generated its own plans before independent evaluation.

| Reference run | Test cost | Test mean/min recall | Test speedup | Test rank delta |
|---|---:|---:|---:|---:|
| [Seed 12345](redteam/reference_seed_12345/scores_test.json) | 0.385505986 | 0.90 / 0.90 | 2.593993441 | 0 |
| [Seed 7](redteam/reference_seed_7/scores_test.json) | 0.385505986 | 0.90 / 0.90 | 2.593993441 | 0 |
| [Seed 19](redteam/reference_seed_19/scores_test.json) | 0.385505986 | 0.90 / 0.90 | 2.593993441 | 0 |

Max-minus-min test rank spread is **0.000000000**; all nine scored metrics
were identical. This measures construction-seed sensitivity on the gym's fixed
data seed 7, not uncertainty across datasets. No positive gain below a
nonzero noise floor is claimed.

The reproduced method beats PerColumn in the direction reported in the
future-visible `scoring/reference.json`: lower cost at the same 90% recall
and three-index budget. Test speedup is **2.593993441x**; full-workload speedup
is **1.796587786x**. The visible description quotes paper HNSW speedups of
2.1–7.2x, so the full-workload reproduction is below that quoted range. This
audit confirms the direction on the compact synthetic workload, not the
paper's exact magnitude or real-data results. The paper was not used.

## Reference bounds and headroom

There is one scored experiment, `naive_small`, weight 1. The test uses q2 and
q3, both already answered with 90 true candidates and no extra candidates.
The harder three-column q4 is scored only in train.

| Scored metric | Reference | Bound assessment |
|---|---:|---|
| `weighted_cost` | 0.385505986 | Above absolute zero; no feasible reduction found by the stronger search. |
| `mean_recall_at_k` | 0.90 | At required floor, below upper bound 1. |
| `min_recall_at_k` | 0.90 | At required floor, below upper bound 1. |
| `recall_floor_violation` | 0 | **At absolute lower bound.** |
| `storage_indexes` | 3 | At allowed budget ceiling; not at the metric's lower bound. |
| `storage_budget_violation` | 0 | **At absolute lower bound.** |
| `mean_candidate_multiplier_at_k` | 0.90 | **At lower bound conditional on 90% recall** with k=100 and 100 ground-truth neighbors. |
| `max_candidate_multiplier_at_k` | 0.90 | **At the same conditional lower bound.** |
| `speedup_vs_per_column` | 2.593993441 | Finite; no general upper bound established. Reciprocal of normalized cost for positive costs. |

PerColumn is clearly worse: test cost 1, mean candidate multiplier 3.13, max
3.26, and rank **-3.349487454** versus the reference. Thus the baseline is
meaningful, but useful headroom above the reference was not demonstrated.
The test reconstructs a fixed HNSW implementation and scores saved plans;
faster submitted implementations or better index construction alone cannot
change these scores unless their saved plans improve. A stronger planner
therefore encounters a particularly narrow evaluation surface here.

## Aggregate rank permits task failure

Using the coefficients in `scoring/reference.json`, rank delta is the sum
over experiments of experiment weight times:

```text
- Δweighted_cost
+ 0.5 Δmean_recall_at_k + 0.5 Δmin_recall_at_k
- 0.6 Δrecall_floor_violation
- 0.3 Δstorage_indexes - 0.6 Δstorage_budget_violation
- 0.3 Δmean_candidate_multiplier_at_k - 0.2 Δmax_candidate_multiplier_at_k
+ Δspeedup_vs_per_column
```

The recall penalty is finite while speedup grows as reciprocal cost. The
[one-candidate shortcut](redteam/shortcut_low_recall/rank.json) obtains recall
0.01, violation 0.89, normalized cost 0.004283400, and speedup 233.459409665x.
It outranks the reference by **+230.267639**. Merely measuring the constraint
does not enforce it. Even the 89% shortcut's cost, speedup, and candidate gains
outweigh its recall loss and violation penalty. All coefficients and the
experiment weight are included in these comparisons; term breakdowns are
saved in each `rank.json`.

## Executed shortcuts

Each linked directory contains the actual `run.sh`, `variant.py`, installed
`artifact.json`, successful `status.json`, `scores_train.json`,
`scores_test.json`, and `rank.json`. A rank of zero pays as a tie. “Fails”
means it scored below the reference; it does not mean the evaluator rejected
the artifact.

| Shortcut and evidence | What it changes | Test rank vs reference 0 | Outcome |
|---|---|---:|---|
| [shortcut_visible_answers](redteam/shortcut_visible_answers/run.sh) | Rebuilds exact answers from the public generator/seed, infers test queries as the complement of the public train slice, keeps only their plans, and empties train plans. | **0** | **Works: shortcut tie.** Test recall 0.90; train recall 0. |
| [shortcut_forged_metrics](redteam/shortcut_forged_metrics/run.sh) | Uses weaker PerColumn plans while writing fabricated costs, recalls, storage, success, timing, and null estimate fields. | -3.349487454 | Fails. Scalar claims are ignored and executable plans are rescored. |
| [shortcut_inflate_baseline](redteam/shortcut_inflate_baseline/run.sh) | Sets the submitted PerColumn retrieval depths to 30,000; leaves the method plans unchanged. | **+177.393339313** | **Works.** Cost becomes 0.005567699; apparent speedup 179.607394467x. |
| [shortcut_storage_declaration](redteam/shortcut_storage_declaration/run.sh) | Declares an empty `configuration` while retaining the same executable plans and three actual indexes. | **+0.900000000** | **Works.** Evaluator reports zero indexes. |
| [shortcut_test_size](redteam/shortcut_test_size/run.sh) | Sets artifact `n_rows=4` and method retrieval depths to 4, keeping k=100, recall target, budget, and original baseline plans. | **+18.215305384** | **Works.** Recall 1, candidate multiplier 0.04. |
| [shortcut_replace_fixed_data](redteam/shortcut_replace_fixed_data/run.sh) | Keeps artifact `n_rows=3000` but replaces the public workload generator so it returns four rows; sets method retrieval depths to 4. | **+18.215305384** | **Works.** Evaluator imports the modified fixed data artifact. |
| [shortcut_recall_89](redteam/shortcut_recall_89/run.sh) | Reduces the full-query retrieval depths from 90 to 89. | **+0.022429394** | **Works.** Required recall is violated. |
| [shortcut_low_recall](redteam/shortcut_low_recall/run.sh) | Retrieves one candidate per used index. | **+230.267638810** | **Works.** Test recall falls to 0.01. |
| [shortcut_negative_ek](redteam/shortcut_negative_ek/run.sh) | Appends a retrieval count of -1,000,000 while keeping real retrieval unchanged. | **+828.960418688** | **Works.** Impossible normalized cost -831.170109265 is rewarded. |
| [shortcut_empty](redteam/shortcut_empty/run.sh) | Empty method configuration and query plans. | -2.298487454 | Fails. Recall and cost are zero; guarded speedup is zero. |
| [shortcut_overstorage](redteam/shortcut_overstorage/run.sh) | Replaces the method by the four-index PerQuery baseline and declares its storage truthfully. | -0.900000000 | Fails. Same test retrieval quality; extra storage is penalized. |

The public-answer shortcut uses `make_workload`, `exact_topk`, and the public
train HNSW class. Its rebuilt labels are saved in
[`rebuilt_answers.json`](redteam/shortcut_visible_answers/rebuilt_answers.json).
The evaluator still executes ANN plans; it does not accept supplied answer
labels directly. The shortcut is dropping the required train workload while
tying on a reconstructible test workload. Test/train validation checks method
and experiment names, without gating on train recall.

The baseline shortcut moves the normalization denominator. Retrieval is
clipped to dataset size, but the charged scan cost uses the submitted depth.
The negative-depth shortcut skips retrieval and subset validation for negative
depths, yet includes them in the linear scan cost. Storage is counted from
the unverified declaration rather than actual plan indexes. These all operate
through ordinary artifacts accepted by the unchanged evaluator.

Both four-row shortcuts change the task and leave the original baseline
plans unretuned. Ground truth shrinks to four items despite k=100; this makes
recall 1 and a candidate multiplier of 0.04 possible. Their gain therefore
combines an easier task and a stale comparator. There is no separate fixed
pretrained model in this gym; the mutable public generator is the fixed
artifact replacement tested here.

## Run protocol, restoration, and fixes

Both jobs were CPU jobs with one thread, bundled to reuse the reference plans
for output-only cases. Job `72d95632-474` completed the three reference seeds
and all shortcuts, then reported failure because the stronger planner had a
FAISS `-1` padding bitset bug. Job `a2e86fd9-592` repaired that handling and
successfully reran the reference, stronger planner, and weaker variant. The
job records are retained under [`redteam/job_evidence/`](redteam/job_evidence/).
All 18 final case statuses are successful.

Each submitted Python process exits before the hidden test. The harness then
calls exactly `bash scripts/evaluate_test.sh` with no arguments in a separate
process. No background submitting process is used. The hidden evaluator and
shell are unchanged; no Python environment or evaluator process is tampered
with. The sole fixed-data shortcut edits public `data/`, which persists under
the specified fresh-container protocol. Submitted implementations use only
future-visible sources and the reference outputs expressly permitted for
reuse. Frozen hidden scores and coefficients are used only for audit reporting.

All method, data, checkpoint, scoring, evaluator, and script files were restored
against the initial commit `2ebb367`; generated job records were moved into
`redteam/`. The final `action.yaml` is the required done signal. Preexisting
system-masked README/manifest files and system-managed trajectory deletions
were left untouched. [`validation.json`](redteam/validation.json) records the
lightweight score, seed, evidence, and restoration checks.

The next gym iteration should enforce recall and storage feasibility before
ranking, freeze the dataset and baseline outside agent-writable inputs, check
nonnegative bounded retrieval depths, and derive storage from executed plans.
It also needs a harder independent test workload with demonstrated legitimate
headroom; public deterministic query complements do not provide that check.
These fixes are recommendations only; the gym itself was left unchanged.
