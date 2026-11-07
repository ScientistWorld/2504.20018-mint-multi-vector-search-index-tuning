# Hardening report

The previous gym failed the scientific-quality check: on its hidden test, lowering recall below the required 90% raised the rank, and so did inflating the submitted baseline, declaring fake storage, shrinking the dataset or submitting negative retrieval depths. A stronger planner tied the reference, and the test queries could be regenerated from public code. After hardening, better plans score higher, worse plans and train-tuned plans score lower, and every major shortcut fails or ties.

All "after" numbers come from job `949f7afa-637` (pass 2; CPU, FAISS 1.14.2, numpy 2.4.6), rerun from the final method code, and are recomputed with the final coefficients. A final run of `scripts/run.sh` from the committed code (job `e1a26fb6-6be`, pinned packages) regenerated both reference artifacts and `scores_train.json` / `scores_test.json` / `scores.json` byte-identically (only `runtime_seconds` differs). Evidence for every case (submitted artifacts, train/test scores, `rank.json`) is in [`redteam/hardening/cases/`](redteam/hardening/cases/), summarized in [`redteam/hardening/summary.md`](redteam/hardening/summary.md). The driver is [`redteam/hardening/run_hardening.py`](redteam/hardening/run_hardening.py). "Before" numbers are the redteam's ([`REDTEAM.md`](REDTEAM.md)). Rank = Σ_experiments weight × Σ_metrics coefficient × (case − reference), reference row = `mint` in `scoring/scores_test.json`.

## What changed

1. **Held-out test queries.** A workload query is now a *template* (column subset + probability). Methods plan from 200 public training instances per template. The test executes the submitted per-template plans on 500 held-out instances per template, generated only in `eval/test/hidden_workload.py` from a seed that exists only there. The old test scored two of the four public query vectors, which anyone could regenerate from `data/`.
2. **Evaluator-owned protocol and data.** `eval/test/` has its own copy of the database generator and fixes rows, k, recall floor and budget. Artifact metadata (`n_rows`, `storage_budget`, `configuration`, reported metrics) is ignored. `eval/test/` does not import `data/`. `eval/train/` refuses to score if the public generator no longer matches the protocol.
3. **Evaluator-owned comparator.** PerColumn (and the over-budget PerQuery reference) are frozen plan files in `eval/{train,test}/baselines.json`, planned once by the reference planner. Artifact entries named `per_column`/`per_query` are ignored.
4. **Plan validation.** Unknown or out-of-template indexes, duplicates, non-integer, negative or `> n_rows` retrieval depths, and missing templates make the plan invalid.
5. **Storage is derived from the plans:** distinct indexes with `ek > 0` across all templates.
6. **Feasibility gate.** An invalid plan, any template with mean recall@100 below 0.90, or storage above the budget scores the run as *failed* on that workload. Every metric is set to a value at least as bad as any feasible run can reach: cost max(100, 2× the largest cost any valid plan can incur) × PerColumn, speedup 0, recall 0, violation 0.9, storage ≥ budget, candidate multiplier 30. A method missing a workload is failed on it. A failed run therefore never outranks a feasible one; feasible runs rank between −1.5 and +0.1 here, failed ones at about −52 per failed workload.
7. **Headroom experiment** `bicomplex_small` (weight 0.5; `naive_small` 0.5): the paper's BiComplex-style workload, scaled to 5 columns, 8 overlapping templates and a budget of 5 indexes, where indexes must be shared.
8. **Noise.** Training instances went from 50 to 200 and held-out instances from 100 to 500 per template. The reference planner's recall allowance went from 0.02 to 0.03 (see "Pass 1" below). The evaluator and the method run FAISS single-threaded with fixed seeds, so scores are deterministic.
9. **Reproduction setup (method unchanged in substance).** Planning pools the ground truth of a template's training instances, as the paper's sample-based recall estimator. Algorithm 1 is vectorized with the paper's "derive the last ek" optimization. Beam-search stopping uses the paper's im = 5% (was 1%). The paper's di restriction for useful indexes is applied. A latent bug was fixed: the post-search recall repair could switch on an unused index and exceed the storage budget; it now only raises already-used indexes.

## Do better results score higher? (hidden test)

| Variant | Before | After: naive cost / bicomplex cost | After rank |
|---|---:|---|---:|
| Reference (MINT, paper settings) | 0 | 0.5723 / 0.4575 | 0 |
| **Better: measured re-rank** (`--rerank-measured 40`; picks among the 40 best configurations by cost measured on the training queries; the paper's estimator counts duplicate candidates) | — (old "stronger search": tie 0) | **0.5531** / 0.4575 | **+0.0822** |
| **Better: variance-aware allowance** (`--trim-deterministic`; zero-variance single-index plans drop the allowance, ek 93→91) | — | 0.5664 / 0.4550 | **+0.0068** |
| **Better: both** | — | 0.5482 / 0.4550 | **+0.0874** |
| Stronger search under the same estimated cost (beam 10, no early stop, swap local search) | 0 (tie) | 0.5723 / 0.4575 | 0 (tie: honest alternative, same configuration) |
| Higher recall (plans to 0.98; recall/cost trade-off) | −0.2522 | 0.6632 / 0.5974 | −0.4362 |
| Weaker: no multi-column indexes (PerColumn plans) | −3.3495 | 1.0 / 1.0 | **−1.4662** |
| Train-tuned: no recall allowance (plans to exactly 0.90 on train) | — | 0.5221 / **failed** (q4 0.8935) | **−52.13** (train slice: +0.29) |

Ordering is now as it should be: better > reference = honest alternative > trade-off > weaker ≫ train-tuned/failed. The measured re-rank finds `{x1, x1,2, x2,3}` on naive, where MINT's estimate picked `{x1, x2,3, x1,2,3}` and served q2 from `x1` at ek 1040. The train-tuned variant shows the instance split at work: it is the best method on train (+0.29) and fails the hidden test.

## Shortcuts (hidden-test rank vs the reference)

| Shortcut | Before | Fix | After |
|---|---:|---|---:|
| Visible answers (rebuild public test queries, drop the other plans) | 0 (tie) | test queries hidden in `eval/test/`; all templates scored | drop plans: **−104.22** (failed) |
| Forged metrics (PerColumn plans + fake cost/recall/storage fields) | −3.3495 | fields ignored | −1.4662 (scored as the PerColumn plans it contains) |
| Inflate the submitted baseline (ek 30,000) | **+177.39** | evaluator-owned frozen comparator | **0.0000** (no effect) |
| Storage declaration (empty configuration, budget 99) | **+0.90** | storage from executed plans; protocol fixed | **0.0000** (no effect) |
| Test size (`n_rows=4`, ek 4) | **+18.22** | protocol fixed by evaluator | **−104.22** (failed) |
| Replace the fixed data generator (4 rows) + ek 4 | **+18.22** | `eval/test` has its own data copy; `eval/train` rejects a mismatching generator | **−104.22** (failed; train scoring refuses) |
| Recall just below the floor (old 89%; now every ek ×0.95) | **+0.0224** | feasibility gate | **−104.22** (failed) |
| One candidate per index | **+230.27** | feasibility gate | **−104.22** (failed) |
| Negative ek appended | **+828.96** | plan validation | **−104.22** (failed) |
| Empty plans | −2.2985 | gate | −104.22 (failed) |
| Over storage (PerQuery plans) | −0.90 | gate | −104.92 (failed) |
| New: fractional ek | — | validation | −104.22 (failed) |
| New: submit only one workload | — | missing workload = failed | −52.23 |

All eight shortcuts that paid before now fail or have no effect.

## Shortcuts left (and why)

- **Training-sample shopping.** The query distribution is public, so a method can draw its own training queries with other seeds. Reference reruns on other training samples land at +0.074 and +0.123. A method that tries samples and keeps the luckiest on train could gain about 0.1, because train and test ranks track each other closely. Using more training data is also a legitimate improvement, and the effect is bounded by sampling noise. Closing it would mean hiding the distribution the method must generalize to. Left.
- **Calibrating the allowance against the hidden test by trial.** Tighter allowances pay when they still pass. A train-only check cannot reveal this, and repeated test submissions are outside the gym. The no-allowance variant shows that guessing too tight fails (−52). This is honest calibration headroom, not a loophole.
- **Reserved names.** An artifact method called `per_column`/`per_query` is silently not scored. It gains nothing.
- `validate.py` soft warning: `eval/test/hidden_workload.py` generates queries itself. This is intended: the held-out queries must live only in `eval/test/`.

## Reference score before and after

- **Before** (old test: public q2/q3 only): weighted cost 0.3855, speedup 2.594, recall 0.90/0.90, 3 indexes. The reference sat at the bound: exact-match indexes retrieving exactly 90 items.
- **After** (hidden test, 500 instances × every template):
  - `naive_small`: cost 0.5723, speedup 1.747, recall mean/min 0.933/0.930, 3 indexes, candidate multipliers 3.30/10.4.
  - `bicomplex_small`: cost 0.4575, speedup 2.186, recall 0.929/0.916, 5 indexes, multipliers 4.32/13.57.
  - Train slice: 0.5727 / 0.4578, so train ≈ test for an honest method.
- The reproduced claim holds in the paper's direction on both workloads: lower cost than PerColumn at the same storage and recall floor (the paper shows 2.1–7.2× on HNSW). PerColumn ranks −1.466 relative to the reference.
- The paper's Naive case has a cost ratio of about 0.33 (Fig. 7C) and BiComplex about 0.33. The scaled reproduction reaches 0.57 / 0.46. On naive, one of four templates is single-column and carries probability 0.63, so no layout can help it.

## Headroom check

| Metric | Reference (naive / bicomplex) | Bound? |
|---|---|---|
| `weighted_cost` | 0.572 / 0.458 | Not at a bound. Measured re-rank reaches 0.548 on naive. PerQuery (over budget, `_raw`) reaches 0.312 / 0.180. On bicomplex, q7/q8 are served from the single-column `x1` at ek 1633/822, costing 1.9×/1.7× PerColumn's own q7/q8 plans. |
| `speedup_vs_per_column` | 1.747 / 2.186 | Not at a bound (reciprocal of cost). |
| `mean_recall_at_k`, `min_recall_at_k` | 0.933–0.930 / 0.929–0.916 | Above the 0.90 floor, below 1. |
| `recall_floor_violation`, `storage_budget_violation` | 0 / 0 | At the lower bound by construction: constraint metrics, 0 for every feasible run. |
| `storage_indexes` | 3 / 5 | At the budget ceiling, not at the metric's lower bound. |
| candidate multipliers | 3.30–10.4 / 4.32–13.57 | Not at a bound. |

The old test had the reference at the bound of every benefit metric. The added experiment is `bicomplex_small`, scaled from the paper's BiComplex empirical study, where the storage budget forces index sharing. The reference and both baselines were run on it: PerColumn 1.0, PerQuery 0.180 over budget, MINT 0.4575. The demonstrated gain there is small: 0.4575→0.4550 with the variance-aware allowance; other training samples give 0.449/0.442. Larger structural gains (the q7/q8 plans above) are visible but not demonstrated by a variant.

## Seed noise

Scores are deterministic for a fixed training sample (single-threaded FAISS, fixed seeds). Across three training-instance samples, the reference's test rank is 0, +0.074 and +0.123 (spread 0.12). The paper's effect, multi-column indexes vs PerColumn, is 1.47 rank units, about 12× the spread.

The demonstrated better variant (+0.08) is a systematic, deterministic improvement on the same public training sample (lower cost, both constraints met), not a lucky draw. It is still of the same size as the training-sample spread, so differences under about 0.1 between methods that use different training samples should not be over-read.

Pass 1 (`redteam/hardening/pass1/`) had 50 training and 100 held-out instances and a 0.02 allowance. There, the frozen PerColumn plans and both reference reruns on other samples failed the hidden recall floor: train→test drift for multi-index plans was about −0.01 ± 0.013. That is why the samples were enlarged and the allowance raised to 0.03. In pass 2 every reference run passes, with worst template recall 0.916 against the 0.90 floor.

## Weights and coefficients

- **Experiment weights: 0.5 / 0.5.** The paper reports all workloads side by side (Fig. 7). Naive alone has little headroom.
- **`storage_indexes`: −0.3 → −0.05; candidate multipliers: −0.3/−0.2 → −0.02/−0.01.** The paper minimizes probability-weighted latency at a *fixed* storage budget. Storage is a constraint, and candidate counts are already part of cost. With the old coefficients, dropping one index was worth about 6% cost, and the unweighted max-multiplier (a 9%-probability template) moved the rank more than the cost itself:
  - the higher-recall variant, which costs 16% more, nearly tied the reference (−0.019);
  - most of the measured re-rank's +0.83 came from multipliers.

  With the new values, cost and speedup dominate, and the other terms only break ties. Recall coefficients stay at 0.5/0.5, a small reward for margin above the floor, below the cost terms. The violation coefficients are unchanged (they only apply to failed runs).
- Paper values in `scoring/reference.json` are unchanged for `naive_small`. The description now notes the Figure 7 readings. `bicomplex_small` uses values read from Fig. 7 (approximate, scaled setup noted).

## Not fixed / caveats

- **Small headroom demonstration on `bicomplex_small`:** +0.5% by a demonstrated variant. The candidate improvements are visible, but no variant here realizes them.
- **Recall gate stability:** the reference passes with a 0.016 recall margin at its tightest template. The gate is a hard cliff, so a future reference with a much smaller allowance would be fragile by design.
- **Scale:** 3,000-row synthetic data and FAISS HNSW instead of 1M-row public embeddings with HNSW/DiskANN. Magnitudes differ from the paper; only the direction is reproduced.
- **Pass-1 evidence** is kept for transparency. Its scores used the earlier settings and coefficients and are not comparable to the final numbers.
