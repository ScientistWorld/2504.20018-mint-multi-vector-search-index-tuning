# Hardening report (final iteration)

The red-team audit (`REDTEAM.md`) found that better feasible results already scored higher, but two things were wrong:

- The only shortcut with a positive score was the **seed lottery**: rerun the unchanged reference on other training samples and keep the one with the lowest visible train cost. It scored **+0.123**, more than a genuine algorithmic improvement (measured re-rank, +0.082), and equal to the training-seed spread (0.123).
- Three shortcuts **tied** the reference: inflated baseline entries, a fake storage declaration and a rewritten protocol.

After this iteration:

- The seed spread is **0.042** (3× smaller).
- The seed lottery pays **+0.042**, well below the calibrated better variant (+0.149 to +0.268).
- The three tie shortcuts **fail** (−104.30).
- The reference keeps the paper's claim in the paper's direction: 1.81× (naive) and 2.24× (bicomplex) cheaper than PerColumn at the same storage budget and recall floor.

Rank = Σ over experiments of weight × Σ over metrics of coefficient × (case − reference). Weights and coefficients are in `scoring/reference.json`. The reference row is `mint` in `scoring/scores_test.json`. "Before" numbers are the red team's on the previous gym. "After" numbers are from jobs `366fde4c-042` and `8ce7e61e-228`. Evidence for every case (submitted artifacts, train and test scores, per-metric rank terms) is in [`redteam/final_hardening/cases/`](redteam/final_hardening/cases/), summarized in [`redteam/final_hardening/summary.md`](redteam/final_hardening/summary.md). The drivers are [`run_final.py`](redteam/final_hardening/run_final.py) and [`run_final2.py`](redteam/final_hardening/run_final2.py). Before and after ranks are each relative to their own gym's reference.

## What changed

1. **Training sample 200 → 1000 queries per template** (`data/synthetic_multivector.py`; pinned in `eval/train/evaluate.py`).
   - **Why it matters.** The seed spread came from a few plans that serve a multi-column template from one single-column index at large `ek`: naive q2 from `x1`, bicomplex q7 and q8 from `x1`, and the q1/q4 mixes. That `ek` is a 0.93-quantile of pooled ranks, estimated from the training sample. With 200 queries, naive q2's `ek` ranged from 928 to 1040 across seeds. With 1000 queries it ranges from 929 to 993.
   - **Scope.** This is a data-size change. MINT's algorithm, cost model, recall allowance (0.03) and search settings are unchanged.
2. **Held-out test 500 → 1000 queries per template** (`eval/test/hidden_workload.py`; the first 500 are the old ones). Recall-floor decisions and paired comparisons rest on twice as many instances.
3. **Comparator re-frozen** from the new reference run (`eval/{train,test}/baselines.json`). PerColumn and PerQuery are now planned by the same planner on the same 1000 training queries as the reference.
4. **Artifact consistency in both evaluators.** These are independent copies with no shared code. Every method in an artifact fails on that workload if:
   - the artifact declares `n_rows`, `top_k`, `recall_threshold` or `storage_budget` differently from the protocol; or
   - an entry under a reserved name (`per_column`, `per_query`) is not a valid plan set.

   A method that declares a `configuration` must include every index its plans scan. Storage = |declared configuration ∪ indexes used|, so a declaration can only add storage. Leaving these fields out is fine.
5. **Planner speed-up, results unchanged.** `method/mint.py:greedy_cover` now sorts each index's ranks once instead of on every step. At 1000 training queries the old loop made one bicomplex run take about 60 min, and over 75 min on one seed; it now takes about 20 min.
   - **Equivalence checks.** Job 2 self-tested the new function against the old one on 60 random cases (identical). It also reran the reference from the final code: every plan and configuration of `mint`, `per_column` and `per_query` equals the committed checkpoints ([`comparison.json`](redteam/final_hardening/cases/reference_rerun/comparison.json)).

## Do better results score higher? (hidden test)

| Variant | Before | After: naive cost / bicomplex cost (worst-template recall) | After rank |
|---|---:|---|---:|
| Reference (MINT, paper settings) | 0 | 0.5536 / 0.4467 (0.930 / 0.915) | 0 |
| **Better: calibrated** (allowance 0.015, measured re-rank of 40 configurations, variance-aware trimming; same training data) | +0.243 (with 800 queries) | 0.5164 / 0.4245 (0.910 / 0.910) | **+0.149** |
| Better: calibrated, training seed 101 | — | 0.4964 / 0.4160 | +0.268 (paired vs reference seed 101: **+0.231**) |
| Better: calibrated, training seed 202 | — | 0.5086 / 0.4222 | +0.214 (paired vs reference seed 202: **+0.172**) |
| Better: measured re-rank only (picks `{x1, x1,2, x2,3}` on naive) | +0.082 | 0.5509 / 0.4467 | +0.045 |
| Weaker: no multi-column indexes (PerColumn plans) | −1.466 | 1.0000 / 1.0000 (0.925 / 0.925) | **−1.539** |

The calibrated variant beats the reference on each of the three training samples, by **+0.149, +0.231 and +0.172** in paired comparisons (same training queries, same held-out queries). That is 3.5–5.5× the reference's seed spread of 0.042. The paper's effect (reference vs PerColumn, 1.539) is 37× the spread. Train ranks track test ranks: calibrated +0.149 / +0.149, weaker −1.535 / −1.539.

## Seed noise

| Reference training seed | Naive / bicomplex test cost | Test rank |
|---|---|---:|
| 20240501 (default, public) | 0.5536 / 0.4467 | 0 |
| 101 | 0.5398 / 0.4455 | +0.037 |
| 202 | 0.5355 / 0.4473 | +0.042 |

The spread is **0.042**; it was 0.123 with 200 training queries. For a fixed training sample the scores are deterministic (single-threaded FAISS, fixed seeds). The measured re-rank gain (+0.045) is a deterministic, structural configuration change on naive. It is about the size of the spread, so an unpaired comparison cannot separate it from training-sample luck (see "Not fixed").

## Shortcuts (hidden-test rank vs the reference)

| Shortcut | Before | Fix | After |
|---|---:|---|---:|
| Seed lottery (best of 3 training seeds by visible train cost; picks 202) | **+0.123** | 5× larger training sample shrinks the spread | **+0.042** (left; see below) |
| Inflate submitted baseline entries (`ek` 30,000) | **0 (tie)** | reserved entries must be valid plan sets | **−104.30** (failed) |
| Storage declaration (empty configuration, storage 0, budget 999) | **0 (tie)** | declared protocol must match; configuration must cover scanned indexes | **−104.30** (failed) |
| Rewrite protocol (k=1, rows=4, threshold 0, budget 999, fake GT) | **0 (tie)** | declared protocol must match | **−104.30** (failed) |
| Forged metrics (PerColumn plans, fake costs, empty configuration) | −1.466 | configuration check | −104.30 (failed) |
| Test size (`n_rows`=4, `ek` 4) | −104.22 | — | −104.30 |
| Below floor (every `ek` × 0.95) | −104.22 | — | −104.30 |
| Over storage (PerQuery plans) | −104.92 | — | −105.00 |
| Partial experiment (naive only) | −52.23 | — | −52.27 |
| Duplicate index alias with negative `ek` | −104.22 | — | −104.30 |
| Rank trade-off (raise one exact plan's `ek` 93 → 100) | −0.010 | — | −0.011 |
| Probe: edit the artifact's own baseline entries within the valid range | — | — | 0 (no-op: never scored) |

The cases not fixed in this iteration were replayed cheaply: they are transforms of the new reference artifacts. They still fail. Not replayed: the public-answers / zero-allowance run, the empty/zero/NaN/null/negative `ek` plans, constant plans, dropped templates, and fixed-data replacement. These failed in the red-team audit and nothing that rejects them has changed.

## Shortcuts left, and why

- **Seed lottery / training-sample shopping (+0.042).** The query distribution is public, so a method can draw its own training queries. Closing this would mean hiding the distribution a method must generalize to. Drawing more training queries is also a legitimate improvement. The payoff is now the size of the seed spread and well below a real improvement. The separate shortcut grader should flag selection on luck.
- **Valid edits to reserved baseline entries.** These have no effect (rank 0), because the comparator is frozen in `eval/`. Nothing is gained.
- **Calibrating the recall allowance against the hidden test by repeated submissions.** This is outside the gym; a single train-only check cannot reveal it. Guessing too tight fails the workload (about −52 per workload). Spending recall margin that still holds on unseen queries is what the paper's objective rewards (minimum cost subject to recall ≥ 0.90), so the calibrated variant is an honest improvement, not a loophole.
- `validate.py` soft warning: `eval/test/hidden_workload.py` generates the held-out queries itself. This is intended: the test queries must live only in `eval/test/`.

## Reference score before and after

- **Before** (200 training / 500 held-out queries; comparator planned on 200): naive cost 0.5723 (1.747×), bicomplex 0.4575 (2.186×), worst-template recall 0.930 / 0.916, storage 3 / 5.
- **After** (1000 / 1000; comparator re-planned on 1000):
  - Test: naive cost **0.5536** (1.806×), bicomplex **0.4467** (2.239×), worst-template recall 0.930 / 0.915, storage 3 / 5.
  - Train slice: 0.5538 / 0.4469.
  - Both experiments still show the paper's direction: lower cost than PerColumn at the same storage and recall floor. The paper reports 2.1–7.2× on 1M-row data with HNSW.
- `scoring/scores_{train,test}.json` and `scores.json` come from the reference artifacts alone (`checkpoints/mint_results.json`, `checkpoints/mint_bicomplex_results.json`). `scripts/run.sh` → `scripts/reproduce.sh` regenerates them.

## Headroom check

| Metric | Reference (naive / bicomplex) | At a bound? |
|---|---|---|
| `weighted_cost`, `speedup_vs_per_column` | 0.554 / 0.447; 1.81 / 2.24 | No. Calibrated reaches 0.496–0.516 / 0.416–0.425. PerQuery (over budget) costs 0.309 / 0.181. |
| `mean_recall_at_k`, `min_recall_at_k` | 0.932–0.930 / 0.929–0.915 | Above the 0.90 floor and below 1. |
| `recall_floor_violation`, `storage_budget_violation` | 0 / 0 | At the lower bound by construction: these are constraint metrics, 0 for every feasible run and worst-case for failed runs. |
| `storage_indexes` | 3 / 5 | At the budget ceiling, as for every tested feasible method. Not the metric's optimum: a method may use fewer indexes. |
| candidate multipliers | 3.18–9.93 / 4.22–13.59 | No. |

No benefit metric sits at its bound. `bicomplex_small` is the headroom experiment added by the previous iteration: scaled BiComplex, where the budget forces index sharing. Its demonstrated headroom is now 0.447 → 0.416–0.425. No further experiment was needed.

## Weights and coefficients

Unchanged: 0.5 / 0.5 per experiment. Cost and speedup dominate; recall above the floor (0.5 each) and storage and candidate counts are tie-breakers. The violation coefficients only act on failed runs. The rank terms behave as intended:
- The calibrated variant gains +0.083 on naive and +0.066 on bicomplex, while losing a little on recall margin.
- PerColumn loses −0.63 / −0.91.
- A failed workload costs about −52.

Only the `storage_indexes` description in `scoring/reference.json` was updated, to say that a declared configuration counts toward storage. The paper's reported values are unchanged.

## Not fixed / caveats

- **The measured re-rank's gain (+0.045) is the size of the seed spread (0.042).** It is a real, deterministic configuration improvement on naive, but its effect is too small to separate from training-sample noise without paired runs. The calibrated variant clearly exceeds the noise on every seed.
- **Tightest template.** On bicomplex, q7 is served from `x1` at `ek` ≈ 1670. There HNSW (efSearch 300, ef = `ek`) plateaus at measured recall 0.914 on train and 0.915 on test: below the planner's 0.93 target, but above the 0.90 floor. The recall gate is a hard cliff by design.
- **Runtime for future agents.** A reference run now takes about 3.5 min (naive) plus about 20 min (bicomplex) on one core. The training sample is 5× larger, and the `greedy_cover` speed-up offsets most of the cost.
- **Job 1 was cancelled at 4 h** because seed 101's bicomplex run was stuck in the old `greedy_cover` loop. Its completed stages (reference, freeze, scoring, weaker variant, shortcuts, calibrated variant) are kept as they are. The rerun from the final code reproduced the same reference plans.
- **Scale.** 3,000-row synthetic data and FAISS HNSW replace the paper's 1M-row public embeddings with HNSW/DiskANN. Only the direction of the paper's results is reproduced, not their magnitudes.
- **Older evidence.** The previous iteration's evidence (`redteam/hardening/`) and the red-team audit (`redteam/audit_20261107/`) describe earlier gyms. Their scores are not comparable to these.
