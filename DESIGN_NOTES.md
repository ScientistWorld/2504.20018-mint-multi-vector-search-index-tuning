# Train/Test Evaluation Design Notes

## Checks performed
- Confirmed `scripts/evaluate_train.sh` invokes `eval.train.evaluate` and writes `scoring/scores_train.json`.
- Confirmed `scripts/evaluate_test.sh` invokes `eval.test.evaluate` and writes `scoring/scores_test.json`.
- Confirmed both scripts enumerate `checkpoints/*_results.json` when no explicit artifacts are supplied, instead of hardcoding one method name.
- Confirmed both train and test evaluators read method result artifacts and score the submitted query plans, rather than regenerating the paper method inline.
- Checked for train/test score stubs; `scoring/scores_train.json` and `scoring/scores_test.json` are absent.
- Ran `bash -n` on both train/test evaluation scripts.
- Parsed `scoring/reference.json` as JSON.

## Changes made
- Added explicit recall and storage violation metrics to both train and test slices:
  - `recall_floor_violation`
  - `storage_budget_violation`
- Added candidate-pressure metrics to both slices:
  - `mean_candidate_multiplier_at_k`
  - `max_candidate_multiplier_at_k`
- Updated `scoring/reference.json` so every active metric has `higher_is_better`, `coefficient`, and `description`.
- Kept the current query-heldout split (`train`: `q1`, `q4`; `test`: `q2`, `q3`) but strengthened it with multi-metric triangulation. This should make a method that tunes visible plans by over-retrieving candidates or exceeding the storage budget show a clearer train/test gap.

## Runtime concerns for the runner iteration
- The split is still small: each slice has only two queries because `naive_small` has four query patterns. Watch for saturation in recall-related metrics; the candidate-pressure and explicit violation metrics were added to reduce that risk.
- The evaluators build FAISS HNSW indexes during scoring. This is method-independent and lightweight relative to training, but the runner should confirm the train/test scripts remain fast enough for repeated agent evaluations.
- `eval/test/evaluate.py` optionally compares method keys against `scoring/scores_train.json` if that file exists. This is useful for final consistency, but if the runner evaluates test without first evaluating train, the check is intentionally skipped.
