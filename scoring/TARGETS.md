# Evaluation Targets

Both experiments (`naive_small`, `bicomplex_small`, weight 0.5 each) are scored on held-out query instances of every template (`scoring/scores_test.json`); the visible train slice (`scores_train.json`) uses the public training instances with the same rules.

## Primary: naive_small, bicomplex_small
Lower probability-weighted query cost on the workload.
- **Metric**: `weighted_cost` (lower is better; normalized by the evaluator's fixed PerColumn plans)

## Constraints: naive_small, bicomplex_small
The lower cost must keep every template's mean recall@100 at or above 0.90 on held-out queries and use no more distinct indexes than the budget (3 / 5). A violation fails the run.
- **Metrics**: `min_recall_at_k`, `mean_recall_at_k` (higher is better), `recall_floor_violation` (must be 0), `storage_indexes` (lower is better), `storage_budget_violation` (must be 0)

## Benefit ratio: naive_small, bicomplex_small
Improvement over the one-index-per-column approach on the same queries.
- **Metric**: `speedup_vs_per_column` (higher is better)

## Diagnostics (tie-breakers): naive_small, bicomplex_small
Retrieval pressure behind the cost.
- **Metrics**: `mean_candidate_multiplier_at_k`, `max_candidate_multiplier_at_k` (lower is better)
