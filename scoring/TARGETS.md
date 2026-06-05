# Evaluation Targets

## Primary: naive_small
Tests whether an index-tuned multi-vector search system lowers weighted query cost on a Naive-style workload.
- **Metric**: `weighted_cost` (lower is better)

## Constraint: naive_small
Tests whether the lower cost is achieved while maintaining recall and respecting the fixed storage budget.
- **Metrics**: `mean_recall_at_k` and `min_recall_at_k` (higher is better), `storage_indexes` (lower is better)

## Benefit Ratio: naive_small
Tests whether the tuned system improves over the one-index-per-column baseline.
- **Metric**: `speedup_vs_per_column` (higher is better)

## Additional Setting: bicomplex_small
Tests whether the same evaluation target holds on a wider five-column workload.
- **Metrics**: `weighted_cost` (lower is better), `mean_recall_at_k` and `min_recall_at_k` (higher is better), `storage_indexes` (lower is better), `speedup_vs_per_column` (higher is better)
