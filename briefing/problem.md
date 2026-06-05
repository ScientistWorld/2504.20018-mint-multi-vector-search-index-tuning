# Problem Definition

The task is multi-vector search index tuning. A database row represents one item and contains several vector-valued columns. A query names a subset of columns and provides one query vector per selected column. The query result is the top-k rows by the sum of cosine similarities over the selected columns.

Given a workload of such queries with probabilities, choose a set of indexes that minimizes probability-weighted query cost while satisfying two constraints:

- each query must meet a recall@k threshold against exact full-scan top-k results
- the number of created indexes must not exceed the storage budget

The benchmark uses a deterministic semi-synthetic multi-vector workload shaped after the paper's Naive setting: three vector columns, four queries, recall@50, and a storage budget of three indexes.
