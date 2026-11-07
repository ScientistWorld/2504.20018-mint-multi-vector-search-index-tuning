# Problem Definition

The task is multi-vector search index tuning. A database row represents one item and contains several vector-valued columns. A query names a subset of columns (its *template*) and provides one query vector per named column. The query result is the top-k rows by the sum of cosine similarities over the named columns.

A workload is a set of query templates with probabilities. Given the database and the workload, choose

- a configuration: a set of ANN indexes, each built over the concatenation of one or more columns, and
- a plan per query template: which of the configuration's indexes to scan and how many items (`ek`) to retrieve from each; the union of retrieved items is re-ranked with the exact score,

so that the probability-weighted query cost is minimal while

- every template meets a recall@k floor (mean recall over that template's queries, measured against the exact full-scan top-k), and
- the number of distinct indexes used by the plans does not exceed the storage budget.

Plans are per template, so they must serve *unseen* queries of that template: the visible data provides a sample of training queries per template, and scoring uses held-out queries drawn from the same per-template distribution.

Two deterministic semi-synthetic workloads are provided (`data/synthetic_multivector.py`):

| workload | rows | columns (dims) | templates | k | recall floor | storage budget |
|---|---:|---|---:|---:|---:|---:|
| `naive_small` | 3,000 | 3 (32, 40, 48) | 4: {1}, {1,2}, {2,3}, {1,2,3} | 100 | 0.90 | 3 |
| `bicomplex_small` | 3,000 | 5 (16, 20, 24, 28, 32) | 8 subsets with >= 2 columns | 100 | 0.90 | 5 |

Each template has 200 visible training queries (`make_workload(name).train_queries`).

Cost is a machine-independent latency proxy: for each query, the sum over scanned indexes of `index dimension x ek`, plus `query dimension x number of distinct retrieved items` (re-ranking).
