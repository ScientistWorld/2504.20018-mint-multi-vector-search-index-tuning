# Experiment Overview

Both experiments ask the same question on a different workload: can a system choose a fixed set of vector indexes (within a storage budget) and per-query-template retrieval plans that make the workload's queries cheaper than answering each query from one index per column, while every query template keeps recall@100 >= 0.90 on queries it has never seen? Each experiment has weight 0.5.

## naive_small
A small workload (three columns, four query templates, budget of three indexes) shaped after the paper's case-study workload. One template is a single-column query, so part of the workload cannot benefit from any layout; the others reward layouts that serve multi-column queries. It establishes whether the cost reduction over the one-index-per-column baseline appears at all under the storage and recall constraints, and whether the plans generalize from the visible training queries to held-out queries.

## bicomplex_small
A larger workload (five columns, eight overlapping query templates over 2-5 columns, budget of five indexes) shaped after the paper's complex binomial workload. There are more templates than indexes, so indexes have to be shared across templates and the choice of configuration matters. It establishes whether the cost reduction holds when the storage budget forces trade-offs between templates, which is where most of the remaining room for improvement lies.

In both experiments the primary metric is the probability-weighted query cost normalized by the evaluator's fixed one-index-per-column plans on the same held-out queries; recall and storage are hard constraints (violations fail the run).
