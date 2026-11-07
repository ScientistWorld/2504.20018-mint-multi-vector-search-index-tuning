# Research Direction

## Research Problem
This environment studies index tuning for multi-vector search workloads. The goal is to choose reusable indexes and per-template query plans that lower retrieval cost while keeping recall on unseen queries and staying within a storage budget.

## Core Contribution
The paper's contribution is a workload-aware index tuner for multi-vector search. Improvements should stay in this family: candidate-index generation, query planning, cost and recall estimation, and configuration search.

## Approach Scope
Scientists may improve the planner, the cost and recall estimators (including how they account for uncertainty on unseen queries), the search strategy, candidate pruning, or how training queries are used, as long as the system still solves the same constrained multi-vector index tuning problem and submits per-template plans.

## Out of Bounds
Replacing the problem with exact full scans, using more indexes than the budget, changing the benchmark data or protocol under the same experiment name, or lowering cost by letting recall fall below the floor.
