# Research Direction

## Research Problem
This environment studies index tuning for multi-vector search workloads. The goal is to choose reusable indexes and query plans that lower retrieval cost while maintaining recall and storage constraints.

## Core Contribution
The paper's contribution is a workload-aware index tuner for multi-vector search. Improvements should stay in this family: candidate-index generation, query planning, cost/recall estimation, and configuration search.

## Approach Scope
Scientists may improve the planner, estimators, search strategy, candidate pruning, or graph-index execution details as long as the system still solves the same constrained multi-vector index tuning problem.

## Out of Bounds
Replacing the problem with exact full scans, using extra indexes beyond the storage budget for the constrained method, changing the benchmark data under the same experiment name, or optimizing only by relaxing recall is out of scope.
