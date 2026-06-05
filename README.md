# Overview

The paper proposes MINT, a multi-vector search index tuner. Instead of creating only one index per vector column or one full index per query, MINT can recommend shared multi-column indexes and query plans that retrieve fewer candidates while maintaining recall and storage constraints.

This reproduction packages a compact semi-synthetic benchmark, a FAISS-HNSW implementation of MINT's query planner and beam-search configuration tuner, PerColumn and PerQuery baselines, and a reusable evaluator that scores cost, recall, storage, and speedup.

---

# Problem Definition

The task is multi-vector search index tuning. A database row represents one item and contains several vector-valued columns. A query names a subset of columns and provides one query vector per selected column. The query result is the top-k rows by the sum of cosine similarities over the selected columns.

Given a workload of such queries with probabilities, choose a set of indexes that minimizes probability-weighted query cost while satisfying two constraints:

- each query must meet a recall@k threshold against exact full-scan top-k results
- the number of created indexes must not exceed the storage budget

The benchmark uses a deterministic semi-synthetic multi-vector workload shaped after the paper's Naive setting: three vector columns, four queries, recall@50, and a storage budget of three indexes.

---

# Method

MINT tunes indexes for multi-vector search workloads. It considers candidate indexes over column subsets and asks a query planner how a query would use a proposed configuration.

The reproduced method implements the paper's main computational steps:

- build graph-based HNSW ANN indexes for single-column and multi-column vector concatenations
- compute exact ground truth for each workload query
- derive relevant extended-k scan depths from the ground-truth items' ranks in each candidate index
- enumerate relevant extended-k combinations for query plans and choose the minimum-cost plan meeting recall
- search configurations with a beam search under storage and recall constraints
- compare the chosen configuration against PerColumn and PerQuery baselines

The implementation uses FAISS CPU `IndexHNSWFlat` as the graph ANN backend. The paper used HNSW and DiskANN C++ implementations; FAISS HNSW preserves the graph-index behavior needed for this reproduction while avoiding a custom extension build.

---

# Evaluation Protocol

Evaluation reads a saved method artifact from `checkpoints/` and writes `scoring/scores.json`. It does not import from `method/` or rerun the tuner.

The main experiment is `naive_small`. It compares:

- `per_column`: one single-column index per vector column
- `per_query`: one index per query column set, reported as a lower-bound baseline that may exceed the storage budget
- `mint`: the proposed tuned configuration under the same storage budget as `per_column`

`scoring/reference.json` and `scoring/scores.json` record the paper-reported PerColumn-normalized MINT comparison. PerQuery is still computed by the reproduction and retained in the raw checkpoint artifact, but it is not included in standardized scores because the converted paper text reports it qualitatively as a lower bound without a stable numeric value to transcribe.

Metrics:

- `weighted_cost`: probability-weighted machine-independent cost proxy; lower is better
- `mean_recall_at_k`: mean recall@k across queries; higher is better
- `min_recall_at_k`: worst query recall@k; higher is better
- `storage_indexes`: number of used indexes; lower is better and must respect the configured budget for constrained methods
- `speedup_vs_per_column`: `per_column weighted_cost / mint weighted_cost`; higher is better

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 9m | **GPU**: 0.0h
- **Jobs**: 1 total (1 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-05 03:07] - none
- Started reproduction work.
- Read the paper and identified MINT's core components: HNSW/DiskANN candidate indexes, relevant-extended-k query planning, cost/recall constraints, and beam-search configuration tuning.

### [2026-06-05 03:07] - method_runs
- Implemented the paper's MINT computation with FAISS HNSW multi-column indexes, relevant-extended-k query planning, and beam-search configuration search.
- Implemented PerColumn and PerQuery baselines plus method-independent evaluation.
- Verified lightweight local smoke runs complete end to end.

### [2026-06-05 03:10] - core_claim
- Completed the 8000-row `naive_small` reproduction job successfully.
- MINT reached mean/min recall@50 of 0.90 with 3 stored indexes, satisfying the recall and storage constraints.
- MINT reduced weighted cost from PerColumn's 50562.84 to 16643.23, a 3.04x speedup/cost reduction.

</details>


---

# Reproduction Milestones

**Current: method_runs**

## Stop Justification
- Completed at milestone `method_runs`.
