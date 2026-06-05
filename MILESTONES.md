# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-06-05 03:07] - none
- Started reproduction work.
- Read the paper and identified MINT's core components: HNSW/DiskANN candidate indexes, relevant-extended-k query planning, cost/recall constraints, and beam-search configuration tuning.

### [2026-06-05 03:07] - method_runs
- Implemented the paper's MINT computation with FAISS HNSW multi-column indexes, relevant-extended-k query planning, and beam-search configuration search.
- Implemented PerColumn and PerQuery baselines plus method-independent evaluation.
- Verified lightweight local smoke runs complete end to end.

## Stop Justification
- Not stopped. An 8000-row CPU reproduction job is being submitted to test the core claim.
