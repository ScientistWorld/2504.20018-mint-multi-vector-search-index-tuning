# Overview

The paper proposes MINT, a multi-vector search index tuner. Instead of creating only one index per vector column or one full index per query, MINT can recommend shared multi-column indexes and query plans that retrieve fewer candidates while maintaining recall and storage constraints.

This reproduction packages two compact semi-synthetic workloads shaped after the paper's Naive and BiComplex workloads, a FAISS-HNSW implementation of MINT's query planner and beam-search configuration tuner, the PerColumn and PerQuery baselines, and an evaluator that executes submitted per-template plans on held-out queries and scores cost, recall, storage and speedup, failing any run that breaks the recall floor or the storage budget.
