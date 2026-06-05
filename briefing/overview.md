# Overview

The paper proposes MINT, a multi-vector search index tuner. Instead of creating only one index per vector column or one full index per query, MINT can recommend shared multi-column indexes and query plans that retrieve fewer candidates while maintaining recall and storage constraints.

This reproduction packages a compact semi-synthetic benchmark, a FAISS-HNSW implementation of MINT's query planner and beam-search configuration tuner, PerColumn and PerQuery baselines, and a reusable evaluator that scores cost, recall, storage, and speedup.
