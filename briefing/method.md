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
