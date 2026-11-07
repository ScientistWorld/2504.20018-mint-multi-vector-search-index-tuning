"""The reference algorithm, seeded, with exact duplicate computations cached."""
from __future__ import annotations

from pathlib import Path


def run(seed: int, output: Path):
    import faiss
    import numpy as np
    import method.mint as mint

    class SeededHnswIndex(mint.HnswIndex):
        def __init__(self, columns, col_ids, ef=300):
            self.col_ids = tuple(col_ids)
            self.dim = int(sum(columns[c].shape[1] for c in self.col_ids))
            matrix = np.concatenate([columns[c] for c in self.col_ids], axis=1).astype("float32")
            matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
            self.matrix = matrix
            self.index = faiss.IndexHNSWFlat(self.dim, 16, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = 120
            self.index.hnsw.efSearch = ef
            self.index.hnsw.rng = faiss.RandomGenerator(seed)
            self.index.add(matrix)
            self.intercept = 80.0
            self.slope = 1.25
            self._ranks = {}

        def rank_ground_truth(self, query, gt):
            key = (query.name, tuple(int(x) for x in gt))
            if key not in self._ranks:
                self._ranks[key] = super().rank_ground_truth(query, gt)
            return self._ranks[key]

    mint.HnswIndex = SeededHnswIndex
    original_plan_query = mint.plan_query
    cache = {}

    def memoized_plan_query(workload, query, gt, indexes, config, max_rank):
        useful = tuple(sorted(k for k in config if set(k).issubset(query.columns)))
        key = (query.name, useful, max_rank)
        if key not in cache:
            cache[key] = original_plan_query(workload, query, gt, indexes, useful, max_rank)
        return cache[key]

    mint.plan_query = memoized_plan_query
    mint.run_experiment("naive_small", 3000, output)
