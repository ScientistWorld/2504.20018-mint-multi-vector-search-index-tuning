# Method

MINT tunes indexes for multi-vector search workloads. It considers candidate indexes over column subsets and asks a query planner ("what-if") how each query template would use a proposed configuration.

The reproduced method (`method/mint.py`) implements the paper's computational steps:

- graph-based HNSW ANN indexes (FAISS `IndexHNSWFlat`) on single-column and multi-column vector concatenations;
- exact ground truth by full scan for the training queries of each template;
- relevant extended-k (ek) values: the rank of every ground-truth item in each candidate index; Algorithm 1 enumerates the relevant ek of all but one index and derives the minimal last ek meeting the recall target (plans over <= 3 indexes); Algorithm 2 (dynamic programming over the power set of k' = 5 sampled ground-truth items, several samples) for larger plans;
- the simplified cost estimator: per-index scan cost `dim * (80 + 1.25 * ek)` plus re-ranking cost `query dim * sum(ek)` (duplicates counted, as in the paper);
- Algorithm 3 beam search over configurations under the storage and recall constraints (candidate indexes per template with at least |q| - di columns, di = 2; seeds of <= se = 2 per-template candidates plus covering seeds; beam width 5; stop when the improvement is <= im = 5%), with plan caching;
- PerColumn and PerQuery baselines planned by the same planner.

Scaled-setup choices (not algorithm changes):

- Plans are per template and must generalize to unseen queries, so planning pools the ground truth of the template's 200 training queries (the relevant-ek grid is thinned to at most 100-256 values per index, as the paper bounds it by k).
- The recall estimator is the training-sample recall minus a fixed generalization allowance of 0.03 (plans target 0.93 on the training queries for a 0.90 floor).
- Chosen plans are checked against the actual HNSW results on the training queries and ek of already-used indexes is raised where the measured recall falls short.
- The paper used HNSW and DiskANN C++ libraries on 1M-row public embeddings; this reproduction uses FAISS HNSW on 3,000-row synthetic workloads.

Optional flags (off for the reference): `--rerank-measured K` re-ranks the K best configurations by cost measured on the training queries; `--trim-deterministic` drops the allowance for single-index plans whose recall is identical on all training queries; `--beam-width/--seed-limit/--improvement-threshold/--max-iters/--local-search` widen the search; `--recall-margin` and `--train-seed` change the allowance and the training sample.
