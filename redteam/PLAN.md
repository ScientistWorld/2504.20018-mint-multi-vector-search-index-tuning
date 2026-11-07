# Audit plan and initial inspection

- Initial commit: `2ebb367`.
- Canonical submitted workload: 3,000 rows, k=100, recall floor 0.9,
  three indexes. The method CLI default is 8,000, but the reproduced artifact
  and `scripts/reproduce.sh` use 3,000. All reference/quality checks use 3,000.
- The public generator exposes all four exact query vectors and their seed;
  public train scores q1/q4, while test scores q2/q3.
- Initial test reference retrieves 90 candidates with 0.9 recall on each
  scored query, using that query's full multi-column index.
- Test rebuilds ANN indexes and recomputes recall/cost, but accepts dataset
  size, per-column normalization plans, and configuration lists from artifacts.
- Rank is computed against initial `scoring/scores_test.json`, with metric
  coefficients and experiment weights from `scoring/reference.json`.
- Genuine quality check: exhaustive three-index configuration search and
  measured-union planning on all four queries. The harder q4 is only in train.
- Weak check: replace the submitted method by PerColumn. Additional honest
  quality tradeoff: more retrieval and higher recall, with unchanged indexes.
- Seed check: original procedure with HNSW seeds 12345 (FAISS default), 7, 19.
  Workload data seed stays fixed, as it does in evaluation. Cache only exact
  repeated ranks and identical planner invocations; no algorithm replacement.
- Shortcuts: public answer reconstruction/drop unscored queries, forged metric
  fields, empty outputs, one-candidate retrieval, 89% recall, negative retrieval
  counts, empty storage declaration, excessive true storage, inflated baseline,
  four-row metadata, and replacement of the fixed public data generator.
- Every case has a run script and an installed method wrapper. No hidden
  evaluator contents or reference scores are used by submitted implementations.

Initial unrelated workspace differences were already present: README.md and
manifest.json are system-masked character devices and trajectory logs are
absent. These are recorded in `original/initial_status.txt` and left untouched.
The temporary job entry point and action file will be restored at completion,
with the final done signal as required by the protocol.
