# Train/Test Evaluation Design Notes

Current design (final hardening). Earlier notes about a query-template split are superseded.

- **Unit of submission.** One artifact per workload with a plan (indexes, `ek`) for every query template. The evaluators execute the plans with a fixed FAISS HNSW engine and recompute every metric; nothing the artifact reports is trusted.
- **Train slice** (`eval/train/`, `scripts/evaluate_train.sh`): the 1000 public training queries per template from `data/synthetic_multivector.py` (seed 20240501). In-sample for any method that plans on them.
- **Test slice** (`eval/test/`, `scripts/evaluate_test.sh`): 1000 held-out queries per template from the same per-template distribution, generated only in `eval/test/hidden_workload.py`, which also holds its own copy of the database and protocol (no import of `data/`).
- **Comparator.** Frozen PerColumn plans in `eval/{train,test}/baselines.json`, planned once by the reproduced planner on the training queries; artifact entries named `per_column`/`per_query` are never scored.
- **Feasibility gate.** Invalid plans, an inconsistent artifact (declared protocol fields that differ from the protocol, invalid reserved entries, a declared configuration missing a scanned index), any template below recall@100 0.90, or storage above the budget fail the run with worst-case values on every metric.
- **Independence.** `eval/train/` and `eval/test/` share no code; neither imports `method/`.
- Honest methods score alike on both slices (reference: 0.5538 / 0.5536 naive, 0.4469 / 0.4467 bicomplex); plans tuned to the training queries without a generalization allowance fail the held-out recall floor.
