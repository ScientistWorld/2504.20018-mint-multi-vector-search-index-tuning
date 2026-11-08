# Scientist Constraints

## Data
Use the provided deterministic workloads (`data/synthetic_multivector.py`): database, query templates, probabilities, k = 100, recall floor 0.90 and storage budgets (3 for `naive_small`, 5 for `bicomplex_small`). Each template has 1000 visible training queries; the test uses 1000 unseen queries per template. Editing the generator does not change what the evaluator measures; the evaluator fixes its own copy of the database and protocol.

## Plans
Submit one artifact per workload with a plan for every template. Plans may only use indexes over the template's own columns, each listed once with an integer `ek` in `[0, 3000]`. The evaluator executes plans with its fixed HNSW engine; index construction or search settings cannot be changed by a submission.

## Artifact consistency
Declared protocol fields (`n_rows`, `top_k`, `recall_threshold`, `storage_budget`) must match the protocol or be omitted. Entries named `per_column` / `per_query` are never scored but must be valid plan sets if present. A declared `configuration` must list every index the plans scan; any extra indexes it lists count toward storage. An inconsistent artifact fails on that workload.

## Storage and recall
The number of distinct indexes with `ek > 0` across all template plans must not exceed the budget, and each template's mean recall@100 on held-out queries must be at least 0.90. Either violation fails the run on that workload; submitting only one workload fails the other.

## Evaluation
Scores come from `scripts/evaluate_train.sh` / `scripts/evaluate_test.sh` (and `scripts/evaluate.sh`) over `checkpoints/*_results.json`. Evaluation code must stay independent of `method/`. Method names `per_column` and `per_query` are reserved for the evaluator's baselines.

## Compliance Checklist
- [ ] artifacts for both `naive_small` and `bicomplex_small`
- [ ] a valid plan for every template
- [ ] declared protocol fields (if any) equal the protocol; declared configuration (if any) covers every scanned index
- [ ] no more indexes than the storage budget
- [ ] recall@100 >= 0.90 per template on unseen queries (train-slice recall alone is not enough)
- [ ] `eval/` does not import from `method/`
