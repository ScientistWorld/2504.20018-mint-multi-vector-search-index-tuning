# Scientist Constraints

## Data
Use the provided deterministic `naive_small` workload generator and its default evaluation setting unless adding a clearly labeled extra experiment. Do not train or tune on a different dataset and report it under `naive_small`.

## Storage
For constrained methods, the number of used indexes must not exceed the workload storage budget. Reporting a lower cost by using the PerQuery-style extra indexes is not a valid improvement for the constrained `mint` method.

## Evaluation
Scores must be produced by `scripts/evaluate.sh` from saved artifacts in `checkpoints/`. Evaluation code must remain independent from `method/`.

## Compliance Checklist
- [ ] `naive_small` uses the provided generator and default recall/top-k settings
- [ ] constrained method uses no more than the storage budget
- [ ] `scripts/evaluate.sh` writes `scoring/scores.json`
- [ ] `eval/` does not import from `method/`
