# Scientist Constraints

## Data
Use the provided deterministic workload generator and default evaluation settings for `naive_small` and `bicomplex_small`. Do not train or tune on a different dataset and report it under either standardized experiment name.

## Storage
For constrained methods, the number of used indexes must not exceed the workload storage budget. Reporting a lower cost by using the PerQuery-style extra indexes is not a valid improvement for the constrained `mint` method.

## Evaluation
Scores must be produced by `scripts/evaluate.sh` from saved artifacts in `checkpoints/`. Evaluation code must remain independent from `method/`.

## Compliance Checklist
- [ ] `naive_small` uses the provided generator and default recall/top-k settings
- [ ] `bicomplex_small` uses the provided generator and default recall/top-k settings
- [ ] constrained method uses no more than the storage budget
- [ ] `scripts/evaluate.sh` writes `scoring/scores.json`
- [ ] `eval/` does not import from `method/`
