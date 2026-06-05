# Evaluation Protocol

Evaluation reads a saved method artifact from `checkpoints/` and writes `scoring/scores.json`. It does not import from `method/` or rerun the tuner. Instead, it reconstructs the deterministic workload, executes the saved query plans, and computes cost, recall, and storage independently.

The main experiment is `naive_small`. It compares:

- `per_column`: one single-column index per vector column
- `per_query`: one index per query column set, reported as a lower-bound baseline that may exceed the storage budget
- `mint`: the proposed tuned configuration under the same storage budget as `per_column`

`scoring/reference.json` and `scoring/scores.json` record the paper-reported PerColumn-normalized MINT comparison. PerQuery is still computed by the reproduction and retained in the raw checkpoint artifact, but it is not included in standardized scores because the converted paper text reports it qualitatively as a lower bound without a stable numeric value to transcribe.

Metrics:

- `weighted_cost`: probability-weighted machine-independent cost proxy at recall@100; lower is better
- `mean_recall_at_k`: mean recall@k across queries; higher is better
- `min_recall_at_k`: worst query recall@k; higher is better
- `storage_indexes`: number of used indexes; lower is better and must respect the configured budget for constrained methods
- `speedup_vs_per_column`: `per_column weighted_cost / mint weighted_cost`; higher is better
