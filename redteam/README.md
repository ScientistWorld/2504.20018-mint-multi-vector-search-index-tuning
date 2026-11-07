# Scoring audit

Run `bash redteam/run_all.sh` in the gym container (CPU only). Each case also
has a standalone `run.sh`. Runs install a submission under `method/`, produce
`checkpoints/mint_results.json`, run the visible evaluator, and then invoke
`bash scripts/evaluate_test.sh` **with no arguments in a separate process**.
No evaluator code or Python environment is modified. The fixed-data shortcut
changes the public generator; the other cases change only submission artifacts.
The harness restores its temporary changes after each run.

`original/` holds immutable audit fixtures. Reference scores are used only by
the reporting harness, never by a submitted method. Output-only shortcuts reuse
the reference's computed plans, as permitted by the audit request. The visible
answer shortcut rebuilds the workload and ground truth from public `data/` and
uses the public train evaluator's HNSW class.

The reference seed runs execute the original algorithm, with memoization of
identical query-planning calls and rank computations, and explicit HNSW random
seeds. They do not change the generated dataset, planner, beam settings, or
evaluation settings. The genuine better variant performs exhaustive search
under the original three-index budget, using measured candidate unions in the
stated cost objective. The weaker variant removes multi-column indexing.

`rank.json` in each case records the weighted sum of coefficient times metric
improvement over the frozen initial test reference. `scores_test.json` is the
unmodified hidden evaluator output. `summary.json` aggregates completed runs.
