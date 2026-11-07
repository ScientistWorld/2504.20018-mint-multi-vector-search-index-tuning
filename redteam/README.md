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
evaluation settings. The attempted stronger planner in `better_actual_cost/`
searches all configurations under the original three-index budget and relevant
retrieval depths up to 2,500, using measured candidate unions in the stated
cost objective. It found the reference's active plans and costs: this is an
honest alternative that the test cannot tell apart, not a demonstrated quality
improvement. The weaker variant removes multi-column indexing. `higher_recall/`
is an honest recall/cost tradeoff, not a Pareto improvement.

`rank.json` in each case records the weighted sum of coefficient times metric
improvement over the frozen initial test reference. `scores_test.json` is the
unmodified hidden evaluator output. `summary.json` aggregates completed runs.

The final findings are in [REDTEAM.md](../REDTEAM.md). All 18 cases completed;
eight of the 11 shortcut cases pay, including the visible-answer shortcut's
tie. Run `python3 redteam/validate_audit.py` for lightweight evidence and
restoration checks; this does not rerun any scientific computation. The two
jobs' completion records are saved under `job_evidence/`.
