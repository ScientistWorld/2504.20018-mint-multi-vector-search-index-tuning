# Current gym audit (starting commit f5f1408)

Earlier redteam results apply to earlier scoring. These cases audit the current
frozen held-out workload, frozen comparators, and feasibility-gated scoring.

Run all cases in a CPU job with `python redteam/audit_20261107/harness.py all`.
Every case has a standalone `run.sh` and `variant.py`. The harness installs the
candidate as ordinary `method/` files, runs it to completion, and then invokes
exactly `bash scripts/evaluate_train.sh` and `bash scripts/evaluate_test.sh`,
both without arguments and in new processes. The submitting process terminates
before evaluation and the Python environment and evaluator remain untouched.
If the training evaluator rejects a shortcut, its exit code is recorded and
the hidden test still runs independently; stale training scores are discarded.
Each case restores all non-audit files in a finally block. Persistent file
changes under data/ are kept through evaluation for the fixed-data shortcut.

The scheduler supplies the clean container. Since candidates only emit ordinary
JSON files (and one appends text to the public data generator), their only
cross-process state is that persistent workspace. No process tricks are used.
The hidden evaluator is inspected only for audit design/reporting; candidate
code reads only method/, public data/, and the explicitly permitted cached
reference outputs. Ranking coefficients and reference scores are read only by
the reporting harness after hidden scoring finishes.

Every `shortcut_*` case is called a shortcut even when it ties by changing an
ignored field. A tie does not establish a metric forgery: the report separates
ignored-input shortcut ties from changes that could obtain a spurious gain.

Initial reference outputs/scores and workspace hashes are saved under initial/.
All case scores, artifacts, status, and coefficient-by-coefficient rank terms
are retained. Large rebuilt public answer arrays are ignored.
