"""Run each candidate, terminate it, then run the restored hidden test shell.

Only this reporting harness reads hidden scores, coefficients, or evaluator
hashes. Candidate code lives separately in variants.py and uses visible inputs.
All evaluator calls are exactly `bash scripts/evaluate_test.sh`, no arguments.
No mutation to evaluator code, its process, or the Python environment is made.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INITIAL = HERE / "initial"
EXPERIMENTS = ("naive_small", "bicomplex_small")
CASES = ["reference_cached", "reference_seed_20240501", "reference_seed_101", "reference_seed_202",
         "better_measured_cost", "better_more_queries", "weaker_no_multicolumn", "shortcut_seed_lottery",
         "shortcut_public_answers", "shortcut_forged_metrics", "shortcut_inflate_baseline",
         "shortcut_storage_declaration", "shortcut_rewrite_protocol", "shortcut_test_size",
         "shortcut_replace_fixed_data", "shortcut_rank_tradeoff", "shortcut_low_recall",
         "shortcut_below_floor", "shortcut_empty", "shortcut_constant", "shortcut_zero",
         "shortcut_negative_ek", "shortcut_nan_ek", "shortcut_null_ek", "shortcut_overstorage",
         "shortcut_drop_templates", "shortcut_partial_experiment", "shortcut_duplicate_index_alias"]


def rank(scores, method="mint"):
    reference = json.loads((INITIAL / "scoring/scores_test.json").read_text())
    spec = json.loads((INITIAL / "scoring/reference.json").read_text())
    terms = {}
    for name, config in spec["experiments"].items():
        actual = scores["experiments"][name]["results"][method]
        target = reference["experiments"][name]["results"]["mint"]
        terms[name] = {key: config["weight"] * metric["coefficient"] * (actual[key] - target[key])
                       for key, metric in config["metrics"].items()}
    return {"rank_delta": sum(sum(t.values()) for t in terms.values()), "terms": terms}


def run_command(cmd, log):
    with log.open("a") as fh:
        fh.write("\n$ " + " ".join(cmd) + "\n")
        fh.flush()
        subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, check=True)


def run_case(case):
    if case not in CASES:
        raise ValueError(case)
    out = HERE / case
    out.mkdir(exist_ok=True)
    if (out / "status.json").is_file() and json.loads((out / "status.json").read_text()).get("ok"):
        print(f"Skip completed {case}", flush=True)
        return
    saved = {p: p.read_bytes() for p in (ROOT / "checkpoints").glob("*_results.json")}
    saved.update({ROOT / f"scoring/scores_{sl}.json": (ROOT / f"scoring/scores_{sl}.json").read_bytes()
                  for sl in ("train", "test")})
    data_path = ROOT / "data/synthetic_multivector.py"
    saved[data_path] = data_path.read_bytes()
    installed = [ROOT / "method/audit_support.py", ROOT / "method/audit_variant.py"]
    protected = [*ROOT.glob("eval/**/*.py"), *ROOT.glob("eval/**/*.json"), ROOT / "scripts/evaluate_test.sh"]
    protected_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    start = time.time()
    ok = False
    error = None
    try:
        for p in list((ROOT / "checkpoints").glob("*_results.json")):
            p.unlink()
        installed[0].write_bytes((HERE / "variants.py").read_bytes())
        installed[1].write_bytes((out / "variant.py").read_bytes())
        run_command([sys.executable, "-m", "method.audit_variant"], out / "candidate.log")
        for p in (ROOT / "checkpoints").glob("*_results.json"):
            (out / p.name).write_bytes(p.read_bytes())
        # The submitting process is finished. Persistent workspace edits remain
        # in place; no submitting process can influence either evaluator.
        # A shortcut may break the public training protocol while the restored
        # hidden evaluator still accepts or rejects its persistent artifacts.
        # A training failure must not prevent that independent test from running.
        train_exit_code = 0
        try:
            run_command(["bash", "scripts/evaluate_train.sh"], out / "evaluation.log")
        except subprocess.CalledProcessError as exc:
            train_exit_code = exc.returncode
        train_scores = out / "scores_train.json"
        if train_exit_code == 0:
            train_scores.write_bytes((ROOT / "scoring/scores_train.json").read_bytes())
        else:
            train_scores.unlink(missing_ok=True)
        (out / "train_status.json").write_text(json.dumps({
            "exit_code": train_exit_code, "scores_saved": train_exit_code == 0,
        }, indent=2))
        run_command(["bash", "scripts/evaluate_test.sh"], out / "evaluation.log")
        (out / "scores_test.json").write_bytes((ROOT / "scoring/scores_test.json").read_bytes())
        scores = json.loads((out / "scores_test.json").read_text())
        result = rank(scores)
        result["status"] = {name: scores["experiments"][name]["_raw"]["mint"]["status"] for name in EXPERIMENTS}
        result["metrics"] = {name: scores["experiments"][name]["results"]["mint"] for name in EXPERIMENTS}
        result["evaluator_command"] = ["bash", "scripts/evaluate_test.sh"]
        result["fresh_process"] = True
        result["train_exit_code"] = train_exit_code
        result["protected_hashes"] = protected_hashes
        (out / "rank.json").write_text(json.dumps(result, indent=2))
        assert protected_hashes == {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
        ok = True
        print(f"{case}: {result['rank_delta']:+.9f}; {result['status']}", flush=True)
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
    finally:
        for p in list((ROOT / "checkpoints").glob("*_results.json")):
            p.unlink()
        for p, content in saved.items():
            p.write_bytes(content)
        for p in installed:
            p.unlink(missing_ok=True)
        (out / "status.json").write_text(json.dumps({"ok": ok, "seconds": time.time() - start, "error": error}, indent=2))
    if not ok:
        raise RuntimeError(f"case failed: {case}")


def summarize():
    rows = {}
    for case in CASES:
        p = HERE / case / "rank.json"
        if p.exists():
            rows[case] = json.loads(p.read_text())
    (HERE / "summary.json").write_text(json.dumps(rows, indent=2))
    lines = ["| case | test rank delta | statuses |", "|---|---:|---|"]
    for name, row in rows.items():
        lines.append(f"| {name} | {row['rank_delta']:+.9f} | {row['status']} |")
    (HERE / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    os.chdir(ROOT)
    if len(sys.argv) > 1 and sys.argv[1] == "all":
        failures = []
        for case in CASES:
            try:
                run_command(["bash", str(HERE / case / "run.sh")], HERE / "job.log")
            except Exception:
                failures.append(case)
            summarize()
        if failures:
            raise RuntimeError(f"failed audit cases: {failures}")
        run_command([sys.executable, str(HERE / "analyze.py")], HERE / "job.log")
    else:
        run_case(sys.argv[1])
        summarize()
