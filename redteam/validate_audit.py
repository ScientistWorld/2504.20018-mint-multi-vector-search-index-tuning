"""Check saved audit evidence and restoration using only the standard library.

This is a lightweight report check, not an experiment or evaluator rerun.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
START = "2ebb367"
CASES = (
    "reference_cached",
    "reference_seed_12345", "reference_seed_7", "reference_seed_19",
    "better_actual_cost", "weaker_no_multicolumn", "higher_recall",
    "shortcut_recall_89", "shortcut_low_recall", "shortcut_empty",
    "shortcut_forged_metrics", "shortcut_storage_declaration",
    "shortcut_overstorage", "shortcut_inflate_baseline",
    "shortcut_negative_ek", "shortcut_visible_answers", "shortcut_test_size",
    "shortcut_replace_fixed_data",
)


def read(path):
    return json.loads(path.read_text())


def git(*args):
    return subprocess.check_output(["git", *args], cwd=WORK)


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def results(scores, method="mint"):
    return scores["experiments"]["naive_small"]["results"][method]


def active_plans(payload):
    return {
        query: [(index, ek) for index, ek in zip(plan["indexes"], plan["ek"]) if ek > 0]
        for query, plan in payload["results"]["mint"]["plans"].items()
    }


def main():
    spec = read(ROOT / "original/reference.json")
    reference = read(ROOT / "original/scores_test.json")
    summary = read(ROOT / "summary.json")
    assert set(CASES) == set(summary["cases"])
    checks = {}
    for case in CASES:
        folder = ROOT / case
        for name in ("run.sh", "variant.py", "artifact.json", "status.json",
                     "scores_train.json", "scores_test.json", "rank.json"):
            assert (folder / name).is_file(), (case, name)
        status = read(folder / "status.json")
        assert status["ok"] is True and status["case"] == case
        assert status["test_command"] == ["bash", "scripts/evaluate_test.sh"]
        run_script = (folder / "run.sh").read_text()
        assert f"python -m redteam.harness {case}" in run_script
        scores = read(folder / "scores_test.json")
        assert scores["_slice"] == "test"
        assert scores["_split_queries"] == ["q2", "q3"]
        train = read(folder / "scores_train.json")
        assert train["_slice"] == "train"
        assert train["_split_queries"] == ["q1", "q4"]
        total = 0.0
        recorded = read(folder / "rank.json")
        for experiment, config in spec["experiments"].items():
            target = reference["experiments"][experiment]["results"]["mint"]
            actual = scores["experiments"][experiment]["results"]["mint"]
            for metric, entry in config["metrics"].items():
                assert math.isfinite(actual[metric])
                term = config.get("weight", 1) * entry["coefficient"] * (actual[metric] - target[metric])
                assert close(term, recorded["terms"][experiment][metric])
                total += term
        assert close(total, recorded["rank_delta"])
        assert close(total, summary["cases"][case]["rank_delta"])
        assert results(scores) == recorded["metrics"]
        checks[case] = {"ok": True, "rank_delta": total}

    seeds = [case for case in CASES if case.startswith("reference_seed_")]
    spread = max(checks[case]["rank_delta"] for case in seeds) - min(checks[case]["rank_delta"] for case in seeds)
    assert spread == summary["seed_spread"] == 0.0
    seed_metrics = [results(read(ROOT / case / "scores_test.json")) for case in seeds]
    assert all(metrics == results(reference) for metrics in seed_metrics)
    metric_spreads = {metric: max(m[metric] for m in seed_metrics) - min(m[metric] for m in seed_metrics)
                      for metric in results(reference)}

    for case in ("reference_cached", "better_actual_cost", "weaker_no_multicolumn"):
        assert (ROOT / case / "scores_full.json").is_file()
    for filename in ("scores_test.json", "scores_full.json"):
        assert results(read(ROOT / "reference_cached" / filename)) == results(read(ROOT / "better_actual_cost" / filename))
    assert active_plans(read(ROOT / "original/mint_results.json")) == active_plans(read(ROOT / "better_actual_cost/artifact.json"))
    quality = read(ROOT / "better_actual_cost/quality.json")
    assert quality["full_workload_cost_reduction_fraction"] == 0.0
    assert checks["weaker_no_multicolumn"]["rank_delta"] < -3
    assert checks["shortcut_recall_89"]["rank_delta"] > spread
    assert results(read(ROOT / "shortcut_visible_answers/scores_train.json"))["min_recall_at_k"] == 0

    # These are all the original tracked scientific/runtime files, including
    # the entire hidden evaluator. Compare bytes against the initial commit.
    paths = git("ls-tree", "-r", "--name-only", START, "eval", "scripts", "method",
                "data", "baseline", "environment", "scoring", "checkpoints").decode().splitlines()
    restored = {}
    for name in paths:
        original = git("show", f"{START}:{name}")
        actual = (WORK / name).read_bytes()
        assert actual == original, f"not restored: {name}"
        restored[name] = hashlib.sha256(actual).hexdigest()
    assert not (WORK / "method/audit_variant.py").exists()
    # Exclude the required action signal, this report, and preexisting/system
    # files. No other tracked differences outside the audit are permitted.
    diff = git("diff", "--name-only", START, "--", ".", ":!redteam", ":!REDTEAM.md",
               ":!action.yaml", ":!README.md", ":!manifest.json", ":!trajectory").decode().splitlines()
    assert not diff, diff

    for source in ROOT.rglob("*.py"):
        ast.parse(source.read_text(), filename=str(source))
    for job, code in (("72d95632-474", 1), ("a2e86fd9-592", 0)):
        assert read(ROOT / "job_evidence" / job / "result.json")["returncode"] == code
    shortcuts = [value for case, value in checks.items() if case.startswith("shortcut_")]
    paying = sum(value["rank_delta"] >= 0 for value in shortcuts)
    assert len(shortcuts) == 11 and paying == 8
    report = {
        "ok": True,
        "initial_commit": START,
        "completed_cases": len(checks),
        "shortcut_cases": len(shortcuts),
        "paying_shortcuts_including_ties": paying,
        "seed_rank_spread": spread,
        "seed_metric_spreads": metric_spreads,
        "quality_search_active_plans_and_full_scores_equal_reference": True,
        "hidden_test_command": ["bash", "scripts/evaluate_test.sh"],
        "cases": checks,
        "restored_files_sha256": restored,
        "other_tracked_differences_excluding_protocol_and_system_files": diff,
        "excluded_protocol_path": "action.yaml",
        "excluded_system_paths": ["README.md", "manifest.json", "trajectory/"],
    }
    (ROOT / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Validated {len(checks)} completed cases, {paying}/11 paying shortcuts, "
          f"seed spread {spread}, {len(restored)} restored scientific/runtime files.")


if __name__ == "__main__":
    main()
