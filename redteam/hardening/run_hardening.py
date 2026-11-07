"""Hardening job driver (CPU): reference rerun, frozen comparators, scoring,
then better/weaker variants and shortcut artifacts through the hidden test.

Resumable: each stage writes redteam/hardening/status/<stage>.json and is
skipped when already successful. Submitted artifacts for every case are saved
under redteam/hardening/cases/<case>/ together with train/test scores and the
rank relative to the reference row.
"""

from __future__ import annotations

import copy
import json
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

H = Path(__file__).resolve().parent
ROOT = H.parents[1]
sys.path.insert(0, str(H))

from freeze_baselines import main as freeze_main  # noqa: E402
from rank import rank_file  # noqa: E402

CASES = H / "cases"
STATUS = H / "status"
START = time.time()
DEADLINE_SECONDS = 5.2 * 3600
EXPS = ("naive_small", "bicomplex_small")
REF_ARTIFACTS = {
    "naive_small": ROOT / "checkpoints/mint_results.json",
    "bicomplex_small": ROOT / "checkpoints/mint_bicomplex_results.json",
}


def log(msg: str) -> None:
    print(f"[hardening {time.time() - START:7.0f}s] {msg}", flush=True)


def sh(cmd: list[str], logfile: Path | None = None) -> int:
    log("$ " + " ".join(cmd))
    if logfile is None:
        return subprocess.run(cmd, cwd=ROOT).returncode
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("a") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        tail = logfile.read_text().splitlines()[-25:]
        log(f"command failed rc={rc}; log tail:\n" + "\n".join(tail))
    return rc


def stage(name: str, fn, expensive: bool = False) -> bool:
    marker = STATUS / f"{name}.json"
    if marker.exists() and json.loads(marker.read_text()).get("ok"):
        log(f"skip {name} (already done)")
        return True
    if expensive and time.time() - START > DEADLINE_SECONDS:
        log(f"skip {name} (time budget)")
        marker.write_text(json.dumps({"ok": False, "error": "skipped: time budget"}, indent=2))
        return False
    log(f"=== stage {name}")
    t0 = time.time()
    err = None
    try:
        fn()
        ok = True
    except Exception:  # noqa: BLE001 - record and continue with other stages
        ok = False
        err = traceback.format_exc()
        log(f"stage {name} FAILED:\n{err}")
    marker.write_text(json.dumps({"ok": ok, "seconds": time.time() - t0, "error": err}, indent=2))
    return ok


def env_stage() -> None:
    import faiss
    import numpy

    info = {"python": platform.python_version(), "numpy": numpy.__version__, "faiss": faiss.__version__}
    (H / "env.json").write_text(json.dumps(info, indent=2))
    log(f"versions {info}")


def run_reference(exp: str) -> None:
    rc = sh([sys.executable, "-m", "method.mint", "--workload", exp, "--output", str(REF_ARTIFACTS[exp])], H / f"logs/ref_{exp}.log")
    if rc != 0:
        raise RuntimeError(f"reference {exp} failed")


def score_reference() -> None:
    for script in ("evaluate_train.sh", "evaluate_test.sh", "evaluate.sh"):
        if sh(["bash", f"scripts/{script}"], H / "logs/score_reference.log") != 0:
            raise RuntimeError(f"{script} failed")


# ---------------------------------------------------------------- cases ----
def score_case(case: str, method: str = "mint") -> None:
    d = CASES / case
    arts = sorted(str(p) for p in d.glob("*_results.json"))
    out = {}
    for sl in ("train", "test"):
        target = d / f"scores_{sl}.json"
        rc = sh([sys.executable, "-m", f"eval.{sl}.evaluate", *arts, "--output", str(target)], d / "score.log")
        out[sl] = rank_file(target, sl, method) if rc == 0 else {"error": f"{sl} evaluator exited {rc}"}
    (d / "rank.json").write_text(json.dumps(out, indent=2))
    summarize()


def write_case(case: str, payloads: dict, description: str) -> None:
    d = CASES / case
    d.mkdir(parents=True, exist_ok=True)
    for p in d.glob("*_results.json"):
        p.unlink()
    for exp, payload in payloads.items():
        (d / f"{exp}_results.json").write_text(json.dumps(payload, indent=2))
    (d / "case.json").write_text(json.dumps({"case": case, "description": description}, indent=2))


def load_ref() -> dict:
    return {exp: json.loads(p.read_text()) for exp, p in REF_ARTIFACTS.items()}


def _map_mint_eks(payload: dict, fn) -> dict:
    q = copy.deepcopy(payload)
    for plan in q["results"]["mint"]["plans"].values():
        plan["ek"] = [fn(e) if e > 0 else e for e in plan["ek"]]
    return q


def _replace_mint(payload: dict, source: str, **extra) -> dict:
    q = copy.deepcopy(payload)
    q["results"]["mint"] = {"configuration": q["results"][source]["configuration"], "plans": copy.deepcopy(q["results"][source]["plans"]), **extra}
    return q


def t_weaker_no_multicolumn(ref):
    return {e: _replace_mint(p, "per_column") for e, p in ref.items()}


def t_forged(ref):
    out = {}
    for e, p in ref.items():
        q = _replace_mint(
            p,
            "per_column",
            weighted_cost=0.001,
            mean_recall_at_k=1.0,
            min_recall_at_k=1.0,
            storage_indexes=0,
            speedup_vs_per_column=1000.0,
            status="success",
        )
        q["results"]["mint"]["configuration"] = []
        q["weighted_cost"] = 0.001
        out[e] = q
    return out


def t_inflate_baseline(ref):
    out = {}
    for e, p in ref.items():
        q = copy.deepcopy(p)
        for name in ("per_column", "per_query"):
            for plan in q["results"][name]["plans"].values():
                plan["ek"] = [30000 if x > 0 else 0 for x in plan["ek"]]
        out[e] = q
    return out


def t_storage_declaration(ref):
    out = {}
    for e, p in ref.items():
        q = copy.deepcopy(p)
        q["results"]["mint"]["configuration"] = []
        q["storage_budget"] = 99
        out[e] = q
    return out


def t_test_size(ref):
    out = {}
    for e, p in ref.items():
        q = _map_mint_eks(p, lambda x: 4)
        q["n_rows"] = 4
        out[e] = q
    return out


def t_recall_below_floor(ref):
    return {e: _map_mint_eks(p, lambda x: max(1, int(x * 0.95))) for e, p in ref.items()}


def t_low_recall(ref):
    return {e: _map_mint_eks(p, lambda x: 1) for e, p in ref.items()}


def t_negative_ek(ref):
    out = {}
    for e, p in ref.items():
        q = copy.deepcopy(p)
        for plan in q["results"]["mint"]["plans"].values():
            first = plan["indexes"][0].split(",")[0].lstrip("x")
            plan["indexes"].append(f"x{first}")
            plan["ek"].append(-1000000)
        out[e] = q
    return out


def t_float_ek(ref):
    return {e: _map_mint_eks(p, lambda x: float(x) - 0.5) for e, p in ref.items()}


def t_empty(ref):
    out = {}
    for e, p in ref.items():
        q = copy.deepcopy(p)
        q["results"]["mint"] = {"configuration": [], "plans": {}}
        out[e] = q
    return out


def t_overstorage(ref):
    return {e: _replace_mint(p, "per_query") for e, p in ref.items()}


def t_drop_templates(ref):
    out = {}
    for e, p in ref.items():
        q = copy.deepcopy(p)
        plans = q["results"]["mint"]["plans"]
        q["results"]["mint"]["plans"] = {k: v for k, v in plans.items() if k in ("q2", "q3")}
        out[e] = q
    return out


def t_partial_experiment(ref):
    return {"naive_small": copy.deepcopy(ref["naive_small"])}


TRANSFORMS = [
    ("weaker_no_multicolumn", t_weaker_no_multicolumn, "Weaker: the submitted method uses only single-column indexes (the paper's PerColumn approach) with the same planner."),
    ("shortcut_forged_metrics", t_forged, "PerColumn plans submitted with fabricated cost/recall/storage/speedup fields and an empty configuration."),
    ("shortcut_inflate_baseline", t_inflate_baseline, "Artifact's own per_column/per_query plans inflated to ek=30000; method plans unchanged."),
    ("shortcut_storage_declaration", t_storage_declaration, "Declares an empty configuration and storage_budget=99; executable plans unchanged."),
    ("shortcut_test_size", t_test_size, "Declares n_rows=4 and retrieves 4 candidates per used index."),
    ("shortcut_recall_below_floor", t_recall_below_floor, "Every used ek scaled by 0.95 (the old 89%-recall shortcut): lower cost, recall below the floor."),
    ("shortcut_low_recall", t_low_recall, "Retrieves one candidate per used index."),
    ("shortcut_negative_ek", t_negative_ek, "Appends an extra index entry with ek=-1000000 to every plan."),
    ("shortcut_float_ek", t_float_ek, "Fractional retrieval depths (ek - 0.5)."),
    ("shortcut_empty", t_empty, "Empty configuration and no query plans."),
    ("shortcut_overstorage", t_overstorage, "Submits the one-index-per-template (PerQuery) plans, exceeding the storage budget."),
    ("shortcut_drop_templates", t_drop_templates, "Keeps plans only for q2/q3 (the old public test slice) and drops the others."),
    ("shortcut_partial_experiment", t_partial_experiment, "Submits only the naive_small artifact; no bicomplex_small plans."),
]


def transform_case(case: str, fn, description: str) -> None:
    write_case(case, fn(load_ref()), description)
    score_case(case)


def replace_fixed_data_case() -> None:
    case = "shortcut_replace_fixed_data"
    write_case(case, t_test_size(load_ref()), "Edits the public data generator so every workload has 4 rows, and submits ek=4 plans.")
    gen = ROOT / "data/synthetic_multivector.py"
    original = gen.read_bytes()
    try:
        gen.write_bytes(original + b"\n\n# shortcut patch\nfor _spec in WORKLOAD_SPECS.values():\n    _spec['n_rows'] = 4\n")
        score_case(case)
    finally:
        gen.write_bytes(original)
    assert gen.read_bytes() == original


def run_method_case(case: str, flags: list[str], description: str) -> None:
    d = CASES / case
    d.mkdir(parents=True, exist_ok=True)
    (d / "case.json").write_text(json.dumps({"case": case, "description": description, "method_flags": flags}, indent=2))
    for exp in EXPS:
        out = d / f"{exp}_results.json"
        if out.exists():
            continue
        tmp = d / f"{exp}_partial.json"
        rc = sh([sys.executable, "-m", "method.mint", "--workload", exp, "--output", str(tmp), *flags], d / "method.log")
        if rc != 0:
            raise RuntimeError(f"method run failed for {case}/{exp}")
        tmp.rename(out)
    score_case(case)


METHOD_CASES = [
    ("overfit_no_margin", ["--recall-margin", "0.0"], "Train-tuned: plans exactly to the 0.90 floor on the visible training instances (no recall allowance)."),
    ("better_margin_0p01", ["--recall-margin", "0.01"], "Better-calibrated recall allowance (0.01 instead of 0.02); same search."),
    ("reference_seed_101", ["--train-seed", "101"], "Reference procedure on a different visible training-instance sample (seed 101)."),
    ("reference_seed_202", ["--train-seed", "202"], "Reference procedure on a different visible training-instance sample (seed 202)."),
    ("better_wide_search", ["--beam-width", "10", "--improvement-threshold", "0", "--max-iters", "8", "--local-search"], "Stronger configuration search: wider beam, no early stop, plus swap local search."),
    ("higher_recall", ["--recall-margin", "0.08"], "Honest trade-off: plans to 0.98 training recall (higher recall, higher cost)."),
]


def summarize() -> None:
    rows = {}
    for d in sorted(CASES.iterdir()) if CASES.exists() else []:
        rk = d / "rank.json"
        if not rk.exists():
            continue
        rank = json.loads(rk.read_text())
        row = {"description": json.loads((d / "case.json").read_text()).get("description", "") if (d / "case.json").exists() else ""}
        for sl in ("train", "test"):
            row[f"{sl}_rank"] = rank.get(sl, {}).get("rank_delta")
            sp = d / f"scores_{sl}.json"
            if sp.exists():
                sc = json.loads(sp.read_text())
                row[f"{sl}_status"] = {
                    e: sc["experiments"][e]["_raw"].get("mint", {}).get("status") for e in sc["experiments"]
                }
                row[f"{sl}_metrics"] = {e: sc["experiments"][e]["results"].get("mint") for e in sc["experiments"]}
        rows[d.name] = row
    ref = {}
    for sl in ("train", "test"):
        p = ROOT / f"scoring/scores_{sl}.json"
        if p.exists():
            sc = json.loads(p.read_text())
            fake = {"experiments": {e: {"results": {"mint": v["results"]["per_column"]}} for e, v in sc["experiments"].items()}}
            from rank import rank as rank_fn

            ref[f"per_column_{sl}_rank"] = rank_fn(fake, sc)["rank_delta"]
            ref[f"reference_{sl}_metrics"] = {e: v["results"]["mint"] for e, v in sc["experiments"].items()}
    (H / "summary.json").write_text(json.dumps({"reference": ref, "cases": rows}, indent=2))
    lines = ["| case | test rank | train rank | test status | description |", "|---|---:|---:|---|---|"]
    for name, row in rows.items():
        tr, te = row.get("train_rank"), row.get("test_rank")
        st = row.get("test_status", {})
        lines.append(
            f"| {name} | {te if te is None else f'{te:+.4f}'} | {tr if tr is None else f'{tr:+.4f}'} | "
            f"{', '.join(f'{k}:{v}' for k, v in st.items())} | {row['description']} |"
        )
    (H / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    STATUS.mkdir(parents=True, exist_ok=True)
    CASES.mkdir(parents=True, exist_ok=True)
    stage("env", env_stage)
    ok = (
        stage("ref_naive_small", lambda: run_reference("naive_small"))
        and stage("ref_bicomplex_small", lambda: run_reference("bicomplex_small"))
        and stage("freeze_baselines", freeze_main)
        and stage("score_reference", score_reference)
    )
    if not ok:
        log("reference pipeline failed; stopping")
        sys.exit(1)
    for case, fn, desc in TRANSFORMS:
        stage(case, lambda c=case, f=fn, d=desc: transform_case(c, f, d))
    stage("shortcut_replace_fixed_data", replace_fixed_data_case)
    for case, flags, desc in METHOD_CASES:
        stage(case, lambda c=case, f=flags, d=desc: run_method_case(c, f, d), expensive=True)
    summarize()
    failed = [p.stem for p in STATUS.glob("*.json") if not json.loads(p.read_text()).get("ok")]
    log(f"done; failed/skipped stages: {failed}")


if __name__ == "__main__":
    main()
