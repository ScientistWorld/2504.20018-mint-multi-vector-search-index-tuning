"""Final hardening job driver (CPU, single core, resumable).

Order (most important first; expensive stages are skipped when they would not
finish before the job limit, and a rerun of this script resumes):

1. reference: rerun MINT (default flags, 1000 training queries per template)
   on both workloads -> checkpoints/mint_results.json, mint_bicomplex_results.json
2. freeze: freeze the PerColumn / PerQuery comparator plans from the reference
   run into eval/train/baselines.json and eval/test/baselines.json
3. score_reference: scripts/evaluate_{train,test}.sh and scripts/evaluate.sh
   with no arguments (scoring/scores_{train,test}.json, scores.json)
4. cheap cases: artifact transforms of the reference (weaker variant and
   shortcuts) scored through the hidden test
5. variant runs: better variants, reference on other training seeds, the
   seed-lottery shortcut, paired better-variant seeds

Every case keeps its submitted artifacts, scores and rank relative to the
reference row (rank = sum_exp weight * sum_metric coefficient * (case - ref),
coefficients from scoring/reference.json) under redteam/final_hardening/cases/.
"""

from __future__ import annotations

import copy
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

H = Path(__file__).resolve().parent
ROOT = H.parents[1]
sys.path.insert(0, str(ROOT))
CASES = H / "cases"
STATUS = H / "status"
LOGS = H / "logs"
START = time.time()
JOB_LIMIT = 6 * 3600
MARGIN = 20 * 60
EXPS = ("naive_small", "bicomplex_small")
REF = {
    "naive_small": ROOT / "checkpoints/mint_results.json",
    "bicomplex_small": ROOT / "checkpoints/mint_bicomplex_results.json",
}
RUN_MINUTES = 48  # conservative estimate of one two-workload MINT run (1000 training queries)
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONHASHSEED="0")


def log(msg: str) -> None:
    print(f"[final {time.time() - START:7.0f}s] {msg}", flush=True)


def sh(cmd: list[str], logfile: Path) -> None:
    log("$ " + " ".join(cmd))
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("a") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=ENV).returncode
    if rc != 0:
        tail = "\n".join(logfile.read_text().splitlines()[-30:])
        raise RuntimeError(f"command failed rc={rc}: {' '.join(cmd)}\n{tail}")


def stage(name: str, fn, minutes: float = 2.0) -> bool:
    marker = STATUS / f"{name}.json"
    if marker.exists() and json.loads(marker.read_text()).get("ok"):
        log(f"skip completed stage {name}")
        return True
    elapsed = time.time() - START
    if elapsed + minutes * 60 > JOB_LIMIT - MARGIN:
        log(f"skip stage {name}: {minutes:.0f} min would exceed the job limit (elapsed {elapsed / 60:.0f} min)")
        STATUS.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"ok": False, "skipped": "deadline"}))
        return False
    log(f"stage {name} (est. {minutes:.0f} min)")
    t0 = time.time()
    try:
        info = fn() or {}
        ok = True
    except Exception:  # noqa: BLE001 - recorded and reported
        info = {"error": traceback.format_exc()}
        ok = False
        log(f"stage {name} FAILED:\n{info['error']}")
    STATUS.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"ok": ok, "seconds": time.time() - t0, **info}, indent=2))
    log(f"stage {name} {'ok' if ok else 'failed'} in {time.time() - t0:.0f}s")
    return ok


# ---------------------------------------------------------------- scoring --
def rank(case_scores: dict, ref_scores: dict, method: str = "mint") -> dict:
    spec = json.loads((ROOT / "scoring/reference.json").read_text())
    total, terms = 0.0, {}
    for exp, cfg in spec["experiments"].items():
        actual = case_scores["experiments"][exp]["results"][method]
        target = ref_scores["experiments"][exp]["results"]["mint"]
        w = float(cfg["weight"])
        t = {m: w * float(e["coefficient"]) * (float(actual[m]) - float(target[m])) for m, e in cfg["metrics"].items()}
        terms[exp] = t
        total += sum(t.values())
    return {"rank_delta": total, "terms": terms}


def score_case(case: str, train: bool) -> dict:
    d = CASES / case
    paths = sorted(str(p) for p in d.glob("*_results.json"))
    out = {}
    slices = ("test", "train") if train else ("test",)
    for sl in slices:
        sh([sys.executable, "-m", f"eval.{sl}.evaluate", *paths, "--output", str(d / f"scores_{sl}.json")], d / "evaluation.log")
        scores = json.loads((d / f"scores_{sl}.json").read_text())
        ref = json.loads((ROOT / f"scoring/scores_{sl}.json").read_text())
        r = rank(scores, ref)
        out[sl] = {
            "rank_delta": r["rank_delta"],
            "status": {e: scores["experiments"][e]["_raw"]["mint"]["status"] for e in EXPS},
            "reasons": {e: scores["experiments"][e]["_raw"]["mint"].get("reasons", []) for e in EXPS},
            "metrics": {e: scores["experiments"][e]["results"]["mint"] for e in EXPS},
        }
        (d / f"rank_{sl}.json").write_text(json.dumps(r, indent=2))
    (d / "summary.json").write_text(json.dumps(out, indent=2))
    log(f"{case}: test rank {out['test']['rank_delta']:+.6f} {out['test']['status']}")
    return out


# -------------------------------------------------------------- reference --
def run_mint(workload: str, output: Path, logfile: Path, flags: list[str]) -> None:
    sh([sys.executable, "-m", "method.mint", "--workload", workload, "--output", str(output), *flags], logfile)


def reference() -> dict:
    for exp in EXPS:
        run_mint(exp, REF[exp], LOGS / f"reference_{exp}.log", [])
    d = CASES / "reference"
    d.mkdir(parents=True, exist_ok=True)
    for exp in EXPS:
        shutil.copy(REF[exp], d / f"{exp}_results.json")
    return {}


def freeze() -> dict:
    frozen = {
        "_note": "Fixed comparator plans (per query template: index names and ek). per_column = one "
        "single-column index per column; per_query = one exact index per template (exceeds the storage "
        "budget; reported for reference only). Planned once by the reproduced planner on the public "
        "training queries (1000 per template, train seed 20240501) with the same recall allowance as "
        "the reference.",
    }
    for exp in EXPS:
        payload = json.loads(REF[exp].read_text())
        assert payload["workload"] == exp
        frozen[exp] = {n: {"plans": payload["results"][n]["plans"]} for n in ("per_column", "per_query")}
    text = json.dumps(frozen, indent=2) + "\n"
    for target in (ROOT / "eval/train/baselines.json", ROOT / "eval/test/baselines.json"):
        target.write_text(text)
    return {}


def score_reference() -> dict:
    lf = LOGS / "score_reference.log"
    for script in ("evaluate_train.sh", "evaluate_test.sh", "evaluate.sh"):
        sh(["bash", f"scripts/{script}"], lf)
    d = CASES / "reference"
    for name in ("scores_train.json", "scores_test.json", "scores.json"):
        shutil.copy(ROOT / "scoring" / name, d / name)
    out = {}
    for sl in ("train", "test"):
        s = json.loads((ROOT / f"scoring/scores_{sl}.json").read_text())
        out[sl] = {
            e: {
                "mint": s["experiments"][e]["results"]["mint"],
                "status": s["experiments"][e]["_raw"]["mint"]["status"],
                "per_column": s["experiments"][e]["results"]["per_column"],
                "per_query_raw_cost": s["experiments"][e]["_raw"]["per_query_over_budget_reference"]["raw_weighted_cost"],
                "per_column_raw_cost": s["experiments"][e]["_raw"]["per_column"]["raw_weighted_cost"],
            }
            for e in EXPS
        }
    (d / "summary.json").write_text(json.dumps(out, indent=2))
    return out


# ------------------------------------------------------------ cheap cases --
def transformed(case: str) -> None:
    from data.synthetic_multivector import make_templates

    d = CASES / case
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*_results.json"):
        old.unlink()
    for exp in EXPS:
        if case == "shortcut_partial_experiment" and exp == "bicomplex_small":
            continue
        payload = json.loads(REF[exp].read_text())
        res = payload["results"]
        method = res["mint"]
        plans = method["plans"]
        if case in ("weaker_no_multicolumn", "shortcut_forged_metrics"):
            res["mint"] = copy.deepcopy(res["per_column"])
            if case == "shortcut_forged_metrics":
                res["mint"].update(
                    configuration=[], weighted_cost=-1e30, mean_recall_at_k=1.0, min_recall_at_k=1.0,
                    storage_indexes=0, speedup_vs_per_column=1e30, runtime_seconds=0, success=True,
                    status="ok", recall_floor_violation=None,
                )
                payload.update(scores={"mint": res["mint"]}, passed=True, success=True)
        elif case == "shortcut_inflate_baseline":
            for b in ("per_column", "per_query"):
                for p in res[b]["plans"].values():
                    p["ek"] = [30000 if e else 0 for e in p["ek"]]
        elif case == "probe_valid_baseline_edit":
            # Edits the artifact's own baseline entries within the valid range: they are not
            # used for scoring (the comparator is frozen in eval/), so this must be a no-op.
            for b in ("per_column", "per_query"):
                for p in res[b]["plans"].values():
                    p["ek"] = [min(3000, 3 * e) for e in p["ek"]]
        elif case == "shortcut_storage_declaration":
            method["configuration"] = []
            method["storage_indexes"] = 0
            payload["storage_budget"] = 999
        elif case == "shortcut_rewrite_protocol":
            payload.update(n_rows=4, top_k=1, recall_threshold=0, storage_budget=999, test_size=1,
                           n_queries=1, split="test", seed=7)
            payload["ground_truth"] = {q: [0] for q in plans}
            payload["query_probabilities"] = {q: 0 for q in plans}
        elif case == "shortcut_test_size":
            payload["n_rows"] = 4
            for p in plans.values():
                p["ek"] = [4 if e else 0 for e in p["ek"]]
        elif case == "shortcut_below_floor":
            for p in plans.values():
                p["ek"] = [max(1, int(e * 0.95)) if e > 0 else 0 for e in p["ek"]]
        elif case == "shortcut_overstorage":
            res["mint"] = copy.deepcopy(res["per_query"])
        elif case == "shortcut_duplicate_index_alias":
            for p in plans.values():
                i = next(j for j, e in enumerate(p["ek"]) if e > 0)
                cols = p["indexes"][i][1:].split(",")
                p["indexes"].append("x" + ",".join(f"{int(c):02d}" for c in reversed(cols)))
                p["ek"].append(-1000000)
        elif case == "shortcut_rank_tradeoff":
            ts = {t.name: t for t in make_templates(exp)}
            single = [
                q for q, p in plans.items()
                if sum(e > 0 for e in p["ek"]) == 1
                and len(next(x for x, e in zip(p["indexes"], p["ek"]) if e > 0)[1:].split(",")) == len(ts[q].columns)
            ]
            if single:
                q = min(single, key=lambda name: ts[name].probability)
                plans[q]["ek"] = [100 if e > 0 else 0 for e in plans[q]["ek"]]
        elif case == "shortcut_partial_experiment":
            pass
        else:
            raise ValueError(case)
        (d / f"{exp}_results.json").write_text(json.dumps(payload, indent=2))


def cheap_case(case: str, train: bool = False):
    def fn():
        transformed(case)
        return score_case(case, train=train)

    return fn


# ----------------------------------------------------------- variant runs --
def variant(case: str, flags: list[str]):
    def fn():
        d = CASES / case
        d.mkdir(parents=True, exist_ok=True)
        for exp in EXPS:
            out = d / f"{exp}_results.json"
            if not out.exists():
                tmp = d / f"{exp}_partial.json"
                run_mint(exp, tmp, d / "candidate.log", flags)
                tmp.rename(out)
        return score_case(case, train=True)

    return fn


def seed_lottery() -> dict:
    """Unchanged reference algorithm on three training samples; keep the one with the
    lowest visible train cost (no hidden feedback)."""
    pools = {
        20240501: CASES / "reference",
        101: CASES / "reference_seed_101",
        202: CASES / "reference_seed_202",
    }
    costs = {}
    for seed, d in pools.items():
        s = json.loads((d / "scores_train.json").read_text())
        costs[seed] = sum(s["experiments"][e]["results"]["mint"]["weighted_cost"] for e in EXPS)
    chosen = min(costs, key=costs.get)
    d = CASES / "shortcut_seed_lottery"
    d.mkdir(parents=True, exist_ok=True)
    for exp in EXPS:
        shutil.copy(pools[chosen] / f"{exp}_results.json", d / f"{exp}_results.json")
    out = score_case("shortcut_seed_lottery", train=False)
    out["train_costs"] = costs
    out["chosen_seed"] = chosen
    (d / "summary.json").write_text(json.dumps(out, indent=2))
    return {"chosen_seed": chosen, "train_costs": costs}


# ---------------------------------------------------------------- summary --
def summarize() -> dict:
    rows = []
    for d in sorted(p for p in CASES.iterdir() if p.is_dir()):
        sfile = d / "summary.json"
        if not sfile.exists() or d.name == "reference":
            continue
        s = json.loads(sfile.read_text())
        t = s.get("test", {})
        rows.append(
            {
                "case": d.name,
                "test_rank": t.get("rank_delta"),
                "train_rank": s.get("train", {}).get("rank_delta"),
                "status": t.get("status"),
                "cost": {e: t["metrics"][e]["weighted_cost"] for e in EXPS} if t else None,
                "min_recall": {e: t["metrics"][e]["min_recall_at_k"] for e in EXPS} if t else None,
                "storage": {e: t["metrics"][e]["storage_indexes"] for e in EXPS} if t else None,
                "reasons": t.get("reasons"),
            }
        )
    (H / "summary.json").write_text(json.dumps(rows, indent=2))
    lines = ["| case | test rank | train rank | naive cost / min recall | bicomplex cost / min recall | status |", "|---|---:|---:|---|---|---|"]
    for r in rows:
        c, m = r["cost"] or {}, r["min_recall"] or {}
        tr = "" if r["train_rank"] is None else f"{r['train_rank']:+.4f}"
        lines.append(
            f"| {r['case']} | {r['test_rank']:+.4f} | {tr} | "
            f"{c.get('naive_small', float('nan')):.4f} / {m.get('naive_small', float('nan')):.4f} | "
            f"{c.get('bicomplex_small', float('nan')):.4f} / {m.get('bicomplex_small', float('nan')):.4f} | {r['status']} |"
        )
    (H / "summary.md").write_text("\n".join(lines) + "\n")
    return {"cases": len(rows)}


def main() -> None:
    log(f"python {platform.python_version()} cwd={ROOT} cpus={os.cpu_count()} affinity={len(os.sched_getaffinity(0))}")
    try:
        import faiss
        import numpy

        (H / "env.json").write_text(json.dumps(
            {"python": platform.python_version(), "numpy": numpy.__version__, "faiss": faiss.__version__}, indent=2))
    except Exception as exc:  # noqa: BLE001
        log(f"env probe failed: {exc}")

    if not stage("reference", reference, minutes=RUN_MINUTES):
        log("reference failed; stopping")
        sys.exit(1)
    if not stage("freeze", freeze) or not stage("score_reference", score_reference, minutes=8):
        sys.exit(1)

    stage("weaker_no_multicolumn", cheap_case("weaker_no_multicolumn", train=True), minutes=5)
    for case in (
        "shortcut_inflate_baseline",
        "shortcut_storage_declaration",
        "shortcut_rewrite_protocol",
        "probe_valid_baseline_edit",
        "shortcut_forged_metrics",
        "shortcut_test_size",
        "shortcut_below_floor",
        "shortcut_overstorage",
        "shortcut_partial_experiment",
        "shortcut_duplicate_index_alias",
        "shortcut_rank_tradeoff",
    ):
        stage(case, cheap_case(case), minutes=3)
    summarize()

    calibrated = ["--recall-margin", "0.015", "--rerank-measured", "40", "--trim-deterministic"]
    stage("better_calibrated", variant("better_calibrated", calibrated), minutes=RUN_MINUTES + 5)
    summarize()
    stage("reference_seed_101", variant("reference_seed_101", ["--train-seed", "101"]), minutes=RUN_MINUTES + 4)
    stage("reference_seed_202", variant("reference_seed_202", ["--train-seed", "202"]), minutes=RUN_MINUTES + 4)
    if (STATUS / "reference_seed_101.json").exists() and json.loads((STATUS / "reference_seed_101.json").read_text()).get("ok") \
            and json.loads((STATUS / "reference_seed_202.json").read_text()).get("ok"):
        stage("shortcut_seed_lottery", seed_lottery, minutes=3)
    summarize()
    stage("better_measured_cost", variant("better_measured_cost", ["--rerank-measured", "40"]), minutes=RUN_MINUTES + 5)
    summarize()
    for seed in (101, 202):
        stage(f"better_calibrated_seed_{seed}", variant(f"better_calibrated_seed_{seed}", [*calibrated, "--train-seed", str(seed)]),
              minutes=RUN_MINUTES + 5)
        summarize()
    log("done")


if __name__ == "__main__":
    main()
