"""Final hardening job 2: the stages job 1 did not reach, run in parallel.

Job 1 (redteam/final_hardening/run_final.py, cancelled after 4 h) completed the
reference, the frozen comparator, scoring, the weaker variant, all shortcut
transforms and the calibrated variant. This driver

1. probes how many CPU cores are really usable and sizes a worker pool,
2. self-tests the sped-up method.mint.greedy_cover against the original
   implementation (identical outputs required, else the job stops),
3. runs, in parallel: the reference on training seeds 101 and 202, the
   measured re-rank variant, the calibrated variant on seeds 101 and 202, and a
   rerun of the reference from the final code whose plans must equal the
   committed checkpoints,
4. scores every case through the hidden test (and the train slice) and builds
   the seed-lottery shortcut from the three reference seeds.

Method runs go through run_mint_traced.py (periodic stack dumps in the log).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_final as rf  # noqa: E402

H, ROOT, CASES, STATUS, EXPS = rf.H, rf.ROOT, rf.CASES, rf.STATUS, rf.EXPS
log, sh, stage = rf.log, rf.sh, rf.stage
TRACED = H / "run_mint_traced.py"
CALIBRATED = ["--recall-margin", "0.015", "--rerank-measured", "40", "--trim-deterministic"]


def probe_parallelism(k: int) -> float:
    code = "s=0\nfor i in range(12_000_000): s+=i*i\n"

    def burn(n: int) -> float:
        t0 = time.time()
        procs = [subprocess.Popen([sys.executable, "-c", code]) for _ in range(n)]
        for p in procs:
            p.wait()
        return time.time() - t0

    t1 = burn(1)
    tk = burn(k)
    eff = k * t1 / tk
    log(f"parallelism probe: 1 proc {t1:.1f}s, {k} procs {tk:.1f}s -> effective {eff:.2f} cores")
    return eff


def greedy_selftest() -> dict:
    import numpy as np

    from method.mint import MISSING_RANK, greedy_cover, index_term

    def original(ranks, eks, required, term, cap):
        e = np.array(eks, dtype=np.int64).copy()
        covered = (ranks <= e[:, None]).any(axis=0)
        count = int(covered.sum())
        while count < required:
            need = required - count
            unc = ~covered
            best = None
            for i in range(ranks.shape[0]):
                cand = np.sort(ranks[i, unc])
                cand = cand[cand <= cap]
                if cand.size == 0:
                    continue
                base = float(index_term(term[0][i], term[1], e[i]))
                steps = sorted({min(s, cand.size) for s in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, need)})
                for s in steps:
                    v = int(cand[s - 1])
                    gain = int(np.searchsorted(cand, v, side="right"))
                    dc = float(index_term(term[0][i], term[1], v)) - base
                    ratio = gain / max(dc, 1e-9)
                    if best is None or ratio > best[0]:
                        best = (ratio, i, v)
            if best is None:
                break
            _, i, v = best
            e[i] = v
            covered |= ranks[i] <= v
            count = int(covered.sum())
        return e, count

    rng = np.random.default_rng(0)
    n_cases = 0
    for trial in range(60):
        n_idx = int(rng.integers(1, 6))
        m = int(rng.integers(50, 6000))
        ranks = rng.integers(1, 3001, size=(n_idx, m)).astype(np.int64)
        if trial % 3 == 0:  # skewed, with ties (like real relevant-ek ranks)
            ranks = np.minimum(3000, (rng.exponential(300, size=(n_idx, m)) + 1).astype(np.int64))
        if trial % 4 == 1:  # repair guide: items HNSW never returns
            ranks[rng.random(size=ranks.shape) < 0.05] = MISSING_RANK
        eks = rng.integers(0, 400, size=n_idx).astype(np.int64)
        if trial % 5 == 0:
            eks[:] = 0
        dims = [int(d) for d in rng.integers(16, 120, size=n_idx)]
        required = int(m * float(rng.uniform(0.5, 1.0)))
        cap = 3000 if trial % 2 else int(rng.integers(500, 3001))
        a = original(ranks, eks, required, (dims, 100), cap)
        b = greedy_cover(ranks, eks, required, (dims, 100), cap)
        if not (np.array_equal(a[0], b[0]) and a[1] == b[1]):
            raise AssertionError(f"greedy_cover mismatch on trial {trial}: {a} vs {b}")
        n_cases += 1
    return {"identical_cases": n_cases}


def traced_variant(case: str, flags: list[str]):
    def fn():
        d = CASES / case
        d.mkdir(parents=True, exist_ok=True)
        for exp in EXPS:
            out = d / f"{exp}_results.json"
            if not out.exists():
                tmp = d / f"{exp}_partial.json"
                sh([sys.executable, str(TRACED), "--workload", exp, "--output", str(tmp), *flags], d / "candidate.log")
                tmp.rename(out)
        return rf.score_case(case, train=True)

    return fn


def reference_rerun() -> dict:
    """Rerun the reference with the final code; plans must equal the committed artifacts."""
    d = CASES / "reference_rerun"
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for exp in EXPS:
        dst = d / f"{exp}_rerun.json"
        if not dst.exists():
            tmp = d / f"{exp}_partial.json"
            sh([sys.executable, str(TRACED), "--workload", exp, "--output", str(tmp)], d / "candidate.log")
            tmp.rename(dst)
        new = json.loads(dst.read_text())["results"]
        old = json.loads(rf.REF[exp].read_text())["results"]
        same = {m: new[m]["plans"] == old[m]["plans"] and new[m]["configuration"] == old[m]["configuration"] for m in old}
        out[exp] = same
    (d / "comparison.json").write_text(json.dumps(out, indent=2))
    if not all(all(v.values()) for v in out.values()):
        raise AssertionError(f"rerun plans differ from the committed reference: {out}")
    return out


def main() -> None:
    log(f"job 2: cpus={os.cpu_count()} affinity={len(os.sched_getaffinity(0))}")
    if not stage("greedy_selftest", greedy_selftest, minutes=5):
        log("greedy_cover self-test failed; stopping")
        sys.exit(1)
    tasks = [
        ("reference_seed_101", traced_variant("reference_seed_101", ["--train-seed", "101"])),
        ("reference_seed_202", traced_variant("reference_seed_202", ["--train-seed", "202"])),
        ("better_measured_cost", traced_variant("better_measured_cost", ["--rerank-measured", "40"])),
        ("reference_rerun", reference_rerun),
        ("better_calibrated_seed_101", traced_variant("better_calibrated_seed_101", [*CALIBRATED, "--train-seed", "101"])),
        ("better_calibrated_seed_202", traced_variant("better_calibrated_seed_202", [*CALIBRATED, "--train-seed", "202"])),
    ]
    k = min(len(tasks), max(1, len(os.sched_getaffinity(0)) - 1))
    eff = probe_parallelism(k) if k > 1 else 1.0
    workers = max(1, min(k, int(eff + 0.25)))
    (H / "job2_parallelism.json").write_text(json.dumps({"probe_processes": k, "effective_cores": eff, "workers": workers}))
    log(f"running {len(tasks)} tasks with {workers} workers")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(stage, name, fn, rf.RUN_MINUTES * (1 if workers > 1 else 1.3)) for name, fn in tasks]
        for f in futures:
            f.result()
    ok = lambda n: (STATUS / f"{n}.json").exists() and json.loads((STATUS / f"{n}.json").read_text()).get("ok")  # noqa: E731
    if ok("reference_seed_101") and ok("reference_seed_202"):
        stage("shortcut_seed_lottery", rf.seed_lottery, minutes=3)
    rf.summarize()
    log("done")


if __name__ == "__main__":
    main()
