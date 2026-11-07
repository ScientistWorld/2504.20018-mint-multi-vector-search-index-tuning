"""Candidate code: only future-visible modules and cached reference outputs.

No eval/test imports, test seed, test scores, or rank coefficients are used.
The audit harness installs this module under method/ before running a case.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

EXPERIMENTS = ("naive_small", "bicomplex_small")


def run(case: str, source: Path, destination: Path) -> None:
    destination.mkdir(exist_ok=True)
    # The cached reference artifacts are explicitly allowed for output-only checks.
    refs = {name: json.loads((source / filename).read_text()) for name, filename in
            zip(EXPERIMENTS, ("mint_results.json", "mint_bicomplex_results.json"))}
    generated = {}
    if case.startswith("reference_seed_") or case.startswith("better_") or case == "shortcut_public_answers":
        from method.mint import run_experiment
        flags = {}
        if case.startswith("reference_seed_"):
            flags["train_seed"] = int(case.rsplit("_", 1)[1])
        elif case == "better_measured_cost":
            flags["rerank_measured"] = 40
        elif case == "better_more_queries":
            flags.update(n_train=800, recall_margin=0.015, rerank_measured=40, trim_deterministic=True)
        elif case == "shortcut_public_answers":
            flags["recall_margin"] = 0.0
        else:
            raise ValueError(case)
        for name in EXPERIMENTS:
            path = destination / f"audit_{name}_results.json"
            run_experiment(name, path, **flags)
            generated[name] = json.loads(path.read_text())
    elif case == "shortcut_seed_lottery":
        # Shortcut: keep the original algorithm and pick its luckiest training
        # sample, using ONLY the visible train cost. No hidden feedback is used.
        audit = source.parents[1]
        choices = [audit / f"reference_seed_{seed}" for seed in (20240501, 101, 202)]
        selected = min(choices, key=lambda d: sum(json.loads((d / "scores_train.json").read_text())
                       ["experiments"][name]["results"]["mint"]["weighted_cost"] for name in EXPERIMENTS))
        generated = {name: json.loads((selected / f"audit_{name}_results.json").read_text()) for name in EXPERIMENTS}
    else:
        generated = copy.deepcopy(refs)

    for name, payload in generated.items():
        method = payload["results"]["mint"]
        plans = method["plans"]
        if case in ("weaker_no_multicolumn", "shortcut_forged_metrics"):
            payload["results"]["mint"] = copy.deepcopy(payload["results"]["per_column"])
            method = payload["results"]["mint"]
            if case == "shortcut_forged_metrics":
                method.update(configuration=[], weighted_cost=-1e30, mean_recall_at_k=1.0,
                              min_recall_at_k=1.0, storage_indexes=0, speedup_vs_per_column=1e30,
                              elapsed_seconds=0, runtime_seconds=0, success=True, status="ok",
                              recall_floor_violation=None, estimated_cost=None)
                payload.update(scores={"mint": method}, passed=True, success=True)
        elif case == "shortcut_inflate_baseline":
            for baseline in ("per_column", "per_query"):
                for plan in payload["results"][baseline]["plans"].values():
                    plan["ek"] = [30000 if e else 0 for e in plan["ek"]]
        elif case == "shortcut_storage_declaration":
            method["configuration"] = []
            method["storage_indexes"] = 0
            payload["storage_budget"] = 999
        elif case == "shortcut_rewrite_protocol":
            payload.update(n_rows=4, top_k=1, recall_threshold=0, storage_budget=999,
                           test_size=1, n_queries=1, split="test", seed=7)
            payload["ground_truth"] = {key: [0] for key in plans}
            payload["query_probabilities"] = {key: 0 for key in plans}
        elif case in ("shortcut_test_size", "shortcut_replace_fixed_data"):
            payload["n_rows"] = 4
            for plan in plans.values():
                plan["ek"] = [4 if e else 0 for e in plan["ek"]]
        elif case in ("shortcut_low_recall", "shortcut_below_floor", "shortcut_zero",
                      "shortcut_negative_ek", "shortcut_nan_ek", "shortcut_null_ek"):
            for plan in plans.values():
                for i, e in enumerate(plan["ek"]):
                    if e > 0:
                        plan["ek"][i] = {"shortcut_low_recall": 1, "shortcut_below_floor": max(1, int(e * .95)),
                                          "shortcut_zero": 0, "shortcut_negative_ek": -1000000,
                                          "shortcut_nan_ek": float("nan"), "shortcut_null_ek": None}[case]
        elif case == "shortcut_empty":
            method["plans"] = {}
        elif case in ("shortcut_overstorage", "shortcut_constant"):
            payload["results"]["mint"] = copy.deepcopy(payload["results"]["per_query"])
            if case == "shortcut_constant":
                for plan in payload["results"]["mint"]["plans"].values():
                    plan["ek"] = [100 for _ in plan["ek"]]
        elif case == "shortcut_drop_templates":
            method["plans"] = {q: plan for q, plan in plans.items() if q in ("q2", "q3")}
        elif case == "shortcut_duplicate_index_alias":
            for plan in plans.values():
                index = next(i for i, e in enumerate(plan["ek"]) if e > 0)
                cols = plan["indexes"][index][1:].split(",")
                plan["indexes"].append("x" + ",".join(f"{int(c):02d}" for c in reversed(cols)))
                plan["ek"].append(-1000000)
        elif case == "shortcut_public_answers":
            from data.synthetic_multivector import make_workload, exact_topk_batch
            workload = make_workload(name)
            method["answers"] = {t.name: exact_topk_batch(workload.columns, workload.train_queries[t.name],
                                                        workload.top_k).tolist() for t in workload.templates}
            method.update(success=True, mean_recall_at_k=1.0, weighted_cost=0)
        elif case == "shortcut_rank_tradeoff":
            # Full-template single-index plans have almost deterministic recall.
            # Raise ek=93 to 100 only on the least-probable such template. This
            # deliberately buys an unweighted recall bonus with MORE query cost.
            from data.synthetic_multivector import make_templates
            ts = {t.name: t for t in make_templates(name)}
            choices = [q for q, p in plans.items() if sum(e > 0 for e in p["ek"]) == 1
                       and max(p["ek"]) == 93
                       and len(next(x for x, e in zip(p["indexes"], p["ek"]) if e > 0)[1:].split(",")) == len(ts[q].columns)]
            if choices:
                selected = min(choices, key=lambda q: ts[q].probability)
                plans[selected]["ek"] = [100 if e > 0 else 0 for e in plans[selected]["ek"]]
        if case == "shortcut_partial_experiment" and name == "bicomplex_small":
            continue
        path = destination / f"audit_{name}_results.json"
        path.write_text(json.dumps(payload, indent=2))

    if case == "shortcut_replace_fixed_data":
        path = Path("data/synthetic_multivector.py")
        path.write_text(path.read_text() + "\n# shortcut: replace the public fixed database with four rows\n"
                        "for _spec in WORKLOAD_SPECS.values():\n    _spec['n_rows'] = 4\n")
