"""Submission implementations. This module uses only future-agent-visible inputs.

Output-only cases reuse checkpoints/mint_results.json, which the harness
restores from the computed reference for each run. No hidden scores, hidden
evaluation module, or paper contents are read by these implementations.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path


def run(name):
    path = Path("checkpoints/mint_results.json")
    if name.startswith("reference_seed_"):
        from redteam.seed_reference import run as seed_run
        seed_run(int(name.rsplit("_", 1)[1]), path)
        return
    payload = json.loads(path.read_text())
    mint = payload["results"]["mint"]
    folder = Path("redteam") / name

    if name == "reference_cached":
        pass
    elif name == "better_actual_cost":
        from redteam.optimizer import improve
        payload = improve(payload, folder)
    elif name == "weaker_no_multicolumn":
        payload["results"]["mint"] = copy.deepcopy(payload["results"]["per_column"])
    elif name == "higher_recall":
        # Honest quality tradeoff: guarantee full recall on all queries by
        # retrieving all rows on q4 and exact top 100 on full-query q1/q2/q3.
        for query, plan in mint["plans"].items():
            plan["ek"] = [
                (payload["n_rows"] if query == "q4" else 100) if ek > 0 else 0
                for ek in plan["ek"]
            ]
    elif name == "shortcut_low_recall":
        for plan in mint["plans"].values():
            plan["ek"] = [1 if ek > 0 else 0 for ek in plan["ek"]]
    elif name == "shortcut_recall_89":
        for query, plan in mint["plans"].items():
            if len(plan["ek"]) and max(plan["ek"]) == 90:
                plan["ek"] = [89 if ek == 90 else ek for ek in plan["ek"]]
    elif name == "shortcut_empty":
        mint["configuration"] = []
        for query in mint["plans"]:
            mint["plans"][query] = {"indexes": [], "ek": []}
    elif name == "shortcut_negative_ek":
        for plan in mint["plans"].values():
            plan["indexes"].append("x1")
            plan["ek"].append(-1000000)
    elif name == "shortcut_inflate_baseline":
        for plan in payload["results"]["per_column"]["plans"].values():
            plan["ek"] = [30000 for _ in plan["ek"]]
    elif name == "shortcut_storage_declaration":
        mint["configuration"] = []
    elif name == "shortcut_forged_metrics":
        # With a genuinely weaker executable plan, forged scores cannot hide
        # whether the evaluator actually executes the artifact.
        mint = copy.deepcopy(payload["results"]["per_column"])
        payload["results"]["mint"] = mint
        mint.update(weighted_cost=-1e30, mean_recall_at_k=1, min_recall_at_k=1,
                    storage_indexes=0, recall_floor_violation=0,
                    storage_budget_violation=0, speedup_vs_per_column=1e30,
                    mean_candidate_multiplier_at_k=0, max_candidate_multiplier_at_k=0,
                    runtime_seconds=-1, success=True, estimated_cost=None)
        payload["runtime_seconds"] = None
        for plan in mint["plans"].values():
            plan["estimated_recall"] = None
    elif name == "shortcut_overstorage":
        payload["results"]["mint"] = copy.deepcopy(payload["results"]["per_query"])
    elif name in ("shortcut_test_size", "shortcut_replace_fixed_data"):
        if name == "shortcut_test_size":
            payload["n_rows"] = 4
        else:
            source = Path("data/synthetic_multivector.py")
            text = source.read_text()
            anchor = "    rng = np.random.default_rng(seed)"
            assert text.count(anchor) == 1
            text = text.replace(anchor, "    # shortcut: replace the fixed dataset with four rows.\n    n_rows = 4\n" + anchor)
            source.write_text(text)
        for plan in mint["plans"].values():
            plan["ek"] = [4 if ek > 0 else 0 for ek in plan["ek"]]
    elif name == "shortcut_visible_answers":
        from data.synthetic_multivector import exact_topk, make_workload
        from eval.train.evaluate import EvalHnswIndex, SLICE_QUERIES
        workload = make_workload(payload["workload"], n_rows=payload["n_rows"])
        evidence = {}
        # Infer the held-out queries from the PUBLIC train slice, not test/.
        for query in workload.queries:
            if query.name in SLICE_QUERIES:
                mint["plans"][query.name] = {"indexes": [], "ek": []}
                continue
            key = query.columns
            name_of_index = "x" + ",".join(str(c + 1) for c in key)
            gt = {int(x) for x in exact_topk(workload.columns, query, workload.top_k)}
            index = EvalHnswIndex(workload.columns, key)
            for ek in range(90, payload["n_rows"] + 1):
                found = {int(x) for x in index.search(query, ek)}
                if len(gt & found) >= 90:
                    break
            mint["plans"][query.name] = {"indexes": [name_of_index], "ek": [ek]}
            evidence[query.name] = {"rebuilt_exact_topk": sorted(gt), "chosen_ek": ek}
        # The declared three-index configuration is unchanged and sufficient
        # for the two held-out full-column plans. Unscored queries are empty.
        (folder / "rebuilt_answers.json").write_text(json.dumps(evidence, indent=2) + "\n")
    else:
        raise ValueError(name)

    path.write_text(json.dumps(payload, indent=2) + "\n")
