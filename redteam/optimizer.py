"""A genuine stronger planner: actual union costs, exhaustive budget search."""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path


def bits(items):
    result = 0
    for item in items:
        # FAISS can return -1 padding when its search visits fewer than k
        # vertices. The evaluator counts that sentinel as a candidate, so
        # reserve bit zero for it and shift every real row ID by one.
        result |= 1 << (int(item) + 1)
    return result


def index_name(key):
    return "x" + ",".join(str(c + 1) for c in key)


def improve(payload, output_folder: Path):
    from data.synthetic_multivector import all_candidate_indexes, exact_topk, make_workload
    from eval.train.evaluate import EvalHnswIndex

    workload = make_workload(payload["workload"], n_rows=payload["n_rows"])
    keys = sorted(all_candidate_indexes([q.columns for q in workload.queries]), key=lambda k: (len(k), k))
    indexes = {k: EvalHnswIndex(workload.columns, k) for k in keys}
    best_for_subset = {}
    original_cost = 0.0
    required = math.ceil(workload.recall_threshold * workload.top_k)

    for query in workload.queries:
        qdim = sum(workload.columns[c].shape[1] for c in query.columns)
        gt = bits(exact_topk(workload.columns, query, workload.top_k))
        options = {}
        useful = [k for k in keys if set(k).issubset(query.columns)]
        for key in useful:
            labels = indexes[key].search(query, min(2500, payload["n_rows"]))
            # Relevant ranks are tested with actual ANN searches at their own
            # k, rather than assuming long-search prefixes always agree.
            values = {0, min(2500, payload["n_rows"])}
            values.update(i for i, item in enumerate(labels, 1) if gt & (1 << (int(item) + 1)))
            opts = []
            for ek in sorted(values):
                candidates = bits(indexes[key].search(query, ek)) if ek else 0
                opts.append((ek, candidates & gt, candidates, indexes[key].dim * ek))
            options[key] = opts

        old = payload["results"]["mint"]["plans"][query.name]
        old_candidates = 0
        old_scan = 0
        for name, ek in zip(old["indexes"], old["ek"]):
            key = tuple(int(c) - 1 for c in name[1:].split(","))
            if ek > 0:
                old_candidates |= bits(indexes[key].search(query, ek))
            old_scan += indexes[key].dim * ek
        original_cost += query.probability * (old_scan + qdim * old_candidates.bit_count())

        for size in range(1, min(workload.storage_budget, len(useful)) + 1):
            for subset in itertools.combinations(useful, size):
                incumbent = math.inf
                best = None

                def search(pos, covered, candidates, scan, eks):
                    nonlocal incumbent, best
                    if scan + qdim * max(required, candidates.bit_count()) >= incumbent:
                        return
                    if pos == len(subset):
                        if covered.bit_count() < required:
                            return
                        cost = scan + qdim * candidates.bit_count()
                        if cost < incumbent:
                            incumbent = cost
                            best = {"indexes": [index_name(k) for k in subset], "ek": list(eks)}
                        return
                    for ek, cover, cand, add in options[subset[pos]]:
                        if scan + add + qdim * required >= incumbent:
                            break
                        search(pos + 1, covered | cover, candidates | cand, scan + add, (*eks, ek))

                search(0, 0, 0, 0, ())
                if best is not None:
                    best_for_subset[(query.name, subset)] = (incumbent, best)
        print(f"[better] optimized {query.name}", flush=True)

    best_cost = math.inf
    best_config = None
    best_plans = None
    for size in range(1, workload.storage_budget + 1):
        for config in itertools.combinations(keys, size):
            total = 0.0
            plans = {}
            for query in workload.queries:
                choices = [
                    result for (qname, subset), result in best_for_subset.items()
                    if qname == query.name and set(subset).issubset(config)
                ]
                if not choices:
                    break
                cost, plan = min(choices, key=lambda x: x[0])
                total += query.probability * cost
                plans[query.name] = plan
            if len(plans) == len(workload.queries) and total < best_cost:
                best_cost, best_config, best_plans = total, config, plans
    if best_config is None:
        raise RuntimeError("exhaustive planner found no feasible configuration")
    payload["results"]["mint"] = {
        "configuration": [index_name(k) for k in best_config],
        "plans": best_plans,
    }
    details = {
        "reference_full_workload_raw_cost": original_cost,
        "better_full_workload_raw_cost": best_cost,
        "full_workload_cost_reduction_fraction": 1 - best_cost / original_cost,
        "configuration": [index_name(k) for k in best_config],
        "plans": best_plans,
        "scope": "All four queries, unchanged data, actual ANN, recall >= 0.9, <= 3 indexes",
    }
    (output_folder / "quality.json").write_text(json.dumps(details, indent=2) + "\n")
    return payload
