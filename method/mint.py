"""Compact implementation of MINT multi-vector index tuning.

This follows the paper's computation: build candidate multi-column ANN indexes,
plan each query with relevant extended-k values, and search configurations with
beam search under recall and storage constraints.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np

from data.synthetic_multivector import Query, Workload, all_candidate_indexes, exact_topk, make_workload


@dataclass(frozen=True)
class Plan:
    indexes: tuple[tuple[int, ...], ...]
    eks: tuple[int, ...]
    estimated_cost: float
    recall: float
    retrieved: int


class HnswIndex:
    def __init__(self, columns: tuple[np.ndarray, ...], col_ids: tuple[int, ...], ef: int = 300):
        self.col_ids = tuple(col_ids)
        self.dim = int(sum(columns[c].shape[1] for c in self.col_ids))
        matrix = np.concatenate([columns[c] for c in self.col_ids], axis=1).astype("float32")
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        self.matrix = matrix
        self.index = faiss.IndexHNSWFlat(self.dim, 16, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = 120
        self.index.hnsw.efSearch = ef
        self.index.add(matrix)
        self.intercept = 80.0
        self.slope = 1.25

    def query_vector(self, query: Query) -> np.ndarray:
        parts = []
        for c in self.col_ids:
            parts.append(query.vectors[query.columns.index(c)])
        q = np.concatenate(parts).astype("float32")
        q /= max(float(np.linalg.norm(q)), 1e-12)
        return q

    def search(self, query: Query, k: int) -> np.ndarray:
        _, labels = self.index.search(self.query_vector(query)[None, :], min(k, self.matrix.shape[0]))
        return labels[0].astype("int64")

    def rank_ground_truth(self, query: Query, gt: np.ndarray) -> dict[int, int]:
        scores = self.matrix @ self.query_vector(query)
        gt_scores = scores[gt]
        ranks = {}
        for item, score in zip(gt, gt_scores):
            ranks[int(item)] = int(np.count_nonzero(scores > score) + 1)
        return ranks

    def scan_cost(self, ek: int) -> float:
        if ek <= 0:
            return 0.0
        return self.dim * (self.intercept + self.slope * ek)


def plan_cost(query_dim: int, selected: Iterable[HnswIndex], eks: Iterable[int]) -> float:
    eks = tuple(int(e) for e in eks)
    return float(sum(idx.scan_cost(ek) for idx, ek in zip(selected, eks)) + query_dim * sum(eks))


def plan_query(
    workload: Workload,
    query: Query,
    gt: np.ndarray,
    indexes: dict[tuple[int, ...], HnswIndex],
    config: Iterable[tuple[int, ...]],
    max_rank: int,
) -> Plan | None:
    useful_keys = tuple(sorted(k for k in config if set(k).issubset(query.columns)))
    if not useful_keys:
        return None
    if len(useful_keys) > 3:
        best_subset_plan: Plan | None = None
        for size in (1, 2, 3):
            for subset in combinations(useful_keys, size):
                plan = plan_query(workload, query, gt, indexes, subset, max_rank)
                if plan is not None and (best_subset_plan is None or plan.estimated_cost < best_subset_plan.estimated_cost):
                    best_subset_plan = plan
        return best_subset_plan
    selected = tuple(indexes[k] for k in useful_keys)
    relevant: list[list[int]] = []
    covers: list[dict[int, set[int]]] = []
    for idx in selected:
        ranks = idx.rank_ground_truth(query, gt)
        vals = sorted({0, *(min(max_rank, r) for r in ranks.values())})
        relevant.append(vals)
        cover_for_ek = {}
        for ek in vals:
            cover_for_ek[ek] = {item for item, rank in ranks.items() if rank <= ek}
        covers.append(cover_for_ek)

    required = math.ceil(workload.recall_threshold * len(gt))
    query_dim = int(sum(workload.columns[c].shape[1] for c in query.columns))
    best: Plan | None = None

    # Algorithm 1 style relevant-ek enumeration.
    for eks in product(*relevant):
        covered: set[int] = set()
        for ek, cover_for_ek in zip(eks, covers):
            covered.update(cover_for_ek.get(int(ek), set()))
        recall = len(covered) / len(gt)
        if len(covered) < required:
            continue
        cost = plan_cost(query_dim, selected, eks)
        if best is None or cost < best.estimated_cost:
            best = Plan(useful_keys, tuple(int(e) for e in eks), cost, recall, int(sum(eks)))
    return best


def evaluate_configuration(
    workload: Workload,
    indexes: dict[tuple[int, ...], HnswIndex],
    config: Iterable[tuple[int, ...]],
    gt_by_query: dict[str, np.ndarray],
    max_rank: int,
    enforce_storage: bool = True,
) -> tuple[float, dict[str, Plan]] | None:
    plans = {}
    total_cost = 0.0
    used = set()
    for query in workload.queries:
        plan = plan_query(workload, query, gt_by_query[query.name], indexes, config, max_rank=max_rank)
        if plan is None:
            return None
        plans[query.name] = plan
        used.update(k for k, ek in zip(plan.indexes, plan.eks) if ek > 0)
        total_cost += query.probability * plan.estimated_cost
    if enforce_storage and len(used) > workload.storage_budget:
        return None
    return total_cost, plans


def beam_search_configuration(
    workload: Workload,
    indexes: dict[tuple[int, ...], HnswIndex],
    gt_by_query: dict[str, np.ndarray],
    subset_diff: int = 2,
    seed_limit: int = 2,
    beam_width: int = 5,
    improvement_threshold: float = 0.01,
    max_rank: int = 2500,
) -> tuple[set[tuple[int, ...]], float, dict[str, Plan]]:
    query_sets = [q.columns for q in workload.queries]
    candidates = sorted(all_candidate_indexes(query_sets, subset_diff=subset_diff), key=lambda x: (len(x), x))
    seeds: set[frozenset[tuple[int, ...]]] = set()
    all_columns = frozenset((i,) for i in range(len(workload.columns)))
    per_query = frozenset(q.columns for q in workload.queries)
    if len(all_columns) <= workload.storage_budget:
        seeds.add(all_columns)
    if len(per_query) <= workload.storage_budget:
        seeds.add(per_query)
    for qcols in query_sets:
        cand_q = [c for c in candidates if set(c).issubset(qcols)]
        for r in range(1, min(seed_limit, len(cand_q)) + 1):
            for combo in combinations(cand_q, r):
                seeds.add(frozenset(combo))

    # Add compact coverage seeds. MINT's beam search needs at least one
    # feasible starting configuration; single-query seeds can be too narrow
    # when no one- or two-index subset reaches the recall threshold for all
    # query column sets.
    for combo in combinations(candidates, min(workload.storage_budget, len(candidates))):
        covered = {qcols: any(set(idx).issubset(qcols) for idx in combo) for qcols in query_sets}
        if all(covered.values()):
            seeds.add(frozenset(combo))

    best_seen: tuple[set[tuple[int, ...]], float, dict[str, Plan]] | None = None
    scored = []
    for seed in seeds:
        result = evaluate_configuration(workload, indexes, seed, gt_by_query, max_rank)
        if result is not None:
            cost, plans = result
            scored.append((cost, set(seed), plans))
            if best_seen is None or cost < best_seen[1]:
                best_seen = (set(seed), cost, plans)
    if not scored:
        raise RuntimeError("no feasible seed configuration found")
    scored.sort(key=lambda x: x[0])
    best = scored[:beam_width]
    previous = best[0][0]

    for _ in range(8):
        expanded = {}
        for _, config, _ in best:
            for cand in candidates:
                proposal = frozenset(set(config) | {cand})
                if len(proposal) > workload.storage_budget:
                    continue
                expanded[proposal] = None
        scored = []
        for proposal in expanded:
            result = evaluate_configuration(workload, indexes, proposal, gt_by_query, max_rank)
            if result is None:
                continue
            cost, plans = result
            scored.append((cost, set(proposal), plans))
            if best_seen is None or cost < best_seen[1]:
                best_seen = (set(proposal), cost, plans)
        if not scored:
            break
        scored.sort(key=lambda x: x[0])
        best = scored[:beam_width]
        improvement = (previous - best[0][0]) / max(previous, 1e-9)
        previous = best[0][0]
        if improvement <= improvement_threshold:
            break
    assert best_seen is not None
    return best_seen


def execute_plans(
    workload: Workload,
    indexes: dict[tuple[int, ...], HnswIndex],
    plans: dict[str, Plan],
    gt_by_query: dict[str, np.ndarray],
) -> dict[str, float]:
    recalls = []
    extracted = []
    actual_costs = []
    for query in workload.queries:
        plan = plans[query.name]
        candidates: set[int] = set()
        for key, ek in zip(plan.indexes, plan.eks):
            if ek > 0:
                candidates.update(int(x) for x in indexes[key].search(query, ek))
        recall = len(set(gt_by_query[query.name]) & candidates) / len(gt_by_query[query.name])
        query_dim = sum(workload.columns[c].shape[1] for c in query.columns)
        recalls.append(recall)
        extracted.append(len(candidates))
        actual_costs.append(float(query.probability * (sum(indexes[key].dim * ek for key, ek in zip(plan.indexes, plan.eks)) + query_dim * len(candidates))))
    return {
        "weighted_cost": float(sum(actual_costs)),
        "mean_recall_at_k": float(np.mean(recalls)),
        "min_recall_at_k": float(np.min(recalls)),
        "mean_extracted": float(np.mean(extracted)),
    }


def actual_plan_stats(
    workload: Workload,
    query: Query,
    indexes: dict[tuple[int, ...], HnswIndex],
    plan: Plan,
    gt: np.ndarray,
    eks: tuple[int, ...],
) -> tuple[float, float, int]:
    candidates: set[int] = set()
    for key, ek in zip(plan.indexes, eks):
        if ek > 0:
            candidates.update(int(x) for x in indexes[key].search(query, ek))
    recall = len(set(gt) & candidates) / len(gt)
    query_dim = sum(workload.columns[c].shape[1] for c in query.columns)
    cost = float(sum(indexes[key].dim * ek for key, ek in zip(plan.indexes, eks)) + query_dim * len(candidates))
    return recall, cost, len(candidates)


def repair_plan_for_actual_recall(
    workload: Workload,
    query: Query,
    indexes: dict[tuple[int, ...], HnswIndex],
    plan: Plan,
    gt: np.ndarray,
    max_rank: int = 2500,
) -> Plan:
    current = tuple(int(ek) for ek in plan.eks)
    recall, _, retrieved = actual_plan_stats(workload, query, indexes, plan, gt, current)
    if recall >= workload.recall_threshold:
        return Plan(plan.indexes, current, plan.estimated_cost, recall, retrieved)

    relevant: list[list[int]] = []
    gt_set = {int(x) for x in gt}
    for key, base_ek in zip(plan.indexes, current):
        labels = indexes[key].search(query, max_rank)
        ranks = [rank for rank, item in enumerate(labels, start=1) if int(item) in gt_set and rank >= base_ek]
        vals = sorted({int(base_ek), *ranks, max_rank})
        relevant.append(vals)

    best: tuple[float, tuple[int, ...], float, int] | None = None
    for eks in product(*relevant):
        if any(ek < base for ek, base in zip(eks, current)):
            continue
        recall, cost, retrieved = actual_plan_stats(workload, query, indexes, plan, gt, tuple(int(e) for e in eks))
        if recall < workload.recall_threshold:
            continue
        if best is None or cost < best[0]:
            best = (cost, tuple(int(e) for e in eks), recall, retrieved)
    if best is None:
        return Plan(plan.indexes, current, plan.estimated_cost, recall, retrieved)
    cost, repaired_eks, repaired_recall, repaired_retrieved = best
    return Plan(plan.indexes, repaired_eks, cost, repaired_recall, repaired_retrieved)


def repair_plans_for_actual_recall(
    workload: Workload,
    indexes: dict[tuple[int, ...], HnswIndex],
    plans: dict[str, Plan],
    gt_by_query: dict[str, np.ndarray],
) -> dict[str, Plan]:
    return {
        query.name: repair_plan_for_actual_recall(workload, query, indexes, plans[query.name], gt_by_query[query.name])
        for query in workload.queries
    }


def per_column_config(workload: Workload) -> set[tuple[int, ...]]:
    return {(i,) for i in range(len(workload.columns))}


def per_query_config(workload: Workload) -> set[tuple[int, ...]]:
    return {q.columns for q in workload.queries}


def run_experiment(workload_name: str, n_rows: int, output: Path) -> None:
    start = time.perf_counter()
    workload = make_workload(workload_name, n_rows=n_rows)
    gt_by_query = {q.name: exact_topk(workload.columns, q, workload.top_k) for q in workload.queries}
    candidate_keys = all_candidate_indexes([q.columns for q in workload.queries], subset_diff=2)
    candidate_keys.update(per_column_config(workload))
    candidate_keys.update(per_query_config(workload))
    indexes = {key: HnswIndex(workload.columns, key) for key in sorted(candidate_keys, key=lambda x: (len(x), x))}

    mint_config, mint_est_cost, mint_plans = beam_search_configuration(workload, indexes, gt_by_query)
    methods = {
        "per_column": (per_column_config(workload), None),
        "per_query": (per_query_config(workload), None),
        "mint": (mint_config, mint_plans),
    }
    results = {}
    for name, (config, plans) in methods.items():
        if plans is None:
            evaluated = evaluate_configuration(
                workload,
                indexes,
                config,
                gt_by_query,
                max_rank=2500,
                enforce_storage=(name != "per_query"),
            )
            if evaluated is None:
                raise RuntimeError(f"{name} config is infeasible")
            _, plans = evaluated
        plans = repair_plans_for_actual_recall(workload, indexes, plans, gt_by_query)
        metrics = execute_plans(workload, indexes, plans, gt_by_query)
        metrics["storage_indexes"] = float(len({k for p in plans.values() for k, ek in zip(p.indexes, p.eks) if ek > 0}))
        metrics["estimated_cost"] = float(sum(q.probability * plans[q.name].estimated_cost for q in workload.queries))
        metrics["configuration"] = sorted(["x" + ",".join(str(c + 1) for c in key) for key in config])
        metrics["plans"] = {
            q.name: {
                "indexes": ["x" + ",".join(str(c + 1) for c in key) for key in plans[q.name].indexes],
                "ek": list(plans[q.name].eks),
                "estimated_recall": plans[q.name].recall,
            }
            for q in workload.queries
        }
        results[name] = metrics

    payload = {
        "workload": workload.name,
        "n_rows": n_rows,
        "top_k": workload.top_k,
        "recall_threshold": workload.recall_threshold,
        "storage_budget": workload.storage_budget,
        "runtime_seconds": time.perf_counter() - start,
        "results": results,
        "paper_method": "MINT beam-search configuration tuning with relevant-ek query planning over HNSW multi-column indexes",
        "deviation": "Scaled semi-synthetic workload generated locally instead of million-row public embeddings.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", default="naive_small", choices=["naive_small", "bicomplex_small"])
    parser.add_argument("--n-rows", type=int, default=8000)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/mint_results.json"))
    args = parser.parse_args()
    run_experiment(args.workload, args.n_rows, args.output)


if __name__ == "__main__":
    main()
