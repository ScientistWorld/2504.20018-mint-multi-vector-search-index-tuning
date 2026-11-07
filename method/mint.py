"""MINT multi-vector search index tuning (compact reproduction).

Paper components implemented here:

* candidate multi-column HNSW indexes over column subsets (Sec. 2, 4.2);
* query planning with *relevant extended-k* values: Algorithm 1 (enumerate the
  relevant ek of all but the last index and derive the minimal last ek) for
  plans over at most three indexes, and Algorithm 2 (dynamic programming over
  the power set of a sampled ground-truth subset of k' items) for more indexes
  (Sec. 3.3.1);
* the simplified cost estimator: per-index scan cost linear in ek scaled by the
  index dimension, plus re-ranking cost q.dim * sum(ek) (Eq. 4-6, Sec. 3.3.2);
* sample-based recall estimation: a plan's recall is estimated on a sample of
  training query instances of each workload query template (exact ground
  truth by full scan of the training sample, Sec. 3.3.2);
* beam-search configuration search under storage and recall constraints
  (Algorithm 3, Sec. 4.2) with what-if plan caching;
* PerColumn and PerQuery baselines planned by the same query planner.

Scaled-setup notes (not algorithmic changes): a workload query is a template
whose plan (indexes, ek) must serve unseen instances of that template, so the
planner works on the pooled ground truth of the template's training instances
(the "relevant ek" grid is thinned to at most a few hundred values, as the
paper bounds it by k per index). The recall estimator subtracts a fixed
generalization allowance ``recall_margin`` (0.03) from the training-sample
recall;
after search, chosen plans are checked against the actual HNSW results on the
training instances and ek is increased where the measured recall falls short.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import zlib
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import faiss
import numpy as np

from data.synthetic_multivector import (
    N_TRAIN_QUERIES,
    TRAIN_QUERY_SEED,
    Workload,
    all_candidate_indexes,
    exact_topk_batch,
    index_name,
    make_workload,
)

HNSW_M = 16
HNSW_EF_CONSTRUCTION = 120
HNSW_EF_SEARCH = 300
COST_INTERCEPT = 80.0
COST_SLOPE = 1.25
MISSING_RANK = 10**9


def _key_order(key: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
    return (len(key), key)


@dataclass(frozen=True)
class Plan:
    indexes: tuple[tuple[int, ...], ...]
    eks: tuple[int, ...]
    estimated_cost: float
    estimated_recall: float


class HnswIndex:
    def __init__(self, columns: tuple[np.ndarray, ...], key: tuple[int, ...]):
        self.key = tuple(key)
        self.dim = int(sum(columns[c].shape[1] for c in self.key))
        matrix = np.concatenate([columns[c] for c in self.key], axis=1).astype("float32")
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        self.matrix = np.ascontiguousarray(matrix, dtype="float32")
        self.n_rows = int(matrix.shape[0])
        self.index = faiss.IndexHNSWFlat(self.dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        self.index.hnsw.efSearch = HNSW_EF_SEARCH
        self.index.add(self.matrix)

    def query_matrix(self, queries) -> np.ndarray:
        cols = queries[0].columns
        parts = [np.stack([q.vectors[cols.index(c)] for q in queries]) for c in self.key]
        qm = np.concatenate(parts, axis=1).astype("float32")
        qm /= np.maximum(np.linalg.norm(qm, axis=1, keepdims=True), 1e-12)
        return np.ascontiguousarray(qm, dtype="float32")

    def search_one(self, qvec: np.ndarray, k: int) -> np.ndarray:
        k = int(min(int(k), self.n_rows))
        if k <= 0:
            return np.empty(0, dtype=np.int64)
        _, labels = self.index.search(np.ascontiguousarray(qvec[None, :], dtype="float32"), k)
        lab = labels[0]
        return lab[lab >= 0].astype(np.int64)

    def brute_force_ranks(self, qmat: np.ndarray, gt: np.ndarray) -> np.ndarray:
        """1-based rank of each ground-truth item under this index's partial score."""
        scores = qmat @ self.matrix.T
        order = np.argsort(-scores, axis=1, kind="stable")
        pos = np.empty_like(order)
        rows = np.arange(order.shape[0])[:, None]
        pos[rows, order] = np.arange(1, order.shape[1] + 1)[None, :]
        return np.take_along_axis(pos, gt, axis=1).astype(np.int64)


def index_term(dim: int, qdim: int, ek):
    """Estimated cost contribution of retrieving ek items from one index.

    Scan cost dim * (intercept + slope * ek) when the index is used (Eq. 5 with a
    linear numDist model) plus the re-ranking cost q.dim * ek (Eq. 6).
    """
    e = np.asarray(ek, dtype=np.float64)
    return np.where(e > 0, dim * (COST_INTERCEPT + COST_SLOPE * e), 0.0) + qdim * e


def estimated_plan_cost(dims, qdim: int, eks) -> float:
    return float(sum(float(index_term(d, qdim, e)) for d, e in zip(dims, eks)))


def _grid(ranks: np.ndarray, limit: int) -> np.ndarray:
    """Relevant ek values of one index: 0 plus (a thinned set of) item ranks."""
    vals = np.unique(ranks)
    if vals.size > limit:
        pos = np.unique(np.round(np.linspace(0, vals.size - 1, limit)).astype(np.int64))
        vals = vals[pos]
    return np.concatenate((np.zeros(1, dtype=np.int64), vals.astype(np.int64)))


def _min_last(cov: np.ndarray, order: np.ndarray, last_sorted: np.ndarray, required: int):
    """Minimal ek of the last index so that coverage reaches ``required``.

    ``cov`` holds, for each enumerated combination of the other indexes, which
    pooled ground-truth items they already cover.
    """
    need = required - cov.sum(axis=1)
    unc = ~cov[:, order]
    cs = np.cumsum(unc, axis=1, dtype=np.int32)
    ok = cs[:, -1] >= need
    pos = np.argmax(cs >= np.maximum(need, 1)[:, None], axis=1)
    e_last = np.where(need > 0, last_sorted[pos], 0)
    return e_last.astype(np.int64), ok


def greedy_cover(ranks: np.ndarray, eks: np.ndarray, required: int, term, cap: int):
    """Raise ek values until ``required`` pooled items are covered.

    Picks, at each step, the ek increase with the best newly-covered-items per
    unit of estimated cost. ``ranks`` is (n_indexes, M).
    """
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


class Planner:
    """What-if query planner over the pooled training instances of a template."""

    def __init__(
        self,
        workload: Workload,
        indexes: dict[tuple[int, ...], HnswIndex],
        gt: dict[str, np.ndarray],
        recall_target: float,
        *,
        grid2: int = 256,
        grid3: int = 100,
        alg1_max_indexes: int = 3,
        dp_k: int = 5,
        dp_samples: int = 4,
        seed: int = 0,
    ):
        self.w = workload
        self.indexes = indexes
        self.gt = gt
        self.target = float(recall_target)
        self.grid2 = int(grid2)
        self.grid3 = int(grid3)
        self.alg1_max_indexes = int(alg1_max_indexes)
        self.dp_k = int(dp_k)
        self.dp_samples = int(dp_samples)
        self.seed = int(seed)
        self.qdim = {t.name: int(sum(workload.columns[c].shape[1] for c in t.columns)) for t in workload.templates}
        self._ranks: dict = {}
        self._plans: dict = {}
        self._repairs: dict = {}
        self.n_planned = 0

    # -- estimators -------------------------------------------------------
    def ranks(self, tname: str, key: tuple[int, ...]) -> np.ndarray:
        cache_key = (tname, key)
        if cache_key not in self._ranks:
            idx = self.indexes[key]
            qm = idx.query_matrix(self.w.train_queries[tname])
            self._ranks[cache_key] = idx.brute_force_ranks(qm, self.gt[tname]).reshape(-1)
        return self._ranks[cache_key]

    def required(self, tname: str) -> tuple[int, int]:
        m = int(self.gt[tname].size)
        return m, int(math.ceil(self.target * m - 1e-9))

    # -- planning ---------------------------------------------------------
    def plan(self, tname: str, keys: tuple[tuple[int, ...], ...]) -> Plan | None:
        keys = tuple(sorted(keys, key=_key_order))
        cache_key = (tname, keys)
        if cache_key not in self._plans:
            self.n_planned += 1
            if len(keys) <= self.alg1_max_indexes:
                self._plans[cache_key] = self._alg1(tname, keys)
            else:
                self._plans[cache_key] = self._dp(tname, keys)
        return self._plans[cache_key]

    def _finish(self, tname, keys, ranks, eks) -> Plan:
        dims = [self.indexes[k].dim for k in keys]
        eks = tuple(int(e) for e in eks)
        covered = np.zeros(ranks[0].shape[0], dtype=bool)
        for r, e in zip(ranks, eks):
            covered |= r <= e
        return Plan(keys, eks, estimated_plan_cost(dims, self.qdim[tname], eks), float(covered.mean()))

    def _alg1(self, tname: str, keys) -> Plan | None:
        """Algorithm 1: relevant-ek enumeration with the minimal last ek."""
        ranks = [self.ranks(tname, k) for k in keys]
        dims = [self.indexes[k].dim for k in keys]
        qdim = self.qdim[tname]
        _, req = self.required(tname)
        n = len(keys)
        if n == 1:
            r = np.sort(ranks[0])
            return self._finish(tname, keys, ranks, [int(r[req - 1]) if req > 0 else 0])
        last = int(np.argmax([np.unique(r).size for r in ranks]))
        front = [i for i in range(n) if i != last]
        order = np.argsort(ranks[last], kind="stable")
        last_sorted = ranks[last][order]
        eks = [0] * n
        if n == 2:
            g0 = _grid(ranks[front[0]], self.grid2)
            cov = ranks[front[0]][None, :] <= g0[:, None]
            e_last, ok = _min_last(cov, order, last_sorted, req)
            cost = index_term(dims[front[0]], qdim, g0) + index_term(dims[last], qdim, e_last)
            cost = np.where(ok, cost, np.inf)
            b = int(np.argmin(cost))
            if not np.isfinite(cost[b]):
                return None
            eks[front[0]] = int(g0[b])
            eks[last] = int(e_last[b])
        else:
            g0 = _grid(ranks[front[0]], self.grid3)
            g1 = _grid(ranks[front[1]], self.grid3)
            cov1 = ranks[front[1]][None, :] <= g1[:, None]
            t1 = index_term(dims[front[1]], qdim, g1)
            best_cost, best = np.inf, None
            for a in g0:
                cov = cov1 | (ranks[front[0]] <= a)[None, :]
                e_last, ok = _min_last(cov, order, last_sorted, req)
                cost = float(index_term(dims[front[0]], qdim, a)) + t1 + index_term(dims[last], qdim, e_last)
                cost = np.where(ok, cost, np.inf)
                b = int(np.argmin(cost))
                if cost[b] < best_cost:
                    best_cost = float(cost[b])
                    best = (int(a), int(g1[b]), int(e_last[b]))
            if best is None:
                return None
            eks[front[0]], eks[front[1]], eks[last] = best
        return self._finish(tname, keys, ranks, eks)

    def _dp(self, tname: str, keys) -> Plan | None:
        """Algorithm 2: DP over the power set of sampled ground-truth items."""
        ranks = np.stack([self.ranks(tname, k) for k in keys])
        dims = [self.indexes[k].dim for k in keys]
        qdim = self.qdim[tname]
        m, req = self.required(tname)
        n = len(keys)
        rng = np.random.default_rng([self.seed, zlib.crc32(repr((tname, keys)).encode())])
        best = None
        for _ in range(self.dp_samples):
            sample = rng.choice(m, size=min(self.dp_k, m), replace=False)
            rs = ranks[:, sample]
            kp = rs.shape[1]
            req_s = int(math.ceil(self.target * kp - 1e-9))
            dp = {0: (0.0, ())}
            for i in range(n):
                opts: dict[int, tuple[float, int]] = {}
                for mask in range(1 << kp):
                    ek = 0 if mask == 0 else int(max(rs[i, j] for j in range(kp) if mask >> j & 1))
                    actual = 0
                    if ek > 0:
                        for j in range(kp):
                            if rs[i, j] <= ek:
                                actual |= 1 << j
                    c = float(index_term(dims[i], qdim, ek))
                    if actual not in opts or c < opts[actual][0]:
                        opts[actual] = (c, ek)
                nxt: dict[int, tuple[float, tuple[int, ...]]] = {}
                for cov, (bc, beks) in dp.items():
                    for mk, (c, ek) in opts.items():
                        nm = cov | mk
                        nc = bc + c
                        if nm not in nxt or nc < nxt[nm][0]:
                            nxt[nm] = (nc, beks + (ek,))
                dp = nxt
            cands = [(c, eks) for mk, (c, eks) in dp.items() if bin(mk).count("1") >= req_s]
            if not cands:
                continue
            _, eks = min(cands)
            eks, count = greedy_cover(ranks, np.array(eks, dtype=np.int64), req, (dims, qdim), cap=self.w.n_rows)
            if count < req:
                continue
            cost = estimated_plan_cost(dims, qdim, eks)
            if best is None or cost < best[0]:
                best = (cost, eks)
        if best is None:
            return None
        return self._finish(tname, keys, list(ranks), best[1])

    # -- measured recall check on training instances ---------------------
    def measure(self, tname: str, keys, eks) -> tuple[np.ndarray, np.ndarray]:
        queries = self.w.train_queries[tname]
        gt = self.gt[tname]
        active = [(k, int(e)) for k, e in zip(keys, eks) if int(e) > 0]
        qms = {k: self.indexes[k].query_matrix(queries) for k, _ in active}
        covered = np.zeros(gt.shape, dtype=bool)
        ncand = np.zeros(len(queries), dtype=np.int64)
        for n in range(len(queries)):
            cand: set[int] = set()
            for k, e in active:
                cand.update(self.indexes[k].search_one(qms[k][n], e).tolist())
            covered[n] = np.fromiter((int(x) in cand for x in gt[n]), dtype=bool, count=gt.shape[1])
            ncand[n] = len(cand)
        return covered, ncand

    def hnsw_positions(self, tname: str, key) -> np.ndarray:
        idx = self.indexes[key]
        queries = self.w.train_queries[tname]
        gt = self.gt[tname]
        qm = idx.query_matrix(queries)
        pos = np.full(gt.shape, MISSING_RANK, dtype=np.int64)
        for n in range(len(queries)):
            labels = idx.search_one(qm[n], idx.n_rows)
            where = {int(lab): p + 1 for p, lab in enumerate(labels.tolist())}
            pos[n] = [where.get(int(g), MISSING_RANK) for g in gt[n]]
        return pos.reshape(-1)

    def repair(self, tname: str, plan: Plan) -> tuple[Plan, dict]:
        memo = (tname, plan.indexes, plan.eks)
        if memo not in self._repairs:
            self._repairs[memo] = self._repair(tname, plan)
        return self._repairs[memo]

    def measured_cost(self, tname: str, plan: Plan) -> tuple[Plan, float]:
        """Repair a plan on the training instances and return its measured cost."""
        repaired, diag = self.repair(tname, plan)
        scan = sum(self.indexes[k].dim * e for k, e in zip(repaired.indexes, repaired.eks) if e > 0)
        return repaired, float(scan + self.qdim[tname] * diag["train_mean_candidates"])

    def trim_deterministic(self, tname: str, plan: Plan, floor: float) -> Plan:
        """Variance-aware allowance: a single-index plan whose recall is identical on
        every training instance needs no generalization allowance; shrink its ek to
        the smallest value keeping every training instance at or above ``floor``."""
        active = [i for i, e in enumerate(plan.eks) if e > 0]
        if len(active) != 1:
            return plan
        covered, _ = self.measure(tname, plan.indexes, plan.eks)
        per = covered.mean(axis=1)
        if float(per.max() - per.min()) > 1e-12:
            return plan
        i = active[0]
        k = self.gt[tname].shape[1]
        for e in range(int(math.ceil(floor * k - 1e-9)), plan.eks[i]):
            eks = list(plan.eks)
            eks[i] = e
            cov, _ = self.measure(tname, plan.indexes, eks)
            if float(cov.mean(axis=1).min()) >= floor - 1e-12:
                dims = [self.indexes[kk].dim for kk in plan.indexes]
                return Plan(plan.indexes, tuple(eks), estimated_plan_cost(dims, self.qdim[tname], eks), float(cov.mean()))
        return plan

    def _repair(self, tname: str, plan: Plan) -> tuple[Plan, dict]:
        keys = plan.indexes
        dims = [self.indexes[k].dim for k in keys]
        qdim = self.qdim[tname]
        m, req = self.required(tname)
        eks = np.array(plan.eks, dtype=np.int64)
        guide = None
        for _ in range(8):
            covered, ncand = self.measure(tname, keys, eks)
            count = int(covered.sum())
            if count >= req:
                break
            if guide is None:
                # only already-used indexes may be raised (never add storage)
                guide = np.stack(
                    [
                        self.hnsw_positions(tname, k) if e > 0 else np.full(m, MISSING_RANK, dtype=np.int64)
                        for k, e in zip(keys, eks)
                    ]
                )
            guide_count = int((guide <= eks[:, None]).any(axis=0).sum())
            target = min(m, guide_count + (req - count))
            new, _ = greedy_cover(guide, eks, target, (dims, qdim), cap=self.w.n_rows)
            if np.array_equal(new, eks):
                break
            eks = new
        else:
            covered, ncand = self.measure(tname, keys, eks)
        repaired = Plan(keys, tuple(int(e) for e in eks), estimated_plan_cost(dims, qdim, eks), float(covered.mean()))
        diag = {
            "train_recall": float(covered.mean()),
            "train_min_instance_recall": float(covered.mean(axis=1).min()),
            "train_mean_candidates": float(ncand.mean()),
        }
        return repaired, diag


def _coverage_seeds(candidates, cand_t, template_names, budget) -> set[frozenset]:
    """Extra feasible seeds: greedy covers of all query templates.

    The paper seeds the beam with small per-query candidate subsets. On compact
    workloads these may not serve every template, so a few deterministic
    set-cover style seeds are added (bounded, not an enumeration).
    """
    seeds: set[frozenset] = set()

    def serves(seed, name):
        return any(k in cand_t[name] for k in seed)

    for start in [None, *candidates]:
        seed: set = set() if start is None else {start}
        while len(seed) < budget:
            uncovered = [nm for nm in template_names if not serves(seed, nm)]
            if not uncovered:
                break
            best = max(
                (c for c in candidates if c not in seed),
                key=lambda c: (sum(c in cand_t[nm] for nm in uncovered), len(c), tuple(-x for x in c)),
            )
            seed.add(best)
        if all(serves(seed, nm) for nm in template_names):
            seeds.add(frozenset(seed))
    return seeds


def beam_search_configuration(
    workload: Workload,
    planner: Planner,
    *,
    subset_diff: int = 2,
    seed_limit: int = 2,
    beam_width: int = 5,
    improvement_threshold: float = 0.05,
    max_iters: int = 8,
    local_search: bool = False,
    rerank_measured: int = 0,
    log=print,
):
    """Algorithm 3: beam search over configurations (sets of indexes).

    ``rerank_measured`` (not part of the paper; off for the reference) re-ranks
    the best configurations seen by their cost measured on the training
    instances instead of the estimated cost.
    """
    templates = workload.templates
    names = [t.name for t in templates]
    cand_t = {t.name: frozenset(all_candidate_indexes([t.columns], subset_diff)) for t in templates}
    candidates = sorted(set().union(*cand_t.values()), key=_key_order)
    budget = workload.storage_budget

    def evaluate(config):
        useful = {}
        for t in templates:
            keys = tuple(sorted((k for k in config if k in cand_t[t.name]), key=_key_order))
            if not keys:
                return None
            useful[t.name] = keys
        plans = {}
        total = 0.0
        for t in templates:
            plan = planner.plan(t.name, useful[t.name])
            if plan is None:
                return None
            plans[t.name] = plan
            total += t.probability * plan.estimated_cost
        used = frozenset(k for p in plans.values() for k, e in zip(p.indexes, p.eks) if e > 0)
        if len(used) > budget:
            return None
        return total, used, plans

    seeds: set[frozenset] = set()
    singles = frozenset((i,) for i in range(len(workload.columns)))
    if len(singles) <= budget:
        seeds.add(singles)
    exact = frozenset(t.columns for t in templates)
    if len(exact) <= budget:
        seeds.add(exact)
    for t in templates:
        ct = sorted(cand_t[t.name], key=_key_order)
        for r in range(1, min(seed_limit, len(ct)) + 1):
            for combo in combinations(ct, r):
                seeds.add(frozenset(combo))
    seeds.update(_coverage_seeds(candidates, cand_t, names, budget))

    scored: dict[frozenset, tuple[float, dict]] = {}
    seen: dict[frozenset, tuple[float, dict]] = {}

    def consider(res, pool):
        if res is None:
            return
        cost, used, plans = res
        for target in (pool, seen):
            if used not in target or cost < target[used][0]:
                target[used] = (cost, plans)

    for seed in sorted(seeds, key=lambda s: sorted(s, key=_key_order)):
        consider(evaluate(seed), scored)
    if not scored:
        raise RuntimeError("no feasible seed configuration found")
    beam = sorted(scored.items(), key=lambda kv: kv[1][0])[:beam_width]
    best_seen = beam[0]
    previous = beam[0][1][0]
    log(f"[mint] {len(seeds)} seeds, {len(scored)} feasible; best seed cost {previous:.1f}")

    for it in range(max_iters):
        proposals: dict[frozenset, tuple[float, dict]] = {}
        for config, _ in beam:
            for cand in candidates:
                if cand in config:
                    continue
                consider(evaluate(config | {cand}), proposals)
        if not proposals:
            break
        beam = sorted(proposals.items(), key=lambda kv: kv[1][0])[:beam_width]
        if beam[0][1][0] < best_seen[1][0]:
            best_seen = beam[0]
        improvement = (previous - beam[0][1][0]) / max(previous, 1e-12)
        log(f"[mint] beam iter {it + 1}: best {beam[0][1][0]:.1f} improvement {improvement:.4f} plans={planner.n_planned}")
        previous = beam[0][1][0]
        if improvement <= improvement_threshold:
            break

    if local_search:
        improved = True
        while improved:
            improved = False
            config, (cost, _) = best_seen
            best_move = None
            for out in sorted(config, key=_key_order):
                for cand in candidates:
                    if cand in config:
                        continue
                    res = evaluate((config - {out}) | {cand})
                    if res is not None and res[0] < cost - 1e-9 and (best_move is None or res[0] < best_move[0]):
                        best_move = res
            if best_move is not None:
                best_seen = (best_move[1], (best_move[0], best_move[2]))
                improved = True
                log(f"[mint] local swap: cost {best_move[0]:.1f}")

    if rerank_measured > 0:
        shortlist = dict(sorted(seen.items(), key=lambda kv: kv[1][0])[:rerank_measured])
        shortlist.setdefault(best_seen[0], best_seen[1])
        best_measured = None
        for config, (est, plans) in sorted(shortlist.items(), key=lambda kv: kv[1][0]):
            measured = sum(t.probability * planner.measured_cost(t.name, plans[t.name])[1] for t in templates)
            if best_measured is None or measured < best_measured[0]:
                best_measured = (measured, config, est, plans)
        log(
            f"[mint] measured re-rank of {len(shortlist)} configurations: picked "
            f"{sorted(index_name(k) for k in best_measured[1])} measured {best_measured[0]:.1f} est {best_measured[2]:.1f}"
        )
        best_seen = (best_measured[1], (best_measured[2], best_measured[3]))

    config, (cost, plans) = best_seen
    return set(config), cost, plans


def run_experiment(
    workload_name: str,
    output: Path,
    *,
    train_seed: int = TRAIN_QUERY_SEED,
    n_train: int = N_TRAIN_QUERIES,
    recall_margin: float = 0.03,
    beam_width: int = 5,
    seed_limit: int = 2,
    improvement_threshold: float = 0.05,
    max_iters: int = 8,
    local_search: bool = False,
    rerank_measured: int = 0,
    trim_deterministic: bool = False,
) -> None:
    faiss.omp_set_num_threads(1)
    start = time.perf_counter()

    def log(msg):
        print(f"{msg} [{time.perf_counter() - start:.0f}s]", flush=True)

    log(f"[mint] building workload={workload_name} train_seed={train_seed} n_train={n_train}")
    workload = make_workload(workload_name, n_train=n_train, train_seed=train_seed)
    templates = workload.templates
    gt = {t.name: exact_topk_batch(workload.columns, workload.train_queries[t.name], workload.top_k) for t in templates}
    keys = set(all_candidate_indexes([t.columns for t in templates], subset_diff=2))
    keys |= {(i,) for i in range(len(workload.columns))}
    keys |= {t.columns for t in templates}
    log(f"[mint] building {len(keys)} HNSW indexes")
    indexes = {k: HnswIndex(workload.columns, k) for k in sorted(keys, key=_key_order)}
    target = min(1.0, workload.recall_threshold + recall_margin)
    planner = Planner(workload, indexes, gt, target, seed=train_seed)

    log(f"[mint] beam search (planning recall target {target:.3f})")
    config, est_cost, mint_plans = beam_search_configuration(
        workload,
        planner,
        seed_limit=seed_limit,
        beam_width=beam_width,
        improvement_threshold=improvement_threshold,
        max_iters=max_iters,
        local_search=local_search,
        rerank_measured=rerank_measured,
        log=log,
    )
    log(f"[mint] chosen configuration {sorted(index_name(k) for k in config)} est cost {est_cost:.1f}")

    exact_keys = sorted({t.columns for t in templates}, key=_key_order)
    per_column = {t.name: planner.plan(t.name, tuple((c,) for c in t.columns)) for t in templates}
    per_query = {t.name: planner.plan(t.name, (t.columns,)) for t in templates}
    methods = {"per_column": per_column, "per_query": per_query, "mint": mint_plans}
    configs = {
        "per_column": [(i,) for i in range(len(workload.columns))],
        "per_query": exact_keys,
        "mint": sorted(config, key=_key_order),
    }
    results = {}
    for name, plans in methods.items():
        if any(p is None for p in plans.values()):
            raise RuntimeError(f"{name} has an infeasible template plan")
        out_plans = {}
        diags = {}
        for t in templates:
            plan, diag = planner.repair(t.name, plans[t.name])
            if trim_deterministic and name == "mint":
                plan = planner.trim_deterministic(t.name, plan, workload.recall_threshold + 0.005)
                covered, ncand = planner.measure(t.name, plan.indexes, plan.eks)
                diag = {
                    "train_recall": float(covered.mean()),
                    "train_min_instance_recall": float(covered.mean(axis=1).min()),
                    "train_mean_candidates": float(ncand.mean()),
                }
            out_plans[t.name] = {"indexes": [index_name(k) for k in plan.indexes], "ek": [int(e) for e in plan.eks]}
            diags[t.name] = {"estimated_cost": plan.estimated_cost, **diag}
        results[name] = {
            "configuration": [index_name(k) for k in configs[name]],
            "plans": out_plans,
            "train_diagnostics": diags,
        }
        log(f"[mint] {name}: " + ", ".join(f"{tn}={p['indexes']}:{p['ek']}" for tn, p in out_plans.items()))

    payload = {
        "workload": workload.name,
        "n_rows": workload.n_rows,
        "top_k": workload.top_k,
        "recall_threshold": workload.recall_threshold,
        "storage_budget": workload.storage_budget,
        "train_seed": int(train_seed),
        "train_queries_per_template": int(n_train),
        "planning_recall_target": target,
        "search": {
            "beam_width": beam_width,
            "seed_limit": seed_limit,
            "improvement_threshold": improvement_threshold,
            "max_iters": max_iters,
            "local_search": local_search,
            "rerank_measured": rerank_measured,
            "trim_deterministic": trim_deterministic,
            "plans_computed": planner.n_planned,
        },
        "runtime_seconds": time.perf_counter() - start,
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"[mint] wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", default="naive_small", choices=["naive_small", "bicomplex_small"])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--train-seed", type=int, default=TRAIN_QUERY_SEED)
    parser.add_argument("--n-train", type=int, default=N_TRAIN_QUERIES)
    parser.add_argument("--recall-margin", type=float, default=0.03)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--seed-limit", type=int, default=2)
    parser.add_argument("--improvement-threshold", type=float, default=0.05)
    parser.add_argument("--max-iters", type=int, default=8)
    parser.add_argument("--local-search", action="store_true")
    parser.add_argument("--rerank-measured", type=int, default=0)
    parser.add_argument("--trim-deterministic", action="store_true")
    args = parser.parse_args()
    output = args.output
    if output is None:
        suffix = "" if args.workload == "naive_small" else f"_{args.workload.split('_')[0]}"
        output = Path(f"checkpoints/mint{suffix}_results.json")
    run_experiment(
        args.workload,
        output,
        train_seed=args.train_seed,
        n_train=args.n_train,
        recall_margin=args.recall_margin,
        beam_width=args.beam_width,
        seed_limit=args.seed_limit,
        improvement_threshold=args.improvement_threshold,
        max_iters=args.max_iters,
        local_search=args.local_search,
        rerank_measured=args.rerank_measured,
        trim_deterministic=args.trim_deterministic,
    )


if __name__ == "__main__":
    main()
