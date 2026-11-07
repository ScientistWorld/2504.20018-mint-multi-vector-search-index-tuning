"""Held-out test scoring for multi-vector index tuning submissions.

Reads submitted result artifacts (``checkpoints/*_results.json``), executes the
submitted per-template query plans on held-out query instances with a fixed
HNSW engine, and recomputes every metric. Nothing the submission reports
(costs, recalls, configuration, dataset size, thresholds, baselines) is
trusted: the protocol, database, held-out queries, ground truth, ANN engine and
the PerColumn comparator are all fixed here.

A run that violates a constraint (worst template's mean recall@k below the
floor, more distinct indexes than the budget) or submits an invalid plan is
scored as failed: every metric is set to a value at least as bad as the worst
value any feasible run can obtain.

An artifact must also be self-consistent. Declared protocol fields (n_rows,
top_k, recall_threshold, storage_budget) must equal the fixed protocol, any
entry under a reserved baseline name must be a valid plan set, and a declared
``configuration`` must contain every index the plans scan. Otherwise every
method in the artifact fails on that workload. Storage counts the declared
configuration together with the indexes the plans use.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import eval.test.hidden_workload as hw

SLICE_NAME = "test"
RESERVED_BASELINES = ("per_column", "per_query")
PROTOCOL_KEYS = ("n_rows", "top_k", "recall_threshold", "storage_budget")
BASELINES_PATH = Path(__file__).with_name("baselines.json")
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 120
HNSW_EF_SEARCH = 300
FAIL_COST_FLOOR = 100.0
METRICS = (
    "weighted_cost",
    "mean_recall_at_k",
    "min_recall_at_k",
    "recall_floor_violation",
    "storage_indexes",
    "storage_budget_violation",
    "mean_candidate_multiplier_at_k",
    "max_candidate_multiplier_at_k",
    "speedup_vs_per_column",
)


class PlanError(ValueError):
    pass


def _faiss():
    import faiss

    faiss.omp_set_num_threads(1)
    return faiss


class EvalHnswIndex:
    def __init__(self, columns, key: tuple[int, ...]):
        import numpy as np

        faiss = _faiss()
        self.key = tuple(key)
        self.dim = int(sum(columns[c].shape[1] for c in self.key))
        m = np.concatenate([columns[c] for c in self.key], axis=1).astype("float32")
        m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)
        self.n_rows = int(m.shape[0])
        self.index = faiss.IndexHNSWFlat(self.dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        self.index.hnsw.efSearch = HNSW_EF_SEARCH
        self.index.add(np.ascontiguousarray(m, dtype="float32"))

    def query_matrix(self, queries):
        import numpy as np

        cols = queries[0].columns
        parts = [np.stack([q.vectors[cols.index(c)] for q in queries]) for c in self.key]
        qm = np.concatenate(parts, axis=1).astype("float32")
        qm /= np.maximum(np.linalg.norm(qm, axis=1, keepdims=True), 1e-12)
        return np.ascontiguousarray(qm, dtype="float32")

    def search(self, qvec, k: int) -> list[int]:
        k = int(min(k, self.n_rows))
        if k <= 0:
            return []
        _, labels = self.index.search(qvec[None, :], k)
        return [int(x) for x in labels[0] if x >= 0]


class Experiment:
    def __init__(self, name: str):
        spec = hw.SPECS[name]
        self.name = name
        self.n_rows = int(spec["n_rows"])
        self.k = int(spec["top_k"])
        self.theta = float(spec["recall_threshold"])
        self.budget = int(spec["storage_budget"])
        self.columns = hw.database(name)
        self.n_cols = len(self.columns)
        self.templates = hw.templates(name)
        self.queries = hw.held_out_queries(name, self.columns)
        self.gt = {t.name: hw.exact_topk(self.columns, self.queries[t.name], self.k) for t in self.templates}
        self.qdim = {t.name: int(sum(self.columns[c].shape[1] for c in t.columns)) for t in self.templates}
        self._indexes: dict[tuple[int, ...], EvalHnswIndex] = {}
        self._qmats: dict = {}

    def index(self, key):
        if key not in self._indexes:
            self._indexes[key] = EvalHnswIndex(self.columns, key)
        return self._indexes[key]

    def qmat(self, tname, key):
        if (tname, key) not in self._qmats:
            self._qmats[(tname, key)] = self.index(key).query_matrix(self.queries[tname])
        return self._qmats[(tname, key)]

    def max_feasible_raw_cost(self) -> float:
        # Any valid plan uses distinct indexes over subsets of the template's
        # columns with ek <= n_rows, so its per-query cost is at most
        # (sum of dims over all non-empty subsets + query dim) * n_rows.
        return float(
            sum(t.probability * (2 ** (len(t.columns) - 1) + 1) * self.qdim[t.name] * self.n_rows for t in self.templates)
        )


def _parse_index_name(name, n_cols: int) -> tuple[int, ...]:
    if not isinstance(name, str) or not name.startswith("x"):
        raise PlanError(f"invalid index name {name!r}")
    try:
        cols = [int(p) for p in name[1:].split(",")]
    except ValueError as exc:
        raise PlanError(f"invalid index name {name!r}") from exc
    if not cols or len(set(cols)) != len(cols) or any(c < 1 or c > n_cols for c in cols):
        raise PlanError(f"invalid index name {name!r}")
    return tuple(sorted(c - 1 for c in cols))


def parse_plans(method_payload, exp: Experiment) -> dict:
    plans = method_payload.get("plans") if isinstance(method_payload, dict) else None
    if not isinstance(plans, dict):
        raise PlanError("missing 'plans' mapping")
    parsed = {}
    for t in exp.templates:
        plan = plans.get(t.name)
        if not isinstance(plan, dict):
            raise PlanError(f"missing plan for {t.name}")
        names, eks = plan.get("indexes"), plan.get("ek")
        if not isinstance(names, list) or not isinstance(eks, list) or not names or len(names) != len(eks):
            raise PlanError(f"plan for {t.name} needs equal-length non-empty 'indexes' and 'ek' lists")
        keys, vals = [], []
        for name, ek in zip(names, eks):
            key = _parse_index_name(name, exp.n_cols)
            if not set(key) <= set(t.columns):
                raise PlanError(f"plan for {t.name} uses {name} outside template columns")
            if key in keys:
                raise PlanError(f"plan for {t.name} lists {name} twice")
            if isinstance(ek, bool) or not isinstance(ek, int):
                raise PlanError(f"plan for {t.name} has non-integer ek {ek!r}")
            if ek < 0 or ek > exp.n_rows:
                raise PlanError(f"plan for {t.name} has ek {ek} outside [0, {exp.n_rows}]")
            keys.append(key)
            vals.append(int(ek))
        parsed[t.name] = (tuple(keys), tuple(vals))
    return parsed


def execute(parsed: dict, exp: Experiment) -> dict:
    per_template = {}
    used: set = set()
    raw_cost = 0.0
    for t in exp.templates:
        keys, eks = parsed[t.name]
        active = [(k, e) for k, e in zip(keys, eks) if e > 0]
        used.update(k for k, _ in active)
        scan = sum(exp.index(k).dim * e for k, e in active)
        recalls, costs, ncands = [], [], []
        gt = exp.gt[t.name]
        for n in range(len(exp.queries[t.name])):
            cand: set[int] = set()
            for k, e in active:
                cand.update(exp.index(k).search(exp.qmat(t.name, k)[n], e))
            hits = sum(1 for x in gt[n].tolist() if x in cand)
            recalls.append(hits / gt.shape[1])
            ncands.append(len(cand))
            costs.append(scan + exp.qdim[t.name] * len(cand))
        per_template[t.name] = {
            "recall": sum(recalls) / len(recalls),
            "min_instance_recall": min(recalls),
            "mean_candidates": sum(ncands) / len(ncands),
            "mean_cost": sum(costs) / len(costs),
        }
        raw_cost += t.probability * per_template[t.name]["mean_cost"]
    return {"per_template": per_template, "raw_cost": raw_cost, "used": used}


def _index_label(key: tuple[int, ...]) -> str:
    return "x" + ",".join(str(c + 1) for c in key)


def declared_configuration(method_payload, exp: Experiment) -> set | None:
    """Index set a method declares it builds (None when not declared)."""
    if not isinstance(method_payload, dict) or "configuration" not in method_payload:
        return None
    conf = method_payload["configuration"]
    if not isinstance(conf, list):
        raise PlanError("'configuration' must be a list of index names")
    keys: set = set()
    for name in conf:
        key = _parse_index_name(name, exp.n_cols)
        if key in keys:
            raise PlanError(f"configuration lists {name} twice")
        keys.add(key)
    return keys


def _artifact_context(payload: dict, results: dict) -> dict:
    """Parts of an artifact that bind every method in it."""
    return {
        "declared": {k: payload[k] for k in PROTOCOL_KEYS if k in payload},
        "reserved": {r: results[r] for r in RESERVED_BASELINES if r in results},
    }


def artifact_problems(context: dict, exp: Experiment) -> list[str]:
    fixed = {"n_rows": exp.n_rows, "top_k": exp.k, "recall_threshold": exp.theta, "storage_budget": exp.budget}
    problems = []
    for key, value in context["declared"].items():
        numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
        if not (numeric and math.isfinite(float(value)) and float(value) == float(fixed[key])):
            problems.append(f"artifact declares {key}={value!r}; the protocol fixes {fixed[key]!r}")
    for name, entry in context["reserved"].items():
        try:
            parse_plans(entry, exp)
        except PlanError as exc:
            problems.append(f"artifact's {name} entry is not a valid plan set ({exc})")
    return problems


def failed_metrics(exp: Experiment, fail_cost: float, storage: int | None) -> dict:
    s = float(max(storage or 0, exp.budget))
    worst_mult = exp.n_rows / exp.k
    return {
        "weighted_cost": fail_cost,
        "mean_recall_at_k": 0.0,
        "min_recall_at_k": 0.0,
        "recall_floor_violation": exp.theta,
        "storage_indexes": s,
        "storage_budget_violation": float(max(s - exp.budget, 1.0)),
        "mean_candidate_multiplier_at_k": worst_mult,
        "max_candidate_multiplier_at_k": worst_mult,
        "speedup_vs_per_column": 0.0,
    }


def score_method(
    method_payload, exp: Experiment, pc_raw: float, fail_cost: float, problems=()
) -> tuple[dict, dict]:
    if problems:
        return failed_metrics(exp, fail_cost, None), {"status": "failed", "reasons": list(problems)}
    try:
        parsed = parse_plans(method_payload, exp)
        declared = declared_configuration(method_payload, exp)
    except PlanError as exc:
        return failed_metrics(exp, fail_cost, None), {"status": "failed", "reasons": [f"invalid plan: {exc}"]}
    run = execute(parsed, exp)
    used = run["used"]
    storage = len(used | declared) if declared is not None else len(used)
    pt = run["per_template"]
    recalls = [pt[t.name]["recall"] for t in exp.templates]
    mults = [pt[t.name]["mean_candidates"] / exp.k for t in exp.templates]
    raw = {
        "raw_weighted_cost": run["raw_cost"],
        "storage": storage,
        "per_template": pt,
    }
    reasons = []
    if declared is not None and not used <= declared:
        missing = sorted(_index_label(k) for k in used - declared)
        reasons.append(f"plans scan indexes missing from the declared configuration: {missing}")
    if min(recalls) < exp.theta - 1e-9:
        worst = min(exp.templates, key=lambda t: pt[t.name]["recall"])
        reasons.append(f"recall floor: {worst.name} mean recall@{exp.k} {pt[worst.name]['recall']:.4f} < {exp.theta}")
    if storage > exp.budget:
        reasons.append(f"storage: {storage} indexes > budget {exp.budget}")
    if not (run["raw_cost"] > 0 and math.isfinite(run["raw_cost"])):
        reasons.append("non-positive or non-finite cost")
    if reasons:
        return failed_metrics(exp, fail_cost, storage), {"status": "failed", "reasons": reasons, **raw}
    metrics = {
        "weighted_cost": run["raw_cost"] / pc_raw,
        "mean_recall_at_k": sum(recalls) / len(recalls),
        "min_recall_at_k": min(recalls),
        "recall_floor_violation": max(0.0, exp.theta - min(recalls)),
        "storage_indexes": float(storage),
        "storage_budget_violation": float(max(0, storage - exp.budget)),
        "mean_candidate_multiplier_at_k": sum(mults) / len(mults),
        "max_candidate_multiplier_at_k": max(mults),
        "speedup_vs_per_column": pc_raw / run["raw_cost"],
    }
    return metrics, {"status": "ok", **raw}


def _load_baselines() -> dict:
    if not BASELINES_PATH.is_file():
        raise SystemExit(f"missing fixed baseline plans: {BASELINES_PATH}")
    return json.loads(BASELINES_PATH.read_text(encoding="utf-8"))


def _load_artifacts(result_paths: list[Path]) -> tuple[dict, list]:
    grouped: dict[str, dict[str, object]] = {name: {} for name in hw.SPECS}
    ignored = []
    for path in result_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            ignored.append(f"{path}: unreadable ({exc})")
            continue
        workload = payload.get("workload") if isinstance(payload, dict) else None
        results = payload.get("results") if isinstance(payload, dict) else None
        if workload not in hw.SPECS or not isinstance(results, dict):
            ignored.append(f"{path}: unknown workload {workload!r} or missing results")
            continue
        methods = grouped[workload]
        context = _artifact_context(payload, results)
        for method_name, method_payload in results.items():
            if method_name in RESERVED_BASELINES:
                continue  # baselines are fixed by the evaluator (entries are only validated)
            entry = {"payload": method_payload, "context": context}
            target = str(method_name)
            if target in methods:
                if methods[target] == entry:
                    continue
                target = f"{method_name}__{path.stem}"
                if target in methods:
                    raise ValueError(f"duplicate method name after disambiguation: {target}")
            methods[target] = entry
    return grouped, ignored


def build_scores(result_paths: list[Path]) -> dict:
    grouped, ignored = _load_artifacts(result_paths)
    baselines = _load_baselines()
    all_methods = sorted({m for methods in grouped.values() for m in methods})
    if not all_methods:
        raise SystemExit("no submitted methods found in the result artifacts")
    experiments = {}
    for name in sorted(hw.SPECS):
        exp = Experiment(name)
        fixed = baselines[name]
        pc_parsed = parse_plans(fixed["per_column"], exp)
        pc_run = execute(pc_parsed, exp)
        pc_raw = pc_run["raw_cost"]
        fail_cost = max(FAIL_COST_FLOOR, 2.0 * exp.max_feasible_raw_cost() / pc_raw)
        results, raw = {}, {}
        results["per_column"], raw["per_column"] = score_method(fixed["per_column"], exp, pc_raw, fail_cost)
        _, raw["per_query_over_budget_reference"] = score_method(fixed["per_query"], exp, pc_raw, fail_cost)
        for method in all_methods:
            entry = grouped[name].get(method)
            if entry is None:
                results[method] = failed_metrics(exp, fail_cost, None)
                raw[method] = {"status": "failed", "reasons": [f"no {name} artifact for this method"]}
            else:
                problems = artifact_problems(entry["context"], exp)
                results[method], raw[method] = score_method(entry["payload"], exp, pc_raw, fail_cost, problems)
        experiments[name] = {"results": results, "_raw": raw}
    return {
        "_slice": SLICE_NAME,
        "_protocol": {
            name: {k: v for k, v in spec.items() if k in ("n_rows", "top_k", "recall_threshold", "storage_budget")}
            | {"held_out_queries_per_template": hw.N_HELD_OUT}
            for name, spec in hw.SPECS.items()
        },
        "_ignored_artifacts": ignored,
        "experiments": experiments,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_paths", nargs="*")
    parser.add_argument("--output", default="scoring/scores_test.json")
    args = parser.parse_args()
    if not args.result_paths:
        raise SystemExit("no result artifacts supplied")
    scores = build_scores([Path(p) for p in args.result_paths])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
