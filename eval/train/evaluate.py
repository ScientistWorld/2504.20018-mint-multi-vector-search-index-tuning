from __future__ import annotations

import argparse
import json
from pathlib import Path


SLICE_NAME = "train"
SLICE_QUERIES = ("q1", "q4")

EXPECTED_PROTOCOL = {
    "naive_small": {
        "top_k": 100,
        "recall_threshold": 0.90,
        "storage_budget": 3,
    }
}


def ratio(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def validate_protocol(payload: dict, result_path: Path) -> None:
    experiment = payload.get("workload")
    expected = EXPECTED_PROTOCOL.get(experiment)
    if expected is None:
        return
    for key, expected_value in expected.items():
        observed = payload.get(key)
        if isinstance(expected_value, float):
            ok = abs(float(observed) - expected_value) < 1e-9
        else:
            ok = observed == expected_value
        if not ok:
            raise ValueError(
                f"{result_path} is not a valid {experiment} artifact: "
                f"expected {key}={expected_value!r}, found {observed!r}"
            )


def _normalize(x):
    import numpy as np

    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _parse_index_name(name: str) -> tuple[int, ...]:
    if not name.startswith("x"):
        raise ValueError(f"invalid index name {name!r}")
    return tuple(int(part) - 1 for part in name[1:].split(",") if part)


class EvalHnswIndex:
    def __init__(self, columns, col_ids: tuple[int, ...], ef: int = 300):
        import faiss
        import numpy as np

        self.col_ids = tuple(col_ids)
        self.dim = int(sum(columns[c].shape[1] for c in self.col_ids))
        self.matrix = _normalize(np.concatenate([columns[c] for c in self.col_ids], axis=1).astype("float32"))
        self.index = faiss.IndexHNSWFlat(self.dim, 16, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = 120
        self.index.hnsw.efSearch = ef
        self.index.add(self.matrix)

    def query_vector(self, query):
        import numpy as np

        parts = [query.vectors[query.columns.index(c)] for c in self.col_ids]
        q = np.concatenate(parts).astype("float32")
        return q / max(float(np.linalg.norm(q)), 1e-12)

    def search(self, query, k: int):
        _, labels = self.index.search(self.query_vector(query)[None, :], min(k, self.matrix.shape[0]))
        return labels[0].astype("int64")


def _collect_plan_indexes(method_payloads: dict[str, dict]) -> set[tuple[int, ...]]:
    keys: set[tuple[int, ...]] = set()
    for method_payload in method_payloads.values():
        for plan in method_payload.get("plans", {}).values():
            keys.update(_parse_index_name(name) for name in plan["indexes"])
    if not keys:
        raise ValueError("artifacts contain no executable query plans")
    return keys


def _slice_queries(workload) -> tuple:
    wanted = set(SLICE_QUERIES)
    queries = tuple(query for query in workload.queries if query.name in wanted)
    found = {query.name for query in queries}
    missing = wanted - found
    if missing:
        raise ValueError(f"{workload.name} is missing {SLICE_NAME} queries: {sorted(missing)}")
    return queries


def _score_method(
    method_payload: dict,
    workload,
    slice_queries: tuple,
    indexes: dict[tuple[int, ...], EvalHnswIndex],
    gt_by_query: dict[str, object],
) -> dict[str, float]:
    plans = method_payload.get("plans")
    if not isinstance(plans, dict):
        raise ValueError("method artifact is missing query plans")

    probability_mass = sum(float(query.probability) for query in slice_queries)
    if probability_mass <= 0:
        raise ValueError(f"{SLICE_NAME} query probabilities must have positive mass")

    recalls = []
    candidate_counts = []
    weighted_cost = 0.0
    for query in slice_queries:
        if query.name not in plans:
            raise ValueError(f"method artifact is missing plan for {query.name}")
        plan = plans[query.name]
        index_keys = tuple(_parse_index_name(name) for name in plan["indexes"])
        eks = tuple(int(ek) for ek in plan["ek"])
        if len(index_keys) != len(eks):
            raise ValueError(f"plan for {query.name} has mismatched indexes and ek values")

        candidates: set[int] = set()
        for key, ek in zip(index_keys, eks):
            if ek <= 0:
                continue
            if not set(key).issubset(query.columns):
                raise ValueError(f"plan for {query.name} uses index {key} outside query columns {query.columns}")
            candidates.update(int(item) for item in indexes[key].search(query, ek))

        gt = set(int(item) for item in gt_by_query[query.name])
        recall = len(gt & candidates) / len(gt)
        candidate_counts.append(len(candidates))
        query_dim = int(sum(workload.columns[c].shape[1] for c in query.columns))
        query_cost = sum(indexes[key].dim * ek for key, ek in zip(index_keys, eks)) + query_dim * len(candidates)
        weighted_cost += (float(query.probability) / probability_mass) * float(query_cost)
        recalls.append(recall)

    mean_recall = float(sum(recalls) / len(recalls))
    min_recall = float(min(recalls))
    configuration = method_payload.get("configuration")
    if isinstance(configuration, list):
        storage_indexes = float(len({_parse_index_name(str(name)) for name in configuration}))
    else:
        storage_indexes = float(
            len(
                {
                    key
                    for plan in plans.values()
                    for key in (_parse_index_name(name) for name in plan["indexes"])
                }
            )
        )
    mean_candidates = float(sum(candidate_counts) / len(candidate_counts))
    return {
        "weighted_cost": float(weighted_cost),
        "mean_recall_at_k": mean_recall,
        "min_recall_at_k": min_recall,
        "recall_floor_violation": float(max(0.0, workload.recall_threshold - min_recall)),
        "storage_indexes": storage_indexes,
        "storage_budget_violation": float(max(0.0, storage_indexes - workload.storage_budget)),
        "mean_candidate_multiplier_at_k": float(mean_candidates / workload.top_k),
        "max_candidate_multiplier_at_k": float(max(candidate_counts) / workload.top_k),
    }


def _load_payloads(result_paths: list[Path]) -> dict[str, dict]:
    grouped: dict[str, dict] = {}
    for path in result_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_protocol(payload, path)
        workload_name = payload["workload"]
        group = grouped.setdefault(
            workload_name,
            {
                "metadata": {
                    "workload": payload["workload"],
                    "n_rows": int(payload["n_rows"]),
                    "top_k": int(payload["top_k"]),
                },
                "methods": {},
            },
        )
        metadata = group["metadata"]
        if metadata["n_rows"] != int(payload["n_rows"]) or metadata["top_k"] != int(payload["top_k"]):
            raise ValueError(f"{path} does not match the {workload_name} evaluation metadata")
        for method_name, method_payload in payload.get("results", {}).items():
            target_name = method_name
            existing = group["methods"].get(target_name)
            if existing == method_payload:
                continue
            if existing is not None:
                target_name = f"{method_name}__{path.stem}"
                if target_name in group["methods"]:
                    raise ValueError(f"duplicate method name after disambiguation: {target_name}")
            group["methods"][target_name] = method_payload
    return grouped


def build_experiment_scores(workload_name: str, group: dict) -> dict:
    from data.synthetic_multivector import exact_topk, make_workload

    metadata = group["metadata"]
    workload = make_workload(workload_name, n_rows=int(metadata["n_rows"]))
    if workload.top_k != metadata["top_k"]:
        raise ValueError(f"{workload_name} top_k does not match generated workload")

    methods = group["methods"]
    if "per_column" not in methods:
        raise ValueError(f"{workload_name} artifacts must include a per_column baseline for normalized cost")

    slice_queries = _slice_queries(workload)
    gt_by_query = {q.name: exact_topk(workload.columns, q, workload.top_k) for q in slice_queries}
    index_keys = _collect_plan_indexes(methods)
    indexes = {key: EvalHnswIndex(workload.columns, key) for key in sorted(index_keys, key=lambda x: (len(x), x))}

    raw = {
        name: _score_method(method, workload, slice_queries, indexes, gt_by_query)
        for name, method in sorted(methods.items())
    }
    per_column_cost = raw["per_column"]["weighted_cost"]
    results = {}
    for name, metrics in raw.items():
        normalized = {
            "weighted_cost": ratio(metrics["weighted_cost"], per_column_cost),
            "mean_recall_at_k": metrics["mean_recall_at_k"],
            "min_recall_at_k": metrics["min_recall_at_k"],
            "recall_floor_violation": metrics["recall_floor_violation"],
            "storage_indexes": metrics["storage_indexes"],
            "storage_budget_violation": metrics["storage_budget_violation"],
            "mean_candidate_multiplier_at_k": metrics["mean_candidate_multiplier_at_k"],
            "max_candidate_multiplier_at_k": metrics["max_candidate_multiplier_at_k"],
            "speedup_vs_per_column": ratio(per_column_cost, metrics["weighted_cost"]),
        }
        results[name] = normalized
    return {"results": results}


def build_scores(result_paths: list[Path]) -> dict:
    grouped = _load_payloads(result_paths)
    experiments = {
        workload_name: build_experiment_scores(workload_name, group)
        for workload_name, group in sorted(grouped.items())
    }
    return {
        "_slice": SLICE_NAME,
        "_split_queries": list(SLICE_QUERIES),
        "experiments": experiments,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_paths", nargs="*")
    parser.add_argument("--output", default="scoring/scores_train.json")
    args = parser.parse_args()
    if not args.result_paths:
        raise SystemExit("no result artifacts supplied")
    scores = build_scores([Path(p) for p in args.result_paths])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
