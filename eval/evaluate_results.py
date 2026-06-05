from __future__ import annotations

import argparse
import json
from pathlib import Path


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


def _normalize(x: np.ndarray) -> np.ndarray:
    import numpy as np

    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _parse_index_name(name: str) -> tuple[int, ...]:
    if not name.startswith("x"):
        raise ValueError(f"invalid index name {name!r}")
    return tuple(int(part) - 1 for part in name[1:].split(",") if part)


class EvalHnswIndex:
    def __init__(self, columns: tuple[np.ndarray, ...], col_ids: tuple[int, ...], ef: int = 300):
        import faiss
        import numpy as np

        self.col_ids = tuple(col_ids)
        self.dim = int(sum(columns[c].shape[1] for c in self.col_ids))
        self.matrix = _normalize(np.concatenate([columns[c] for c in self.col_ids], axis=1).astype("float32"))
        self.index = faiss.IndexHNSWFlat(self.dim, 16, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = 120
        self.index.hnsw.efSearch = ef
        self.index.add(self.matrix)

    def query_vector(self, query: Query) -> np.ndarray:
        import numpy as np

        parts = [query.vectors[query.columns.index(c)] for c in self.col_ids]
        q = np.concatenate(parts).astype("float32")
        return q / max(float(np.linalg.norm(q)), 1e-12)

    def search(self, query: Query, k: int) -> np.ndarray:
        _, labels = self.index.search(self.query_vector(query)[None, :], min(k, self.matrix.shape[0]))
        return labels[0].astype("int64")


def _collect_plan_indexes(payload: dict) -> set[tuple[int, ...]]:
    keys: set[tuple[int, ...]] = set()
    for method_payload in payload["results"].values():
        for plan in method_payload.get("plans", {}).values():
            keys.update(_parse_index_name(name) for name in plan["indexes"])
    if not keys:
        raise ValueError("artifact contains no executable query plans")
    return keys


def _score_method(
    method_payload: dict,
    workload,
    indexes: dict[tuple[int, ...], EvalHnswIndex],
    gt_by_query: dict[str, np.ndarray],
) -> dict[str, float]:
    plans = method_payload.get("plans")
    if not isinstance(plans, dict):
        raise ValueError("method artifact is missing query plans")

    recalls = []
    candidate_counts = []
    weighted_cost = 0.0
    used_indexes: set[tuple[int, ...]] = set()
    for query in workload.queries:
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
            used_indexes.add(key)
            candidates.update(int(item) for item in indexes[key].search(query, ek))

        gt = set(int(item) for item in gt_by_query[query.name])
        recall = len(gt & candidates) / len(gt)
        candidate_counts.append(len(candidates))
        query_dim = int(sum(workload.columns[c].shape[1] for c in query.columns))
        query_cost = sum(indexes[key].dim * ek for key, ek in zip(index_keys, eks)) + query_dim * len(candidates)
        weighted_cost += query.probability * float(query_cost)
        recalls.append(recall)

    mean_recall = float(sum(recalls) / len(recalls))
    min_recall = float(min(recalls))
    configuration = method_payload.get("configuration")
    if isinstance(configuration, list):
        storage_indexes = float(len({_parse_index_name(str(name)) for name in configuration}))
    else:
        storage_indexes = float(len(used_indexes))
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


def build_experiment_scores(result_path: Path) -> tuple[str, dict]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    validate_protocol(payload, result_path)

    from data.synthetic_multivector import exact_topk, make_workload

    workload = make_workload(payload["workload"], n_rows=int(payload["n_rows"]))
    if workload.top_k != payload["top_k"]:
        raise ValueError(f"{result_path} top_k does not match generated workload")
    gt_by_query = {q.name: exact_topk(workload.columns, q, workload.top_k) for q in workload.queries}
    index_keys = _collect_plan_indexes(payload)
    indexes = {key: EvalHnswIndex(workload.columns, key) for key in sorted(index_keys, key=lambda x: (len(x), x))}

    raw = {name: _score_method(method, workload, indexes, gt_by_query) for name, method in payload["results"].items()}
    per_column_cost = raw["per_column"]["weighted_cost"]
    results = {}
    for name, metrics in raw.items():
        results[name] = {
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
    return payload["workload"], {
        "results": results
    }


def build_scores(result_paths: list[Path]) -> dict:
    experiments = {}
    for path in result_paths:
        name, scores = build_experiment_scores(path)
        experiments[name] = scores
    return {"experiments": experiments}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_paths", nargs="*", default=["checkpoints/mint_results.json"])
    parser.add_argument("--output", default="scoring/scores.json")
    args = parser.parse_args()
    scores = build_scores([Path(p) for p in args.result_paths])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
