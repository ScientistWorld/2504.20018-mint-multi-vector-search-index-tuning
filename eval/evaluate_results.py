from __future__ import annotations

import argparse
import json
from pathlib import Path


def ratio(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def build_experiment_scores(result_path: Path) -> tuple[str, dict]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    res = payload["results"]
    experiment = payload["workload"]
    per_column_cost = res["per_column"]["weighted_cost"]
    return experiment, {
        "results": {
            "per_column": {
                "weighted_cost": ratio(res["per_column"]["weighted_cost"], per_column_cost),
                "mean_recall_at_k": res["per_column"]["mean_recall_at_k"],
                "min_recall_at_k": res["per_column"]["min_recall_at_k"],
                "storage_indexes": res["per_column"]["storage_indexes"],
            },
            "mint": {
                "weighted_cost": ratio(res["mint"]["weighted_cost"], per_column_cost),
                "mean_recall_at_k": res["mint"]["mean_recall_at_k"],
                "min_recall_at_k": res["mint"]["min_recall_at_k"],
                "storage_indexes": res["mint"]["storage_indexes"],
                "speedup_vs_per_column": ratio(res["per_column"]["weighted_cost"], res["mint"]["weighted_cost"]),
            }
        }
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
