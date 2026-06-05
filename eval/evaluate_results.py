from __future__ import annotations

import argparse
import json
from pathlib import Path


def ratio(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def build_scores(result_path: Path) -> dict:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    res = payload["results"]
    experiment = payload["workload"]
    scores = {
        "experiments": {
            experiment: {
                "results": {
                    "per_column": {
                        "weighted_cost": res["per_column"]["weighted_cost"],
                        "mean_recall_at_k": res["per_column"]["mean_recall_at_k"],
                        "min_recall_at_k": res["per_column"]["min_recall_at_k"],
                        "storage_indexes": res["per_column"]["storage_indexes"],
                    },
                    "per_query": {
                        "weighted_cost": res["per_query"]["weighted_cost"],
                        "mean_recall_at_k": res["per_query"]["mean_recall_at_k"],
                        "min_recall_at_k": res["per_query"]["min_recall_at_k"],
                        "storage_indexes": res["per_query"]["storage_indexes"],
                    },
                    "mint": {
                        "weighted_cost": res["mint"]["weighted_cost"],
                        "mean_recall_at_k": res["mint"]["mean_recall_at_k"],
                        "min_recall_at_k": res["mint"]["min_recall_at_k"],
                        "storage_indexes": res["mint"]["storage_indexes"],
                        "speedup_vs_per_column": ratio(res["per_column"]["weighted_cost"], res["mint"]["weighted_cost"]),
                    },
                }
            }
        }
    }
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_path", nargs="?", default="checkpoints/mint_results.json")
    parser.add_argument("--output", default="scoring/scores.json")
    args = parser.parse_args()
    scores = build_scores(Path(args.result_path))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
