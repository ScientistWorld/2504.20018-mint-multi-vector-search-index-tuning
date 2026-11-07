"""Reporting only: never imported by a submitted method."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def rank(scores, method="mint"):
    spec = json.loads((ROOT / "original/reference.json").read_text())
    ref = json.loads((ROOT / "original/scores_test.json").read_text())
    terms = {}
    for experiment, config in spec["experiments"].items():
        actual = scores["experiments"][experiment]["results"][method]
        target = ref["experiments"][experiment]["results"]["mint"]
        weight = config.get("weight", 1)
        terms[experiment] = {
            metric: weight * entry["coefficient"] * (actual[metric] - target[metric])
            for metric, entry in config["metrics"].items()
        }
    return {"method": method, "rank_delta": sum(sum(v.values()) for v in terms.values()), "terms": terms}


def summarize():
    completed = {}
    for path in sorted(ROOT.glob("*/scores_test.json")):
        if path.parent.name == "original":
            continue
        scores = json.loads(path.read_text())
        value = rank(scores)
        value["metrics"] = scores["experiments"]["naive_small"]["results"]["mint"]
        value["per_column"] = rank(scores, "per_column")
        (path.parent / "rank.json").write_text(json.dumps(value, indent=2) + "\n")
        completed[path.parent.name] = value
    seeds = [v["rank_delta"] for k, v in completed.items() if k.startswith("reference_seed_")]
    summary = {"cases": completed, "seed_spread": max(seeds) - min(seeds) if seeds else None}
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    summarize()
