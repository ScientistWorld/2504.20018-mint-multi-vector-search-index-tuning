"""Reporting-only: scientific ordering, reference bounds, and cross-task bounds."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from data.synthetic_multivector import WORKLOAD_SPECS, make_templates


def main():
    summary = json.loads((HERE / "summary.json").read_text())
    ref = json.loads((HERE / "initial/scoring/scores_test.json").read_text())
    spec = json.loads((HERE / "initial/scoring/reference.json").read_text())
    seeds = {str(seed): summary[f"reference_seed_{seed}"]["rank_delta"] for seed in (20240501, 101, 202)}
    spread = max(seeds.values()) - min(seeds.values())
    checks = {
        "reference_seed_ranks": seeds,
        "seed_spread": spread,
        "better_variants": {case: {"rank_delta": summary[case]["rank_delta"],
                                   "exceeds_seed_spread": summary[case]["rank_delta"] > spread,
                                   "all_feasible": all(v == "ok" for v in summary[case]["status"].values())}
                            for case in ("better_measured_cost", "better_more_queries")},
        "weaker_delta": summary["weaker_no_multicolumn"]["rank_delta"],
        "weaker_exceeds_seed_spread": -summary["weaker_no_multicolumn"]["rank_delta"] > spread,
        "reference_cost_speedup": {}, "reference_metrics_at_bounds": {},
        "failure_compensation_bounds": {},
        "shortcut_ties": [name for name, row in summary.items() if name.startswith("shortcut_") and abs(row["rank_delta"]) < 1e-12],
        "shortcut_gains": [name for name, row in summary.items() if name.startswith("shortcut_") and row["rank_delta"] > 1e-12],
    }
    best_possible = {}
    failed_possible = {}
    for name in WORKLOAD_SPECS:
        experiment = ref["experiments"][name]
        metrics = experiment["results"]["mint"]
        pc_raw = experiment["_raw"]["per_column"]["raw_weighted_cost"]
        # At recall >= .9 every template averages >=90 candidates. The sum
        # of retrieval depths must also be >=90, each charged at >=min dim.
        cfg = WORKLOAD_SPECS[name]
        lower_raw = sum(t.probability * 90 * (sum(cfg["dims"][c] for c in t.columns) +
                                             min(cfg["dims"][c] for c in t.columns)) for t in make_templates(name))
        lower_cost = lower_raw / pc_raw
        optimistic = dict(weighted_cost=lower_cost, speedup_vs_per_column=1/lower_cost,
                          mean_recall_at_k=1, min_recall_at_k=1, recall_floor_violation=0,
                          storage_indexes=0, storage_budget_violation=0,
                          mean_candidate_multiplier_at_k=.9, max_candidate_multiplier_at_k=.9)
        weight = spec["experiments"][name]["weight"]
        coefficients = spec["experiments"][name]["metrics"]
        best_possible[name] = weight * sum(v["coefficient"] * (optimistic[m] - metrics[m]) for m,v in coefficients.items())
        failed = summary["shortcut_empty"]["metrics"][name]
        failed_possible[name] = weight * sum(v["coefficient"] * (failed[m] - metrics[m]) for m,v in coefficients.items())
        checks["reference_cost_speedup"][name] = {
            "cost": metrics["weighted_cost"], "speedup": metrics["speedup_vs_per_column"],
            "beats_per_column": metrics["weighted_cost"] < 1,
            "recall_floor_met": metrics["min_recall_at_k"] >= .9,
            "storage_budget_met": metrics["storage_budget_violation"] == 0,
            "loose_feasible_cost_lower_bound": lower_cost,
            "optimistic_maximum_rank_contribution": best_possible[name],
        }
        checks["reference_metrics_at_bounds"][name] = {
            "absolute_optimal_bounds": ["recall_floor_violation", "storage_budget_violation"],
            "budget_ceiling": {"storage_indexes": metrics["storage_indexes"]},
            "other_metrics_at_optimal_bounds": [],
        }
    for name in WORKLOAD_SPECS:
        checks["failure_compensation_bounds"][name] = {
            "failed_workload_plus_optimistically_perfect_other_workload_rank_upper_bound":
                failed_possible[name] + sum(v for other,v in best_possible.items() if other != name),
            "bound_is_below_reference": failed_possible[name] + sum(v for other,v in best_possible.items() if other != name) < 0,
        }
    (HERE / "analysis.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == "__main__":
    main()
