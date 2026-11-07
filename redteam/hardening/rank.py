"""Reporting only: rank of a scored case relative to the reference row.

rank = sum over experiments of weight * sum over metrics of
       coefficient * (case metric - reference metric),
with weights/coefficients from scoring/reference.json and the reference row
("mint") from scoring/scores_{test,train}.json.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def rank(case_scores: dict, ref_scores: dict, method: str = "mint", ref_method: str = "mint") -> dict:
    spec = json.loads((ROOT / "scoring/reference.json").read_text())
    terms = {}
    total = 0.0
    for exp, cfg in spec["experiments"].items():
        actual = case_scores["experiments"][exp]["results"][method]
        target = ref_scores["experiments"][exp]["results"][ref_method]
        w = float(cfg.get("weight", 1.0))
        t = {m: w * float(e["coefficient"]) * (float(actual[m]) - float(target[m])) for m, e in cfg["metrics"].items()}
        terms[exp] = t
        total += sum(t.values())
    return {"method": method, "rank_delta": total, "terms": terms}


def rank_file(case_path: Path, slice_name: str, method: str = "mint") -> dict:
    case_scores = json.loads(case_path.read_text())
    ref_scores = json.loads((ROOT / f"scoring/scores_{slice_name}.json").read_text())
    return rank(case_scores, ref_scores, method=method)
