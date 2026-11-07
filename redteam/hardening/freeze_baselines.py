"""One-time: freeze the PerColumn / PerQuery comparator plans into eval/.

The comparators are planned once by the reproduced planner (single-column
indexes per query template, and one exact index per template) on the public
training instances. After freezing they belong to the evaluator: submitted
artifacts can no longer move the normalization baseline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "naive_small": ROOT / "checkpoints/mint_results.json",
    "bicomplex_small": ROOT / "checkpoints/mint_bicomplex_results.json",
}
TARGETS = (ROOT / "eval/train/baselines.json", ROOT / "eval/test/baselines.json")


def main(force: bool = False) -> None:
    if all(p.is_file() for p in TARGETS) and not force:
        print("baselines already frozen; leaving them unchanged")
        return
    frozen = {
        "_note": "Fixed comparator plans (per query template: index names and ek). "
        "per_column = one single-column index per column; per_query = one exact index per template "
        "(exceeds the storage budget; reported for reference only). Planned once on the public "
        "training instances by the reproduced planner with the same recall allowance as the reference.",
    }
    for exp, path in SOURCES.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["workload"] == exp, (path, payload["workload"])
        frozen[exp] = {
            name: {"plans": payload["results"][name]["plans"]} for name in ("per_column", "per_query")
        }
    text = json.dumps(frozen, indent=2) + "\n"
    for target in TARGETS:
        target.write_text(text, encoding="utf-8")
        print(f"wrote {target}")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
