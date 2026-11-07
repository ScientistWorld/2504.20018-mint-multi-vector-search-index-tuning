"""Standard evaluation entry point (``scripts/evaluate.sh`` -> scoring/scores.json).

Scores saved result artifacts on the visible slice (public training query
instances) with exactly the train-slice rules in ``eval/train/evaluate.py``.
It never imports ``method/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.train.evaluate import build_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_paths", nargs="*")
    parser.add_argument("--output", default="scoring/scores.json")
    args = parser.parse_args()
    paths = [Path(p) for p in args.result_paths] or sorted(Path("checkpoints").glob("*_results.json"))
    if not paths:
        raise SystemExit("no result artifacts found under checkpoints/*_results.json")
    scores = build_scores(paths)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
