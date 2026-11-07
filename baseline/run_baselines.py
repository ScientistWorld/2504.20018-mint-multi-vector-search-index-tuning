from __future__ import annotations

import argparse
from pathlib import Path

from method.mint import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", default="naive_small", choices=["naive_small", "bicomplex_small"])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or Path(f"checkpoints/baseline_{args.workload}_results.json")
    # The shared runner plans PerColumn and PerQuery with the same query planner
    # alongside the tuned configuration. Note that the evaluator scores its own
    # frozen PerColumn plans (eval/*/baselines.json) as the comparator and
    # ignores artifact entries named per_column / per_query.
    run_experiment(args.workload, output)


if __name__ == "__main__":
    main()
