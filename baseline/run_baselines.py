from __future__ import annotations

import argparse
from pathlib import Path

from method.mint import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", default="naive_small")
    parser.add_argument("--n-rows", type=int, default=8000)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/baseline_results.json"))
    args = parser.parse_args()
    # The shared runner emits PerColumn and PerQuery alongside MINT so the
    # evaluator can compare the same workload and ground truth.
    run_experiment(args.workload, args.n_rows, args.output)


if __name__ == "__main__":
    main()
