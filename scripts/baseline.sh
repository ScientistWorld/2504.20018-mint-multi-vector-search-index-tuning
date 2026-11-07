#!/bin/bash
# Plan the paper's PerColumn / PerQuery baselines (together with the tuned
# configuration) with the shared planner and score the artifact on the visible
# slice. The scored PerColumn comparator is the evaluator's frozen plan set in
# eval/*/baselines.json.
#
# Usage:
#   bash scripts/baseline.sh [naive_small|bicomplex_small] [output]

set -e

cd /home/user

WORKLOAD="${1:-naive_small}"
OUTPUT="${2:-checkpoints/baseline_${WORKLOAD}_results.json}"

python -m baseline.run_baselines --workload "$WORKLOAD" --output "$OUTPUT"
bash scripts/evaluate.sh "$OUTPUT"
