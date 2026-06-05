#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

WORKLOAD="${1:-naive_small}"
N_ROWS="${2:-8000}"
OUTPUT="${3:-checkpoints/mint_results.json}"

python -m method.mint --workload "$WORKLOAD" --n-rows "$N_ROWS" --output "$OUTPUT"
bash scripts/evaluate.sh "$OUTPUT"
