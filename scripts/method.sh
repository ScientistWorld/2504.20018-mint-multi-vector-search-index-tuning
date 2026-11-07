#!/bin/bash
# Run the paper's proposed method (MINT index tuning) on one workload and score
# the artifact on the visible slice.
#
# Usage:
#   bash scripts/method.sh [naive_small|bicomplex_small] [output]

set -e

cd /home/user

WORKLOAD="${1:-naive_small}"
if [ "$WORKLOAD" = "naive_small" ]; then DEFAULT_OUT=checkpoints/mint_results.json; else DEFAULT_OUT=checkpoints/mint_bicomplex_results.json; fi
OUTPUT="${2:-$DEFAULT_OUT}"

python -m method.mint --workload "$WORKLOAD" --output "$OUTPUT"
bash scripts/evaluate.sh "$OUTPUT"
