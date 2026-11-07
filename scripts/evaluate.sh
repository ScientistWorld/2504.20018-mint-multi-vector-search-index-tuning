#!/bin/bash
# Evaluation script — the standard way to evaluate work in this environment.
#
# Self-contained: uses only code in eval/ (never method/). It reads saved
# result artifacts, executes their per-template query plans with the fixed
# HNSW engine on the public training query instances, and writes
# scoring/scores.json (same rules as scripts/evaluate_train.sh).
#
# Usage:
#   bash scripts/evaluate.sh                         # all checkpoints/*_results.json
#   bash scripts/evaluate.sh checkpoints/a_results.json checkpoints/b_results.json

set -e

cd /home/user

mkdir -p scoring

python -m eval.evaluate_results "$@" --output scoring/scores.json
