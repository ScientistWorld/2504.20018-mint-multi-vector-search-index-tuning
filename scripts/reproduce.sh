#!/bin/bash
# Reproduce the reference end to end (CPU): tune both workloads with MINT and
# score them on the visible train slice and the held-out test slice.
#
# Assumes the container from environment/ and runs from /home/user.

set -e

cd /home/user

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python -m method.mint --workload naive_small --output checkpoints/mint_results.json
python -m method.mint --workload bicomplex_small --output checkpoints/mint_bicomplex_results.json
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
bash scripts/evaluate.sh
