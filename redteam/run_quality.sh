#!/bin/bash
set -euo pipefail
cd /home/user
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
trap 'python -m redteam.harness restore' EXIT
# Re-evaluate the cached reference and weaker variant to collect independent
# full-workload scores. Do not retrain the three completed seed runs.
for name in reference_cached better_actual_cost weaker_no_multicolumn; do
  echo "[audit quality] running $name"
  bash "redteam/$name/run.sh" > "redteam/$name/run.log" 2>&1
  echo "[audit quality] completed $name"
done
python -m redteam.rank
