#!/bin/bash
set -euo pipefail
cd /home/user
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for seed in 20240501 101 202; do
  bash "redteam/audit_20261107/reference_seed_${seed}/run.sh" "$@"
done
python redteam/audit_20261107/harness.py shortcut_seed_lottery "$@"
