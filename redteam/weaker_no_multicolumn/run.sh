#!/bin/bash
set -euo pipefail
cd /home/user
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m redteam.harness weaker_no_multicolumn
