#!/bin/bash
# Current scoring audit: all computation is bundled into this CPU job.
set -euo pipefail
cd /home/user
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python redteam/audit_20261107/harness.py all
