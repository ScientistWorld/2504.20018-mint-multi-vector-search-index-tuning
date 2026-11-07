#!/bin/bash
# Hardening job (CPU, no internet): rerun the reference end to end on the
# hardened gym, freeze the evaluator-owned comparator plans, score train/test,
# then run better/weaker variants and shortcut artifacts through the hidden
# test. Resumable: finished stages are skipped on resubmission.

set -e

cd /home/user

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

python redteam/hardening/run_hardening.py
