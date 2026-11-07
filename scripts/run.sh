#!/bin/bash
# Job entry point (CPU, no internet) for the final hardening check:
# reference rerun (1000 training queries per template), frozen comparators,
# scoring, then better/weaker variants, training-seed reruns and shortcuts
# through the hidden test. Resumable; see redteam/final_hardening/run_final.py.
# (scripts/reproduce.sh reproduces the reference alone.)

set -e

cd /home/user

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python redteam/final_hardening/run_final.py
