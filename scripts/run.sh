#!/bin/bash
# Job entry point (CPU, no internet): reproduce the reference end to end.
# Tunes both workloads with MINT (1000 training queries per template), then
# scores every checkpoints/*_results.json on the visible train slice, the
# held-out test slice and scores.json. The final hardening audit drivers are
# redteam/final_hardening/run_final.py and run_final2.py.

set -e

cd /home/user

bash scripts/reproduce.sh
