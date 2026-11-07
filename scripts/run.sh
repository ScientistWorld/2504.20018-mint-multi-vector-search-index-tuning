#!/bin/bash
# Job entry point (CPU, no internet): reproduce the reference end to end.
# Tunes both workloads with MINT, then scores every checkpoints/*_results.json
# on the visible train slice, the held-out test slice and scores.json.
# (The hardening audit itself is redteam/hardening/run_hardening.py.)

set -e

cd /home/user

bash scripts/reproduce.sh
