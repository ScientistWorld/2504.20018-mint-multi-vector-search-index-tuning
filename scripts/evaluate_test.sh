#!/bin/bash
# Held-out test-slice evaluation for final anti-shortcut scoring.
#
# Mirrors scripts/evaluate.sh's result-artifact contract, but writes
# scoring/scores_test.json and invokes eval/test/.

set -e

cd /home/user

mkdir -p scoring

if [ "$#" -eq 0 ]; then
  mapfile -t RESULT_PATHS < <(find checkpoints -maxdepth 1 -type f -name '*_results.json' | sort)
  if [ "${#RESULT_PATHS[@]}" -eq 0 ]; then
    echo "no result artifacts found under checkpoints/*_results.json" >&2
    exit 1
  fi
else
  RESULT_PATHS=("$@")
fi

python -m eval.test.evaluate "${RESULT_PATHS[@]}" --output scoring/scores_test.json
