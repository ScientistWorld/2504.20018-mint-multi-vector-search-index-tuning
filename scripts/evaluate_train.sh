#!/bin/bash
# Visible train-slice evaluation for future method development.
#
# Mirrors scripts/evaluate.sh's result-artifact contract, but writes
# scoring/scores_train.json and invokes eval/train/.

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

python -m eval.train.evaluate "${RESULT_PATHS[@]}" --output scoring/scores_train.json
