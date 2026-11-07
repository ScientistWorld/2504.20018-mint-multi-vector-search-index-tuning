#!/bin/bash
set -uo pipefail
cd /home/user
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
trap 'python -m redteam.harness restore' EXIT
cases=(
  reference_cached
  reference_seed_12345 reference_seed_7 reference_seed_19
  better_actual_cost weaker_no_multicolumn higher_recall
  shortcut_recall_89 shortcut_low_recall shortcut_empty
  shortcut_forged_metrics shortcut_storage_declaration shortcut_overstorage
  shortcut_inflate_baseline shortcut_negative_ek
  shortcut_visible_answers shortcut_test_size shortcut_replace_fixed_data
)
failed=0
for name in "${cases[@]}"; do
  if [ -f "redteam/$name/scores_test.json" ]; then
    echo "[audit] already completed $name"
    continue
  fi
  echo "[audit] running $name"
  if timeout 4500 bash "redteam/$name/run.sh" > "redteam/$name/run.log" 2>&1; then
    echo "[audit] completed $name"
  else
    code=$?
    echo "[audit] FAILED $name: $code"
    failed=1
    python -m redteam.harness restore
  fi
done
python -m redteam.rank
exit "$failed"
