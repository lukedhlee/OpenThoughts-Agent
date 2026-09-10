#!/bin/bash
# sync_table.sh <run>... — one line per train step per run: reward pass8 entropy logratio masked rejected grad step_s + rollout profile.
E=/e/fscratch/reformo/lee27/experiments; S=/e/project1/transfernetx/lee27/code/snowball
for R in "$@"; do L=$(ls -t $E/$R/logs/*.out 2>/dev/null | head -1); [ -n "$L" ] || { echo "$R: no log"; continue; }
  python3 $S/extract_metrics.py "$L" 2>/dev/null > /tmp/lee27_st_$R.txt
  python3 $S/rollout_profile.py $E/$R/$R/exports/dumped_data 2>/dev/null > /tmp/lee27_pf_$R.txt
  python3 $S/sync_table.py $R /tmp/lee27_st_$R.txt /tmp/lee27_pf_$R.txt
done
