#!/bin/bash
# hist_readout_all.sh — run the history-think read-outs on Jupiter as each probe group finishes (2026-09-06).
# Group = r2egym trio (keep/drop/last2 on val441, split read-out) + three easy keep/drop pairs (full 300 and audited-clean).
# A group is read when every member is tabled (pass8_summary.json) or its job is gone; reads trace_jobs or trace_archive.tar.
E=/e/fscratch/reformo/lee27/experiments; C=/e/project1/transfernetx/lee27/code/snowball; O=$E/hist_readouts; A=$E/easy3_audit
X=/e/fscratch/reformo/lee27/experiments; mkdir -p $O; export OMP_NUM_THREADS=1
done_group() { for p in "$@"; do [ -f $E/$p/$p/pass8_summary.json ] || [ -z "$(squeue -u $USER -h -n $p -o %T)" ] || return 1; done; return 0; }
for i in $(seq 1 600); do
  if [ ! -f $O/hist_r2egym_readout.md ] && done_group snowball_hist_keep_base snowball_hist_drop_base snowball_hist_last2_base; then
    echo "$(date) r2egym trio tabled: reading out" >> $O/log
    python3 $C/hist_readout.py --probes keep=snowball_hist_keep_base drop=snowball_hist_drop_base last2=snowball_hist_last2_base \
      --splits idval=$X/tt_v2_idval.txt,oodval=$X/tt_v2_oodval.txt,heldout=$X/tt_v2_heldout.txt --out $O/hist_r2egym >> $O/log 2>&1
  fi
  for s in curriculumeasy:curriculum-easy pymethods2testv3:pymethods2test-v3 unitsynpythonv4:unitsyn-python-v4; do
    short=${s%%:*}; src=${s##*:}
    if [ ! -f $O/easy_${short}_readout.md ] && done_group snowball_easy_${short}_keep_base snowball_easy_${short}_drop_base; then
      echo "$(date) $src pair tabled: reading out" >> $O/log
      python3 $C/hist_readout.py --probes keep=snowball_easy_${short}_keep_base drop=snowball_easy_${short}_drop_base --out $O/easy_${short} >> $O/log 2>&1
      python3 $C/hist_readout.py --probes keep=snowball_easy_${short}_keep_base drop=snowball_easy_${short}_drop_base --exclude $A/exclude_${src}.txt --out $O/easy_${short}_clean >> $O/log 2>&1
    fi
  done
  n=$(ls $O/*_readout.md 2>/dev/null | wc -l)
  [ "$n" -ge 7 ] && { echo "$(date) ALL_READOUTS_DONE" >> $O/log; exit 0; }
  sleep 60
done
echo "$(date) waiter expired" >> $O/log
