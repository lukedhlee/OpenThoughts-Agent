#!/bin/bash
# easy3_after_trio.sh — second attempt at the six easy-source probes (2026-09-06, after the 18:30 PT bridge outage).
# Waits until the r2egym trio is tabled, reads it out, then launches the six easy probes (keep/drop x 3 sources) on the
# EXISTING 768 seats of fleets 14229634-36 — no new fleets: the outage was the bridge's thread-per-request server hitting
# the login node's 4096-pid cap once 2,304 workers + 9 probe clients polled it. Reads out each pair (full 300 + audited-clean).
E=/e/fscratch/reformo/lee27/experiments; C=/e/project1/transfernetx/lee27/code/snowball; T=/e/fscratch/reformo/lee27/tasks/tt-easy3
O=$E/hist_readouts; A=$E/easy3_audit; X=$E; mkdir -p $O; export OMP_NUM_THREADS=1
tabled() { for p in "$@"; do [ -f $E/$p/$p/pass8_summary.json ] || [ -z "$(squeue -u $USER -h -n $p -o %T)" ] || return 1; done; return 0; }
log() { echo "$(date) $*" >> $O/log; }
log "easy3_after_trio start: waiting for the r2egym trio"
until tabled snowball_hist_keep_base snowball_hist_drop_base snowball_hist_last2_base; do sleep 60; done
log "r2egym trio tabled: read-out"
python3 $C/hist_readout.py --probes keep=snowball_hist_keep_base drop=snowball_hist_drop_base last2=snowball_hist_last2_base \
  --splits idval=$X/tt_v2_idval.txt,oodval=$X/tt_v2_oodval.txt,heldout=$X/tt_v2_heldout.txt --out $O/hist_r2egym >> $O/log 2>&1
log "launching the six easy probes on the existing seats"
for s in curriculumeasy:curriculum-easy pymethods2testv3:pymethods2test-v3 unitsynpythonv4:unitsyn-python-v4; do
  short=${s%%:*}; src=${s##*:}
  for m in keep drop; do bash $C/probe_history.sh snowball_easy2_${short}_${m}_base $T/$src base $m >> $O/log 2>&1; done
done
sleep 180
for s in curriculumeasy:curriculum-easy pymethods2testv3:pymethods2test-v3 unitsynpythonv4:unitsyn-python-v4; do
  short=${s%%:*}; src=${s##*:}
  until tabled snowball_easy2_${short}_keep_base snowball_easy2_${short}_drop_base; do sleep 60; done
  log "$src pair tabled: read-out"
  python3 $C/hist_readout.py --probes keep=snowball_easy2_${short}_keep_base drop=snowball_easy2_${short}_drop_base --out $O/easy_${short} >> $O/log 2>&1
  python3 $C/hist_readout.py --probes keep=snowball_easy2_${short}_keep_base drop=snowball_easy2_${short}_drop_base --exclude $A/exclude_${src}.txt --out $O/easy_${short}_clean >> $O/log 2>&1
done
log ALL_DONE
