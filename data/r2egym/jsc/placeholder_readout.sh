#!/bin/bash
# placeholder_readout.sh — when the placeholder sub160 probe has tabled and archived, read it out against the trio (base+top-up
# merged per arm) on the 160-task subset and its split intersections. 2026-09-06.
E=/e/fscratch/reformo/lee27/experiments; C=/e/project1/transfernetx/lee27/code/snowball; O=$E/hist_readouts; export OMP_NUM_THREADS=1
P=snowball_hist_placeholder_sub160
settled() { for p in "$@"; do d=$E/$p/$p; job=$(squeue -u $USER -h -n $p -o %T); { [ -f $E/$p/pass8_summary.json ] || [ -z "$job" ]; } || return 1
  if [ -f $d/trace_archive.tar ] && [ ! -d $d/trace_jobs ]; then continue; fi
  if [ -z "$job" ] && [ -d $d/trace_jobs ]; then m=$(( $(date +%s) - $(stat -c %Y $d/trace_jobs) )); [ $m -ge 900 ] && [ ! -f $d/trace_archive.tar ] && continue; fi
  return 1; done; return 0; }
until settled $P; do sleep 60; done
echo "$(date) placeholder tabled: read-out" >> $O/log
K=snowball_hist_keep_base; D=snowball_hist_drop_base; L=snowball_hist_last2_base
[ -f $E/snowball_hist_keep_topup/pass8_summary.json ] && { K=$K+snowball_hist_keep_topup; D=$D+snowball_hist_drop_topup; L=$L+snowball_hist_last2_topup; }
python3 $C/hist_readout.py --probes keep=$K drop=$D last2=$L placeholder=$P \
  --splits sub160=$E/hist_sub160.txt,idval=$E/tt_v2_idval.txt,oodval=$E/tt_v2_oodval.txt,heldout=$E/tt_v2_heldout.txt --out $O/hist_placeholder >> $O/log 2>&1
echo "$(date) PLACEHOLDER_READOUT_DONE" >> $O/log
