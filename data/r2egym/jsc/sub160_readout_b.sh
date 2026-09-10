#!/bin/bash
# sub160_readout.sh — when both the placeholder and the head:300 probes have tabled and archived, read out all five contracts
# on the 160-task subset (keep/drop/last2 = base+top-up merged per arm). 2026-09-07.
E=/e/fscratch/reformo/lee27/experiments; C=/e/project1/transfernetx/lee27/code/snowball; O=$E/hist_readouts; export OMP_NUM_THREADS=1
settled() { for p in "$@"; do d=$E/$p/$p; job=$(squeue -u $USER -h -n $p -o %T); { [ -f $E/$p/pass8_summary.json ] || [ -z "$job" ]; } || return 1
  if [ -f $d/trace_archive.tar ] && [ ! -d $d/trace_jobs ]; then continue; fi
  if [ -z "$job" ] && [ -d $d/trace_jobs ]; then m=$(( $(date +%s) - $(stat -c %Y $d/trace_jobs) )); [ $m -ge 900 ] && [ ! -f $d/trace_archive.tar ] && continue; fi
  return 1; done; return 0; }
until settled snowball_hist_head300_sub160b; do sleep 60; done
echo "$(date) placeholder + head300 tabled: five-arm sub160 read-out" >> $O/log
K=snowball_hist_keep_base+snowball_hist_keep_topup; D=snowball_hist_drop_base+snowball_hist_drop_topup; L=snowball_hist_last2_base+snowball_hist_last2_topup
python3 $C/hist_readout.py --probes keep=$K drop=$D last2=$L placeholder=snowball_hist_placeholder_sub160 head300=snowball_hist_head300_sub160b \
  --min-scored 7 --splits sub160=$E/hist_sub160.txt,idval=$E/tt_v2_idval.txt,oodval=$E/tt_v2_oodval.txt,heldout=$E/tt_v2_heldout.txt --out $O/hist_sub160_five >> $O/log 2>&1
echo "$(date) SUB160_FIVE_READOUT_DONE" >> $O/log
