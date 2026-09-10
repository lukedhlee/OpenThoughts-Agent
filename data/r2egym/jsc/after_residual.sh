#!/bin/bash
# after_residual.sh — when all 8 residual shard tables exist: union of scored attempts over pass 1 + r1 partial + r1,
# build the band + task dir, sanity-gate it, then generate+submit the GRPO arms (lr 1e-5 primary, 3e-6 hedge).
E=/e/fscratch/reformo/lee27/experiments; C=/e/project1/transfernetx/lee27/code/snowball; T=/e/fscratch/reformo/lee27/tasks
B=$E/snowball_pool60k_band; LOG=$E/after_residual.log; mkdir -p $B
cd /e/project1/transfernetx/lee27/code/OpenThoughts-Agent || exit 1; export DCFT=$PWD
echo "$(date) start" >> $LOG
while true; do n=0; for i in 0 1 2 3 4 5 6 7; do [ -f $E/snowball_pool60k_r1_s$i/pass8_summary.json ] && n=$((n+1)); done
  [ $n -eq 8 ] && break; sleep 120; done
echo "$(date) all 8 r1 tables present" >> $LOG
SRC=""; for i in 0 1 2 3 4 5 6 7; do SRC="$SRC $E/snowball_pool60k_s$i/snowball_pool60k_s$i/trace_jobs $E/snowball_pool60k_r1_s$i/run1_partial/trace_jobs $E/snowball_pool60k_r1_s$i/snowball_pool60k_r1_s$i/trace_jobs"; done
python3 $C/merge_scored_union.py --trace-jobs $SRC --out $B/union >> $LOG 2>&1 || { echo "$(date) ABORT union failed" >> $LOG; exit 1; }
python3 $C/build_band.py --tables $B/union_pass8_table.csv --out $B --task-dir $T/r2egym-raw-v3-train-band60k >> $LOG 2>&1 || { echo "$(date) ABORT band failed" >> $LOG; exit 1; }
read mixed full tasks <<< $(python3 -c "import json; s=json.load(open('$B/summary.json')); print(s['strict_mixed'], s['fully_sampled'], s['tasks'])")
echo "$(date) band: strict_mixed=$mixed fully_sampled=$full tasks_with_data=$tasks" >> $LOG
if [ "$mixed" -lt 300 ] || [ "$full" -lt 2000 ]; then echo "$(date) GATE FAILED (need >=300 mixed and >=2000 fully sampled); NOT launching GRPO — inspect $B and decide on a second residual" >> $LOG; exit 2; fi
inodes=$(timeout 900 find /e/fscratch/reformo/lee27 -xdev -type f 2>/dev/null | wc -l); echo "$(date) inodes in tree: $inodes" >> $LOG
[ "$inodes" -gt 1700000 ] && { echo "$(date) GATE FAILED: inode headroom < 400k; NOT launching" >> $LOG; exit 3; }
for arm in "snowball_grpo_b60k_lr1e5 1e-5" "snowball_grpo_b60k_lr3e6 3e-6"; do set -- $arm
  python3 $C/make_snowball_grpo.py --name $1 --train-dir $T/r2egym-raw-v3-train-band60k --lr $2 --submit >> $LOG 2>&1 && echo "$(date) LAUNCHED $1 (lr $2)" >> $LOG
  sleep 120; done
echo "$(date) done" >> $LOG
