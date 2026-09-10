#!/bin/bash
# after_s3.sh — once pool shard s3 reaches its eval block: scancel it, table it, wait for a trial-free window, restart the
# bridge (backlog 1024 / reaper 2400), then launch the residual wave (snowball_pool60k_r1) and the clean parity re-probe.
E=/e/fscratch/reformo/lee27/experiments; C=/e/project1/transfernetx/lee27/code/snowball; LOG=$E/after_s3.log
cd /e/project1/transfernetx/lee27/code/OpenThoughts-Agent || exit 1; export DCFT=$PWD
echo "$(date) start" >> $LOG
while true; do L=$(ls -t $E/snowball_pool60k_s3/logs/*.out | head -1); jid=$(squeue -h -u $USER -n snowball_pool60k_s3 -o %i | head -1)
  if grep -q "WANDB_MIRROR kind=eval" "$L"; then echo "$(date) s3 eval block present (job ${jid:-gone})" >> $LOG; [ -n "$jid" ] && scancel $jid; break; fi
  [ -z "$jid" ] && { echo "$(date) s3 job gone before eval block" >> $LOG; break; }
  sleep 60; done
[ -f $E/snowball_pool60k_s3/pass8_summary.json ] || (python3 $C/pass8_table.py $E/snowball_pool60k_s3/snowball_pool60k_s3/trace_jobs --k 8 --out $E/snowball_pool60k_s3/pass8 >> $LOG 2>&1 &)
for i in $(seq 1 30); do n=$(squeue -h -u $USER -o %j | grep -c "pool60k\|probe_val\|grpo\|smoke"); [ "$n" -eq 0 ] && break; sleep 20; done
echo "$(date) trial jobs remaining: $n; sleeping 60 s for stragglers" >> $LOG; sleep 60
bash $C/bridge_restart.sh >> $LOG 2>&1; rc=$?; echo "$(date) bridge_restart rc=$rc" >> $LOG
[ $rc -ne 0 ] && { echo "$(date) ABORT: bridge not healthy; not launching" >> $LOG; exit 1; }
tmux new -d -s pool_launch_r1 "bash $C/pool_launch.sh snowball_pool60k_r1 8 150" && tmux new -d -s pool_watch_r1 "bash $C/pool_watch.sh snowball_pool60k_r1 8" && echo "$(date) LAUNCHED residual wave snowball_pool60k_r1 (8 shards, 150 s stagger)" >> $LOG
sleep 30; out=$(sbatch $E/snowball_probe_val_b60k_parity2/sbatch/snowball_probe_val_b60k_parity2_rl.sbatch); echo "$(date) parity2: $out" >> $LOG
J=$(echo "$out" | awk '{print $NF}'); tmux new -d -s val_watch_b60k_parity2 "bash $C/val_watch.sh snowball_probe_val_b60k_parity2 $J b60k_parity2"
echo "$(date) done" >> $LOG
