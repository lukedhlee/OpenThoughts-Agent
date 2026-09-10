#!/bin/bash
# pool_gate.sh — fallback auto-launch of the 64k PARITY pool pass (snowball_pool60k_s{0..7}) once the parity val probe
# 1632285 has >=MIN attempts AND the SFT-parity signature check passes AND the trial pass rate is in the 60k range.
# Kill this tmux (pool_gate) to disable. Log: $E/pool_gate.log. Never launches twice (checks squeue + shard logs).
J=1632285; N=snowball_probe_val_b60k_parity; MIN=${1:-100}
E=/e/fscratch/reformo/lee27/experiments; D=$E/$N/$N/trace_jobs; PY=/e/project1/transfernetx/lee27/code/envs/snowball/bin/python
C=/e/project1/transfernetx/lee27/code/snowball; LOG=$E/pool_gate.log
echo "$(date) pool_gate start job=$J min=$MIN" >> $LOG
while true; do
  if squeue -h -u $USER -n snowball_pool60k_s0 -o %i | grep -q . || ls $E/snowball_pool60k_s0/logs/*.out >/dev/null 2>&1; then echo "$(date) pool already launched; exiting" >> $LOG; exit 0; fi
  st=$(squeue -h -j $J -o %T 2>/dev/null)
  att=$(ls -d $D/eval_sessions/*/*/attempts/*/result.json 2>/dev/null | wc -l)
  if [ -z "$st" ] && [ "$att" -lt "$MIN" ]; then echo "$(date) probe gone with only $att attempts; NOT launching" >> $LOG; exit 1; fi
  if [ "$att" -ge "$MIN" ]; then
    python3 $C/pass8_table.py $D --k 8 --out $E/$N/pass8_gate_partial >/dev/null 2>&1
    verdict=$(python3 $C/parity_check.py $D/eval_sessions $E/$N/pass8_gate_partial_summary.json 2>&1 | tail -1)
    echo "$(date) attempts=$att verdict: $verdict" >> $LOG
    if [[ "$verdict" == PASS* ]]; then
      tmux new -d -s pool_launch "bash $C/pool_launch.sh snowball_pool60k 8 150" && tmux new -d -s pool_watch "bash $C/pool_watch.sh snowball_pool60k 8"
      echo "$(date) LAUNCHED pool_launch + pool_watch" >> $LOG; exit 0
    else
      echo "$(date) gate FAILED; not launching (rerun manually after inspection)" >> $LOG; exit 1
    fi
  fi
  sleep 120
done
