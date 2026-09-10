#!/bin/bash
# gate_watch.sh — runs ON the Jupiter login node: polls the juwels gate job + bridge #2 + trial results every 60 s.
J=1657050; N=ttsample_s0; E=/e/fscratch/reformo/lee27/experiments; D=$E/$N/$N/trace_jobs
W="ssh -o BatchMode=yes -S ~/.ssh/cm_juwels/bridge juwels"
for i in $(seq 1 300); do
  st=$(squeue -h -j $J -o "%T %M %R" 2>/dev/null); br=$(curl -s -m 5 localhost:9926/status | python3 -c "import sys,json; d=json.load(sys.stdin); e=d['envs']; print('ready',e['ready'],'starting',e['starting'],'stopping',e['stopping'],'jobs',d['active_jobs'],'errs',d['stats']['jobs_errors'],'alive',d['workers_alive'])" 2>/dev/null)
  res=$(ls $D/eval_sessions/*/*/attempts/*/result.json 2>/dev/null | wc -l)
  log=$(ls -t $E/$N/logs/*.out 2>/dev/null | head -1); gen=$(grep -c "Starting batch generation" "$log" 2>/dev/null); dp=$(grep -c "DP rank -> node" "$log" 2>/dev/null); to=$(grep -c "BridgeOperationTimeoutError" "$log" 2>/dev/null)
  fl=$($W "squeue -h -j 14226172 -o '%T %M'" 2>/dev/null)
  echo "$(date +%H:%M:%S) gate[$st] fleet[$fl] bridge[$br] results=$res gen_starts=$gen dp_lines=$dp bridge_timeouts=$to"
  if [ "$res" -ge 512 ] || [ -z "$st" ]; then echo DONE; break; fi
  sleep 60
done
