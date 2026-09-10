#!/bin/bash
# store_reaper.sh — every 30 min, for each RUNNING snowball_ttband arm with an artifact store: on its batch host run
# code/snowball/store_compact.py over trace_jobs (trial dirs older than 20 min: rewrite attempts/000/result.json compact without the
# token/logprob blob, delete the two byte-identical duplicate result copies, keep the readable trace + logs; 78 MB -> ~5 MB per trial).
# Nothing is deleted at the trial level — Luke wants the agent traces retained (2026-09-05 10:00 PT). Log: experiments/store_reaper.log.
export OMP_NUM_THREADS=1; E=/e/fscratch/reformo/lee27/experiments; C=/e/project1/transfernetx/lee27/code/snowball
while true; do
  for line in $(squeue -h -u $USER -o "%i:%j:%N" | grep snowball_ttband | grep -v _val_); do
    J=${line%%:*}; rest=${line#*:}; R=${rest%%:*}; NL=${rest#*:}
    CFG=$E/$R/configs/${R}_rl_config.json; [ -f "$CFG" ] || continue
    M=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("artifact_store_mount",""))' "$CFG"); [ -z "$M" ] && continue
    BH=$(scontrol show hostnames "$NL" 2>/dev/null | head -1); [ -z "$BH" ] && continue
    echo "$(date +%F_%T) $R $(timeout 1500 srun --overlap --jobid=$J -N1 -n1 -w $BH --cpus-per-task=16 bash -c "use=\$(df --output=pcent $M 2>/dev/null | tail -1 | tr -dc 0-9); echo use=\${use:-?}%; python3 $C/store_compact.py $M/trace_jobs --min-age-min 20 --workers 16 2>&1 | tail -1; echo after=\$(df --output=pcent $M | tail -1 | tr -dc 0-9)%" 2>&1 | grep -v "^srun" | tr '\n' ' ')"
  done
  sleep 1800
done
