#!/bin/bash
# Sweep the ghost-GPU probe over suspected nodes (login-node script).
#   ./ghost_probe_sweep.sh jpbo-037-36 jpbo-037-39 jpbo-074-02 ...
# Submits one 1-node/6-min probe per node that is currently idle (a probe aimed
# at an allocated node would just pend). Results land in
#   $F/experiments/ghost_probe/logs/ghost_probe_<node>_<jobid>.out
# Collect with:  grep -A6 "allocation probe" $F/experiments/ghost_probe/logs/*.out
F=/e/fscratch/reformo/lee27
SB=$F/OpenThoughts-Agent/scripts/jupiter/ghost_gpu_probe.sbatch
for n in "$@"; do
  st=$(sinfo -h -n "$n" -o %T 2>/dev/null | head -1)
  if [ "$st" != "idle" ]; then
    echo "$n: state=${st:-unknown} — skipped (only idle nodes are probeable)"
    continue
  fi
  jid=$(sbatch --parsable --nodelist="$n" "$SB") && echo "$n: probe job $jid"
done
