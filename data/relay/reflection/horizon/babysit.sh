#!/bin/bash
# babysit.sh CONFIG... — login-node loop (tmux): every 10 min, one status line per run (phase, committed update, Slurm
# state of its job) into <output_dir>/babysit.status, and prune_native.py on each run. It never submits, cancels or
# resumes anything: a failed phase or a vanished job is only reported (FAILED lines), for the operator to reconcile.
OTA=$(cd "$(dirname "$0")/../../../.." && pwd); PY=$HOME/snowball/envs/snowball/bin/python
OUT=/scratch/11584/$USER/experiments/calibforge_reflection; ST=$OUT/babysit.status
while :; do
  for cfg in "$@"; do
    name=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['name'])" "$cfg")
    d=$OUT/$name
    job=$(ls -t "$d"/logs/job_*.out 2>/dev/null | head -1 | sed -E 's#.*job_([0-9]+)\.out#\1#')
    js=$(squeue -h -j "$job" -o %T 2>/dev/null); [ -z "$js" ] && js="gone($(sacct -j "$job" -X -n -o State 2>/dev/null | head -1 | tr -d ' '))"
    s=$($PY -c "import json,sys; s=json.load(open(sys.argv[1])); print(s['phase'], s['iteration'], s.get('error','')[:200])" "$d/state.json" 2>/dev/null || echo "no_state")
    echo "$(date -Is) $name job=$job $js state=$s" >> "$ST"
    case "$s" in failed*|stopped*) echo "$(date -Is) FAILED $name: $s" >> "$ST";; esac
    PYTHONPATH=$OTA $PY -m data.relay.reflection.horizon.prune_native --config "$cfg" >> "$ST" 2>&1
  done
  sleep 600
done
