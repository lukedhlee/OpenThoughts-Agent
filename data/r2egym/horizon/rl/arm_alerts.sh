#!/bin/bash
# arm_alerts.sh <run ...> — one line per event worth acting on, for a Monitor: job state change, new tracebacks,
# a stop-rule trip at any step (q = reward_given_done < .60, tis log-ratio > 0.0736 = 2x gate 5, masked
# trajectories > 0), and a heartbeat every 6th step. Polls every 60 s. Read-only.
E=/scratch/11584/lukedhlee/experiments/rl
declare -A ST TB STEP
while true; do
  for r in "$@"; do
    J=$(tail -n 1 $E/$r/jobs.txt 2>/dev/null); L=$E/$r/logs/${r}_$J.out
    s=$(squeue -h -j "$J" -o %T 2>/dev/null); s=${s:-ENDED}
    [ "${ST[$r]:-}" != "$s" ] && { echo "$r job $J state $s"; ST[$r]=$s; }
    t=$(grep -c "Traceback (most recent" $L 2>/dev/null); t=${t:-0}
    [ "$t" != "${TB[$r]:-0}" ] && { echo "$r tracebacks $t (was ${TB[$r]:-0})"; TB[$r]=$t; }
    out=$(python3 - "$L" "${STEP[$r]:-0}" <<'PY'
import json, re, sys
log, last = sys.argv[1], int(sys.argv[2]); top = last
try:
    lines = open(log, errors='replace').read().splitlines()
except OSError:
    lines = []
for line in lines:
    m = re.search(r'WANDB_MIRROR kind=train step=(\d+) metrics=(.*)', line)
    if not m or int(m.group(1)) <= last:
        continue
    s = int(m.group(1)); top = max(top, s)
    try:
        d = json.JSONDecoder().raw_decode(m.group(2))[0]
    except ValueError:
        continue
    q, tis, mk = d.get('diag/reward_given_done', 1), d.get('policy/tis/log_ratio_abs_mean', 0), d.get('generate/num_masked_trajectories', 0)
    trip = [x for x, c in (('q<.60', q < 0.60), ('tis>0.0736', tis > 0.0736), ('masked>0', mk > 0)) if c]
    if trip or s % 6 == 0:
        print(f"step {s} {'TRIP ' + ','.join(trip) + ' ' if trip else ''}q={q:.3f} tis={tis:.4f} H={d.get('policy/policy_entropy', 0):.3f} "
              f"reward={d.get('reward/avg_raw_reward', 0):.3f} mixed={d.get('diag/frac_groups_mixed', 0):.2f} masked={mk:.0f} step_s={d.get('timing/step', 0):.0f}")
print(f"LAST {top}")
PY
)
    STEP[$r]=$(echo "$out" | sed -n 's/^LAST //p'); echo "$out" | grep -v "^LAST" | sed "s/^/$r /"
  done
  sleep 60
done
