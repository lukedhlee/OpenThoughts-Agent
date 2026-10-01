#!/bin/bash
# rl_watch.sh [run name ...] — one status block per Horizon RL run/probe (default: every run dir with a jobs.txt):
# job state, log age, engine DP placement, train/eval steps logged, trial results written, tracebacks, and the last
# step's key metrics (tis log_ratio_abs_mean, reward, entropy) from the WANDB_MIRROR lines. Read-only.
E=/scratch/11584/lukedhlee/experiments/rl
runs=("$@"); [ ${#runs[@]} = 0 ] && runs=($(cd $E && ls -d */jobs.txt 2>/dev/null | cut -d/ -f1))
for r in "${runs[@]}"; do
  J=$(tail -n 1 $E/$r/jobs.txt 2>/dev/null); L=$E/$r/logs/${r}_$J.out
  st=$(squeue -h -j "$J" -o "%T %M %D" 2>/dev/null); [ -z "$st" ] && st="ENDED $(sacct -j $J -X -n -o State,Elapsed 2>/dev/null | head -1 | xargs)"
  age=$([ -f $L ] && echo $(( $(date +%s) - $(stat -c %Y $L) ))s || echo "-")
  dp=$(grep -c "DP rank -> node" $L 2>/dev/null); tb=$(grep -c "^Traceback\|Traceback (most recent" $L 2>/dev/null)
  tr=$(grep -c "WANDB_MIRROR kind=train" $L 2>/dev/null); ev=$(grep -c "WANDB_MIRROR kind=eval" $L 2>/dev/null)
  res=$(find $E/$r/trials -maxdepth 4 -name result.json 2>/dev/null | wc -l)
  echo "== $r job $J [$st] log age $age | DP lines $dp | train steps $tr | eval logs $ev | probe results $res | tracebacks $tb"
  # exception mix of finished trials (probes: /scratch trials; arms: /dev/shm on the head node, last ~10 min kept)
  if [ -d $E/$r/trials ]; then TD=$E/$r/trials; ex=$(find $TD -maxdepth 4 -name result.json -mmin -30 2>/dev/null | head -400 | xargs -r grep -ho '"exception_type": *"[A-Za-z]*"' 2>/dev/null | sort | uniq -c | sort -rn | head -4 | awk '{printf "%s=%s ", $3, $1}' | tr -d '"')
  else N=$(scontrol show hostnames "$(squeue -h -j $J -o %N 2>/dev/null)" 2>/dev/null | head -1)
    [ -n "$N" ] && ex=$(timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=10 $N "find /dev/shm/otagent_trials/$r -maxdepth 3 -name result.json 2>/dev/null | head -400 | xargs -r grep -ho '\"exception_type\": *\"[A-Za-z]*\"' | sort | uniq -c | sort -rn | head -4" 2>/dev/null | awk '{printf "%s=%s ", $3, $1}' | tr -d '"'); fi
  [ -n "${ex:-}" ] && echo "   recent trial exceptions (<=400 trials): $ex"; ex=""
  last=$(grep "WANDB_MIRROR kind=train" $L 2>/dev/null | tail -n 1)
  [ -n "$last" ] && python3 - "$last" <<'PY'
import re, sys, json
line = sys.argv[1]; step = re.search(r"step=(\d+)", line).group(1); m = line.split("metrics=", 1)[1]
try:
    d = json.JSONDecoder().raw_decode(m)[0]
except Exception:
    d = {k: v for k, v in re.findall(r"'([^']+)': ([-0-9.eE+]+)", m)}
keys = [k for k in d if re.search(r"tis/log_ratio_abs_mean|avg_raw_reward|policy_entropy|reward_given_done|pass_at_8|frac_groups_mixed|avg_num_tokens|timing/step$|num_masked", k)]
print("   step", step, " ".join(f"{k}={float(d[k]):.4g}" for k in sorted(keys)[:14]))
PY
done
