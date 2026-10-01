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
  dp=$(grep -c "DP rank" $L 2>/dev/null); tb=$(grep -c "^Traceback\|Traceback (most recent" $L 2>/dev/null)
  tr=$(grep -c "WANDB_MIRROR kind=train" $L 2>/dev/null); ev=$(grep -c "WANDB_MIRROR kind=eval" $L 2>/dev/null)
  res=$(find $E/$r/trials -maxdepth 4 -name result.json 2>/dev/null | wc -l)
  echo "== $r job $J [$st] log age $age | DP lines $dp | train steps $tr | eval logs $ev | probe results $res | tracebacks $tb"
  last=$(grep "WANDB_MIRROR kind=train" $L 2>/dev/null | tail -n 1)
  [ -n "$last" ] && python3 - "$last" <<'PY'
import re, sys, json
line = sys.argv[1]; step = re.search(r"step=(\d+)", line).group(1); m = line.split("metrics=", 1)[1]
try:
    d = json.loads(m)
except Exception:
    d = {k: v for k, v in re.findall(r"'([^']+)': ([-0-9.eE+]+)", m)}
keys = [k for k in d if re.search(r"tis/log_ratio_abs_mean|reward/avg|avg_raw_reward|entropy|q_correct|finish|pass_at|response_length/avg|timing/step", k)]
print("   step", step, " ".join(f"{k}={float(d[k]):.4g}" for k in sorted(keys)[:14]))
PY
done
