#!/bin/bash
set -euo pipefail
C=/e/project1/transfernetx/lee27/code/snowball; E=/e/fscratch/reformo/lee27/experiments
W=$(sacct -j 1630184 -o WorkDir%200 -n | head -1 | xargs); echo "workdir=$W"
VD=$(python3 - <<'PY'
import json
a=json.load(open("/e/fscratch/reformo/lee27/experiments/snowball_probe_val_b28k/configs/snowball_probe_val_b28k_rl_config.json"))["skyrl_hydra_args"]
v=[x for x in a if x.startswith("data.val_data=")][0]
print(json.loads(v.split("=",1)[1])[0])
PY
); echo "valdir=$VD"
SB=$(python3 $C/make_snowball_probe.py --name snowball_probe_val_b60k --val-dir "$VD" --k 8 --conc 128 --max-in 61440 --max-out 4096 --max-model-len 65536 --eval-timeout 1800 --wall 07:00:00); echo "sbatch=$SB"
CFG=$E/snowball_probe_val_b60k/configs/snowball_probe_val_b60k_rl_config.json
grep -o '"++generator.engine_init_kwargs[^"]*"' $CFG
grep -oE '"[^"]*(max_input|max_prompt|max_generate|max_output|max_model)[a-z_]*=[0-9]*"' $CFG | sort -u
grep -o '"++terminal_bench_config.harbor.eval_timeout_override_sec=[0-9]*"' $CFG || echo "NO eval timeout override!"
grep -c enable_summarize=true $CFG || true
wc -c $SB
cd "$W" && J=$(sbatch --parsable "$SB") && echo "JOB=$J" && tmux new -d -s val_watch_b60k "bash $C/val_watch.sh snowball_probe_val_b60k $J b60k" && tmux ls | grep b60k
