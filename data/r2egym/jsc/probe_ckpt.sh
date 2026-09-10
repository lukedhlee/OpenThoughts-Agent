#!/bin/bash
# probe_ckpt.sh <probe_name> <val_dir> <model_dir|base> [bridge_url] — standard-budget held-out probe of an exported checkpoint (or the
# Stage-3 base) on a task tree: k 8, 61,440/4,096, conc 256, 12 nodes / 8 engines, parity, eval 1,800 s, verifier 1,200 s, use_tis=false,
# bridge pinned. The export_probe.sh recipe without the checkpoint wait/export (2026-09-06). Run on the Jupiter login node.
set -uo pipefail
P=$1; V=$2; M=$3; B=${4:-http://10.128.1.2:9924}
E=${SNOWBALL_EXP:-/e/fscratch/reformo/lee27/experiments}; export SNOWBALL_EXP=$E; C=/e/project1/transfernetx/lee27/code/snowball
PY=/e/project1/transfernetx/lee27/code/envs/snowball/bin/python; O=/e/project1/transfernetx/lee27/code/OpenThoughts-Agent
export OMP_NUM_THREADS=1
MODELARG=(); [ "$M" != base ] && { [ -f "$M/config.json" ] || { echo "no config.json under $M"; exit 1; }; MODELARG=(--model "$M"); }
python3 $C/make_snowball_probe.py --name $P --val-dir $V --k 8 --conc 256 --max-in 61440 --max-out 4096 --max-model-len 65536 \
  --nodes 12 --engines 8 --parity --eval-timeout 1800 --verifier-timeout 1200 --wall 05:00:00 "${MODELARG[@]}" > /dev/null 2>&1 || { echo "PROBE BUILD FAILED ($P)"; exit 1; }
python3 - $E/$P/configs/${P}_rl_config.json <<'PYEOF'
import json, sys
p = sys.argv[1]; c = json.load(open(p)); a = [x for x in c["skyrl_hydra_args"] if not x.startswith("trainer.algorithm.use_tis=")]
a.append("trainer.algorithm.use_tis=false"); c["skyrl_hydra_args"] = a; json.dump(c, open(p, "w"), indent=2); print("model", c["model_path"].split("/")[-2] if c["model_path"].endswith("model") else c["model_path"].split("/")[-1])
PYEOF
python3 $C/fix_merged_keys.py $E/$P/configs/${P}_rl_config.json > /dev/null
$PY $C/validate_hydra_args.py $E/$P/configs/${P}_rl_config.json 2>&1 | tail -1
sed -i "/^export DCFT_RL_ENV=/a export APPTAINER_BRIDGE_URL=$B" $E/$P/sbatch/${P}_rl.sbatch
grep -q "APPTAINER_BRIDGE_URL=$B" $E/$P/sbatch/${P}_rl.sbatch || { echo "bridge pin failed"; exit 1; }
cd $O && DCFT=$PWD sbatch $E/$P/sbatch/${P}_rl.sbatch
tmux new -d -s probe_watch_$P "bash $C/probe_watch.sh $P"
echo "$P: val=$V model=$M bridge=$B watch=$(tmux ls | grep -c probe_watch_$P)"
