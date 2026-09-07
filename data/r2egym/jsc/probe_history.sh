#!/bin/bash
# probe_history.sh <probe_name> <val_dir> <model_dir|base> <keep|drop|last:N|placeholder|head:N> [bridge_url]
# (placeholder / head:N need PROBE_OVERLAY=.../harbor_overlay/014e7562 or newer)
#
# probe_ckpt.sh under the history-think contract: the harbor overlay that carries
# HARBOR_TERMINUS2_HISTORY_THINK (fork commit e6adddd8) goes first on PYTHONPATH and the
# mode is exported right before the launch line, the way the x16 arm carries its overlay.
# Same budget as probe_ckpt.sh (k 8, 61,440/4,096, conc 256, 12 nodes / 8 engines, parity,
# eval 1,800 s, verifier 1,200 s, use_tis=false); wall 08:00:00 because dropped-reasoning
# episodes run more turns (PROBE_WALL overrides). Lives in code/snowball on Jupiter; this
# copy is the record. 2026-09-06.
set -uo pipefail
P=$1; V=$2; M=$3; MODE=$4; B=${5:-http://10.128.1.2:9926}
OVERLAY=${PROBE_OVERLAY:-/e/project1/transfernetx/lee27/code/harbor_overlay/e6adddd8}
E=/e/fscratch/reformo/lee27/experiments; C=/e/project1/transfernetx/lee27/code/snowball
PY=/e/project1/transfernetx/lee27/code/envs/snowball/bin/python; O=/e/project1/transfernetx/lee27/code/OpenThoughts-Agent
export OMP_NUM_THREADS=1
case "$MODE" in keep|drop|last:[0-9]*|placeholder|head:[0-9]*) ;; *) echo "mode must be keep, drop, last:N, placeholder or head:N"; exit 1;; esac
[ -d "$OVERLAY/harbor" ] || { echo "no overlay at $OVERLAY"; exit 1; }
MODELARG=(); [ "$M" != base ] && { [ -f "$M/config.json" ] || { echo "no config.json under $M"; exit 1; }; MODELARG=(--model "$M"); }
python3 $C/make_snowball_probe.py --name $P --val-dir $V --k 8 --conc 256 --max-in 61440 --max-out 4096 --max-model-len 65536 \
  --nodes 12 --engines 8 --parity --eval-timeout 1800 --verifier-timeout 1200 --wall ${PROBE_WALL:-08:00:00} "${MODELARG[@]}" > /dev/null 2>&1 || { echo "PROBE BUILD FAILED ($P)"; exit 1; }
python3 - $E/$P/configs/${P}_rl_config.json <<'PYEOF'
import json, sys
p = sys.argv[1]; c = json.load(open(p)); a = [x for x in c["skyrl_hydra_args"] if not x.startswith("trainer.algorithm.use_tis=")]
a.append("trainer.algorithm.use_tis=false"); c["skyrl_hydra_args"] = a; json.dump(c, open(p, "w"), indent=2); print("model", c["model_path"].split("/")[-2] if c["model_path"].endswith("model") else c["model_path"].split("/")[-1])
PYEOF
python3 $C/fix_merged_keys.py $E/$P/configs/${P}_rl_config.json > /dev/null
$PY $C/validate_hydra_args.py $E/$P/configs/${P}_rl_config.json 2>&1 | tail -1
S=$E/$P/sbatch/${P}_rl.sbatch
sed -i "/^export DCFT_RL_ENV=/a export APPTAINER_BRIDGE_URL=$B" $S
grep -q "APPTAINER_BRIDGE_URL=$B" $S || { echo "bridge pin failed"; exit 1; }
# overlay + mode immediately before the launch line
python3 - "$S" "$OVERLAY" "$MODE" <<'PYEOF'
import sys
path, overlay, mode = sys.argv[1:4]
lines = open(path).read().split("\n")
launch = [i for i, l in enumerate(lines) if l.startswith('"$RL_PYTHON" -m hpc.rl_launch_utils --config')]
assert len(launch) == 1, f"expected one launch line, found {len(launch)}"
block = [
    f"# history-think probe: harbor overlay {overlay.rsplit('/', 1)[-1]} + HARBOR_TERMINUS2_HISTORY_THINK={mode}",
    f"export PYTHONPATH={overlay}${{PYTHONPATH:+:$PYTHONPATH}}",
    f"export HARBOR_TERMINUS2_HISTORY_THINK={mode}",
    'echo "harbor overlay: $("$RL_PYTHON" -c \'import os, harbor.agents.terminus_2.terminus_2 as t; print(t.__file__, t.parse_history_think_mode(os.environ.get(t.HISTORY_THINK_ENV)))\')"',
    "",
]
lines[launch[0]:launch[0]] = block
open(path, "w").write("\n".join(lines))
PYEOF
grep -q "HARBOR_TERMINUS2_HISTORY_THINK=$MODE" $S || { echo "mode export failed"; exit 1; }
cd $O && DCFT=$PWD sbatch $S
tmux new -d -s probe_watch_$P "bash $C/probe_watch.sh $P"
echo "$P: val=$V model=$M mode=$MODE overlay=$OVERLAY bridge=$B watch=$(tmux ls | grep -c probe_watch_$P)"
