#!/bin/bash
# probe_ckpt_v2.sh <probe_name> <val_dir> <model_dir|base> [bridge_url] — standard-budget pass@8 probe of an exported
# checkpoint (or the Stage-3 base) on a task tree. probe_ckpt.sh (2026-09-06) plus the two 2026-09-07 lessons:
#   * the probe dir is moved to /e/data1/mmlaion + symlinked BEFORE sbatch (fscratch inode quota; moving it while the
#     job runs orphans the Slurm log), and
#   * NODES / ENGINES / CONC / WALL / VERIF are overridable, so a 1,840-task tree can run wider than the 12-node default.
# Geometry rule: NODES = 4 + ENGINES, and CONC = 32 x ENGINES is the ratio every probe so far has used.
#   e.g. NODES=20 ENGINES=16 CONC=512 WALL=10:00:00 bash probe_ckpt_v2.sh <name> tasks/r2egym-tt-v2-rest <export>/model http://10.128.1.2:9922
# Run on the Jupiter login node that hosts the bridge.
set -uo pipefail
P=$1; V=$2; M=$3; B=${4:-http://10.128.1.2:9922}
NODES=${NODES:-12}; ENGINES=${ENGINES:-8}; CONC=${CONC:-256}; WALL=${WALL:-05:00:00}; VERIF=${VERIF:-1200}
# coordinators: the dispatcher splits n_concurrent_trials across process_pool.num_coordinators, and a coordinator that
# multiplexes far more than ~66 trials starves its own asyncio loops (2026-09-07: 12 x 132 produced 150-250 spurious 30 s
# connect timeouts per step; 2026-09-09: the probe template's 4 coordinators x 256 trials halved probe throughput).
# Default keeps the proven ~64 trials per coordinator. CONNECT is the 09-07 arm fix, applied to probes too.
COORD=${COORD:-$(( CONC / 64 < 4 ? 4 : CONC / 64 ))}; CONNECT=${CONNECT:-120}
E=${SNOWBALL_EXP:-/e/fscratch/reformo/lee27/experiments}; export SNOWBALL_EXP=$E; C=/e/project1/transfernetx/lee27/code/snowball
# the task tree MUST reach the config as an absolute path: hydra's data.val_data is resolved by the trainer, not by the
# sbatch's CWD, so a relative "tasks/..." loads zero tasks and the job dies at eval_before_train with
# "evaluation dataloader produced no batches" ~17 min in (2026-09-09).
case "$V" in /*) ;; *) V=/e/fscratch/reformo/lee27/${V#./} ;; esac
[ -d "$V" ] || { echo "task tree not found: $V"; exit 1; }
MM=/e/data1/mmlaion/lee27/experiments
PY=/e/project1/transfernetx/lee27/code/envs/snowball/bin/python; O=/e/project1/transfernetx/lee27/code/OpenThoughts-Agent
export OMP_NUM_THREADS=1
MODELARG=(); [ "$M" != base ] && { [ -f "$M/config.json" ] || { echo "no config.json under $M"; exit 1; }; MODELARG=(--model "$M"); }
# relaunching the same probe name: make_snowball_probe.py rmtree's the dst and cannot rmtree a symlink, so clear the
# mmlaion dir + the fscratch symlink first. Refuse while a job of that name is in the queue.
if [ -L "$E/$P" ]; then
  squeue -h -u $USER -n $P -o %i | grep -q . && { echo "$P is still in squeue — scancel it first"; exit 1; }
  echo "clearing previous probe dir $(readlink $E/$P)"; rm -rf "$(readlink $E/$P)"; rm -f "$E/$P"
fi
python3 $C/make_snowball_probe.py --name $P --val-dir $V --k 8 --conc $CONC --max-in 61440 --max-out 4096 --max-model-len 65536 \
  --nodes $NODES --engines $ENGINES --parity --eval-timeout 1800 --verifier-timeout $VERIF --wall $WALL "${MODELARG[@]}" > /dev/null 2>&1 \
  || { echo "PROBE BUILD FAILED ($P)"; exit 1; }
python3 $C/fix_merged_keys.py $E/$P/configs/${P}_rl_config.json > /dev/null
python3 - $E/$P/configs/${P}_rl_config.json $COORD <<'PYEOF'
import json, sys
p, coord = sys.argv[1], sys.argv[2]
c = json.load(open(p))
a = [x for x in c["skyrl_hydra_args"]
     if not x.startswith("trainer.algorithm.use_tis=") and not x.startswith("trajectory_runner.process_pool.num_coordinators=")]
a += ["trainer.algorithm.use_tis=false", "trajectory_runner.process_pool.num_coordinators=%s" % coord]
c["skyrl_hydra_args"] = a; json.dump(c, open(p, "w"), indent=2)
PYEOF
$PY $C/validate_hydra_args.py $E/$P/configs/${P}_rl_config.json 2>&1 | tail -1
sed -i "/^export DCFT_RL_ENV=/a export APPTAINER_BRIDGE_URL=$B\nexport HARBOR_OPENAI_CONNECT_TIMEOUT_SEC=$CONNECT" $E/$P/sbatch/${P}_rl.sbatch
grep -q "APPTAINER_BRIDGE_URL=$B" $E/$P/sbatch/${P}_rl.sbatch || { echo "bridge pin failed"; exit 1; }
grep -q "HARBOR_OPENAI_CONNECT_TIMEOUT_SEC=$CONNECT" $E/$P/sbatch/${P}_rl.sbatch || { echo "connect-timeout pin failed"; exit 1; }
# storage: probe trees live on mmlaion, reached through the fscratch symlink every script hard-codes (2026-09-07)
mkdir -p $MM
if [ ! -L $E/$P ]; then rm -rf $MM/$P; mv $E/$P $MM/$P && ln -s $MM/$P $E/$P && echo "probe dir -> $(readlink $E/$P)"; fi
cd $O && DCFT=$PWD sbatch $E/$P/sbatch/${P}_rl.sbatch
tmux new -d -s probe_watch_$P "bash $C/probe_watch.sh $P"
echo "$P: val=$V model=$M bridge=$B nodes=$NODES engines=$ENGINES conc=$CONC coord=$COORD connect=$CONNECT wall=$WALL watch=$(tmux ls 2>/dev/null | grep -c probe_watch_$P)"
