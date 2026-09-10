#!/bin/bash
# export_probe.sh <run> <step> <probe_name> [max_wait_min] — wait for checkpoint <step> of <run>, export it to HF layout, verify,
# build + submit a standard-budget held-out probe (224 val × 8, 61,440/4,096, conc 256, 12 nodes, use_tis=false), and start
# probe_watch.sh in a tmux. Session 20ae8169, 2026-09-05. Run inside a tmux on the login node; log to experiments/export_probe_<probe>.log.
set -uo pipefail
R=$1; STEP=$2; P=$3; MAXW=${4:-360}
E=/e/fscratch/reformo/lee27/experiments; C=/e/project1/transfernetx/lee27/code/snowball
PY=/e/project1/transfernetx/lee27/code/envs/snowball/bin/python; O=/e/project1/transfernetx/lee27/code/OpenThoughts-Agent
V=/e/fscratch/reformo/lee27/tasks/r2egym-tt-v2-val441; CK=$E/$R/$R/checkpoints
export OMP_NUM_THREADS=1
echo "$(date '+%F %T') waiting for $R global_step_$STEP (max ${MAXW} min)"
for i in $(seq 1 $((MAXW*2))); do
  [ "$(cat $CK/latest_ckpt_global_step.txt 2>/dev/null)" = "$STEP" ] && [ -f $CK/global_step_$STEP/trainer_state.pt ] \
    && [ -z "$(find $CK/global_step_$STEP -mmin -2 2>/dev/null | head -1)" ] && break
  sleep 30
done
[ "$(cat $CK/latest_ckpt_global_step.txt 2>/dev/null)" = "$STEP" ] || { echo "TIMEOUT: latest ckpt is $(cat $CK/latest_ckpt_global_step.txt 2>/dev/null)"; exit 1; }
echo "ckpt$STEP ready at $(date): $(du -sh $CK/global_step_$STEP | cut -f1)"
cd $C && J=$(RUN=$R STEP=$STEP sbatch --parsable --export=ALL,RUN=$R,STEP=$STEP export_hf.sbatch); echo "export job $J"
while squeue -h -j $J -o %T | grep -q .; do sleep 20; done
grep -E "index tensors|tensor-name set|done$" $E/exports/logs/snowball_export_hf_$J.out | cut -c1-120
M=$E/exports/${R}_step${STEP}/model
[ -f $M/model.safetensors.index.json ] || { echo "EXPORT MISSING"; exit 1; }
python3 $C/make_snowball_probe.py --name $P --val-dir $V --k 8 --conc ${CONC:-256} --max-in 61440 --max-out 4096 --max-model-len 65536 \
  --nodes ${NODES:-12} --engines ${ENGINES:-8} --parity --eval-timeout 1800 --verifier-timeout 1200 --wall ${WALL:-06:00:00} --model $M > /dev/null 2>&1 || { echo "PROBE BUILD FAILED"; exit 1; }
python3 - $E/$P/configs/${P}_rl_config.json <<'PYEOF'
import json, sys
p = sys.argv[1]; c = json.load(open(p)); a = c["skyrl_hydra_args"]
a = [x for x in a if not x.startswith("trainer.algorithm.use_tis=")]; a.append("trainer.algorithm.use_tis=false")
c["skyrl_hydra_args"] = a; json.dump(c, open(p, "w"), indent=2); print("model", c["model_path"])
PYEOF
python3 $C/fix_merged_keys.py $E/$P/configs/${P}_rl_config.json > /dev/null
$PY $C/validate_hydra_args.py $E/$P/configs/${P}_rl_config.json 2>&1 | tail -1
# Optional: BRIDGE=http://10.128.1.2:9926 export_probe.sh ... pins the probe to a sandbox pool other than the default 9922
# (JUWELS pool10 = 9926). Same line patch_arms.py writes; inserted after the DCFT_RL_ENV export the generator always emits.
if [ -n "${BRIDGE:-}" ]; then sed -i "/^export DCFT_RL_ENV=/a export APPTAINER_BRIDGE_URL=$BRIDGE" $E/$P/sbatch/${P}_rl.sbatch; echo "bridge pinned: $(grep -c "APPTAINER_BRIDGE_URL=$BRIDGE" $E/$P/sbatch/${P}_rl.sbatch)"; fi
M=/e/data1/mmlaion/lee27/experiments; mkdir -p $M; if [ ! -L $E/$P ]; then mv $E/$P $M/$P && ln -s $M/$P $E/$P && echo "probe dir moved to mmlaion: $(readlink $E/$P)"; fi   # 2026-09-07: fscratch inode quota
cd $O && DCFT=$PWD sbatch $E/$P/sbatch/${P}_rl.sbatch
tmux new -d -s probe_watch_$P "bash $C/probe_watch.sh $P"
echo "probe_watch tmux: $(tmux ls | grep -c probe_watch_$P)"
squeue -u $USER -h -o "%i %j %T %M %D"
echo "$(date '+%F %T') DONE"
