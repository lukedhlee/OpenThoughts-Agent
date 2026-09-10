#!/bin/bash
# relaunch_arm16.sh <run> <lr> [engines=16] [eval_interval=24] — relaunch a band arm on the DP-placement fix (MarinSkyRL fork
# fcecf833+) at <engines> engine nodes @ 64 trials/node (16 policy nodes @ fsdp 64 + <engines> = 32 nodes at 16 engines).
# REFUSES while the run still has a job in squeue (make_snowball_probe.py rmtree-s the run dir). Archives the previous attempt
# (logs, dumped rollouts, eval tables, config, sbatch) to <run>_logs_0903b/, regenerates the run with make_snowball_grpo.py,
# re-applies patch_arm.py (staleness 3, epochs 10, RLOO-N min-group 4, mask list, verifier 600 s) and submits.
# eval_before_train stays TRUE (generator default): the step-0 K=8 eval costs ~25 min at 16 engines and pairs with later evals.
set -euo pipefail
RUN=$1; LR=$2; ENG=${3:-16}; EVI=${4:-24}
E=/e/fscratch/reformo/lee27/experiments; C=/e/project1/transfernetx/lee27/code/snowball
OTA=/e/project1/transfernetx/lee27/code/OpenThoughts-Agent; TRAIN=/e/fscratch/reformo/lee27/tasks/r2egym-raw-v3-train-band60k-clean
squeue -h -u $USER -n $RUN -o %i | grep -q . && { echo "$RUN still has a job in squeue - cancel it first"; exit 1; }
[ -d $E/$RUN ] || { echo "no run dir $E/$RUN"; exit 1; }
A=$E/${RUN}_logs_0903b; mkdir -p $A
mv $E/$RUN/logs/*.out $A/ 2>/dev/null || true
[ -d $E/$RUN/$RUN/exports/dumped_data ] && cp -r $E/$RUN/$RUN/exports/dumped_data $A/dumped_data_prev || true
[ -d $E/$RUN/$RUN/eval_tables ] && cp -r $E/$RUN/$RUN/eval_tables $A/eval_tables_prev || true
cp $E/$RUN/configs/${RUN}_rl_config.json $A/prev_rl_config.json; cp $E/$RUN/sbatch/${RUN}_rl.sbatch $A/prev_rl.sbatch
echo "archived previous attempt to $A"
python3 $C/make_snowball_grpo.py --name $RUN --train-dir $TRAIN --lr $LR --fsdp 64 --engines $ENG --conc-per-engine 64 --eval-interval $EVI
python3 $C/patch_arm.py $E/$RUN/configs/${RUN}_rl_config.json 3 10
python3 $C/fix_merged_keys.py $E/$RUN/configs/${RUN}_rl_config.json  # merged-schema key rewrite (make_snowball_grpo.py still emits vllm_stats_interval / rollout.fanout.*; 2026-09-04 session 04b0b663)
$OTA/../envs/snowball/bin/python $C/validate_hydra_args.py $E/$RUN/configs/${RUN}_rl_config.json
grep -m1 -- "--nodes=" $E/$RUN/sbatch/${RUN}_rl.sbatch
cd $OTA && DCFT=$PWD sbatch $E/$RUN/sbatch/${RUN}_rl.sbatch
