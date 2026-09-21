#!/bin/bash
# rl_d1034_chain.sh — Luke 2026-09-21 ("Later, RL the epoch 2 as well"): once D's epoch-2 export (step 1034) and the D-517
# learnability band (band_swap.sh -> $E/rl_d517/max_steps) both exist, run rl_from_sft_pipeline.sh for a second plain-GRPO
# arm from the epoch-2 export on the SAME band and step count, so the two RL readouts are paired on the start checkpoint.
set -uo pipefail
E=/e/fscratch/reformo/lee27/experiments; C=/e/project1/transfernetx/lee27/code/snowball
X=/e/data1/mmlaion/lee27/snowball-sft/experiments/snowball-ota-sft/ota3_if_rstsucc/lr1e-4-sched2/export-step1034-hf-bf16
LOG=$E/rl_d1034/chain.log; mkdir -p $E/rl_d1034
log(){ echo "$(date '+%F %T') $*" | tee -a $LOG; }
log "waiting for $X and $E/rl_d517/max_steps"
for i in $(seq 1 288); do [ -f $X/model.safetensors.index.json ] && [ -f $E/rl_d517/max_steps ] && break; sleep 300; done   # up to 24 h
[ -f $X/model.safetensors.index.json ] && [ -f $E/rl_d517/max_steps ] || { log "not ready after 24 h"; exit 1; }
TREE=$(grep -oE 'released arm [0-9]+ on [^,]+' $E/rl_d517/band_swap.log | tail -1 | awk '{print $NF}'); TREE=$(basename "$TREE")
MS=$(cat $E/rl_d517/max_steps); log "band tree $TREE, max_steps $MS"
[ -d /e/fscratch/reformo/lee27/tasks/$TREE ] || { log "tree $TREE missing"; exit 1; }
echo $MS > $E/rl_d1034/max_steps
NAME=rl_d1034 ARM=snowball_ttband_ota3d1034_a MODEL=$X TAG=rld1034 TRIALS=3 TRAIN_TREE=$TREE MAX_STEPS=$MS bash $C/rl_from_sft_pipeline.sh
