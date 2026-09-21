#!/bin/bash
# rl_union_chain.sh — Luke 2026-09-21 11:25 PT: the RL band must come from the WHOLE Daytona R2E-Gym pool, not the base's
# training pool. Waits for the never-solved-pool screen (POOL=rest) of the same checkpoint, builds the union learnable tree
# (train-pool learnable ∪ rest-pool learnable; both trees are dirs of symlinks <task>__rN -> r2egym-daytona-v3/<task>), sizes
# 2 epochs of it, then launches rl_from_sft_pipeline.sh for the epoch-1 and epoch-2 exports on that tree (two arms, two fleets).
# Usage (tmux): TRAIN_SCREEN=screen_ota3d517_20260921 REST_SCREEN=screen_ota3d517rest_20260921 bash rl_union_chain.sh
set -uo pipefail
TRAIN_SCREEN=${TRAIN_SCREEN:?}; REST_SCREEN=${REST_SCREEN:?}; EPOCHS=${EPOCHS:-2}; CKPT=${CKPT:-6}
E=/e/fscratch/reformo/lee27/experiments; T=/e/fscratch/reformo/lee27/tasks; C=/e/project1/transfernetx/lee27/code/snowball
SFT=/e/data1/mmlaion/lee27/snowball-sft/experiments/snowball-ota-sft/ota3_if_rstsucc/lr1e-4-sched2
UNION=${UNION:-screen_ota3d517_union_20260921_learnable_x16}; LOG=$E/rl_union/chain.log; mkdir -p $E/rl_union
log(){ echo "$(date '+%F %T') $*" | tee -a $LOG; }
log "waiting for LEARNABLE TREE of $REST_SCREEN"
for i in $(seq 1 240); do grep -q "LEARNABLE TREE" $E/$REST_SCREEN/controller.log 2>/dev/null && break; sleep 300; done   # up to 20 h
grep -q "LEARNABLE TREE" $E/$REST_SCREEN/controller.log 2>/dev/null || { log "no learnable tree after 20 h"; exit 1; }
A=$T/${TRAIN_SCREEN}_learnable_x16; B=$T/${REST_SCREEN}_learnable_x16
[ -d $A ] && [ -d $B ] || { log "missing tree: $A or $B"; exit 1; }
if [ ! -d $T/$UNION ]; then
  mkdir -p $T/$UNION.tmp && cp -P $A/* $T/$UNION.tmp/ && cp -Pn $B/* $T/$UNION.tmp/ && mv $T/$UNION.tmp $T/$UNION || { log "union build failed"; exit 1; }
fi
NA=$(ls $A | sed 's/__r[0-9]*$//' | sort -u | wc -l); NB=$(ls $B | sed 's/__r[0-9]*$//' | sort -u | wc -l); N=$(ls $T/$UNION | sed 's/__r[0-9]*$//' | sort -u | wc -l)
STEPS=$(( ( (EPOCHS*N + 63) / 64 + CKPT - 1 ) / CKPT * CKPT ))
log "union tree $UNION: $N tasks ($NA train-pool + $NB rest-pool learnable) x16 = $(ls $T/$UNION | wc -l) entries; $EPOCHS epochs -> max_steps $STEPS"
for pair in "rl_d517u snowball_ttband_ota3d517u_a export-step517-hf-bf16 rld517u" "rl_d1034u snowball_ttband_ota3d1034u_a export-step1034-hf-bf16 rld1034u"; do
  set -- $pair; mkdir -p $E/$1; echo $STEPS > $E/$1/max_steps
  tmux new-session -d -s ${1}_pipeline "NAME=$1 ARM=$2 MODEL=$SFT/$3 TAG=$4 TRIALS=3 TRAIN_TREE=$UNION MAX_STEPS=$STEPS bash $C/rl_from_sft_pipeline.sh; sleep 3600"
  log "launched pipeline tmux ${1}_pipeline (ARM $2, MODEL $3, tag $4, tree $UNION, max_steps $STEPS)"; sleep 20
done
