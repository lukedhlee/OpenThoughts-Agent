#!/bin/bash
# rl_union_swap.sh — successor of rl_union_chain.sh (Luke 2026-09-21 12:10 PT: "accelerate"): the two RL arms are submitted NOW
# (rl_from_sft_pipeline.sh on the train-pool band as a placeholder) and HELD so they age in the queue; once the never-solved-
# pool screen has its LEARNABLE TREE, build the union tree (train-pool learnable ∪ rest-pool learnable), size 2 epochs of it,
# patch both held arms' configs (train_data + max_steps, as band_swap.sh does), write $D/max_steps for each pipeline, release.
# Usage (tmux): TRAIN_SCREEN=screen_ota3d517_20260921 REST_SCREEN=screen_ota3d517rest_20260921 bash rl_union_swap.sh
set -uo pipefail
TRAIN_SCREEN=${TRAIN_SCREEN:?}; REST_SCREEN=${REST_SCREEN:?}; EPOCHS=${EPOCHS:-2}; CKPT=${CKPT:-6}
E=/e/fscratch/reformo/lee27/experiments; T=/e/fscratch/reformo/lee27/tasks; C=/e/project1/transfernetx/lee27/code/snowball
SFT=/e/data1/mmlaion/lee27/snowball-sft/experiments/snowball-ota-sft/ota3_if_rstsucc/lr1e-4-sched2
UNION=${UNION:-screen_ota3d517_union_20260921_learnable_x16}; LOG=$E/rl_union/swap.log; mkdir -p $E/rl_union
log(){ echo "$(date '+%F %T') $*" | tee -a $LOG; }
A=$T/${TRAIN_SCREEN}_learnable_x16; [ -d $A ] || { log "missing $A"; exit 1; }
ARMS="rl_d517u:snowball_ttband_ota3d517u_a:export-step517-hf-bf16:rld517u rl_d1034u:snowball_ttband_ota3d1034u_a:export-step1034-hf-bf16:rld1034u"
# 1. submit both arms now on the placeholder band, then hold them
for spec in $ARMS; do IFS=: read n arm ex tag <<< "$spec"; mkdir -p $E/$n
  if [ ! -f $E/$n/job_B ]; then
    tmux new-session -d -s ${n}_pipeline "NAME=$n ARM=$arm MODEL=$SFT/$ex TAG=$tag TRIALS=3 TRAIN_TREE=$(basename $A) MAX_STEPS=24 bash $C/rl_from_sft_pipeline.sh; sleep 3600"
    for i in $(seq 1 60); do [ -f $E/$n/job_B ] && break; sleep 5; done
    [ -f $E/$n/job_B ] || { log "$n: no job_B after 5 min (pipeline log: $E/$n/pipeline.log)"; exit 1; }
  fi
  J=$(cat $E/$n/job_B); scontrol hold $J && log "$n: arm $J submitted on the placeholder band and HELD" || log "$n: hold $J failed"
done
# 2. wait for the rest-pool screen's learnable tree
log "waiting for LEARNABLE TREE of $REST_SCREEN"
for i in $(seq 1 240); do grep -q "LEARNABLE TREE" $E/$REST_SCREEN/controller.log 2>/dev/null && break; sleep 300; done   # up to 20 h
grep -q "LEARNABLE TREE" $E/$REST_SCREEN/controller.log 2>/dev/null || { log "no learnable tree after 20 h; arms left on hold"; exit 1; }
B=$T/${REST_SCREEN}_learnable_x16; [ -d $B ] || { log "missing $B"; exit 1; }
# 3. union tree (dirs of symlinks <task>__rN -> r2egym-daytona-v3/<task>)
if [ ! -d $T/$UNION ]; then
  mkdir -p $T/$UNION.tmp && cp -P $A/* $T/$UNION.tmp/ && cp -Pn $B/* $T/$UNION.tmp/ && mv $T/$UNION.tmp $T/$UNION || { log "union build failed"; exit 1; }
fi
NA=$(ls $A | sed 's/__r[0-9]*$//' | sort -u | wc -l); NB=$(ls $B | sed 's/__r[0-9]*$//' | sort -u | wc -l); N=$(ls $T/$UNION | sed 's/__r[0-9]*$//' | sort -u | wc -l)
STEPS=$(( ( (EPOCHS*N + 63) / 64 + CKPT - 1 ) / CKPT * CKPT ))
log "union tree $UNION: $N tasks ($NA train-pool + $NB rest-pool learnable) x16 = $(ls $T/$UNION | wc -l) entries; $EPOCHS epochs -> max_steps $STEPS"
# 4. patch + release each held arm (a started arm reads its config at launch, so refuse to patch one that already runs)
for spec in $ARMS; do IFS=: read n arm ex tag <<< "$spec"
  J=$(cat $E/$n/job_B); st=$(squeue -h -j $J -o %T 2>/dev/null); CFG=$E/$arm/configs/${arm}_rl_config.json
  [ "$st" = PENDING ] || { log "$n: arm $J is '$st', not PENDING: NOT patched, operator decides"; continue; }
  python3 - "$CFG" "$T/$UNION" "$STEPS" <<'PY'
import json, sys
cp, tree, steps = sys.argv[1], sys.argv[2], sys.argv[3]
c = json.load(open(cp)); k = 0
for i, a in enumerate(c["skyrl_hydra_args"]):
    if a.startswith("data.train_data="): c["skyrl_hydra_args"][i] = f'data.train_data=["{tree}"]'; k += 1
    elif a.startswith("trainer.max_steps="): c["skyrl_hydra_args"][i] = f"trainer.max_steps={steps}"; k += 1
c["train_data"] = [tree]; c["train_data_sources"] = [tree]
assert k == 2, k
json.dump(c, open(cp, "w"), indent=2); print("patched", cp)
PY
  [ $? -eq 0 ] || { log "$n: config patch FAILED; arm left on hold"; continue; }
  echo $STEPS > $E/$n/max_steps
  scontrol release $J && log "$n: released arm $J on $UNION, max_steps $STEPS" || log "$n: release $J FAILED"
done
