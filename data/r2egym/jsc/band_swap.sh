#!/bin/bash
# band_swap.sh — point a HELD, still-pending RL arm at the learnable band of its own start checkpoint once the Daytona
# screen (refresh_screen.py DAYTONA=1, launch_screen_daytona.sh) has sized it, then release the arm (Luke 2026-09-21:
# "run the probe for the 1-epoch model and RL on the learnable band for 2 epochs"). Two epochs over N band tasks at 64
# prompts per step = 2N/64 steps, rounded UP to a multiple of the 6-step checkpoint interval so the pipeline's
# cancel-at-final-checkpoint and export find it; trainer.epochs stays 3 (non-binding: the x16 tree makes one trainer
# epoch N/4 steps). Writes $D/max_steps for rl_from_sft_pipeline.sh.
# Usage (tmux): SCREEN=screen_ota3d517_20260921 NAME=rl_d517 ARM=snowball_ttband_ota3d517_a bash band_swap.sh
set -uo pipefail
SCREEN=${SCREEN:?}; NAME=${NAME:?}; ARM=${ARM:?}; EPOCHS=${EPOCHS:-2}; CKPT=${CKPT:-6}
E=/e/fscratch/reformo/lee27/experiments; T=/e/fscratch/reformo/lee27/tasks; D=$E/$NAME; LOG=$D/band_swap.log
CFG=$E/$ARM/configs/${ARM}_rl_config.json; TREE=$T/${SCREEN}_learnable_x16
log(){ echo "$(date '+%F %T') $*" | tee -a $LOG; }
log "waiting for LEARNABLE TREE of $SCREEN ($E/$SCREEN/controller.log)"
for i in $(seq 1 240); do grep -q "LEARNABLE TREE" $E/$SCREEN/controller.log 2>/dev/null && [ -d $TREE ] && break; sleep 300; done   # up to 20 h
grep -q "LEARNABLE TREE" $E/$SCREEN/controller.log 2>/dev/null && [ -d $TREE ] || { log "no learnable tree after 20 h; arm left on hold"; exit 1; }
line=$(grep "LEARNABLE TREE" $E/$SCREEN/controller.log | tail -1); log "$line"
N=$(echo "$line" | grep -oE '[0-9]+ tasks x16' | grep -oE '^[0-9]+'); [ -n "$N" ] || N=$(ls $TREE | sed 's/__r[0-9]*$//' | sort -u | wc -l)
STEPS=$(( ( (EPOCHS*N + 63) / 64 + CKPT - 1 ) / CKPT * CKPT ))
log "band N=$N tasks -> $EPOCHS epochs = $(( (EPOCHS*N + 63) / 64 )) steps -> max_steps $STEPS (multiple of $CKPT)"
J=$(cat $D/job_B); st=$(squeue -h -j $J -o %T 2>/dev/null)
[ "$st" = PENDING ] || { log "arm $J is '$st', not PENDING: config NOT patched (a started arm reads its config at launch); operator decides"; exit 1; }
python3 - "$CFG" "$TREE" "$STEPS" <<'PY'
import json, sys
cp, tree, steps = sys.argv[1], sys.argv[2], sys.argv[3]
c = json.load(open(cp)); n = 0
for i, a in enumerate(c["skyrl_hydra_args"]):
    if a.startswith("data.train_data="): c["skyrl_hydra_args"][i] = f'data.train_data=["{tree}"]'; n += 1
    elif a.startswith("trainer.max_steps="): c["skyrl_hydra_args"][i] = f"trainer.max_steps={steps}"; n += 1
c["train_data"] = [tree]; c["train_data_sources"] = [tree]
assert n == 2, n
json.dump(c, open(cp, "w"), indent=2); print("patched", cp)
PY
[ $? -eq 0 ] || { log "config patch FAILED; arm left on hold"; exit 1; }
echo $STEPS > $D/max_steps
grep -E "train_data=|max_steps=" $CFG | tee -a $LOG
scontrol release $J && log "released arm $J on $TREE, max_steps $STEPS" || log "scontrol release $J FAILED"
