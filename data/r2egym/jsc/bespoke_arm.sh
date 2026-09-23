#!/bin/bash
# bespoke_arm.sh — one SFT arm of the Grug Datakit 09-21 checkpoint (grug-datakit-sft-20260921) on one variant of the
# private bespoke Qwen/GLM successful-trace corpus (STAGE = bespoke_fold_all | bespoke_fold_noglm | bespoke_think_all |
# bespoke_think_noglm; marin vista_snowball_chat.py STAGES). Modelled on ota3_arm.sh; what differs:
#   base      09-21, imported once to $INIT with its own qk_mult 1.75 / max_position_embeddings 262,144 (snowball_base.json
#             sidecar) and pending_qb_betas = -router_bias; the stage freezes the router bias there for the whole run
#   template  09-21 training_chat_template.jinja, per-row enable_thinking, reasoning -> reasoning_content; 09-21 tokenizer
#   layout    16 sequences x 65,536 tokens per step on 4 nodes (SEQ_LEN / BATCH / NODES override)
#   recipe    LR 3e-5, EPOCHS 3 on one cosine over those epochs, warmup 5 % of the steps (min 2), a permanent
#             checkpoint at each epoch end, export (09-21 config/templates/tokenizer) + held-out NLL per kept checkpoint
# The data is private and lives only under $D on Jupiter: nothing here (or in the marin path) resolves the dataset id,
# and the Hub is forced offline for every job this script starts.
#
# Modes (run on a Jupiter LOGIN node inside tmux; every heavy step is a Slurm job):
#   STAGE=bespoke_think_all PREP_ONLY=1 bash bespoke_arm.sh   # cache for the variant + the one-time 09-21 import, then stop
#   STAGE=bespoke_think_all PROBE=1 bash bespoke_arm.sh       # PROBE_STEPS (2) train steps at the layout, peak memory, no ckpt
#   STAGE=bespoke_think_all bash bespoke_arm.sh               # the full arm
#
# Cost, stated before any submission (standing rule; fill the numbers from PREP_ONLY's cache size before the go):
#   prep    1 node  x ~__ min                          = __ node-h   (4 min per 50M tokens measured)
#   import  4 nodes x ~5 min (once for all variants)   = ~0.3 node-h
#   probe   4 nodes x ~__ min (startup + 2 steps)      = __ node-h   (~0.5 expected)
#   run     4 nodes x ~__ h (__ M tokens x 3 epochs)   = __ node-h   (~5.1k tokens/s/GPU measured at 32k -> ~82k/s on 16 GPUs)
#   exports 4 nodes x ~5 min x 3 kept checkpoints       = ~1.0 node-h
#   scoring 1 node  x ~__ min x 3                       = __ node-h
#   total                                               = __ node-h
set -uo pipefail
S=/e/data1/mmlaion/lee27/snowball-sft
C=/e/project1/transfernetx/lee27/code/snowball
MOE=/e/project1/transfernetx/lee27/code/marin-sft/experiments/june_tpu_67b_a2b/moe
: "${STAGE:?bespoke_fold_all | bespoke_fold_noglm | bespoke_think_all | bespoke_think_noglm}"
case $STAGE in bespoke_fold_all|bespoke_fold_noglm|bespoke_think_all|bespoke_think_noglm) ;; *) echo "unknown STAGE=$STAGE" >&2; exit 1;; esac
VARIANT=${STAGE#bespoke_}
# v1 = rows <= 65,536 tokens; v1_48k = the same render filtered to <= 49,152 (16 x 65,536 OOMs on 4 nodes, 09-23)
DATA_REV=${DATA_REV:-v1}
RUN=$STAGE${DATA_REV#v1}   # per-rev paths; SNOWBALL_STAGE stays $STAGE (the provenance pin)
D=${DATA_DIR:-$S/data/bespoke_$DATA_REV/$VARIANT}
EXP=$S/experiments/snowball-bespoke-sft
CACHE=$EXP/cache-$RUN-v1
HELDOUT=$D/heldout-00000-of-00001.parquet
LR=${LR:-3e-5}
EPOCHS=${EPOCHS:-3}
PROBE_STEPS=${PROBE_STEPS:-2}

# base: 09-21 (tokenizer = its dir; import / train / export read its config through the chain and the sidecar)
export SNOWBALL_BASE=grug0921
export SNOWBALL_HF_BASE=/e/data1/mmlaion/lee27/models/grug-datakit-sft-20260921
export SNOWBALL_TOKENIZER=$SNOWBALL_HF_BASE
export SNOWBALL_INIT=$S/experiments/snowball-base-inits/init-dk0921-step0
# layout: sequences x packing length on NODES x 4 ranks (one sequence per GPU)
NODES=${NODES:-4}
export SNOWBALL_SEQ_LEN=${SEQ_LEN:-65536} SNOWBALL_BATCH=${BATCH:-16} SNOWBALL_NODES=$NODES SNOWBALL_DEVICES=$((NODES * 4))
STEP_TOKENS=$((SNOWBALL_SEQ_LEN * SNOWBALL_BATCH))
# provenance must equal the stage's pin in vista_snowball_chat.py (a string label; never an HF repo)
export SNOWBALL_STAGE=$STAGE
export SNOWBALL_DATASET_ID=private/bespoke-qwen-glm-successful-20260922 SNOWBALL_DATASET_REVISION=bespoke_v1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-laionize}
# held-out scoring at the packing length (heldout_nll.sbatch knobs)
export HELDOUT_MAX_LEN=$SNOWBALL_SEQ_LEN HELDOUT_MAX_MODEL_LEN=$((SNOWBALL_SEQ_LEN + 1024)) HELDOUT_MAX_POS=$((2 * SNOWBALL_SEQ_LEN))

mkdir -p "$S/logs/$RUN" "$EXP"; LOG=$S/logs/$RUN/arm.log
say() { echo "[$(date -u +%FT%TZ)] [$RUN] $*" | tee -a "$LOG"; }
[ -f "$D/parquet.list" ] || { say "no corpus at $D (parquet.list)"; exit 1; }
[ -f "$SNOWBALL_HF_BASE/config.json" ] || { say "no 09-21 base at $SNOWBALL_HF_BASE"; exit 1; }
say "ARM_START lr=$LR epochs=$EPOCHS layout=${SNOWBALL_BATCH}x${SNOWBALL_SEQ_LEN} nodes=$NODES data=$D prep_only=${PREP_ONLY:-0} probe=${PROBE:-0}"

# 1. cache (1 node) and the one-time 09-21 import (4 CPU nodes); both skip when done
if [ ! -f "$CACHE/train/.stats.json" ]; then
  rm -f "$S/logs/$STAGE/.done.prep"
  SBATCH_TIMELIMIT=02:00:00 SNOWBALL_PARQUET_LIST=$D/parquet.list SNOWBALL_EXP=$EXP SNOWBALL_CACHE=$CACHE \
  CHAIN_STEPS=prep bash -l "$C/snowball_sft_chain.sh" || { say "prep failed"; exit 1; }
fi
say "cache ready: $(head -c 300 "$CACHE/train/.stats.json" 2>/dev/null)"
if [ ! -f "$SNOWBALL_INIT/snowball_base.json" ]; then
  SNOWBALL_PARQUET_LIST=$D/parquet.list SNOWBALL_EXP=$EXP SNOWBALL_CACHE=$CACHE \
  CHAIN_STEPS=import bash -l "$C/snowball_sft_chain.sh" || { say "import failed"; exit 1; }
fi
say "init ready: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print({k: d[k] for k in ('qk_mult','max_position_embeddings','pending_qb_betas_from_router_bias')})" "$SNOWBALL_INIT/snowball_base.json")"
[ "${PREP_ONLY:-0}" = 1 ] && { say "PREP_DONE"; exit 0; }

TOK=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['total_tokens'])" "$CACHE/train/.stats.json")
EPOCH_STEPS=$(( (TOK + STEP_TOKENS - 1) / STEP_TOKENS ))
STEPS=$((EPOCHS * EPOCH_STEPS))
WARMUP=$(( (STEPS * 5 + 50) / 100 )); [ "$WARMUP" -ge 2 ] || WARMUP=2   # 5 % of the steps, at least 2 (1 would read as 100 %)
export SNOWBALL_WARMUP=$WARMUP
say "cache $TOK tokens -> $EPOCH_STEPS steps per epoch at $STEP_TOKENS tokens/step; $STEPS steps, warmup $WARMUP"

# 2. PROBE: the launcher alone (no chain: a probe leaves no checkpoint), then the max PROBE_MEM over all ranks
if [ "${PROBE:-0}" = 1 ]; then
  OUT=$EXP/$RUN/probe-$(date -u +%Y%m%dT%H%M%S)
  mkdir -p "$(dirname "$OUT")"
  out=$(SNOWBALL_SCRATCH=$S MARIN_ROOT=/e/project1/transfernetx/lee27/code/marin-sft \
        MARIN_PYTHON=/e/project1/transfernetx/lee27/code/envs/marin-grug-sft/bin/python \
        SNOWBALL_CACHE=$CACHE SNOWBALL_OUTPUT=$OUT SNOWBALL_RUN_ID=snowball-$RUN-probe SNOWBALL_LR=$LR \
        SNOWBALL_PROBE_STEPS=$PROBE_STEPS EPOCHS=$EPOCHS SNOWBALL_WALL=${PROBE_WALL:-00:45:00} \
        bash "$MOE/launch_jupiter_snowball_r2egym.sh" 2>&1); rc=$?
  echo "$out" | tee -a "$LOG"
  [ $rc -eq 0 ] || { say "PROBE launch failed rc=$rc"; exit 1; }
  j=$(echo "$out" | grep -oE 'Submitted batch job [0-9]+' | awk '{print $NF}' | tail -1)
  [ -n "$j" ] || { say "PROBE: no job id"; exit 1; }
  say "PROBE_SUBMITTED job $j ($PROBE_STEPS steps) -> $S/logs/snowball-$STAGE.$j.log"
  while squeue -j "$j" -h 2>/dev/null | grep -q .; do sleep ${POLL:-900}; done   # watcher cadence rule: 15 min
  JL=$S/logs/snowball-$STAGE.$j.log
  st=$(sacct -j "$j" -X -n -o State%20 2>/dev/null | head -1 | awk '{print $1}')
  peak=$(grep -h 'PROBE_MEM step=' "$JL" "$S/logs/forensics/$j/run.log" 2>/dev/null | grep -oE 'peak_gib=[0-9.]+' | cut -d= -f2 | sort -n | tail -1)
  limit=$(grep -h 'PROBE_MEM step=' "$JL" 2>/dev/null | grep -oE 'limit_gib=[0-9.]+' | cut -d= -f2 | sort -n | head -1)
  oom=$(grep -c -iE 'RESOURCE_EXHAUSTED|out of memory' "$JL" 2>/dev/null)
  say "PROBE_DONE job $j state=$st max_peak_gib=${peak:-none} limit_gib=${limit:-?} oom_lines=$oom log=$JL"
  exit 0
fi

# 3. the arm: train (chain run step via the lane), then export + held-out NLL for every kept (epoch-end) checkpoint
SNOWBALL_RESUME=0 SNOWBALL_SCHEDULE_EPOCHS=$EPOCHS LANE=${LANE:-$RUN} CACHE=$CACHE PARQUET_LIST=$D/parquet.list LRS="$LR" \
EPOCHS=$EPOCHS EXPORT_EPOCHS=kept HELDOUT=$HELDOUT OUT_ROOT=$EXP/$RUN SNOWBALL_WALL=${WALL:-03:00:00} \
SCORE_TIME=${SCORE_TIME:-01:00:00} bash -l "$C/ota_lane.sh"
say "ARM_DONE"
