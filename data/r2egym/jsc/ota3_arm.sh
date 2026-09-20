#!/bin/bash
# ota3_arm.sh — one SFT arm on a 2026-09-20 corpus (STAGE = ota3_if | ota3_if_rst | ota3_if_rstsucc), the ota_if_arm.sh
# recipe unchanged: lr 1e-4, one epoch of a 2-epoch cosine, a permanent checkpoint every KEEP_EVERY steps, export +
# held-out NLL per checkpoint. PREP_ONLY=1 builds the cache and stops (step 5 of the plan; ~1 node-h per corpus).
#   STAGE=ota3_if PREP_ONLY=1 bash ota3_arm.sh
#   STAGE=ota3_if bash ota3_arm.sh            # cost line first: prep ~1 + run ~20 + exports/scores ~3 node-h
set -uo pipefail
S=/e/data1/mmlaion/lee27/snowball-sft
C=/e/project1/transfernetx/lee27/code/snowball
: "${STAGE:?ota3_if | ota3_if_rst | ota3_if_rstsucc}"
D=$S/data/${STAGE}_v1
EXP=$S/experiments/snowball-ota-sft
CACHE=$EXP/cache-$STAGE-v1
HELDOUT=$D/ota3-heldout-00000-of-00001.parquet
LR=${LR:-1e-4}
KEEP_EVERY=${KEEP_EVERY:-}   # default: a third of the epoch, computed from the cache's step count below
export SNOWBALL_STAGE=$STAGE
export SNOWBALL_DATASET_ID=open-thoughts/OpenThoughts-Agent-SFT-100K+open-athena/nemotron-gym-if-v2-qwen3.5-122b-32k-traces+open-athena/recursive-task-synthesis-glm-5.3-rollouts
export SNOWBALL_DATASET_REVISION=45fb28fcc38d352133cb28a1c8a43a2f14fea97b+50b7f77+dd6f34cb
mkdir -p "$S/logs/$STAGE"; LOG=$S/logs/$STAGE/arm.log
say() { echo "[$(date -u +%FT%TZ)] [$STAGE] $*" | tee -a "$LOG"; }
[ -f "$D/parquet.list" ] || { say "no corpus at $D"; exit 1; }
say "ARM_START lr=$LR data=$D prep_only=${PREP_ONLY:-0}"
if [ ! -f "$CACHE/train/.stats.json" ]; then
  rm -f "$S/logs/$STAGE/.done.prep"
  SBATCH_TIMELIMIT=03:00:00 SNOWBALL_PARQUET_LIST=$D/parquet.list SNOWBALL_EXP=$EXP SNOWBALL_CACHE=$CACHE \
  SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-laionize} CHAIN_STEPS=prep bash -l "$C/snowball_sft_chain.sh" || { say "prep failed"; exit 1; }
fi
say "cache ready: $(head -c 300 "$CACHE/train/.stats.json" 2>/dev/null)"
[ "${PREP_ONLY:-0}" = 1 ] && { say "PREP_DONE"; exit 0; }
SNOWBALL_SCHEDULE_EPOCHS=2 LANE=${LANE:-$STAGE} CACHE=$CACHE PARQUET_LIST=$D/parquet.list LRS="$LR" EPOCHS=1 EXPORT_EPOCHS=kept \
KEEP_EVERY=${KEEP_EVERY:-210} HELDOUT=$HELDOUT OUT_ROOT=$EXP/$STAGE SNOWBALL_WALL=02:00:00 SCORE_TIME=01:00:00 bash -l "$C/ota_lane.sh"
say "ARM_DONE"
