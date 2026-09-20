#!/bin/bash
# ota3_arm.sh — one SFT arm on a 2026-09-20 corpus (STAGE = ota3_if | ota3_if_rst | ota3_if_rstsucc | rst_if |
# ota3_if_rst_fmt | ota3_if_rst_beh, the last two = data/rst/filter_corpus.py filter arms on the ota3_if_rst mix), the
# ota_if_arm.sh recipe unchanged: lr 1e-4, one epoch of a 2-epoch cosine, a permanent checkpoint at each third of the
# epoch (KEEP_EVERY = epoch steps / 3, from the cache's token count at 64 x 32,768 tokens per step), export + held-out
# NLL per checkpoint. PREP_ONLY=1 builds the cache and stops (step 5 of the plan; ~1 node-h per corpus).
#   STAGE=ota3_if PREP_ONLY=1 bash ota3_arm.sh
#   STAGE=ota3_if bash ota3_arm.sh            # cost line first: prep ~1 + run ~20 + exports/scores ~3 node-h
set -uo pipefail
S=/e/data1/mmlaion/lee27/snowball-sft
C=/e/project1/transfernetx/lee27/code/snowball
: "${STAGE:?ota3_if | ota3_if_rst | ota3_if_rstsucc | rst_if | ota3_if_rst_fmt | ota3_if_rst_beh}"
case $STAGE in ota3_if) D=$S/data/ota3_if_sft_v1;; rst_if) D=$S/data/rst_if_sft_v1;; *) D=$S/data/${STAGE}_v1;; esac   # corpus dirs from data/rst/build_corpora.py
EXP=$S/experiments/snowball-ota-sft
CACHE=$EXP/cache-$STAGE-v1
HELDOUT=$D/ota3-heldout-00000-of-00001.parquet
LR=${LR:-1e-4}
KEEP_EVERY=${KEEP_EVERY:-}   # default: a third of the epoch, computed from the cache's token count below
STEP_TOKENS=$((64 * 32768))  # packed sequences per step x packing length (the ota3_if cache: 1,036,278,000 tokens -> 495 steps)
export SNOWBALL_STAGE=$STAGE
# provenance must equal the stage's pin in vista_snowball_chat.py STAGES (validate_cache_provenance compares the strings)
OTA_ID=open-thoughts/OpenThoughts-Agent-SFT-100K; IF_ID=open-athena/nemotron-gym-if-v2-qwen3.5-122b-32k-traces; RST_ID=open-athena/recursive-task-synthesis-glm-5.3-rollouts
OTA_REV=45fb28fcc38d352133cb28a1c8a43a2f14fea97b; IF_REV=50b7f77; RST_REV=dd6f34cb
case $STAGE in
  ota3_if) export SNOWBALL_DATASET_ID=$OTA_ID+$IF_ID SNOWBALL_DATASET_REVISION=$OTA_REV+$IF_REV;;
  rst_if)  export SNOWBALL_DATASET_ID=$IF_ID+$RST_ID SNOWBALL_DATASET_REVISION=$IF_REV+$RST_REV;;
  *)       export SNOWBALL_DATASET_ID=$OTA_ID+$IF_ID+$RST_ID SNOWBALL_DATASET_REVISION=$OTA_REV+$IF_REV+$RST_REV;;
esac
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
TOK=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['total_tokens'])" "$CACHE/train/.stats.json")
EPOCH_STEPS=$(( (TOK + STEP_TOKENS - 1) / STEP_TOKENS )); [ -n "$KEEP_EVERY" ] || KEEP_EVERY=$(( EPOCH_STEPS / 3 ))
say "epoch ~$EPOCH_STEPS steps; checkpoint every $KEEP_EVERY"
SNOWBALL_SCHEDULE_EPOCHS=2 LANE=${LANE:-$STAGE} CACHE=$CACHE PARQUET_LIST=$D/parquet.list LRS="$LR" EPOCHS=1 EXPORT_EPOCHS=kept \
KEEP_EVERY=$KEEP_EVERY HELDOUT=$HELDOUT OUT_ROOT=$EXP/$STAGE SNOWBALL_WALL=02:00:00 SCORE_TIME=01:00:00 bash -l "$C/ota_lane.sh"
say "ARM_DONE"
