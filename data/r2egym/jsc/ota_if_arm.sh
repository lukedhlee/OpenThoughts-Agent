#!/bin/bash
# ota_if_arm.sh — the OTA + instruction-following mix arm (stage ota_if), paired against the standard OTA arm:
# same recipe (1e-4, one epoch of a 2-epoch cosine, a permanent checkpoint every 210 steps, export + held-out
# score per checkpoint), only the data differs: ota_if_sft_v1 = OTA v2 (87,697) + if-v2 clean rows (12,484).
# Cost, stated before starting: prep 1 node ~1 h 50 (1.8 node-h); run ~656 steps at 6.5 s on 16 nodes
# (~21 node-h); three exports (1.2) + three scorings (1.5) -> about 25 node-hours. Luke's go 2026-09-19 12:35 PT.
set -uo pipefail
S=/e/data1/mmlaion/lee27/snowball-sft
C=/e/project1/transfernetx/lee27/code/snowball
D=$S/data/ota_if_sft_v1
EXP=$S/experiments/snowball-ota-sft
CACHE=$EXP/cache-ifmix-v1
HELDOUT=$D/ota-all-heldout-00000-of-00001.parquet
LR=${LR:-1e-4}
export SNOWBALL_STAGE=ota_if
export SNOWBALL_DATASET_ID=open-thoughts/OpenThoughts-Agent-SFT-100K+open-athena/nemotron-gym-if-v2-qwen3.5-122b-32k-traces
export SNOWBALL_DATASET_REVISION=45fb28fcc38d352133cb28a1c8a43a2f14fea97b+50b7f77
mkdir -p "$S/logs/ota_if"; LOG=$S/logs/ota_if/ifmix_arm.log
say() { echo "[$(date -u +%FT%TZ)] [ifmix] $*" | tee -a "$LOG"; }
say "IFMIX_START lr=$LR data=$D"
if [ ! -f "$CACHE/train/.stats.json" ]; then
  rm -f "$S/logs/ota_if/.done.prep"
  SBATCH_TIMELIMIT=03:00:00 SNOWBALL_PARQUET_LIST=$D/parquet.list SNOWBALL_EXP=$EXP SNOWBALL_CACHE=$CACHE \
  SBATCH_ACCOUNT=laionize CHAIN_STEPS=prep bash -l "$C/snowball_sft_chain.sh" || { say "prep failed"; exit 1; }
fi
say "cache ready: $(head -c 300 "$CACHE/train/.stats.json" 2>/dev/null)"
SNOWBALL_STAGE=ota_if SNOWBALL_DATASET_ID=$SNOWBALL_DATASET_ID SNOWBALL_DATASET_REVISION=$SNOWBALL_DATASET_REVISION \
SNOWBALL_SCHEDULE_EPOCHS=2 LANE=ifmix CACHE=$CACHE PARQUET_LIST=$D/parquet.list LRS="$LR" EPOCHS=1 EXPORT_EPOCHS=kept \
KEEP_EVERY=210 HELDOUT=$HELDOUT OUT_ROOT=$EXP/ifmix SNOWBALL_WALL=02:00:00 SCORE_TIME=01:00:00 bash -l "$C/ota_lane.sh"
say "IFMIX_DONE"
