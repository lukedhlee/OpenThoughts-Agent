#!/bin/bash
# ota_full_pair.sh — the scale-up: standard SFT and TailSFT on the FULL OTA v2 set (~94k traces), each at the lr
# the sweep picked (LR_STD, LR_TAIL), EPOCHS packed epochs with a permanent checkpoint, an export and a held-out
# score per epoch, so each arm reads its own dose curve.
# Cost, stated before starting, at EPOCHS=2 (~600 steps per epoch, 6.4 s per step on 16 nodes): prep 1 node
# ~1 h 45 (1.8 node-h); each arm ~2 h 10 + start-up on 16 nodes (~36 node-h); TailSFT reference pass ~40 min on
# 16 nodes (~11 node-h); 4 exports (1.6); base + 4 scorings of the 2,728-row held-out at ~50 min (4.2).
# About 90 node-hours in all, ~5 h wall with the two lanes side by side.
set -uo pipefail
: "${LR_STD:?}" "${LR_TAIL:?}"
EPOCHS=${EPOCHS:-2}
S=/e/data1/mmlaion/lee27/snowball-sft
C=/e/project1/transfernetx/lee27/code/snowball
D=$S/data/ota_sft_100k_v2
EXP=$S/experiments/snowball-ota-sft
CACHE=$EXP/cache-all-v2
HELDOUT=$D/ota-sub-heldout-00000-of-00001.parquet   # 300 rows per slice, the same file the sweep scored, ~25 min per export
LOG=$S/logs/ota/full_pair.log; mkdir -p "$S/logs/ota"
say() { echo "[$(date -u +%FT%TZ)] [full] $*" | tee -a "$LOG"; }
say "FULL_PAIR_START lr_std=$LR_STD lr_tail=$LR_TAIL epochs=$EPOCHS"

if [ ! -f "$CACHE/train/.stats.json" ]; then
  rm -f "$S/logs/ota/.done.prep"
  SBATCH_TIMELIMIT=03:00:00 SNOWBALL_STAGE=ota SNOWBALL_PARQUET_LIST=$D/parquet.list SNOWBALL_DATASET_ID=open-thoughts/OpenThoughts-Agent-SFT-100K \
  SNOWBALL_DATASET_REVISION=45fb28fcc38d352133cb28a1c8a43a2f14fea97b SNOWBALL_EXP=$EXP SNOWBALL_CACHE=$CACHE \
  SBATCH_ACCOUNT=laionize CHAIN_STEPS=prep bash -l "$C/snowball_sft_chain.sh" || { say "prep failed"; exit 1; }
fi
say "cache ready: $(head -c 300 "$CACHE/train/.stats.json" 2>/dev/null)"

jb=$(SBATCH_TIMELIMIT=01:00:00 sbatch --parsable --account=laionize --export=ALL,MODEL=/e/fscratch/reformo/lee27/models/snowball-s3-nemotron-terminal-step1888,PARQUET="$HELDOUT",NAME=ota-sub-base-full "$C/heldout_nll.sbatch") && say "base score job $jb"

tmux new -d -s ota_full_std "LANE=fullstd CACHE=$CACHE PARQUET_LIST=$D/parquet.list LRS='$LR_STD' EPOCHS=$EPOCHS EXPORT_EPOCHS=all HELDOUT=$HELDOUT OUT_ROOT=$EXP/full SNOWBALL_WALL=05:00:00 SCORE_TIME=01:00:00 bash -l $C/ota_lane.sh; sleep 3600"
say "std lane started (tmux ota_full_std)"
for i in $(seq 1 120); do squeue -u lee27 -h -o "%j %T" | grep -q "snowball-ota-fullstd.* RUNNING" && break; sleep 30; done
sleep 300
tmux new -d -s ota_full_tail "CACHE=$CACHE REF_OUT=$EXP/tail/ref_all_v2.npy REF_EPOCHS=4 SNOWBALL_WALL=02:00:00 bash -l $C/ota_tail_ref.sh && LANE=fulltail CACHE=$CACHE PARQUET_LIST=$D/parquet.list LRS='$LR_TAIL' EPOCHS=$EPOCHS EXPORT_EPOCHS=all HELDOUT=$HELDOUT OUT_ROOT=$EXP/full TAIL_FRACTION=0.25 TAIL_REF=$EXP/tail/ref_all_v2.npy SNOWBALL_WALL=05:00:00 SCORE_TIME=01:00:00 bash -l $C/ota_lane.sh; sleep 3600"
say "tail lane started (tmux ota_full_tail)"
say "FULL_PAIR_LAUNCHED"
