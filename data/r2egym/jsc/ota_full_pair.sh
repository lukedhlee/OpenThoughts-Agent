#!/bin/bash
# ota_full_pair.sh — the scale-up: standard SFT and TailSFT on the FULL OTA v2 set (87,697 traces), each at the lr
# the sweep picked (LR_STD, LR_TAIL). Luke 2026-09-19 03:00 PT: ONE epoch for now, saved so it can be resumed any
# time. So each arm trains EPOCHS=1 of a SCHEDULE_EPOCHS=2 cosine (the lr schedule spans two epochs; the epoch-1
# checkpoint keeps the optimizer state), exports and scores the epoch-1 checkpoint, and can be continued later
# with the resume line below: the trainer picks up its own checkpoint and the schedule continues unchanged.
# Cost, stated before starting, at EPOCHS=1 (~630 steps, 6.4 s per step on 16 nodes): prep 1 node ~1 h 45
# (1.8 node-h); each arm ~1 h 10 + start-up on 16 nodes (~20 node-h); TailSFT reference pass ~50 min on 16 nodes
# (~13 node-h); a checkpoint every 210 steps (~29k / 58k / 88k trajectories seen) -> 6 exports (2.4) and base + 6
# scorings of the 1,200-row held-out (2.8), the data-scaling ladder inside one epoch. About 60 node-hours in all.
# Resume an arm to its second epoch (~20 node-h) later:
#   SNOWBALL_RESUME=1 LANE=fullstd CACHE=$EXP/cache-all-v2 LRS=<lr> EPOCHS=2 SNOWBALL_SCHEDULE_EPOCHS=2 \
#     EXPORT_EPOCHS=final HELDOUT=<same> OUT_ROOT=$EXP/full [TAIL_FRACTION=0.25 TAIL_REF=...] bash ota_lane.sh
#   (the arm's output dir name carries the run's EPOCHS, so pass TAG=-resumed or reuse the dir by hand).
set -uo pipefail
: "${LR_STD:?}" "${LR_TAIL:?}"
EPOCHS=${EPOCHS:-1}
export SNOWBALL_SCHEDULE_EPOCHS=${SNOWBALL_SCHEDULE_EPOCHS:-2}
S=/e/data1/mmlaion/lee27/snowball-sft
C=/e/project1/transfernetx/lee27/code/snowball
D=$S/data/ota_sft_100k_v2
EXP=$S/experiments/snowball-ota-sft
CACHE=$EXP/cache-all-v2
HELDOUT=$D/ota-all-heldout-00000-of-00001.parquet   # all 4,728 held-out rows: scoring 1,200 took 200 s, so the full file is ~13 min per export
LOG=$S/logs/ota/full_pair.log; mkdir -p "$S/logs/ota"
say() { echo "[$(date -u +%FT%TZ)] [full] $*" | tee -a "$LOG"; }
say "FULL_PAIR_START lr_std=$LR_STD lr_tail=$LR_TAIL epochs=$EPOCHS schedule_epochs=$SNOWBALL_SCHEDULE_EPOCHS"

if [ ! -f "$CACHE/train/.stats.json" ]; then
  rm -f "$S/logs/ota/.done.prep"
  SBATCH_TIMELIMIT=03:00:00 SNOWBALL_STAGE=ota SNOWBALL_PARQUET_LIST=$D/parquet.list SNOWBALL_DATASET_ID=open-thoughts/OpenThoughts-Agent-SFT-100K \
  SNOWBALL_DATASET_REVISION=45fb28fcc38d352133cb28a1c8a43a2f14fea97b SNOWBALL_EXP=$EXP SNOWBALL_CACHE=$CACHE \
  SBATCH_ACCOUNT=laionize CHAIN_STEPS=prep bash -l "$C/snowball_sft_chain.sh" || { say "prep failed"; exit 1; }
fi
say "cache ready: $(head -c 300 "$CACHE/train/.stats.json" 2>/dev/null)"

jb=$(SBATCH_TIMELIMIT=01:00:00 sbatch --parsable --account=laionize --export=ALL,MODEL=/e/fscratch/reformo/lee27/models/snowball-s3-nemotron-terminal-step1888,PARQUET="$HELDOUT",NAME=ota-all-base "$C/heldout_nll.sbatch") && say "base score job $jb"

# tmux sessions take the tmux SERVER's environment, not this script's, so every knob goes on the command line.
tmux new -d -s ota_full_std "SNOWBALL_SCHEDULE_EPOCHS=$SNOWBALL_SCHEDULE_EPOCHS LANE=fullstd CACHE=$CACHE PARQUET_LIST=$D/parquet.list LRS='$LR_STD' EPOCHS=$EPOCHS EXPORT_EPOCHS=kept KEEP_EVERY=210 HELDOUT=$HELDOUT OUT_ROOT=$EXP/full SNOWBALL_WALL=03:00:00 SCORE_TIME=01:00:00 bash -l $C/ota_lane.sh; sleep 3600"
say "std lane started (tmux ota_full_std)"
for i in $(seq 1 120); do squeue -u lee27 -h -o "%j %T" | grep -q "snowball-ota-fullstd.* RUNNING" && break; sleep 30; done
sleep 300
tmux new -d -s ota_full_tail "CACHE=$CACHE REF_OUT=$EXP/tail/ref_all_v2.npy REF_EPOCHS=4 SNOWBALL_WALL=02:00:00 bash -l $C/ota_tail_ref.sh && SNOWBALL_SCHEDULE_EPOCHS=$SNOWBALL_SCHEDULE_EPOCHS LANE=fulltail CACHE=$CACHE PARQUET_LIST=$D/parquet.list LRS='$LR_TAIL' EPOCHS=$EPOCHS EXPORT_EPOCHS=kept KEEP_EVERY=210 HELDOUT=$HELDOUT OUT_ROOT=$EXP/full TAIL_FRACTION=0.25 TAIL_REF=$EXP/tail/ref_all_v2.npy SNOWBALL_WALL=03:00:00 SCORE_TIME=01:00:00 bash -l $C/ota_lane.sh; sleep 3600"
say "tail lane started (tmux ota_full_tail)"
say "FULL_PAIR_LAUNCHED"
