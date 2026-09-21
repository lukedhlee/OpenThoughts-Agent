#!/bin/bash
# ota_lane.sh — one LANE of Snowball SFT arms on an OTA cache: for each learning rate in LRS, train one arm
# (standard SFT, or TailSFT when TAIL_FRACTION > 0), then export the requested epoch checkpoints and score each
# export on HELDOUT with heldout_nll.sbatch. Arms in a lane run one after another (the chain waits on each run
# job); two lanes may run side by side, started a few minutes apart (gotchas 2026-09-17: two 64-rank JAX jobs
# compiling in the same minute wedged one of them).
#
# Required env: LANE (name), CACHE (built cache dir), LRS ("2e-5 5e-5"), EPOCHS, HELDOUT (parquet), OUT_ROOT.
# Optional: TAIL_FRACTION (0 = standard), TAIL_REF (.npy, required when TAIL_FRACTION > 0), EXPORT_EPOCHS
# ("final" = last checkpoint only, "all" = every packed epoch, "kept" = every kept checkpoint), KEEP_EVERY,
# SNOWBALL_WALL, SBATCH_ACCOUNT, TAG (run-id suffix),
# SCORE_TIME (scoring job wall; ~1 s per held-out row + 5 min server start).
# Runs on a LOGIN node inside tmux; every heavy step is a Slurm job. Log: $S/logs/$SNOWBALL_STAGE/lane_$LANE.log
set -uo pipefail
S=/e/data1/mmlaion/lee27/snowball-sft
C=/e/project1/transfernetx/lee27/code/snowball
MOE=/e/project1/transfernetx/lee27/code/marin-sft/experiments/june_tpu_67b_a2b/moe
TOK=/e/fscratch/reformo/lee27/models/snowball-s3-nemotron-terminal-step1888
: "${LANE:?}" "${CACHE:?}" "${LRS:?}" "${EPOCHS:?}" "${HELDOUT:?}" "${OUT_ROOT:?}"
TAIL_FRACTION=${TAIL_FRACTION:-0}
EXPORT_EPOCHS=${EXPORT_EPOCHS:-final}
TAG=${TAG:-}
# stage + provenance are overridable so a sibling stage (ota_if = OTA + the if-v2 slice) reuses this lane
export SNOWBALL_STAGE=${SNOWBALL_STAGE:-ota}
export SNOWBALL_DATASET_ID=${SNOWBALL_DATASET_ID:-open-thoughts/OpenThoughts-Agent-SFT-100K}
export SNOWBALL_DATASET_REVISION=${SNOWBALL_DATASET_REVISION:-45fb28fcc38d352133cb28a1c8a43a2f14fea97b}
export SNOWBALL_EXP=$S/experiments/snowball-ota-sft
export SNOWBALL_CACHE=$CACHE
export SNOWBALL_PARQUET_LIST=${PARQUET_LIST:-$S/data/ota_sft_100k_v2/parquet.list}
export EPOCHS
# permanent checkpoints: one per packed epoch by default, or every KEEP_EVERY steps (the full pair keeps every
# 210 so one epoch yields a data-scaling ladder at ~29k / 58k / 88k trajectories seen)
if [ -n "${KEEP_EVERY:-}" ]; then export SNOWBALL_KEEP_EVERY=$KEEP_EVERY; else export SNOWBALL_KEEP_PER_EPOCH=1; fi
export SNOWBALL_WALL=${SNOWBALL_WALL:-02:00:00}
export SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-laionize}
export CHAIN_STEPS=run
# lr-schedule horizon and in-place resume (launch_jupiter_snowball_r2egym.sh): both pass through the environment
export SNOWBALL_SCHEDULE_EPOCHS=${SNOWBALL_SCHEDULE_EPOCHS:-$EPOCHS} SNOWBALL_RESUME=${SNOWBALL_RESUME:-0}
if [ "$TAIL_FRACTION" != 0 ]; then
  : "${TAIL_REF:?TailSFT needs the reference .npy}"
  [ -f "$TAIL_REF" ] || { echo "no reference vector at $TAIL_REF"; exit 1; }
  export SNOWBALL_TAIL_FRACTION=$TAIL_FRACTION SNOWBALL_TAIL_REF=$TAIL_REF
fi
mkdir -p "$S/logs/$SNOWBALL_STAGE" "$OUT_ROOT"
LOG=$S/logs/$SNOWBALL_STAGE/lane_$LANE.log
say() { echo "[$(date -u +%FT%TZ)] [$LANE] $*" | tee -a "$LOG"; }
say "LANE_START lrs='$LRS' epochs=$EPOCHS schedule_epochs=$SNOWBALL_SCHEDULE_EPOCHS resume=$SNOWBALL_RESUME tail=$TAIL_FRACTION cache=$CACHE heldout=$HELDOUT"

for LR in $LRS; do
  ARM=lr$LR-sched$SNOWBALL_SCHEDULE_EPOCHS$TAG   # the dir is named by the schedule, not the epochs run, so a resume reuses it
  [ "$TAIL_FRACTION" != 0 ] && ARM=$ARM-tail${TAIL_FRACTION#0.}
  export SNOWBALL_LR=$LR
  export SNOWBALL_OUTPUT=$OUT_ROOT/$ARM
  export SNOWBALL_RUN_ID=snowball-$SNOWBALL_STAGE-$LANE-$ARM
  # the chain keeps its "run done" marker per stage; give every arm its own marker dir via SNOWBALL_SCRATCH? No:
  # the marker is $S/logs/$SNOWBALL_STAGE/.done.run, shared. Remove it before each arm so a finished earlier arm is not
  # mistaken for this one (the chain also checks the checkpoint dir, which is per arm).
  rm -f "$S/logs/$SNOWBALL_STAGE/.done.run"
  say "ARM_START $ARM lr=$LR output=$SNOWBALL_OUTPUT"
  if ! bash -l "$C/snowball_sft_chain.sh" >> "$LOG" 2>&1; then say "ARM_FAILED $ARM (chain); continuing with the next lr"; continue; fi
  STEPS=$(cat "$S/logs/$SNOWBALL_STAGE/.done.run" 2>/dev/null || true)
  [ -n "$STEPS" ] || { say "ARM_FAILED $ARM (no step count)"; continue; }
  EPOCH_STEPS=$((STEPS / EPOCHS))
  say "ARM_DONE $ARM steps=$STEPS epoch=$EPOCH_STEPS"
  if [ "$EXPORT_EPOCHS" = all ]; then points=$(seq 1 "$EPOCHS" | awk -v E="$EPOCH_STEPS" -v S="$STEPS" -v N="$EPOCHS" '{print ($1==N)?S:$1*E}')
  elif [ "$EXPORT_EPOCHS" = kept ]; then points=$(ls "$SNOWBALL_OUTPUT/checkpoints" | grep -oE '^step-[0-9]+$' | cut -d- -f2 | sort -n)
  else points=$STEPS; fi
  for st in $points; do
    e=$st
    CK=$SNOWBALL_OUTPUT/checkpoints/step-$st
    EX=$SNOWBALL_OUTPUT/export-step$st-hf-bf16
    [ -d "$CK" ] || { say "no checkpoint $CK; skipping"; continue; }
    if [ -f "$EX/config.json" ]; then jx=""; else
      jx=$(sbatch --parsable -o "$S/logs/snowball-export.%j.log" --account="$SBATCH_ACCOUNT" \
        --export=ALL,MARIN_ROOT=/e/project1/transfernetx/lee27/code/marin-sft,MARIN_PYTHON=/e/project1/transfernetx/lee27/code/envs/marin-grug-sft/bin/python,SNOWBALL_EXPORT_CHECKPOINT="$CK",SNOWBALL_EXPORT_OUTPUT="$EX",SNOWBALL_EXPORT_TOKENIZER="$TOK" \
        "$MOE/jupiter_snowball_export.sbatch") || { say "export submit failed for $ARM step $st"; continue; }
      say "EXPORT_SUBMITTED $ARM step $st job $jx -> $EX"
    fi
    NAME=$SNOWBALL_STAGE-$LANE-$ARM-step$st
    [ -f "$S/logs/heldout_nll_$NAME.json" ] && { say "score exists for step $st; skipping"; continue; }   # a resume re-lists the old kept steps
    js=$(SBATCH_TIMELIMIT=${SCORE_TIME:-00:45:00} sbatch --parsable ${jx:+--dependency=afterok:$jx} --account="$SBATCH_ACCOUNT" \
      --export=ALL,MODEL="$EX",PARQUET="$HELDOUT",NAME="$NAME" "$C/heldout_nll.sbatch") \
      && say "SCORE_SUBMITTED $ARM step $st job $js -> $S/logs/heldout_nll_$NAME.json" \
      || say "score submit failed for $ARM step $st"
  done
done
say "LANE_DONE; curve: grep -h '\"nll\"' $S/logs/heldout_nll_$SNOWBALL_STAGE-$LANE-*.json"
