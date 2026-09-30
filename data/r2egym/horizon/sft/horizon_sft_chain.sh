#!/bin/bash
# horizon_sft_chain.sh — the Snowball SFT chain on TACC Horizon (4x GB200 per node), one Slurm job after another,
# each gated on the previous job's state AND its OK marker: prep -> gate -> import -> run -> export. Horizon port of
# data/r2egym/jsc/snowball_sft_chain.sh as it was at d1fd1767 (the version the 2026-09-17 Kimi SWE-smith reference
# arms ran with, marin a3840f826): plain Stage-3 import (zeroed pending_qb_betas, per-batch router bias), no base
# selection. The marin side is branch lukedhlee/horizon-snowball-sft (a3840f826 + horizon_* scripts).
#
# Runs on the Horizon LOGIN node inside tmux (it only submits and polls; every heavy step is a Slurm job):
#   tmux new -d -s sft_chain "bash data/r2egym/horizon/sft/horizon_sft_chain.sh"
# Steps already done are skipped by their artifacts, so a rerun resumes at the failed step. CHAIN_STEPS narrows it.
# Cost for the Kimi reference (96 steps): prep 1 node ~15 min, gate 2 nodes ~5 min, import 4 nodes ~15 min,
# run 16 nodes ~25 min, export 4 nodes ~10 min -> about 9 node-hours.
set -uo pipefail
U=${USER}
S=${SNOWBALL_SCRATCH:-/scratch/11584/$U/snowball-sft}
C=${CODE_ROOT:-$HOME/snowball}
export MARIN_ROOT=${MARIN_ROOT:-$C/marin-sft}
export MARIN_PYTHON=${MARIN_PYTHON:-$C/envs/marin-grug-sft/bin/python}
export SNOWBALL_SCRATCH=$S
TOK=${SNOWBALL_TOKENIZER:-$(ls -d /scratch/11584/$U/hf_hub/models--laion--snowball-67b-a2b-sft-s3-nemotron-terminal-step1888/snapshots/* | head -1)}
EPOCHS=${EPOCHS:-3}
# The stage selects the marin STAGES entry (dataset pin, step ceiling). r2egym keeps its original paths;
# any other stage (e.g. kimi_swesmith) gets its own experiment dir and must name its parquet list and the
# dataset id / revision the stage pins. The imported step-0 init is shared: every agentic stage starts
# from the same Stage-3 export.
SFT_STAGE=${SNOWBALL_STAGE:-r2egym}
export SNOWBALL_STAGE=$SFT_STAGE
if [ "$SFT_STAGE" = r2egym ]; then EXP=$S/experiments/snowball-r2egym-sft; else EXP=${SNOWBALL_EXP:-$S/experiments/snowball-$SFT_STAGE-sft}; fi
PARQUET_LIST=${SNOWBALL_PARQUET_LIST:-$S/data/r2egym_glm47_solved_v1/parquet.list}
DATASET_ID=${SNOWBALL_DATASET_ID:-DCAgent/g1_clean_hybrid_scaffold_plus_r2eg_gfi_38k_glm47_traces}
DATASET_REVISION=${SNOWBALL_DATASET_REVISION:-4243a8f5cd39799803a6a0d52457fa0833068566}
CACHE=${SNOWBALL_CACHE:-$EXP/cache-v1}
INIT=${SNOWBALL_INIT:-$S/experiments/snowball-r2egym-sft/init-s3-step1888}
RUN_ID=${SNOWBALL_RUN_ID:-snowball-$SFT_STAGE-sft-run1}
if [ "$SFT_STAGE" = r2egym ]; then OUT=${SNOWBALL_OUTPUT:-$EXP/r2egym-glm47-solved-v1-run1}; else OUT=${SNOWBALL_OUTPUT:-$EXP/$SFT_STAGE-run1}; fi
MOE=$MARIN_ROOT/experiments/june_tpu_67b_a2b/moe
CHAIN_STEPS=${CHAIN_STEPS:-prep gate import run export}
LOG=$S/logs/chain.log
mkdir -p "$S/logs"
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
die() { say "CHAIN_FAILED: $*"; exit 1; }

job_state() {  # prints a terminal Slurm state, or "" while the job is queued/running/settling
  local st
  st=$(squeue -j "$1" -h -o %T 2>/dev/null | head -1)
  if [ -n "$st" ]; then echo ""; return; fi
  st=$(sacct -j "$1" -X -n -o State%24 2>/dev/null | head -1 | awk '{print $1}')
  case "$st" in
    COMPLETED|FAILED|CANCELLED*|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|BOOT_FAIL|DEADLINE) echo "$st";;
    *) echo "";;
  esac
}
wait_job() {  # $1 job id, $2 log file, $3 marker regex that must appear in the log ("" = exit code only)
  local j=$1 f=$2 m=$3 st n=0
  say "waiting on job $j -> $f"
  while :; do
    st=$(job_state "$j")
    [ -n "$st" ] && break
    n=$((n + 1)); sleep 60
  done
  say "job $j ended $st after ~${n} min of polling"
  [ "$st" = COMPLETED ] || return 1
  [ -z "$m" ] || grep -q -E "$m" "$f" || { say "marker '$m' missing in $f"; return 1; }
  return 0
}
submit() {  # runs a submitting command, logs its output, prints the job id it announced
  local out
  out=$("$@" 2>&1) || { echo "$out" | tee -a "$LOG" >&2; return 1; }
  echo "$out" | tee -a "$LOG" >&2
  echo "$out" | grep -oE 'Submitted batch job [0-9]+' | awk '{print $NF}' | tail -1
}

for p in "$MARIN_PYTHON" "$MOE/horizon_snowball_guarded.sbatch" "$TOK/tokenizer.json" "$PARQUET_LIST"; do
  [ -e "$p" ] || die "missing $p"
done
# Step markers are per stage so a second dataset never inherits the r2egym chain's "done" state.
if [ "$SFT_STAGE" = r2egym ]; then MARK=$S/logs; else MARK=$S/logs/$SFT_STAGE; mkdir -p "$MARK"; fi
say "CHAIN_START stage=$SFT_STAGE steps='$CHAIN_STEPS' epochs=$EPOCHS parquet_list=$PARQUET_LIST marin=$(git -C "$MARIN_ROOT" rev-parse --short HEAD) python=$MARIN_PYTHON scratch=$S"
STEPS=$(cat "$MARK/.done.run" 2>/dev/null || true)

for step in $CHAIN_STEPS; do
  case $step in
    prep)
      if [ -f "$CACHE/train/.stats.json" ] && [ -f "$MARK/.done.prep" ]; then say "prep: done already"; continue; fi
      j=$(submit sbatch -o "$S/logs/snowball-$SFT_STAGE-prep.%j.log" \
            --export=ALL,MARIN_ROOT="$MARIN_ROOT",SNOWBALL_ENV="$(dirname "$(dirname "$MARIN_PYTHON")")",SNOWBALL_SCRATCH="$S",SNOWBALL_TOKENIZER="$TOK",SNOWBALL_CACHE="$CACHE",SNOWBALL_PARQUET_LIST="$PARQUET_LIST",SNOWBALL_STAGE="$SFT_STAGE",SNOWBALL_DATASET_ID="$DATASET_ID",SNOWBALL_DATASET_REVISION="$DATASET_REVISION" \
            "$MOE/horizon_r2egym_prep.sbatch") || die "prep submit"
      [ -n "$j" ] || die "prep: no job id"
      wait_job "$j" "$S/logs/snowball-$SFT_STAGE-prep.$j.log" "PREP_DONE" || die "prep job $j"
      touch "$MARK/.done.prep"; say "prep OK ($j): $(grep -oE 'cache_tokens=[0-9]+ cache_examples=[0-9]+ cache_shards=[0-9]+ epoch_steps=[0-9]+' "$S/logs/snowball-$SFT_STAGE-prep.$j.log" | tail -1)";;
    gate)
      if [ -f "$S/logs/.done.gate" ]; then say "gate: done already"; continue; fi
      j=$(submit bash "$MOE/gate_horizon_snowball_env.sh") || die "gate submit"
      [ -n "$j" ] || die "gate: no job id"
      wait_job "$j" "$S/logs/snowball-env-gate.$j.log" "SNOWBALL_DISTRIBUTED_PROBE_OK" || die "gate job $j"
      touch "$S/logs/.done.gate"; say "gate OK ($j)";;
    import)
      if [ -f "$INIT/metadata.json" ]; then say "import: done already ($INIT)"; continue; fi
      j=$(submit sbatch -o "$S/logs/snowball-import.%j.log" \
            --export=ALL,MARIN_ROOT="$MARIN_ROOT",MARIN_PYTHON="$MARIN_PYTHON",SNOWBALL_HF_CHECKPOINT="$TOK",SNOWBALL_INIT="$INIT" \
            "$MOE/horizon_snowball_import.sbatch") || die "import submit"
      [ -n "$j" ] || die "import: no job id"
      wait_job "$j" "$S/logs/snowball-import.$j.log" "" || die "import job $j"
      [ -f "$INIT/metadata.json" ] || die "import job $j left no metadata.json in $INIT"
      say "import OK ($j): $(cat "$INIT/metadata.json")";;
    run)
      if [ -n "$STEPS" ] && [ -d "$OUT/checkpoints/step-$STEPS" ]; then say "run: done already (step-$STEPS)"; continue; fi
      out=$(EPOCHS=$EPOCHS SNOWBALL_STAGE="$SFT_STAGE" SNOWBALL_TOKENIZER="$TOK" SNOWBALL_OUTPUT="$OUT" SNOWBALL_RUN_ID="$RUN_ID" \
            SNOWBALL_CACHE="$CACHE" SNOWBALL_INIT="$INIT" bash "$MOE/launch_horizon_snowball_sft.sh" 2>&1); rc=$?
      echo "$out" | tee -a "$LOG"
      [ $rc -eq 0 ] || die "launch script rc=$rc"
      STEPS=$(echo "$out" | awk '/^steps +=/{print $3}')
      j=$(echo "$out" | grep -oE 'Submitted batch job [0-9]+' | awk '{print $NF}' | tail -1)
      { [ -n "$j" ] && [ -n "$STEPS" ]; } || die "could not parse the run job id / step count from the launch output"
      wait_job "$j" "$S/logs/snowball-$SFT_STAGE.$j.log" "GUARDED_RUN_EXIT rc=0" || die "run job $j (forensics under $S/logs/forensics/$j)"
      [ -d "$OUT/checkpoints/step-$STEPS" ] || die "no checkpoint step-$STEPS under $OUT"
      echo "$STEPS" > "$MARK/.done.run"; say "run OK ($j): $OUT/checkpoints/step-$STEPS";;
    export)
      [ -n "$STEPS" ] || die "export: unknown step count (no $MARK/.done.run)"
      EXPORT=$OUT/export-step$STEPS-hf-bf16
      if [ -f "$EXPORT/config.json" ]; then say "export: done already ($EXPORT)"; continue; fi
      j=$(submit sbatch -o "$S/logs/snowball-export.%j.log" \
            --export=ALL,MARIN_ROOT="$MARIN_ROOT",MARIN_PYTHON="$MARIN_PYTHON",SNOWBALL_EXPORT_CHECKPOINT="$OUT/checkpoints/step-$STEPS",SNOWBALL_EXPORT_OUTPUT="$EXPORT",SNOWBALL_EXPORT_TOKENIZER="$TOK" \
            "$MOE/horizon_snowball_export.sbatch") || die "export submit"
      [ -n "$j" ] || die "export: no job id"
      wait_job "$j" "$S/logs/snowball-export.$j.log" "EXPORT_CHECK_OK" || die "export job $j"
      say "export OK ($j): $EXPORT";;
    *) die "unknown step '$step'";;
  esac
done
say "CHAIN_DONE"
