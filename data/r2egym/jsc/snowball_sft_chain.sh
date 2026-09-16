#!/bin/bash
# snowball_sft_chain.sh — run the Snowball R2E-Gym SFT chain on Jupiter end to end, one job after another,
# each gated on the previous job's Slurm state AND its OK marker in the log. Same order as the Vista
# runbook (ai_memory/active/snowball-sft/runbooks/vista_levanter_sft.md): prep -> gate -> import -> run
# -> export, with the Jupiter scripts from marin branch lukedhlee/vista-snowball-sft
# (experiments/june_tpu_67b_a2b/moe/jupiter_*).
#
# Runs on a LOGIN node inside tmux (it only submits and polls; every heavy step is a Slurm job):
#   tmux new -d -s sft_chain "bash -l /e/project1/transfernetx/$USER/code/snowball/snowball_sft_chain.sh"
# Steps already done are skipped by their artifacts (prep marker + cache stats, init metadata.json, the
# run's checkpoint, the export's config.json), so a rerun after a failure resumes at the failed step.
# CHAIN_STEPS narrows the list, e.g. CHAIN_STEPS="run export".
#
# Cost, stated before starting (standing rule): prep 1 node ~15 min, gate 2 nodes ~5 min, import 4 nodes
# ~10 min, run 16 nodes ~1 h (2 h wall; the watchdog cancels a wedge after 2 x 900 s), export 4 nodes
# ~15 min -> about 20 node-hours expected, under 40 at the wall caps. Nothing is submitted until this
# script is started.
set -uo pipefail
U=${USER}
S=${SNOWBALL_SCRATCH:-/e/data1/mmlaion/$U/snowball-sft}
C=${CODE_ROOT:-/e/project1/transfernetx/$U/code}
export MARIN_ROOT=${MARIN_ROOT:-$C/marin-sft}
export MARIN_PYTHON=${MARIN_PYTHON:-$C/envs/marin-grug-sft/bin/python}
export SNOWBALL_SCRATCH=$S
TOK=${SNOWBALL_TOKENIZER:-/e/fscratch/reformo/$U/models/snowball-s3-nemotron-terminal-step1888}
EPOCHS=${EPOCHS:-3}
EXP=$S/experiments/snowball-r2egym-sft
CACHE=${SNOWBALL_CACHE:-$EXP/cache-v1}
INIT=${SNOWBALL_INIT:-$EXP/init-s3-step1888}
RUN_ID=${SNOWBALL_RUN_ID:-snowball-r2egym-sft-run1}
OUT=${SNOWBALL_OUTPUT:-$EXP/r2egym-glm47-solved-v1-run1}
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

for p in "$MARIN_PYTHON" "$MOE/jupiter_snowball_guarded.sbatch" "$TOK/tokenizer.json" \
         "$S/data/r2egym_glm47_solved_v1/parquet.list"; do
  [ -e "$p" ] || die "missing $p"
done
say "CHAIN_START steps='$CHAIN_STEPS' epochs=$EPOCHS marin=$(git -C "$MARIN_ROOT" rev-parse --short HEAD) python=$MARIN_PYTHON scratch=$S"
STEPS=$(cat "$S/logs/.done.run" 2>/dev/null || true)

for step in $CHAIN_STEPS; do
  case $step in
    prep)
      if [ -f "$CACHE/train/.stats.json" ] && [ -f "$S/logs/.done.prep" ]; then say "prep: done already"; continue; fi
      j=$(submit sbatch -o "$S/logs/snowball-r2egym-prep.%j.log" \
            --export=ALL,MARIN_ROOT="$MARIN_ROOT",SNOWBALL_ENV="$(dirname "$(dirname "$MARIN_PYTHON")")",SNOWBALL_SCRATCH="$S",SNOWBALL_TOKENIZER="$TOK",SNOWBALL_CACHE="$CACHE",SNOWBALL_PARQUET_LIST="$S/data/r2egym_glm47_solved_v1/parquet.list" \
            "$MOE/jupiter_r2egym_prep.sbatch") || die "prep submit"
      [ -n "$j" ] || die "prep: no job id"
      wait_job "$j" "$S/logs/snowball-r2egym-prep.$j.log" "PREP_DONE" || die "prep job $j"
      touch "$S/logs/.done.prep"; say "prep OK ($j): $(grep -oE 'cache_tokens=[0-9]+ cache_examples=[0-9]+ cache_shards=[0-9]+ epoch_steps=[0-9]+' "$S/logs/snowball-r2egym-prep.$j.log" | tail -1)";;
    gate)
      if [ -f "$S/logs/.done.gate" ]; then say "gate: done already"; continue; fi
      j=$(submit bash "$MOE/gate_jupiter_snowball_env.sh") || die "gate submit"
      [ -n "$j" ] || die "gate: no job id"
      wait_job "$j" "$S/logs/snowball-env-gate.$j.log" "SNOWBALL_DISTRIBUTED_PROBE_OK" || die "gate job $j"
      touch "$S/logs/.done.gate"; say "gate OK ($j)";;
    import)
      if [ -f "$INIT/metadata.json" ]; then say "import: done already ($INIT)"; continue; fi
      j=$(submit sbatch -o "$S/logs/snowball-import.%j.log" \
            --export=ALL,MARIN_ROOT="$MARIN_ROOT",MARIN_PYTHON="$MARIN_PYTHON",SNOWBALL_HF_CHECKPOINT="$TOK",SNOWBALL_INIT="$INIT" \
            "$MOE/jupiter_snowball_import.sbatch") || die "import submit"
      [ -n "$j" ] || die "import: no job id"
      wait_job "$j" "$S/logs/snowball-import.$j.log" "" || die "import job $j"
      [ -f "$INIT/metadata.json" ] || die "import job $j left no metadata.json in $INIT"
      say "import OK ($j): $(cat "$INIT/metadata.json")";;
    run)
      if [ -n "$STEPS" ] && [ -d "$OUT/checkpoints/step-$STEPS" ]; then say "run: done already (step-$STEPS)"; continue; fi
      out=$(EPOCHS=$EPOCHS SNOWBALL_TOKENIZER="$TOK" SNOWBALL_OUTPUT="$OUT" SNOWBALL_RUN_ID="$RUN_ID" \
            SNOWBALL_CACHE="$CACHE" SNOWBALL_INIT="$INIT" bash "$MOE/launch_jupiter_snowball_r2egym.sh" 2>&1); rc=$?
      echo "$out" | tee -a "$LOG"
      [ $rc -eq 0 ] || die "launch script rc=$rc"
      STEPS=$(echo "$out" | awk '/^steps +=/{print $3}')
      j=$(echo "$out" | grep -oE 'Submitted batch job [0-9]+' | awk '{print $NF}' | tail -1)
      { [ -n "$j" ] && [ -n "$STEPS" ]; } || die "could not parse the run job id / step count from the launch output"
      wait_job "$j" "$S/logs/snowball-r2egym.$j.log" "GUARDED_RUN_EXIT rc=0" || die "run job $j (forensics under $S/logs/forensics/$j)"
      [ -d "$OUT/checkpoints/step-$STEPS" ] || die "no checkpoint step-$STEPS under $OUT"
      echo "$STEPS" > "$S/logs/.done.run"; say "run OK ($j): $OUT/checkpoints/step-$STEPS";;
    export)
      [ -n "$STEPS" ] || die "export: unknown step count (no $S/logs/.done.run)"
      EXPORT=$OUT/export-step$STEPS-hf-bf16
      if [ -f "$EXPORT/config.json" ]; then say "export: done already ($EXPORT)"; continue; fi
      j=$(submit sbatch -o "$S/logs/snowball-export.%j.log" \
            --export=ALL,MARIN_ROOT="$MARIN_ROOT",MARIN_PYTHON="$MARIN_PYTHON",SNOWBALL_EXPORT_CHECKPOINT="$OUT/checkpoints/step-$STEPS",SNOWBALL_EXPORT_OUTPUT="$EXPORT",SNOWBALL_EXPORT_TOKENIZER="$TOK" \
            "$MOE/jupiter_snowball_export.sbatch") || die "export submit"
      [ -n "$j" ] || die "export: no job id"
      wait_job "$j" "$S/logs/snowball-export.$j.log" "EXPORT_CHECK_OK" || die "export job $j"
      say "export OK ($j): $EXPORT";;
    *) die "unknown step '$step'";;
  esac
done
say "CHAIN_DONE"
