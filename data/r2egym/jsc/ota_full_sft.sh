#!/bin/bash
# ota_full_sft.sh — one SFT arm on the OpenThoughts-Agent SFT-100K traces, with the dose curve read out of
# that single run: a permanent checkpoint per packed epoch (SNOWBALL_KEEP_PER_EPOCH), an export of each, and
# the held-out NLL of each against the base. A standard-versus-TailSFT pair would cost twice this and answer
# a question the Kimi run already answered (fit is a wash); the open question here is dose.
#
# Data (staged and checksummed 2026-09-19): $S/data/ota_sft_100k_v1, built by
# OpenThoughts-Agent data/swesmith/ota_sft_convert.py from the 2026-09-18 audit manifest.
#   SCOPE=all  (default) four slices, 50,053 train rows / 617.9M tokens -> ~304 packed steps per epoch
#   SCOPE=swe3            drops IssueTasks, 38,695 rows / 477.3M tokens -> ~235 steps per epoch
# Held-out is the same 2,728-row four-slice file either way, so SCOPE=swe3 still reads IssueTasks transfer.
#
# Cost, stated before starting (standing rule), SCOPE=all at EPOCHS=3: prep 1 node ~15 min (0.3 node-h),
# run 16 nodes ~1 h 45 (28 node-h), three exports 4 nodes x ~15 min (3 node-h), four scoring jobs
# (base + three epochs) 1 node x ~15 min (1 node-h) -> about 32 node-hours, ~2 h 45 wall.
# SCOPE=swe3: about 26 node-hours, ~2 h 20 wall. Nothing is submitted until this script is started.
#
# The base reference for the curve is independent of the run and can go first (1 node, ~15 min, 0.25 node-h):
#   sbatch --export=ALL,MODEL=/e/fscratch/reformo/lee27/models/snowball-s3-nemotron-terminal-step1888,\
#     PARQUET=$S/data/ota_sft_100k_v1/ota-all-heldout-00000-of-00001.parquet,NAME=ota-base \
#     /e/project1/transfernetx/lee27/code/snowball/heldout_nll.sbatch
# An arm counts if held-out NLL falls below that base with think-span NLL not above the base's think value,
# and the dose to keep is the earliest epoch within 0.01 of the best one.
set -uo pipefail
S=/e/data1/mmlaion/lee27/snowball-sft
D=$S/data/ota_sft_100k_v1
SCOPE=${SCOPE:-all}
EXP=$S/experiments/snowball-ota-sft
export SNOWBALL_STAGE=ota
export SNOWBALL_PARQUET_LIST=$([ "$SCOPE" = all ] && echo "$D/parquet.list" || echo "$D/parquet.swe3.list")
export SNOWBALL_DATASET_ID=open-thoughts/OpenThoughts-Agent-SFT-100K
export SNOWBALL_DATASET_REVISION=45fb28fcc38d352133cb28a1c8a43a2f14fea97b
export SNOWBALL_EXP=$EXP
export SNOWBALL_CACHE=$EXP/cache-$SCOPE-v1
export SNOWBALL_OUTPUT=$EXP/$SCOPE-lr${SNOWBALL_LR:-2e-5}-ep${EPOCHS:-3}
export SNOWBALL_RUN_ID=snowball-ota-$SCOPE-lr${SNOWBALL_LR:-2e-5}-ep${EPOCHS:-3}
# 2e-5, not the Kimi pick of 5e-5: that came from a 96-step run, and this one is ~900 steps at the same
# batch, so the same lr is ~10x the dose. The per-epoch curve is what says whether 2e-5 was too low.
export SNOWBALL_LR=${SNOWBALL_LR:-2e-5}
export EPOCHS=${EPOCHS:-3}
export SNOWBALL_KEEP_PER_EPOCH=1
export SNOWBALL_WALL=${SNOWBALL_WALL:-03:00:00}
export CHAIN_STEPS=${CHAIN_STEPS:-"prep run"}   # gate + import are done (markers under $S/logs); export is per epoch below
export SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-laionize}
C=/e/project1/transfernetx/lee27/code/snowball
MOE=/e/project1/transfernetx/lee27/code/marin-sft/experiments/june_tpu_67b_a2b/moe
TOK=/e/fscratch/reformo/lee27/models/snowball-s3-nemotron-terminal-step1888
say() { echo "[$(date -u +%FT%TZ)] $*"; }

bash -l $C/snowball_sft_chain.sh || { say "chain failed; no exports submitted"; exit 1; }
STEPS=$(cat "$S/logs/ota/.done.run")
EPOCH_STEPS=$((STEPS / EPOCHS))
say "run done at step $STEPS, epoch every $EPOCH_STEPS steps"

# One export + one scoring job per epoch checkpoint, each scoring job gated on its own export.
for e in $(seq 1 "$EPOCHS"); do
  st=$((e * EPOCH_STEPS)); [ "$e" = "$EPOCHS" ] && st=$STEPS
  CK=$SNOWBALL_OUTPUT/checkpoints/step-$st
  EX=$SNOWBALL_OUTPUT/export-step$st-hf-bf16
  [ -d "$CK" ] || { say "no checkpoint $CK (keep interval wrong?); skipping epoch $e"; continue; }
  if [ -f "$EX/config.json" ]; then jx=""; say "export for epoch $e exists"; else
    jx=$(sbatch --parsable -o "$S/logs/snowball-export.%j.log" \
      --export=ALL,MARIN_ROOT=/e/project1/transfernetx/lee27/code/marin-sft,MARIN_PYTHON=/e/project1/transfernetx/lee27/code/envs/marin-grug-sft/bin/python,SNOWBALL_EXPORT_CHECKPOINT="$CK",SNOWBALL_EXPORT_OUTPUT="$EX",SNOWBALL_EXPORT_TOKENIZER="$TOK" \
      "$MOE/jupiter_snowball_export.sbatch") || { say "export submit failed for epoch $e"; continue; }
    say "epoch $e export job $jx -> $EX"
  fi
  js=$(sbatch --parsable ${jx:+--dependency=afterok:$jx} \
    --export=ALL,MODEL="$EX",PARQUET="$D/ota-all-heldout-00000-of-00001.parquet",NAME="ota-$SCOPE-ep$e" \
    $C/heldout_nll.sbatch) \
    && say "epoch $e scoring job $js -> $S/logs/heldout_nll_ota-$SCOPE-ep$e.json"
done
say "ALL_SUBMITTED; read the curve with: grep -h heldout_nll $S/logs/heldout_nll_ota-$SCOPE-ep*.json"
