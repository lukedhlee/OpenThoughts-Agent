#!/bin/bash
# Build gsm8k train/validation parquet for SkyRL main_base (Vista).
# Prefer running on a compute node if HF download is large; login OK for small pulls.
set -euo pipefail

: "${SCRATCH:=/scratch/11584/lukedhlee}"
: "${PENFEVER_SCRATCH:=/scratch/10635/penfever}"
: "${DATA_DIR:=$SCRATCH/data/gsm8k}"
: "${SKYRL_HOME:=}"
: "${PYTHON:=$PENFEVER_SCRATCH/miniconda3/envs/otagent/bin/python}"

if [[ -z "$SKYRL_HOME" ]]; then
  for cand in "$PENFEVER_SCRATCH/MarinSkyRL" "$PENFEVER_SCRATCH/SkyRL" \
              "$SCRATCH/MarinSkyRL" "$SCRATCH/SkyRL"; do
    if [[ -d "$cand/skyrl-train" ]]; then
      SKYRL_HOME="$cand"
      break
    fi
  done
fi

if [[ -z "${SKYRL_HOME:-}" || ! -d "$SKYRL_HOME/skyrl-train" ]]; then
  echo "ERROR: MarinSkyRL/SkyRL not found. Set SKYRL_HOME=/path/to/MarinSkyRL"
  exit 1
fi

mkdir -p "$DATA_DIR"
export HF_HOME="${HF_HOME:-$SCRATCH/cache/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$HF_HUB_CACHE"

DATASET_PY=""
for p in \
  "$SKYRL_HOME/skyrl-train/examples/gsm8k/gsm8k_dataset.py" \
  "$SKYRL_HOME/examples/gsm8k/gsm8k_dataset.py"; do
  if [[ -f "$p" ]]; then
    DATASET_PY="$p"
    break
  fi
done

if [[ -z "$DATASET_PY" ]]; then
  echo "ERROR: gsm8k_dataset.py not found under $SKYRL_HOME"
  exit 1
fi

echo "Using $DATASET_PY -> $DATA_DIR"
cd "$(dirname "$DATASET_PY")"
"$PYTHON" "$(basename "$DATASET_PY")" --output_dir "$DATA_DIR" "$@"
ls -lh "$DATA_DIR"
