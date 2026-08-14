#!/bin/bash
# Build gsm8k train/validation parquet for SkyRL main_base (Jupiter login OK).
set -euo pipefail

: "${JSC_SCRATCH:=/e/scratch/reformo/lee27}"
: "${SCRATCH:=$JSC_SCRATCH}"
: "${DATA_DIR:=$SCRATCH/data/gsm8k}"
: "${SKYRL_HOME:=$SCRATCH/MarinSkyRL}"
: "${PYTHON:=$SCRATCH/miniforge3/envs/otagent/bin/python}"
: "${HF_HOME:=$SCRATCH/cache/hf}"
: "${HF_HUB_CACHE:=$HF_HOME/hub}"

if [[ ! -x "$PYTHON" ]]; then
  # fall back to base miniforge if otagent not ready yet
  PYTHON="$SCRATCH/miniforge3/bin/python"
fi
if [[ ! -d "$SKYRL_HOME/skyrl-train" ]]; then
  echo "ERROR: MarinSkyRL not found at $SKYRL_HOME"
  exit 1
fi

mkdir -p "$DATA_DIR" "$HF_HUB_CACHE"
export HF_HOME HF_HUB_CACHE

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
# datasets + pyarrow needed
"$PYTHON" - <<'PY'
import importlib.util, sys
need = ["datasets", "pyarrow", "pandas"]
missing = [m for m in need if importlib.util.find_spec(m) is None]
if missing:
    print("Missing packages:", ", ".join(missing), file=sys.stderr)
    print("Install with: pip install datasets pyarrow pandas", file=sys.stderr)
    sys.exit(1)
PY

cd "$(dirname "$DATASET_PY")"
"$PYTHON" "$(basename "$DATASET_PY")" --output_dir "$DATA_DIR" "$@"
ls -lh "$DATA_DIR"
