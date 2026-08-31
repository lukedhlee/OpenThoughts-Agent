#!/bin/bash
# Build gsm8k train/validation parquet for SkyRL main_base (run on a Jupiter
# login node — it downloads from the HF Hub). Locations come from
# hpc/dotenv/jupiter.env (SCRATCH, DATASETS_DIR, SKYRL_HOME, HF_*).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
: "${DCFT:=$REPO_ROOT}"
if [[ -z "${SCRATCH:-}" && -f "$DCFT/hpc/dotenv/jupiter.env" ]]; then
  # shellcheck disable=SC1091
  source "$DCFT/hpc/dotenv/jupiter.env"
fi
: "${SCRATCH:?SCRATCH unset — source hpc/dotenv/jupiter.env first}"
: "${DATA_DIR:=${DATASETS_DIR:-$SCRATCH/data}/gsm8k}"
: "${SKYRL_HOME:?SKYRL_HOME unset — source hpc/dotenv/jupiter.env first}"
# Any python with datasets + pyarrow + pandas (the RL venv qualifies).
: "${PYTHON:=${DCFT_RL_ENV:+$DCFT_RL_ENV/bin/python}}"
: "${PYTHON:=$(command -v python3)}"
: "${HF_HOME:=$SCRATCH/cache/hf}"
: "${HF_HUB_CACHE:=$HF_HOME/hub}"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: PYTHON=$PYTHON is not executable"
  exit 1
fi
if [[ ! -d "$SKYRL_HOME/skyrl-train" ]]; then
  echo "ERROR: SkyRL checkout not found at $SKYRL_HOME"
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
