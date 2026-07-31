#!/usr/bin/env bash
set -euo pipefail

: "${JSC_SCRATCH:=/e/scratch/reformo/lee27}"
: "${DCFT:=$JSC_SCRATCH/OpenThoughts-Agent}"
: "${RL_VENV:=$DCFT/envs/rl}"
: "${RL_REPO_DIR:=$JSC_SCRATCH/MarinSkyRL}"
: "${EXPERIMENTS_DIR:=$JSC_SCRATCH/experiments}"
: "${PARTITION:=booster}"
: "${ACCOUNT:=reformo}"
: "${TIME_LIMIT:=11:59:00}"
: "${NUM_NODES:=6}"
: "${MAX_RESTARTS:=2}"
: "${MAX_STEPS:=50}"
: "${CKPT_INTERVAL:=5}"
: "${HF_SAVE_INTERVAL:=5}"
: "${JOB_NAME:=jupiter_qwen3_30b_a3b_r2egym_grpo}"
: "${HF_HUB_REPO_ID:=lukedhlee/$JOB_NAME}"
: "${HF_HUB_PRIVATE:=false}"
: "${RL_CONFIG:=$DCFT/hpc/skyrl_yaml/jupiter/6node_qwen3_30b_a3b_r2egym_grpo.yaml}"
: "${TRAIN_DATA:=DCAgent/r2egym-patched-full-oracle}"

export SCRATCH="$JSC_SCRATCH"
export DCFT
export DCFT_RL_ENV="$RL_VENV"
export DC_AGENT_SECRET_ENV="${DC_AGENT_SECRET_ENV:-$JSC_SCRATCH/keys/secrets.env}"
export HF_HOME="${HF_HOME:-$JSC_SCRATCH/cache/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export WANDB_DIR="${WANDB_DIR:-$JSC_SCRATCH/wandb}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-jupiter-r2egym-grpo}"
export RL_REPO_DIR
export SKYRL_HOME="$RL_REPO_DIR"
export PYTHONPATH="$SKYRL_HOME/skyrl-train:$SKYRL_HOME/skyrl-gym:$DCFT${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -x "$RL_VENV/bin/python" ]]; then
  echo "ERROR: missing RL runtime at $RL_VENV/bin/python" >&2
  exit 1
fi
if [[ ! -f "$DC_AGENT_SECRET_ENV" ]]; then
  echo "ERROR: missing secret environment file at $DC_AGENT_SECRET_ENV" >&2
  exit 1
fi
if [[ ! -f "$RL_CONFIG" ]]; then
  echo "ERROR: missing RL config at $RL_CONFIG" >&2
  exit 1
fi

set -a
source "$DC_AGENT_SECRET_ENV"
set +a

"$RL_VENV/bin/python" - <<'PY'
from huggingface_hub import HfApi

HfApi().whoami()
PY

MODEL_PATH=""
for candidate in "$HF_HUB_CACHE/models--Qwen--Qwen3-30B-A3B/snapshots"/*; do
  if [[ -f "$candidate/config.json" ]] && { compgen -G "$candidate/model-*.safetensors" >/dev/null || [[ -f "$candidate/model.safetensors" ]]; }; then
    MODEL_PATH="$candidate"
    break
  fi
done
if [[ -z "$MODEL_PATH" ]]; then
  echo "ERROR: Qwen3-30B-A3B is not cached under $HF_HUB_CACHE" >&2
  exit 1
fi

mkdir -p "$EXPERIMENTS_DIR" "$WANDB_DIR"
cd "$DCFT"

"$RL_VENV/bin/python" -m hpc.launch \
  --job_type rl \
  --rl_config "$RL_CONFIG" \
  --model_path "$MODEL_PATH" \
  --train_data "[\"$TRAIN_DATA\"]" \
  --num_nodes "$NUM_NODES" \
  --partition "$PARTITION" \
  --account "$ACCOUNT" \
  --time_limit "$TIME_LIMIT" \
  --max_restarts "$MAX_RESTARTS" \
  --experiments_dir "$EXPERIMENTS_DIR" \
  --job_name "$JOB_NAME" \
  --hf_hub_repo_id "$HF_HUB_REPO_ID" \
  --hf_hub_private "$HF_HUB_PRIVATE" \
  --skyrl_override trainer.max_steps="$MAX_STEPS" \
  --skyrl_override trainer.ckpt_interval="$CKPT_INTERVAL" \
  --skyrl_override trainer.hf_save_interval="$HF_SAVE_INTERVAL"
