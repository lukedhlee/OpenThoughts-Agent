#!/usr/bin/env bash
# Qwen3-8B (dense) FSDP2 + GRPO on r2egym, Jupiter/booster, disaggregated.
#
# Geometry: 3 nodes = 1 policy node (4x GH200, FSDP2 fsdp_size=4)
#                   + 2 inference nodes (8 vLLM engines, TP=1).
# Sandboxes: Daytona (r2egym is 7 unique snapshots for all 3,328 tasks).
# Validation: OFFLINE on saved checkpoints -- SWE-bench-100 through the JURECA
#             apptainer bridge. Nothing in-loop; eval_interval is pinned off.
#
# NOTE the env: envs/rl is BROKEN on Jupiter (its vllm _C.so needs
# libcudart.so.12, Jupiter only ships CUDA/13). envs/rl-megatron is the only
# runnable vLLM; "megatron" is just the directory name -- strategy is fsdp2.
# RL_ENV_DIR is forced via container.extra_env in the YAML, because
# rl_launch_utils.py:780 would otherwise -d-test its way into the broken env.
set -euo pipefail

: "${JSC_SCRATCH:=/e/scratch/reformo/lee27}"
: "${DCFT:=$JSC_SCRATCH/OpenThoughts-Agent}"
: "${RL_VENV:=$DCFT/envs/rl-megatron}"
: "${RL_REPO_DIR:=$JSC_SCRATCH/MarinSkyRL}"
: "${EXPERIMENTS_DIR:=$JSC_SCRATCH/experiments}"
: "${PARTITION:=booster}"
: "${ACCOUNT:=reformo}"
: "${TIME_LIMIT:=11:59:00}"
: "${NUM_NODES:=3}"
: "${MAX_RESTARTS:=2}"
: "${MAX_STEPS:=50}"
: "${CKPT_INTERVAL:=5}"
: "${HF_SAVE_INTERVAL:=5}"
: "${JOB_NAME:=jupiter_qwen3_8b_r2egym_grpo}"
# Trainer HF auto-push is OFF by default: Jupiter's secrets.env carries no HF
# token (only DAYTONA/WANDB/SUPABASE), and the auto-pushed lukedhlee/<job> repo
# has the wrong nested layout anyway. Publishing goes through
# rl-agentic-job-cleanup -> laion/<job>-<step>-<size>. hf_save_interval still
# writes local HF exports, which is what the offline SWE-bench-100 val consumes.
: "${HF_HUB_REPO_ID:=}"
: "${HF_HUB_PRIVATE:=false}"
: "${RL_CONFIG:=$DCFT/hpc/skyrl_yaml/jupiter/3node_qwen3_8b_r2egym_grpo.yaml}"
: "${TRAIN_DATA_REPO:=DCAgent/r2egym-patched-full-oracle}"

export SCRATCH="$JSC_SCRATCH"
export DCFT
export DCFT_RL_ENV="$RL_VENV"
export RL_ENV_DIR="$RL_VENV"
export DC_AGENT_SECRET_ENV="${DC_AGENT_SECRET_ENV:-$JSC_SCRATCH/keys/secrets.env}"
export HF_HOME="${HF_HOME:-$JSC_SCRATCH/cache/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export WANDB_DIR="${WANDB_DIR:-$JSC_SCRATCH/wandb}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-jupiter-r2egym-grpo-8b}"
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

# Only assert HF credentials when we actually intend to push.
if [[ -n "$HF_HUB_REPO_ID" ]]; then
  "$RL_VENV/bin/python" - <<'PY'
from huggingface_hub import HfApi

HfApi().whoami()
PY
else
  echo "HF auto-push disabled (HF_HUB_REPO_ID empty); skipping whoami preflight."
fi

MODEL_PATH=""
for candidate in "$HF_HUB_CACHE/models--Qwen--Qwen3-8B/snapshots"/*; do
  if [[ -f "$candidate/config.json" ]] && { compgen -G "$candidate/model-*.safetensors" >/dev/null || [[ -f "$candidate/model.safetensors" ]]; }; then
    MODEL_PATH="$candidate"
    break
  fi
done
if [[ -z "$MODEL_PATH" ]]; then
  echo "ERROR: Qwen3-8B is not cached under $HF_HUB_CACHE" >&2
  exit 1
fi

# Resolve the r2egym parquet to a LOCAL path. scripts/datagen/extract_tasks_from_parquet.py
# uses a local path directly, but falls back to importing data.commons for anything
# it can't resolve locally -- and data.commons pulls rapidfuzz, which this env lacks.
# Passing the cached snapshot dir keeps extraction off that branch entirely (and
# off the network). Falls back to the repo id if the cache is somehow absent.
TRAIN_DATA="${TRAIN_DATA:-}"
if [[ -z "$TRAIN_DATA" ]]; then
  for candidate in "$HF_HUB_CACHE/datasets--DCAgent--r2egym-patched-full-oracle/snapshots"/*; do
    if [[ -d "$candidate" ]] && compgen -G "$candidate/**/*.parquet" >/dev/null 2>&1 || \
       { [[ -d "$candidate" ]] && [[ -n "$(find "$candidate" -maxdepth 2 -name '*.parquet' -print -quit 2>/dev/null)" ]]; }; then
      TRAIN_DATA="$candidate"
      break
    fi
  done
fi
if [[ -z "$TRAIN_DATA" ]]; then
  echo "WARNING: r2egym parquet not found in $HF_HUB_CACHE; falling back to repo id" >&2
  TRAIN_DATA="$TRAIN_DATA_REPO"
else
  echo "Using local r2egym parquet: $TRAIN_DATA"
fi

mkdir -p "$EXPERIMENTS_DIR" "$WANDB_DIR"
cd "$DCFT"

HF_ARGS=()
if [[ -n "$HF_HUB_REPO_ID" ]]; then
  HF_ARGS=(--hf_hub_repo_id "$HF_HUB_REPO_ID" --hf_hub_private "$HF_HUB_PRIVATE")
fi

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
  ${HF_ARGS[@]+"${HF_ARGS[@]}"} \
  --skyrl_override trainer.max_steps="$MAX_STEPS" \
  --skyrl_override trainer.ckpt_interval="$CKPT_INTERVAL" \
  --skyrl_override trainer.hf_save_interval="$HF_SAVE_INTERVAL"
