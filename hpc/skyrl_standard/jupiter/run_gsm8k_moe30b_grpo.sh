#!/bin/bash
# Jupiter STANDARD (non-agentic) GRPO on Qwen3-30B-A3B(-Base) + gsm8k.
# Default: 6 nodes / 24 GPU, FSDP2 EP=4, flash_attn (the *_fsdp2_arms_fa yaml).
#
# Prerequisites (login node):
#   cd <OpenThoughts-Agent> && source hpc/dotenv/jupiter.env
#   set -a; source $DC_AGENT_SECRET_ENV; set +a       # WANDB_API_KEY
#   model snapshot cached under $HF_HUB_CACHE (compute has no internet)
#   gsm8k parquet under $DATA_DIR (bash hpc/skyrl_standard/jupiter/prep_gsm8k_parquet.sh)
#
# Usage (from the repo root on Jupiter):
#   bash hpc/skyrl_standard/jupiter/run_gsm8k_moe30b_grpo.sh
#   POLICY_LR=1.0e-6 JOB_NAME=base30b_gsm8k_lr1e6 bash hpc/skyrl_standard/jupiter/run_gsm8k_moe30b_grpo.sh
#   RESERVATION=develbooster NUM_NODES=4 TIME_LIMIT=02:00:00 bash ...   # reservation smoke
#
# Every location below is taken from hpc/dotenv/jupiter.env (or the caller's
# environment); nothing user-specific is hard-coded here.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
: "${DCFT:=$REPO_ROOT}"
if [[ -z "${SCRATCH:-}" && -f "$DCFT/hpc/dotenv/jupiter.env" ]]; then
  # shellcheck disable=SC1091
  source "$DCFT/hpc/dotenv/jupiter.env"
  if [[ -f "$DCFT/hpc/dotenv/jupiter.local.env" ]]; then
    # shellcheck disable=SC1091
    source "$DCFT/hpc/dotenv/jupiter.local.env"
  fi
fi
: "${SCRATCH:?SCRATCH unset — source hpc/dotenv/jupiter.env first}"
: "${DATA_DIR:=${DATASETS_DIR:-$SCRATCH/data}/gsm8k}"
: "${EXPERIMENTS_DIR:=$DCFT/experiments}"
: "${PARTITION:=booster}"
: "${ACCOUNT:=${DCFT_GROUP:-}}"
# No reservation by default. RESERVATION=develbooster for the 4-node smoke queue.
: "${RESERVATION:=}"
if [[ "${RESERVATION}" == "none" || "${RESERVATION}" == "off" || "${RESERVATION}" == "-" ]]; then
  RESERVATION=
fi
: "${TIME_LIMIT:=11:59:00}"
: "${NUM_NODES:=6}"
: "${JOB_NAME:=jupiter_moe30b_gsm8k_grpo_6n}"
: "${RL_CONFIG:=./hpc/skyrl_yaml/jupiter/6node_qwen3_30b_a3b_base_gsm8k_grpo_fsdp2_arms_fa.yaml}"
: "${MODEL_ID:=Qwen/Qwen3-30B-A3B-Base}"
: "${MODEL_PATH:=}"
# RL runtime: DCFT_RL_ENV (host venv, see hpc/env_builds/jupiter/build_rl_fa_env.sh)
# > $DCFT/envs/rl (hpc/setup_rl_env.sh). RL_CONDA_ENV=<name|/abs/path> uses conda instead.
: "${RL_VENV:=${DCFT_RL_ENV:-$DCFT/envs/rl}}"
: "${RL_CONDA_ENV:=}"
: "${CONDA_BASE:=${CONDA_PREFIX:-$SCRATCH/miniforge3}}"
: "${WANDB_PROJECT:=jupiter-moe-gsm8k-grpo}"

export DCFT HF_HOME="${HF_HOME:-$SCRATCH/cache/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT
export DC_AGENT_SECRET_ENV="${DC_AGENT_SECRET_ENV:-$HOME/secrets.env}"
if [[ -n "${SKYRL_HOME:-}" ]]; then
  export RL_REPO_DIR="${RL_REPO_DIR:-$SKYRL_HOME}"
  export PYTHONPATH="${SKYRL_HOME}/skyrl-train:${SKYRL_HOME}/skyrl-gym:${DCFT}${PYTHONPATH:+:$PYTHONPATH}"
fi
LAUNCH_WANDB_PROJECT="$WANDB_PROJECT"

_snapshot_complete() {
  local d="$1"
  [[ -d "$d" && -f "$d/config.json" ]] || return 1
  [[ -e "$d/tokenizer.json" || -e "$d/vocab.json" ]] || return 1
  compgen -G "$d/model-*.safetensors" >/dev/null || [[ -e "$d/model.safetensors" ]] || return 1
  return 0
}
if [[ -z "$MODEL_PATH" ]]; then
  for cand in "$HF_HUB_CACHE/models--${MODEL_ID//\//--}/snapshots"/*; do
    if _snapshot_complete "$cand"; then
      MODEL_PATH="$cand"
      break
    fi
  done
fi
if [[ -z "$MODEL_PATH" ]]; then
  echo "ERROR: no complete local $MODEL_ID snapshot under $HF_HUB_CACHE"
  echo "  Download on a login node first (compute has no internet)."
  exit 1
fi

if [[ -f "$DC_AGENT_SECRET_ENV" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$DC_AGENT_SECRET_ENV"
  set +a
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: WANDB_API_KEY unset. Expected $DC_AGENT_SECRET_ENV"
  exit 1
fi

if [[ ! -f "$DATA_DIR/train.parquet" || ! -f "$DATA_DIR/validation.parquet" ]]; then
  echo "ERROR: gsm8k parquet missing under $DATA_DIR"
  echo "Run: bash hpc/skyrl_standard/jupiter/prep_gsm8k_parquet.sh"
  exit 1
fi

USE_CONDA=0
if [[ -n "${RL_CONDA_ENV}" ]]; then
  if [[ "$RL_CONDA_ENV" == /* ]]; then
    LAUNCH_PY="${RL_CONDA_ENV}/bin/python"
  else
    LAUNCH_PY="${CONDA_BASE}/envs/${RL_CONDA_ENV}/bin/python"
  fi
  USE_CONDA=1
elif [[ -x "${RL_VENV}/bin/python" ]]; then
  LAUNCH_PY="${RL_VENV}/bin/python"
else
  echo "ERROR: no RL env. Run: cd \$DCFT && bash hpc/setup_rl_env.sh (or set DCFT_RL_ENV)"
  echo "  expected venv: $RL_VENV"
  exit 1
fi
if [[ ! -x "$LAUNCH_PY" ]]; then
  echo "ERROR: missing $LAUNCH_PY"
  exit 1
fi

cd "$DCFT"
# launch.py re-loads hpc/dotenv/jupiter.env; keep the job-side values aligned.
export DCFT_RL_ENV="${RL_VENV}"
export WANDB_PROJECT="$LAUNCH_WANDB_PROJECT"
export EXPERIMENTS_DIR
export OT_AGENT_RAY_LOG_DIR="${OT_AGENT_RAY_LOG_DIR:-$EXPERIMENTS_DIR/_ray_logs}"
# vLLM wheels need the pip nvidia-* libs on LD_LIBRARY_PATH (CUDA 12 ABI) even
# with the CUDA/13 modules loaded — build the list from the venv itself.
_NV_LIBS=""
if [[ -d "${RL_VENV}/lib" ]]; then
  while IFS= read -r _d; do
    _NV_LIBS="${_NV_LIBS:+$_NV_LIBS:}${_d}"
  done < <(find "${RL_VENV}/lib" -type d \( -path '*/nvidia/*/lib' -o -path '*/torch/lib' \) 2>/dev/null | sort -u)
fi
if [[ -n "$_NV_LIBS" ]]; then
  export LD_LIBRARY_PATH="${_NV_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
mkdir -p "$EXPERIMENTS_DIR" "$OT_AGENT_RAY_LOG_DIR"

EXTRA_LAUNCH_ARGS=()
if [[ -n "${RESERVATION:-}" ]]; then
  EXTRA_LAUNCH_ARGS+=(--reservation "$RESERVATION")
fi
if [[ -n "${ACCOUNT:-}" ]]; then
  EXTRA_LAUNCH_ARGS+=(--account "$ACCOUNT")
fi

echo "WandB: mode=$WANDB_MODE project=$WANDB_PROJECT entity=${WANDB_ENTITY:-unset} dir=${WANDB_DIR:-<launcher default>}"
echo "Launching $JOB_NAME on $PARTITION ($NUM_NODES nodes, wall $TIME_LIMIT, account=${ACCOUNT:-default} reservation=${RESERVATION:-none})"
echo "  model=$MODEL_PATH"
echo "  data=$DATA_DIR"
echo "  python=$LAUNCH_PY"

LAUNCH_ARGS=(
  --job_type rl
  --rl_config "$RL_CONFIG"
  --model_path "$MODEL_PATH"
)
if [[ "$USE_CONDA" -eq 1 ]]; then
  LAUNCH_ARGS+=(--rl_use_conda --rl_conda_env "$RL_CONDA_ENV")
fi

SKYRL_OVERRIDES=(
  --skyrl_override environment.env_class=gsm8k
  --skyrl_override trainer.enable_db_registration=false
  --skyrl_override trainer.logger=wandb
  --skyrl_override trainer.project_name="$WANDB_PROJECT"
  --skyrl_override trainer.run_name="$JOB_NAME"
  --skyrl_override trainer.max_steps="${MAX_STEPS:-50}"
  --skyrl_override trainer.eval_interval="${EVAL_INTERVAL:-5}"
  --skyrl_override trainer.eval_before_train="${EVAL_BEFORE_TRAIN:-true}"
  --skyrl_override trainer.ckpt_interval="${CKPT_INTERVAL:-5}"
)
if [[ -n "${HF_SAVE_INTERVAL+x}" ]]; then
  SKYRL_OVERRIDES+=(--skyrl_override trainer.hf_save_interval="$HF_SAVE_INTERVAL")
fi
if [[ -n "${RESUME_MODE+x}" ]]; then
  SKYRL_OVERRIDES+=(--skyrl_override trainer.resume_mode="$RESUME_MODE")
fi
if [[ -n "${RESUME_PATH+x}" ]]; then
  SKYRL_OVERRIDES+=(--skyrl_override trainer.resume_path="$RESUME_PATH")
fi
if [[ -n "${TRAIN_BATCH_SIZE+x}" ]]; then
  SKYRL_OVERRIDES+=(--skyrl_override trainer.train_batch_size="$TRAIN_BATCH_SIZE")
fi
if [[ -n "${POLICY_MINI_BATCH_SIZE+x}" ]]; then
  SKYRL_OVERRIDES+=(--skyrl_override trainer.policy_mini_batch_size="$POLICY_MINI_BATCH_SIZE")
fi
if [[ -n "${EVAL_BATCH_SIZE+x}" ]]; then
  SKYRL_OVERRIDES+=(--skyrl_override trainer.eval_batch_size="$EVAL_BATCH_SIZE")
fi
if [[ -n "${POLICY_LR+x}" ]]; then
  SKYRL_OVERRIDES+=(--skyrl_override trainer.policy.optimizer_config.lr="$POLICY_LR")
fi
if [[ -n "${MAX_GENERATE_LENGTH+x}" ]]; then
  SKYRL_OVERRIDES+=(--skyrl_override context_budget.max_new_tokens_per_turn="$MAX_GENERATE_LENGTH")
fi
if [[ -n "${MAX_MODEL_LEN+x}" ]]; then
  SKYRL_OVERRIDES+=(--skyrl_override context_budget.request_window_tokens="$MAX_MODEL_LEN")
fi
if [[ -n "${USE_KL_LOSS+x}" ]]; then
  SKYRL_OVERRIDES+=(--skyrl_override trainer.algorithm.use_kl_loss="$USE_KL_LOSS")
fi
if [[ -n "${GENERATOR_NUM_INFERENCE_ENGINES+x}" ]]; then
  SKYRL_OVERRIDES+=(--skyrl_override generator.num_inference_engines="$GENERATOR_NUM_INFERENCE_ENGINES")
fi
if [[ -n "${GENERATOR_INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE+x}" ]]; then
  SKYRL_OVERRIDES+=(--skyrl_override generator.inference_engine_tensor_parallel_size="$GENERATOR_INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE")
fi

"$LAUNCH_PY" -m hpc.launch \
  "${LAUNCH_ARGS[@]}" \
  --train_data "[\"${DATA_DIR}/train.parquet\"]" \
  --val_data "[\"${DATA_DIR}/validation.parquet\"]" \
  --num_nodes "$NUM_NODES" \
  --partition "$PARTITION" \
  --time_limit "$TIME_LIMIT" \
  --experiments_dir "$EXPERIMENTS_DIR" \
  --job_name "$JOB_NAME" \
  "${SKYRL_OVERRIDES[@]}" \
  "${EXTRA_LAUNCH_ARGS[@]}"
