#!/bin/bash
# Jupiter STANDARD (non-agentic) GRPO on Qwen/Qwen3-30B-A3B + gsm8k.
# Default: develbooster smoke = 4 nodes / 16 GPU (reservation max_nodes_per_job=4).
# Vista 24-GPU parity = 6 Jupiter nodes — use RESERVATION= NUM_NODES=6 + 6node yaml.
#
# Prerequisites (login node):
#   source /e/project1/reformo/lee27/env.sh
#   set -a; source $DC_AGENT_SECRET_ENV; set +a
#   model cached under $HF_HUB_CACHE; gsm8k parquet under $DATA_DIR
#
# Usage (from OpenThoughts-Agent repo root on Jupiter):
#   bash hpc/skyrl_standard/jupiter/run_gsm8k_moe30b_grpo.sh
#   # full 24-GPU parity on open booster (no reservation):
#   RESERVATION=none NUM_NODES=6 RL_CONFIG=./hpc/skyrl_yaml/jupiter/6node_qwen3_30b_a3b_gsm8k_grpo.yaml \
#     JOB_NAME=jupiter_moe30b_gsm8k_grpo_6n TIME_LIMIT=11:59:00 bash ...
set -euo pipefail

: "${JSC_SCRATCH:=/e/scratch/reformo/lee27}"
: "${SCRATCH:=$JSC_SCRATCH}"
: "${DCFT:=$SCRATCH/OpenThoughts-Agent}"
: "${DATA_DIR:=$SCRATCH/data/gsm8k}"
: "${EXPERIMENTS_DIR:=$SCRATCH/experiments}"
: "${PARTITION:=booster}"
: "${ACCOUNT:=reformo}"
# Default develbooster. To use open booster: RESERVATION=none (empty is NOT enough —
# bash `${VAR:=default}` treats null as missing and re-applies the default).
if [[ -z "${RESERVATION+set}" ]]; then
  RESERVATION=develbooster
fi
if [[ "${RESERVATION}" == "none" || "${RESERVATION}" == "off" || "${RESERVATION}" == "-" ]]; then
  RESERVATION=
fi
: "${TIME_LIMIT:=02:00:00}"
: "${NUM_NODES:=4}"
: "${JOB_NAME:=jupiter_moe30b_gsm8k_grpo_4n}"
: "${RL_CONFIG:=./hpc/skyrl_yaml/jupiter/4node_qwen3_30b_a3b_gsm8k_grpo.yaml}"
: "${MODEL_PATH:=}"
: "${CONDA_BASE:=$SCRATCH/miniforge3}"
# Prefer the SkyRL venv from hpc/setup_rl_env.sh; fall back to conda otagent if present.
: "${RL_VENV:=$DCFT/envs/rl}"
: "${RL_CONDA_ENV:=}"
: "${RL_REPO_DIR:=$SCRATCH/MarinSkyRL}"
: "${WANDB_PROJECT:=jupiter-moe-gsm8k-grpo}"

export SCRATCH DCFT HF_HOME="${HF_HOME:-$SCRATCH/cache/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export WANDB_DIR="${WANDB_DIR:-$SCRATCH/wandb}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT
export DC_AGENT_SECRET_ENV="${DC_AGENT_SECRET_ENV:-$SCRATCH/keys/secrets.env}"
export SKYRL_HOME="${RL_REPO_DIR}"
export RL_REPO_DIR
export PYTHONPATH="${SKYRL_HOME}/skyrl-train:${SKYRL_HOME}/skyrl-gym:${DCFT}${PYTHONPATH:+:$PYTHONPATH}"
LAUNCH_WANDB_PROJECT="$WANDB_PROJECT"

_snapshot_complete() {
  local d="$1"
  [[ -d "$d" && -f "$d/config.json" ]] || return 1
  [[ -e "$d/tokenizer.json" || -e "$d/vocab.json" ]] || return 1
  compgen -G "$d/model-*.safetensors" >/dev/null || [[ -e "$d/model.safetensors" ]] || return 1
  return 0
}
if [[ -z "$MODEL_PATH" ]]; then
  for cand in "$HF_HUB_CACHE/models--Qwen--Qwen3-30B-A3B/snapshots"/*; do
    if _snapshot_complete "$cand"; then
      MODEL_PATH="$cand"
      break
    fi
  done
fi
if [[ -z "$MODEL_PATH" ]]; then
  echo "ERROR: no complete local Qwen3-30B-A3B snapshot under $HF_HUB_CACHE"
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
  echo "ERROR: no RL env. Run: cd \$DCFT && bash hpc/setup_rl_env.sh"
  echo "  expected venv: $RL_VENV"
  exit 1
fi
if [[ ! -x "$LAUNCH_PY" ]]; then
  echo "ERROR: missing $LAUNCH_PY"
  exit 1
fi

cd "$DCFT"
# Prefer lee27 dotenv (reformo). launch.py will re-load hpc/dotenv/jupiter.env —
# on this clone that file should be the lee27 rewrite (see jupiter.lee27.env).
if [[ -f "$DCFT/hpc/dotenv/jupiter.lee27.env" ]]; then
  # shellcheck disable=SC1091
  source "$DCFT/hpc/dotenv/jupiter.lee27.env"
elif [[ -f "$DCFT/hpc/dotenv/jupiter.env" ]]; then
  # shellcheck disable=SC1091
  source "$DCFT/hpc/dotenv/jupiter.env"
fi
# Force lee27 paths after any dotenv (launch.set_environment also overwrites from jupiter.env).
export SCRATCH="$JSC_SCRATCH"
export DCFT="$SCRATCH/OpenThoughts-Agent"
export DCFT_RL_ENV="${RL_VENV}"
export DC_AGENT_SECRET_ENV="$SCRATCH/keys/secrets.env"
export HF_HOME="$SCRATCH/cache/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export WANDB_MODE=offline
export WANDB_PROJECT="$LAUNCH_WANDB_PROJECT"
export WANDB_DIR="$SCRATCH/wandb"
export SKYRL_HOME="$RL_REPO_DIR"
export RL_REPO_DIR
export EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-$SCRATCH/experiments}"
export OT_AGENT_RAY_LOG_DIR="${OT_AGENT_RAY_LOG_DIR:-$SCRATCH/experiments/_ray_logs}"
export IMAGE=
# vLLM wheels need nvidia-* pip libs on LD_LIBRARY_PATH (CUDA 12 ABI) even with CUDA/13 modules.
_NV_LIBS=""
if [[ -d "${RL_VENV}/lib" ]]; then
  while IFS= read -r _d; do
    _NV_LIBS="${_NV_LIBS:+$_NV_LIBS:}${_d}"
  done < <(find "${RL_VENV}/lib" -type d \( -path '*/nvidia/*/lib' -o -path '*/torch/lib' \) 2>/dev/null | sort -u)
fi
if [[ -n "$_NV_LIBS" ]]; then
  export LD_LIBRARY_PATH="${_NV_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
mkdir -p "$EXPERIMENTS_DIR" "$WANDB_DIR" "$OT_AGENT_RAY_LOG_DIR"

EXTRA_LAUNCH_ARGS=()
if [[ -n "${RESERVATION:-}" ]]; then
  EXTRA_LAUNCH_ARGS+=(--reservation "$RESERVATION")
fi
if [[ -n "${ACCOUNT:-}" ]]; then
  EXTRA_LAUNCH_ARGS+=(--account "$ACCOUNT")
fi

echo "WandB: mode=$WANDB_MODE project=$WANDB_PROJECT entity=${WANDB_ENTITY:-unset} dir=$WANDB_DIR"
echo "Launching $JOB_NAME on $PARTITION ($NUM_NODES nodes, wall $TIME_LIMIT, account=$ACCOUNT reservation=${RESERVATION:-none})"
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
if [[ -n "${SAVE_OPTIMIZER_STATES+x}" ]]; then
  SKYRL_OVERRIDES+=(--skyrl_override ++trainer.save_optimizer_states="$SAVE_OPTIMIZER_STATES")
fi
if [[ -n "${LOAD_OPTIMIZER_STATES+x}" ]]; then
  SKYRL_OVERRIDES+=(--skyrl_override ++trainer.load_optimizer_states="$LOAD_OPTIMIZER_STATES")
fi
if [[ -n "${SAVE_LR_SCHEDULER_STATES+x}" ]]; then
  SKYRL_OVERRIDES+=(--skyrl_override ++trainer.save_lr_scheduler_states="$SAVE_LR_SCHEDULER_STATES")
fi
if [[ -n "${LOAD_LR_SCHEDULER_STATES+x}" ]]; then
  SKYRL_OVERRIDES+=(--skyrl_override ++trainer.load_lr_scheduler_states="$LOAD_LR_SCHEDULER_STATES")
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
  SKYRL_OVERRIDES+=(--skyrl_override generator.sampling_params.max_generate_length="$MAX_GENERATE_LENGTH")
fi
if [[ -n "${MAX_MODEL_LEN+x}" ]]; then
  SKYRL_OVERRIDES+=(--skyrl_override generator.engine_init_kwargs.max_model_len="$MAX_MODEL_LEN")
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
