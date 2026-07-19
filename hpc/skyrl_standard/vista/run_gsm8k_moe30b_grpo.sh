#!/bin/bash
# Vista STANDARD (non-agentic) GRPO on Qwen/Qwen3-30B-A3B + gsm8k.
# 8 GH200 nodes, colocate EP=4 x FSDP=2, max_steps=10, verifier reward only.
#
# Prerequisites (login node OK for git/light; prep data may download):
#   1. Branch lukedhlee/vista-moe-grpo-30b pulled into $DCFT
#   2. gsm8k parquet at $DATA_DIR/{train,validation}.parquet
#   3. Model weights cached (or let compute download with HF_TOKEN)
#   4. secrets: source $SCRATCH/keys.env or penfever keys.env for WANDB/HF
#
# Usage (from OpenThoughts-Agent repo root on Vista):
#   bash hpc/skyrl_standard/vista/run_gsm8k_moe30b_grpo.sh
#   PARTITION=gh TIME_LIMIT=06:00:00 bash hpc/skyrl_standard/vista/run_gsm8k_moe30b_grpo.sh
set -euo pipefail

: "${SCRATCH:=/scratch/11584/lukedhlee}"
: "${DCFT:=$SCRATCH/OpenThoughts-Agent}"
: "${PENFEVER_SCRATCH:=/scratch/10635/penfever}"
: "${DATA_DIR:=$SCRATCH/data/gsm8k}"
: "${EXPERIMENTS_DIR:=$SCRATCH/experiments}"
: "${PARTITION:=gh-dev}"
: "${TIME_LIMIT:=02:00:00}"
: "${NUM_NODES:=8}"
: "${JOB_NAME:=vista_moe30b_gsm8k_grpo10}"
: "${MODEL_PATH:=Qwen/Qwen3-30B-A3B}"
: "${CONDA_BASE:=$PENFEVER_SCRATCH/miniconda3}"
: "${RL_CONDA_ENV:=otagent}"

export SCRATCH DCFT
export HF_HOME="${HF_HOME:-$SCRATCH/cache/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$SCRATCH/flashinfer_ws}"
# Prefer Luke-owned MarinSkyRL (penfever's tree is often mode 700 / unreadable).
if [[ -z "${RL_REPO_DIR:-}" ]]; then
  for cand in "$SCRATCH/MarinSkyRL" "$SCRATCH/SkyRL" \
              "$PENFEVER_SCRATCH/MarinSkyRL" "$PENFEVER_SCRATCH/SkyRL"; do
    if [[ -d "$cand/skyrl-train" ]]; then
      export RL_REPO_DIR="$cand"
      break
    fi
  done
fi

cd "$DCFT"
# shellcheck disable=SC1091
source "$DCFT/hpc/dotenv/tacc.env"
export DCFT="$DCFT"
export PYTHONPATH="${DCFT}${PYTHONPATH:+:$PYTHONPATH}"
export CONDA_EXE="${CONDA_BASE}/bin/conda"

# Prefer Luke's secrets (never penfever's keys for WandB/Daytona identity).
if [[ -f "$SCRATCH/keys.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$SCRATCH/keys.env"
  set +a
elif [[ -f "$HOME/.config/otagent/secrets.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$HOME/.config/otagent/secrets.env"
  set +a
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: WANDB_API_KEY unset after sourcing secrets. Expected $SCRATCH/keys.env"
  exit 1
fi
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-vista-moe-gsm8k-grpo}"
export DC_AGENT_SECRET_ENV="${DC_AGENT_SECRET_ENV:-$SCRATCH/keys.env}"
# Ensure HF hub sees a token if secrets use either common name.
if [[ -z "${HF_TOKEN:-}" && -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  export HF_TOKEN="$HUGGING_FACE_HUB_TOKEN"
fi
echo "WandB: mode=$WANDB_MODE project=$WANDB_PROJECT entity=${WANDB_ENTITY:-unset}"

if [[ ! -f "$DATA_DIR/train.parquet" || ! -f "$DATA_DIR/validation.parquet" ]]; then
  echo "ERROR: gsm8k parquet missing under $DATA_DIR"
  echo "Run: bash hpc/skyrl_standard/vista/prep_gsm8k_parquet.sh"
  exit 1
fi

LAUNCH_PY="${CONDA_BASE}/envs/${RL_CONDA_ENV}/bin/python"
if [[ ! -x "$LAUNCH_PY" ]]; then
  echo "ERROR: missing $LAUNCH_PY"
  exit 1
fi

echo "Launching $JOB_NAME on $PARTITION ($NUM_NODES nodes, wall $TIME_LIMIT)"
echo "  model=$MODEL_PATH"
echo "  data=$DATA_DIR"
echo "  SKYRL/RL_REPO_DIR=${RL_REPO_DIR:-unset}"
echo "  python=$LAUNCH_PY"

"$LAUNCH_PY" -m hpc.launch \
  --job_type rl \
  --rl_config ./hpc/skyrl_yaml/vista/8node_qwen3_30b_a3b_gsm8k_grpo.yaml \
  --rl_use_conda \
  --rl_conda_env "$RL_CONDA_ENV" \
  --model_path "$MODEL_PATH" \
  --train_data "[\"${DATA_DIR}/train.parquet\"]" \
  --val_data "[\"${DATA_DIR}/validation.parquet\"]" \
  --num_nodes "$NUM_NODES" \
  --partition "$PARTITION" \
  --time_limit "$TIME_LIMIT" \
  --experiments_dir "$EXPERIMENTS_DIR" \
  --job_name "$JOB_NAME" \
  --skyrl_override environment.env_class=gsm8k \
  --skyrl_override trainer.enable_db_registration=false \
  --skyrl_override trainer.logger=wandb \
  --skyrl_override trainer.project_name=vista-moe-gsm8k-grpo \
  --skyrl_override trainer.run_name="$JOB_NAME"
