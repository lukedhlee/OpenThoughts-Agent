#!/bin/bash
# Jupiter AGENTIC GRPO on Qwen/Qwen3-30B-A3B + r2egym over the apptainer bridge.
# Default: 6 nodes / 24 GPU, FSDP2 EP=4, flash_attn (6node_qwen3_30b_a3b_r2egym_grpo.yaml).
#
# Prerequisites (login node):
#   cd <OpenThoughts-Agent> && source hpc/dotenv/jupiter.env
#   set -a; source $DC_AGENT_SECRET_ENV; set +a       # WANDB_API_KEY, HF token
#   model snapshot cached under $HF_HUB_CACHE (compute has no internet)
#   train dataset cached: python -c "from huggingface_hub import snapshot_download; \
#     snapshot_download('DCAgent/r2egym-patched-full-oracle', repo_type='dataset')"
#   apptainer bridge up (harbor server.py + worker fleet, see
#     harbor src/harbor/environments/apptainer/jureca_workers.sbatch) and
#   APPTAINER_BRIDGE_URL exported (put it in hpc/dotenv/jupiter.local.env)
#
# Usage (from the repo root on Jupiter):
#   bash hpc/skyrl_standard/jupiter/run_r2egym_moe30b_grpo.sh
#   MAX_STEPS=3 NUM_NODES=6 TIME_LIMIT=02:00:00 JOB_NAME=r2egym30b_smoke bash ...
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
: "${DCFT:=$REPO_ROOT}"
if [[ -z "${SCRATCH:-}" && -f "$DCFT/hpc/dotenv/jupiter.env" ]]; then
  # shellcheck disable=SC1091
  source "$DCFT/hpc/dotenv/jupiter.env"
fi
: "${SCRATCH:?SCRATCH unset — source hpc/dotenv/jupiter.env first}"
: "${EXPERIMENTS_DIR:=$DCFT/experiments}"
: "${PARTITION:=booster}"
: "${ACCOUNT:=${DCFT_GROUP:-}}"
: "${RESERVATION:=}"
if [[ "${RESERVATION}" == "none" || "${RESERVATION}" == "off" || "${RESERVATION}" == "-" ]]; then
  RESERVATION=
fi
: "${TIME_LIMIT:=11:59:00}"
: "${NUM_NODES:=6}"
: "${JOB_NAME:=jupiter_moe30b_r2egym_grpo_6n}"
: "${RL_CONFIG:=./hpc/skyrl_yaml/jupiter/6node_qwen3_30b_a3b_r2egym_grpo.yaml}"
: "${MODEL_ID:=Qwen/Qwen3-30B-A3B}"
: "${MODEL_PATH:=}"
: "${TRAIN_DATA:=DCAgent/r2egym-patched-full-oracle}"
: "${RL_VENV:=${DCFT_RL_ENV:-$DCFT/envs/rl}}"
: "${RL_CONDA_ENV:=}"
: "${CONDA_BASE:=${CONDA_PREFIX:-$SCRATCH/miniforge3}}"
: "${WANDB_PROJECT:=jupiter-r2egym-grpo}"

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

# The apptainer environment resolves the bridge from this variable at trial
# time; without it every trial fails at construction. Keep it in the per-user
# overlay so the job-side re-source of jupiter.env restores it on compute.
if [[ -z "${APPTAINER_BRIDGE_URL:-}" ]]; then
  echo "ERROR: APPTAINER_BRIDGE_URL unset — the r2egym sandboxes run over the apptainer bridge."
  echo "  Start the bridge (harbor apptainer server + workers), then export"
  echo "  APPTAINER_BRIDGE_URL=http://<bridge-host>:<port> in hpc/dotenv/jupiter.local.env"
  exit 1
fi

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

if ! compgen -G "$HF_HUB_CACHE/datasets--${TRAIN_DATA//\//--}/snapshots/*" >/dev/null; then
  echo "ERROR: $TRAIN_DATA not cached under $HF_HUB_CACHE (compute has no internet)."
  echo "  On a login node: python -c \"from huggingface_hub import snapshot_download; snapshot_download('$TRAIN_DATA', repo_type='dataset')\""
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
  echo "ERROR: no RL env. Run: cd \$DCFT && bash hpc/env_builds/jupiter/build_rl_fa_env.sh (or set DCFT_RL_ENV)"
  echo "  expected venv: $RL_VENV"
  exit 1
fi
if [[ ! -x "$LAUNCH_PY" ]]; then
  echo "ERROR: missing $LAUNCH_PY"
  exit 1
fi

cd "$DCFT"
export DCFT_RL_ENV="${RL_VENV}"
export WANDB_PROJECT="$LAUNCH_WANDB_PROJECT"
export EXPERIMENTS_DIR
export OT_AGENT_RAY_LOG_DIR="${OT_AGENT_RAY_LOG_DIR:-$EXPERIMENTS_DIR/_ray_logs}"
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
echo "  data=$TRAIN_DATA"
echo "  bridge=$APPTAINER_BRIDGE_URL"
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
  --skyrl_override trainer.enable_db_registration=false
  --skyrl_override trainer.logger=wandb
  --skyrl_override trainer.project_name="$WANDB_PROJECT"
  --skyrl_override trainer.run_name="$JOB_NAME"
  --skyrl_override trainer.max_steps="${MAX_STEPS:-200}"
  --skyrl_override trainer.ckpt_interval="${CKPT_INTERVAL:-20}"
)
if [[ -n "${HF_SAVE_INTERVAL+x}" ]]; then
  SKYRL_OVERRIDES+=(--skyrl_override trainer.hf_save_interval="$HF_SAVE_INTERVAL")
fi
if [[ -n "${RESUME_MODE+x}" ]]; then
  SKYRL_OVERRIDES+=(--skyrl_override trainer.resume_mode="$RESUME_MODE")
fi
if [[ -n "${TRAIN_BATCH_SIZE+x}" ]]; then
  SKYRL_OVERRIDES+=(--skyrl_override trainer.train_batch_size="$TRAIN_BATCH_SIZE")
fi
if [[ -n "${POLICY_MINI_BATCH_SIZE+x}" ]]; then
  SKYRL_OVERRIDES+=(--skyrl_override trainer.policy_mini_batch_size="$POLICY_MINI_BATCH_SIZE")
fi
if [[ -n "${POLICY_LR+x}" ]]; then
  SKYRL_OVERRIDES+=(--skyrl_override trainer.policy.optimizer_config.lr="$POLICY_LR")
fi
if [[ -n "${N_SAMPLES_PER_PROMPT+x}" ]]; then
  SKYRL_OVERRIDES+=(--skyrl_override generator.n_samples_per_prompt="$N_SAMPLES_PER_PROMPT")
fi
if [[ -n "${N_CONCURRENT_TRIALS+x}" ]]; then
  SKYRL_OVERRIDES+=(--skyrl_override terminal_bench.harbor.n_concurrent_trials="$N_CONCURRENT_TRIALS")
fi

"$LAUNCH_PY" -m hpc.launch \
  "${LAUNCH_ARGS[@]}" \
  --train_data "[\"$TRAIN_DATA\"]" \
  --num_nodes "$NUM_NODES" \
  --partition "$PARTITION" \
  --time_limit "$TIME_LIMIT" \
  --experiments_dir "$EXPERIMENTS_DIR" \
  --job_name "$JOB_NAME" \
  "${SKYRL_OVERRIDES[@]}" \
  "${EXTRA_LAUNCH_ARGS[@]}"
