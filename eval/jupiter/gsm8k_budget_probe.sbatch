#!/bin/bash
# ==============================================================================
# eval/jupiter/gsm8k_budget_probe.sbatch — single-node, inference-only probe
# ==============================================================================
# Answers: was step-0 pass_at_1=0.4511 on GSM8K a generation-budget truncation
# artifact rather than the model's math ability?
#
# ONE generation pass at 4096 tokens, then the SAME rollouts re-scored at 1024
# and 2048 tokens (paired — no sampling noise between arms). See
# scripts/analysis/gsm8k_budget_probe.py for the full rationale.
#
# NO training: no optimizer, no FSDP, no checkpoints, no Ray, no Daytona.
# 1 node / 4 GH200 (TP=4). Reads the val parquet + HF cache; writes one JSON.
#
# Env setup is copied from the rendered sbatch of the run this probe explains
# (jupiter_moe30b_gsm8k_grpo_6n_fast_50step_eval5) so vLLM sees an identical
# environment — same venv, same CUDA libs, same VLLM_* flags.
#
# Usage (from the repo root on a Jupiter login node):
#   sbatch eval/jupiter/gsm8k_budget_probe.sbatch
# Override via env, e.g. a fast smoke over 40 examples:
#   PROBE_LIMIT=40 TIME_LIMIT=00:40:00 sbatch eval/jupiter/gsm8k_budget_probe.sbatch
# ==============================================================================
#SBATCH --job-name=jupiter_gsm8k_budget_probe
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH -p booster
#SBATCH --account=reformo
#SBATCH --output=/e/scratch/reformo/lee27/experiments/jupiter_gsm8k_budget_probe/logs/%x_%j.out
#SBATCH --mail-type=END,TIME_LIMIT,FAIL

set -eo pipefail
ulimit -c 0
ulimit -n 65536 2>/dev/null || true

SCRATCH_ROOT="${SCRATCH_ROOT:-/e/scratch/reformo/lee27}"
WORKDIR="${WORKDIR:-$SCRATCH_ROOT/OpenThoughts-Agent}"
RUN_DIR="$SCRATCH_ROOT/experiments/jupiter_gsm8k_budget_probe"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/results"

# --- knobs -------------------------------------------------------------------
PARQUET="${PROBE_PARQUET:-$SCRATCH_ROOT/data/gsm8k/validation.parquet}"
MODEL="${PROBE_MODEL:-Qwen/Qwen3-30B-A3B}"
MAX_GEN="${PROBE_MAX_GEN:-4096}"
TRUNCATE_AT="${PROBE_TRUNCATE_AT:-1024 2048}"
TP="${PROBE_TP:-4}"
MAX_MODEL_LEN="${PROBE_MAX_MODEL_LEN:-8192}"
GPU_UTIL="${PROBE_GPU_UTIL:-0.85}"
LIMIT="${PROBE_LIMIT:-0}"
OUT_JSON="${PROBE_OUT:-$RUN_DIR/results/probe_${SLURM_JOB_ID:-manual}.json}"

# --- modules (same as the RL run) -------------------------------------------
module load GCC/14.3.0
module load nvidia-compilers/25.9-CUDA-13

# --- venv (NOT conda; conda is deliberately left out to avoid import conflicts)
RL_ENV_DIR="${RL_ENV_DIR:-$WORKDIR/envs/rl}"
if [ ! -x "$RL_ENV_DIR/bin/python" ]; then
  echo "FATAL: no python at $RL_ENV_DIR/bin/python" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$RL_ENV_DIR/bin/activate"
if [ -n "${CUDA_HOME:-}" ]; then
  export PATH="$CUDA_HOME/bin:$PATH"
fi

SP="$RL_ENV_DIR/lib/python3.12/site-packages"
# The nvidia wheel libs must precede anything else or vLLM fails to load its
# compiled extensions on aarch64/GH200 (same list as the RL sbatch).
export LD_LIBRARY_PATH="$SP/nvidia/cu13/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_cupti/lib:$SP/nvidia/cuda_nvrtc/lib:$SP/nvidia/cuda_runtime/lib:$SP/nvidia/cudnn/lib:$SP/nvidia/cufft/lib:$SP/nvidia/cufile/lib:$SP/nvidia/curand/lib:$SP/nvidia/cusolver/lib:$SP/nvidia/cusparse/lib:$SP/nvidia/cusparselt/lib:$SP/nvidia/nccl/lib:$SP/nvidia/nvjitlink/lib:$SP/nvidia/nvshmem/lib:$SP/nvidia/nvtx/lib:$SP/torch/lib:${LD_LIBRARY_PATH:-}"

# --- HF cache (model is already pulled; stay offline-friendly) ---------------
export HF_HOME="${HF_HOME:-$SCRATCH_ROOT/cache/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

# --- vLLM flags (mirror the RL run) -----------------------------------------
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND="FLASH_ATTN"
export VLLM_FORCE_IPV4=1
export VLLM_SKIP_FLAG_DISCOVERY=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# GH200/aarch64: TP>1 spawned workers fail Triton linking without this.
export LIBRARY_PATH="/.singularity.d/libs:${LIBRARY_PATH:-}"

# skyrl_gym supplies the scorer; the probe hard-fails if it can't import it.
export PYTHONPATH="$SCRATCH_ROOT/MarinSkyRL/skyrl-train:$SCRATCH_ROOT/MarinSkyRL/skyrl-gym:$WORKDIR:${PYTHONPATH:-}"

echo "=============================================================="
echo "GSM8K budget probe (INFERENCE ONLY — no training, no Daytona)"
echo "  job          : ${SLURM_JOB_ID:-manual} on $(hostname)"
echo "  model        : $MODEL"
echo "  parquet      : $PARQUET"
echo "  max_generate : $MAX_GEN   (re-scored at: $TRUNCATE_AT)"
echo "  TP           : $TP        max_model_len: $MAX_MODEL_LEN"
echo "  limit        : ${LIMIT} (0 = all 1319)"
echo "  out          : $OUT_JSON"
echo "=============================================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

# shellcheck disable=SC2086
srun --ntasks=1 python "$WORKDIR/scripts/analysis/gsm8k_budget_probe.py" \
  --parquet "$PARQUET" \
  --model "$MODEL" \
  --output "$OUT_JSON" \
  --max-generate-length "$MAX_GEN" \
  --truncate-at $TRUNCATE_AT \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --limit "$LIMIT"

echo "Probe complete. Results: $OUT_JSON"
