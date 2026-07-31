#!/bin/bash
#SBATCH --job-name=opencode_vllm_qwen3_0p6b
#SBATCH --account=synthlaion
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/p/scratch/synthlaion/lee27/dc_agent_eval/logs/opencode_vllm_qwen3_0p6b_%j.out

set -euo pipefail

VLLM_ENV="${VLLM_ENV:-/p/scratch/synthlaion/lee27/envs/skyrl-venv}"
MODEL_SNAPSHOT="${MODEL_SNAPSHOT:-/p/scratch/synthlaion/lee27/cache/hf/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-0.6b}"
SERVE_PORT="${SERVE_PORT:-8000}"

export HF_HOME="${HF_HOME:-/p/scratch/synthlaion/lee27/cache/hf}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/p/scratch/synthlaion/lee27/cache/vllm}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/p/scratch/synthlaion/lee27/cache/xdg}"
export TOKENIZERS_PARALLELISM=false

mkdir -p \
  /p/scratch/synthlaion/lee27/dc_agent_eval/logs \
  "$VLLM_CACHE_ROOT" \
  "$XDG_CACHE_HOME"

if [[ ! -x "$VLLM_ENV/bin/python" ]]; then
  echo "ERROR: vLLM environment is missing: $VLLM_ENV" >&2
  exit 1
fi
if [[ ! -f "$MODEL_SNAPSHOT/config.json" ]]; then
  echo "ERROR: cached model snapshot is missing: $MODEL_SNAPSHOT" >&2
  exit 1
fi

source "$VLLM_ENV/bin/activate"

NODE_SHORT="$(hostname -s)"
NODE_IB="${NODE_SHORT}i"
echo "vLLM node: $NODE_SHORT"
echo "OpenCode endpoint: http://${NODE_IB}:${SERVE_PORT}/v1"
echo "Model snapshot: $MODEL_SNAPSHOT"
python -c 'import torch, vllm; print(f"torch={torch.__version__} vllm={vllm.__version__}")'

exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_SNAPSHOT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$SERVE_PORT" \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.80 \
  --max-num-seqs 8 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
