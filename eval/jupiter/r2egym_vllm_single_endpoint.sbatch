#!/usr/bin/env bash
#SBATCH --partition=booster
#SBATCH --account=reformo
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --job-name=r2e_vllm_strict

set -eo pipefail

: "${JSC_SCRATCH:=/e/scratch/reformo/lee27}"
: "${RL_VENV:=$JSC_SCRATCH/OpenThoughts-Agent/envs/rl-megatron}"
: "${MODEL_PATH:=$JSC_SCRATCH/cache/hf/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
: "${ENDPOINT_FILE:=$JSC_SCRATCH/experiments/r2egym_step4/vllm_${SLURM_JOB_ID}.endpoint}"

module load GCC/14.3.0
module load nvidia-compilers/25.9-CUDA-13
source "$RL_VENV/bin/activate"

site_packages="$RL_VENV/lib/python3.12/site-packages"
export LD_LIBRARY_PATH="$site_packages/nvidia/cu13/lib:$site_packages/nvidia/cublas/lib:$site_packages/nvidia/cuda_cupti/lib:$site_packages/nvidia/cuda_nvrtc/lib:$site_packages/nvidia/cuda_runtime/lib:$site_packages/nvidia/cudnn/lib:$site_packages/nvidia/cufft/lib:$site_packages/nvidia/cufile/lib:$site_packages/nvidia/curand/lib:$site_packages/nvidia/cusolver/lib:$site_packages/nvidia/cusparse/lib:$site_packages/nvidia/cusparselt/lib:$site_packages/nvidia/nccl/lib:$site_packages/nvidia/nvjitlink/lib:$site_packages/nvidia/nvshmem/lib:$site_packages/nvidia/nvtx/lib:$site_packages/torch/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="$JSC_SCRATCH/cache/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

host="$(hostname -I | awk '{print $1}')"
test -n "$host"
export VLLM_HOST_IP="$host"

mkdir -p "$(dirname "$ENDPOINT_FILE")"
printf 'http://%s:8000/v1\n' "$host" > "$ENDPOINT_FILE"
echo "R2E_VLLM_ENDPOINT=http://${host}:8000/v1"

exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name qwen3-8b \
  --host "$host" \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 4 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --reasoning-parser qwen3
