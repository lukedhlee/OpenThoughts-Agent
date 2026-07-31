#!/usr/bin/env bash
# Start one Qwen3-8B vLLM endpoint per Slurm task inside an existing Jupiter
# allocation. Rank 0 uses GPU 1 / port 8001, rank 1 GPU 2 / port 8002, etc.;
# GPU 0 / port 8000 is reserved for the allocation's primary server.
set -eo pipefail

: "${JSC_SCRATCH:=/e/scratch/reformo/lee27}"
: "${RL_VENV:=$JSC_SCRATCH/OpenThoughts-Agent/envs/rl-megatron}"
: "${MODEL_PATH:=$JSC_SCRATCH/cache/hf/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"

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

rank="${SLURM_PROCID:?SLURM_PROCID is required}"
gpu=$((rank + 1))
port=$((8001 + rank))
host="$(hostname -I | awk '{print $1}')"
test -n "$host"
export CUDA_VISIBLE_DEVICES="$gpu"

echo "R2E_VLLM_ENDPOINT=http://${host}:${port}/v1 GPU=$gpu"
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name qwen3-8b \
  --host "$host" \
  --port "$port" \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 4 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --reasoning-parser qwen3
