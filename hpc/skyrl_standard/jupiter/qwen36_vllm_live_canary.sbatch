#!/usr/bin/env bash
# One-GPU exact-model endpoint for allocation-free OpenCode context canaries.
# This is a diagnostic server only; it does not replace the distributed RL gate.
#SBATCH --partition=booster
#SBATCH --account=reformo
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --job-name=qwen36_vllm_canary
#SBATCH --output=/e/scratch/reformo/lee27/experiments/qwen36_vllm_canary_%j.out

set -euo pipefail

JSC_SCRATCH=/e/scratch/reformo/lee27
RL_VENV="$JSC_SCRATCH/OpenThoughts-Agent/envs/rl-megatron"
MODEL_REVISION=995ad96eacd98c81ed38be0c5b274b04031597b0
# MODEL_ROOT selects the FILESYSTEM the 26 shards are read from; the pinned
# revision stays in the directory name either way so the identity assert below
# still holds. vLLM loads shards SEQUENTIALLY in a single stream, so the load is
# bounded by per-stream read bandwidth: /e/scratch delivers ~57-98 MB/s per
# stream (measured O_DIRECT on compute jpbo-018-14, job 1224678) which is the
# whole of the 701s "Loading weights" in job 1217900, while /e/fscratch delivers
# 2.51 GiB/s per stream. Override to /e/fscratch/reformo/lee27 to read the
# flash-staged copy. Default is unchanged.
MODEL_ROOT="${MODEL_ROOT:-$JSC_SCRATCH}"
MODEL_PATH="${MODEL_PATH:-$MODEL_ROOT/models/Qwen3.6-35B-A3B/$MODEL_REVISION}"

# Fail loud rather than silently serving a different revision: the pinned hash
# is load-bearing for the paired comparison, and it lives in the dir name only.
case "$MODEL_PATH" in
  *"$MODEL_REVISION"*) ;;
  *) echo "ERROR: MODEL_PATH does not carry revision $MODEL_REVISION: $MODEL_PATH" >&2; exit 1 ;;
esac
test -s "$MODEL_PATH/model.safetensors.index.json" || {
  echo "ERROR: no safetensors index under $MODEL_PATH" >&2; exit 1; }
echo "MODEL_PATH=$MODEL_PATH"
ENDPOINT_FILE="$JSC_SCRATCH/experiments/qwen36_vllm_canary_${SLURM_JOB_ID}.endpoint"

module load GCC/14.3.0
module load nvidia-compilers/25.9-CUDA-13
source "$RL_VENV/bin/activate"

site_packages="$RL_VENV/lib/python3.12/site-packages"
export LD_LIBRARY_PATH="$site_packages/nvidia/cu13/lib:$site_packages/nvidia/cublas/lib:$site_packages/nvidia/cuda_cupti/lib:$site_packages/nvidia/cuda_nvrtc/lib:$site_packages/nvidia/cuda_runtime/lib:$site_packages/nvidia/cudnn/lib:$site_packages/nvidia/cufft/lib:$site_packages/nvidia/cufile/lib:$site_packages/nvidia/curand/lib:$site_packages/nvidia/cusolver/lib:$site_packages/nvidia/cusparse/lib:$site_packages/nvidia/cusparselt/lib:$site_packages/nvidia/nccl/lib:$site_packages/nvidia/nvjitlink/lib:$site_packages/nvidia/nvshmem/lib:$site_packages/nvidia/nvtx/lib:$site_packages/torch/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export HF_HOME="$JSC_SCRATCH/cache/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_FORCE_IPV4=1
export FLASHINFER_WORKSPACE_BASE="/tmp/flashinfer_${USER}_${SLURM_JOB_ID}"
mkdir -p "$FLASHINFER_WORKSPACE_BASE"

host="$(hostname -I | awk '{print $1}')"
test -n "$host"
export VLLM_HOST_IP="$host"
printf 'http://%s:8000/v1\n' "$host" >"$ENDPOINT_FILE"
echo "QWEN36_VLLM_ENDPOINT=http://${host}:8000/v1"

# --- Startup / throughput knobs (defaults reproduce the previous behaviour) ---
#
# MM_LIMIT=0 adds --limit-mm-per-prompt '{"image":0,"video":0}'.
#
# WHY: Qwen3.6 is a multimodal SHELL. SkyRL unwraps it to the text tower
# (SKYRL_QWEN3_5_VLM_UNWRAP=1), the vision tower is parked and gets no gradients,
# and no rollout ever sends an image. But vLLM still profiles it: job 1217900
# logged "Encoder cache will be initialized with a budget of 16384 tokens, and
# profiled with 1 image items of the maximum feature size" at 06:37:08 and did
# not emit "Available KV cache memory" until 06:56:07 -- ~19 MINUTES of
# memory-profiling forward passes, in eager mode, through a 35B MoE, plus a
# further "Multi-modal warmup completed in 15.008s".
#
# With the checkpoint on /e/fscratch the weight load is 38s, so that profile is
# now ~95% of startup. Skipping the image profile should also FREE KV cache (the
# profile is what sizes it), on top of the time saved.
#
# MAX_NUM_SEQS: 1217900 measured "GPU KV cache size: 560,505 tokens" and
# "Maximum concurrency for 32,768 tokens per request: 17.11x" -- so KV supports
# ~17 full-length concurrent sequences while the RL config caps at 8. At the
# real prompt length (client window 20480 => ~15,360) headroom is ~36x. Raising
# this is the cheapest rollout-throughput lever available.
#
# MAX_NUM_BATCHED_TOKENS: the agentic workload is PREFILL-dominated -- measured
# 150,223 input tokens vs 3,377 output per trial on 1221005, a 44:1 ratio. At
# 4096 a ~15k prompt takes ~4 chunked-prefill passes.
: "${MAX_NUM_SEQS:=4}"
: "${MAX_NUM_BATCHED_TOKENS:=4096}"
: "${MM_LIMIT:=}"

VLLM_ARGS=(
  --model "$MODEL_PATH"
  --served-model-name "$MODEL_REVISION"
  --host "$host"
  --port 8000
  --tensor-parallel-size 1
  --max-model-len 32768
  --dtype bfloat16
  --gpu-memory-utilization 0.85
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --enable-prefix-caching
  --enable-chunked-prefill
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --gdn-prefill-backend triton
  --enforce-eager
)
if [[ -n "$MM_LIMIT" ]]; then
  VLLM_ARGS+=(--limit-mm-per-prompt "{\"image\":$MM_LIMIT,\"video\":$MM_LIMIT}")
fi
echo "VLLM_ARGS: ${VLLM_ARGS[*]}"

exec python -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}"
