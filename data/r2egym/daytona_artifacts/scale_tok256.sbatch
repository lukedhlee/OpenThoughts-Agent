#!/bin/bash
#SBATCH --job-name=daytona-scale-tok256
#SBATCH --partition=booster
#SBATCH --account=reformo
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=00:45:00
#SBATCH --output=/e/fscratch/reformo/lee27/experiments/daytona_scale_1024_20260915/%x_%j.out
set -uo pipefail
umask 077
ulimit -c 0
ulimit -n 65536
module load GCC/14.3.0
module load nvidia-compilers/25.9-CUDA-13
export SCALE_X=/e/fscratch/reformo/lee27/experiments/daytona_scale_1024_20260915
export SCALE_T=/e/fscratch/reformo/lee27/tasks/r2egym-daytona-v3
V=/e/project1/transfernetx/lee27/code/envs/snowball-v2
export PATH="$V/bin:$PATH"
export PYTHONPATH=/e/project1/transfernetx/lee27/code/src/marin_vllm_eagle3:/e/fscratch/reformo/lee27/code/harbor-pacing/src:/e/fscratch/reformo/lee27/experiments/tokenization_transport_20260915/deps:$SCALE_X
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_ALLREDUCE_USE_SYMM_MEM=0 TORCH_CUDNN_SDPA_ENABLED=0 VLLM_MQ_MAX_CHUNKS=240 VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_SERVER_DEV_MODE=1
CACHE=/e/fscratch/reformo/lee27/cache
export VLLM_CACHE_ROOT=$CACHE/vllm XDG_CACHE_HOME=$CACHE/xdg TRITON_CACHE_DIR=$CACHE/triton TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor FLASHINFER_WORKSPACE_BASE=$CACHE/flashinfer TMPDIR=$CACHE/tmp
set -a
source /e/fscratch/reformo/lee27/keys/socks5_currease.env
source /e/fscratch/reformo/lee27/keys/daytona_eval.env
set +a
export OPENAI_API_KEY=EMPTY
export PYTHONFAULTHANDLER=1
# 4 coordinator processes share the org's 5 creates/s
export HARBOR_DAYTONA_CREATE_RATE=5 HARBOR_DAYTONA_CREATE_SHARES=4
# RL-faithful: rollout details on, so harbor tokenizes every turn (the path that timed out at 256 agents)
export SCALE_PROBE_FLAGS="--collect-rollout-details"
# reuse slices 8..11 (tasks 512..767 of the 1,024) so the tasks differ from nothing else running
export SCALE_NODE_OFFSET=4
export HTTPS_PROXY=http://127.0.0.1:18946 HTTP_PROXY=http://127.0.0.1:18946
export NO_PROXY=127.0.0.1,localhost,10.128.1.2
mkdir -p "$SCALE_X/run_$SLURM_JOB_ID"
srun --kill-on-bad-exit=0 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=64 --gres=gpu:4 bash "$SCALE_X/scale_node_run_v2.sh"
echo "all nodes finished $(date +%s)" >> "$SCALE_X/run_$SLURM_JOB_ID/ALL_DONE"
