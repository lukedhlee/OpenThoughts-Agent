#!/bin/bash
# forward_nll.sbatch — Marin's teacher-forced NLL of ROWS under a native checkpoint (forward only), in the training layout
# (16 one-GPU ranks on 4 nodes, train_update.sh's environment). Pair with probe.sbatch NLL_UPDATE mode (vLLM side).
#   sbatch -N 4 -t 01:00:00 -o <out>/fnll_%j.out --export=ALL,INIT=<native>,BASE=<hf dir>,ROWS=<jsonl>,OUT=<json>,N=32 forward_nll.sbatch
#SBATCH -p debug
#SBATCH -A CCR24067
#SBATCH --exclusive
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=36
set -uo pipefail
MARIN_ROOT=${MARIN_ROOT:-$HOME/snowball/marin-calibforge-roft}; PYM=$HOME/snowball/envs/marin-grug-sft/bin/python
unset HTTPS_PROXY https_proxy HTTP_PROXY http_proxy ALL_PROXY all_proxy NO_PROXY no_proxy
export CC=gcc CXX=g++ CUDA_HOME=${CUDA_HOME:-/home1/apps/nvidia/Linux_aarch64/26.9/cuda/13.3}
export GIT_PYTHON_REFRESH=quiet WANDB_DISABLE_GIT=true WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda,cpu HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 TRITON_CC="$(command -v gcc)" LD_PRELOAD="$(gcc -print-file-name=libstdc++.so.6)"
export JAX_ENABLE_PGLE=false XLA_FLAGS="--xla_gpu_enable_command_buffer="
export JAX_COMPILATION_CACHE_DIR=/scratch/11584/$USER/snowball-sft/jax-cache/online-reflection
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
export SNOWBALL_SEQ_LEN=65536 SNOWBALL_BATCH=16 SNOWBALL_NODES=$SLURM_NNODES SNOWBALL_DEVICES=$((4 * SLURM_NNODES))
export NCCL_SOCKET_IFNAME=ib NCCL_IB_TIMEOUT=22 NCCL_IB_RETRY_CNT=13 NCCL_WIN_ENABLE=0
export PYTHONPATH="$MARIN_ROOT:$MARIN_ROOT/lib/levanter/src:$MARIN_ROOT/lib/haliax/src:$MARIN_ROOT/lib/marin/src:$MARIN_ROOT/lib/rigging/src:$MARIN_ROOT/lib/fray/src:$MARIN_ROOT/lib/zephyr/src:$MARIN_ROOT/lib/iris/src"
ulimit -c 0; cd "$MARIN_ROOT"
echo "forward_nll job=$SLURM_JOB_ID marin=$(git rev-parse --short HEAD) init=$INIT base=$BASE rows=$ROWS"
srun --nodes=$SLURM_NNODES --ntasks=$((4 * SLURM_NNODES)) --ntasks-per-node=4 --gpu-bind=none --cpu-bind=cores --kill-on-bad-exit=1 \
  "$PYM" -m experiments.june_tpu_67b_a2b.moe.forward_nll --init "$INIT" --base "$BASE" --rows "$ROWS" --n ${N:-32} \
  --expert-axis ${EXPERT_AXIS:-8} --out "$OUT"
echo "FORWARD_NLL_DONE rc=$?"
