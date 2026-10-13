#!/bin/bash
# train_update.sh REQUEST RESPONSE — backend.update_command for one native reflection update (marin
# experiments/june_tpu_67b_a2b/moe/online_reflection.py), inside job.sbatch's allocation: 16 one-GPU ranks on the job's
# training nodes (every node but the first, which serves and runs the controller). The environment is
# horizon_snowball_guarded.sbatch's (GCC, CUDA 13.3, NCCL over IPoIB, command buffers off) plus relay_sft_arm.sh's
# 64k memory settings. marin_adapter runs this with cwd = backend.marin_root. A GPU sampler logs utilization and memory
# of the training nodes every 10 s beside the request.
set -uo pipefail
REQ=${1:?request}; RES=${2:?response}
: "${SLURM_JOB_ID:?run inside job.sbatch}"
MARIN_ROOT=$(pwd); PYM=${MARIN_PYTHON:-$HOME/snowball/envs/marin-grug-sft/bin/python}
NODE0=$(hostname -s)
TRAIN=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | grep -vx "$NODE0" | paste -sd,)
N=$(echo "$TRAIN" | tr ',' '\n' | wc -l)
[ "$N" = "${REFLECTION_TRAIN_NODES:-4}" ] || { echo "expected ${REFLECTION_TRAIN_NODES:-4} training nodes, got $N ($TRAIN)"; exit 1; }
OUT=$(dirname "$REQ")
unset HTTPS_PROXY https_proxy HTTP_PROXY http_proxy ALL_PROXY all_proxy NO_PROXY no_proxy   # gRPC (JAX coordinator) honors these
export CC=gcc CXX=g++ CUDA_HOME=${CUDA_HOME:-/home1/apps/nvidia/Linux_aarch64/26.9/cuda/13.3}
export GIT_PYTHON_REFRESH=quiet WANDB_DISABLE_GIT=true WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 JAX_PLATFORMS=cuda,cpu
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1
export TRITON_CC="$(command -v gcc)"; export LD_PRELOAD="$(gcc -print-file-name=libstdc++.so.6)"
export JAX_ENABLE_PGLE=false XLA_FLAGS="--xla_gpu_enable_command_buffer="
export JAX_COMPILATION_CACHE_DIR=/scratch/11584/$USER/snowball-sft/jax-cache/online-reflection
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1 JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
export SNOWBALL_SEQ_LEN=65536 SNOWBALL_BATCH=16 SNOWBALL_NODES=$N SNOWBALL_DEVICES=$((4 * N))
export NCCL_SOCKET_IFNAME=ib NCCL_IB_TIMEOUT=22 NCCL_IB_RETRY_CNT=13 NCCL_WIN_ENABLE=0
export PYTHONPATH="$MARIN_ROOT:$MARIN_ROOT/lib/levanter/src:$MARIN_ROOT/lib/haliax/src:$MARIN_ROOT/lib/marin/src:$MARIN_ROOT/lib/rigging/src:$MARIN_ROOT/lib/fray/src:$MARIN_ROOT/lib/zephyr/src:$MARIN_ROOT/lib/iris/src"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
ulimit -c 0
echo "[$(date -Is)] train_update job=$SLURM_JOB_ID nodes=$TRAIN marin=$(git -C "$MARIN_ROOT" rev-parse --short HEAD) request=$REQ"
srun --overlap --jobid="$SLURM_JOB_ID" --nodelist="$TRAIN" --ntasks-per-node=1 --gres=gpu:4 --gpu-bind=none --cpu-bind=none --mem=500M \
  bash -c 'while :; do nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader | sed "s/^/$(hostname -s) /"; sleep 10; done' \
  > "$OUT/gpu_samples.log" 2>&1 &
SAMPLER=$!
t0=$(date +%s)
srun --nodelist="$TRAIN" --nodes="$N" --ntasks=$((4 * N)) --ntasks-per-node=4 --gpu-bind=none --cpu-bind=cores \
  --kill-on-bad-exit=1 "$PYM" -m experiments.june_tpu_67b_a2b.moe.online_reflection --request "$REQ" --response "$RES"
rc=$?
kill "$SAMPLER" 2>/dev/null; wait "$SAMPLER" 2>/dev/null
echo "[$(date -Is)] train_update rc=$rc wall=$(( $(date +%s) - t0 ))s"
exit $rc
