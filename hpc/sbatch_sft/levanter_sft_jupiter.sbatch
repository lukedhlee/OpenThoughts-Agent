#!/bin/bash
#SBATCH --job-name=ota10k_lev_sft
#SBATCH --partition=booster
#SBATCH --account=reformo
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=11:59:00
#SBATCH --output=/e/fscratch/reformo/lee27/experiments/ota10k_lev_sft/logs/%x_%j.out

# Raw levanter.main.train_lm SFT on Jupiter GH200 (no iris/ArtifactStep).
# LEV_CONFIG must be exported at submit time (path to a TrainLmConfig yaml).
# Extra CLI overrides pass through via LEV_EXTRA_ARGS (e.g. smoke-test shrinkage).
set -uo pipefail
F=/e/fscratch/reformo/lee27
mkdir -p $F/experiments/ota10k_lev_sft/logs

source $F/envs/levanter/bin/activate
export HF_HUB_CACHE=$F/hf_hub
export HF_HOME=$F/cache/hf
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export WANDB_DIR=$F/experiments/ota10k_lev_sft
export TMPDIR=/tmp

# marin #7115: resume-time BFC fragmentation OOM -> cuda_async defrag allocator.
export JAX_PJRT_CLIENT_CREATE_OPTIONS=allocator:cuda_async

export NCCL_DEBUG=WARN
export NCCL_NET_GDR_LEVEL=0
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_TIMEOUT=60

# levanter #wall-SIGKILL: a mid-save kill orphans a metadata-only ghost step in the
# staging mirror that poisons resume discovery -> self-clean before every (re)start.
rm -rf "$F/checkpoints/ota10k_sft_30ba3b_levanter/ckpt-temp" 2>/dev/null || true

srun --kill-on-bad-exit=1 \
  python -m levanter.main.train_lm --config_path "$LEV_CONFIG" ${LEV_EXTRA_ARGS:-}
