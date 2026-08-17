#!/bin/bash
#SBATCH --job-name=ota10k_lev_sft
#SBATCH --partition=booster
#SBATCH --account=reformo
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=11:59:00
#SBATCH --output=/e/fscratch/reformo/lee27/experiments/ota10k_lev_sft/logs/%x_%j.out
# Targeted ghost-node exclusions (2026-08-17 storm): NOT the full RL blacklist —
# only sets with confirmed stranded-memory incidents this session. Prod v2 (1397721)
# died in 8 min on jpbo-077 residue from our own scancel; exclusion-free SFT keeps
# landing on poisoned idle nodes.
#SBATCH --exclude=jpbo-076-[01-48],jpbo-026-[01-48],jpbo-107-[01-06],jpbo-040-[17-20,23-24],jpbo-077-[01-06],jpbo-095-[21-26],jpbo-106-20,jpbo-114-[01-03,05-06,08,11,13]

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
# compute nodes have no git on PATH; GitPython hard-fails without this
export GIT_PYTHON_REFRESH=quiet
export WANDB_DISABLE_GIT=true
export TMPDIR=/tmp

# marin #7115: resume-time BFC fragmentation OOM -> cuda_async defrag allocator.
export JAX_PJRT_CLIENT_CREATE_OPTIONS=allocator:cuda_async

# incident 41 (job 1396849): fused-CE autotune sweep at the 32k/151936 prod shape
# compiled pathological candidates (3.3TB-arg modules) for 2h23m until a rank died
# and the shutdown barrier killed all 8. Sweep results were unshareable anyway
# (jaxpr hash unavailable) -> pin to inferred block sizes, no sweep.
export LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS=0
# Persistent XLA module cache: restart links skip recompile entirely.
export JAX_COMPILATION_CACHE_DIR=$F/cache/jax_comp_cache

export NCCL_DEBUG=WARN
export NCCL_NET_GDR_LEVEL=0
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_TIMEOUT=60

# levanter #wall-SIGKILL: a mid-save kill orphans a metadata-only ghost step in the
# staging mirror that poisons resume discovery -> self-clean before every (re)start.
rm -rf "$F/checkpoints/ota10k_sft_30ba3b_levanter/ckpt-temp" 2>/dev/null || true

srun --kill-on-bad-exit=1 \
  python -m levanter.main.train_lm --config_path "$LEV_CONFIG" ${LEV_EXTRA_ARGS:-}
