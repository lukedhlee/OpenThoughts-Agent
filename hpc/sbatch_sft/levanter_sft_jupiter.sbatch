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
#SBATCH --exclude=jpbo-076-[01-48],jpbo-026-[01-48],jpbo-107-[01-06],jpbo-040-[17-20,23-24],jpbo-077-[01-06],jpbo-095-[21-26],jpbo-106-20,jpbo-114-[01-03,05-06,08,11,13],jpbo-046-[02-05,07,09,11-12]

# Raw levanter.main.train_lm SFT on Jupiter GH200 (no iris/ArtifactStep).
# LEV_CONFIG must be exported at submit time (path to a TrainLmConfig yaml).
# Extra CLI overrides pass through via LEV_EXTRA_ARGS (e.g. smoke-test shrinkage).
set -uo pipefail
F=/e/fscratch/reformo/lee27
# incident 44: 8-rank SIGABRT crashes dumped 258GB of cores into the repo cwd,
# blowing project quota and cascading into the next runs' failures.
ulimit -c 0
mkdir -p $F/experiments/ota10k_lev_sft/logs

source $F/envs/levanter/bin/activate
export HF_HUB_CACHE=$F/hf_hub
export HF_HOME=$F/cache/hf
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export WANDB_DIR=$F/experiments/ota10k_lev_sft
export WANDB_DATA_DIR=$F/cache/wandb_data
export WANDB_CACHE_DIR=$F/cache/wandb_cache
# incident 45: JAX default mem fraction (0.75 = 71.25GiB cap on 96GB GH200) was the
# real killer of every prod attempt — load/compile transients hit the self-imposed
# limit and RESOURCE_EXHAUSTED'd (the "ghost node" reads on 077/046/001 were wrong).
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
# compute nodes have no git on PATH; GitPython hard-fails without this
export GIT_PYTHON_REFRESH=quiet
export WANDB_DISABLE_GIT=true
export TMPDIR=/tmp

# marin #7115: resume-time BFC fragmentation OOM -> cuda_async defrag allocator.
export JAX_PJRT_CLIENT_CREATE_OPTIONS=allocator:cuda_async

# incident 48 REVERSAL of incident 41: autotune-OFF is the fast killer — the
# inferred CE block sizes OOM train_step in 7 min (9.6GB blocks) at ANY mem limit
# (v2/v2b at 71.25GiB, v3 at 87.40GiB). v1 with autotune ON survived this phase for
# 2h23m and its rank death is explained by the old 71.25 cap during the sweep.
# So: sweep ON + 0.92 fraction + compile cache. Expect a long (~2-3h) first compile.
# Persistent XLA module cache: restart links skip recompile entirely.
export JAX_COMPILATION_CACHE_DIR=$F/cache/jax_comp_cache

# incident 50: v4 (1398566) survived the CE sweep but died 1h48m in — XLA's own
# GEMM/fusion autotuner (default level 4: operand init + correctness check +
# redzone) RESOURCE_EXHAUSTED'd allocating 24GiB for 5 of 646 instructions.
# incident 51: level 1 still dies — v6 (1399224) at 22m50s: the autotuner tried to
# materialize 3.00TiB/4.50TiB operand buffers for 2 of 646 instructions, both
# __triton_gemm fusions of MoELinear/ragged_dot_general (the MoE expert GEMMs).
# Level 0 = no autotune runs at all; XLA picks heuristic configs. Perf hit, correct.
export XLA_FLAGS="--xla_gpu_autotune_level=0 ${XLA_FLAGS:-}"

export NCCL_DEBUG=WARN
export NCCL_NET_GDR_LEVEL=0
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_TIMEOUT=60

# levanter #wall-SIGKILL: a mid-save kill orphans a metadata-only ghost step in the
# staging mirror that poisons resume discovery -> self-clean before every (re)start.
rm -rf "$F/checkpoints/ota10k_sft_30ba3b_levanter/ckpt-temp" 2>/dev/null || true

srun --kill-on-bad-exit=1 \
  python -m levanter.main.train_lm --config_path "$LEV_CONFIG" ${LEV_EXTRA_ARGS:-}
