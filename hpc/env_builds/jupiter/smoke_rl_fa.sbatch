#!/bin/bash
#SBATCH --account reformo
#SBATCH -p booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:15:00
#SBATCH --job-name=rl_fa_smoke
#SBATCH --output=rl_fa_smoke_%j.out
# GPU smoke for the rl-fa venv. Submit from the OpenThoughts-Agent repo root
# with the dotenv sourced (ENV_DIR / DCFT_RL_ENV select the venv):
#   source hpc/dotenv/jupiter.env && sbatch hpc/env_builds/jupiter/smoke_rl_fa.sbatch
module load GCC/14.3.0 nvidia-compilers/25.9-CUDA-13 2>/dev/null || module load GCC NVIDIA 2>/dev/null || true
bash "${DCFT:-$SLURM_SUBMIT_DIR}/hpc/env_builds/jupiter/smoke_rl_fa_gpu.sh"
