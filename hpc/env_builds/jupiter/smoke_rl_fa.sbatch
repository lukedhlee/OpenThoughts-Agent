#!/bin/bash
#SBATCH --account reformo
#SBATCH -p booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:15:00
#SBATCH --job-name=rl_fa_smoke
#SBATCH --exclude=jpbo-031-[01-48],jpbo-011-[01-48],jpbo-038-38,jpbo-004-46,jpbo-065-17,jpbo-074-22,jpbo-074-40,jpbo-048-41,jpbo-091-05,jpbo-044-0[1-5]
#SBATCH --output=/e/fscratch/reformo/lee27/envs/rl_fa_smoke_%j.out
module load GCC/14.3.0 nvidia-compilers/25.9-CUDA-13 2>/dev/null || module load GCC NVIDIA 2>/dev/null || true
bash /e/fscratch/reformo/lee27/envs/smoke_rl_fa_gpu.sh
