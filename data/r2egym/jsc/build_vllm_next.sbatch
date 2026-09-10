#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --output=/e/project1/transfernetx/lee27/code/logs/%x_%j.out
#SBATCH --job-name=vllm_next_build
#SBATCH -p booster
#SBATCH --account reformo
#SBATCH --gres=gpu:4
#SBATCH --exclude=jpbo-031-[01-48],jpbo-011-[01-48],jpbo-038-38,jpbo-004-46,jpbo-065-17,jpbo-074-22,jpbo-074-40,jpbo-048-41,jpbo-091-05,jpbo-044-0[1-5]
# COMPUTE-NODE build of marin-community/vllm @ main into envs/snowball-vnext (copy of envs/snowball).
# Everything that needs internet was staged on the login node; the recipe logs to logs/build_marin_vllm_next.log.
echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date -Is)"
bash /e/project1/transfernetx/lee27/code/build_marin_vllm_next.sh
echo "rc=$? end=$(date -Is)"
