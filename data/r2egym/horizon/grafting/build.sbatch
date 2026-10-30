#!/bin/bash
#SBATCH -A CCR24067
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 01:00:00
#SBATCH -J graft_build
set -euo pipefail
unset LD_PRELOAD
export CC=gcc CXX=g++ OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
: "${GRAFT_CODE:?}" "${GRAFT_ROOT:?}" "${GRAFT_REVISION:?}"
R=/scratch/11584/$USER
srun -N1 -n1 -c16 "$HOME/snowball/envs/snowball/bin/python" \
  "$GRAFT_CODE/data/r2egym/horizon/grafting/build_grafts.py" \
  --donor "$R/models/grug-datakit-sft-20260921" \
  --donor-ft "$R/experiments/cf2-epochs57-20261022/training/fdv/export-step959-hf-bf16" \
  --target "$R/experiments/paired-sft-rl-20261026/mixed/rl/exports/rl_mix_base_1026_step30/model" \
  --output "$GRAFT_ROOT/models" --revision "$GRAFT_REVISION"
