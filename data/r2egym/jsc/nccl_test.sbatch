#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --output=/e/project1/transfernetx/lee27/code/snowball/logs/%x_%j.out
#SBATCH --job-name=snowball_nccl_test
#SBATCH -p booster
#SBATCH --account reformo
#SBATCH --exclude=jpbo-031-[01-48],jpbo-011-[01-48],jpbo-038-38,jpbo-004-46,jpbo-065-17,jpbo-074-22,jpbo-074-40,jpbo-048-41,jpbo-091-05,jpbo-044-0[1-5]
# Variants of the 2-node NCCL scatter test in the snowball venv: baseline / PXN disabled / GDR default.
set -uo pipefail
C=/e/project1/transfernetx/lee27/code; L=$C/snowball/logs
module load GCC/14.3.0; module load nvidia-compilers/25.9-CUDA-13
export NCCL_SOCKET_IFNAME=ib0 NCCL_SOCKET_FAMILY=AF_INET NCCL_IB_TIMEOUT=23 NCCL_BLOCKING_WAIT=1
export NCCL_CUMEM_ENABLE=0 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,ENV
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1); export MASTER_PORT=29517
echo "nodes=$SLURM_JOB_NODELIST master=$MASTER_ADDR $(date -Is)"; df -h /dev/shm | tail -1
source $C/OpenThoughts-Agent/hpc/dotenv/jupiter.env 2>/dev/null || true
source $C/OpenThoughts-Agent/hpc/dotenv/jupiter.local.env
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH//envs\/rl-fa/envs\/snowball}"
PY=$C/envs/snowball/bin/python
run_variant() {
  local name=$1; shift
  LOG=$L/nccl_${name}_$SLURM_JOB_ID.log
  echo "=== VARIANT=$name env: $* start=$(date -Is) ==="
  env "$@" timeout -k 20 300 srun --export=ALL --gpu-bind=none --cpu-bind=none --kill-on-bad-exit=1 $PY -u $C/snowball/nccl_test.py > $LOG 2>&1
  rc=$?
  grep -h "^\[r0\]\|^\[r4\]" $LOG | cut -c1-160
  grep -h -i "NCCL WARN" $LOG | sed "s/^.*NCCL WARN/NCCL WARN/" | sort | uniq -c | head -n 4 | cut -c1-200
  echo "=== VARIANT=$name rc=$rc DONE_ranks=$(grep -c "] DONE" $LOG) end=$(date -Is) ==="
  export MASTER_PORT=$((MASTER_PORT+1)); rm -f /dev/shm/nccl_probe_r* 2>/dev/null
}
run_variant baseline      NCCL_NET_GDR_LEVEL=0
run_variant pxn_off       NCCL_NET_GDR_LEVEL=0 NCCL_PXN_DISABLE=1
run_variant gdr_default   NCCL_PXN_DISABLE=0
run_variant pxn_off_gdr   NCCL_PXN_DISABLE=1
echo "=== DONE $(date -Is)"
