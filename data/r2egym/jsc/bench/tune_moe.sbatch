#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --output=/e/project1/transfernetx/lee27/code/snowball/bench/logs/%x_%j.out
#SBATCH --job-name=snowball_moe_tune
#SBATCH -p booster
#SBATCH --account reformo
#SBATCH --gres=gpu:4
#SBATCH --exclude=jpbo-031-[01-48],jpbo-011-[01-48],jpbo-038-38,jpbo-004-46,jpbo-065-17,jpbo-074-22,jpbo-074-40,jpbo-048-41,jpbo-091-05,jpbo-044-0[1-5]
# Tune the Triton fused_moe tile config for Snowball's EP4 local-expert shape on GH200, and benchmark
# default vs H200-copied vs GH200-tuned kernel times at decode-relevant batch sizes.
set -uo pipefail
module load GCC/14.3.0
module load nvidia-compilers/25.9-CUDA-13
C=/e/project1/transfernetx/lee27/code
PY=$C/envs/snowball/bin/python
B=$C/snowball/bench
mkdir -p $B/tuned_gh200
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CACHE=/e/fscratch/reformo/lee27/cache
export VLLM_CACHE_ROOT=$CACHE/vllm XDG_CACHE_HOME=$CACHE/xdg TRITON_CACHE_DIR=$CACHE/triton TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor TMPDIR=$CACHE/tmp
export RAY_TMPDIR=/tmp/ray_$SLURM_JOB_ID; mkdir -p $RAY_TMPDIR
export PYTHONPATH=$C/src/marin_vllm/benchmarks/kernels:${PYTHONPATH:-}   # ray actors import benchmark_moe by module name
export PATH=$C/envs/snowball/bin:$PATH
cd $C/src/marin_vllm/benchmarks/kernels
BS="4 8 16 32 48 64 96 128 256 1024 4096"
echo "node=$(hostname) job=$SLURM_JOB_ID ($(date -Is))"
echo "=== A: default (untuned) config"
$PY $B/tune_moe_gh200.py --batch-size $BS 2>&1 | grep -E "Batch size|Kernel time|Error|Traceback"
echo "=== B: H200 config served as GH200"
VLLM_TUNED_CONFIG_FOLDER=$B/tuned_h200 $PY $B/tune_moe_gh200.py --batch-size $BS 2>&1 | grep -E "Batch size|Kernel time|Error|Traceback"
echo "=== C: tune ($(date -Is))"
$PY $B/tune_moe_gh200.py --tune --save-dir $B/tuned_gh200 2>&1 | grep -vE "^\s*$" | tail -60
ls -la $B/tuned_gh200
echo "=== D: GH200-tuned config ($(date -Is))"
VLLM_TUNED_CONFIG_FOLDER=$B/tuned_gh200 $PY $B/tune_moe_gh200.py --batch-size $BS 2>&1 | grep -E "Batch size|Kernel time|Error|Traceback"
echo "=== DONE $(date -Is)"
