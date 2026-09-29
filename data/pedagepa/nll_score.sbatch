#!/bin/bash
# nll_score.sbatch: serve 09-21 (no speculative draft; prompt logprobs only) on one GH200 node and score rendered SFT rows
# with nll_client.py. ROWS="<jsonl>:<label> ..." OUT=<file> N=<rows per set> sbatch nll_score.sbatch
#SBATCH --job-name=pg_nll
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH -p booster
#SBATCH --account transfernetx
#SBATCH --output=/e/data1/mmlaion/lee27/experiments/pedagepa/logs/%x_%j.out
#SBATCH --exclude=jpbo-031-[01-48],jpbo-011-[01-48],jpbo-038-38,jpbo-004-46,jpbo-065-17,jpbo-074-22,jpbo-074-40,jpbo-048-41,jpbo-091-05,jpbo-044-0[1-5]
set -uo pipefail
module load GCC/14.3.0
module load nvidia-compilers/25.9-CUDA-13
C=/e/project1/transfernetx/lee27/code; PY=$C/envs/snowball-v2/bin/python
export PYTHONPATH=$C/src/marin_vllm_eagle3 PATH=$C/envs/snowball-v2/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLREDUCE_USE_SYMM_MEM=0
CACHE=/e/fscratch/reformo/lee27/cache
export VLLM_CACHE_ROOT=$CACHE/vllm XDG_CACHE_HOME=$CACHE/xdg TRITON_CACHE_DIR=$CACHE/triton TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor FLASHINFER_WORKSPACE_BASE=$CACHE/flashinfer TMPDIR=$CACHE/tmp
MODEL=${MODEL:-/e/data1/mmlaion/lee27/models/grug-datakit-sft-20260921}
HERE=/e/project1/transfernetx/lee27/code/pedagepa/data/pedagepa
echo "pg_nll on $(hostname) job=$SLURM_JOB_ID ($(date -Is)) model=$MODEL rows=$ROWS"
$PY -m vllm.entrypoints.openai.api_server --model $MODEL --served-model-name snowball --port 8000 \
  --tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel \
  --max-model-len 65536 --hf-overrides '{"max_position_embeddings": 65536, "max_seq_len": 65536}' \
  --max-num-seqs 8 --gpu-memory-utilization 0.78 --max-num-batched-tokens 2048 --no-enable-prefix-caching > /e/data1/mmlaion/lee27/experiments/pedagepa/logs/pg_nll_serve_$SLURM_JOB_ID.log 2>&1 &
SP=$!
for i in $(seq 1 180); do curl -sf localhost:8000/health >/dev/null && break; kill -0 $SP 2>/dev/null || { echo "server died"; exit 1; }; sleep 10; done
curl -sf localhost:8000/health >/dev/null || { echo "server not up"; exit 1; }
$PY $HERE/nll_client.py --url http://localhost:8000/v1 --rows $ROWS --n ${N:-120} --out $OUT --workers 6
echo "done $(date -Is)"; kill $SP
