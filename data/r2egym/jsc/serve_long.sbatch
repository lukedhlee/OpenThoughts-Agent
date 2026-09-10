#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --output=/e/project1/transfernetx/lee27/code/snowball/logs/%x_%j.out
#SBATCH --job-name=snowball_serve_long
#SBATCH -p booster
#SBATCH --account reformo
#SBATCH --gres=gpu:4
#SBATCH --exclude=jpbo-031-[01-48],jpbo-011-[01-48],jpbo-038-38,jpbo-004-46,jpbo-065-17,jpbo-074-22,jpbo-074-40,jpbo-048-41,jpbo-091-05,jpbo-044-0[1-5]
set -uo pipefail
module load GCC/14.3.0
module load nvidia-compilers/25.9-CUDA-13
C=/e/project1/transfernetx/lee27/code
PY=${SNOWBALL_PY:-$C/envs/snowball/bin/python}
M=/e/fscratch/reformo/lee27/models/snowball-s3-nemotron-terminal-step1888
D=$C/snowball; LOGS=$D/logs; mkdir -p $LOGS
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
CACHE=/e/fscratch/reformo/lee27/cache; mkdir -p $CACHE/vllm $CACHE/xdg $CACHE/triton $CACHE/inductor $CACHE/flashinfer $CACHE/tmp
export VLLM_CACHE_ROOT=$CACHE/vllm XDG_CACHE_HOME=$CACHE/xdg TRITON_CACHE_DIR=$CACHE/triton TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor FLASHINFER_WORKSPACE_BASE=$CACHE/flashinfer TMPDIR=$CACHE/tmp
echo "node=$(hostname) job=$SLURM_JOB_ID"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
$PY -c "import vllm,torch;print('vllm',vllm.__version__,'torch',torch.__version__,torch.version.cuda)"
SPID=""
run_serve() {
  local label=$1; shift
  echo "=== SERVE $label: $* ($(date -Is))"
  $PY -m vllm.entrypoints.openai.api_server --model $M --served-model-name snowball --port 8000 \
      --max-model-len 65536 --hf-overrides '{"max_position_embeddings": 65536}' --gpu-memory-utilization 0.90 --chat-template $M/chat_template.jinja \
      "$@" > $LOGS/serve_${label}_$SLURM_JOB_ID.log 2>&1 &
  SPID=$!
  for i in $(seq 1 150); do
    curl -sf localhost:8000/health >/dev/null && { echo "UP after $((i*10))s"; return 0; }
    kill -0 $SPID 2>/dev/null || { echo "server died (see serve_${label} log)"; tail -40 $LOGS/serve_${label}_$SLURM_JOB_ID.log; return 1; }
    sleep 10
  done
  echo "timeout waiting for server"; kill $SPID 2>/dev/null; return 1
}
stop_serve() { [ -n "$SPID" ] && { kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; }; pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null; sleep 20; SPID=""; }
if run_serve dp4ep --tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel; then
  grep -m2 -iE "max_model_len|max_position" $LOGS/serve_dp4ep_$SLURM_JOB_ID.log || true
  for pad in 0 20000 28000 40000 56000; do
    echo "=== PAD $pad ($(date -Is))"
    $PY $D/smoke_client.py --label pad$pad --pad-tokens $pad --max-tokens 256 --conc 4 2>&1 | tee $LOGS/client_long_pad${pad}_$SLURM_JOB_ID.log
  done
fi
stop_serve
echo "=== DONE $(date -Is)"
