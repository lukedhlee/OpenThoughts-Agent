#!/bin/bash
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --output=/e/project1/transfernetx/lee27/code/snowball/logs/%x_%j.out
#SBATCH --job-name=snowball_serve_ab
#SBATCH -p booster
#SBATCH --account reformo
#SBATCH --gres=gpu:4
#SBATCH --exclude=jpbo-031-[01-48],jpbo-011-[01-48],jpbo-038-38,jpbo-004-46,jpbo-065-17,jpbo-074-22,jpbo-074-40,jpbo-048-41,jpbo-091-05,jpbo-044-0[1-5]
# Attention-backend A/B for Snowball on one GH200 node: same layout, same prompts, same concurrency sweep,
# one server per backend. LAYOUT_ARGS and BACKENDS are overridable at sbatch time via --export.
set -uo pipefail
module load GCC/14.3.0
module load nvidia-compilers/25.9-CUDA-13
C=/e/project1/transfernetx/lee27/code
PY=${SNOWBALL_PY:-$C/envs/snowball/bin/python}
M=/e/fscratch/reformo/lee27/models/snowball-s3-nemotron-terminal-step1888
D=$C/snowball; LOGS=$D/logs; mkdir -p $LOGS
LAYOUT=${LAYOUT:-dp4ep}
case $LAYOUT in
  dp4ep) LAYOUT_ARGS="--tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel";;
  tp4ep) LAYOUT_ARGS="--tensor-parallel-size 4 --enable-expert-parallel";;
  *) echo "unknown LAYOUT=$LAYOUT"; exit 2;;
esac
BACKENDS=${BACKENDS:-"FLASH_ATTN TRITON_ATTN FLASHINFER"}
CONC=${CONC:-"16,64,128"}
PAD_TOKENS=${PAD_TOKENS:-0}; MAXTOK=${MAXTOK:-512}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
CACHE=/e/fscratch/reformo/lee27/cache; mkdir -p $CACHE/vllm $CACHE/xdg $CACHE/triton $CACHE/inductor $CACHE/flashinfer $CACHE/tmp
export VLLM_CACHE_ROOT=$CACHE/vllm XDG_CACHE_HOME=$CACHE/xdg TRITON_CACHE_DIR=$CACHE/triton TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor FLASHINFER_WORKSPACE_BASE=$CACHE/flashinfer TMPDIR=$CACHE/tmp
echo "node=$(hostname) job=$SLURM_JOB_ID layout=$LAYOUT args='$LAYOUT_ARGS' backends='$BACKENDS' conc=$CONC"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
$PY -c "import vllm,torch;print('vllm',vllm.__version__,'torch',torch.__version__,torch.version.cuda)"
SPID=""
run_serve() {
  local label=$1; shift
  echo "=== SERVE $label: VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-auto} $* ($(date -Is))"
  $PY -m vllm.entrypoints.openai.api_server --model $M --served-model-name snowball --port 8000 \
      --max-model-len 32768 --gpu-memory-utilization 0.90 --chat-template $M/chat_template.jinja \
      "$@" > $LOGS/serve_${label}_$SLURM_JOB_ID.log 2>&1 &
  SPID=$!
  for i in $(seq 1 180); do
    curl -sf localhost:8000/health >/dev/null && { echo "UP after $((i*10))s"; return 0; }
    kill -0 $SPID 2>/dev/null || { echo "server died (see serve_${label} log)"; grep -i -m5 "error\|Traceback" $LOGS/serve_${label}_$SLURM_JOB_ID.log | cut -c1-200; return 1; }
    sleep 10
  done
  echo "timeout waiting for server"; kill $SPID 2>/dev/null; return 1
}
stop_serve() { [ -n "$SPID" ] && { kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; }; pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null; sleep 20; SPID=""; }
for be in $BACKENDS; do
  export VLLM_ATTENTION_BACKEND=$be
  label="ab_${LAYOUT}_${be,,}_p${PAD_TOKENS}"
  if run_serve $label $LAYOUT_ARGS; then
    grep -m2 -i -E "attention backend|MoE backend" $LOGS/serve_${label}_$SLURM_JOB_ID.log | cut -c1-200 || true
    $PY $D/smoke_client.py --label $label --conc $CONC --pad-tokens $PAD_TOKENS --max-tokens $MAXTOK 2>&1 | tee $LOGS/client_${label}_$SLURM_JOB_ID.log
  fi
  stop_serve
done
echo "=== AB SUMMARY ==="; grep -h "^RESULT" $LOGS/client_ab_*_$SLURM_JOB_ID.log 2>/dev/null
echo "=== DONE $(date -Is)"
