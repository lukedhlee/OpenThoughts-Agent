#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --output=/e/fscratch/reformo/lee27/experiments/p2o/logs/%x_%j.out
#SBATCH --job-name=p2o_prescreen
#SBATCH -p booster
#SBATCH --account reformo
#SBATCH --gres=gpu:4
#SBATCH --exclude=jpbo-031-[01-48],jpbo-011-[01-48],jpbo-038-38,jpbo-004-46,jpbo-065-17,jpbo-074-22,jpbo-074-40,jpbo-048-41,jpbo-091-05,jpbo-044-0[1-5]
# P2O shift scoring on one GH200 node: the 09-10 prefill-cap bench's proven standalone server (DP4 + EP, 65k, prefix
# caching, chunked prefill) + score_shift.py as an HTTP client. MODE=prescreen (default) or MODE=wave (TRACES=<glob>).
# prompt_logprobs materialises chunk x vocab fp32 logits in the sampler: 16k chunks OOM'd GPU 3 (6.6 GiB ask, 3 GiB free,
# job 1772690), so chunks are 4k, KV budget 0.70, 16 seqs, no async scheduling.
set -uo pipefail
module load GCC/14.3.0
module load nvidia-compilers/25.9-CUDA-13
C=/e/project1/transfernetx/lee27/code; E=/e/fscratch/reformo/lee27/experiments
M=${MODEL:-/e/fscratch/reformo/lee27/models/snowball-s3-nemotron-terminal-step1888}   # MODEL=<exported checkpoint dir> scores that checkpoint
PY=$C/envs/snowball/bin/python
MODE=${MODE:-prescreen}; OUT=${OUT:-$E/p2o/$MODE}; TRACES=${TRACES:-}
mkdir -p $OUT $E/p2o/logs
export PATH=$C/envs/snowball/bin:$PATH
export VLLM_HOST_IP=127.0.0.1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ALLREDUCE_USE_SYMM_MEM=0 TORCH_CUDNN_SDPA_ENABLED=0 VLLM_MQ_MAX_CHUNKS=240 VLLM_USE_FLASHINFER_SAMPLER=0
CACHE=/e/data1/mmlaion/lee27/experiments/snowball_ep_layout_20260909/cache
export VLLM_CACHE_ROOT=$CACHE/vllm XDG_CACHE_HOME=$CACHE/xdg TRITON_CACHE_DIR=$CACHE/triton TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor FLASHINFER_WORKSPACE_BASE=$CACHE/flashinfer TMPDIR=/tmp/p2o_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
pids=(); cleanup() { for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; }; trap cleanup EXIT TERM INT
echo "model=$M mode=$MODE out=$OUT job=$SLURM_JOB_ID node=$(hostname) start=$(date -Is)"
"$PY" -c 'import vllm,torch; print("vllm", vllm.__version__, "torch", torch.__version__)'
CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_PORT=30000 "$PY" -m vllm.entrypoints.openai.api_server \
 --model "$M" --served-model-name snowball --port 8000 \
 --max-model-len 65536 --hf-overrides '{"max_position_embeddings":65536,"max_seq_len":65536}' \
 --gpu-memory-utilization 0.70 --max-num-seqs 16 --max-num-batched-tokens 4096 \
 --max-cudagraph-capture-size 64 --enable-prefix-caching --enable-chunked-prefill \
 --tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel \
 --chat-template "$M/chat_template.jinja" --chat-template-content-format string --no-enable-log-requests \
 > "$OUT/server_$SLURM_JOB_ID.log" 2>&1 &
pids+=("$!")
ready=0
for i in $(seq 1 150); do
 curl -sf http://localhost:8000/health >/dev/null && { ready=1; break; }
 for pid in "${pids[@]}"; do kill -0 "$pid" 2>/dev/null || { tail -60 "$OUT/server_$SLURM_JOB_ID.log"; echo SERVER_DIED; exit 1; }; done
 sleep 5
done
[ "$ready" = 1 ] || { echo STARTUP_TIMEOUT; exit 1; }
echo "SERVER_READY $(date -Is)"
# leave ~8 min for the merge and shutdown inside the 2 h wall
BUDGET=$(( ${WALL_SEC:-7200} - 8*60 - $(( $(date +%s) - $(date -d "$(squeue -h -j $SLURM_JOB_ID -o %S)" +%s 2>/dev/null || date +%s) )) ))
[ "$BUDGET" -gt 600 ] || BUDGET=$(( ${WALL_SEC:-7200} - 15*60 ))
ARGS="--mode $MODE --out $OUT --workers ${WORKERS:-8} --time-budget $BUDGET --tokenizer $M"
[ -n "$TRACES" ] && ARGS="$ARGS --traces $TRACES"
"$PY" $E/p2o/score_shift.py $ARGS > "$OUT/client_$SLURM_JOB_ID.log" 2>&1
tail -5 "$OUT/client_$SLURM_JOB_ID.log"
"$PY" $E/p2o/score_shift.py --mode $MODE --out $OUT --merge
echo "end=$(date -Is)"
