#!/bin/bash
# draft_bench.sbatch — EAGLE-3 draft on vs off for one SFT checkpoint at RL-like load, on ONE Horizon node (4x GB200).
# Serves the model twice back to back on the same node with the eval/RL student serve (data/relay/pilot/serve_node.sh
# 'student': TP1 x DP4 + expert parallel, 65,536 context, 32 seqs per replica, gpu-mem 0.90, sampler 1.0/1.0/-1, the
# model's chat_template.jinja, draft k=3 or no --speculative-config), and at each concurrency in CONCS runs
# bench_client.py (closed-loop replay of the model's own SWE-bench eval sessions, 120 s warmup + 240 s window).
# Env setup is gate2_horizon.sbatch's with compile caches node-local under /tmp (SERVE_CACHE_LOCAL in serve_node.sh).
# Deltas from serve_node.sh: VLLM_SERVER_DEV_MODE=1 (only for POST /reset_prefix_cache between points).
#
#   MODEL=<hf dir> SESS=<sessions.jsonl> LABEL=h9 CONCS="32 128" ORDER="on off" sbatch -t 01:05:00 draft_bench.sbatch
#SBATCH -J draft_bench
#SBATCH -A CCR24067
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 01:05:00
#SBATCH -o /scratch/11584/lukedhlee/experiments/rl/draft_bench/logs/%x.%j.out
set -uo pipefail
ROOT=${ROOT:-$HOME/snowball}; S=/scratch/11584/$USER
HERE=${HERE:-$HOME/OpenThoughts-Agent-rl/data/r2egym/horizon/rl/draft_bench}
O=$S/experiments/rl/draft_bench; mkdir -p $O/logs
PY=$ROOT/envs/snowball/bin/python
export PYTHONPATH=$ROOT/src/marin_vllm_eagle3${PYTHONPATH:+:$PYTHONPATH}   # the EAGLE-3 overlay shadows the venv's vllm
export PATH=$ROOT/envs/snowball/bin:$PATH                                     # ninja for FlashInfer JIT
MODEL=${MODEL:?MODEL=<hf dir>}; SESS=${SESS:?SESS=<sessions.jsonl>}; LABEL=${LABEL:?LABEL}
DRAFT=${DRAFT:-$(ls -d $S/hf_hub/models--laion--snowball-64k-eagle3-draft-r2egym/snapshots/* | head -1)}
CONCS=${CONCS:-"32 128"}; ORDER=${ORDER:-"on off"}; WARMUP=${WARMUP:-120}; WINDOW=${WINDOW:-240}
GEN='{"temperature":1.0,"top_p":1.0,"top_k":-1}'
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLREDUCE_USE_SYMM_MEM=0 VLLM_SERVER_DEV_MODE=1
export CC=gcc CXX=g++   # TACC's login env sets CC=nvc, which Triton's JIT rejects (-Wno-psabi)
C=/tmp/$USER-dbcache-$SLURM_JOB_ID; df -h /tmp | tail -1
mkdir -p $C/vllm $C/xdg $C/triton $C/inductor $C/flashinfer $C/tmp
export VLLM_CACHE_ROOT=$C/vllm XDG_CACHE_HOME=$C/xdg TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor FLASHINFER_WORKSPACE_BASE=$C/flashinfer TMPDIR=$C/tmp
echo "node=$(hostname) job=$SLURM_JOB_ID label=$LABEL model=$MODEL draft=$DRAFT concs=$CONCS order=$ORDER $(date -Is)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
[ -f "$MODEL/config.json" ] && [ -f "$DRAFT/model.safetensors" ] && [ -f "$SESS" ] || { echo "missing model/draft/sessions"; exit 1; }
CHAT=(); [ -f "$MODEL/chat_template.jinja" ] && CHAT=(--chat-template "$MODEL/chat_template.jinja")
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,power.draw,memory.used --format=csv -l 10 > $O/logs/gpu_$LABEL.$SLURM_JOB_ID.csv 2>&1 &
SMI=$!
SPID=
stop_server() {
  [ -n "$SPID" ] || return 0
  kill -TERM -- -$SPID 2>/dev/null
  for i in $(seq 1 30); do kill -0 -- -$SPID 2>/dev/null || break; sleep 2; done
  kill -KILL -- -$SPID 2>/dev/null; sleep 5; SPID=
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo
}
trap 'stop_server; kill $SMI 2>/dev/null; rm -rf $C' EXIT
trap 'exit 143' TERM
RES=$O/results_$LABEL.$SLURM_JOB_ID.jsonl
port=8000
for cfg in $ORDER; do
  SPEC=(); [ "$cfg" = on ] && SPEC=(--speculative-config "{\"method\":\"eagle3\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":3}")
  SLOG=$O/logs/server_${LABEL}_$cfg.$SLURM_JOB_ID.log
  up=0
  for try in 1 2; do   # one retry: DP workers can race for torch.distributed ports (EADDRINUSE) at start
    t0=$(date +%s)
    setsid $PY -m vllm.entrypoints.openai.api_server --model "$MODEL" --served-model-name snowball --port $port \
      --tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel \
      --max-model-len 65536 --hf-overrides '{"max_position_embeddings": 65536, "max_seq_len": 65536}' \
      --max-num-seqs 32 --gpu-memory-utilization 0.90 \
      "${SPEC[@]}" --override-generation-config "$GEN" "${CHAT[@]}" >> $SLOG 2>&1 &
    SPID=$!
    for i in $(seq 1 120); do
      curl -sf --max-time 5 localhost:$port/health >/dev/null && { up=1; break; }
      kill -0 $SPID 2>/dev/null || break
      sleep 10
    done
    [ $up = 1 ] && break
    echo "server $cfg not healthy on try $try"; tail -30 $SLOG; stop_server; port=$((port + 1))
  done
  [ $up = 1 ] || { echo "FAILED: server $cfg never came up"; exit 1; }
  echo "=== $LABEL draft=$cfg UP on :$port after $(( $(date +%s) - t0 ))s ($(date -Is))"
  grep -m2 -E "max_num_batched_tokens|SpeculativeConfig\(" $SLOG | cut -c1-300
  for conc in $CONCS; do
    curl -s -X POST localhost:$port/reset_prefix_cache >/dev/null; sleep 2
    $PY $HERE/bench_client.py --url http://localhost:$port --sessions $SESS --conc $conc --warmup $WARMUP \
        --window $WINDOW --label ${LABEL}_c${conc}_$cfg --out $RES 2>&1 | tail -5
  done
  stop_server; port=$((port + 1))
done
echo "=== DONE $(date -Is)"
