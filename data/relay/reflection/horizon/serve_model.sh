#!/bin/bash
# serve_model.sh refresh|stop student|teacher [MODEL] [VERSION] — this experiment's own vLLM server on the job's first
# node, started and stopped by its pidfile only (never pkill). Run inside job.sbatch, which exports REFLECTION_SERVE_DIR.
#   student  Grug, port 29010, served name cf-student: serve_node.sh's student arguments (TP1 x DP4 x EP, 65,536
#            context, the EAGLE-3 draft the RLVR1 evals used, temperature 1 / top_p 1 / top_k -1, the model's template)
#   teacher  Qwen3.8-27B, port 29011, served name cf-teacher: serve_node.sh's teacher arguments with QWEN_SPEC=0 (no MTP:
#            MTP engines crashed in the original teacher runs), qwen3 reasoning parser, DP4, 65,536 context
# refresh = stop this role's server if one runs, start MODEL, wait until /health and /v1/models answer. The refresh
# adapter (data.relay.reflection.refresh) then checks that /models reports MODEL as the root.
set -euo pipefail
ACTION=${1:?refresh|stop}; ROLE=${2:?student|teacher}; MODEL=${3:-}; VERSION=${4:-0}
D=${REFLECTION_SERVE_DIR:?set by job.sbatch}
OTA=${REFLECTION_OTA:-$(cd "$(dirname "$0")/../../../.." && pwd)}
case $ROLE in student) PORT=29010 NAME=cf-student;; teacher) PORT=29011 NAME=cf-teacher;; *) echo "bad role $ROLE"; exit 2;; esac
PIDF=$D/serve_$ROLE.pid
log() { echo "[$(date -Is)] serve_model $ROLE: $*"; }

stop() {
  [ -f "$PIDF" ] || return 0
  local pg; pg=$(cat "$PIDF")
  if kill -0 -- "-$pg" 2>/dev/null; then
    log "stopping process group $pg"
    kill -TERM -- "-$pg" 2>/dev/null || true
    for _ in $(seq 1 60); do kill -0 -- "-$pg" 2>/dev/null || break; sleep 2; done
    kill -KILL -- "-$pg" 2>/dev/null || true
  fi
  rm -f "$PIDF"
  # The next server needs the GPUs back: wait until no compute process holds them.
  for _ in $(seq 1 60); do
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] && break; sleep 2
  done
  log "stopped; GPU processes left: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)"
}

start() {
  [ -f "$MODEL/config.json" ] || { log "no model at $MODEL"; exit 1; }
  # serve_env.sh: the Horizon venv, EAGLE-3 overlay, relay-extra and node-local compile caches the eval serves use.
  source "$OTA/data/relay/horizon/serve_env.sh"
  local C=$SERVE_CODE PY=$SERVE_VENV/bin/python CACHE=$SERVE_CACHE_LOCAL/$USER-cfr-${SLURM_JOB_ID:-0}
  mkdir -p $CACHE/vllm $CACHE/xdg $CACHE/triton $CACHE/inductor $CACHE/flashinfer $CACHE/tmp
  local -a ARGS
  if [ "$ROLE" = student ]; then
    local draft=${STUDENT_DRAFT}
    [ -f "$draft/model.safetensors" ] || { log "no draft at $draft"; exit 1; }
    ARGS=(--model "$MODEL" --served-model-name $NAME --port $PORT
      --tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel
      --max-model-len 65536 --hf-overrides '{"max_position_embeddings": 65536, "max_seq_len": 65536}'
      --max-num-seqs 32 --gpu-memory-utilization 0.90
      --speculative-config "{\"method\":\"eagle3\",\"model\":\"$draft\",\"num_speculative_tokens\":3}"
      --override-generation-config '{"temperature":1.0,"top_p":1.0,"top_k":-1}')
    [ -f "$MODEL/chat_template.jinja" ] && ARGS+=(--chat-template "$MODEL/chat_template.jinja")
    local extra=""
  else
    ARGS=(--model "$MODEL" --served-model-name $NAME --port $PORT --tensor-parallel-size 1 --data-parallel-size 4
      --max-model-len 65536 --gpu-memory-utilization 0.90 --max-num-seqs 96
      --enable-prefix-caching --enable-chunked-prefill --no-enable-log-requests --reasoning-parser qwen3)
    local extra=":$RELAY_EXTRA"
  fi
  local logf=$D/serve_${ROLE}_v${VERSION}.log
  log "starting $MODEL (version $VERSION) on port $PORT -> $logf"
  # No proxy for the server: it only talks to localhost and its own DP workers.
  setsid env -u HTTPS_PROXY -u https_proxy -u HTTP_PROXY -u http_proxy -u ALL_PROXY -u all_proxy \
    PYTHONPATH="$SERVE_OVERLAY$extra" PATH="$SERVE_VENV/bin:$PATH" CC=gcc CXX=g++ \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 \
    VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
    VLLM_CACHE_ROOT=$CACHE/vllm XDG_CACHE_HOME=$CACHE/xdg TRITON_CACHE_DIR=$CACHE/triton \
    TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor FLASHINFER_WORKSPACE_BASE=$CACHE/flashinfer TMPDIR=$CACHE/tmp \
    "$PY" -m vllm.entrypoints.openai.api_server "${ARGS[@]}" > "$logf" 2>&1 < /dev/null &
  local pid=$!
  echo "$pid" > "$PIDF"   # setsid makes the server its own process-group leader
  for i in $(seq 1 120); do
    kill -0 "$pid" 2>/dev/null || { log "server exited before it was ready; tail:"; tail -20 "$logf"; rm -f "$PIDF"; exit 1; }
    if curl -sf --noproxy '*' --max-time 5 "localhost:$PORT/health" >/dev/null &&
       curl -sf --noproxy '*' --max-time 10 "localhost:$PORT/v1/models" | grep -q "\"$NAME\""; then
      log "ready after $((i * 10)) s: $(curl -sf --noproxy '*' localhost:$PORT/v1/models | head -c 400)"
      return 0
    fi
    sleep 10
  done
  log "not ready after 20 min"; stop; exit 1
}

case $ACTION in
  refresh) [ -n "$MODEL" ] || { log "refresh needs MODEL"; exit 2; }; stop; start;;
  stop) stop;;
  *) echo "bad action $ACTION"; exit 2;;
esac
