#!/bin/bash
# serve_node.sh student|teacher — one vLLM server on this GH200 node for the relay pilot (run by serve_relay.sbatch
# through srun, one node each). The two command lines are the eval/bench serves, copied by diff:
#   student = data/tb2/jupiter/serve_snowball.sbatch as the 09-21 thinking-on TB2 evals ran it (serves 1961314 /
#             1967259 / 1974067: POLICY=trained = temperature 1.0 / top_p 1.0 / top_k -1, the adapted EAGLE-3 draft
#             probe_adapt_20260911/checkpoints/3, the model's own chat_template.jinja, no reasoning parser, TP1 x DP4
#             x EP, 65,536 context, 32 seqs per replica). skip_special_tokens=false comes per request from harbor.
#   teacher = data/mini_swe_host/serve_qwen38.sbatch (= data/relay/bench/qwen_bench.sbatch's pick: TP1 x DP4, MTP 2,
#             64k (TEACHER_MAXLEN=131072 for relay from 21:15 PT; Qwen3.8's native length is 262k), prefix caching,
#             96 seqs per replica, the relay-extra torchvision side dir) with --reasoning-parser
#             qwen3; no sampler override (Qwen3.8's generation_config).
set -uo pipefail
ROLE=${1:?student|teacher}
# Start-up retry: vLLM's DP workers race for torch.distributed ports and a start can die with EADDRINUSE (run 3's first
# submission, job 2033358, lost both teacher nodes this way). A server that dies BEFORE its /health answers is started
# again, up to START_TRIES times; once healthy, it is never restarted (a mid-run death ends the job, as before).
START_TRIES=${START_TRIES:-6}
serve() {  # serve <vllm args...>   (PORT env, default 8000)
  local try rc pid port=${PORT:-8000}
  for try in $(seq 1 $START_TRIES); do
    "$@" & pid=$!
    while kill -0 $pid 2>/dev/null; do
      curl -sf --max-time 5 localhost:$port/health >/dev/null && { echo "serve_node: port $port healthy on try $try"; wait $pid; exit $?; }
      sleep 10
    done
    wait $pid; rc=$?
    echo "serve_node: port $port server exited with $rc before it was healthy (try $try of $START_TRIES)"
    sleep 20
  done
  exit 1
}
# Site knobs: the defaults are Jupiter's; data/relay/horizon/serve_env.sh sets Horizon's (SERVE_MODULES="" loads none).
for m in ${SERVE_MODULES-GCC/14.3.0 nvidia-compilers/25.9-CUDA-13}; do module load $m; done
C=${SERVE_CODE:-/e/project1/transfernetx/lee27/code}
VENV=${SERVE_VENV:-$C/envs/snowball-v2}
PY=$VENV/bin/python
export PYTHONPATH=${SERVE_OVERLAY:-$C/src/marin_vllm_eagle3}${PYTHONPATH:+:$PYTHONPATH}   # the EAGLE-3 overlay shadows the venv's vllm
export PATH=$VENV/bin:$PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
CACHE=${SERVE_CACHE:-/e/fscratch/reformo/lee27/cache}; mkdir -p $CACHE/vllm $CACHE/xdg $CACHE/triton $CACHE/inductor $CACHE/flashinfer $CACHE/tmp
export VLLM_CACHE_ROOT=$CACHE/vllm XDG_CACHE_HOME=$CACHE/xdg TRITON_CACHE_DIR=$CACHE/triton TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor FLASHINFER_WORKSPACE_BASE=$CACHE/flashinfer TMPDIR=$CACHE/tmp
echo "serve_node $ROLE on $(hostname) job=$SLURM_JOB_ID step=$SLURM_STEP_ID ($(date -Is))"
nvidia-smi --query-gpu=name,memory.total --format=csv
case $ROLE in
  student)
    MODEL=${STUDENT_MODEL:-/e/data1/mmlaion/lee27/models/grug-datakit-sft-20260921}
    DRAFT=${STUDENT_DRAFT:-/e/data1/mmlaion/lee27/eagle3/probe_adapt_20260911/checkpoints/3}
    [ -f "$MODEL/config.json" ] || { echo "no model at $MODEL"; exit 1; }
    [ -f "$DRAFT/model.safetensors" ] || { echo "no draft at $DRAFT"; exit 1; }
    CHAT=(); [ -f "$MODEL/chat_template.jinja" ] && CHAT=(--chat-template "$MODEL/chat_template.jinja")
    GEN='{"temperature":1.0,"top_p":1.0,"top_k":-1}'
    serve $PY -m vllm.entrypoints.openai.api_server --model "$MODEL" --served-model-name snowball --port 8000 \
      --tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel \
      --max-model-len 65536 --hf-overrides '{"max_position_embeddings": 65536, "max_seq_len": 65536}' \
      --max-num-seqs 32 --gpu-memory-utilization 0.90 \
      --speculative-config "{\"method\":\"eagle3\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":3}" \
      --override-generation-config "$GEN" "${CHAT[@]}";;
  teacher)
    export PYTHONPATH=$PYTHONPATH:${RELAY_EXTRA:-$C/envs/relay-extra}   # torchvision for vLLM's qwen3_5 import (qwen_bench.sbatch)
    MODEL=${TEACHER_MODEL:-/e/data1/mmlaion/lee27/models/Qwen3.8-27B}
    [ -f "$MODEL/config.json" ] || { echo "no model at $MODEL"; exit 1; }
    $PY -c "import torchvision, vllm.model_executor.models.qwen3_5; print('torchvision', torchvision.__version__, 'qwen3_5 import ok')" || { echo "qwen3_5 import failed"; exit 1; }
    QARGS=(--model "$MODEL" --served-model-name qwen38 --tensor-parallel-size 1
      --max-model-len ${TEACHER_MAXLEN:-65536} --gpu-memory-utilization 0.90 --max-num-seqs 96
      --enable-prefix-caching --enable-chunked-prefill --no-enable-log-requests
      --speculative-config '{"method":"mtp","num_speculative_tokens":2}' --reasoning-parser qwen3 ${TEACHER_EXTRA_ARGS:-})
    if [ "${PER_GPU:-0}" = 1 ]; then
      # one server per GPU (TP1, DP1, ports 8000-8003): the router balances episodes across all of them itself; a DP4
      # server spread its load unevenly over its engines (2026-09-26 diagnosis: busy engines at 85-95 % KV with 10-19
      # waiting while siblings idled)
      for g in 0 1 2 3; do
        ( export CUDA_VISIBLE_DEVICES=$g PORT=$((8000 + g)); VLLM_PORT=$((29500 + 100 * g)) serve $PY -m vllm.entrypoints.openai.api_server "${QARGS[@]}" --port $((8000 + g)) ) &
      done
      wait; exit 1
    fi
    serve $PY -m vllm.entrypoints.openai.api_server "${QARGS[@]}" --port 8000 --data-parallel-size 4;;
  *) echo "ROLE must be student or teacher"; exit 2;;
esac
