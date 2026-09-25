#!/bin/bash
# serve_qwen38.sbatch — serve Qwen3.8-27B on one Jupiter GH200 node for a mini-swe-agent-host smoke.
#
# The serve is the relay bench's best layout (data/relay/bench/qwen_bench.sbatch @ 76e12202: TP1 x DP4, MTP 2, 64k,
# prefix caching, the relay-extra torchvision side dir) plus one flag: --reasoning-parser qwen3. Without it the reply's
# thinking stays in `content` ("...</think>answer"), and Qwen3.8's template, which re-renders every prior turn as
# "<think>\n{reasoning_content}\n</think>\n\n{content}", would re-feed it as an empty think block followed by the raw
# reasoning. With it, harbor's Chat (interleaved_thinking: true) re-sends reasoning_content and the template renders
# the trained format. Writes the endpoint URL to $ENDPOINT_DIR/<jobid> after one real completion, like
# serve_snowball.sbatch, and holds the server until the wall or scancel.
#
#   sbatch -A transfernetx --time=00:55:00 serve_qwen38.sbatch
#   TOOL_PARSER=qwen3_coder sbatch ...   # tool mode: Qwen3.8's template writes <tool_call><function=…><parameter=…>,
#                                        # the qwen3_coder parser's format; adds --enable-auto-tool-choice
#
#SBATCH --job-name=msa_qwen38
#SBATCH --time=00:55:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH -p booster
#SBATCH --account transfernetx
#SBATCH --output=/e/fscratch/reformo/lee27/experiments/mini_swe_host/logs/%x_%j.out
#SBATCH --exclude=jpbo-031-[01-48],jpbo-011-[01-48],jpbo-038-38,jpbo-004-46,jpbo-065-17,jpbo-074-22,jpbo-074-40,jpbo-048-41,jpbo-091-05,jpbo-044-0[1-5]
set -uo pipefail
module load GCC/14.3.0
module load nvidia-compilers/25.9-CUDA-13
C=/e/project1/transfernetx/lee27/code
PY=$C/envs/snowball-v2/bin/python
export PYTHONPATH=$C/src/marin_vllm_eagle3${PYTHONPATH:+:$PYTHONPATH}
export PYTHONPATH=$PYTHONPATH:$C/envs/relay-extra   # torchvision for vLLM's qwen3_5 import (qwen_bench.sbatch)
export PATH=$C/envs/snowball-v2/bin:$PATH
MODEL=${MODEL:-/e/data1/mmlaion/lee27/models/Qwen3.8-27B}
SERVED=qwen38
E=/e/fscratch/reformo/lee27/experiments/mini_swe_host; ENDPOINT_DIR=$E/endpoints; LOGS=$E/logs; mkdir -p $ENDPOINT_DIR $LOGS
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
CACHE=/e/fscratch/reformo/lee27/cache; mkdir -p $CACHE/vllm $CACHE/xdg $CACHE/triton $CACHE/inductor $CACHE/flashinfer $CACHE/tmp
export VLLM_CACHE_ROOT=$CACHE/vllm XDG_CACHE_HOME=$CACHE/xdg TRITON_CACHE_DIR=$CACHE/triton TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor FLASHINFER_WORKSPACE_BASE=$CACHE/flashinfer TMPDIR=$CACHE/tmp
[ -f "$MODEL/config.json" ] || { echo "no model at $MODEL"; exit 1; }
TOOLS=(); [ -n "${TOOL_PARSER:-}" ] && TOOLS=(--enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER")
# Strict tool calling (the default) constrains "auto" tool calls with the parser's structural-tag grammar; on this build
# the grammar rejects MTP draft tokens ("grammar rejected tokens ... Terminating request", HTTP 500 on 28 of 64 calls,
# job 2024000). Parse-only tool calling keeps the served text unconstrained, like the text-mode runs.
export VLLM_ENFORCE_STRICT_TOOL_CALLING=0
echo "node=$(hostname) job=$SLURM_JOB_ID model=$MODEL tool_parser=${TOOL_PARSER:-none} ($(date -Is))"
$PY -c "import torchvision, vllm.model_executor.models.qwen3_5; print('torchvision', torchvision.__version__, 'qwen3_5 import ok')" || { echo "qwen3_5 import failed"; exit 1; }
SLOG=$LOGS/serve_$SLURM_JOB_ID.log
$PY -m vllm.entrypoints.openai.api_server --model "$MODEL" --served-model-name $SERVED --port 8000 \
    --tensor-parallel-size 1 --data-parallel-size 4 \
    --max-model-len 65536 --gpu-memory-utilization 0.90 --max-num-seqs 96 \
    --enable-prefix-caching --enable-chunked-prefill --no-enable-log-requests \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
    --reasoning-parser qwen3 "${TOOLS[@]}" > $SLOG 2>&1 &
SPID=$!
for i in $(seq 1 120); do
  curl -sf localhost:8000/health >/dev/null && { echo "UP after $((i*10))s"; break; }
  kill -0 $SPID 2>/dev/null || { echo "server died"; grep -iE -m8 "error|Traceback|ValueError" $SLOG | cut -c1-240; exit 1; }
  sleep 10
done
curl -sf localhost:8000/health >/dev/null || { echo "timeout waiting for the server"; kill $SPID; exit 1; }
# one real completion before harbor points at it: the served name, reasoning split off, usage counts
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Print hello in bash. Be brief.\"}],\"max_tokens\":2000}" \
  | $PY -c 'import json,sys; r=json.load(sys.stdin); m=r["choices"][0]["message"]; print("SMOKE finish=%s usage=%s reasoning=%r content=%r" % (r["choices"][0]["finish_reason"], r.get("usage"), (m.get("reasoning_content") or m.get("reasoning") or "")[:120], (m.get("content") or "")[:200]))' | tee -a $SLOG
URL="http://$(hostname -s).jupiter.internal:8000/v1"
echo "$URL" > $ENDPOINT_DIR/$SLURM_JOB_ID
printf 'model=%s\nserved=%s\n' "$MODEL" "$SERVED" > $ENDPOINT_DIR/$SLURM_JOB_ID.meta
echo "ENDPOINT $URL (file $ENDPOINT_DIR/$SLURM_JOB_ID)"
trap 'echo "wall/term: stopping server"; kill $SPID 2>/dev/null; pkill -u $USER -f vllm.entrypoints.openai.api_server 2>/dev/null; rm -f $ENDPOINT_DIR/$SLURM_JOB_ID' TERM EXIT
wait $SPID
