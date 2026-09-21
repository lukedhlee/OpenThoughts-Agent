#!/bin/bash
# gepa_serve_node.sh — ONE vLLM server on ONE Jupiter node, for the standing GEPA serve job.
#
# This is data/tb2/jupiter/serve_snowball.sbatch's body, adapted by diff. Same marin vLLM fork with the adapted
# EAGLE-3 draft (Luke 2026-09-12: every job uses the draft), same geometry (TP1 x DP4 x EP4, 65k context), same
# caches, same health wait, same smoke completion. Four deltas, all because this one is an srun task inside a
# multi-node allocation instead of a whole job:
#   1. the endpoint file is per TASK, $ENDPOINT_DIR/$SLURM_JOB_ID.$SLURM_PROCID, so the runner sees N of them;
#   2. MAX_NUM_SEQS is a knob (default 32 -- the TB2-proven value; raising it doubles KV demand at 65k context,
#      so do not raise it without a bench);
#   3. the trap removes only this node's endpoint file and kills only this server, never the whole job (the
#      idle watchdog in gepa_serve.sbatch owns the job's life);
#   4. no POLICY branch: GEPA always serves the arm's sampler, because the loop compares prompts, not samplers.
#
# Not run by hand -- gepa_serve.sbatch sruns it.
set -uo pipefail
module load GCC/14.3.0
module load nvidia-compilers/25.9-CUDA-13
C=/e/project1/transfernetx/lee27/code
PY=$C/envs/snowball-v2/bin/python
export PYTHONPATH=$C/src/marin_vllm_eagle3${PYTHONPATH:+:$PYTHONPATH}   # the EAGLE-3 overlay shadows the venv's vllm
MODEL=${MODEL:-/e/fscratch/reformo/lee27/models/snowball-s3-nemotron-terminal-step1888}
DRAFT=${DRAFT:-/e/data1/mmlaion/lee27/eagle3/probe_adapt_20260911/checkpoints/3}
SERVED=${SERVED:-snowball}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}
ENDPOINT_DIR=${GEPA_ENDPOINTS:-/e/fscratch/reformo/lee27/experiments/gepa/endpoints}
LOGS=${GEPA_LOGS:-/e/fscratch/reformo/lee27/experiments/gepa/logs}
mkdir -p "$ENDPOINT_DIR" "$LOGS"
GEN='{"temperature":1.0,"top_p":1.0,"top_k":-1}'   # the RL arm's sampler, as the screens and the probes served it
[ -f "$MODEL/config.json" ] || { echo "no model at $MODEL"; exit 1; }
[ -f "$DRAFT/model.safetensors" ] || { echo "no draft at $DRAFT"; exit 1; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
CACHE=/e/fscratch/reformo/lee27/cache; mkdir -p $CACHE/vllm $CACHE/xdg $CACHE/triton $CACHE/inductor $CACHE/flashinfer $CACHE/tmp
export VLLM_CACHE_ROOT=$CACHE/vllm XDG_CACHE_HOME=$CACHE/xdg TRITON_CACHE_DIR=$CACHE/triton TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor FLASHINFER_WORKSPACE_BASE=$CACHE/flashinfer TMPDIR=$CACHE/tmp
export PATH=$C/envs/snowball-v2/bin:$PATH   # ninja for FlashInfer JIT
RANK=${SLURM_PROCID:-0}
EPF=$ENDPOINT_DIR/$SLURM_JOB_ID.$RANK
echo "node=$(hostname) job=$SLURM_JOB_ID rank=$RANK model=$MODEL seqs=$MAX_NUM_SEQS gen=$GEN"
CHAT=(); [ -f "$MODEL/chat_template.jinja" ] && CHAT=(--chat-template "$MODEL/chat_template.jinja")
SLOG=$LOGS/serve_${SLURM_JOB_ID}_$RANK.log
$PY -m vllm.entrypoints.openai.api_server --model "$MODEL" --served-model-name "$SERVED" --port 8000 \
    --tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel \
    --max-model-len "$MAX_MODEL_LEN" --hf-overrides "{\"max_position_embeddings\": $MAX_MODEL_LEN, \"max_seq_len\": $MAX_MODEL_LEN}" \
    --max-num-seqs "$MAX_NUM_SEQS" --gpu-memory-utilization 0.90 \
    --speculative-config "{\"method\":\"eagle3\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":3}" \
    --override-generation-config "$GEN" "${CHAT[@]}" > $SLOG 2>&1 &
SPID=$!
trap 'echo "rank '"$RANK"': stopping server"; kill $SPID 2>/dev/null; rm -f '"$EPF"'' TERM EXIT
for i in $(seq 1 180); do
  curl -sf localhost:8000/health >/dev/null && { echo "rank $RANK UP after $((i*10))s"; break; }
  kill -0 $SPID 2>/dev/null || { echo "rank $RANK server died"; tail -60 $SLOG; exit 1; }
  sleep 10
done
curl -sf localhost:8000/health >/dev/null || { echo "rank $RANK timeout waiting for the server"; exit 1; }
# one real completion before the runner points harbor at it: served name, think markers surviving, usage counts
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Print hello in bash.\"}],\"max_tokens\":400,\"skip_special_tokens\":false}" \
  | $PY -c 'import json,sys; r=json.load(sys.stdin); m=r["choices"][0]["message"]; print("SMOKE finish=%s usage=%s content=%r" % (r["choices"][0]["finish_reason"], r.get("usage"), (m.get("content") or "")[:200]))' | tee -a $SLOG
URL="http://$(hostname -s).jupiter.internal:8000/v1"
echo "$URL" > "$EPF"
echo "ENDPOINT rank $RANK $URL (file $EPF)"
wait $SPID
