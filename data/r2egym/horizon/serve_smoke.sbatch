#!/bin/bash
# serve_smoke.sbatch — serve Snowball on one Horizon node (4x GB200) with the production serving config of
# data/tb2/jupiter/serve_snowball.sbatch (DP4 + expert parallel, 64k context, EAGLE-3 draft k=3, trained sampler),
# then one real completion and smoke_client.py (coherence + tokens/s). ~1 node-hour.
#SBATCH -J sb_serve_smoke
#SBATCH -A CCR24067
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -o %x.%j.out
set -uo pipefail
ROOT=${ROOT:-$HOME/snowball}; S=/scratch/11584/$USER
PY=$ROOT/envs/snowball/bin/python
export PYTHONPATH=$ROOT/src/marin_vllm_eagle3${PYTHONPATH:+:$PYTHONPATH}   # the EAGLE-3 overlay shadows the venv's vllm
export PATH=$ROOT/envs/snowball/bin:$PATH                                     # ninja for FlashInfer JIT
snap() { ls -d $S/hf_hub/models--laion--$1/snapshots/* | head -1; }
MODEL=${MODEL:-$(snap snowball-67b-a2b-sft-s3-nemotron-terminal-step1888)}
DRAFT=${DRAFT:-$(snap snowball-64k-eagle3-draft-r2egym)}
SERVED=snowball; MAXLEN=${MAXLEN:-65536}
GEN='{"temperature":1.0,"top_p":1.0,"top_k":-1}'   # the RL arm's sampler (POLICY=trained)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLREDUCE_USE_SYMM_MEM=0
export CC=gcc CXX=g++   # TACC's login env sets CC=nvc, which Triton's JIT rejects (-Wno-psabi)
C=$S/cache; mkdir -p $C/vllm $C/xdg $C/triton $C/inductor $C/flashinfer $C/tmp
export VLLM_CACHE_ROOT=$C/vllm XDG_CACHE_HOME=$C/xdg TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor FLASHINFER_WORKSPACE_BASE=$C/flashinfer TMPDIR=$C/tmp
echo "node=$(hostname) job=$SLURM_JOB_ID model=$MODEL draft=$DRAFT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
$PY -c "import vllm,torch;print('vllm',vllm.__file__,vllm.__version__,'torch',torch.__version__,torch.version.cuda)"
CHAT=(); [ -f "$MODEL/chat_template.jinja" ] && CHAT=(--chat-template "$MODEL/chat_template.jinja")
SLOG=$PWD/serve_smoke_server.$SLURM_JOB_ID.log
$PY -m vllm.entrypoints.openai.api_server --model "$MODEL" --served-model-name "$SERVED" --port 8000 \
    --tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel \
    --max-model-len $MAXLEN --hf-overrides "{\"max_position_embeddings\": $MAXLEN, \"max_seq_len\": $MAXLEN}" \
    --max-num-seqs 32 --gpu-memory-utilization 0.90 \
    --speculative-config "{\"method\":\"eagle3\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":3}" \
    --override-generation-config "$GEN" "${CHAT[@]}" > $SLOG 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null; pkill -u $USER -f vllm.entrypoints.openai.api_server 2>/dev/null' TERM EXIT
for i in $(seq 1 180); do
  curl -sf localhost:8000/health >/dev/null && { echo "UP after $((i*10))s"; break; }
  kill -0 $SPID 2>/dev/null || { echo "server died"; tail -60 $SLOG; exit 1; }
  sleep 10
done
curl -sf localhost:8000/health >/dev/null || { echo "timeout waiting for the server"; tail -40 $SLOG; exit 1; }
grep -m3 -i -E "attention backend|Using .* backend|speculative|eagle" $SLOG || true
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Print hello in bash.\"}],\"max_tokens\":400,\"skip_special_tokens\":false}" \
  | $PY -c 'import json,sys; r=json.load(sys.stdin); m=r["choices"][0]["message"]; print("SMOKE finish=%s usage=%s content=%r" % (r["choices"][0]["finish_reason"], r.get("usage"), (m.get("content") or "")[:300]))'
$PY $ROOT/snowball/smoke_client.py --label horizon_dp4ep_eagle3 --conc 16
grep -E "SpecDecoding metrics|Mean acceptance length" $SLOG | tail -2
echo "=== DONE $(date -Is)"
