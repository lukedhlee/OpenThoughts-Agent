#!/bin/bash
# gate2_horizon.sbatch — Horizon port gate 2, config (a): score the fixed gate-2 set with prompt logprobs on one Horizon
# node in the serve_smoke.sbatch geometry (DP4 + expert parallel, 64k, trained sampler). Same as serve_smoke.sbatch
# except: no EAGLE-3 draft (scoring must not touch the spec-decode path; all three gate-2 configs run draft-off) and
# --gpu-memory-utilization 0.80 (headroom for the 8192 x 128k fp32 prompt-logprob tensors; KV size only, numerics
# unchanged). Both deltas are identical in gate2_jupiter.sbatch. ~0.35 node-hours; exits when the scores are written.
#
#   sbatch data/r2egym/horizon/gate2_horizon.sbatch        (from ~/snowball/ota; set at $S/horizon_gates/gate2_set.jsonl)
#SBATCH -J sb_gate2
#SBATCH -A CCR24067
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 00:45:00
#SBATCH -o /scratch/11584/lukedhlee/horizon_gates/logs/%x.%j.out
set -uo pipefail
ROOT=${ROOT:-$HOME/snowball}; S=/scratch/11584/$USER
OTA=${SLURM_SUBMIT_DIR:-$ROOT/ota}
PY=$ROOT/envs/snowball/bin/python
export PYTHONPATH=$ROOT/src/marin_vllm_eagle3${PYTHONPATH:+:$PYTHONPATH}   # the EAGLE-3 overlay shadows the venv's vllm
export PATH=$ROOT/envs/snowball/bin:$PATH                                     # ninja for FlashInfer JIT
snap() { ls -d $S/hf_hub/models--laion--$1/snapshots/* | head -1; }
MODEL=${MODEL:-$(snap snowball-67b-a2b-sft-s3-nemotron-terminal-step1888)}
SERVED=snowball; MAXLEN=${MAXLEN:-65536}
G=$S/horizon_gates; SET=${SET:-$G/gate2_set.jsonl}; mkdir -p $G/logs
GEN='{"temperature":1.0,"top_p":1.0,"top_k":-1}'   # the RL arm's sampler (POLICY=trained)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLREDUCE_USE_SYMM_MEM=0
export CC=gcc CXX=g++   # TACC's login env sets CC=nvc, which Triton's JIT rejects (-Wno-psabi)
C=$S/cache; mkdir -p $C/vllm $C/xdg $C/triton $C/inductor $C/flashinfer $C/tmp
export VLLM_CACHE_ROOT=$C/vllm XDG_CACHE_HOME=$C/xdg TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor FLASHINFER_WORKSPACE_BASE=$C/flashinfer TMPDIR=$C/tmp
[ -f "$SET" ] || { echo "no gate-2 set at $SET"; exit 1; }
echo "node=$(hostname) job=$SLURM_JOB_ID model=$MODEL set=$SET ota=$(git -C $OTA rev-parse --short HEAD)"
sha256sum $SET
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
$PY -c "import vllm,torch;print('vllm',vllm.__file__,vllm.__version__,'torch',torch.__version__,torch.version.cuda)"
CHAT=(); [ -f "$MODEL/chat_template.jinja" ] && CHAT=(--chat-template "$MODEL/chat_template.jinja")
SLOG=$G/logs/gate2_server.$SLURM_JOB_ID.log
$PY -m vllm.entrypoints.openai.api_server --model "$MODEL" --served-model-name "$SERVED" --port 8000 \
    --tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel \
    --max-model-len $MAXLEN --hf-overrides "{\"max_position_embeddings\": $MAXLEN, \"max_seq_len\": $MAXLEN}" \
    --max-num-seqs 32 --gpu-memory-utilization 0.80 \
    --override-generation-config "$GEN" "${CHAT[@]}" > $SLOG 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null; pkill -u $USER -f vllm.entrypoints.openai.api_server 2>/dev/null' TERM EXIT
for i in $(seq 1 180); do
  curl -sf localhost:8000/health >/dev/null && { echo "UP after $((i*10))s"; break; }
  kill -0 $SPID 2>/dev/null || { echo "server died"; tail -60 $SLOG; exit 1; }
  sleep 10
done
curl -sf localhost:8000/health >/dev/null || { echo "timeout waiting for the server"; tail -40 $SLOG; exit 1; }
grep -m3 -i -E "attention backend|Using .* backend|max_num_batched_tokens" $SLOG || true
$PY $OTA/data/r2egym/horizon/gate2_score.py score --set $SET --out $G/gate2_horizon_dp4ep.$SLURM_JOB_ID.jsonl \
    --label horizon_dp4ep --conc 4 --reps 2
echo "=== DONE $(date -Is)"
