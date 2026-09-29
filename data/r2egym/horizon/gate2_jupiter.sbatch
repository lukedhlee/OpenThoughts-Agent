#!/bin/bash
# gate2_jupiter.sbatch — Horizon port gate 2, configs (b) and (c): score the fixed gate-2 set with prompt logprobs on one
# Jupiter GH200 node, first DP4 + expert parallel (b), then TP4 + expert parallel (c), one server after the other.
# Ported from data/tb2/jupiter/serve_snowball.sbatch (modules, env, geometry, trained sampler) with the same two deltas
# as gate2_horizon.sbatch: no EAGLE-3 draft, --gpu-memory-utilization 0.80. The TP4 server differs from DP4 only in
# --tensor-parallel-size 4 --data-parallel-size 1. ~0.6 node-hours; exits when both score files are written.
#
#   sbatch data/r2egym/horizon/gate2_jupiter.sbatch        (from a detached OTA worktree at the pushed SHA)
#SBATCH --job-name=sb_gate2
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH -p booster
#SBATCH --account laionize
#SBATCH --output=/e/data1/mmlaion/lee27/horizon_gates/logs/%x_%j.out
#SBATCH --exclude=jpbo-031-[01-48],jpbo-011-[01-48],jpbo-038-38,jpbo-004-46,jpbo-065-17,jpbo-074-22,jpbo-074-40,jpbo-048-41,jpbo-091-05,jpbo-044-0[1-5]
set -uo pipefail
module load GCC/14.3.0
module load nvidia-compilers/25.9-CUDA-13
C=/e/project1/transfernetx/lee27/code
OTA=${SLURM_SUBMIT_DIR:-$C/ota-horizon-gates}
PY=$C/envs/snowball-v2/bin/python
export PYTHONPATH=$C/src/marin_vllm_eagle3${PYTHONPATH:+:$PYTHONPATH}   # the EAGLE-3 overlay shadows the venv's vllm
MODEL=${MODEL:-/e/fscratch/reformo/lee27/models/snowball-s3-nemotron-terminal-step1888}   # sha256 == laion/ HF snapshot
SERVED=snowball; MAXLEN=${MAXLEN:-65536}
G=/e/data1/mmlaion/lee27/horizon_gates; SET=${SET:-$G/gate2_set.jsonl}; mkdir -p $G/logs
LAYOUTS=${LAYOUTS:-"dp4ep tp4ep"}
GEN='{"temperature":1.0,"top_p":1.0,"top_k":-1}'   # the RL arm's sampler (POLICY=trained)
[ -f "$MODEL/config.json" ] || { echo "no model at $MODEL"; exit 1; }
[ -f "$SET" ] || { echo "no gate-2 set at $SET"; exit 1; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
CACHE=/e/fscratch/reformo/lee27/cache; mkdir -p $CACHE/vllm $CACHE/xdg $CACHE/triton $CACHE/inductor $CACHE/flashinfer $CACHE/tmp
export VLLM_CACHE_ROOT=$CACHE/vllm XDG_CACHE_HOME=$CACHE/xdg TRITON_CACHE_DIR=$CACHE/triton TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor FLASHINFER_WORKSPACE_BASE=$CACHE/flashinfer TMPDIR=$CACHE/tmp
export PATH=$C/envs/snowball-v2/bin:$PATH   # ninja for FlashInfer JIT
echo "node=$(hostname) job=$SLURM_JOB_ID model=$MODEL set=$SET ota=$(git -C $OTA rev-parse --short HEAD) layouts=$LAYOUTS"
sha256sum $SET
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
$PY -c "import vllm,torch;print('vllm',vllm.__file__,vllm.__version__,'torch',torch.__version__,torch.version.cuda)"
CHAT=(); [ -f "$MODEL/chat_template.jinja" ] && CHAT=(--chat-template "$MODEL/chat_template.jinja")
SPID=
trap 'kill $SPID 2>/dev/null; pkill -u $USER -f vllm.entrypoints.openai.api_server 2>/dev/null' TERM EXIT
for L in $LAYOUTS; do
  case $L in
    dp4ep) PAR=(--tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel);;
    tp4ep) PAR=(--tensor-parallel-size 4 --data-parallel-size 1 --enable-expert-parallel);;
    *) echo "unknown layout $L"; exit 2;;
  esac
  SLOG=$G/logs/gate2_server_${L}.$SLURM_JOB_ID.log
  $PY -m vllm.entrypoints.openai.api_server --model "$MODEL" --served-model-name "$SERVED" --port 8000 \
      "${PAR[@]}" \
      --max-model-len $MAXLEN --hf-overrides "{\"max_position_embeddings\": $MAXLEN, \"max_seq_len\": $MAXLEN}" \
      --max-num-seqs 32 --gpu-memory-utilization 0.80 \
      --override-generation-config "$GEN" "${CHAT[@]}" > $SLOG 2>&1 &
  SPID=$!
  for i in $(seq 1 180); do
    curl -sf localhost:8000/health >/dev/null && { echo "$L UP after $((i*10))s"; break; }
    kill -0 $SPID 2>/dev/null || { echo "$L server died"; tail -60 $SLOG; exit 1; }
    sleep 10
  done
  curl -sf localhost:8000/health >/dev/null || { echo "$L timeout waiting for the server"; tail -40 $SLOG; exit 1; }
  grep -m3 -i -E "attention backend|Using .* backend|max_num_batched_tokens" $SLOG || true
  $PY $OTA/data/r2egym/horizon/gate2_score.py score --set $SET --out $G/gate2_jupiter_${L}.$SLURM_JOB_ID.jsonl \
      --label jupiter_$L --conc 4 --reps 2 || { echo "$L scoring failed"; tail -40 $SLOG; exit 1; }
  kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
  pkill -u $USER -f vllm.entrypoints.openai.api_server 2>/dev/null; sleep 20   # free the GPUs before the next layout
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
done
echo "=== DONE $(date -Is)"
