#!/bin/bash
# serve_snowball.sbatch — serve one Snowball checkpoint on one Jupiter GH200 node for a Terminal-Bench 2 run.
#
# The marin vLLM fork with the adapted EAGLE-3 draft (Luke's 2026-09-12 policy: every job uses the draft), the arm's
# geometry (TP1 x DP4 x EP4, 65k context), and marin's eval-policy sampler (#8151: temperature 0.7, top_p 1.0,
# repetition_penalty 1.1) applied server-side, so harbor's terminus-2 sends no sampling parameters of its own.
# Writes the endpoint URL to $ENDPOINT_DIR/<jobid> for run_tb2.sh on the login node, then holds the server up
# until the wall ends. One server per harbor run; the run reads the endpoint file.
#
#   MODEL=<hf-format dir> POLICY=marin|trained sbatch serve_snowball.sbatch
#
#SBATCH --job-name=tb2_serve
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH -p booster
#SBATCH --account laionize   # 2026-09-17: reformo is 18 % left and queues ~6 h on fairshare; laionize starts in minutes
#SBATCH --output=/e/fscratch/reformo/lee27/experiments/tb2/logs/%x_%j.out
#SBATCH --exclude=jpbo-031-[01-48],jpbo-011-[01-48],jpbo-038-38,jpbo-004-46,jpbo-065-17,jpbo-074-22,jpbo-074-40,jpbo-048-41,jpbo-091-05,jpbo-044-0[1-5]
set -uo pipefail
module load GCC/14.3.0
module load nvidia-compilers/25.9-CUDA-13
C=/e/project1/transfernetx/lee27/code
PY=$C/envs/snowball-v2/bin/python
export PYTHONPATH=$C/src/marin_vllm_eagle3${PYTHONPATH:+:$PYTHONPATH}   # the EAGLE-3 overlay shadows the venv's vllm
MODEL=${MODEL:-/e/fscratch/reformo/lee27/models/snowball-s3-nemotron-terminal-step1888}
DRAFT=${DRAFT:-/e/data1/mmlaion/lee27/eagle3/probe_adapt_20260911/checkpoints/3}
POLICY=${POLICY:-trained}   # v0.1 = no sampler override = vLLM defaults (temp 1.0); 'marin' is the August #8151 sampler, not the current policy
SERVED=${SERVED:-snowball}
E=/e/fscratch/reformo/lee27/experiments/tb2; ENDPOINT_DIR=$E/endpoints; LOGS=$E/logs; mkdir -p $ENDPOINT_DIR $LOGS
case "$POLICY" in
  marin)   GEN='{"temperature":0.7,"top_p":1.0,"repetition_penalty":1.1}';;   # marin eval policy #8151, shared decoding rules
  trained) GEN='{"temperature":1.0,"top_p":1.0,"top_k":-1}';;                 # the RL arm's sampler
  *) echo "POLICY must be marin or trained"; exit 2;;
esac
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
echo "node=$(hostname) job=$SLURM_JOB_ID model=$MODEL policy=$POLICY gen=$GEN"
nvidia-smi --query-gpu=name,memory.total --format=csv
$PY -c "import vllm,torch;print('vllm',vllm.__file__,vllm.__version__,'torch',torch.__version__)"
CHAT=(); [ -f "$MODEL/chat_template.jinja" ] && CHAT=(--chat-template "$MODEL/chat_template.jinja")
SLOG=$LOGS/serve_$SLURM_JOB_ID.log
# NSYS=1 (2026-09-21, ai_memory/notes/jupiter_nsys_r2egym_profiling.md): wrap the server in Nsight Systems, one warmed-up
# capture of NSYS_DURATION s starting NSYS_DELAY s after launch (default 600 s: server up ~5 min, first trials ~2 min later);
# --kill=none keeps the server alive after the capture; report -> /e/data1/mmlaion/lee27/nsys/serve_<job>.nsys-rep.
NSYS=${NSYS:-0}; NSYSW=()
if [ "$NSYS" = 1 ]; then
  module load Nsight-Systems/2025.5.1 || { echo "no Nsight-Systems module"; exit 1; }
  NSYS_DIR=/e/data1/mmlaion/lee27/nsys; mkdir -p $NSYS_DIR
  NSYSW=(nsys profile -t cuda,nvtx,osrt --sample=none --cpuctxsw=none --delay "${NSYS_DELAY:-600}" --duration "${NSYS_DURATION:-60}" --kill=none -f true -o "$NSYS_DIR/serve_$SLURM_JOB_ID")
  echo "NSYS capture: delay ${NSYS_DELAY:-600}s duration ${NSYS_DURATION:-60}s -> $NSYS_DIR/serve_$SLURM_JOB_ID.nsys-rep"
fi
"${NSYSW[@]}" $PY -m vllm.entrypoints.openai.api_server --model "$MODEL" --served-model-name "$SERVED" --port 8000 \
    --tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel \
    --max-model-len 65536 --hf-overrides '{"max_position_embeddings": 65536, "max_seq_len": 65536}' \
    --max-num-seqs 32 --gpu-memory-utilization 0.90 \
    --speculative-config "{\"method\":\"eagle3\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":3}" \
    --override-generation-config "$GEN" "${CHAT[@]}" > $SLOG 2>&1 &
SPID=$!
for i in $(seq 1 180); do
  curl -sf localhost:8000/health >/dev/null && { echo "UP after $((i*10))s"; break; }
  kill -0 $SPID 2>/dev/null || { echo "server died"; tail -60 $SLOG; exit 1; }
  sleep 10
done
curl -sf localhost:8000/health >/dev/null || { echo "timeout waiting for the server"; kill $SPID; exit 1; }
grep -m1 -E "Initializing a V1 LLM engine" $SLOG || true
# one real completion before anyone points harbor at it: the served name, the think markers surviving, usage counts
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Print hello in bash.\"}],\"max_tokens\":400,\"skip_special_tokens\":false}" \
  | $PY -c 'import json,sys; r=json.load(sys.stdin); m=r["choices"][0]["message"]; print("SMOKE finish=%s usage=%s content=%r" % (r["choices"][0]["finish_reason"], r.get("usage"), (m.get("content") or "")[:300]))' | tee -a $SLOG
URL="http://$(hostname -s).jupiter.internal:8000/v1"
echo "$URL" > $ENDPOINT_DIR/$SLURM_JOB_ID
printf 'model=%s\npolicy=%s\ngen=%s\ndraft=%s\nserved=%s\n' "$MODEL" "$POLICY" "$GEN" "$DRAFT" "$SERVED" > $ENDPOINT_DIR/$SLURM_JOB_ID.meta
echo "ENDPOINT $URL (file $ENDPOINT_DIR/$SLURM_JOB_ID)"
trap 'echo "wall/term: stopping server"; kill $SPID 2>/dev/null; pkill -u $USER -f vllm.entrypoints.openai.api_server 2>/dev/null; rm -f $ENDPOINT_DIR/$SLURM_JOB_ID' TERM EXIT
wait $SPID
