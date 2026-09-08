#!/bin/bash
#SBATCH --job-name=tito_probe
#SBATCH --account=CCR24067
#SBATCH --partition=gh-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#
# Reproduce MarinSkyRL#520 on TACC Vista (GH200, aarch64) and test repair A.
#
# Serves one model on the allocated node and runs scripts/tito_repro/probe.py in
# both transport modes. Vista does NOT gres-track GPUs and misreports RealMemory,
# so this asks for whole nodes with no --gres / --mem (see .claude/ops/tacc/ops.md).
# gh-dev caps at 2h wall and ONE running job per user; use -p gh for longer runs.
#
# Env knobs: MODEL, MAX_MODEL_LEN, TRAJECTORIES, TURNS, MAX_TOKENS, TEMPERATURE,
#            EXTRA_PROBE_ARGS, OUT_DIR.

set -uo pipefail

SCRATCH="${SCRATCH:-/scratch/11584/lukedhlee}"
DCFT="${DCFT:-$SCRATCH/OpenThoughts-Agent-tito}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
TRAJECTORIES="${TRAJECTORIES:-8}"   # first pass is a smoke; re-drive bigger via srun --overlap
TURNS="${TURNS:-6}"
MAX_TOKENS="${MAX_TOKENS:-384}"
TEMPERATURE="${TEMPERATURE:-1.0}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
PORT="${PORT:-8000}"
OUT_DIR="${OUT_DIR:-$SCRATCH/experiments/tito_repro/${SLURM_JOB_ID:-manual}}"

mkdir -p "$OUT_DIR"
echo "[tito] job=${SLURM_JOB_ID:-manual} node=$(hostname) model=$MODEL out=$OUT_DIR"

# --- aarch64 toolchain (matches the fork vLLM wheel build) ---
module purge 2>/dev/null || true
module load gcc/13.2.0 cuda/12.8 2>/dev/null || true

# --- otagent env (penfever's, used READ-ONLY; nothing is installed into it) ---
source "$SCRATCH/miniconda3/etc/profile.d/conda.sh"
conda activate otagent
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PYTHON_BIN="$(which python)"
echo "[tito] python=$PYTHON_BIN"

# --- caches on scratch; /home1 is 23 GB and must stay empty ---
export VLLM_CACHE_ROOT="$SCRATCH/vllm"
export VLLM_CONFIG_ROOT="$SCRATCH/vllm"
export TRITON_CACHE_DIR="/tmp/triton_cache_${USER}_${SLURM_JOB_ID:-manual}"
export FLASHINFER_WORKSPACE_BASE="$SCRATCH/flashinfer_cache"
export TORCH_EXTENSIONS_DIR="$SCRATCH/torch_extensions"
export TORCH_HOME="$SCRATCH/torch_home"
export XDG_CACHE_HOME="$SCRATCH/.cache"
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/.cache/torchinductor"
export HF_HOME="${HF_HOME:-$SCRATCH/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$SCRATCH/hf_home/hub}"
export HF_HUB_OFFLINE=0          # Vista compute nodes have direct egress
export MAX_JOBS=8                # cap ninja during any JIT; 72-way OOM-kills nvcc
export TOKENIZERS_PARALLELISM=false

cleanup() {
    if [ -n "${VLLM_PID:-}" ]; then
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "[tito] starting vLLM"
"$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$MODEL" \
    --port "$PORT" \
    --host 127.0.0.1 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --disable-log-requests \
    > "$OUT_DIR/vllm.log" 2>&1 &
VLLM_PID=$!

HEALTH="http://127.0.0.1:${PORT}/v1/models"
for i in $(seq 1 60); do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[tito] FATAL: vLLM died during startup; tail of vllm.log:"
        tail -40 "$OUT_DIR/vllm.log"
        exit 1
    fi
    if "$PYTHON_BIN" -c "
import urllib.request, sys
try:
    urllib.request.urlopen('$HEALTH', timeout=10); sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
        echo "[tito] vLLM ready after $((i*20))s"
        break
    fi
    echo "[tito] waiting for vLLM ($i/60)"
    sleep 20
done

"$PYTHON_BIN" "$DCFT/scripts/tito_repro/probe.py" \
    --base-url "http://127.0.0.1:${PORT}/v1" \
    --model "$MODEL" \
    --trajectories "$TRAJECTORIES" \
    --turns "$TURNS" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --out "$OUT_DIR/tito_probe.json" \
    ${EXTRA_PROBE_ARGS:-} 2>&1 | tee "$OUT_DIR/probe.log"
RC=${PIPESTATUS[0]}
echo "[tito] first probe exit=$RC"

# Keep the server up for the rest of the wall so further probes can be driven
# into this allocation with `srun --overlap --jobid=$SLURM_JOB_ID` instead of
# paying model load again. Touch $OUT_DIR/STOP to release the node early.
echo "[tito] serving; drive more probes with:"
echo "  srun --overlap --jobid=${SLURM_JOB_ID:-?} --pty bash -lc 'PROBE_ARGS...'"
echo "[tito] READY host=$(hostname) port=$PORT out=$OUT_DIR" > "$OUT_DIR/READY"
while [ ! -f "$OUT_DIR/STOP" ]; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[tito] vLLM exited; stopping"
        break
    fi
    sleep 30
done

echo "[tito] done rc=$RC"
exit "$RC"
