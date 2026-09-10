#!/usr/bin/env bash
# Build marin-community/vllm @ fa50698a9a30 (MarinSkyRL main pin) from source for GH200 (sm_90, aarch64)
# against torch 2.11.0+cu130 in the `snowball` venv. On success, submits the serve smoke.
set -uo pipefail
C=/e/project1/transfernetx/lee27/code
F=/e/fscratch/reformo/lee27
ENV=$C/envs/snowball; PY=$ENV/bin/python
SRC=$C/src/marin_vllm
LOG=$C/logs/build_marin_vllm.log
exec >>"$LOG" 2>&1
echo "=== START $(date -Is) host=$(hostname) ==="
export UV_CACHE_DIR=$F/cache/uv PIP_CACHE_DIR=$F/cache/pip TMPDIR=$F/cache/tmp XDG_CACHE_HOME=$F/cache/xdg UV_LINK_MODE=copy
export PATH="$HOME/.local/bin:$PATH"
for i in $(seq 1 90); do $PY -c "import torch" 2>/dev/null && break; echo "waiting for venv ($i)"; sleep 10; done
module purge 2>/dev/null || true
module load Stages/2026 2>/dev/null || true
module load CUDA/13 2>/dev/null || module load CUDA 2>/dev/null || true
module load GCCcore 2>/dev/null || module load GCC/14.3.0 2>/dev/null || true
module load CMake 2>/dev/null || true
module load Ninja 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"; export PATH="$CUDA_HOME/bin:$PATH"
echo "nvcc=$(which nvcc)"; nvcc --version | tail -2
echo "torch before build:"; $PY -c "import torch; print(torch.__version__, torch.version.cuda)"
export TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=${MAX_JOBS:-16} NVCC_THREADS=2
export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0.dev20260804+marin.fa50698a9a30
export VLLM_TARGET_DEVICE=cuda
cd "$SRC" && git rev-parse HEAD
[ -f use_existing_torch.py ] && $PY use_existing_torch.py || true
uv pip install -p "$PY" packaging ninja cmake psutil setuptools-scm wheel requests
echo "=== compiling editable vllm $(date -Is) ==="
uv pip install -p "$PY" -e . --no-build-isolation
rc=$?
echo "=== compile rc=$rc $(date -Is) ==="
echo "torch after build:"; $PY -c "import torch; print(torch.__version__, torch.version.cuda)"
$PY -c "import torch; assert torch.version.cuda.startswith('13'), torch.version.cuda" || { echo "torch was replaced; reinstalling cu130"; uv pip install -p "$PY" torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130; }
$PY - <<'PY'
import torch, vllm
print("vllm", vllm.__version__, vllm.__file__, "torch", torch.__version__, torch.version.cuda)
from vllm.model_executor.models.registry import ModelRegistry
archs = ModelRegistry.get_supported_archs()
print("GrugMoeForCausalLM registered:", "GrugMoeForCausalLM" in archs)
assert "GrugMoeForCausalLM" in archs
PY
rc2=$?
echo "=== BUILD_RC=$rc smoke_rc=$rc2 $(date -Is) ==="
if [ "$rc" -eq 0 ] && [ "$rc2" -eq 0 ]; then
  sbatch "$C/snowball/serve_smoke.sbatch" && echo "serve smoke submitted"
fi
echo "=== END $(date -Is) ==="
