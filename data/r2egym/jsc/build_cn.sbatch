#!/bin/bash
#SBATCH --time=03:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --output=/e/project1/transfernetx/lee27/code/snowball/logs/%x_%j.out
#SBATCH --job-name=snowball_build_cn
#SBATCH -p booster
#SBATCH --account reformo
#SBATCH --gres=gpu:4
#SBATCH --exclude=jpbo-031-[01-48],jpbo-011-[01-48],jpbo-038-38,jpbo-004-46,jpbo-065-17,jpbo-074-22,jpbo-074-40,jpbo-048-41,jpbo-091-05,jpbo-044-0[1-5]
# COMPUTE-NODE hedge build of marin-community/vllm @ fa50698a for GH200 (sm_90, aarch64), then the serve
# smoke on the same node's 4 GPUs. No internet here: everything was staged by cn_prep.sh on the login node.
set -uo pipefail
C=/e/project1/transfernetx/lee27/code
F=/e/fscratch/reformo/lee27
ENV=$C/envs/snowball_cn; PY=$ENV/bin/python
SRC=$C/src/marin_vllm_cn
echo "=== BUILD_CN START $(date -Is) node=$(hostname) job=$SLURM_JOB_ID ncpu=$(nproc) mem=$(free -g | awk '/Mem/{print $2}')G ==="
export UV_CACHE_DIR=$F/cache/uv PIP_CACHE_DIR=$F/cache/pip TMPDIR=$F/cache/tmp XDG_CACHE_HOME=$F/cache/xdg UV_LINK_MODE=copy
export PATH="$HOME/.local/bin:$PATH"
module purge 2>/dev/null || true
module load Stages/2026 2>/dev/null || true
module load CUDA/13 2>/dev/null || module load CUDA 2>/dev/null || true
module load GCC/14.3.0 2>/dev/null || true
module load Rust/1.88.0 2>/dev/null || true
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"; export PATH="$CUDA_HOME/bin:$PATH"
echo "nvcc=$(which nvcc) gcc=$(which gcc) cargo=$(which cargo 2>/dev/null || echo none)"; nvcc --version | tail -1
export CARGO_HOME=$F/cache/cargo CARGO_NET_OFFLINE=true
export TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=72 NVCC_THREADS=2
export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0.dev20260804+marin.fa50698a9a30
export VLLM_TARGET_DEVICE=cuda
echo "torch before build:"; $PY -c "import torch; print(torch.__version__, torch.version.cuda)"
cd "$SRC" && git rev-parse HEAD
echo "=== compiling editable vllm (MAX_JOBS=$MAX_JOBS) $(date -Is) ==="
uv pip install -p "$PY" -e . --no-build-isolation --no-deps --offline
rc=$?
echo "=== compile rc=$rc $(date -Is) ==="
$PY -c "import torch; assert torch.version.cuda.startswith('13'), torch.version.cuda" || echo "TORCH_REPLACED"
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
  echo "=== running serve smoke in-job with SNOWBALL_PY=$PY $(date -Is) ==="
  ( module purge 2>/dev/null; SNOWBALL_PY=$PY bash $C/snowball/serve_smoke.sbatch )
fi
echo "=== BUILD_CN END $(date -Is) ==="
