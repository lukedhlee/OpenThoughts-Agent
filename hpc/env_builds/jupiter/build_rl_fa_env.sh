#!/bin/bash
# Build $F/envs/rl-fa — clone of validated rl-megatron stack (torch 2.11.0+cu128, vllm 0.22.0)
# + flash_attn 2.8.3 exact-tag wheel (mjun0812 v0.9.22 cu128torch2.11-cp312 aarch64).
# Oracle: Ben's otagent combo (ENVIRONMENT_MAP §2a) + rl-megatron freeze.
set -uxo pipefail
F=/e/fscratch/reformo/lee27
export UV_CACHE_DIR=$F/cache/uv PIP_CACHE_DIR=$F/cache/pip
export UV_PYTHON_INSTALL_DIR=$F/cache/uv-python TMPDIR=$F/cache/tmp
export XDG_CACHE_HOME=$F/cache/xdg
mkdir -p $F/cache/{uv,pip,uv-python,tmp,xdg} $F/envs
UV=$HOME/.local/bin/uv
ENV=$F/envs/rl-fa
PY=$ENV/bin/python

$UV venv $ENV --python 3.12 --clear || exit 10

# freeze minus editables/vcs/compile-risk (installed explicitly after)
grep -vE "^(skyrl-gym|skyrl-train|torchtitan|transformer_engine_torch|flash_attn|dynamic-semaphore|dynamic_semaphore|harbor)==" \
  $F/envs/rl-megatron.freeze \
  | grep -viE "^(conda|conda-libmamba-solver|conda-package-handling|conda_package_streaming|libmambapy|menuinst|pycosat)==" \
  > $F/envs/rl-fa.freeze.in

# --no-deps: the freeze is already a complete closed set; the oracle env holds
# pin pairs a resolver rejects (fsspec 2026.6.0 vs datasets 5.0.0), so replicate
# it exactly instead of re-resolving.
$UV pip install -p $PY --no-deps -r $F/envs/rl-fa.freeze.in \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match || exit 20

$UV pip install -p $PY --no-deps \
  "torchtitan @ git+https://github.com/pytorch/torchtitan@a1fdd7e43694bbfeff5d6ad8ac738c067bb90d41" \
  "dynamic-semaphore @ git+https://github.com/penfever/dynamic-semaphore@4d5f49f290889f4826219b241e1aa42d6466163e" \
  "harbor @ git+https://github.com/marin-community/harbor.git@725fc069555013da7ae7f895dd3658a2c2452d55" || exit 30

$UV pip install -p $PY --no-deps \
  -e $F/repos/MarinSkyRL/skyrl-train -e $F/repos/MarinSkyRL/skyrl-gym || exit 40

# transformer_engine_torch: best-effort (FSDP2 path does not import TE); never fail the build on it
$UV pip install -p $PY transformer_engine_torch==2.11.0 || echo "TE_SKIPPED"

W=flash_attn-2.8.3+cu128torch2.11-cp312-cp312-manylinux_2_34_aarch64.whl
curl -sL -o $F/cache/tmp/$W \
  https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.22/$W || exit 50
$UV pip install -p $PY --no-deps $F/cache/tmp/$W || exit 55

echo "=== import smoke (login node, no GPU) ==="
$PY - <<'PYEOF'
import torch; print("torch", torch.__version__, "cuda", torch.version.cuda)
import flash_attn, flash_attn_2_cuda
from flash_attn import flash_attn_func
print("flash_attn", flash_attn.__version__, "OK")
import vllm; print("vllm", vllm.__version__)
import vllm._C; print("vllm._C OK")
from torch.distributed.fsdp import fully_shard; print("fully_shard OK")
from torch.distributed.tensor.placement_types import _StridedShard; print("_StridedShard OK")
import torchtitan; print("torchtitan OK")
import skyrl_train, skyrl_gym; print("skyrl editables OK")
import ray; print("ray", ray.__version__)
PYEOF
echo "BUILD_DONE rc=$?"
