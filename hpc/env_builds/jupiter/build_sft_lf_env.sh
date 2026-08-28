#!/bin/bash
# Build $F/envs/sft-lf — rl-fa freeze clone + LLaMA-Factory (submodule, editable)
# + deepspeed/liger/trl for LF SFT on GH200. Oracle: build_rl_fa_env.sh.
set -uxo pipefail
F=/e/fscratch/reformo/lee27
export UV_CACHE_DIR=$F/cache/uv PIP_CACHE_DIR=$F/cache/pip
export UV_PYTHON_INSTALL_DIR=$F/cache/uv-python TMPDIR=$F/cache/tmp
export XDG_CACHE_HOME=$F/cache/xdg
UV=$HOME/.local/bin/uv
ENV=$F/envs/sft-lf
PY=$ENV/bin/python
DCFT=$F/repos/OpenThoughts-Agent

$UV venv $ENV --python 3.12 --clear || exit 10
$UV pip install -p $PY --no-deps -r $F/envs/rl-fa.freeze.in \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match || exit 20
W=flash_attn-2.8.3+cu128torch2.11-cp312-cp312-manylinux_2_34_aarch64.whl
[ -f $F/cache/tmp/$W ] || curl -sL -o $F/cache/tmp/$W \
  https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.22/$W || exit 50
$UV pip install -p $PY --no-deps $F/cache/tmp/$W || exit 55
$UV pip install -p $PY --no-deps -e $DCFT/sft/llamafactory || exit 60
$UV pip install -p $PY --no-deps deepspeed liger-kernel trl || exit 65
$UV pip install -p $PY ninja py-cpuinfo hjson msgpack matplotlib fire uvicorn sse-starlette tyro || exit 70

echo "=== import smoke (login node, no GPU) ==="
export DISABLE_VERSION_CHECK=1
$PY - <<PYEOF
import torch; print("torch", torch.__version__, "cuda", torch.version.cuda)
import flash_attn; print("flash_attn", flash_attn.__version__)
import transformers; print("transformers", transformers.__version__)
import deepspeed; print("deepspeed", deepspeed.__version__)
import liger_kernel; print("liger OK")
import trl; print("trl", trl.__version__)
import llamafactory; print("llamafactory", llamafactory.__version__ if hasattr(llamafactory,"__version__") else "OK")
from llamafactory.train.tuner import run_exp; print("run_exp import OK")
PYEOF
echo "BUILD_DONE rc=$?"
