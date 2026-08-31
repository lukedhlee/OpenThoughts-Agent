#!/bin/bash
# Build the Jupiter (aarch64 GH200) host-venv RL runtime with FlashAttention:
# torch 2.11.0+cu128 + vllm 0.22.0 stack pinned by rl-fa.freeze.in, plus the
# flash_attn 2.8.3 exact-tag wheel (mjun0812 v0.9.22, cu128torch2.11-cp312
# manylinux_2_34_aarch64). rl-fa.freeze.in is the validated rl-megatron freeze
# minus editables / VCS installs / compile-risk packages (installed explicitly
# below) and minus conda's own packages.
#
# Locations come from hpc/dotenv/jupiter.env (SCRATCH, DCFT_SCRATCH, SKYRL_HOME)
# or the caller's environment:
#   ENV_DIR      target venv           (default $SCRATCH/envs/rl-fa — point DCFT_RL_ENV at it)
#   BUILD_CACHE  uv/pip/tmp caches     (default $DCFT_SCRATCH/cache — $HOME is quota-bound)
#   SKYRL_HOME   MarinSkyRL checkout   (skyrl-train / skyrl-gym installed editable)
#   UV           uv binary             (default: `uv` on PATH, else ~/.local/bin/uv)
# Run on a login node (needs internet for the wheel indexes). GPU smoke:
# sbatch hpc/env_builds/jupiter/smoke_rl_fa.sbatch
set -uxo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
if [[ -z "${SCRATCH:-}" && -f "$REPO_ROOT/hpc/dotenv/jupiter.env" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/hpc/dotenv/jupiter.env"
fi
: "${SCRATCH:?SCRATCH unset — source hpc/dotenv/jupiter.env first}"
: "${SKYRL_HOME:?SKYRL_HOME unset — source hpc/dotenv/jupiter.env first}"
# Harbor must match what MarinSkyRL main pins (uv.lock: `harbor` git source + the harbor-config
# release wheel of the same sha) — the HarborTrajectoryRunner imports harbor_config.errors, which
# older harbor commits (e.g. 725fc069) do not have.
: "${HARBOR_REF:=df866b3086f386221e6e04ecf4b09e3cc9ffe44e}"
: "${ENV_DIR:=$SCRATCH/envs/rl-fa}"
: "${BUILD_CACHE:=${DCFT_SCRATCH:-$SCRATCH/otagent}/cache}"
if [[ -z "${UV:-}" ]]; then
  UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
fi

export UV_CACHE_DIR=$BUILD_CACHE/uv PIP_CACHE_DIR=$BUILD_CACHE/pip
export UV_PYTHON_INSTALL_DIR=$BUILD_CACHE/uv-python TMPDIR=$BUILD_CACHE/tmp
export XDG_CACHE_HOME=$BUILD_CACHE/xdg
mkdir -p "$BUILD_CACHE"/{uv,pip,uv-python,tmp,xdg} "$(dirname "$ENV_DIR")"
PY=$ENV_DIR/bin/python
FREEZE=$HERE/rl-fa.freeze.in

"$UV" venv "$ENV_DIR" --python 3.12 --clear || exit 10

# --no-deps: the freeze is already a complete closed set; the oracle env holds
# pin pairs a resolver rejects (fsspec 2026.6.0 vs datasets 5.0.0), so replicate
# it exactly instead of re-resolving.
"$UV" pip install -p "$PY" --no-deps -r "$FREEZE" \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match || exit 20

"$UV" pip install -p "$PY" --no-deps \
  "torchtitan @ git+https://github.com/pytorch/torchtitan@a1fdd7e43694bbfeff5d6ad8ac738c067bb90d41" \
  "dynamic-semaphore @ git+https://github.com/penfever/dynamic-semaphore@4d5f49f290889f4826219b241e1aa42d6466163e" \
  "harbor[daytona] @ git+https://github.com/marin-community/harbor.git@${HARBOR_REF}" \
  "https://github.com/marin-community/harbor/releases/download/harbor-config-${HARBOR_REF}/harbor_config-0.1.0-py3-none-any.whl" || exit 30

# MarinSkyRL is one root distribution (PR #284) whose hatch `force-include` COPIES
# skyrl_train/skyrl_gym into site-packages even for `-e`, so a `git pull` would not be
# live. Install the root for its metadata, then swap the copies for a .pth onto the
# checkout — that is the editable install upstream's packaging no longer gives us.
"$UV" pip install -p "$PY" --no-deps -e "$SKYRL_HOME" || exit 40
SP="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
rm -rf "$SP/skyrl_train" "$SP/skyrl_gym"
printf '%s\n%s\n' "$SKYRL_HOME/skyrl-train" "$SKYRL_HOME/skyrl-gym" > "$SP/_marinskyrl_src.pth"

# Deps upstream added after the freeze was taken (skyrl_gym imports reasoning_gym at
# import time). Resolved on top of the pinned set; must not move torch/vllm — checked below.
"$UV" pip install -p "$PY" reasoning-gym || exit 45

# transformer_engine_torch: best-effort (FSDP2 path does not import TE); never fail the build on it
"$UV" pip install -p "$PY" transformer_engine_torch==2.11.0 || echo "TE_SKIPPED"

W=flash_attn-2.8.3+cu128torch2.11-cp312-cp312-manylinux_2_34_aarch64.whl
curl -sL -o "$TMPDIR/$W" \
  "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.22/$W" || exit 50
"$UV" pip install -p "$PY" --no-deps "$TMPDIR/$W" || exit 55

echo "=== import smoke (login node, no GPU) ==="
"$PY" - <<'PYEOF'
import os
import torch; print("torch", torch.__version__, "cuda", torch.version.cuda)
import flash_attn, flash_attn_2_cuda
from flash_attn import flash_attn_func
print("flash_attn", flash_attn.__version__, "OK")
import vllm; print("vllm", vllm.__version__)
import vllm._C; print("vllm._C OK")
from torch.distributed.fsdp import fully_shard; print("fully_shard OK")
from torch.distributed.tensor.placement_types import _StridedShard; print("_StridedShard OK")
import torchtitan; print("torchtitan OK")
import skyrl_train, skyrl_gym, harbor_config.errors
assert skyrl_train.__file__.startswith(os.environ["SKYRL_HOME"]), skyrl_train.__file__
print("skyrl live from", skyrl_train.__file__)
import ray; print("ray", ray.__version__)
PYEOF
echo "BUILD_DONE rc=$?"
echo "Point DCFT_RL_ENV=$ENV_DIR in hpc/dotenv/jupiter.env to use this runtime."
