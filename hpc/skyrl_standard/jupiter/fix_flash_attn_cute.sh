#!/usr/bin/env bash
# Fix flash-attn 2.8.3 + nvidia-cutlass-dsl 4.6.1 mismatch on Jupiter (aarch64).
#
# Root cause:
#   flash_attn 2.8.3 ships a cutlass.cute DSL subtree that still references
#   deprecated APIs (cute.core.ThrMma, cutlass.utils.ampere_helpers, ...).
#   cutlass-dsl 4.6.1 moved/removed those, so `import flash_attn.cute` crashes.
#   vLLM's FA backend imports that cute path when the standalone wheel is present,
#   which previously forced us to uninstall flash-attn entirely.
#
# Fix:
#   1) Install the torch2.9/cu130/cp312 aarch64 FA2 wheel with --no-deps.
#   2) Disable the broken flash_attn/cute subtree. FA2 CUDA kernels + bert_padding
#      remain and are what SkyRL FSDP + transformers flash_attention_2 need.
#
# Usage (on Jupiter login):
#   bash hpc/skyrl_standard/jupiter/fix_flash_attn_cute.sh
#   RL_PYTHON=/path/to/envs/rl/bin/python bash .../fix_flash_attn_cute.sh
set -euo pipefail

RL_PYTHON="${RL_PYTHON:-${DCFT:-/e/scratch/reformo/lee27/OpenThoughts-Agent}/envs/rl/bin/python}"
if [[ ! -x "$RL_PYTHON" ]]; then
  echo "FATAL: RL python not executable: $RL_PYTHON" >&2
  exit 1
fi
SITE="$("$RL_PYTHON" -c 'import site; print(site.getsitepackages()[0])')"
FA_DIR="$SITE/flash_attn"
TMP="${TMPDIR:-/e/scratch/reformo/lee27/tmp_fa_fix}"
mkdir -p "$TMP"

WHL_NAME="flash_attn-2.8.3+cu130torch2.9-cp312-cp312-manylinux_2_34_aarch64.whl"
WHL_URL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3%2Bcu130torch2.9-cp312-cp312-manylinux_2_34_aarch64.whl"
WHL_PATH="$TMP/$WHL_NAME"

echo "=== RL_PYTHON=$RL_PYTHON ==="
"$RL_PYTHON" - <<'PY'
import torch, sys
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.version.cuda)
PY

if [[ ! -s "$WHL_PATH" ]]; then
  echo "=== downloading $WHL_NAME ==="
  curl -fL "$WHL_URL" -o "$WHL_PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
  export PATH="${HOME}/.local/bin:${PATH}"
fi
echo "=== install FA wheel (--no-deps) ==="
uv pip install --python "$RL_PYTHON" --no-deps --force-reinstall "$WHL_PATH"

echo "=== disable broken flash_attn.cute subtree ==="
if [[ -d "$FA_DIR/cute" ]]; then
  DEST="$FA_DIR/cute.DISABLED_cutlass461"
  rm -rf "$DEST"
  mv "$FA_DIR/cute" "$DEST"
  echo "moved $FA_DIR/cute -> $DEST"
elif [[ -d "$FA_DIR/cute.DISABLED_cutlass461" ]]; then
  echo "cute already disabled at $FA_DIR/cute.DISABLED_cutlass461"
else
  echo "WARN: no flash_attn/cute directory found under $FA_DIR"
fi

echo "=== verify ==="
"$RL_PYTHON" - <<'PY'
import importlib.util
import flash_attn
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn import flash_attn_func, flash_attn_varlen_func
assert importlib.util.find_spec("flash_attn.cute") is None, "cute should be disabled"
print("flash_attn", flash_attn.__version__, "FA2+bert_padding OK; cute disabled")
try:
    from transformers.utils import is_flash_attn_2_available
    print("transformers is_flash_attn_2_available", is_flash_attn_2_available())
except Exception as e:
    print("transformers check skipped:", e)
try:
    import skyrl_train.model_wrapper as mw
    print("skyrl _HAS_FLASH", getattr(mw, "_HAS_FLASH", None))
except Exception as e:
    print("skyrl check skipped:", e)
PY

echo "=== DONE: re-enable trainer.flash_attn=true in Jupiter SkyRL YAML ==="
