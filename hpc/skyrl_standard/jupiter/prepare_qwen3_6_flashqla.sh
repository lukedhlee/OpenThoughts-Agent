#!/usr/bin/env bash
# Install the exact FlashQLA layer used by Qwen3.6 GRPO, then submit its GH200
# forward/backward smoke. Run this on a Jupiter login node with internet access.
set -euo pipefail

: "${JSC_SCRATCH:=/e/scratch/reformo/lee27}"
: "${DCFT:=$JSC_SCRATCH/OpenThoughts-Agent-r2egym-bridge}"
: "${RL_VENV:=$JSC_SCRATCH/OpenThoughts-Agent/envs/rl-megatron}"
: "${FLASHQLA_LAYER:=$JSC_SCRATCH/pydeps/qwen36-flashqla-0.1.2}"
: "${INSTALL_ONLY:=0}"

LOCK_DIR="${FLASHQLA_LAYER}.install.lock"
PARTIAL="${FLASHQLA_LAYER}.partial.$$"

if [[ ! -x "$RL_VENV/bin/python" ]]; then
  echo "ERROR: missing runtime: $RL_VENV/bin/python" >&2
  exit 1
fi
if [[ ! -f "$DCFT/hpc/sbatch_rl/jupiter_qwen3_6_flashqla_smoke.sbatch" ]]; then
  echo "ERROR: missing FlashQLA smoke sbatch under $DCFT" >&2
  exit 1
fi
if [[ -e "$FLASHQLA_LAYER" ]]; then
  echo "ERROR: refusing to overwrite existing layer: $FLASHQLA_LAYER" >&2
  exit 1
fi

mkdir -p "$(dirname "$FLASHQLA_LAYER")"
if ! mkdir "$LOCK_DIR"; then
  echo "ERROR: another install may be active: $LOCK_DIR" >&2
  exit 1
fi
cleanup() {
  if [[ -d "$PARTIAL" ]]; then
    rm -rf -- "$PARTIAL"
  fi
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT
mkdir "$PARTIAL"

"$RL_VENV/bin/python" - <<'PY'
import importlib.util

required = (
    "cloudpickle",
    "ml_dtypes",
    "numpy",
    "psutil",
    "torch",
    "tqdm",
    "typing_extensions",
)
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"runtime is missing common TileLang dependencies: {missing}")
PY

"$RL_VENV/bin/python" -m pip install \
  --target "$PARTIAL" \
  --only-binary=:all: \
  --no-deps \
  flash-qla==0.1.2 \
  tilelang==0.1.9 \
  apache-tvm-ffi==0.1.9 \
  torch-c-dlpack-ext==0.1.5 \
  z3-solver==4.15.4.0

"$RL_VENV/bin/python" - "$PARTIAL" <<'PY'
import json
import platform
import sys
from importlib.metadata import distributions
from pathlib import Path

target = Path(sys.argv[1]).resolve()
expected = {
    "apache-tvm-ffi": "0.1.9",
    "flash-qla": "0.1.2",
    "tilelang": "0.1.9",
    "torch-c-dlpack-ext": "0.1.5",
    "z3-solver": "4.15.4.0",
}
found = {
    dist.metadata["Name"].lower().replace("_", "-"): dist.version
    for dist in distributions(path=[str(target)])
}
if found != expected:
    raise SystemExit(f"isolated dependency mismatch: expected={expected}, found={found}")
manifest = {
    "format": "qwen3.6-flashqla-layer-v1",
    "packages": expected,
    "platform": platform.platform(),
    "python": platform.python_version(),
    "gpu_smoke_required": True,
}
(target / "qwen3_6_flashqla_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
PY

mv "$PARTIAL" "$FLASHQLA_LAYER"
trap - EXIT
rmdir "$LOCK_DIR"

echo "Installed isolated FlashQLA layer: $FLASHQLA_LAYER"
if [[ "$INSTALL_ONLY" == "1" ]]; then
  echo "INSTALL_ONLY=1; GPU smoke not submitted."
  exit 0
fi

JOB_ID="$(
  sbatch --parsable \
    --export=ALL,FLASHQLA_LAYER="$FLASHQLA_LAYER" \
    "$DCFT/hpc/sbatch_rl/jupiter_qwen3_6_flashqla_smoke.sbatch"
)"
echo "Submitted FlashQLA GH200 smoke job: $JOB_ID"
echo "The GRPO launcher remains gated until gpu_sm90_smoke.ok is written."
