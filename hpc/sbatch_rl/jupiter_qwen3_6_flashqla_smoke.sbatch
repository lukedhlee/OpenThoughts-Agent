#!/usr/bin/env bash
#SBATCH --job-name=qwen36-flashqla-smoke
#SBATCH --partition=booster
#SBATCH --account=reformo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=/e/scratch/reformo/lee27/experiments/logs/%x_%j.out

# Prove the exact isolated FlashQLA layer can JIT, run, and backpropagate on a
# Jupiter GH200 before a six-node allocation is submitted.
set -euo pipefail

: "${JSC_SCRATCH:=/e/scratch/reformo/lee27}"
: "${RL_VENV:=$JSC_SCRATCH/OpenThoughts-Agent/envs/rl-megatron}"
: "${FLASHQLA_LAYER:=$JSC_SCRATCH/pydeps/qwen36-flashqla-0.1.2}"
: "${TILELANG_CACHE_DIR:=$JSC_SCRATCH/cache/tilelang/qwen36-flashqla-0.1.2-sm90}"

module load GCC/14.3.0
module load nvidia-compilers/25.9-CUDA-13
mkdir -p "$JSC_SCRATCH/experiments/logs" "$TILELANG_CACHE_DIR"
export TILELANG_CACHE_DIR
export PYTHONPATH="$FLASHQLA_LAYER${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -x "$RL_VENV/bin/python" ]]; then
  echo "ERROR: missing runtime: $RL_VENV/bin/python" >&2
  exit 1
fi
if [[ ! -f "$FLASHQLA_LAYER/qwen3_6_flashqla_manifest.json" ]]; then
  echo "ERROR: missing isolated layer manifest: $FLASHQLA_LAYER" >&2
  exit 1
fi

srun --ntasks=1 --gpus=1 "$RL_VENV/bin/python" - "$FLASHQLA_LAYER" <<'PY'
import inspect
import json
import math
import sys
from importlib.metadata import version
from pathlib import Path

import torch
import torch.nn.functional as F
from flash_qla import chunk_gated_delta_rule

layer = Path(sys.argv[1]).resolve()
manifest = json.loads((layer / "qwen3_6_flashqla_manifest.json").read_text())
expected = manifest["packages"]
for package, expected_version in expected.items():
    actual = version(package)
    if actual != expected_version:
        raise SystemExit(f"{package}: expected {expected_version}, got {actual}")
if version("flash-qla") != "0.1.2":
    raise SystemExit("unexpected FlashQLA version")
if "use_qk_l2norm_in_kernel" not in inspect.signature(
    chunk_gated_delta_rule
).parameters:
    raise SystemExit("FlashQLA lacks Qwen3.6's Q/K normalization interface")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
properties = torch.cuda.get_device_properties(0)
if properties.major != 9:
    raise SystemExit(
        f"expected an SM90 GH200 for this smoke, got {properties.name} "
        f"(SM{properties.major}{properties.minor})"
    )

# Qwen3.6-MoE GDN geometry: 16 Q/K heads, 32 value heads, K=V=128.
# A 256-token sample exercises both the fused forward and its custom backward.
batch, sequence, qk_heads, value_heads, width = 1, 256, 16, 32, 128
q = torch.randn(
    batch,
    sequence,
    qk_heads,
    width,
    device="cuda",
    dtype=torch.bfloat16,
    requires_grad=True,
)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn(
    batch,
    sequence,
    value_heads,
    width,
    device="cuda",
    dtype=torch.bfloat16,
    requires_grad=True,
)
g_logits = torch.randn(
    batch,
    sequence,
    value_heads,
    device="cuda",
    dtype=torch.bfloat16,
    requires_grad=True,
)
beta_logits = torch.randn_like(g_logits, requires_grad=True)
# Qwen3.6 computes the decay gate from ``a.float()`` so FlashQLA receives an
# FP32 gate even when the model activations are BF16. Its backward kernel
# requires this contract because the accumulated gate gradient is FP32.
g = F.logsigmoid(g_logits.float())
beta = beta_logits.sigmoid()
output, final_state = chunk_gated_delta_rule(
    q,
    k,
    v,
    g,
    beta,
    scale=1.0 / math.sqrt(width),
    output_final_state=False,
    use_qk_l2norm_in_kernel=True,
    head_first=False,
)
if final_state is not None:
    raise SystemExit("unexpected final state")
if output.shape != v.shape or not torch.isfinite(output).all():
    raise SystemExit(f"invalid FlashQLA output: shape={tuple(output.shape)}")
output.float().square().mean().backward()
for name, tensor in {
    "q": q,
    "k": k,
    "v": v,
    "g_logits": g_logits,
    "beta_logits": beta_logits,
}.items():
    if tensor.grad is None or not torch.isfinite(tensor.grad).all():
        raise SystemExit(f"invalid or missing FlashQLA gradient: {name}")
torch.cuda.synchronize()
print(
    "FlashQLA GH200 forward/backward passed:",
    f"torch={torch.__version__}",
    f"flash-qla={version('flash-qla')}",
    f"tilelang={version('tilelang')}",
    f"gpu={properties.name}",
    f"sm={properties.major}{properties.minor}",
)
PY

MARKER="$FLASHQLA_LAYER/gpu_sm90_smoke.ok"
{
  echo "format=qwen3.6-flashqla-smoke-v1"
  echo "job_id=${SLURM_JOB_ID:-unknown}"
  echo "completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$MARKER"
echo "Wrote verified GPU smoke marker: $MARKER"
