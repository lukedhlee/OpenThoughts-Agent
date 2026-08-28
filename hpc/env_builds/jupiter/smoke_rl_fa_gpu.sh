#!/bin/bash
# GPU smoke for $F/envs/rl-fa: flash_attn fwd/bwd + SDPA parity + varlen + HF/vLLM gates.
# Mirrors run_gsm8k_moe30b_grpo.sh's LD_LIBRARY_PATH construction (venv nvidia + torch libs).
set -uo pipefail
F=/e/fscratch/reformo/lee27
ENV=$F/envs/rl-fa
export VLLM_CACHE_ROOT=$F/cache/vllm XDG_CACHE_HOME=$F/cache/xdg \
       TRITON_CACHE_DIR=$F/cache/triton TORCHINDUCTOR_CACHE_DIR=$F/cache/inductor \
       FLASHINFER_WORKSPACE_BASE=$F/cache/flashinfer
_NV_LIBS=""
while IFS= read -r d; do _NV_LIBS="${_NV_LIBS:+$_NV_LIBS:}$d"; done \
  < <(find "$ENV/lib" -type d \( -path '*/nvidia/*/lib' -o -path '*/torch/lib' \) 2>/dev/null | sort -u)
export LD_LIBRARY_PATH="${_NV_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

$ENV/bin/python - <<'PYEOF'
import torch
print("device:", torch.cuda.get_device_name(0), "| torch", torch.__version__)

import flash_attn
from flash_attn import flash_attn_func, flash_attn_varlen_func
print("flash_attn", flash_attn.__version__)

torch.manual_seed(0)
B, S, H, D = 2, 512, 8, 128
q, k, v = (torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
           for _ in range(3))
out = flash_attn_func(q, k, v, causal=True)
out.sum().backward()
assert torch.isfinite(out).all() and torch.isfinite(q.grad).all()
print("fa2 fwd/bwd OK", tuple(out.shape))

ref = torch.nn.functional.scaled_dot_product_attention(
    q.detach().transpose(1, 2), k.detach().transpose(1, 2), v.detach().transpose(1, 2),
    is_causal=True).transpose(1, 2)
diff = (out.detach() - ref).abs().max().item()
print(f"fa2-vs-sdpa max|diff| = {diff:.5f}")
assert diff < 0.02, "parity fail"

cu = torch.arange(0, (B + 1) * S, S, dtype=torch.int32, device="cuda")
vout = flash_attn_varlen_func(
    q.reshape(-1, H, D), k.reshape(-1, H, D), v.reshape(-1, H, D),
    cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=S, max_seqlen_k=S, causal=True)
assert torch.isfinite(vout).all()
print("fa2 varlen (sample-packing path) OK")

from transformers.utils import is_flash_attn_2_available
print("HF is_flash_attn_2_available:", is_flash_attn_2_available())
assert is_flash_attn_2_available()

import vllm._C  # noqa: F401
print("vllm._C OK on GPU node")
print("GPU_SMOKE_PASS")
PYEOF
