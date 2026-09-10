#!/usr/bin/env bash
# Layer the MarinSkyRL trainer stack (rl-fa recipe) on top of the `snowball` venv AFTER the marin vLLM build.
# Principle: NEVER touch what the fork's vLLM build installed (torch/vllm/flashinfer/triton/transformers/...):
#   1. snapshot the post-build venv (rollback manifest),
#   2. install only the rl-fa pins whose package is NOT already present (--no-deps),
#   3. git pins + harbor + FA2 wheel (--no-deps),
#   4. skyrl_train / skyrl_gym via a .pth (the root pyproject's [vllm] extra pins the x86 wheel; never resolve it),
#   5. assert torch is still cu130 and vllm is still the source build, then an import smoke.
set -uo pipefail
C=/e/project1/transfernetx/lee27/code
F=/e/fscratch/reformo/lee27
ENV=$C/envs/snowball; PY=$ENV/bin/python
LOG=$C/logs/install_trainer_layer.log
exec >>"$LOG" 2>&1
echo "=== START $(date -Is) ==="
export UV_CACHE_DIR=$F/cache/uv PIP_CACHE_DIR=$F/cache/pip TMPDIR=$F/cache/tmp XDG_CACHE_HOME=$F/cache/xdg UV_LINK_MODE=copy
export PATH="$HOME/.local/bin:$PATH"
UV=$(command -v uv); echo "uv=$UV"

# 1. rollback manifest
$UV pip freeze -p "$PY" > $C/envs/snowball.post_build.freeze
echo "post-build freeze: $(wc -l < $C/envs/snowball.post_build.freeze) lines -> $C/envs/snowball.post_build.freeze"

# 2. rl-fa pins minus anything already installed (name-normalised), minus the vllm-owned families
$PY - <<'PY'
import json, re, subprocess, os
C = "/e/project1/transfernetx/lee27/code"
py = f"{C}/envs/snowball/bin/python"
have = {p["name"].lower().replace("_", "-") for p in json.loads(subprocess.check_output(["uv", "pip", "list", "-p", py, "--format=json"]))}
skip = re.compile(r"^(torch|torchvision|torchaudio|vllm|flashinfer|triton|pytorch-triton|transformers|numpy|xformers|nvidia|cuda|flash-attn|tokenizers|safetensors|huggingface|skyrl|harbor|dynamic-semaphore|torchtitan|transformer-engine|marinskyrl)")
out, kept, dropped = [], 0, 0
for line in open(f"{C}/envs/rl-fa.freeze.in"):
    s = line.strip()
    if not s or s.startswith("#") or s.startswith("-e ") or "@ file://" in s or " @ " in s:
        continue
    name = re.split(r"[=<>!~ \[;]", s, 1)[0].lower().replace("_", "-")
    if skip.match(name) or name in have:
        dropped += 1; continue
    out.append(s); kept += 1
open(f"{C}/envs/snowball.freeze.filtered.in", "w").write("\n".join(out) + "\n")
print(f"filtered pins: keep {kept} (missing from venv), drop {dropped} (present or vllm-owned)")
PY
$UV pip install -p "$PY" --no-deps -r $C/envs/snowball.freeze.filtered.in --index-strategy unsafe-best-match || echo "FREEZE_INSTALL_RC=$?"

# 3. git pins, harbor (local clone), harbor-config wheel, FA2 wheel — all --no-deps
$UV pip install -p "$PY" --no-deps \
  "torchtitan @ git+https://github.com/pytorch/torchtitan@a1fdd7e43694bbfeff5d6ad8ac738c067bb90d41" \
  "dynamic-semaphore @ git+https://github.com/penfever/dynamic-semaphore@4d5f49f290889f4826219b241e1aa42d6466163e" || echo "GIT_PINS_RC=$?"
$UV pip install -p "$PY" --no-deps "$C/harbor" || echo "HARBOR_RC=$?"
$UV pip install -p "$PY" --no-deps "https://github.com/marin-community/harbor/releases/download/harbor-config-df866b3086f386221e6e04ecf4b09e3cc9ffe44e/harbor_config-0.1.0-py3-none-any.whl" || echo "HARBOR_CONFIG_RC=$?"
W=flash_attn-2.8.3+cu130torch2.11-cp312-cp312-manylinux_2_34_aarch64.whl
[ -f $F/cache/tmp/$W ] || curl -sL -o $F/cache/tmp/$W https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.22/$W
$UV pip install -p "$PY" --no-deps $F/cache/tmp/$W || echo "FA_RC=$?"

# 4. skyrl_train / skyrl_gym as a .pth (same mechanism rl-fa ended up on: fix_editable.sh)
SP=$($PY -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
rm -rf $SP/skyrl_gym $SP/skyrl_train
printf "%s\n%s\n" $C/MarinSkyRL/skyrl-train $C/MarinSkyRL/skyrl-gym > $SP/_marinskyrl_src.pth
echo "wrote $SP/_marinskyrl_src.pth"

# 5. guards + import smoke
$PY -c "import torch; assert torch.version.cuda.startswith('13'), torch.version.cuda; print('torch', torch.__version__)" || { echo "TORCH_REPLACED -> reinstall cu130"; $UV pip install -p "$PY" torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130; }
$PY -c "import vllm; assert vllm.__file__.startswith('$C/src/marin_vllm'), vllm.__file__; print('vllm', vllm.__version__, vllm.__file__)" || echo "VLLM_CLOBBERED (restore from $C/envs/snowball.post_build.freeze / rebuild)"
echo "=== import smoke (login node, no GPU) ==="
$PY - <<'PY'
import importlib
bad = 0
for m in ("torch","vllm","transformers","ray","harbor","harbor_config","skyrl_train","skyrl_gym","torchtitan","dynamic_semaphore","flash_attn","hydra","omegaconf","wandb","peft","deepspeed","loguru","jax"):
    try:
        mod = importlib.import_module(m); print("OK ", m, getattr(mod, "__version__", ""), getattr(mod, "__file__", ""))
    except Exception as e:
        bad += 1; print("FAIL", m, type(e).__name__, str(e)[:160])
try:
    from skyrl_train.models import GrugMoeForCausalLM; print("OK  skyrl GrugMoeForCausalLM")
except Exception as e:
    bad += 1; print("FAIL skyrl GrugMoeForCausalLM", type(e).__name__, str(e)[:200])
try:
    from vllm.model_executor.models.registry import ModelRegistry; print("OK  vllm GrugMoe registered:", "GrugMoeForCausalLM" in ModelRegistry.get_supported_archs())
except Exception as e:
    bad += 1; print("FAIL vllm registry", type(e).__name__, str(e)[:200])
print("IMPORT_SMOKE_FAILURES", bad)
PY
echo "=== END rc=$? $(date -Is) ==="
