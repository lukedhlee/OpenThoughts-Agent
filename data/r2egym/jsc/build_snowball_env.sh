#!/usr/bin/env bash
# build_snowball_env.sh — reproduce the Snowball RL runtime (Jupiter's `snowball-v2` venv) from public sources only.
#
# What it builds under $ROOT (no access to anyone else's project tree is needed):
#   envs/uv-python/           uv-managed CPython 3.12.13 (aarch64)
#   envs/snowball/            the venv: torch 2.11.0+cu130, marin vLLM built from source (GH200 = sm_90),
#                             MarinSkyRL trainer layer (rl-fa pins), flash-attn 2.8.3, harbor + harbor-config editable,
#                             then every package pinned to the exact version of Jupiter's snowball-v2 (snowball-v2.freeze)
#   src/marin_vllm/           marin-community/vllm @ fa50698a (the MarinSkyRL uv.lock pin), compiled in place (editable)
#   marinskyrl-marin/         marin-community/MarinSkyRL  branch lukedhlee/snowball-r2egym   (on sys.path via .pth)
#   harbor-marin/             marin-community/harbor      branch lukedhlee/snowball-r2egym   (editable install)
#   OpenThoughts-Agent/       lukedhlee/OpenThoughts-Agent branch lukedhlee/rl_acceleration  (the Jupiter launcher)
#   snowball/                 data/r2egym/jsc from branch lukedhlee/vista-moe-grpo-30b (probe/arm generators, this script)
#   jupiter.local.env         per-user overlay for the launcher, generated from ROOT/SCRATCH (copied into the checkout)
#
# Usage (on a Jupiter LOGIN node — compute nodes have no internet; run inside tmux, ~90 min, the vLLM compile is ~60 min):
#   ROOT=/e/project1/reformo/$USER/snowball bash build_snowball_env.sh check      # verify modules + every download, build nothing
#   ROOT=/e/project1/reformo/$USER/snowball bash build_snowball_env.sh            # = all steps, each skipped when already done
#   ROOT=... bash build_snowball_env.sh vllm trainer                              # re-run named steps only
# Steps, in order: tools python clones torch vllm trainer overlay smoke
# Knobs: SCRATCH (default /e/fscratch/reformo/$USER; caches + experiments), CACHE (default $SCRATCH/cache/snowball_build),
#        MAX_JOBS (16), STAGE_MODULE (Stages/2026).
# Quotas: ROOT gets ~170k files / ~16 GB (venv 109k, vLLM tree 62k); the build caches are another ~150k files and go to CACHE.
#         JSC project1 inode quotas are per PROJECT (4 M soft), so pick a project with room: `check` prints the line.
#         2026-09-14: /e/project1/reformo was already over its inode soft limit (4.18 M / 4 M), so reformo members should
#         use ROOT=/e/project1/ccstdl/$USER/snowball SCRATCH=/e/fscratch/ccstdl/$USER (0.33 M / 4 M used) when they are in ccstdl.
#
# Provenance: the recipe is the login-node build of 2026-09-02 (jpbl-s01-02, 63 min compile) + the trainer layer of
# install_trainer_layer.sh + the 09-10 marin migration (harbor-marin/marinskyrl-marin editable), see .claude/ops/jupiter/ops.md.
# Traps it already knows about: the fork has a Rust frontend (needs the Rust module: setuptools-rust + cargo), torch must
# stay cu130 after the build (the fork's requirements would drag in the cu128 wheel — use_existing_torch.py strips them),
# never `pip install -e` the MarinSkyRL root (it copies skyrl_train into site-packages and shadows the checkout).
set -uo pipefail

# ----------------------------------------------------------------------------- pins (public)
VLLM_REPO=https://github.com/marin-community/vllm.git
VLLM_SHA=fa50698a9a303f7282aa0e969f35717703de4911
VLLM_VERSION=0.0.0.dev20260804+marin.fa50698a9a30           # setuptools-scm pretend version, same as Jupiter's build
MSRL_REPO=https://github.com/marin-community/MarinSkyRL.git
MSRL_BRANCH=lukedhlee/snowball-r2egym;  MSRL_SHA=20032472    # 2026-09-13 (what Jupiter runs); `git pull` later to follow the branch
HARBOR_REPO=https://github.com/marin-community/harbor.git
HARBOR_BRANCH=lukedhlee/snowball-r2egym; HARBOR_SHA=b964a5f6 # 2026-09-11
OTA_REPO=https://github.com/lukedhlee/OpenThoughts-Agent.git
OTA_BRANCH=lukedhlee/rl_acceleration;    OTA_SHA=f3edec45    # the launcher branch checked out on Jupiter
JSC_BRANCH=lukedhlee/vista-moe-grpo-30b                      # data/r2egym/jsc lives here (mirror of Jupiter's code/snowball)
TORCH_SPEC="torch==2.11.0"; TORCH_INDEX=https://download.pytorch.org/whl/cu130
UV_VERSION=0.11.31; PY_VERSION=3.12.13
FA_WHL=flash_attn-2.8.3+cu130torch2.11-cp312-cp312-manylinux_2_34_aarch64.whl
FA_URL=https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.22/$FA_WHL
TORCHTITAN_PIN="torchtitan @ git+https://github.com/pytorch/torchtitan@a1fdd7e43694bbfeff5d6ad8ac738c067bb90d41"
DYNSEM_PIN="dynamic-semaphore @ git+https://github.com/penfever/dynamic-semaphore@4d5f49f290889f4826219b241e1aa42d6466163e"
STAGE_MODULE=${STAGE_MODULE:-Stages/2026}                        # JSC already exports $STAGES (a path), hence the different name
MODULES=(CUDA/13 GCC/14.3.0 CMake Ninja Rust/1.88.0)        # what the 2026-09-02 build had on PATH (nvcc, gcc 14, cmake, ninja, cargo)

# ----------------------------------------------------------------------------- layout
: "${ROOT:?set ROOT (e.g. ROOT=/e/project1/reformo/\$USER/snowball) — everything is built under it}"
SCRATCH=${SCRATCH:-/e/fscratch/reformo/$USER}
CACHE=${CACHE:-$SCRATCH/cache/snowball_build}   # uv/pip/cargo caches: inode-heavy, keep them OFF project1 (its inode quota is per project)
ENV=$ROOT/envs/snowball; PY=$ENV/bin/python
SRC=$ROOT/src/marin_vllm
OTA=$ROOT/OpenThoughts-Agent
UV=$ROOT/bin/uv
export UV_CACHE_DIR=$CACHE/uv PIP_CACHE_DIR=$CACHE/pip TMPDIR=$CACHE/tmp XDG_CACHE_HOME=$CACHE/xdg CARGO_HOME=$CACHE/cargo
export UV_PYTHON_INSTALL_DIR=$ROOT/envs/uv-python UV_LINK_MODE=copy UV_NO_MODIFY_PATH=1
export MAX_JOBS=${MAX_JOBS:-16} NVCC_THREADS=2 TORCH_CUDA_ARCH_LIST=9.0 VLLM_TARGET_DEVICE=cuda
export SETUPTOOLS_SCM_PRETEND_VERSION=$VLLM_VERSION
mkdir -p "$ROOT"/{bin,envs,src,logs} "$CACHE"/{uv,pip,tmp,xdg,cargo}
exec > >(tee -a "$ROOT/logs/build_snowball_env.log") 2>&1
say() { echo "--- [$(date -Is)] $*"; }
die() { echo "FATAL: $*"; exit 1; }
done_marker() { [ -f "$ROOT/logs/.done.$1" ]; }
mark_done() { touch "$ROOT/logs/.done.$1"; }

load_modules() {
  type module >/dev/null 2>&1 || source "${LMOD_PKG:-/e/software/default/lmod/8.7.64}/init/bash" 2>/dev/null || die "no module command (Lmod init not found)"
  module purge >/dev/null 2>&1 || true
  module load "$STAGE_MODULE" >/dev/null 2>&1 || say "module load $STAGE_MODULE failed (continuing with the default stage)"
  local m; for m in "${MODULES[@]}"; do module load "$m" >/dev/null 2>&1 || say "WARN: module load $m failed"; done
  export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc || true)")")"; export PATH="$CUDA_HOME/bin:$PATH"
}

quota_line() { # path kind -> the project quota row for that filesystem (jutil is a login-shell function on JSC)
  local proj; proj=$(printf '%s' "$1" | awk -F/ '$2=="e" && ($3=="project1" || $3=="fscratch") {print $4}')
  [ -n "$proj" ] || { echo "  $2: $1 is not under /e/project1 or /e/fscratch; no quota check"; return; }
  local row; row=$(bash -lc "jutil project dataquota -p $proj" 2>/dev/null | awk -v k="exa_$2" '$4==k')
  [ -n "$row" ] || { echo "  $2 quota for project $proj: jutil unavailable"; return; }
  echo "$row" | awk -v p="$proj" -v k="$2" '{du=$7/1024/1024/1024; ds=$8/1024/1024/1024; printf "  %s quota (%s): %.1f / %.0f TB, inodes %d / %d soft (%.0f%%)%s\n", k, p, du, ds, $10, $11, 100*$10/$11, ($10>0.9*$11 ? "  <-- NEAR/OVER the inode limit: pick another project" : "")}'
}

# ----------------------------------------------------------------------------- steps
step_check() {
  say "check: host $(hostname)"
  case "$(hostname)" in jpbl-s01-*|jwlogin*|jrlogin*) ;; *) echo "WARN: not a JSC login node — compute nodes have no internet";; esac
  ( load_modules
    for t in nvcc gcc cmake ninja cargo; do printf "  %-6s %s\n" "$t" "$(command -v $t || echo MISSING)"; done
    command -v nvcc >/dev/null || die "nvcc missing: CUDA module did not load"
    command -v cargo >/dev/null || die "cargo missing: the marin vLLM fork's Rust frontend needs the Rust module" )
  for u in "https://astral.sh/uv/$UV_VERSION/install.sh" "$FA_URL" "$TORCH_INDEX/torch/"; do
    printf "  %s -> %s\n" "$u" "$(curl -s -o /dev/null -L -m 30 -w '%{http_code}' -I "$u")"; done
  for r in "$VLLM_REPO HEAD" "$MSRL_REPO $MSRL_BRANCH" "$HARBOR_REPO $HARBOR_BRANCH" "$OTA_REPO $OTA_BRANCH" "$OTA_REPO $JSC_BRANCH"; do
    # shellcheck disable=SC2086
    printf "  %-75s %s\n" "$r" "$(git ls-remote $r 2>/dev/null | awk 'NR==1{print substr($1,1,8)} END{if(NR==0)print "UNREACHABLE"}')"; done
  echo "  (vllm is pinned at $VLLM_SHA, fetched by the clone step; ls-remote only proves the host is reachable)"
  printf "  ulimit -u %s (login-node pid cgroup; MAX_JOBS=%s NVCC_THREADS=%s fits)\n" "$(ulimit -u)" "$MAX_JOBS" "$NVCC_THREADS"
  quota_line "$ROOT" project1; quota_line "$SCRATCH" fscratch
  say "check: done (nothing built). Budget ~20 GB, ~170k files, ~90 min."
}

step_tools() {
  done_marker tools && { say "tools: done"; return; }
  [ -x "$UV" ] || { say "tools: installing uv $UV_VERSION -> $ROOT/bin"; curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_INSTALL_DIR="$ROOT/bin" sh || die "uv install"; }
  "$UV" --version || die "uv broken"
  mark_done tools
}

step_python() {
  done_marker python && { say "python: done"; return; }
  say "python: uv-managed CPython $PY_VERSION -> $UV_PYTHON_INSTALL_DIR"
  "$UV" python install --no-bin "$PY_VERSION" || die "uv python install"   # --no-bin: no shim in ~/.local/bin
  [ -x "$PY" ] || "$UV" venv "$ENV" --python "$PY_VERSION" --python-preference only-managed || die "uv venv"
  "$PY" -c "import sys; print('venv python', sys.version.split()[0], sys.executable)"
  mark_done python
}

clone_pin() { # dest repo branch sha
  local d=$1 r=$2 b=$3 s=$4
  if [ -d "$d/.git" ]; then say "clone: $d exists ($(git -C "$d" rev-parse --short HEAD))"; return; fi
  git clone -q -b "$b" "$r" "$d" || die "clone $r"
  git -C "$d" checkout -q "$s" 2>/dev/null || { git -C "$d" fetch -q origin "$b" && git -C "$d" checkout -q "$s"; } || die "checkout $s in $d"
  git -C "$d" checkout -q -B "$b" "$s"   # stay on the branch name, pinned at the SHA (git pull to follow later)
  say "clone: $d @ $(git -C "$d" rev-parse --short HEAD) ($b)"
}

step_clones() {
  done_marker clones && { say "clones: done"; return; }
  if [ ! -d "$SRC/.git" ]; then
    say "clones: marin vllm @ $VLLM_SHA (blob:none partial clone)"
    git clone -q --filter=blob:none "$VLLM_REPO" "$SRC" || die "clone vllm"
    git -C "$SRC" checkout -q "$VLLM_SHA" || die "checkout vllm sha"
  fi
  clone_pin "$ROOT/marinskyrl-marin" "$MSRL_REPO" "$MSRL_BRANCH" "$MSRL_SHA"
  clone_pin "$ROOT/harbor-marin"     "$HARBOR_REPO" "$HARBOR_BRANCH" "$HARBOR_SHA"
  clone_pin "$OTA"                    "$OTA_REPO" "$OTA_BRANCH" "$OTA_SHA"
  if [ ! -d "$ROOT/snowball" ]; then
    say "clones: data/r2egym/jsc from $JSC_BRANCH -> $ROOT/snowball"
    git -C "$OTA" fetch -q origin "$JSC_BRANCH" || die "fetch $JSC_BRANCH"
    mkdir -p "$ROOT/snowball" && git -C "$OTA" archive "origin/$JSC_BRANCH" data/r2egym/jsc | tar -x -C "$ROOT/snowball" --strip-components=3 || die "archive jsc"
  fi
  mark_done clones
}

step_torch() {
  done_marker torch && { say "torch: done"; return; }
  say "torch: $TORCH_SPEC from $TORCH_INDEX"
  "$UV" pip install -p "$PY" "$TORCH_SPEC" --index-url "$TORCH_INDEX" || die "torch install"
  "$PY" -c "import torch; assert torch.version.cuda.startswith('13'), torch.version.cuda; print('torch', torch.__version__)" || die "torch is not cu130"
  mark_done torch
}

assert_torch_cu130() {
  "$PY" -c "import torch; assert torch.version.cuda.startswith('13'), torch.version.cuda" 2>/dev/null && return
  say "torch was replaced by a non-cu130 wheel; reinstalling $TORCH_SPEC"
  "$UV" pip install -p "$PY" "$TORCH_SPEC" --index-url "$TORCH_INDEX" || die "torch reinstall"
}

step_vllm() {
  done_marker vllm && { say "vllm: done"; return; }
  load_modules
  say "vllm: nvcc=$(command -v nvcc) cargo=$(command -v cargo) cmake=$(command -v cmake) ninja=$(command -v ninja)"
  command -v nvcc >/dev/null || die "nvcc missing"; command -v cargo >/dev/null || die "cargo missing (Rust module)"
  "$UV" pip install -p "$PY" packaging ninja cmake psutil setuptools-scm "setuptools<81" setuptools-rust jinja2 numpy regex wheel requests || die "build deps"
  cd "$SRC" || die "no $SRC"; git rev-parse HEAD
  [ -f use_existing_torch.py ] && "$PY" use_existing_torch.py   # strips the fork's torch pins so the build keeps our cu130 torch
  say "vllm: compiling editable (MAX_JOBS=$MAX_JOBS, ~60 min on a login node)"
  "$UV" pip install -p "$PY" -e . --no-build-isolation; rc=$?
  say "vllm: compile rc=$rc"; [ "$rc" -eq 0 ] || die "vllm build failed (see $ROOT/logs/build_snowball_env.log)"
  assert_torch_cu130
  "$PY" - <<'PY' || die "vllm smoke"
import torch, vllm
from vllm.model_executor.models.registry import ModelRegistry
print("vllm", vllm.__version__, vllm.__file__, "torch", torch.__version__, torch.version.cuda)
assert "GrugMoeForCausalLM" in ModelRegistry.get_supported_archs(), "GrugMoeForCausalLM not registered: wrong vllm"
print("GrugMoeForCausalLM registered")
PY
  mark_done vllm
}

step_trainer() {
  done_marker trainer && { say "trainer: done"; return; }
  local FREEZE=$OTA/hpc/env_builds/jupiter/rl-fa.freeze.in V2=$ROOT/snowball/snowball-v2.freeze
  [ -f "$FREEZE" ] || die "missing $FREEZE (OTA checkout on $OTA_BRANCH?)"
  [ -f "$V2" ] || V2=$(dirname "${BASH_SOURCE[0]}")/snowball-v2.freeze   # beside this script when run from a clone
  [ -f "$V2" ] || die "missing snowball-v2.freeze (ships next to this script in data/r2egym/jsc)"
  say "trainer: rollback manifest -> $ROOT/envs/snowball.post_build.freeze"
  "$UV" pip freeze -p "$PY" > "$ROOT/envs/snowball.post_build.freeze"
  # rl-fa pins minus what the vLLM build already owns (torch/vllm/triton/transformers/nvidia-*...) minus what is present.
  FREEZE_IN=$FREEZE FILTERED=$ROOT/envs/snowball.freeze.filtered.in UV_BIN=$UV VENV_PY=$PY "$PY" - <<'PY' || die "freeze filter"
import json, os, re, subprocess
have = {p["name"].lower().replace("_", "-") for p in json.loads(subprocess.check_output([os.environ["UV_BIN"], "pip", "list", "-p", os.environ["VENV_PY"], "--format=json"]))}
skip = re.compile(r"^(torch(vision|audio)?$|vllm|flashinfer|triton|pytorch-triton|transformers|numpy|xformers|nvidia|cuda|flash-attn|tokenizers|safetensors|huggingface|skyrl|harbor|dynamic-semaphore|torchtitan|transformer-engine|marinskyrl)")
out, dropped = [], 0
for line in open(os.environ["FREEZE_IN"]):
    s = line.strip()
    if not s or s.startswith("#") or s.startswith("-e ") or "@ file://" in s or " @ " in s: continue
    name = re.split(r"[=<>!~ \[;]", s, 1)[0].lower().replace("_", "-")
    if skip.match(name) or name in have: dropped += 1; continue
    out.append(s)
open(os.environ["FILTERED"], "w").write("\n".join(out) + "\n")
print(f"filtered pins: keep {len(out)} (missing from venv), drop {dropped} (present or vllm-owned)")
PY
  "$UV" pip install -p "$PY" --no-deps -r "$ROOT/envs/snowball.freeze.filtered.in" --index-strategy unsafe-best-match || die "freeze install"
  say "trainer: git pins, harbor (editable, harbor-marin), harbor-config, flash-attn wheel — all --no-deps"
  "$UV" pip install -p "$PY" --no-deps "$TORCHTITAN_PIN" "$DYNSEM_PIN" || die "git pins"
  "$UV" pip install -p "$PY" --no-deps -e "$ROOT/harbor-marin" || die "harbor editable"
  "$UV" pip install -p "$PY" --no-deps -e "$ROOT/harbor-marin/packages/harbor-config" || die "harbor-config editable"
  [ -f "$CACHE/tmp/$FA_WHL" ] || curl -sL -o "$CACHE/tmp/$FA_WHL" "$FA_URL" || die "flash-attn wheel download"
  "$UV" pip install -p "$PY" --no-deps "$CACHE/tmp/$FA_WHL" || die "flash-attn install"
  # Exact pin-down: the vLLM build resolves its unpinned runtime deps at build time (transformers, tokenizers, openai, ...
  # drifted ~30 versions between 09-02 and 09-14), so bring every package to the version snowball-v2 runs. --no-deps,
  # torch/vllm/editables untouched. Idempotent: a second run installs nothing.
  say "trainer: exact pin-down to $(basename "$V2")"
  V2_FREEZE=$V2 PINS=$ROOT/envs/snowball.pindown.in UV_BIN=$UV VENV_PY=$PY "$PY" - <<'PY' || die "pin-down list"
import json, os, re, subprocess
norm = lambda s: s.lower().replace("_", "-")
have = {norm(p["name"]): p["version"] for p in json.loads(subprocess.check_output([os.environ["UV_BIN"], "pip", "list", "-p", os.environ["VENV_PY"], "--format=json"]))}
pins, missing, changed = [], 0, 0
for line in open(os.environ["V2_FREEZE"]):
    s = line.strip()
    if not s or s.startswith("#") or s.startswith("-e ") or "@ file://" in s: continue
    if " @ " in s:                      # VCS pin: only if absent (versions are not comparable)
        name = norm(s.split(" @ ", 1)[0])
        if name not in have: pins.append(s); missing += 1
        continue
    name, ver = s.split("==", 1); name = norm(name)
    if name in ("torch", "vllm"): continue
    if name not in have: pins.append(s); missing += 1
    elif have[name] != ver: pins.append(s); changed += 1
open(os.environ["PINS"], "w").write("\n".join(pins) + "\n")
print(f"pin-down: install {missing} missing + {changed} version changes ({len(have)} packages present)")
PY
  if [ -s "$ROOT/envs/snowball.pindown.in" ]; then
    "$UV" pip install -p "$PY" --no-deps -r "$ROOT/envs/snowball.pindown.in" --index-strategy unsafe-best-match || die "pin-down install"
  fi
  # MarinSkyRL on sys.path via .pth files (never `pip install -e` its root: that copies skyrl_train into site-packages and shadows the checkout)
  local SP; SP=$("$PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
  rm -rf "$SP/skyrl_gym" "$SP/skyrl_train"
  printf "%s\n%s\n" "$ROOT/marinskyrl-marin/skyrl-train" "$ROOT/marinskyrl-marin/skyrl-gym" > "$SP/_marinskyrl_src.pth"
  printf "%s\n" "$ROOT/marinskyrl-marin" > "$SP/_editable_impl_marinskyrl.pth"
  assert_torch_cu130
  "$PY" -c "import vllm; assert vllm.__file__.startswith('$SRC'), vllm.__file__" || die "vllm was clobbered by the trainer layer (restore from $ROOT/envs/snowball.post_build.freeze)"
  mark_done trainer
}

step_overlay() {
  local OV=$ROOT/jupiter.local.env
  say "overlay: $OV (ROOT=$ROOT SCRATCH=$SCRATCH)"
  mkdir -p "$SCRATCH"/{data,otagent,keys,cache/{hf,vllm,xdg,triton,torchinductor,flashinfer},hf_hub,checkpoints,experiments,wandb,ray_spill} 2>/dev/null || true
  cat > "$OV" <<EOF
# Per-user Jupiter overlay, generated by build_snowball_env.sh on $(date -Is). Loaded AFTER hpc/dotenv/jupiter.env by
# hpc.set_environment and by the RL sbatch (later wins). Install: cp to \$DCFT/hpc/dotenv/jupiter.local.env (gitignored); chmod 600.
# Secrets go in \$DC_AGENT_SECRET_ENV (DAYTONA_API_KEY for the org that holds the snapshots, WANDB_API_KEY, HF_TOKEN), never here.
export SCRATCH=$SCRATCH
export CODE_ROOT=$ROOT
export DCFT=\$CODE_ROOT/OpenThoughts-Agent
export DCFT_DATA=\$SCRATCH/data
export DCFT_SCRATCH=\$SCRATCH/otagent
export DCFT_GROUP=reformo
export DC_AGENT_SECRET_ENV=\$SCRATCH/keys/secrets.env
export HF_HOME=\$SCRATCH/cache/hf
export HF_HUB_CACHE=\$SCRATCH/hf_hub
export MODELS_DIR=\$HF_HUB_CACHE
export CHECKPOINTS_DIR=\$SCRATCH/checkpoints
export DATASETS_DIR=\$SCRATCH/data
export TOKENIZED_DATASETS_DIR=\$DCFT_SCRATCH/tokenized_datasets
export VLLM_CACHE_ROOT=\$SCRATCH/cache/vllm
export XDG_CACHE_HOME=\$SCRATCH/cache/xdg
export TRITON_CACHE_DIR=\$SCRATCH/cache/triton
export TORCHINDUCTOR_CACHE_DIR=\$SCRATCH/cache/torchinductor
export FLASHINFER_WORKSPACE_BASE=\$SCRATCH/cache/flashinfer
export WANDB_DIR=\$SCRATCH/wandb
export WANDB_MODE=offline
export EXPERIMENTS_DIR=\$SCRATCH/experiments
export OT_AGENT_RAY_LOG_DIR=\$SCRATCH/experiments/_ray_logs
export OT_AGENT_RAY_SPILL_DIR=\$SCRATCH/ray_spill
export DCFT_CONDA_ACTIVATE=
# --- host-venv RL runtime: the venv this script built; MarinSkyRL checkout FIRST on PYTHONPATH (it composes the Hydra config)
export RL_REPO_DIR=\$CODE_ROOT/marinskyrl-marin
export SKYRL_HOME=\$RL_REPO_DIR
export PYTHONPATH="\${SKYRL_HOME}/skyrl-train:\${SKYRL_HOME}/skyrl-gym:\${DCFT}"
export DCFT_RL_ENV=\$CODE_ROOT/envs/snowball
export RL_PYTHON=\$DCFT_RL_ENV/bin/python
export IMAGE=
_RL_SP=\$DCFT_RL_ENV/lib/python3.12/site-packages
export LD_LIBRARY_PATH=\$_RL_SP/nvidia/cu13/lib:\$_RL_SP/nvidia/cublas/lib:\$_RL_SP/nvidia/cuda_cupti/lib:\$_RL_SP/nvidia/cuda_nvrtc/lib:\$_RL_SP/nvidia/cuda_runtime/lib:\$_RL_SP/nvidia/cudnn/lib:\$_RL_SP/nvidia/cufft/lib:\$_RL_SP/nvidia/cufile/lib:\$_RL_SP/nvidia/curand/lib:\$_RL_SP/nvidia/cusolver/lib:\$_RL_SP/nvidia/cusparse/lib:\$_RL_SP/nvidia/cusparselt/lib:\$_RL_SP/nvidia/nccl/lib:\$_RL_SP/nvidia/nvjitlink/lib:\$_RL_SP/nvidia/nvshmem/lib:\$_RL_SP/nvidia/nvtx/lib:\$_RL_SP/torch/lib:/.singularity.d/libs:/usr/local/cuda/compat/lib.real:/usr/local/cuda-13/lib64/stubs:/usr/local/cuda/lib64/stubs:/usr/local/cuda-13/compat/lib.real/
# --- Daytona egress from compute nodes: JSC TOTP accounts cannot open compute->login ssh -D, so point the launcher's
# proxy block at a microsocks you run on a login node (authenticated; bind the node's 10.128.x.x address). Build
# proxychains-ng once with \$DCFT/hpc/proxychains_setup.sh install (PROXYCHAINS_PREFIX=\$SCRATCH/tools/proxychains-ng).
export PROXYCHAINS_BIN_OVERRIDE=\$SCRATCH/tools/proxychains-ng/bin/proxychains4
# export PROXYCHAINS_SOCKS5_PRESET_HOST=10.128.1.2
# export PROXYCHAINS_SOCKS5_PRESET_PORT=7012
# export PROXYCHAINS_SOCKS5_PRESET_AUTH="<user> <pass>"
EOF
  chmod 600 "$OV"
  if [ -d "$OTA/hpc/dotenv" ] && [ ! -f "$OTA/hpc/dotenv/jupiter.local.env" ]; then cp "$OV" "$OTA/hpc/dotenv/jupiter.local.env"; say "overlay: installed into $OTA/hpc/dotenv/"; fi
}

step_smoke() {
  say "smoke: imports on the login node (no GPU)"
  OMP_NUM_THREADS=1 SRC="$SRC" HB="$ROOT/harbor-marin" MS="$ROOT/marinskyrl-marin" "$PY" - <<'PY'
import importlib, os
bad = 0
for m in ("torch","vllm","transformers","ray","harbor","harbor_config","skyrl_train","skyrl_gym","marinskyrl","torchtitan","torchdata","reasoning_gym","dynamic_semaphore","flash_attn","hydra","omegaconf","wandb","peft","loguru"):
    try:
        mod = importlib.import_module(m); print("OK  ", m, getattr(mod, "__version__", ""), getattr(mod, "__file__", ""))
    except Exception as e:
        bad += 1; print("FAIL", m, type(e).__name__, str(e)[:160])
import vllm, harbor, skyrl_train, torch
for name, mod, root in (("vllm", vllm, os.environ["SRC"]), ("harbor", harbor, os.environ["HB"]), ("skyrl_train", skyrl_train, os.environ["MS"])):
    if not mod.__file__.startswith(root): bad += 1; print("FAIL", name, "imports from", mod.__file__, "expected under", root)
if not torch.version.cuda.startswith("13"): bad += 1; print("FAIL torch cuda", torch.version.cuda)
try:
    from vllm.model_executor.models.registry import ModelRegistry; print("OK   vllm GrugMoe registered:", "GrugMoeForCausalLM" in ModelRegistry.get_supported_archs())
except Exception as e:
    bad += 1; print("FAIL vllm registry", type(e).__name__, str(e)[:200])
print("IMPORT_SMOKE_FAILURES", bad); raise SystemExit(1 if bad else 0)
PY
  local rc=$?; [ $rc -eq 0 ] && say "smoke: PASS — next: sbatch $ROOT/snowball/serve_smoke.sbatch (after fixing its paths) for the GPU serve check" || die "smoke failed (rc=$rc)"
}

# ----------------------------------------------------------------------------- main
STEPS=("$@"); [ ${#STEPS[@]} -eq 0 ] && STEPS=(tools python clones torch vllm trainer overlay smoke)
say "build_snowball_env.sh ROOT=$ROOT SCRATCH=$SCRATCH CACHE=$CACHE steps: ${STEPS[*]}"
for s in "${STEPS[@]}"; do
  case "$s" in
    check|all|tools|python|clones|torch|vllm|trainer|overlay|smoke) ;;
    *) die "unknown step '$s' (check tools python clones torch vllm trainer overlay smoke)";;
  esac
  if [ "$s" = all ]; then for t in tools python clones torch vllm trainer overlay smoke; do "step_$t"; done; else "step_$s"; fi
done
say "END ok. Venv: $PY  overlay: $ROOT/jupiter.local.env  log: $ROOT/logs/build_snowball_env.log"
