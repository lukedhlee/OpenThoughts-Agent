#!/usr/bin/env bash
# LOGIN-NODE prep for the compute-node hedge build of marin-community/vllm @ fa50698a (GH200).
# Everything that needs internet happens here: second venv, torch cu130, build deps, the fork's runtime
# deps, a source copy, and a cargo prefetch for the Rust frontend. Then it submits build_cn.sbatch.
set -uo pipefail
C=/e/project1/transfernetx/lee27/code
F=/e/fscratch/reformo/lee27
ENV=$C/envs/snowball_cn; PY=$ENV/bin/python
SRC0=$C/src/marin_vllm; SRC=$C/src/marin_vllm_cn
LOG=$C/logs/cn_prep.log
exec >>"$LOG" 2>&1
echo "=== PREP START $(date -Is) host=$(hostname) ==="
export UV_CACHE_DIR=$F/cache/uv PIP_CACHE_DIR=$F/cache/pip TMPDIR=$F/cache/tmp XDG_CACHE_HOME=$F/cache/xdg UV_LINK_MODE=copy
export PATH="$HOME/.local/bin:$PATH"
step() { echo "--- $1 $(date -Is)"; }

step "venv"
[ -x "$PY" ] || uv venv "$ENV" --python /e/project1/transfernetx/lee27/code/envs/uv-python/cpython-3.12.13-linux-aarch64-gnu/bin/python3.12 || { echo "PREP_RC=10 venv"; exit 10; }
step "torch cu130"
uv pip install -p "$PY" torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130 || { echo "PREP_RC=11 torch"; exit 11; }
step "build deps"
uv pip install -p "$PY" packaging ninja cmake psutil setuptools-scm "setuptools<81" setuptools-rust jinja2 numpy regex wheel requests || { echo "PREP_RC=12 builddeps"; exit 12; }

step "source copy (rsync, no build/ or .so)"
mkdir -p "$SRC"
rsync -a --delete --exclude 'build/' --exclude '*.so' --exclude '.deps/' --exclude '__pycache__/' "$SRC0/" "$SRC/" || { echo "PREP_RC=13 rsync"; exit 13; }
cd "$SRC" && git rev-parse HEAD
[ -f use_existing_torch.py ] && $PY use_existing_torch.py || true

step "runtime deps (so the compute node can install with --no-deps --offline)"
uv pip install -p "$PY" -r requirements/cuda.txt || { echo "PREP_RC=14 runtime-deps"; exit 14; }
$PY -c "import torch; assert torch.version.cuda.startswith('13'), torch.version.cuda; print('torch ok', torch.__version__)" || { echo "torch replaced; reinstalling cu130"; uv pip install -p "$PY" torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130; }

step "cargo prefetch (Rust frontend crates into CARGO_HOME)"
module load Stages/2026 2>/dev/null || true
module load Rust/1.88.0 2>/dev/null || true
export CARGO_HOME=$F/cache/cargo; mkdir -p "$CARGO_HOME"
if command -v cargo >/dev/null; then
  for m in rust/Cargo.toml rust/src/cmd/Cargo.toml rust/src/tool-parser/python/Cargo.toml; do
    [ -f "$m" ] && { echo "cargo fetch $m"; cargo fetch --manifest-path "$m" || echo "CARGO_FETCH_FAIL $m (non-fatal: rust frontend is optional unless VLLM_REQUIRE_RUST_FRONTEND)"; }
  done
else
  echo "no cargo on PATH (rust frontend will be skipped/fail non-fatally on the compute node)"
fi

step "submit compute-node build"
jid=$(sbatch --parsable "$C/snowball/build_cn.sbatch") && echo "SUBMITTED build_cn job=$jid" || echo "PREP_RC=15 sbatch"
echo "PREP_RC=0"
echo "=== PREP END $(date -Is) ==="
