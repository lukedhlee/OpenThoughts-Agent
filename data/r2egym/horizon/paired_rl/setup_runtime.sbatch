#!/usr/bin/env bash
#SBATCH --job-name=paired-rl-runtime
#SBATCH --partition=debug
#SBATCH --account=CCR24067
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00

set -euo pipefail
unset LD_PRELOAD
export CC=gcc CXX=g++ OMP_NUM_THREADS=1
ROOT=${PAIR_ROOT:?Set PAIR_ROOT to the isolated experiment directory}
SOURCE=$ROOT/code/MarinSkyRL
EXPECTED=82396a73a
test "$(git -C "$SOURCE" rev-parse --short=9 HEAD)" = "$EXPECTED"
mkdir -p "$ROOT/logs" "$ROOT/cache/uv" "$ROOT/gateway"
printf 'INSTALLING_RUNTIME\n' > "$ROOT/STATUS"
finish() {
    code=$?
    if [ "$code" -ne 0 ]; then printf 'RUNTIME_FAILED rc=%s\n' "$code" > "$ROOT/STATUS"; fi
    if [ -n "${BRIDGE_PID:-}" ]; then kill "$BRIDGE_PID" 2>/dev/null || true; fi
}
trap finish EXIT
bash "$HOME/snowball/ota-rl/data/r2egym/horizon/rl/node_bridge.sh" "$ROOT/gateway" 18126 18946 &
BRIDGE_PID=$!
deadline=$((SECONDS + 900))
while ! test -f "$ROOT/gateway/ready-$(hostname -s)"; do
    kill -0 "$BRIDGE_PID"
    test "$SECONDS" -lt "$deadline"
    sleep 3
done
export HTTPS_PROXY=http://127.0.0.1:18946
export HTTP_PROXY=$HTTPS_PROXY
export NO_PROXY=localhost,127.0.0.1
export UV_CACHE_DIR=$ROOT/cache/uv
export UV_PROJECT_ENVIRONMENT=$ROOT/venv
export UV_PYTHON_DOWNLOADS=never
export UV_LINK_MODE=copy
PYTHON=$HOME/snowball/envs/snowball/bin/python
cd "$SOURCE"
"$HOME/snowball/bin/uv" sync --frozen --extra vllm --extra megatron --no-default-groups --no-build --python "$PYTHON"
"$ROOT/venv/bin/python" - <<'PY'
import importlib.metadata
import json
import os
from pathlib import Path
import torch
import skyrl_train
import skyrl_gym
import vllm
import megatron.core

root = Path(os.environ['PAIR_ROOT'])
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 4
report = {
    'torch': torch.__version__,
    'cuda': torch.version.cuda,
    'gpus': torch.cuda.device_count(),
    'gpu_name': torch.cuda.get_device_name(0),
    'vllm': importlib.metadata.version('vllm'),
    'megatron_core': importlib.metadata.version('megatron-core'),
    'skyrl_train': skyrl_train.__file__,
    'skyrl_gym': skyrl_gym.__file__,
}
(root / 'runtime.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
PY
printf 'RUNTIME_IMPORTS_PASSED: model/update smoke still required\n' > "$ROOT/STATUS"
