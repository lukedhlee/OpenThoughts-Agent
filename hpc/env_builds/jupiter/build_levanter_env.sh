#!/bin/bash
# Build $F/envs/levanter — JAX cu13 (GH200 aarch64) + marin monorepo lib/levanter editable.
# For raw levanter.main.train_lm SFT on Jupiter (no iris/ArtifactStep).
set -uxo pipefail
F=/e/fscratch/reformo/lee27
export UV_CACHE_DIR=$F/cache/uv PIP_CACHE_DIR=$F/cache/pip
export UV_PYTHON_INSTALL_DIR=$F/cache/uv-python TMPDIR=$F/cache/tmp
export XDG_CACHE_HOME=$F/cache/xdg
UV=$HOME/.local/bin/uv
ENV=$F/envs/levanter
PY=$ENV/bin/python

[ -d $F/repos/marin ] || git clone --depth 1 https://github.com/marin-community/marin.git $F/repos/marin || exit 5

$UV venv $ENV --python 3.12 --clear || exit 10
$UV pip install -p $PY "jax[cuda13]" || $UV pip install -p $PY "jax[cuda12]" || exit 20
$UV pip install -p $PY -e $F/repos/marin/lib/levanter --prerelease=allow || exit 30

echo "=== import smoke (login node, CPU) ==="
JAX_PLATFORMS=cpu $PY - <<PYEOF
import jax; print("jax", jax.__version__)
import haliax, equinox; print("haliax", haliax.__version__)
import levanter; print("levanter OK")
from levanter.main.train_lm import TrainLmConfig; print("TrainLmConfig OK")
from levanter.models.qwen3_moe import Qwen3MoeConfig; print("Qwen3MoeConfig OK")
from levanter.data.text.formats import ChatLmDatasetFormat; print("ChatLmDatasetFormat OK")
PYEOF
echo "BUILD_DONE rc=$?"
