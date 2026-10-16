#!/bin/bash
# build_relay_env.sh — the relay's additions to the snowball venv on Horizon (build_env.sbatch builds the venv first):
# python-socks in the venv (socks_connect_bridge.py runs on it), the --no-deps side dirs envs/relay-extra (torchvision, for
# the Qwen3.8 teacher serve) and envs/msa-2.4.6 (mini-swe-agent, AGENT=msa), and the HF snapshots serve_env.sh looks up.
# Downloads only, so it runs on the login node: ~1 min for the envs, the models longer. In a Slurm job it uses tunnel.sh's PORT.
#   bash data/relay/horizon/build_relay_env.sh [socks extra msa models]   (default all; knobs SNOWBALL_ROOT SCRATCH_DIR UV PY ENVS)
# By diff from ~/snowball/bootstrap/relay_prep.sh + dl_models.sh (2026-09-29): $HOME/$SCRATCH paths, pinned versions, msa, import checks.
set -uo pipefail
R=${SNOWBALL_ROOT:-$HOME/snowball}; S=${SCRATCH_DIR:-${SCRATCH:?SCRATCH is not set}}
UV=${UV:-$R/bin/uv}; PY=${PY:-$R/envs/snowball/bin/python}; ENVS=${ENVS:-$R/envs}
STEPS=${*:-socks extra msa models}
export UV_CACHE_DIR=$S/cache/snowball_build/uv HF_HOME=$S/cache/hf HF_HUB_CACHE=$S/hf_hub
export CC=gcc CXX=g++   # nvidia/26.9 sets CC=nvc; only matters if a package ever builds from source
if [ -n "${SLURM_JOB_ID:-}" ]; then   # compute node: no internet, only the login-side tunnel
  P=socks5h://127.0.0.1:${PORT:-18080}
  export ALL_PROXY=$P all_proxy=$P HTTPS_PROXY=$P https_proxy=$P HTTP_PROXY=$P http_proxy=$P NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
fi
[ -x $UV ] && [ -x $PY ] || { echo "no uv at $UV or no venv python at $PY (build_env.sbatch builds both)"; exit 1; }
fail=0
for step in $STEPS; do case $step in
  socks)
    echo "$(date -Is) python-socks"; $UV pip install -p $PY "python-socks[asyncio]==3.1.1" 2>&1 | tail -2 &&
      $PY -c "import python_socks.async_.asyncio as m; print('  import ok', m.__file__)" || fail=1;;
  extra)
    echo "$(date -Is) relay-extra torchvision"; $UV pip install -p $PY --target $ENVS/relay-extra --no-deps "torchvision==0.26.0" --index-url https://download.pytorch.org/whl/cu130 2>&1 | tail -2 &&
      PYTHONPATH=$ENVS/relay-extra OMP_NUM_THREADS=1 $PY -c "import torchvision as m; print('  import ok', m.__version__, m.__file__)" || fail=1;;
  msa)   # the modules harbor's mini-swe-relay agent imports; the rest of its deps come from the venv
    echo "$(date -Is) msa-2.4.6 mini-swe-agent"; $UV pip install -p $PY --target $ENVS/msa-2.4.6 --no-deps "mini-swe-agent==2.4.6" 2>&1 | tail -2 &&
      PYTHONPATH=$ENVS/msa-2.4.6 MSWEA_SILENT_STARTUP=1 LITELLM_LOCAL_MODEL_COST_MAP=True $PY -c "import minisweagent as m
from minisweagent.models.litellm_model import LitellmModel; from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel
from minisweagent.models.utils.actions_toolcall import BASH_TOOL; print('  import ok', m.__version__, m.__file__)" || fail=1;;
  models)
    echo "$(date -Is) teacher Qwen/Qwen3.8-27B"; $UV tool run --from "huggingface_hub[hf_xet]" hf download Qwen/Qwen3.8-27B --revision 1d4bf0f2 --cache-dir $S/hf_hub > /dev/null && echo "$(date -Is) teacher done" || { echo "$(date -Is) teacher FAIL"; fail=1; }
    for r in laion/snowball-64k-eagle3-draft-r2egym laion/snowball-67b-a2b-sft-s3-nemotron-terminal-step1888; do
      echo "$(date -Is) start $r"
      $UV tool run --from "huggingface_hub[hf_xet]" hf download $r --cache-dir $S/hf_hub > /dev/null && echo "$(date -Is) done $r" || { echo "$(date -Is) FAIL $r"; fail=1; }
    done;;
  *) echo "unknown step $step (socks extra msa models)"; exit 1;;
esac; done
[ $fail = 0 ] && echo PREP_DONE || { echo PREP_FAILED; exit 1; }
