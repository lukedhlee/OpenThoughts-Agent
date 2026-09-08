#!/bin/bash
# Run probe.py inside an existing Vista allocation that is already serving.
#
#   srun --overlap --jobid=<id> bash $DCFT/scripts/tito_repro/drive_probe.sh --turns 16 ...
#
# Sets up the same env as run_probe_vista.sbatch and forwards all arguments to
# probe.py. Reuses the running server, so a sweep costs no extra model load.
set -uo pipefail

SCRATCH="${SCRATCH:-/scratch/11584/lukedhlee}"
DCFT="${DCFT:-$SCRATCH/OpenThoughts-Agent-tito}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
PORT="${PORT:-8000}"

module purge 2>/dev/null || true
module load gcc/13.2.0 cuda/12.8 2>/dev/null || true
source "$SCRATCH/miniconda3/etc/profile.d/conda.sh"
conda activate otagent
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME="${HF_HOME:-$SCRATCH/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$SCRATCH/hf_home/hub}"
export XDG_CACHE_HOME="$SCRATCH/.cache"
export TOKENIZERS_PARALLELISM=false

exec python "$DCFT/scripts/tito_repro/probe.py" \
    --base-url "http://127.0.0.1:${PORT}/v1" --model "$MODEL" "$@"
