#!/usr/bin/env bash
# Learnable-band p@4 probe: DCAgent/g1_diverse_tezos_100k_8b over r2egym.
#
# PURE ROLLOUT. entrypoint=main_tbench_generate builds vLLM engines and calls
# generator.generate() once over every prompt in the shard. There is no policy
# model, no ref model, no optimizer, no FSDP, no weight sync and no checkpoint,
# so the ~8 serial SPOFs that have cost us nine training runs are simply absent.
#
# Why this model: it is the exact checkpoint the co-lead's band/raw comparison
# used (Qwen3-8B SFT), and the band is a property of the BASE POLICY -- a band
# measured on a different model does not transfer as a measurement.
#
# DELIBERATELY NOT PASSED:
#   --train_data  : rl_launch_utils overwrites data.train_data from the CLI, which
#                   would replace the shard's explicit task list with the whole
#                   3,328-dir tree. The shard list lives in the YAML; keep it.
#   --model_path  : same trap in the other direction (it overrides the YAML path).
set -euo pipefail

: "${JSC_SCRATCH:=/e/scratch/reformo/lee27}"
: "${FSCRATCH:=/e/fscratch/reformo/lee27}"
: "${RUNTIME_ROOT:=$JSC_SCRATCH/OpenThoughts-Agent}"
# /e/fscratch, NOT /e/scratch (inode-exhausted, cannot be fetched into) and NOT
# /p/scratch (not mounted on Jupiter compute).
: "${DCFT:=$FSCRATCH/OpenThoughts-Agent-r2egym-bridge-next}"
: "${RL_VENV:=$RUNTIME_ROOT/envs/rl-megatron}"
: "${RL_REPO_DIR:=$JSC_SCRATCH/MarinSkyRL-apptainer-bridge}"
: "${HARBOR_REPO:=$JSC_SCRATCH/harbor-apptainer-bridge}"
# Probe bridge on 9921 -- the 9920 bridge is serving the 35B milestone run.
: "${APPTAINER_BRIDGE_URL:=http://10.128.1.2:9921}"
: "${EXPERIMENTS_DIR:=$FSCRATCH/experiments}"
: "${PARTITION:=booster}"
: "${ACCOUNT:=reformo}"
: "${TIME_LIMIT:=03:00:00}"
: "${NUM_NODES:=1}"
: "${MAX_RESTARTS:=0}"
: "${SHARD:?Set SHARD to the shard yaml basename, e.g. band_probe_8b_p4_shard00of416}"
: "${RL_CONFIG:=$DCFT/hpc/skyrl_yaml/jupiter/band/${SHARD}.yaml}"
: "${JOB_NAME:=jupiter_band_probe_8b_${SHARD}}"

export SCRATCH="$JSC_SCRATCH"
export DCFT
export DCFT_RL_ENV="$RL_VENV"
export RL_ENV_DIR="$RL_VENV"
export DC_AGENT_SECRET_ENV="${DC_AGENT_SECRET_ENV:-$JSC_SCRATCH/keys/secrets.env}"
export HF_HOME="${HF_HOME:-$JSC_SCRATCH/cache/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export WANDB_DIR="${WANDB_DIR:-$FSCRATCH/wandb}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export RL_REPO_DIR
export SKYRL_HOME="$RL_REPO_DIR"
export APPTAINER_BRIDGE_URL
export PYTHONPATH="$HARBOR_REPO/src:$SKYRL_HOME/skyrl-train:$SKYRL_HOME/skyrl-gym:$DCFT${PYTHONPATH:+:$PYTHONPATH}"

for f in "$RL_VENV/bin/python" "$RL_CONFIG" "$DC_AGENT_SECRET_ENV"; do
  [[ -e "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }
done
# A disconnected bridge looks exactly like a dataset of zero-reward tasks, which
# would silently corrupt the band. Refuse to launch without it.
if ! curl --fail --silent --show-error --max-time 10 "$APPTAINER_BRIDGE_URL/status" >/dev/null; then
  echo "ERROR: probe bridge unhealthy at $APPTAINER_BRIDGE_URL" >&2
  exit 1
fi
echo "Probe bridge preflight passed: $APPTAINER_BRIDGE_URL"

set -a; source "$DC_AGENT_SECRET_ENV"; set +a
mkdir -p "$EXPERIMENTS_DIR" "$WANDB_DIR"
cd "$DCFT"

"$RL_VENV/bin/python" -m hpc.launch \
  --job_type rl \
  --rl_config "$RL_CONFIG" \
  --num_nodes "$NUM_NODES" \
  --partition "$PARTITION" \
  --account "$ACCOUNT" \
  --time_limit "$TIME_LIMIT" \
  --max_restarts "$MAX_RESTARTS" \
  --experiments_dir "$EXPERIMENTS_DIR" \
  --job_name "$JOB_NAME"
