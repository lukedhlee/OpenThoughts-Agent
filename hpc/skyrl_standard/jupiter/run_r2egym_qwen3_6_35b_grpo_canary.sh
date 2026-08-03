#!/usr/bin/env bash
# Intermediate five-node Qwen3.6 pipeline canary. This intentionally reuses the
# production launcher/preflight while selecting a standalone reduced config;
# success here does not replace the established six-node final gate.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${DCFT:=/e/scratch/reformo/lee27/OpenThoughts-Agent-r2egym-bridge}"

export RL_CONFIG="${RL_CONFIG:-$DCFT/hpc/skyrl_yaml/jupiter/5node_qwen3_6_35b_a3b_r2egym_grpo_canary.yaml}"
export NUM_NODES=5
export MAX_STEPS=1
export CKPT_INTERVAL=1
export HF_SAVE_INTERVAL=1
export MAX_RESTARTS=0
export JOB_NAME="${JOB_NAME:-jupiter_qwen36_35b_r2egym_grpo_canary}"

exec "$SCRIPT_DIR/run_r2egym_qwen3_6_35b_grpo.sh"
