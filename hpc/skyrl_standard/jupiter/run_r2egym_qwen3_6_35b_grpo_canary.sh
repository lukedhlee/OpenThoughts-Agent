#!/usr/bin/env bash
# Intermediate five-node Qwen3.6 pipeline canary. This intentionally reuses the
# production launcher/preflight while selecting a standalone reduced config;
# success here does not replace the established six-node final gate.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# `-next` is the deployed tree; the bare path is the STALE 0f04b250 checkout.
: "${DCFT:=/e/fscratch/reformo/lee27/OpenThoughts-Agent-r2egym-bridge-next}"

export RL_CONFIG="${RL_CONFIG:-$DCFT/hpc/skyrl_yaml/jupiter/5node_qwen3_6_35b_a3b_r2egym_grpo_canary.yaml}"
export NUM_NODES=5
# 6h, not the parent's 2h default. The canary now collects 16x8 = 128
# trajectories (was 64) with the agent budget restored to the full 1800s (was
# silently capped at 600s), so the rollout phase can take ~4 waves x up to 30
# min at 32-way concurrency. On 1221005 the 3h wall left only ~40 min after
# setup, and its shards->policy-init phase alone took 110 min.
: "${TIME_LIMIT:=06:00:00}"
export TIME_LIMIT
export MAX_STEPS=1
export CKPT_INTERVAL=1
export HF_SAVE_INTERVAL=1
export MAX_RESTARTS=0
export JOB_NAME="${JOB_NAME:-jupiter_qwen36_35b_r2egym_grpo_canary}"

exec "$SCRIPT_DIR/run_r2egym_qwen3_6_35b_grpo.sh"
