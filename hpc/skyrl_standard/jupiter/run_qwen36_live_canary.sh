#!/usr/bin/env bash
# Allocation-free live gate for one exact-model OpenCode/tool/verifier trial.
set -euo pipefail

: "${JSC_SCRATCH:=/e/scratch/reformo/lee27}"
: "${RUNTIME_ROOT:=$JSC_SCRATCH/OpenThoughts-Agent}"
: "${DCFT:=$JSC_SCRATCH/OpenThoughts-Agent-r2egym-bridge}"
: "${RL_VENV:=$RUNTIME_ROOT/envs/rl-megatron}"
: "${RL_REPO_DIR:=$JSC_SCRATCH/MarinSkyRL-apptainer-bridge}"
: "${HARBOR_REPO:=$JSC_SCRATCH/harbor-apptainer-bridge}"
: "${APPTAINER_BRIDGE_URL:=http://10.128.1.2:9920}"
: "${ROLLOUT_ENDPOINT:=http://jrlogin05i:18000/v1}"
: "${TASKS_DIR:=$JSC_SCRATCH/tasks/r2egym-patched-full-oracle}"
: "${RL_CONFIG:=$DCFT/hpc/skyrl_yaml/jupiter/6node_qwen3_6_35b_a3b_r2egym_grpo.yaml}"
: "${TRIALS_DIR:=$JSC_SCRATCH/experiments/qwen36_live_canary}"
: "${REQUIRE_CLEAN_BRIDGE:=1}"
: "${AGENT_TIMEOUT_SEC:=300}"

for required in \
  "$RL_VENV/bin/python" \
  "$DCFT/hpc/qwen36_live_canary.py" \
  "$RL_CONFIG" \
  "$RL_REPO_DIR/skyrl-train/examples/terminal_bench/harbor_config.py" \
  "$HARBOR_REPO/src/harbor/trial/trial.py" \
  "$TASKS_DIR"; do
  if [[ ! -e "$required" ]]; then
    echo "CANARY_FAIL[RUNTIME]: missing required path: $required" >&2
    exit 1
  fi
done

export APPTAINER_BRIDGE_URL
export PYTHONPATH="$HARBOR_REPO/src:$RL_REPO_DIR/skyrl-train:$DCFT${PYTHONPATH:+:$PYTHONPATH}"

exec "$RL_VENV/bin/python" "$DCFT/hpc/qwen36_live_canary.py" \
  --rl-config "$RL_CONFIG" \
  --tasks-dir "$TASKS_DIR" \
  --trials-dir "$TRIALS_DIR" \
  --endpoint "$ROLLOUT_ENDPOINT" \
  --bridge-url "$APPTAINER_BRIDGE_URL" \
  --require-clean-bridge "$REQUIRE_CLEAN_BRIDGE" \
  --agent-timeout-sec "$AGENT_TIMEOUT_SEC"
