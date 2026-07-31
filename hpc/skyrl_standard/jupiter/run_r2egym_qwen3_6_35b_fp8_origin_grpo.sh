#!/usr/bin/env bash
# Qwen3.6-35B-A3B, initialized from the exact pinned FP8 checkpoint.
# Default is a one-step, six-node compatibility smoke. Promote with
# MAX_STEPS=50 only after the smoke writes a valid HF checkpoint and reward logs.
set -euo pipefail
shopt -s nullglob

: "${JSC_SCRATCH:=/e/scratch/reformo/lee27}"
: "${RUNTIME_ROOT:=$JSC_SCRATCH/OpenThoughts-Agent}"
: "${DCFT:=$JSC_SCRATCH/OpenThoughts-Agent-r2egym-bridge}"
: "${RL_VENV:=$RUNTIME_ROOT/envs/rl-megatron}"
: "${RL_REPO_DIR:=$JSC_SCRATCH/MarinSkyRL-apptainer-bridge}"
: "${HARBOR_REPO:=$JSC_SCRATCH/harbor-apptainer-bridge}"
: "${FLASHQLA_LAYER:=$JSC_SCRATCH/pydeps/qwen36-flashqla-0.1.2}"
: "${APPTAINER_BRIDGE_URL:=http://10.128.1.2:9920}"
: "${MODEL_REVISION:=95a723d08a9490559dae23d0cff1d9466213d989}"
: "${MODEL_PATH:=$JSC_SCRATCH/models/Qwen3.6-35B-A3B-FP8-origin-bf16-text/$MODEL_REVISION}"
: "${TASKS_DIR:=$JSC_SCRATCH/tasks/r2egym-patched-full-oracle}"
: "${EXPERIMENTS_DIR:=$JSC_SCRATCH/experiments}"
: "${RL_CONFIG:=$DCFT/hpc/skyrl_yaml/jupiter/6node_qwen3_6_35b_a3b_fp8_origin_r2egym_grpo.yaml}"
: "${PARTITION:=booster}"
: "${ACCOUNT:=reformo}"
: "${TIME_LIMIT:=02:00:00}"
: "${NUM_NODES:=6}"
: "${MAX_RESTARTS:=0}"
: "${MAX_STEPS:=1}"
: "${CKPT_INTERVAL:=1}"
: "${HF_SAVE_INTERVAL:=1}"
: "${PREFLIGHT_ONLY:=0}"
: "${JOB_NAME:=jupiter_qwen36_35b_fp8_origin_r2egym_grpo_smoke}"
: "${HF_HUB_REPO_ID:=}"
: "${HF_HUB_PRIVATE:=false}"

export SCRATCH="$JSC_SCRATCH"
export DCFT DCFT_RL_ENV="$RL_VENV" RL_ENV_DIR="$RL_VENV"
export DC_AGENT_SECRET_ENV="${DC_AGENT_SECRET_ENV:-$JSC_SCRATCH/keys/secrets.env}"
export HF_HOME="${HF_HOME:-$JSC_SCRATCH/cache/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export WANDB_DIR="${WANDB_DIR:-$JSC_SCRATCH/wandb}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-jupiter-r2egym-grpo-qwen36-35b}"
export RL_REPO_DIR SKYRL_HOME="$RL_REPO_DIR" APPTAINER_BRIDGE_URL
export PYTHONPATH="$FLASHQLA_LAYER:$HARBOR_REPO/src:$RL_REPO_DIR/skyrl-train:$RL_REPO_DIR/skyrl-gym:$DCFT${PYTHONPATH:+:$PYTHONPATH}"

for required in \
  "$RL_VENV/bin/python" \
  "$RL_CONFIG" \
  "$HARBOR_REPO/src/harbor/environments/apptainer/apptainer.py" \
  "$RL_REPO_DIR/skyrl-train/skyrl_train/models/qwen3_5_vlm.py" \
  "$FLASHQLA_LAYER/qwen3_6_flashqla_manifest.json" \
  "$FLASHQLA_LAYER/gpu_sm90_smoke.ok" \
  "$MODEL_PATH/config.json" \
  "$MODEL_PATH/fp8_origin_provenance.json"; do
  if [[ ! -e "$required" ]]; then
    echo "ERROR: missing required path: $required" >&2
    exit 1
  fi
done

if ! curl --fail --silent --show-error --max-time 10 \
  "$APPTAINER_BRIDGE_URL/status" >/dev/null; then
  echo "ERROR: Apptainer bridge is unhealthy: $APPTAINER_BRIDGE_URL" >&2
  echo "A disconnected bridge can otherwise masquerade as zero reward." >&2
  exit 1
fi

"$RL_VENV/bin/python" - "$MODEL_PATH" "$MODEL_REVISION" "$FLASHQLA_LAYER" <<'PY'
import importlib
import json
import sys
from importlib.metadata import distributions
from pathlib import Path

from packaging.version import Version

model = Path(sys.argv[1])
expected_revision = sys.argv[2]
flashqla_layer = Path(sys.argv[3])
import torch
import transformers
import vllm
import vllm._C
from transformers import AutoConfig

if Version(transformers.__version__) < Version("5.8.1"):
    raise SystemExit(f"transformers>=5.8.1 required, got {transformers.__version__}")
if Version(vllm.__version__) < Version("0.19.0"):
    raise SystemExit(f"vllm>=0.19 required, got {vllm.__version__}")
provenance = json.loads((model / "fp8_origin_provenance.json").read_text())
assert provenance["source_repo"] == "Qwen/Qwen3.6-35B-A3B-FP8", provenance
assert provenance["source_revision"] == expected_revision, provenance
assert provenance["format"] == "qwen3.6-fp8-origin-bf16-text-v1", provenance
config = AutoConfig.from_pretrained(model, local_files_only=True)
assert str(config.model_type).startswith("qwen3_5"), config.model_type
assert not getattr(config, "quantization_config", None)
assert not hasattr(config, "text_config"), "checkpoint is still a VLM shell"
layer_types = getattr(config, "layer_types", None)
assert layer_types and any("linear" in str(kind).lower() for kind in layer_types), (
    "text config does not expose its GDN layer_types; SkyRL would skip FlashQLA"
)
manifest = json.loads(
    (flashqla_layer / "qwen3_6_flashqla_manifest.json").read_text()
)
expected_packages = {
    "apache-tvm-ffi": "0.1.9",
    "flash-qla": "0.1.2",
    "tilelang": "0.1.9",
    "torch-c-dlpack-ext": "0.1.5",
    "z3-solver": "4.15.4.0",
}
assert manifest["format"] == "qwen3.6-flashqla-layer-v1", manifest
assert manifest["packages"] == expected_packages, manifest
found_packages = {
    dist.metadata["Name"].lower(): dist.version
    for dist in distributions(path=[str(flashqla_layer)])
}
assert found_packages == expected_packages, found_packages
smoke = (flashqla_layer / "gpu_sm90_smoke.ok").read_text()
assert "format=qwen3.6-flashqla-smoke-v1" in smoke, smoke
importlib.import_module("skyrl_train.models.qwen3_5_vlm")
gdn = importlib.import_module("skyrl_train.models.qwen3_next_gdn")
assert "Qwen3_5MoeGatedDeltaNet" in gdn._FLASHQLA_GDN_TYPES
print(
    "Qwen3.6 runtime preflight passed:",
    f"torch={torch.__version__}",
    f"transformers={transformers.__version__}",
    f"vllm={vllm.__version__}",
    f"model_type={config.model_type}",
)
PY

if [[ ! -d "$TASKS_DIR" ]]; then
  echo "ERROR: extracted r2egym task directory is missing: $TASKS_DIR" >&2
  exit 1
fi
TASK_COUNT="$(find "$TASKS_DIR" -mindepth 2 -maxdepth 2 -name instruction.md -print | wc -l)"
if [[ "$TASK_COUNT" -ne 3328 ]]; then
  echo "ERROR: expected 3,328 extracted r2egym tasks, found $TASK_COUNT" >&2
  exit 1
fi

if [[ "$MAX_STEPS" -gt 1 && "$JOB_NAME" == *smoke* ]]; then
  echo "ERROR: set a non-smoke JOB_NAME when promoting beyond one step" >&2
  exit 1
fi
if [[ "$MAX_STEPS" -gt 1 && "$MAX_RESTARTS" -lt 1 ]]; then
  echo "ERROR: a promoted run requires MAX_RESTARTS>=1" >&2
  exit 1
fi

mkdir -p "$EXPERIMENTS_DIR" "$WANDB_DIR"
if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  echo "Qwen3.6 GRPO preflight complete; no job submitted."
  echo "  model: $MODEL_PATH"
  echo "  tasks: $TASKS_DIR ($TASK_COUNT)"
  echo "  bridge: $APPTAINER_BRIDGE_URL"
  echo "  steps: $MAX_STEPS"
  exit 0
fi

cd "$DCFT"
HF_ARGS=()
if [[ -n "$HF_HUB_REPO_ID" ]]; then
  HF_ARGS=(--hf_hub_repo_id "$HF_HUB_REPO_ID" --hf_hub_private "$HF_HUB_PRIVATE")
fi

"$RL_VENV/bin/python" -m hpc.launch \
  --job_type rl \
  --rl_config "$RL_CONFIG" \
  --model_path "$MODEL_PATH" \
  --train_data "[\"$TASKS_DIR\"]" \
  --num_nodes "$NUM_NODES" \
  --partition "$PARTITION" \
  --account "$ACCOUNT" \
  --time_limit "$TIME_LIMIT" \
  --max_restarts "$MAX_RESTARTS" \
  --experiments_dir "$EXPERIMENTS_DIR" \
  --job_name "$JOB_NAME" \
  ${HF_ARGS[@]+"${HF_ARGS[@]}"} \
  --skyrl_override trainer.max_steps="$MAX_STEPS" \
  --skyrl_override trainer.ckpt_interval="$CKPT_INTERVAL" \
  --skyrl_override trainer.hf_save_interval="$HF_SAVE_INTERVAL"
