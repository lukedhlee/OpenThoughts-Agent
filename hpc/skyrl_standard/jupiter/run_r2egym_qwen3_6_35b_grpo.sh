#!/usr/bin/env bash
# Qwen3.6-35B-A3B, initialized from the exact pinned plain checkpoint.
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
: "${MODEL_REVISION:=995ad96eacd98c81ed38be0c5b274b04031597b0}"
: "${MODEL_PATH:=$JSC_SCRATCH/models/Qwen3.6-35B-A3B/$MODEL_REVISION}"
: "${TASKS_DIR:=$JSC_SCRATCH/tasks/r2egym-patched-full-oracle}"
: "${EXPERIMENTS_DIR:=$JSC_SCRATCH/experiments}"
: "${RL_CONFIG:=$DCFT/hpc/skyrl_yaml/jupiter/6node_qwen3_6_35b_a3b_r2egym_grpo.yaml}"
: "${PARTITION:=booster}"
: "${ACCOUNT:=reformo}"
: "${TIME_LIMIT:=02:00:00}"
: "${NUM_NODES:=6}"
: "${MAX_RESTARTS:=0}"
: "${MAX_STEPS:=1}"
: "${CKPT_INTERVAL:=1}"
: "${HF_SAVE_INTERVAL:=1}"
: "${PREFLIGHT_ONLY:=0}"
: "${JOB_NAME:=jupiter_qwen36_35b_r2egym_grpo_smoke}"
: "${HF_HUB_REPO_ID:=}"
: "${HF_HUB_PRIVATE:=false}"

# The login-node preflight imports the compiled vLLM extension before Slurm's
# generated sbatch can load the Jupiter runtime modules.
module load GCC/14.3.0
module load nvidia-compilers/25.9-CUDA-13

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
  "$MODEL_PATH/config.json"; do
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
import torch.nn as nn
import transformers
import vllm
import vllm._C
from transformers import AutoConfig, AutoModelForCausalLM

from skyrl_train.distributed.fsdp_utils import get_fsdp_wrap_policy

if Version(transformers.__version__) < Version("5.8.1"):
    raise SystemExit(f"transformers>=5.8.1 required, got {transformers.__version__}")
if Version(vllm.__version__) < Version("0.19.0"):
    raise SystemExit(f"vllm>=0.19 required, got {vllm.__version__}")
# The staged tree is a verbatim hub snapshot pinned by revision in its path, so
# the revision check is on the directory name rather than a generated provenance
# file. Both eval arms must resolve to this exact revision or the paired
# comparison is meaningless.
assert model.name == expected_revision, (
    f"staged model dir {model.name!r} is not the pinned revision {expected_revision!r}"
)
config = AutoConfig.from_pretrained(model, local_files_only=True)
assert str(config.model_type).startswith("qwen3_5"), config.model_type
assert not getattr(config, "quantization_config", None), (
    "expected the unquantized release; a quantization_config means a quantized repo was staged"
)
# We deliberately stage the multimodal shell and let SkyRL unwrap it in memory
# (SKYRL_QWEN3_5_VLM_UNWRAP=1). The GDN signature therefore lives one level down,
# on text_config, not on the top-level config.
text_config = getattr(config, "text_config", None)
assert text_config is not None, (
    "expected the Qwen3.6 multimodal shell with a nested text_config; got a bare "
    "text tower. If a pre-unwrapped checkpoint is staged on purpose, set "
    "SKYRL_QWEN3_5_VLM_UNWRAP=0 in the RL config to match."
)
assert hasattr(text_config, "linear_conv_kernel_dim"), (
    "text_config lacks the GatedDeltaNet signature; SkyRL's unwrap probe would "
    "not fire and FlashQLA would be skipped"
)
layer_types = getattr(text_config, "layer_types", None)
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
    dist.metadata["Name"].lower().replace("_", "-"): dist.version
    for dist in distributions(path=[str(flashqla_layer)])
}
assert found_packages == expected_packages, found_packages
smoke = (flashqla_layer / "gpu_sm90_smoke.ok").read_text()
assert "format=qwen3.6-flashqla-smoke-v1" in smoke, smoke
importlib.import_module("skyrl_train.models.qwen3_5_vlm")
gdn = importlib.import_module("skyrl_train.models.qwen3_next_gdn")
assert "Qwen3_5MoeGatedDeltaNet" in gdn._FLASHQLA_GDN_TYPES

# Transformers 5.12 advertises a vision no-split class on the text CausalLM,
# even though the RL policy contains only decoder layers. Exercise SkyRL's
# resolver before reserving six nodes: it must retain the present decoder class
# while ignoring the absent optional vision class.
text_model_cls = AutoModelForCausalLM._model_mapping[type(text_config)]
no_split_modules = list(text_model_cls._no_split_modules)
decoder_cls_name = next(
    name for name in no_split_modules if name.endswith("DecoderLayer")
)
decoder_cls = type(decoder_cls_name, (nn.Module,), {})
wrap_probe = nn.Module()
wrap_probe._no_split_modules = no_split_modules
wrap_probe.add_module("decoder", decoder_cls())
assert get_fsdp_wrap_policy(wrap_probe) is not None
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
TASK_COUNT="$("$RL_VENV/bin/python" - "$TASKS_DIR" <<'PY'
import os
import sys

root = sys.argv[1]
with os.scandir(root) as entries:
    count = sum(
        entry.is_dir() and os.path.isfile(os.path.join(entry.path, "instruction.md"))
        for entry in entries
    )
print(count)
PY
)"
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
