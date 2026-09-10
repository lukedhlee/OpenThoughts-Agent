#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --output=/e/fscratch/reformo/lee27/experiments/exports/logs/%x_%j.out
#SBATCH --job-name=snowball_export_hf
#SBATCH -p booster
#SBATCH --account reformo
#SBATCH --gres=gpu:4
# export_hf.sbatch — offline FSDP2 -> HF consolidation of a Snowball GRPO checkpoint (CPU only, one node, ~858 GB host RAM).
# Usage: RUN=<run name> STEP=<N> sbatch --export=ALL,RUN=...,STEP=... export_hf.sbatch
#   -> /e/fscratch/reformo/lee27/experiments/exports/<RUN>_step<N>/model  (config/tokenizer copied from the base weights)
# Route A of runbooks/rl-training.md § "FSDP -> HF export". Runs --inspect first; aborts before writing on any layout it cannot reassemble.
set -euo pipefail
module load GCC/14.3.0
module load nvidia-compilers/25.9-CUDA-13
PY=/e/project1/transfernetx/lee27/code/envs/snowball/bin/python
SCRIPT=/e/project1/transfernetx/lee27/code/MarinSkyRL/skyrl-train/scripts/consolidate_sharded_checkpoint.py
BASE=/e/fscratch/reformo/lee27/models/snowball-s3-nemotron-terminal-step1888
E=/e/fscratch/reformo/lee27/experiments
: "${RUN:?set RUN}"; : "${STEP:?set STEP}"
SRC=${SRC:-$E/$RUN/$RUN/checkpoints/global_step_$STEP/policy}
DST=${DST:-$E/exports/${RUN}_step${STEP}/model}
echo "$(date) RUN=$RUN STEP=$STEP SRC=$SRC DST=$DST host=$(hostname)"; ls "$SRC" | head -5; ls "$SRC" | wc -l
du -sh "$SRC" 2>/dev/null | tail -1
echo "=== inspect ==="; $PY "$SCRIPT" --src "$SRC" --inspect
echo "=== consolidate ==="; mkdir -p "$(dirname "$DST")"; ( time $PY "$SCRIPT" --src "$SRC" --dst "$DST" --aux-from "$BASE" ) 2>&1 | tail -60
echo "=== verify ==="; ls "$DST" | wc -l; du -sh "$DST" | tail -1
$PY - "$DST" "$BASE" <<'PY'
import json, os, sys
dst, base = sys.argv[1], sys.argv[2]
idx = json.load(open(os.path.join(dst, "model.safetensors.index.json")))
files = set(idx["weight_map"].values()); missing = [f for f in files if not os.path.exists(os.path.join(dst, f))]
print("index tensors:", len(idx["weight_map"]), "shard files:", len(files), "missing files:", missing)
cfg = json.load(open(os.path.join(dst, "config.json"))); bcfg = json.load(open(os.path.join(base, "config.json")))
print("model_type:", cfg.get("model_type"), "architectures:", cfg.get("architectures"), "max_position_embeddings:", cfg.get("max_position_embeddings"), "tie:", cfg.get("tie_word_embeddings"))
diff = {k: (bcfg.get(k), cfg.get(k)) for k in set(bcfg) | set(cfg) if bcfg.get(k) != cfg.get(k)}
print("config diffs vs base:", diff)
try:
    bidx = json.load(open(os.path.join(base, "model.safetensors.index.json")))
    a, b = set(idx["weight_map"]), set(bidx["weight_map"]); print("tensor-name set equal to base:", a == b, "only-export:", sorted(a - b)[:5], "only-base:", sorted(b - a)[:5])
except Exception as e: print("base index compare skipped:", e)
for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
    print(f, os.path.exists(os.path.join(dst, f)))
PY
echo "$(date) done"
