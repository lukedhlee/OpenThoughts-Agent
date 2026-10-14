#!/bin/bash
# probe.sbatch — serve one export (MODEL) as cf-student on this node and run probe_export.py against GENERATIONS.
#   sbatch -N 1 -t 01:00:00 -o <out>/probe_%j.out --export=ALL,MODEL=<hf dir>,GENERATIONS=<gen dir>,OUT=<json>,REFLECTION_OTA=<ota> probe.sbatch
#SBATCH -p debug
#SBATCH -A CCR24067
#SBATCH --exclusive
#SBATCH --gres=gpu:4
set -uo pipefail
OTA=${REFLECTION_OTA:?}; export REFLECTION_OTA=$OTA REFLECTION_SERVE_DIR=$(dirname "$OUT")/probe_serve_$SLURM_JOB_ID
mkdir -p "$REFLECTION_SERVE_DIR"; export CC=gcc CXX=g++
trap 'bash $OTA/data/relay/reflection/horizon/serve_model.sh stop student' EXIT
bash "$OTA/data/relay/reflection/horizon/serve_model.sh" refresh student "$MODEL" 0 || exit 1
cd "$OTA" && PYTHONPATH=$OTA $HOME/snowball/envs/snowball/bin/python -m data.relay.reflection.horizon.probe_export \
  --generations "$GENERATIONS" --n ${N:-32} --out "$OUT"
echo "PROBE_DONE rc=$?"
