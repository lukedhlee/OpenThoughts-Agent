#!/bin/bash
# gate3_horizon.sbatch — Horizon port gate 3: EAGLE-3 acceptance on the 32k held-out replay, the exact method of Jupiter's
# reference (toolkit smoke job 1762082, 2026-09-11: acceptance 2.59): tools/spec_decode/scripts/benchmark.py (identical to
# the snapshot that ran), toolkit serve DP4 EP, max-model-len 32768, k=3, replay_b28k.jsonl, concurrency 16, 48
# trajectories, 120 s, steady after 30 s, 4 parity turns. Both arms serve the public draft
# (laion/snowball-64k-eagle3-draft-r2egym == the reference's release dir, sha256 538e6800), so one job gives two
# independent acceptance readings. ~0.4 node-hours; exits when results.json is complete.
#
#   sbatch data/r2egym/horizon/gate3_horizon.sbatch        (from ~/snowball/ota; replay at $S/horizon_gates/replay_b28k.jsonl)
#SBATCH -J sb_gate3
#SBATCH -A CCR24067
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 00:50:00
#SBATCH -o /scratch/11584/lukedhlee/horizon_gates/logs/%x.%j.out
set -uo pipefail
ROOT=${ROOT:-$HOME/snowball}; S=/scratch/11584/$USER
OTA=${SLURM_SUBMIT_DIR:-$ROOT/ota}
PY=$ROOT/envs/snowball/bin/python
export PYTHONPATH=$ROOT/src/marin_vllm_eagle3${PYTHONPATH:+:$PYTHONPATH}   # the EAGLE-3 overlay shadows the venv's vllm
export PATH=$ROOT/envs/snowball/bin:$PATH                                     # ninja for FlashInfer JIT
snap() { ls -d $S/hf_hub/models--laion--$1/snapshots/* | head -1; }
MODEL=${MODEL:-$(snap snowball-67b-a2b-sft-s3-nemotron-terminal-step1888)}
DRAFT=${DRAFT:-$(snap snowball-64k-eagle3-draft-r2egym)}
G=$S/horizon_gates; REPLAY=${REPLAY:-$G/replay_b28k.jsonl}; mkdir -p $G/logs
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1   # the reference job's env
export CC=gcc CXX=g++   # TACC's login env sets CC=nvc, which Triton's JIT rejects (-Wno-psabi)
[ -f "$REPLAY" ] || { echo "no replay at $REPLAY"; exit 1; }
echo "node=$(hostname) job=$SLURM_JOB_ID model=$MODEL draft=$DRAFT ota=$(git -C $OTA rev-parse --short HEAD)"
sha256sum $REPLAY $DRAFT/model.safetensors
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
$PY $OTA/tools/spec_decode/scripts/benchmark.py --target "$MODEL" --draft "$DRAFT" --baseline-draft "$DRAFT" \
    --replay $REPLAY --out $G/gate3_horizon.$SLURM_JOB_ID --cache $S/cache --python $PY \
    --max-model-len 32768 --seconds 120 --steady-after 30 --parity-turns 4 --repeats 1 > $G/logs/gate3_bench.$SLURM_JOB_ID.log 2>&1
RC=$?
$PY -c "import json,sys; r=json.load(open(sys.argv[1])); [print('GATE3 arm=%s acceptance=%.4f decode_tps_p50=%.1f buckets=%s' % (a['kind'], a['acceptance_length'], a['decode_tps_p50'], json.dumps(a['prompt_buckets']))) for a in r['arms']]" $G/gate3_horizon.$SLURM_JOB_ID/results.json
echo "=== DONE rc=$RC $(date -Is)"
