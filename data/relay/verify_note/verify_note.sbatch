#!/bin/bash
# verify_note.sbatch — one GH200 node: Qwen3.8-27B as the relay teacher served it (data/relay/pilot/serve_node.sh
# teacher, one server per GPU, MTP 2, reasoning parser qwen3, the relay-extra torchvision side dir) at 131,072 context,
# then replay.py sends every done_claim takeover request A/B x 4 and the job exits (servers die with it).
# Cost: 1 node, --time 00:30 = hard ceiling 0.5 node-h (Luke's go, 2026-09-26).
#
#   sbatch verify_note.sbatch      (from a checkout of this repo on Jupiter; OTA=<checkout> overrides)
#
#SBATCH --job-name=relay_verify_note
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH -p booster
#SBATCH --account transfernetx
#SBATCH --output=/e/fscratch/reformo/lee27/experiments/relay/verify_note/logs/%x_%j.out
#SBATCH --exclude=jpbo-031-[01-48],jpbo-011-[01-48],jpbo-038-38,jpbo-004-46,jpbo-065-17,jpbo-074-22,jpbo-074-40,jpbo-048-41,jpbo-091-05,jpbo-044-0[1-5]
set -uo pipefail
export C=/e/project1/transfernetx/lee27/code
export PY=$C/envs/snowball-v2/bin/python
OTA=${OTA:-$C/ota-verify-note}
W=/e/fscratch/reformo/lee27/experiments/relay/verify_note; mkdir -p $W/logs
echo "verify_note job=$SLURM_JOB_ID node=$(hostname) ota=$(git -C $OTA rev-parse --short HEAD) ($(date -Is))"
PER_GPU=1 TEACHER_MAXLEN=131072 bash $OTA/data/relay/pilot/serve_node.sh teacher > $W/logs/serve_$SLURM_JOB_ID.log 2>&1 &
SP=$!
trap 'kill $SP 2>/dev/null; pkill -u $USER -f "vllm.entrypoints.openai.api_server" 2>/dev/null' EXIT
URLS=http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1,http://127.0.0.1:8003/v1
t=0
while :; do
  kill -0 $SP 2>/dev/null || { echo "serve_node exited before up"; exit 1; }
  up=1; for p in 8000 8001 8002 8003; do curl -sf --max-time 5 localhost:$p/health >/dev/null || up=0; done
  [ $up = 1 ] && { echo "servers UP after ${t}s"; break; }
  [ $t -ge 1200 ] && { echo "not up after 1200s"; exit 1; }
  sleep 10; t=$((t+10))
done
curl -s --max-time 300 localhost:8000/v1/models | head -c 400; echo
$PY $OTA/data/relay/verify_note/replay.py --requests $W/requests.jsonl.gz --out $W/replies.jsonl --urls $URLS
echo "replay exit $? ($(date -Is))"
