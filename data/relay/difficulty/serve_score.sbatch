#!/bin/bash
#SBATCH -p debug
#SBATCH -A CCR24067
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 01:30:00
#SBATCH -J sdscore
# serve_score.sbatch — serve_hold.sbatch's student serve, then score.py over the renders, then release the node.
# For checkpoints that only need scoring (e.g. SFT exports): chain it after the export with --dependency=afterok.
#   STUDENT_MODEL=<dir> TAG=<score set> [CONDITIONS="full_history norefeed"] sbatch -o <exp>/logs/score_%j.out serve_score.sbatch
set -eo pipefail
unset LD_PRELOAD
OTA=${OTA:-$HOME/snowball/ota-difficulty}
EXP=${EXP:-/scratch/11584/$USER/experiments/student_difficulty}
source $OTA/data/relay/horizon/serve_env.sh
export STUDENT_MODEL=${STUDENT_MODEL:?STUDENT_MODEL=<checkpoint dir>} STUDENT_DRAFT=none
export STUDENT_MAXLEN=${STUDENT_MAXLEN:-131072}
TAG=${TAG:?TAG=<score set name>}
echo "serve_score job=$SLURM_JOB_ID node=$(hostname) model=$STUDENT_MODEL tag=$TAG ota=$(git -C $OTA rev-parse HEAD)"
setsid bash $OTA/data/relay/pilot/serve_node.sh student &
pg=$!
for _ in $(seq 1 90); do curl -sf --max-time 5 localhost:8000/health >/dev/null && break; sleep 10; done
curl -sf --max-time 5 localhost:8000/health >/dev/null || { echo SERVE_NOT_READY; kill -TERM -- -$pg; exit 1; }
echo "SERVE_READY $(date -Is)"
rc=0
for c in ${CONDITIONS:-full_history}; do
  out=$EXP/scores/$TAG; [ "$c" = full_history ] || out=${out}_$c
  $SERVE_VENV/bin/python $OTA/data/relay/difficulty/score.py --renders $EXP/renders --out $out --tag $TAG \
    --model-path $STUDENT_MODEL --condition $c --serve-args "serve_node.sh student, no draft, max-model-len $STUDENT_MAXLEN" || rc=$?
done
echo "SCORE_EXIT $rc $(date -Is)"
kill -TERM -- -$pg 2>/dev/null || true; sleep 20; kill -KILL -- -$pg 2>/dev/null || true
exit $rc
