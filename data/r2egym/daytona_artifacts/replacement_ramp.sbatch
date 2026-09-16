#!/bin/bash
#SBATCH --job-name=daytona-replacement-ramp
#SBATCH --partition=booster
#SBATCH --account=reformo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=00:25:00
#SBATCH --signal=B:USR1@240
#SBATCH --output=/e/fscratch/reformo/lee27/experiments/daytona_fixed_ramp_20260915/%x_%j.out
# Replacement-under-load stage the 35-minute fixed ramp (1821426) had to skip:
# 1,056 seats on the async gateway, 60 s cold-heavy plateau, 120 s warm plateau, delete 64 seats and
# recreate them under load, 60 s plateau, 128-seat recovery, exact-label cleanup.
# No `set -e`: bash's wait returns on the trapped USR1 and set -e would exit the script before the
# ramp's own cleanup (gotchas 2026-09-15). The trap forwards USR1 so the driver stops after the
# current stage and cleans up; the loop keeps waiting for it.
set -uo pipefail
umask 077
ulimit -c 0
ulimit -n 65536
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
X=/e/fscratch/reformo/lee27/experiments/daytona_fixed_ramp_20260915
V=/e/project1/transfernetx/lee27/code/envs/snowball-v2
export PYTHONPATH=/e/fscratch/reformo/lee27/code/harbor-pacing/src:/e/fscratch/reformo/lee27/experiments/tokenization_transport_20260915/deps
set -a
source /e/fscratch/reformo/lee27/keys/socks5_currease.env
source /e/fscratch/reformo/lee27/keys/daytona_eval.env
set +a
export HTTPS_PROXY=http://127.0.0.1:18946 HTTP_PROXY=http://127.0.0.1:18946 NO_PROXY=127.0.0.1,localhost,10.128.1.2
export HARBOR_DAYTONA_CREATE_RATE=5 HARBOR_DAYTONA_CREATE_SHARES=1
RUN="$X/run_${SLURM_JOB_ID}"
mkdir -p "$RUN"
child=''
gateway=''
trap '[[ -z "$gateway" ]] || kill "$gateway" 2>/dev/null' EXIT
trap '[[ -z "$child" ]] || kill -USR1 "$child" 2>/dev/null' USR1 TERM INT
"$V/bin/python" -u "$X/async_socks_connect_proxy.py" --ready "$RUN/gateway-ready" > "$RUN/gateway.log" 2>&1 &
gateway=$!
for i in $(seq 1 30); do [[ -f "$RUN/gateway-ready" ]] && break; sleep 1; done
[[ -f "$RUN/gateway-ready" ]] || exit 1
"$V/bin/python" -u "$X/command_ramp.py" --run "replacement-${SLURM_JOB_ID}" --out "$RUN/main" --levels 1056 \
  --plateau-seconds 60 --cold-every 4 --cold-gap 35 --replace-seats 64 &
child=$!
rc=0
while kill -0 "$child" 2>/dev/null; do wait "$child"; rc=$?; done
echo "ramp exit $rc" >> "$RUN/main/exit"
exit "$rc"
