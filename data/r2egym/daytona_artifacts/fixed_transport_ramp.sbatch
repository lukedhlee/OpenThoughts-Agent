#!/bin/bash
#SBATCH --job-name=daytona-fixed-ramp
#SBATCH --partition=booster
#SBATCH --account=reformo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=00:35:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=/e/fscratch/reformo/lee27/experiments/daytona_fixed_ramp_20260915/%x_%j.out
set -euo pipefail
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
[[ -f "$RUN/gateway-ready" ]]
"$V/bin/python" -u "$X/preflight_daytona_pair.py" --config /e/fscratch/reformo/lee27/experiments/rollout_transport_pair_20260915/run_1821359/daytona_16_config.json --out "$RUN/preflight" > "$RUN/preflight.log" 2>&1
"$V/bin/python" -u "$X/command_ramp.py" --run "fixed-canary-${SLURM_JOB_ID}" --out "$RUN/canary" --levels 4 --plateau-seconds 45 --cold-every 4 &
child=$!
wait "$child"
"$V/bin/python" -u "$X/command_ramp.py" --run "fixed-main-${SLURM_JOB_ID}" --out "$RUN/main" --plateau-seconds 180 --final-seconds 600 --cold-every 4 --cold-gap 35 --replace-seats 64 &
child=$!
wait "$child"
