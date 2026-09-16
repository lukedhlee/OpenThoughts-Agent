#!/bin/bash
#SBATCH --job-name=daytona-command-ramp
#SBATCH --partition=booster
#SBATCH --account=reformo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=/e/fscratch/reformo/lee27/experiments/daytona_command_ramp_20260915/%x_%j.out
set -euo pipefail
umask 077
ulimit -c 0
ulimit -n 65536
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONFAULTHANDLER=1
export CUDA_VISIBLE_DEVICES=''
X=/e/fscratch/reformo/lee27/experiments/daytona_command_ramp_20260915
V=/e/project1/transfernetx/lee27/code/envs/snowball-v2
PC=/e/fscratch/reformo/lee27/tools/proxychains-ng-install/bin/proxychains4
source /e/fscratch/reformo/lee27/keys/socks5_currease.env
CONF=$X/proxychains_${SLURM_JOB_ID}.conf
cat > "$CONF" <<EOF
strict_chain
quiet_mode
tcp_read_time_out 30000
tcp_connect_time_out 15000
localnet 127.0.0.0/255.0.0.0
localnet 10.0.0.0/255.0.0.0
localnet 172.16.0.0/255.240.0.0
localnet 192.168.0.0/255.255.0.0
localnet 169.254.0.0/255.255.0.0
[ProxyList]
socks5 10.128.1.2 7011 $SOCKS_USER $SOCKS_PASS
EOF
trap 'rm -f "$CONF"' EXIT
cd "$X"
child=''
trap 'if [[ -n "$child" ]]; then kill -USR1 "$child"; fi' USR1 TERM INT
"$PC" -q -f "$CONF" "$V/bin/python" -u command_ramp.py \
  --run "command-canary-${SLURM_JOB_ID}" --out "$X/canary_${SLURM_JOB_ID}" \
  --levels 4 --plateau-seconds 8 &
child=$!
wait "$child"
"$PC" -q -f "$CONF" "$V/bin/python" -u command_ramp.py \
  --run "command-main-${SLURM_JOB_ID}" --out "$X/main_${SLURM_JOB_ID}" &
child=$!
wait "$child"
