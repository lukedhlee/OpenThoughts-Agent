#!/bin/bash
#SBATCH --job-name=tokenization-transport
#SBATCH --partition=booster
#SBATCH --account=reformo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=/e/fscratch/reformo/lee27/experiments/tokenization_transport_20260915/%x_%j.out
set -euo pipefail
umask 077
ulimit -c 0
ulimit -n 65536
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONFAULTHANDLER=1
export CUDA_VISIBLE_DEVICES=''
X=/e/fscratch/reformo/lee27/experiments/tokenization_transport_20260915
V=/e/project1/transfernetx/lee27/code/envs/snowball-v2
PC=/e/fscratch/reformo/lee27/tools/proxychains-ng-install/bin/proxychains4
set -a
source /e/fscratch/reformo/lee27/keys/socks5_currease.env
set +a
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

server=''
trap '[[ -z "$server" ]] || kill "$server" 2>/dev/null; rm -f "$CONF"' EXIT
export PYTHONPATH="$X/deps"
RUN="$X/run_${SLURM_JOB_ID}"
mkdir -p "$RUN"
"$V/bin/python" -u "$X/tokenization_transport_probe.py" --serve --ready "$RUN/ready" > "$RUN/server.log" 2>&1 &
server=$!
for i in $(seq 1 300); do
  [[ -f "$RUN/ready" ]] && break
  kill -0 "$server" || { cat "$RUN/server.log"; exit 1; }
  sleep 1
done
[[ -f "$RUN/ready" ]] || exit 1
for seats in 64 128; do
  "$V/bin/python" -u "$X/tokenization_transport_probe.py" --mode baseline --seats "$seats" --out "$RUN/baseline_${seats}.json"
  "$PC" -q -f "$CONF" "$V/bin/python" -u "$X/tokenization_transport_probe.py" --mode preload --cold --seats "$seats" --out "$RUN/preload_cold_${seats}.json"
  "$V/bin/python" -u "$X/tokenization_transport_probe.py" --mode native --cold --seats "$seats" --out "$RUN/native_cold_${seats}.json"
  "$PC" -q -f "$CONF" "$V/bin/python" -u "$X/tokenization_transport_probe.py" --mode preload --seats "$seats" --out "$RUN/preload_warm_${seats}.json"
done
