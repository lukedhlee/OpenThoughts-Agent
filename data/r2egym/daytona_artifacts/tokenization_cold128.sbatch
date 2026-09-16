#!/bin/bash
#SBATCH --job-name=tokenization-cold128
#SBATCH --partition=booster
#SBATCH --account=reformo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=00:12:00
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
gateway=''
trap '[[ -z "$server" ]] || kill "$server" 2>/dev/null; [[ -z "$gateway" ]] || kill "$gateway" 2>/dev/null; rm -f "$CONF"' EXIT
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

"$V/bin/python" -u "$X/async_socks_connect_proxy.py" --ready "$RUN/gateway-ready" > "$RUN/gateway.log" 2>&1 &
gateway=$!
for i in $(seq 1 30); do [[ -f "$RUN/gateway-ready" ]] && break; sleep 1; done
[[ -f "$RUN/gateway-ready" ]] || exit 1
export HTTPS_PROXY=http://127.0.0.1:18946 HTTP_PROXY=http://127.0.0.1:18946
export NO_PROXY=127.0.0.1,localhost,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16
"$V/bin/python" -u "$X/tokenization_daytona_probe.py" --run "transport-${SLURM_JOB_ID}" --out "$RUN/mixed" --proxychains "$PC" --proxyconf "$CONF" --seats 128 --rounds 2
