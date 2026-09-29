#!/bin/bash
# build_env.sbatch — compile the Snowball runtime for Horizon (GB200 = sm_100) on one compute node.
# The login node is shared (TACC policy: no heavy builds there), so only the download steps (tools python clones torch) run there; this job runs the rest
# (vllm trainer overlay smoke) with 64 build jobs, reaching the internet through tunnel.sh on the login node.
#   sbatch build_env.sbatch  ->  setsid nohup bash tunnel.sh <jobid> > ~/tunnel.<jobid>.log 2>&1 &
#SBATCH -J sb_build
#SBATCH -A CCR24067
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -o %x.%j.out
set -uo pipefail
PORT=${PORT:-18080}
echo "node $(hostname) start $(date -Is); waiting for the reverse tunnel on :$PORT"
for i in $(seq 1 120); do (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break; sleep 5; done
(echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null || { echo "FATAL: no tunnel on :$PORT after 10 min (run tunnel.sh on login1)"; exit 1; }
P=socks5h://127.0.0.1:$PORT
export ALL_PROXY=$P all_proxy=$P HTTPS_PROXY=$P https_proxy=$P HTTP_PROXY=$P http_proxy=$P CARGO_HTTP_PROXY=$P NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
curl -sS -m 20 -o /dev/null -w "preflight pypi %{http_code}\n" https://pypi.org/simple/ || { echo "FATAL: proxy does not reach pypi"; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1
export ROOT=${ROOT:-$HOME/snowball} SCRATCH=${SCRATCH:-/scratch/11584/$USER}
export SNOWBALL_MODULES="nvidia/26.9 cuda/13.1 cmake/4.4.0" STAGE_MODULE=TACC TORCH_CUDA_ARCH_LIST=10.0 SNOWBALL_RUSTUP=1 MAX_JOBS=64
export SNOWBALL_CC=gcc SNOWBALL_CXX=g++   # nvidia/26.9 sets CXX=nvc++, which fails on torch Half.h (float16_t); system GCC 14 = Jupiter's
export MSRL_SHA=${MSRL_SHA:-a03b2773} HARBOR_SHA=${HARBOR_SHA:-dcf609bc}   # what Jupiter runs (2026-09-15 checkouts)
export CMAKE_POLICY_VERSION_MINIMUM=3.5   # cmake 4 refuses FetchContent deps that declare cmake_minimum_required < 3.5
# Run the script from the pulled checkout ($ROOT/ota), not the build-time archive in $ROOT/snowball, so fixes arrive by git pull.
bash "$ROOT/ota/data/r2egym/jsc/build_snowball_env.sh" ${STEPS:-vllm trainer overlay eagle3 smoke}
