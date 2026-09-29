# serve_env.sh — Horizon (TACC, 4x GB200 per node) values for the relay serve knobs of data/relay/pilot/serve_relay.sbatch
# and serve_node.sh, whose defaults are Jupiter's. Source it on the Horizon login node before sbatch (sbatch exports the
# environment to the job; srun passes it on to serve_node.sh). Anything already set in the caller's env wins.
R=${SNOWBALL_ROOT:-$HOME/snowball}; S=${SCRATCH_DIR:-/scratch/11584/$USER}
_snap() { ls -d $S/hf_hub/models--$1/snapshots/${2:-}* 2>/dev/null | head -1; }
export SERVE_CODE=$R SERVE_VENV=$R/envs/snowball SERVE_OVERLAY=$R/src/marin_vllm_eagle3 RELAY_EXTRA=$R/envs/relay-extra
export SERVE_MODULES=""                 # the login env (nvidia/26.9, cuda/13.x) rides along; nothing to load
export CC=gcc CXX=g++                   # nvidia/26.9 sets CC=nvc, which rejects -Wno-psabi when Triton JIT-builds cuda_utils.c
export SERVE_CACHE=$S/cache HF_HOME=$S/cache/hf
export RELAY_PILOT_DIR=${RELAY_PILOT_DIR:-$R/ota/data/relay/pilot} RELAY_EXP_DIR=${RELAY_EXP_DIR:-$S/experiments/relay/pilot}
export NODE_SUFFIX=${NODE_SUFFIX-}      # short names (c101-003) resolve on login and compute nodes
export TEACHER_MODEL=${TEACHER_MODEL:-$(_snap Qwen--Qwen3.8-27B 1d4bf0f2)}
export STUDENT_MODEL=${STUDENT_MODEL:-$(_snap laion--snowball-67b-a2b-sft-s3-nemotron-terminal-step1888)}   # Stage-3 base until the relay-SFT arm-A export lands
export STUDENT_DRAFT=${STUDENT_DRAFT:-$(_snap laion--snowball-64k-eagle3-draft-r2egym)}
export PER_GPU=${PER_GPU:-1} TEACHER_MAXLEN=${TEACHER_MAXLEN:-131072}   # the full runs' teacher: 4 x TP1 servers at 128k
