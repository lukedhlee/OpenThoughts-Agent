#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --output=/e/fscratch/reformo/lee27/experiments/snowball_ttband_ota3d517u_b/logs/%x_%j.out
#SBATCH --job-name=snowball_ttband_ota3d517u_b
#SBATCH --mail-type=END,TIME_LIMIT,FAIL
#SBATCH --mail-user=
#SBATCH --signal=B:USR1@300
#SBATCH -p booster
#SBATCH --account laionize
#SBATCH --gres=gpu:4
#SBATCH --exclude=jpbo-031-[01-48],jpbo-011-[01-48],jpbo-038-38,jpbo-004-46,jpbo-065-17,jpbo-074-22,jpbo-074-40,jpbo-048-41,jpbo-091-05,jpbo-044-0[1-5]

# ==============================================================================
# Universal RL Training SBATCH Template
# ==============================================================================
# This template handles SkyRL-based reinforcement learning training jobs.
# It sets up Ray cluster, environment variables, and runs the SkyRL entrypoint.
#
# Usage: The launcher writes a JSON config file and substitutes /e/fscratch/reformo/lee27/experiments/snowball_ttband_ota3d517u_b/configs/snowball_ttband_ota3d517u_b_rl_config.json
# ==============================================================================

set -eo pipefail
ulimit -c 0  # Disable core dumps to avoid filling disk space
ulimit -n 65536 2>/dev/null || true  # Raise fd limit for high-concurrency Harbor trials

# Handle bash completion scripts that use BASH_COMPLETION_DEBUG
if [ -z "${BASH_COMPLETION_DEBUG+x}" ]; then
  export BASH_COMPLETION_DEBUG=""
fi

# --- Clean up /tmp to prevent state pollution from previous jobs ---
# Some HPC systems retain /tmp contents across job allocations on the same node.
# This can cause issues with tmux sessions, container state, and other temporary files.
rm -rf /tmp/tmux-* 2>/dev/null || true
rm -rf /tmp/ray 2>/dev/null || true
rm -rf /tmp/hf_home 2>/dev/null || true
rm -rf /tmp/containers 2>/dev/null || true
rm -rf /tmp/podman-* 2>/dev/null || true

# --- Module and Conda Setup ---

# --- Module loading (cluster-specific, substituted by launcher) ---
module load GCC/14.3.0
module load nvidia-compilers/25.9-CUDA-13

# --- Environment setup ---
if [ -n "${DCFT_PRIVATE:-}" ]; then
  WORKDIR="$DCFT_PRIVATE"
elif [ -n "${DCFT:-}" ]; then
  WORKDIR="$DCFT"
else
  WORKDIR="$PWD"
fi

# --- WORKDIR sanity guard (HARD FAIL — do not remove) ---
# If WORKDIR does not point at the OpenThoughts-Agent repo root, every downstream
# `source "$WORKDIR/hpc/..."` and `python -m hpc....` fails cryptically and the job
# dies instantly. This happens when a job is submitted from outside the repo with
# DCFT/DCFT_PRIVATE unset, so the fallback resolves WORKDIR=$PWD (the wrong dir).
# Fail loudly and immediately instead of running from the wrong place.
if [ ! -f "$WORKDIR/hpc/shell_utils/triton_cache.sh" ]; then
  echo "FATAL: WORKDIR='$WORKDIR' is not the OpenThoughts-Agent repo root." >&2
  echo "       (marker hpc/shell_utils/triton_cache.sh not found under it)." >&2
  echo "       Submit the job FROM the repo dir with DCFT exported — on Jupiter:" >&2
  echo "         cd /e/scratch/jureap59/feuer1/OpenThoughts-Agent && export DCFT=\$PWD" >&2
  echo "       or pass --chdir <repo> to sbatch. Do NOT submit from \$HOME or a scratch subdir." >&2
  echo "       See .agents/skills/rl-agentic-launch-jupiter for the correct launch procedure." >&2
  exit 1
fi
cd "$WORKDIR"

if [ -z "${DCFT:-}" ]; then
  export DCFT="$WORKDIR"
fi

# --- Conda activation (cluster-specific, substituted by launcher) ---
# No conda activation configured

# --- Source environment files (before RL env activation) ---
# Source dotenv and secrets first, but NOT DCFT_ACTIVATE_ENV which would clobber RL env
# Resolve the dotenv dir from the checkout we were launched from BEFORE sourcing
# <cluster>.env: that file may reset $DCFT/$SCRATCH to a site default (jupiter.env does),
# which would make the overlay lookup below miss the user's checkout and skip silently.
_OT_DOTENV_DIR="${DCFT:-}/hpc/dotenv"
if [ -n "${DCFT:-}" ] && [ -f "$_OT_DOTENV_DIR/jupiter.env" ]; then
  source "$_OT_DOTENV_DIR/jupiter.env"
fi
# Gitignored per-user overlay (<cluster>.local.env); later wins. Mirrors hpc.set_environment.
if [ -f "$_OT_DOTENV_DIR/jupiter.local.env" ]; then
  source "$_OT_DOTENV_DIR/jupiter.local.env"
fi
if [ -n "${DC_AGENT_SECRET_ENV:-}" ] && [ -f "$DC_AGENT_SECRET_ENV" ]; then
  set -a
  source "$DC_AGENT_SECRET_ENV"
  set +a
fi

# --- Daytona API key override (from --daytona_api_key CLI argument) ---
DAYTONA_API_KEY_OVERRIDE="dtn_REDACTED"
if [ -n "$DAYTONA_API_KEY_OVERRIDE" ]; then
  export DAYTONA_API_KEY="$DAYTONA_API_KEY_OVERRIDE"
fi

# --- Daytona target ---
# Intentionally NOT setting DAYTONA_TARGET. Harbor's daytona.py treats unset
# DAYTONA_TARGET as "no region" — snapshot names omit the region segment and
# region_id is left None when creating snapshots. This must stay aligned with
# prebuild_daytona_snapshots() in hpc/rl_launch_utils.py, which is also region-less.

# --- RL Environment Activation ---
# The RL environment is separate from datagen due to dependency conflicts.
# Options:
#   1. venv created by ./hpc/setup_rl_env.sh (default)
#   2. Conda environment specified by --rl_use_conda --rl_conda_env NAME
# NOTE: We intentionally skip DCFT_ACTIVATE_ENV here - it would clobber the RL environment.
# The RL environment is explicitly specified via --rl_use_conda/--rl_conda_env flags.
# Using venv for RL (created by ./hpc/setup_rl_env.sh)
# IMPORTANT: Deactivate conda environment to prevent import conflicts,
# but KEEP conda paths in PATH because the venv's python symlink may point
# to the conda Python that was used when the venv was created.
set +u  # conda deactivate may reference unset variables
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  echo "Deactivating conda environment: $CONDA_PREFIX"
  # Deactivate all stacked conda environments
  while [[ -n "${CONDA_PREFIX:-}" ]]; do
    conda deactivate 2>/dev/null || break
  done
  # Unset the environment name variable so imports don't get confused
  unset CONDA_DEFAULT_ENV
  # NOTE: We keep CONDA_PREFIX and conda paths in PATH because the venv's
  # python binary is often a symlink to the conda Python.
fi
set -u

export DCFT_RL_ENV=/e/project1/transfernetx/lee27/code/envs/snowball-v2
export APPTAINER_BRIDGE_URL=http://10.128.1.2:9930
export RL_PYTHON=/e/project1/transfernetx/lee27/code/envs/snowball-v2/bin/python
export NCCL_PXN_DISABLE=1
export HARBOR_TMUX_BATCH_EXEC_TIMEOUT_MARGIN_SEC=600
export HARBOR_OPENAI_CONNECT_TIMEOUT_SEC=120
export HARBOR_TMUX_CAPTURE_BUDGET_CHARS=400000
export HARBOR_TMUX_CAPTURE_MAX_WINDOW_LINES=2000
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH//envs\/rl-fa/envs\/snowball-v2}"
export VLLM_CACHE_ROOT=/e/fscratch/reformo/lee27/cache/vllm XDG_CACHE_HOME=/e/fscratch/reformo/lee27/cache/xdg TRITON_CACHE_DIR=/e/fscratch/reformo/lee27/cache/triton TORCHINDUCTOR_CACHE_DIR=/e/fscratch/reformo/lee27/cache/inductor
RL_ENV_DIR="${RL_ENV_DIR:-$WORKDIR/envs/rl}"
if [[ -d "$RL_ENV_DIR" ]]; then
  echo "Activating RL environment: $RL_ENV_DIR"
  source "$RL_ENV_DIR/bin/activate"
elif [[ -n "${DCFT_RL_ENV:-}" ]] && [[ -d "$DCFT_RL_ENV" ]]; then
  echo "Activating RL environment from DCFT_RL_ENV: $DCFT_RL_ENV"
  source "$DCFT_RL_ENV/bin/activate"
else
  echo "Warning: RL environment not found at $RL_ENV_DIR"
  echo "Run ./hpc/setup_rl_env.sh to create it, or set DCFT_RL_ENV"
fi

# Verify we're using the correct Python
echo "Python executable: $(which python)"
echo "Python path check: $(python -c 'import sys; print(sys.executable)')"

# Skip DCFT_ACTIVATE_ENV for RL jobs - we use the RL environment activation above instead
# (DCFT_ACTIVATE_ENV typically activates the datagen/SFT environment which conflicts with RL)

# --- CUDA path detection (Perlmutter and similar) ---


# --- Ensure CUDA_HOME/PATH survive conda activation ---
# Module loads set CUDA_HOME but conda activation can reset PATH.
# Re-export to ensure vLLM and DeepSpeed find the right CUDA.
if [ -n "${CUDA_HOME:-}" ]; then
  export PATH="$CUDA_HOME/bin:$PATH"
fi

# --- NCCL/Networking settings (cluster-specific) ---
# Cluster-specific NCCL/networking settings
export NCCL_DEBUG="WARN"
export NCCL_NET_GDR_LEVEL="0"
export NCCL_SOCKET_IFNAME="ib0"
export NCCL_IB_TIMEOUT="23"

# --- RL container env (Apptainer SIF overlays / pydeps / extra exports) ---
# Emitted by the launcher from the RL yaml `container:` section (if present).
# Sets RL_CONTAINER_OVERLAYS / RL_CONTAINER_PYDEPS (read at runtime by
# hpc/rl_launch_utils.py build_apptainer_prefix + _build_container_pythonpath)
# plus any container.extra_env exports (e.g. SKYRL_GDN_MASK_FLA, alloc-conf).
# Empty (comment-only) when the yaml has no container section — byte-identical
# to the prior non-container path. Exported here so every Ray daemon/driver
# (apptainer --env passthrough + srun --export=ALL) inherits them.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG="WARN"
export SKYRL_WORKER_NCCL_TIMEOUT_IN_S="1800"
export VLLM_ATTENTION_BACKEND="FLASHINFER"
export FLASHINFER_WORKSPACE_BASE="/e/fscratch/reformo/lee27/cache/flashinfer"
export VLLM_FORCE_IPV4="1"
export GLOO_USE_IPV6="0"
export NCCL_SOCKET_FAMILY="AF_INET"
export WANDB_PROJECT="jupiter-r2egym-grpo"
export WANDB_MODE="offline"

# --- Standard environment variables ---
# UV_USE_IO_URING=0 forces libuv (bundled in uvloop 0.22.1 as libuv 1.48.0)
# onto the plain, immediate epoll_ctl path instead of the io_uring EPOLL_CTL
# batching path. libuv 1.48.0 defers EPOLL_CTL_DEL through an io_uring SQ; under
# high socket churn (timeout-heavy RL rollouts → constant connect/teardown to
# vLLM + Daytona) a deferred op runs against a closed/recycled fd and libuv
# aborts() in uv__epoll_ctl_flush — surfacing as `uv__epoll_ctl_prep ->
# Fatal Python error: Aborted` in the skyrl_entrypoint Ray driver at ~30-70 min.
# Fixed upstream in libuv 1.49.0 ("don't delay EPOLL_CTL_DEL operations"), which
# no uvloop release ships yet. This var is exported in the top-level batch shell
# and propagates to every Ray head/worker via `srun --export=ALL` (ray_utils.py),
# so the skyrl_entrypoint Ray task inherits it. Harmless on io_uring-disabled
# nodes (they already run the plain-epoll path). See
# notes/ot-agent/agent_logs/2026-05-28_uvloop_libuv_io_uring_epoll_abort_rootcause.md
export UV_USE_IO_URING=0
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_LAUNCH_BLOCKING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
# NCCL_BLOCKING_WAIT=1 makes the NCCL collective itself error out at the
# call boundary rather than spinning while the watchdog races to abort.
# Combined with TORCH_NCCL_ASYNC_ERROR_HANDLING above, this surfaces the
# real failure point (often a peer-node HSN socket close on Perlmutter)
# instead of a downstream SIGABRT in a NCCL destructor.
#
# It is also MEMORY-LOAD-BEARING on the 30B EP/FSDP2 GRPO configs, not just a
# debugging aid. 2026-08-14 A/B on the Qwen3-30B-A3B-Base gsm8k arms (Jupiter,
# 6 nodes): removing it (to let the heartbeat watchdog abort hung collectives,
# per upstream MarinSkyRL 449d1eb) made every LONG-generation arm CUDA-OOM
# within steps 1-13 (594MiB alloc, device-global ~95GB full; five jobs) while
# the exact same yaml ran hours OOM-free with it set — blocking wait
# serializes the FSDP2 allgather pipeline and caps peak memory. The cost: a
# hung EP collective spins at 100% GPU forever with NO watchdog abort (six such
# hangs) — pair any long run with an external log-mtime stall-watcher.
# Do not remove again without first buying back memory headroom.
export NCCL_BLOCKING_WAIT=1
export OMP_NUM_THREADS=1
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"

# --- RL-specific environment variables ---
# RL training environment variables
export TENSOR_PARALLEL_SIZE=1
export NUM_INFERENCE_ENGINES=160
export POLICY_NUM_NODES=40
export SKYRL_EXPORT_PATH=/e/fscratch/reformo/lee27/experiments/snowball_ttband_ota3d517u_b/snowball_ttband_ota3d517u_b/exports
export WANDB_MODE=offline
export GLOO_USE_IPV6=0
export NCCL_SOCKET_FAMILY=AF_INET
export VLLM_FORCE_IPV4=1
export VLLM_SKIP_FLAG_DISCOVERY=1
export SKYRL_ENABLE_NUMA_AFFINITY=1
export DISABLE_AIOHTTP_TRANSPORT=True
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export TORCH_CUDNN_SDPA_ENABLED=0
export PYTHONFAULTHANDLER=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TORCH_NCCL_BLOCKING_WAIT_TIMEOUT_MS=1800000
export VLLM_MQ_MAX_CHUNKS=240
export OT_AGENT_RAY_LOG_DIR=/e/fscratch/reformo/lee27/experiments/_ray_logs
export RAY_object_spilling_config='{"type":"filesystem","params":{"directory_path":"/e/fscratch/reformo/lee27/ray_spill"}}'

# --- RL container env (LATE override — config container.extra_env wins) ---
# Re-emit the yaml container.extra_env exports AFTER both the hardcoded
# export TORCH_NCCL_* lines above and the RL-specific env exports block,
# so a config that intentionally sets a normally-hardcoded var (e.g. #232 lowering
# TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC to 600 so NCCL's flight-recorder fires BEFORE
# vLLM's 900s execute_model watchdog and actually dumps) wins by shell last-write.
# No-op for every config whose extra_env does NOT collide with a hardcoded var
# (the earlier container-env block already exported those, re-export is
# idempotent), so this is byte-identical behavior for all existing configs.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG="WARN"
export SKYRL_WORKER_NCCL_TIMEOUT_IN_S="1800"
export VLLM_ATTENTION_BACKEND="FLASHINFER"
export FLASHINFER_WORKSPACE_BASE="/e/fscratch/reformo/lee27/cache/flashinfer"
export VLLM_FORCE_IPV4="1"
export GLOO_USE_IPV6="0"
export NCCL_SOCKET_FAMILY="AF_INET"
export WANDB_PROJECT="jupiter-r2egym-grpo"
export WANDB_MODE="offline"

# --- Ray defaults ---
# --- Ray defaults ---
export RAY_CGRAPH_get_timeout="${RAY_CGRAPH_get_timeout:-900}"
# Disable Ray OOM monitor: FSDP init transiently spikes CPU RAM
# (e.g., 4 workers × 32B model peaks at ~249GB on 251GB nodes).
# The spike settles after init; Ray's default 0.95 threshold kills
# workers during this transient phase. Already disabled for GH200
# (unified memory), now disabled universally.
export RAY_memory_monitor_refresh_ms=0
# Force Ray (and every worker/actor it spawns) onto CPython stock
# asyncio instead of uvloop. Ray installs uvloop in every worker by
# default (RAY_USE_UVLOOP defaults True -> ray/_private/async_compat.py
# try_install_uvloop in default_worker.py). libuv's io_uring/epoll-ctl
# machinery SIGABRTs and then freezes the harbor agent event loops under
# Daytona sandbox-teardown socket churn (uv__epoll_ctl_prep / uv__io_poll
# asserts; present across libuv 1.45-1.49+). This is the SAME class of
# freeze that the RL path fixed via asyncio.set_event_loop_policy in
# BasePPOExp.run(). Belt: kill it globally for the whole datagen job;
# suspenders: the harbor run_async chokepoint also resets the policy.
# Datagen is network-RTT-bound (vLLM/Daytona) so uvloop's throughput
# edge is moot here.
export RAY_USE_UVLOOP=0
if [ -z "${RAY_TMPDIR:-}" ]; then
  RAY_TMPDIR_BASE="/tmp/ray"
  RAY_TMPDIR="${RAY_TMPDIR_BASE}/ray_${SLURM_JOB_ID:-$$}"
  mkdir -p "$RAY_TMPDIR"
fi
export RAY_TMPDIR="${RAY_TMPDIR}"
echo "[ray] RAY_TMPDIR=$RAY_TMPDIR"

# --- vLLM/Ray settings ---
# RL uses its own conda env pinned to an older vLLM wheel that requires the
# V1 engine. The eval/datagen path uses the newer wheel and runs V2 instead.
export VLLM_USE_V1=1
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook

# --- Triton/TorchInductor cache settings (node-local to avoid shared FS issues) ---
export TRITON_CACHE_VERBOSE=1
source "$WORKDIR/hpc/shell_utils/triton_cache.sh"

# --- Hydra debugging ---
export HYDRA_FULL_ERROR=1

# --- HuggingFace/WandB paths ---
export HF_HOME="${HF_HOME:-${HF_HUB_CACHE:-/tmp/hf_home}}"
export WANDB_DIR="${DCFT_WANDB_DIR:-/e/fscratch/reformo/lee27/experiments/snowball_ttband_ota3d517u_b/wandb}"

# --- Create experiment directories ---
# Note: /e/fscratch/reformo/lee27/experiments/snowball_ttband_ota3d517u_b is an absolute path substituted by the launcher
mkdir -p "/e/fscratch/reformo/lee27/experiments/snowball_ttband_ota3d517u_b"
mkdir -p "/e/fscratch/reformo/lee27/experiments/snowball_ttband_ota3d517u_b/logs"
mkdir -p "/e/fscratch/reformo/lee27/experiments/snowball_ttband_ota3d517u_b/exports"
mkdir -p "/e/fscratch/reformo/lee27/experiments/snowball_ttband_ota3d517u_b/ray_logs"

# Preserve rank-by-rank collective history on the shared experiment filesystem.
# PyTorch appends the global rank to this prefix when its NCCL watchdog fires.
source "$WORKDIR/hpc/shell_utils/nccl_flight_recorder.sh"
setup_nccl_flight_recorder "/e/fscratch/reformo/lee27/experiments/snowball_ttband_ota3d517u_b" || exit $?

# --- Cleanup trap to preserve Ray logs ---
# Ray uses /tmp by default for logs/sockets. This trap copies them to the
# experiments directory for debugging after the job terminates.
cleanup_ray_logs() {
  echo "Preserving Ray logs to /e/fscratch/reformo/lee27/experiments/snowball_ttband_ota3d517u_b/ray_logs/"

  # --- Head node logs ---
  if [[ -d /tmp/ray ]]; then
    rsync -a --ignore-errors /tmp/ray/ "/e/fscratch/reformo/lee27/experiments/snowball_ttband_ota3d517u_b/ray_logs/" 2>/dev/null || true
  fi
  for ray_dir in /tmp/ray_logs /tmp/ray_tmp; do
    if [[ -d "$ray_dir" ]]; then
      rsync -a --ignore-errors "$ray_dir/" "/e/fscratch/reformo/lee27/experiments/snowball_ttband_ota3d517u_b/ray_logs/$(basename $ray_dir)/" 2>/dev/null || true
    fi
  done

  # --- Worker node logs (clusters with intra-node SSH, e.g. Jupiter) ---
  local WORKER_LOG_SCRIPT="$WORKDIR/hpc/ray/collect_worker_ray_logs.sh"
  if [[ -f "$WORKER_LOG_SCRIPT" ]]; then
    source "$WORKER_LOG_SCRIPT"
    collect_worker_ray_logs "/e/fscratch/reformo/lee27/experiments/snowball_ttband_ota3d517u_b/ray_logs"
  fi

  echo "Ray log preservation complete"
}

ARTIFACT_STORE_ENABLED="1"
ARTIFACT_STORE_IMAGE="/e/fscratch/reformo/lee27/experiments/snowball_ttband_ota3d517u_b/snowball_ttband_ota3d517u_b/artifact_store.img"
ARTIFACT_STORE_MOUNT="/tmp/otagent-artifact-stores/snowball_ttband_ota3d517u_b-3bc3d165cf20"
ARTIFACT_STORE_LEASE="${ARTIFACT_STORE_IMAGE}.mount-lease"
ARTIFACT_STORE_MOUNTED=0
ARTIFACT_STORE_LEASE_HELD=0
RL_RUNNER_PID=""

release_artifact_store_lease() {
  if [[ "$ARTIFACT_STORE_LEASE_HELD" != "1" ]]; then
    return
  fi
  rm -f "$ARTIFACT_STORE_LEASE/owner"
  rmdir "$ARTIFACT_STORE_LEASE"
  ARTIFACT_STORE_LEASE_HELD=0
}

acquire_artifact_store_lease() {
  if ! mkdir "$ARTIFACT_STORE_LEASE" 2>/dev/null; then
    local previous_job=""
    local previous_state=""
    if [[ -f "$ARTIFACT_STORE_LEASE/owner" ]]; then
      previous_job=$(sed -n 's/^job_id=//p' "$ARTIFACT_STORE_LEASE/owner" | head -1)
    fi
    if [[ -z "$previous_job" ]]; then
      echo "FATAL: artifact store has an active non-Slurm mount lease: $ARTIFACT_STORE_LEASE" >&2
      return 1
    fi
    if ! previous_state=$(squeue -h -j "$previous_job" -o '%T'); then
      echo "FATAL: could not validate artifact-store lease owner job $previous_job" >&2
      return 1
    fi
    if [[ -n "$previous_state" ]]; then
      echo "FATAL: artifact store is owned by active job $previous_job ($previous_state)" >&2
      return 1
    fi
    echo "Recovering stale artifact-store lease from completed job $previous_job"
    rm -f "$ARTIFACT_STORE_LEASE/owner"
    rmdir "$ARTIFACT_STORE_LEASE"
    mkdir "$ARTIFACT_STORE_LEASE"
  fi
  printf 'kind=writer\njob_id=%s\nnode=%s\n' "$SLURM_JOB_ID" "$(hostname)" > "$ARTIFACT_STORE_LEASE/owner"
  ARTIFACT_STORE_LEASE_HELD=1
}

cleanup_artifact_store() {
  local status=0
  if [[ "$ARTIFACT_STORE_MOUNTED" != "1" ]]; then
    release_artifact_store_lease
    return
  fi
  echo "Flushing artifact store: $ARTIFACT_STORE_IMAGE"
  if ! sync; then
    echo "FATAL: sync failed for artifact store $ARTIFACT_STORE_IMAGE" >&2
    status=1
  fi
  if ! fusermount3 -u "$ARTIFACT_STORE_MOUNT"; then
    echo "FATAL: could not unmount artifact store $ARTIFACT_STORE_MOUNT" >&2
    return 1
  fi
  ARTIFACT_STORE_MOUNTED=0
  flock -u 9 || true
  exec 9>&-
  rmdir "$ARTIFACT_STORE_MOUNT" 2>/dev/null || true
  release_artifact_store_lease
  echo "Artifact store unmounted"
  return "$status"
}

cleanup_job() {
  local status=$?
  local cleanup_status=0
  trap - EXIT TERM USR1
  set +e
  cleanup_artifact_store || cleanup_status=1
  cleanup_ray_logs || cleanup_status=1
  if [[ "$status" == "0" ]] && [[ "$cleanup_status" != "0" ]]; then
    status=$cleanup_status
  fi
  exit "$status"
}

source "$WORKDIR/hpc/shell_utils/rl_shutdown.sh"

forward_termination() {
  echo "Received Slurm shutdown signal; forwarding SIGTERM to the RL runner"
  if ! forward_termination_to_rl_runner "$RL_RUNNER_PID"; then
    exit 143
  fi
}

trap cleanup_job EXIT
trap forward_termination TERM USR1

if [[ "$ARTIFACT_STORE_ENABLED" == "1" ]]; then
  if [[ ! -f "$ARTIFACT_STORE_IMAGE" ]]; then
    echo "FATAL: artifact-store image is missing: $ARTIFACT_STORE_IMAGE" >&2
    exit 1
  fi
  mkdir -p "$ARTIFACT_STORE_MOUNT"
  acquire_artifact_store_lease
  exec 9>"${ARTIFACT_STORE_IMAGE}.lock"
  if ! flock -n 9; then
    echo "FATAL: artifact store already has an active writer: $ARTIFACT_STORE_IMAGE" >&2
    exit 1
  fi
  set +e
  e2fsck -p "$ARTIFACT_STORE_IMAGE"
  E2FSCK_STATUS=$?
  set -e
  if [[ "$E2FSCK_STATUS" -gt 1 ]]; then
    echo "FATAL: e2fsck failed for $ARTIFACT_STORE_IMAGE (status $E2FSCK_STATUS)" >&2
    exit "$E2FSCK_STATUS"
  fi
  fuse2fs -o rw,fakeroot "$ARTIFACT_STORE_IMAGE" "$ARTIFACT_STORE_MOUNT"
  if ! mountpoint -q "$ARTIFACT_STORE_MOUNT"; then
    echo "FATAL: fuse2fs returned without mounting $ARTIFACT_STORE_MOUNT" >&2
    exit 1
  fi
  ARTIFACT_STORE_MOUNTED=1
  mkdir -p "$ARTIFACT_STORE_MOUNT/trace_jobs"
SHM_TRIALS=/dev/shm/otagent_trials/snowball_ttband_ota3d517u_b/trace_jobs; mkdir -p "$SHM_TRIALS"; echo "trials_dir on tmpfs: $SHM_TRIALS ($(df -h /dev/shm | tail -n 1))"
( bash /e/project1/transfernetx/lee27/code/snowball/shm_prune.sh "$SHM_TRIALS" 10 70 60 ) &
  echo "Artifact store mounted at $ARTIFACT_STORE_MOUNT"
fi

# --- SSH Tunneling (JSC clusters only) ---
# ============================================================================
# SSH Tunnel + Proxychains Setup for No-Internet Clusters (JSC)
# Wrapped in a function so `return` works correctly in sbatch scripts.
# ============================================================================
_setup_proxy() {
# ============================================================================
#
# Creates SOCKS5 proxy via SSH tunnel to login node, then uses proxychains
# to route external traffic through the tunnel.
#
# Jupiter (ARM GH200): Uses wrapped binary approach (proxychains4 -f <config> cmd)
# Other JSC clusters: Uses LD_PRELOAD approach for Ray worker inheritance
# ============================================================================

# Determine login node and proxychains paths based on cluster
NODE_HOST=$(hostname -s)
PROXYCHAINS_MODE=""  # "binary" or "ldpreload"

if [[ $NODE_HOST == jrc* ]]; then
    LOGIN_NODE="jrlogin05i"
    PROXYCHAINS_LIB="/p/scratch/synthlaion/dc-agent-shared/tools/proxychains-ng-install/lib/libproxychains4.so"
    PROXYCHAINS_MODE="ldpreload"
elif [[ $NODE_HOST == jwb* ]]; then
    LOGIN_NODE="jwlogin22i"
    PROXYCHAINS_LIB="/p/scratch/synthlaion/dc-agent-shared/tools/proxychains-ng-install/lib/libproxychains4.so"
    PROXYCHAINS_MODE="ldpreload"
elif [[ $NODE_HOST == jpb* ]] || [[ $NODE_HOST == jpc* ]]; then
    LOGIN_NODE="jpbl-s01-01"
    # Jupiter uses aarch64 build - binary wrapper approach (LD_PRELOAD doesn't work reliably)
    # Default is one user's build (others get Permission denied) — override per-user
    # via PROXYCHAINS_BIN_OVERRIDE (dotenv / submit shell).
    PROXYCHAINS_BIN="${PROXYCHAINS_BIN_OVERRIDE:-/e/scratch/jureap59/feuer1/proxychains-ng-aarch64/bin/proxychains4}"
    PROXYCHAINS_MODE="binary"
elif [[ $NODE_HOST == lrdn* ]] || [[ $NODE_HOST == *.leonardo.local ]]; then
    LOGIN_NODE="login05-ext.leonardo.cineca.it"
    # Leonardo uses x86 build - binary wrapper approach
    PROXYCHAINS_BIN="/leonardo/home/userexternal/bfeuer00/proxychains/bin/proxychains4"
    PROXYCHAINS_MODE="binary"
else
    echo "[proxy] Unknown cluster for node $NODE_HOST - skipping proxy setup"
    return 0
fi

TUNNEL_PORT=7003

# Check if proxychains is available
if [[ "$PROXYCHAINS_MODE" == "binary" ]]; then
    if [ ! -x "$PROXYCHAINS_BIN" ]; then
        echo "[proxy] ✗ proxychains binary not found at $PROXYCHAINS_BIN"
        echo "[proxy] Skipping proxy setup - external connectivity will fail"
        return 0
    fi
    echo "[proxy] ✓ Found proxychains binary at $PROXYCHAINS_BIN"
else
    if [ ! -f "$PROXYCHAINS_LIB" ]; then
        echo "[proxy] ✗ proxychains library not found at $PROXYCHAINS_LIB"
        echo "[proxy] Skipping proxy setup - external connectivity will fail"
        return 0
    fi
    echo "[proxy] ✓ Found proxychains library at $PROXYCHAINS_LIB"
fi

if [ -z "${SSH_KEY:-}" ] && [ -z "${PROXYCHAINS_SOCKS5_PRESET_HOST:-}" ]; then
    echo "[proxy] SSH_KEY not set - skipping proxy setup"
    echo "[proxy] Set SSH_KEY in your environment to enable internet access"
else
    if [ -n "${PROXYCHAINS_SOCKS5_PRESET_HOST:-}" ]; then
    # Preset SOCKS5 (e.g. a microsocks on a login node): used when compute→login
    # `ssh -D` is impossible because the account has a second factor enforced
    # after pubkey even for cluster-internal sources. Optional
    # PROXYCHAINS_SOCKS5_PRESET_AUTH="user pass" is appended to the ProxyList line.
    NODE_IP="${PROXYCHAINS_SOCKS5_PRESET_HOST}"
    TUNNEL_PORT="${PROXYCHAINS_SOCKS5_PRESET_PORT:-1080}"
    echo "[proxy] ✓ Using preset SOCKS5 proxy at ${NODE_IP}:${TUNNEL_PORT} (skipping SSH tunnel)"
    else
    # Get this node's IP address for multi-node proxy access
    NODE_IP=$(nslookup $NODE_HOST | grep 'Address' | tail -n1 | awk '{print $2}')
    echo "[proxy] Setting up SSH tunnel to $LOGIN_NODE"
    echo "[proxy] SSH key: $SSH_KEY"
    echo "[proxy] Tunnel port: $TUNNEL_PORT"
    echo "[proxy] Node IP: $NODE_IP (workers will connect here)"

    # Create SSH tunnel with SOCKS5 proxy
    # -g flag allows remote hosts (worker nodes) to connect to the tunnel
    ssh -g -f -N -D ${TUNNEL_PORT} \
        -o StrictHostKeyChecking=no \
        -o ConnectTimeout=1000 \
        -o ServerAliveInterval=10 \
        -o ServerAliveCountMax=30 \
        -o TCPKeepAlive=yes \
        -o ExitOnForwardFailure=yes \
        -o BatchMode=yes \
        -i ${SSH_KEY} \
        ${USER}@${LOGIN_NODE}

    # Give tunnel time to establish
    sleep 5

    # Verify tunnel is running
    if pgrep -f "ssh.*-D.*${TUNNEL_PORT}" > /dev/null; then
        echo "[proxy] ✓ SSH tunnel started successfully"
    else
        echo "[proxy] ✗ SSH tunnel failed to start"
        return 0
    fi
    fi

    # ============================================================================
    # Generate proxychains config
    # Key: Uses NODE_IP (not localhost) so worker nodes can access the tunnel
    # localnet entries ensure internal traffic (Ray, NCCL) bypasses proxy
    # ============================================================================
    SLURM_JOB_ID=${SLURM_JOB_ID:-"local"}
    CFG_PATH=~/.proxychains/proxychains_${SLURM_JOB_ID}.conf
    mkdir -p ~/.proxychains

    cat > "$CFG_PATH" <<PCEOF
strict_chain
quiet_mode
tcp_read_time_out 30000
tcp_connect_time_out 15000
localnet 127.0.0.0/255.0.0.0
localnet 127.0.0.1/255.255.255.255
localnet 10.0.0.0/255.0.0.0
localnet 172.16.0.0/255.240.0.0
localnet 192.168.0.0/255.255.0.0
localnet 169.254.0.0/255.255.0.0
[ProxyList]
socks5 ${NODE_IP} ${TUNNEL_PORT}${PROXYCHAINS_SOCKS5_PRESET_AUTH:+ ${PROXYCHAINS_SOCKS5_PRESET_AUTH}}
PCEOF

    if [ -n "${PROXYCHAINS_SOCKS5_PRESET_AUTH:-}" ]; then
        chmod 600 "$CFG_PATH"   # conf embeds SOCKS credentials
    fi
    echo "[proxy] ✓ Generated proxychains config at $CFG_PATH"
    echo "[proxy]   - Internal traffic (10.x.x.x, 172.x.x.x, 169.254.x.x) → DIRECT"
    echo "[proxy]   - External traffic (internet) → PROXY via tunnel"

    # ============================================================================
    # Export proxychains configuration based on mode
    # ============================================================================
    export PROXYCHAINS_CONF_FILE="$CFG_PATH"
    export PROXYCHAINS_SOCKS5_HOST="${NODE_IP}"
    export PROXYCHAINS_SOCKS5_PORT="${TUNNEL_PORT}"

    # if [[ "$PROXYCHAINS_MODE" == "binary" ]]; then
    #     # Binary wrapper approach (Jupiter ARM GH200)
    #     # Ray workers will use: proxychains4 -f $PROXYCHAINS_CONF_FILE ray start ...
    #     export PROXYCHAINS_BINARY="$PROXYCHAINS_BIN"
    #     echo "[proxy] ✓ PROXYCHAINS_BINARY=$PROXYCHAINS_BIN"
    #     echo "[proxy] ✓ PROXYCHAINS_CONF_FILE=$CFG_PATH"
    #     echo "[proxy] ✓ PROXYCHAINS_SOCKS5_HOST=${NODE_IP} (accessible from worker nodes)"
    #     echo "[proxy] ✓ PROXYCHAINS_SOCKS5_PORT=${TUNNEL_PORT}"
    # else
    #     # LD_PRELOAD approach (Jureca, Juwels)
    #     # Ray workers inherit proxy via LD_PRELOAD environment variable
    #     export LD_PRELOAD="$PROXYCHAINS_LIB"
    #     echo "[proxy] ✓ LD_PRELOAD set to $PROXYCHAINS_LIB"
    #     echo "[proxy] ✓ PROXYCHAINS_CONF_FILE=$CFG_PATH"
    #     echo "[proxy] ✓ PROXYCHAINS_SOCKS5_HOST=${NODE_IP} (accessible from worker nodes)"
    #     echo "[proxy] ✓ PROXYCHAINS_SOCKS5_PORT=${TUNNEL_PORT}"
    # fi

    # ============================================================================
    # Daytona/aiohttp timeout and retry settings
    # ============================================================================
    export DAYTONA_MAX_RETRIES=5
    export DAYTONA_RETRY_DELAY=30
    export DAYTONA_BACKOFF_FACTOR=2
    export DAYTONA_TIMEOUT=1800  # 30 minutes
    export AIOHTTP_CLIENT_TIMEOUT=900  # 15 minutes
    export AIOHTTP_CONNECTOR_TIMEOUT=900
    export AIOHTTP_SOCK_CONNECT_TIMEOUT=300
    export AIOHTTP_TOTAL_TIMEOUT=1800

    # Disable SSL verification (JSC certificate issues)
    export PYTHONHTTPSVERIFY=0
    unset SSL_CERT_FILE
    unset CURL_CA_BUNDLE
    unset REQUESTS_CA_BUNDLE
    unset SSL_CERT_DIR

    echo "[proxy] ✓ Daytona timeout settings configured"

    # Test proxy connectivity
    echo "[proxy] Testing proxy connectivity..."
    if [[ "$PROXYCHAINS_MODE" == "binary" ]]; then
        if "$PROXYCHAINS_BIN" -f "$CFG_PATH" curl -s --connect-timeout 10 https://huggingface.co -o /dev/null; then
            echo "[proxy] ✓ Proxy connectivity test passed (huggingface.co reachable via wrapped binary)"
        else
            echo "[proxy] ⚠ Proxy connectivity test failed (may still work for Daytona)"
        fi
    else
        if curl -s --connect-timeout 10 https://huggingface.co -o /dev/null 2>/dev/null; then
            echo "[proxy] ✓ Proxy connectivity test passed (huggingface.co reachable via LD_PRELOAD)"
        else
            echo "[proxy] ⚠ Proxy connectivity test failed (may still work for Daytona)"
        fi
    fi

    # Test that tunnel is accessible from this node's IP (for worker node access)
    if nc -z ${NODE_IP} ${TUNNEL_PORT} 2>/dev/null; then
        echo "[proxy] ✓ Tunnel accessible at ${NODE_IP}:${TUNNEL_PORT} (workers can connect)"
    else
        echo "[proxy] ⚠ Tunnel not accessible at ${NODE_IP}:${TUNNEL_PORT} (workers may fail)"
    fi

    if [[ "$PROXYCHAINS_MODE" == "binary" ]]; then
        echo "[proxy] ✓ Proxy setup complete (using wrapped binary for Ray workers)"
    else
        echo "[proxy] ✓ Proxy setup complete (using LD_PRELOAD for Ray worker inheritance)"
    fi
fi
}
_setup_proxy


# --- SOCKS5 Proxy Setup (JSC clusters - alternative to SSH tunnel) ---
# No proxy configured for this cluster

# --- Container Runtime Setup ---
source "$WORKDIR/hpc/shell_utils/container_runtime.sh"
setup_container_runtime "apptainer" "$WORKDIR" || exit $?

# --- Run the RL training job via Python runner ---
# Ray cluster management is handled by hpc/ray_utils.py using proper srun calls
# Resolve an absolute interpreter for the runner. On some clusters a bare
# `python` still resolves to the system interpreter (e.g. /bin/python 3.9)
# after a successful venv/conda activation because module/XALT hooks re-order
# PATH. Precedence: explicit RL_PYTHON (dotenv / yaml extra_env) > active venv
# > active conda env > whatever `python` resolves to.
if [[ -n "${RL_PYTHON:-}" && -x "${RL_PYTHON}" ]]; then
  :
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  RL_PYTHON="${VIRTUAL_ENV}/bin/python"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  RL_PYTHON="${CONDA_PREFIX}/bin/python"
else
  RL_PYTHON="$(command -v python || true)"
fi
if [[ -z "${RL_PYTHON}" || ! -x "${RL_PYTHON}" ]]; then
  echo "FATAL: RL_PYTHON='${RL_PYTHON:-}' is not executable" >&2
  exit 1
fi
export RL_PYTHON
# Make bare `python` in child processes agree with the runner's interpreter.
export PATH="$(dirname "${RL_PYTHON}"):${PATH}"
hash -r 2>/dev/null || true
echo "=== Universal RL Training Runner ==="
echo "Config: /e/fscratch/reformo/lee27/experiments/snowball_ttband_ota3d517u_b/configs/snowball_ttband_ota3d517u_b_rl_config.json"
echo "Working directory: $WORKDIR"
echo "Python: $RL_PYTHON (which=$(command -v python || true))"
echo "Python version: $("$RL_PYTHON" --version)"
echo "UV_USE_IO_URING: ${UV_USE_IO_URING:-<unset>}"  # libuv io_uring epoll-abort fix; must be 0

# --- Proxy setup for no-internet clusters ---
# Uses LD_PRELOAD set by ssh_tunnel_setup (proxychains library approach)
# This routes external traffic (Daytona API) through proxy while internal traffic (Ray, NCCL) goes direct
# LD_PRELOAD is inherited by all child processes including Ray workers
if [ -n "${LD_PRELOAD:-}" ]; then
    echo "Proxy: ENABLED via LD_PRELOAD"
    echo "  LD_PRELOAD: $LD_PRELOAD"
    echo "  PROXYCHAINS_CONF_FILE: ${PROXYCHAINS_CONF_FILE:-not set}"
    echo "  PROXYCHAINS_SOCKS5_HOST: ${PROXYCHAINS_SOCKS5_HOST:-not set}"
    echo "  PROXYCHAINS_SOCKS5_PORT: ${PROXYCHAINS_SOCKS5_PORT:-not set}"
else
    echo "Proxy: DISABLED (direct internet or not configured)"
fi
echo "========================================"

# MIGRATION SMOKE: drop the harbor_overlay AND the dotenv PYTHONPATH. hpc/dotenv/jupiter.env and
# jupiter.local.env put $CODE_ROOT/MarinSkyRL/skyrl-{train,gym} on PYTHONPATH, which would shadow the
# venv-editable marinskyrl-marin. Reset to the repo root only so harbor/skyrl_train resolve from
# envs/snowball-v2 (harbor-marin 26c307fc, marinskyrl-marin f5cd0001).
export PYTHONPATH="$WORKDIR"
export PYTHONPATH=/e/project1/transfernetx/lee27/code/src/marin_vllm_eagle3${PYTHONPATH:+:$PYTHONPATH}
echo "vllm overlay: $("$RL_PYTHON" -c 'import vllm; print(vllm.__file__)')"
echo "harbor: $("$RL_PYTHON" -c "import harbor; print(harbor.__file__)")"
echo "skyrl_train: $("$RL_PYTHON" -c "import skyrl_train; print(skyrl_train.__file__)")"

"$RL_PYTHON" -m hpc.rl_launch_utils --config "/e/fscratch/reformo/lee27/experiments/snowball_ttband_ota3d517u_b/configs/snowball_ttband_ota3d517u_b_rl_config.json" &
RL_RUNNER_PID=$!
set +e
while true; do
  wait "$RL_RUNNER_PID"
  RL_STATUS=$?
  if ! kill -0 "$RL_RUNNER_PID" 2>/dev/null; then
    break
  fi
  echo "RL runner is still shutting down; continuing to wait"
done
set -e
RL_RUNNER_PID=""
exit "$RL_STATUS"
