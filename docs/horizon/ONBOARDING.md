# Getting started on TACC Horizon, with Claude Code

**What this is.** Horizon is TACC's GB200 cluster. Each node has 4 GB200 GPUs (~185 GiB each), 2 Grace CPUs (144
aarch64 cores) and ~960 GB of RAM, so one node has about twice the GPU memory of a Jupiter node. During early operations
everything runs in one `debug` partition on project **`CCR24067`**, and it is free to use (TACC said so 2026-09-29;
still size jobs to need). The three things that trip everyone up are: **compute nodes have no internet**, **`/scratch`
deletes files untouched for 10 days**, and **TACC's default compiler (`nvc`) breaks Triton, vLLM and torch builds**.
This file covers how to handle each.

**How to use this file.** Give it to Claude Code on your laptop and say "set me up for Horizon". Steps marked **(you)**
need your password and TACC token code, so Claude must stop and ask you to do them. Claude can run steps marked
**(Claude)** itself once you are logged in. Replace `<you>` with your TACC username everywhere.

---

## Rules for Claude (read before doing anything)

- **Never try to log in.** Login needs a password and a TOTP code. Use only the `horizon` alias below, which reuses
  a connection the person opened. If `ssh horizon true` fails, ask them to run `ssh horizon-login` in their own terminal.
  Never edit the alias's `ProxyCommand` or `BatchMode` lines to get around this.
- Submit every job with **`-A CCR24067 -p debug`** and a time limit `-t`. The account must be uppercase; TACC's filter
  rejects `ccr24067`. `srun` needs `-p`, `-A` and `-t` too, even for a step inside a running job.
- **Keep heavy work off the login node.** Builds, torch, vLLM and anything that uses a GPU go to compute nodes. The login
  node is shared with every user. Downloads are fine there.
- Never cancel a job or kill a process you did not start. Kill by exact pid, never with a broad `pkill -f`.
- Never print tokens or keys (HF, GitHub, Daytona, W&B).
- Keep code and environments in `$HOME` (`/home1/...`), and only data you can regenerate in `$SCRATCH`.
- Always use `$SCRATCH` and `$HOME`, never a literal `/scratch/<number>/...` path. The number differs per user.

---

## 1. Log in from your laptop

**(Claude)** Add this to `~/.ssh/config` and run `mkdir -p ~/.ssh/cm && chmod 700 ~/.ssh/cm`.

```sshconfig
# Fresh login, done by the human in a terminal: password + TACC token code.
# It leaves a shared connection open for 4 hours after the last session closes.
Host horizon-login
    HostName horizon.tacc.utexas.edu
    User <you>
    PreferredAuthentications keyboard-interactive,password
    PubkeyAuthentication no
    NumberOfPasswordPrompts 1
    ConnectTimeout 15
    ServerAliveInterval 60
    ServerAliveCountMax 3
    ControlMaster auto
    ControlPath ~/.ssh/cm/%r@%h:%p
    ControlPersist 4h

# For Claude and scripts: rides the open connection, never prompts, and fails at once if no connection is open.
Host horizon
    HostName horizon.tacc.utexas.edu
    User <you>
    BatchMode yes
    ProxyCommand /usr/bin/false
    ControlMaster no
    ControlPath ~/.ssh/cm/%r@%h:%p
```

**(you)** Run `ssh horizon-login`, then enter your TACC password and the code from the TACC token app. Leave that
terminal open or close it; the connection lasts 4 hours either way.

**(Claude)** Check it works: `ssh horizon 'hostname; echo $HOME $SCRATCH'`. If it fails, ask for a fresh login.
TACC asks for a token code on every fresh login, so after 4 idle hours the person has to log in again.

## 2. One-time setup on the Horizon login node

**(Claude)** Add these lines to the end of `~/.bashrc` on Horizon:

```bash
export PATH="$HOME/.local/bin:$PATH"
# Horizon's first DNS server never answers outside names, so each lookup waits 5 s. This cuts it to 1 s.
export RES_OPTIONS="timeout:1 attempts:2 rotate"
```

Then install the tools that are missing (the login node has internet; Python 3.12, git and `screen` are there already):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh                       # uv; use it instead of pip (see §4)
uv tool install 'huggingface_hub[cli]'                                 # the `hf` command
cd ~/.local && curl -Ls https://micro.mamba.pm/api/micromamba/linux-aarch64/latest | tar -xvj bin/micromamba
~/.local/bin/micromamba create -y -r ~/.local/mamba -n tools -c conda-forge tmux
ln -sf ~/.local/mamba/envs/tools/bin/tmux ~/.local/bin/tmux             # there is no system tmux
```

**(you)** Run `hf auth login` on Horizon and paste your HF token. Do the same for anything else that needs a key
(`gh auth login`, W&B, Daytona). Keep keys in a file only you can read (`chmod 600`).

**Optional: Claude Code on Horizon.** A session on the login node keeps working while your laptop sleeps.
`curl -fsSL https://claude.ai/install.sh | bash`, then **(you)** run `claude` once there and log in. Start long work
in tmux so it survives disconnects: `tmux new -d -s work "claude --remote-control 'Horizon work'"`. With Remote
Control on, you can steer the session from claude.ai/code or your phone. Never copy Claude credentials between machines.

## 3. Submitting jobs

```bash
#!/bin/bash
#SBATCH -J myjob
#SBATCH -A CCR24067
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -o %x.%j.out
export CC=gcc CXX=g++          # see §5; put this in every job that builds code or runs Triton / vLLM / torch.compile
```

- **Nodes are exclusive.** You get all 4 GPUs and 144 cores. Add `--gres=gpu:4` when you use `srun` to bind one GPU
  per rank.
- **Limits per user:** 20 running jobs, 40 submitted, 48 h per job, no node cap. The 20-job cap counts jobs, not
  nodes, so pack work into fewer, bigger jobs. For example, run a client as a step inside its server's job:
  `srun -p debug -A CCR24067 -t <time left> --jobid=<serve job> --overlap -w <node> <cmd>`.
- `sbatch` prints a TACC banner first, so with `--parsable` take the last line: `J=$(sbatch --parsable x.sbatch | tail -1)`.
- Finished jobs stay in `COMPLETING` for 5–10 minutes while TACC's cleanup runs. Don't treat that as a hang.
- `squeue` shows everyone's jobs, but `sacct -a` shows only yours. `sinfo -p debug` shows how many nodes are idle.

## 4. Internet for compute nodes (the tunnel)

Compute nodes can't reach the internet. A compute node also can't ssh back to the login node, because that would
ask for a token code. So the tunnel starts on the login side. `ssh -R PORT <node>` opens a SOCKS5 proxy on the node's
`localhost:PORT`, and its traffic leaves through the login node. TACC puts an ssh key for this in `~/.ssh` when you first
log in, so it works with no setup.

**(Claude)** Save this as `~/bin/tunnel.sh` on Horizon. Once you've cloned Luke's code (§8), the same script is also
at `data/r2egym/horizon/tunnel.sh`.

```bash
#!/usr/bin/env bash
# tunnel.sh <jobid> [PORT]: run on the login node, detached:
#   setsid nohup bash ~/bin/tunnel.sh <jobid> > ~/tunnel.<jobid>.log 2>&1 < /dev/null &
# Waits for the job to start, keeps one reverse SOCKS tunnel per node (reopens dropped ones), exits when the job ends.
set -uo pipefail
JOB=${1:?jobid}; PORT=${2:-18080}
SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=no -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3)
declare -A PID
while :; do
  state=$(squeue -h -j "$JOB" -o %T 2>/dev/null)
  case "$state" in
    RUNNING) ;;
    PENDING|CONFIGURING|"") [ -z "$state" ] && { echo "$(date -Is) job $JOB gone, exiting"; break; }; sleep 15; continue;;
    *) echo "$(date -Is) job $JOB state $state"; sleep 15; continue;;
  esac
  for n in $(scontrol show hostnames "$(squeue -h -j "$JOB" -o %N)"); do
    if [ -z "${PID[$n]:-}" ] || ! kill -0 "${PID[$n]}" 2>/dev/null; then
      ssh "${SSH_OPTS[@]}" -N -R "$PORT" "$n" & PID[$n]=$!
      echo "$(date -Is) tunnel -> $n:$PORT pid ${PID[$n]}"
    fi
  done
  sleep 20
done
for p in "${PID[@]}"; do kill "$p" 2>/dev/null; done
```

In the job, wait for the tunnel to come up, then point everything at it:

```bash
PORT=18080
for i in $(seq 1 120); do (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break; sleep 5; done
P=socks5h://127.0.0.1:$PORT
export ALL_PROXY=$P all_proxy=$P HTTPS_PROXY=$P https_proxy=$P HTTP_PROXY=$P http_proxy=$P \
       NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
```

- curl, git, uv and cargo work through the tunnel. **pip does not** unless PySocks is installed, so use uv.
- The first ssh to a new job can be refused for a few seconds (`pam_slurm_adopt`). The script retries, so that's fine.
- Download models and datasets **on the login node** (HF ran at ~600 MB/s there) into `$SCRATCH`. Then run jobs with
  `HF_HUB_OFFLINE=1`. That's faster and doesn't depend on the tunnel.
- Some Python clients ignore proxy settings. The Daytona sync SDK, for example, hangs on a compute node; use its
  async client there.

## 5. Things that differ from other clusters

- **aarch64 + sm_100.** Use aarch64 wheels, and set `TORCH_CUDA_ARCH_LIST=10.0` when you compile CUDA code. x86
  wheels and Jupiter's Hopper builds (sm_90) won't run here.
- **Compilers.** TACC's login environment loads `nvidia/26.9`, which sets `CC=nvc` and `CXX=nvc++`, and jobs
  inherit it. Triton's JIT rejects `nvc`, so every vLLM engine dies at startup, and torch extension builds fail on
  `float16_t`. Put `export CC=gcc CXX=g++` (system GCC 14) in every job. If something needs `CUDA_HOME`, use
  `/home1/apps/nvidia/Linux_aarch64/26.9/cuda/13.3`.
- **Multi-node NCCL.** Set `NCCL_SOCKET_IFNAME=ib`. Compute nodes name their InfiniBand ports `ibs2`, `ibP2p1s0`, …,
  which differ from the login node's.
- **Compile caches are per node.** `/scratch` is NFS. When several nodes compile into one Triton, Inductor or vLLM
  cache, a node can read another's half-written kernel and crash at its first batch. For multi-node jobs, put caches
  in node-local `/tmp` (`TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `VLLM_CACHE_ROOT`). A single-node job can use
  a shared cache safely.
- **`nproc` says 1 on the login node.** That's only because TACC sets `OMP_NUM_THREADS=1`. The node really has 144
  cores, but other users share them.

## 6. Where to put files

| | path | size | backed up | use for |
|---|---|---|---|---|
| home | `$HOME` = `/home1/<n>/<you>` | ~50 GB, ~500k files | daily | code, venvs, tools |
| scratch | `$SCRATCH` = `/scratch/<n>/<you>` | 1.4 PB shared | **no** | models, data, checkpoints, logs |

- **Scratch deletes any file not read for 10 days**, and TACC forbids touching files to keep them alive. Push results
  (checkpoints, rollouts) to HF or copy them somewhere else soon after they're made.
- A venv is a few GB and tens of thousands of files, so watch the home file count.
- `/work` is not mounted on Horizon, even though `$WORK` is set.

## 7. Check that it all works (one node, ~15 min)

**(Claude)** Save this as `$SCRATCH/smoke/smoke.sbatch`, submit it from that directory, and start the tunnel:

```bash
#!/bin/bash
#SBATCH -J hz_smoke
#SBATCH -A CCR24067
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 00:15:00
#SBATCH -o %x.%j.out
set -uo pipefail
echo "node $(hostname)"; nvidia-smi -L
PORT=18080
for i in $(seq 1 60); do (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break; sleep 5; done
export ALL_PROXY=socks5h://127.0.0.1:$PORT HTTPS_PROXY=socks5h://127.0.0.1:$PORT
curl -sS -m 20 -o /dev/null -w "huggingface.co -> %{http_code}\n" https://huggingface.co
curl -sS -m 20 -o /dev/null -w "pypi.org       -> %{http_code}\n" https://pypi.org/simple/
echo SMOKE_DONE
```

```bash
cd $SCRATCH/smoke && J=$(sbatch --parsable smoke.sbatch | tail -1)
setsid nohup bash ~/bin/tunnel.sh $J > ~/tunnel.$J.log 2>&1 < /dev/null &
```

It passes when `hz_smoke.<J>.out` lists 4 GB200s, both URLs return `200`, and it ends with `SMOKE_DONE`. If the job
stays `PENDING`, run `squeue -j $J -o %R` to see why. The cluster is often busy.

---

## 8. Get Luke's code (for the relay recipes)

**(Claude)** Clone Luke's public fork at the tag `relay-v1.1` (or the newest `relay-v*` tag Luke announces). The working branch moves fast, so always use a tag.
The scripts expect the checkout at `~/snowball/ota` and build their environments under `~/snowball/envs`:

```bash
git clone --filter=blob:none -b relay-v1.1 https://github.com/lukedhlee/OpenThoughts-Agent.git ~/snowball/ota
```

The scripts find your directories through `$SCRATCH` and `$HOME`, so nothing in them needs editing. Then follow
`data/relay/README.md`. It lists what to build and download, starts with a cheap 4-task smoke run, and then covers
the full loop and its variants. You need one thing from Luke first: a key for the Daytona org that holds the task
snapshots.

Other workloads that run on Horizon, each with a doc that starts with its launch commands:

| workload | doc |
|---|---|
| Relay rollouts, relay SFT, the variants and the Qwen-alone comparison | `data/relay/README.md` |
| Serve Snowball (Grug 67B-A2B) with vLLM + EAGLE-3, build the env | `data/r2egym/horizon/serve_smoke.sbatch`, `build_env.sbatch` |
| Grug SFT chain (Levanter / marin, 16 nodes) | `data/r2egym/horizon/SFT.md` |
| Evals: held-out R2E-Gym, TB2.1, SWE-bench, TB-lite | `data/tb2/horizon/EVAL.md` |
| GRPO on R2E-Gym (MarinSkyRL) | `data/r2egym/horizon/RL.md` |
