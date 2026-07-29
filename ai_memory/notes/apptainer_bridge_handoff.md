# Handoff — enable Apptainer on JSC + port the apptainer bridge

**Owner:** lukedhlee (JSC user `lee27`) · **Opened 2026-07-28** · **Status: JURECA CPU bridge smoke passed; next is small-Qwen OpenCode→vLLM smoke**

Parent: [[handoff]] (the r2egym GRPO task this serves).
Full background analysis: [[apptainer_bridge_swebench]] — **read that second**, it has the measurements
and the architecture. This file is the operational state: what's done, what's next, what to watch out for.
Cluster particulars: [[jureca_ssh]], [[jureca_what_goes_where]], [[jsc_paths_hazards]],
[[jureca_marianna_setup_plan]] (her fork refs + the JURECA split-mode design).

Visual explainer (may be stale): https://claude.ai/code/artifact/6447de1d-c124-4830-b2fc-165173afd06b

---

## Why this task exists (one paragraph)

We want **SWE-bench as a 200-task validation set**. We can't: harbor's Daytona backend needs one snapshot
per SWE-bench task (**measured: 500 tasks → 500 unique env hashes**), harbor **never reaps** snapshots, and
our org cap leaves ~40 headroom ⇒ a 200-task eval dies around task 40. Jupiter can't run the images at all
(**Jupiter is aarch64/GH200; every `swebench/sweb.eval.x86_64.*` image is x86_64**; ARM variants cover only
~53% and are repo-biased ⇒ non-comparable). The RL co-lead (Marianna) already solved this with an
**apptainer bridge**: a self-hosted Daytona replacement running task containers on **x86_64 CPU nodes** at
JSC. Porting it removes the snapshot cap entirely. **JURECA is x86_64**, so a single-cluster JURECA
deployment needs no cross-cluster tunnel and has no ARM problem.

**This is NOT on the RL launch's critical path.** In-flight validation = held-out r2egym slice (same 7
Daytona snapshots as training, zero new infra). The bridge buys SWE-bench headline numbers on the best 2–3
checkpoints, needed only once checkpoints exist. Build it in parallel; do not block the launch on it.

---

## DONE ✅

### The `container` group gate — RESOLVED
`apptainer` was installed on every JSC system but refused to run:
```
User lee27 is not in group container and hence not authorized to use apptainer
Please visit JUDOOR to sign the Container Service Level Description, ...
```
No bypass existed: `/usr/bin/apptainer` is a stripped ELF (check compiled in, not a wrapper script), and
`podman` / `docker` / `enroot` / `udocker` are **all absent** on JURECA.

**Exact JuDoor path** (from [JSC docs](https://apps.fz-juelich.de/jsc/hps/jureca/container-runtime.html)):
`Software` → *Request access to restricted software* → **Access to other restricted software** →
**Container Runtime Engine** → **`Get Access`** → accept the SLD.
Gotchas that cost us time: the card looks inert until clicked (`Get Access` is a *second* reveal), and JSC's
error text says "Container Service Level Description" while the UI labels it "Container Runtime Engine" —
so searching JuDoor for the document by name finds nothing. The "Usage Agreement for Access to JUWELS" is
a different, irrelevant document. **Accepted 2026-07-28.** One signature covers ALL JSC systems.

⚠ **`id -Gn` still does NOT list `container`, even though apptainer works.** The authorization check hits
LDAP directly while `id` is served from a stale local cache. **Never gate a check on `id -Gn`** — it gave a
false negative for us. Test with `apptainer --version` instead.

### Verified working on JURECA login node (`jrlogin11`), 2026-07-28
```
apptainer --version                       → apptainer version 1.4.5-1.el9
apptainer pull docker://alpine:latest     → real 0m3.3s   ✅
apptainer exec alpine.sif uname -m        → x86_64        ✅
```

**BIG WIN: the JURECA login node has direct internet.** Marianna's `prebuild_sifs.sh` builds a SOCKS
proxy (`start_proxy_tunnel.sh` + proxychains-ng) *only* when it detects a compute node (`jrc*`/`jwb*`/`jpb*`)
— JSC compute nodes are air-gapped. **Pulling from the login node skips the proxy entirely**, which removed
the single largest unknown in the estimate. Pull SIFs on the login node; run them on compute nodes.

⚠ **Always set the cache to scratch before any pull** — the default `~/.apptainer/cache` will blow the
home quota on 1–3 GB images:
```bash
export APPTAINER_CACHEDIR=/p/scratch/synthlaion/lee27/apptainer_cache
export APPTAINER_TMPDIR=/p/scratch/synthlaion/lee27/apptainer_tmp
# SIFs live in /p/scratch/synthlaion/lee27/sif
```
(Benign warning, ignore: `'nodev' mount option set on /p/scratch`.)

---

## RESOLVED ✅ — no-build runtime injection of tmux + asciinema

```
$ apptainer exec --fakeroot alpine.sif id -u
INFO:    User not listed in /etc/subuid, trying root-mapped namespace
INFO:    No user namespaces available
FATAL:   --fakeroot used without sandbox image or user namespaces
```
Consistent with the docs: *"Building images on JURECA is not possible, because root privileges are
required."* `apptainer pull` (converting a **published** image) works fine; `apptainer build` from a def
file does not.

**Why this matters:** Marianna injects **tmux + asciinema** into every task SIF via a
`Bootstrap: localimage` overlay (`rebuild_r2egym_sifs.sh`). Harbor's agents drive the session *through
tmux inside the container*, so this is not cosmetic. Her build step is the one piece that does not
transfer as-is.

Resolution ladder:
1. ~~Does the SWE-bench image already ship tmux?~~ **ANSWERED: NO** — both tmux and asciinema are
   MISSING (measured below). The injection step is required; this option is dead.
2. **Bind-mount static binaries at runtime** — **VERIFIED WORKING 2026-07-28.** No build, namespaces,
   root, overlay, or network at runtime is required.
3. **`--overlay` ext3 image or `--writable-tmpfs`** — writable layer at runtime instead of build time.
4. **JSC Container Build System** — SLD item 2 promises an external Dockerfile→image webservice with a
   CLI on the login nodes. Sanctioned path; find its CLI name in the system docs.
5. **Are user namespaces available on COMPUTE nodes** even though absent on login? Untested. Marianna
   evidently builds overlays *somewhere* on JSC, so ask her — this may be a one-line answer.

### Option 2 verification details

Staged on our scratch:
```
/p/scratch/synthlaion/lee27/agent_tools/bin/tmux       # tmux 3.7b, static
/p/scratch/synthlaion/lee27/agent_tools/bin/asciinema  # asciinema 3.2.1, static PIE
```
Sources are the official GitHub release artifacts:
- `tmux/tmux-builds` `tmux-3.7b-linux-x86_64.tar.gz`
- `asciinema/asciinema` `asciinema-x86_64-unknown-linux-musl`

Verified against the real 999 MiB
`/p/scratch/synthlaion/lee27/sif/sweb_astropy.sif`, first with one-shot `apptainer exec` and then
with the bridge's actual persistent `apptainer instance start` model. The persistent-instance smoke
replicated Terminus-2's dummy session → global 10M history limit → real 160×40 login-shell session →
`asciinema rec --stdin` → send command → Ctrl-D flow. The cast contains
`APPTAINER_INSTANCE_BIND_OK`. Evidence:
`/p/scratch/synthlaion/lee27/bind_instance_smoke.0OzL4D/instance.cast`.

Use direct file binds:
```bash
--bind "$TOOLS/bin/tmux:/usr/local/bin/tmux:ro" \
--bind "$TOOLS/bin/asciinema:/usr/local/bin/asciinema:ro"
```
Do **not** rely only on `--env PATH=...` passed to `instance start`: that PATH was not present in later
`apptainer exec instance://...` calls. Direct binds at `/usr/local/bin` worked without an overlay or
symlink step.

The static asciinema 3 binary requires a UTF-8 locale. With Apptainer `--cleanenv`, `LANG=C` fails with
`asciinema requires ASCII or UTF-8 character encoding`. Set `LANG=C.UTF-8` and `LC_ALL=C.UTF-8`;
the measured SWE-bench image provides `C.utf8`.

Marianna's live JURECA `harbor_patched` tree independently contains an uncommitted `worker.py` change
which bind-mounts a private `BRIDGE_AGENT_TOOLS` tree and exposes tmux/asciinema via symlinks in an
overlay. Her `/p/scratch/transfernetx/nezhurina1/agent_tools` is permission-denied to us, so our static,
direct-bind implementation is independent and simpler.

## RESOLVED ✅ — offline/rootless OpenCode installation

**Agent choice changed to OpenCode on 2026-07-28; do not port Terminus as the target agent.**
Harbor's installed OpenCode agent runs headlessly:
```
opencode --model <provider/model> run --format=json <instruction>
```
It does not use tmux or asciinema, so the resolved injection path above is compatibility evidence, not
a prerequisite for the chosen agent.

Marianna's current Harbor fork uses the stock
`agents/installed/install-opencode.sh.j2`, which runs:
1. `apt-get update && apt-get install -y curl`
2. the NVM installer
3. `nvm install 22`
4. `npm i -g opencode-ai`

That cannot run per task on air-gapped `dc-cpu`, and the immutable SWE-bench SIF cannot be modified with
apt/npm without fakeroot.

**But Node/NVM/npm are unnecessary.** The `opencode-ai` npm package is only a thin wrapper around
platform-specific standalone binaries. On 2026-07-28 the current version was **1.18.8**. Downloaded
`opencode-linux-x64-baseline-1.18.8.tgz` from the npm registry, verified its published SHA-512 integrity,
and staged:
```
/p/scratch/synthlaion/lee27/agent_tools/bin/opencode  # 1.18.8
```
The baseline glibc build runs both on the JURECA login node and inside the measured SWE-bench SIF. The
baseline-musl package was rejected: despite its name it is dynamically linked to
`/lib/ld-musl-x86_64.so.1`, which is absent on JURECA and cannot be assumed across SWE-bench images.

Verified through a persistent `apptainer instance` that:
- `node` and `npm` are absent;
- direct bind to `/usr/local/bin/opencode` yields `opencode --version == 1.18.8`;
- the headless CLI and help load correctly;
- inline OpenCode configuration using `OPENCODE_CONFIG_CONTENT` accepts a custom
  `@ai-sdk/openai-compatible` provider and enumerates `vllm/smoke-model`.

Evidence:
```
/p/scratch/synthlaion/lee27/opencode_instance_smoke.H1kQdF/opencode-help.txt
/p/scratch/synthlaion/lee27/opencode_config_smoke.jXqgFC/models.txt
```

OpenCode writes state even for CLI startup. Bind per-instance writable directories:
```bash
--bind "$TOOLS/bin/opencode:/usr/local/bin/opencode:ro" \
--bind "$STAGING/root-local:/root/.local:rw" \
--bind "$STAGING/root-cache:/root/.cache:rw" \
--bind "$STAGING/root-config:/root/.config:rw"
```
Attempting to override `HOME` on a later `apptainer exec instance://...` is rejected by Apptainer, and
without the writable `/root/.local` bind OpenCode fails with `EROFS: read-only file system`.

Official current configuration supports any OpenAI-compatible endpoint using:
`provider.<id>.npm = "@ai-sdk/openai-compatible"`, `options.baseURL`, a model map, and optional
`options.apiKey`. Runtime inline config via `OPENCODE_CONFIG_CONTENT` is preferable here: no config file
must be written into the immutable SIF.

### Remaining OpenCode→vLLM wiring

Marianna's live router is currently reachable **from inside `sweb_astropy.sif`**:
```
http://jrlogin05i:8001/router/status
```
It reported eight `jrc*i:8000/v1` backends. A read-only `/v1/models` request was routed to
`jrc0277i`, which refused connection, so at least that router entry was stale/offline. Do not use her live
router as ours or infer readiness from the router process alone.

For our smoke, launch our own vLLM/router only after stating its key config and receiving approval. Then
use an inline provider ID such as `vllm`, base URL `http://<our-router-login-IB>:8001/v1`, the exact served
model ID, and explicit context/output limits. Patch Harbor's `OpenCode` provider validation because the
current `create_run_agent_commands()` rejects unknown provider IDs before OpenCode sees the custom config.

---

## MEASURED — real SWE-bench image pull on JURECA login node (2026-07-28)

```
$ time apptainer pull -F sweb_astropy.sif \
    docker://swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest
real 2m46.9s   user 6m51.4s   sys 1m8.1s
-rwxr-xr-x 1 lee27 synthlaion 1047486464  sweb_astropy.sif      # 999 MiB
```

**⇒ prebuild budget: ~1.0 GB/SIF × 500 ≈ 500 GB** (comfortable against ~124 TiB) and
**~2m47 × 500 ≈ 23 h serial**, so **~6 h at `MAX_PARALLEL=4`**. Confirms the earlier 4–10 h guess.

⚠ **It is CPU-bound, not network-bound** — `user 6m51` ≫ `real 2m47` means time goes to parallel blob
decompression + SIF assembly, not download. Two consequences: (a) more parallelism helps only up to the
core count, and (b) **do not run a 500-image 4-way-parallel prebuild on a shared login node** — that's
hours of heavy CPU on a node other users need, and JSC does police it. Genuine tension to resolve: login
nodes have internet but shouldn't host the load; `dc-cpu` compute nodes have the cores but are air-gapped
and need Marianna's SOCKS proxy. Options: modest parallelism on the login node, or bring the proxy path up
after all (her `prebuild_sifs.sh` already handles it — the proxy work isn't wasted, just deferred).

### Contents of a SWE-bench image
```
tmux:       MISSING          ← the overlay blocker DOES matter
asciinema:  MISSING          ← same
git:        /usr/bin/git
python:     /opt/miniconda3/bin/python
bash:       /usr/bin/bash
/testbed:   astropy/ astropy.egg-info/ cextern/ CHANGES.rst ...
git HEAD:   e66c5e38d SWE-bench
```
**Option 1 of the ladder below is eliminated** — tmux and asciinema are genuinely absent, so the injection
step is required. Start at option 2 (bind-mount static binaries), which needs no build and no namespaces.
Good news: `git`, `bash`, and a conda `python` are all present, so only the two recording tools are missing.
`git HEAD = e66c5e38d "SWE-bench"` confirms the repo is baked in at the buggy commit — re-confirming why the
SweSmith runtime-`git checkout` trick cannot transfer (see [[apptainer_bridge_swebench]]).

---

## NEXT STEPS (ordered)

1. ✅ Port the minimal bridge + OpenCode changes locally and push only to our forks.
2. ✅ Add non-job tests for command/bind construction and fail-closed behavior.
3. ✅ Run the one-task, one-worker JURECA CPU infrastructure smoke. See the measured result below.
4. **Message Marianna** — highest leverage remaining:
   - For your OpenCode migration, how do you stage/install OpenCode on air-gapped JSC workers?
   - **How does your 200-task SWE-bench eval actually run?** Your `docs/workflow.md` covers r2egym /
     swesmith / nl2bash but **omits swe-bench**, yet every published result is a SWE-bench eval ⇒ there may
     be a different path we should copy instead.
   - Exact 200-task list (100 verified folders + 100 random) for comparability.
   - Training-side context length (we train 28,672 prompt / 32,768 total; her clean eval regime is 81,920+).
5. Port any still-needed scripts. Re-point: project `jureap59`/`laionize` →
   `synthlaion`, JUWELS account `projectnucleus` → ours, user `nezhurina1` → `lee27`,
   `/p/scratch/transfernetx/nezhurina1/` → `/p/scratch/synthlaion/lee27/`, keys `~/.ssh/docker_jusuf` +
   `~/.ssh/docker`. **Prefer JURECA (`dc-cpu`/`synthlaion`) or JUSUF (`batch`/`reformo`) — NOT JUWELS**,
   whose sbatch uses her `transfernetx`. All three worker sbatches are **CPU-only, zero `--gres=gpu`**.
6. Run the first model-backed end-to-end smoke using a small Qwen model
   (`Qwen/Qwen3-0.6B` proposed) rather than the 30B training model. State the exact checkpoint/cache,
   vLLM environment and flags, port, walltime/GPU request, served model ID, task, context/output limits,
   and output path before submission. Ordinary experiment submission does not require approval.
7. Prebuild the 500 SWE-bench SIFs (~1–2 TB of ~124 TiB; 500 inodes; one-time, then cached).
8. Wire `--harbor-ref` + `APPTAINER_BRIDGE_URL` into our launcher.

## Implementation state — OpenCode bridge branch (2026-07-28)

The pinned Harbor was already newer than Marianna's documented fork and already contained the maintained
bridge at `harbor.environments.apptainer` (commits `ae5f18f9`, `ec6ebb57`, `4af44c3f`). We did **not**
copy the obsolete `apptainer_bridge/` subsystem wholesale.

Pushed only to our forks; no PRs:
```
OpenThoughts-Agent:
  repo/branch: lukedhlee/OpenThoughts-Agent :: lukedhlee/apptainer-opencode-bridge
  commit:      6d2f9cae7eef2bc5af0e7ab08c2277b96086fdc1

Harbor:
  repo/branch: lukedhlee/harbor :: lukedhlee/apptainer-opencode-bridge
  commit:      3b22fb57de1b902d655c9a5756fcc42e955a70a4
```

Harbor changes:
- `OpenCode(preinstalled: true)` skips the root `/installed-agent` mkdir and all apt/NVM/npm network
  installation, but fails closed unless `opencode --version` succeeds.
- OpenCode run conditionally sources NVM, so the same code supports both normal and preinstalled modes.
- Worker direct-binds staged `opencode`, `tmux`, `asciinema`, and `uv` into `/usr/local/bin` when present.
- With OpenCode, worker binds per-instance writable `/root/.local`, `/root/.cache`, `/root/.config`.
- `BRIDGE_REQUIRE_OPENCODE=1` fails closed if the pinned binary is missing.
- Instance and exec environments explicitly set `LANG=C.UTF-8`, `LC_ALL=C.UTF-8`.
- JURECA worker sbatch uses our `synthlaion/lee27` paths, stock `python3`, current
  `harbor.environments.apptainer`, and requires an explicit `HARBOR_SRC`; Marianna's conda and paths are
  gone.

Validation:
```
55 targeted Harbor tests passed
Ruff check + format check passed
bash -n jureca_workers.sbatch passed
git diff --check passed
```
JURECA-side non-job validation loaded the exact pushed `worker.py` and resolved these direct binds from
our real scratch tree: opencode, tmux, asciinema, plus all three writable OpenCode state directories.

OpenThoughts change:
`hpc/harbor_yaml/eval/configs/eval_opencode_apptainer_ctx32k.yaml` — one-concurrent-trial smoke config,
preinstalled OpenCode 1.18.8, vLLM OpenAI-compatible provider, 32,768 total context
(28,672 prompt + 4,096 output), compaction disabled, Apptainer SIF cache on our scratch. Parsed
successfully as Harbor `JobConfig`.

Clean JURECA clones (always pull these; do not hand-edit):
```
/p/project1/synthlaion/lee27/harbor
  HEAD 3b22fb57, clean
/p/project1/synthlaion/lee27/OpenThoughts-Agent
  HEAD 6d2f9cae, clean
```

Prepared one-task smoke input without running a job:
```
dataset: /p/project1/synthlaion/lee27/swebench-verified-smoke
dataset commit: 0e4345c345395a3371f0b79d891a59fdbf74e9a8
task: astropy__astropy-12907
Dockerfile SHA-256: 3a88c0935413cfc2e0c83f5ceecf73660f34e131037e676fe587ca709ca40894
SIF cache key: /p/scratch/synthlaion/lee27/sif/astropy__astropy-12907.sif
SIF target: /p/scratch/synthlaion/lee27/sif/sweb_astropy.sif (999 MiB)
```

### JURECA CPU infrastructure smoke — PASSED 2026-07-29

The approved infrastructure-only smoke ran as Slurm job `15476272`:
```
cluster/partition/account: JURECA / dc-cpu / synthlaion
allocation:                1 node, CPU-only
node:                      jrc0030
walltime requested:        00:30:00
bridge workers:            1 (WORKERS_PER_NODE=1)
bridge:                    jrlogin04i:9920
Harbor source:             /p/project1/synthlaion/lee27/harbor/src @ 3b22fb57
agent tools:               /p/scratch/synthlaion/lee27/agent_tools
SIF/task:                  astropy__astropy-12907, exact paths above
model/API:                 none — validate bridge lifecycle, writable /testbed, transfers, and
                           preinstalled `opencode --version` only
output:                    /p/scratch/synthlaion/lee27/dc_agent_eval/logs/
auto-chain:                disabled (MAX_CHAIN=0)
```

Measured results:
```
bridge worker registration: workers_alive=true
environment:                env-a3e25db65891, ready on jrc0030
SIF resolved:               /p/scratch/synthlaion/lee27/sif/astropy__astropy-12907.sif
opencode:                   1.18.8
tmux:                       3.7b
asciinema:                  3.2.1
/testbed writable:          yes
/testbed git HEAD:          e66c5e38d35d54e14976a9de6d93f8ae3291cdcf
exec/readback sentinel:     passed
bridge upload/download:     passed, byte-for-byte (32 bytes)
environment stop/delete:    passed
```

Cleanup completed: the environment stopped before allocation teardown; the bridge tmux session and
HTTP listener on port 9920 were stopped; job `15476272` was cancelled after the smoke and disappeared
from `squeue`. `sacct` records the deliberate cancellation after 2m40s; the terminated batch/step states
are cancellation cleanup, not a smoke failure.

### Experiment submission policy

Ordinary non-destructive experiment submissions do **not** require approval. State the key configuration
before submission and proceed. Destructive actions still require explicit approval. Separately, the
standing task-specific guard remains: do not launch an RL job.

**Development-model decision (2026-07-29):** do not use the 30B training model for the bridge/OpenCode
development smoke. Use `Qwen/Qwen3-0.6B` on one JURECA A100 instead; model quality is irrelevant for this
test. The purpose is only to exercise OpenCode → OpenAI-compatible vLLM → tool-call response → sandbox
command end to end. Proposed serving flags include one GPU (`TP=1`), 32,768 max model length,
`--enable-auto-tool-choice --tool-call-parser hermes`, and the existing OpenCode `vllm/<served-id>`
routing.

Exact next-experiment configuration, measured/prepared but not yet submitted:
```
cluster/account:        JURECA / synthlaion
vLLM partition:         dc-gpu-devel
vLLM allocation:        1 node, 1 task, 1x A100 GPU, 16 CPUs, 64 GiB RAM
vLLM walltime:          00:30:00
vLLM environment:       /p/scratch/synthlaion/lee27/envs/skyrl-venv
vLLM version:           0.9.2
checkpoint:             Qwen/Qwen3-0.6B
checkpoint snapshot:    /p/scratch/synthlaion/lee27/cache/hf/hub/models--Qwen--Qwen3-0.6B/
                        snapshots/c1899de289a04d12100db370d81485cdf75e47ca
HF_HOME:                /p/scratch/synthlaion/lee27/cache/hf
VLLM_CACHE_ROOT:        /p/scratch/synthlaion/lee27/cache/vllm
served model ID:        qwen3-0.6b
listen/route:           0.0.0.0:8000; OpenCode uses http://<allocated-gpu-node>:8000/v1
tensor parallel:        1
context/output:         32,768 / 4,096
tool calling:           --enable-auto-tool-choice --tool-call-parser hermes
other serving flags:    --dtype bfloat16 --gpu-memory-utilization 0.80 --max-num-seqs 8
vLLM output:            /p/scratch/synthlaion/lee27/dc_agent_eval/logs/
                        opencode_vllm_qwen3_0p6b_%j.out

bridge partition:       dc-cpu
bridge allocation:      1 CPU node, WORKERS_PER_NODE=1, MAX_CHAIN=0
bridge walltime:        00:30:00
bridge server:          actual login-node IB hostname, port 9920
Harbor/OpenCode:        Harbor 3b22fb57 / OpenCode 1.18.8
task/SIF:               astropy__astropy-12907 / prepared SIF above
OpenCode route:         vllm/qwen3-0.6b
trial count:            1
evaluation output:      /p/scratch/synthlaion/lee27/dc_agent_eval/
```
The installed vLLM CLI was checked read-only and exposes all proposed
`--enable-auto-tool-choice`, `--tool-call-parser hermes`, `--max-model-len`, and
`--served-model-name` flags.

**Estimate: smoke working ~1 day; SWE-bench SIFs ~2–3 days.** Risk is distributed plumbing across two
schedulers (tunnel liveness, worker↔server auth, dispatcher layer, SLURM 6h walltime vs her `MAX_CHAIN=5`
auto-requeue), not algorithms. It always breaks somewhere on the first port.

---

## Source material

Her fork: **`marianna13/harbor` @ branch `marianna/beam`** (`064ec1a`), explored read-only via `gh api`.
`src/harbor/environments/apptainer_bridge/` ≈ 110 KB — a subsystem, not a backend flag:
`worker.py` (72 KB), `server.py` (20 KB), `harbor_env.py` (17 KB), `vllm_router{,_async}.py`,
`start_bridge.sh`, `tunnel_monitor.sh`, `prebuild_sifs.sh`, `rebuild_r2egym_sifs.sh`,
`{jureca,jusuf,juwels}_workers.sbatch`, `docs/workflow.md`, `CONNECTION_INFO.md`.
Her live tree is readable on JURECA at `/p/project1/laionize/marianna/dc_agent`.
Also in that fork: `agents/terminus_structured/` — but **she is migrating off terminus to OpenCode**, so
treat it as a fading target; our pinned harbor's `AgentName` enum has no `terminus-structured`.

## Local ↔ Jupiter checkout state (measured 2026-07-28)

Both checkouts use branch **`lukedhlee/vista-moe-grpo-30b`**, but their HEADs differ:
```
local:   06a4e45bc46970301a0dcbed47fb3024c1edfdd9
Jupiter: 903290a0ce020bfb24557268e7baad7d292b0e5d
```
`903290a0` is the merge-base; local is exactly **one commit ahead, zero behind**:
`06a4e45b add GSM8K generation-budget probe (paired, inference-only)`. The two probe files are already
staged on Jupiter and byte-identical to the local commit, so Jupiter's *worktree content* contains that
commit even though its HEAD has not advanced.

The four shared modified launcher files are byte-identical across local and Jupiter:
```
hpc/ray_utils.py
hpc/rl_config_utils.py
hpc/rl_launch_utils.py
hpc/sbatch_rl/universal_rl.sbatch
```

Jupiter uniquely modifies:
- `hpc/hpc.py` (6 insertions / 4 deletions): lee27-specific CUDA library path, scratch Ray log/spill
  locations, removal of inaccessible feuer1 proxychains, and lee27 miniforge activation.
- `hpc/dotenv/jupiter.env` (25 insertions / 36 deletions; may contain environment-specific values, so
  reconcile deliberately rather than copying/blindly diffing into chat).
- several `.bak_20260726_*` files plus runtime `envs/`; these are cluster-local artifacts, not branch
  source.

Local uniquely has numerous untracked Vista configs plus the current `ai_memory/` updates. Thus the
**committed code does not significantly diverge**, but both worktrees are dirty and Jupiter has two
important machine-local tracked edits. Switching the current checkout to a new branch would carry all
dirty changes with it and would not isolate the bridge.

**Bridge isolation created 2026-07-28:**
```
branch:   lukedhlee/apptainer-opencode-bridge
worktree: /Users/lukedhlee/OpenThoughts-Agent-apptainer-bridge
base:     06a4e45bc46970301a0dcbed47fb3024c1edfdd9
```
The new worktree is clean. Port only bridge / OpenCode files there. The original dirty local checkout and
Jupiter checkout were left untouched. Do not branch Jupiter independently; clusters should pull the
pushed local bridge branch.

## Standing constraints (do not violate)

- **Local clones are ground truth; clusters never diverge.** Edit locally → push → `git pull` on the
  cluster. No hand-editing, no patch-by-rsync. An unmerged marin-fork fix rides `--harbor-ref`.
  ⚠ The Jupiter clone currently has ~6 uncommitted files — pre-existing violation, needs reconciling.
- **Never `find`/`du` on Jupiter GPFS** — inode quota is the binding constraint; exhaustion can get you
  banned. Same caution on JURECA scratch.
- Daytona snapshot caps are **HARD** — clean stale, never raise `max_new_snapshots` / `max_org_snapshots`.
- Snapshot manager: always pass `--api-key-env DAYTONA_API_KEY` (default is a different org).
- `enable_db_registration: false` in YAMLs; ≤6 RUNNING RL jobs per cluster; never kill a RUNNING job
  without explicit permission.
- Secrets only from `$DC_AGENT_SECRET_ENV`; a committed `dtn_…` key is a leak — rotate, don't fix-forward.
- Push access: `origin` (open-thoughts) **denies** lukedhlee. Use the `fork` remote; on Jupiter the remote
  is `lukefork`. Commit `06a4e45b` (the GSM8K probe) is on the fork only, NOT origin.
- **Push/PR guard (added 2026-07-28): push only to repositories owned by `lukedhlee` for now. Do not
  push branches to upstream or another person's fork, and do not create PRs.** Current successful pushes:
  `lukedhlee/OpenThoughts-Agent` branch `lukedhlee/apptainer-opencode-bridge` @ `6d2f9cae` and
  `lukedhlee/harbor` branch `lukedhlee/apptainer-opencode-bridge` @ `3b22fb57`. No PRs created.
  Permission checks against `marin-community/harbor` and `marianna13/harbor` used `git push --dry-run`;
  both returned 403 and changed no refs.

## Also parked (unrelated to apptainer)

**GSM8K budget probe, Jupiter job `1075086`** — PENDING behind Jupiter's acceptance-benchmarking
reservation (started 09:00 CEST 2026-07-28). Inference-only; tests whether step-0 `pass_at_1=0.4511` was a
generation-budget truncation artifact (`max_generate_length: 1024`, failures pinned at ~950 tokens vs ~640
for successes, against 78.2% reported zero-shot). Files: `scripts/analysis/gsm8k_budget_probe.py`,
`eval/jupiter/gsm8k_budget_probe.sbatch`.

---

## Final bridge/evaluation status (2026-07-29)

### Go/no-go: PASSED

The non-RL, model-backed OpenCode → vLLM smoke passed on JURECA:

- vLLM job `15476440`: one A100 serving cached `Qwen/Qwen3-0.6B`.
- bridge worker job `15476441`: one `dc-cpu` worker.
- task: `astropy__astropy-12907`.
- OpenCode successfully issued a Bash tool call that created the requested sentinel inside `/testbed`,
  then issued a Read tool call that verified it. This proves the vLLM route, automatic tool parsing,
  OpenCode tool dispatch, bridge execution, and sandbox mutation all work together.
- The small 0.6B model later looped until context exhaustion; that is model quality, not a bridge
  integration failure. Both smoke jobs were cancelled only after the required behavior was verified.

No RL job was involved.

### Forks and launcher wiring

Both clean worktrees are on `lukedhlee/apptainer-opencode-bridge`, clean, pushed, and equal to upstream:

```
OpenThoughts-Agent: /Users/lukedhlee/OpenThoughts-Agent-apptainer-bridge
fork:               lukedhlee/OpenThoughts-Agent
HEAD:               bc8378ed

Harbor:             /Users/lukedhlee/harbor-apptainer-bridge
fork:               lukedhlee/harbor
HEAD:               02b06bc5
```

The normal evaluation launcher now accepts and propagates `--harbor-ref` and
`--apptainer-bridge-url`, verifies the exact Harbor checkout HEAD fail-closed, and sets the required
runtime import path. The relevant launcher wiring landed in OpenThoughts-Agent commit `2c2b6be0`.

Harbor integration fixes include offline verifier-wheel handling and artifact mounts, clean exec
environments, library-path preservation, and a narrow Django compatibility shim. The final Django fix
at `02b06bc5` changes only the offline verifier copy of
`django/utils/version.py` for affected Django tasks, avoiding Python 3.5 `Popen(cwd=repo_dir)` `EPERM`
on JSC while preserving repository/current-directory semantics. Focused tests and Ruff pass.

### Multi-task validation

Final pristine oracle validation:

```
/tmp/harbor-jobs/apptainer-swebench-three-oracle-v2-final
```

Tasks:

- `astropy__astropy-13236`
- `django__django-10097`
- `matplotlib__matplotlib-20826`

Result: **3/3 reward 1, zero exceptions**.

### Comparable subset provenance

Marianna's live JURECA scripts consistently use:

```
DCAgent2/swebench-verified-random-100-folders
HF snapshot: a2e51e9e0e8029156ed340719eb8cc7ceee3ed1a
JURECA copy: /p/project1/synthlaion/lee27/swebench-verified-random-100-a2e51e9
```

The pinned snapshot contains exactly 100 task directories. No credible manifest or deployed script
for a distinct additional 100 tasks was found. The public `sample100` overlaps this set by 18 tasks,
so combining them would produce only 182 unique tasks and would not be Marianna's exact comparable
200-task evaluation. Do not invent or label a 200-task subset until its second manifest is obtained.

### SIF cache and JSC inode outcome

Prebuild job `15476943` completed `0:0` in `00:09:14`:

```
cache: /p/scratch/synthlaion/lee27/sif
built this run: 24
already cached: 76
independent audit: total=100 valid=100 missing=0 invalid=0
```

The final prebuild script uses node-local `${LOCALSCRATCH:-${TMPDIR:-/tmp}}` for Apptainer
unpack/build work and atomically copies only completed SIFs to GPFS scratch. This avoids the inode
and Apptainer futex failures seen when earlier attempts unpacked directly on GPFS.

Earlier cancelled builds left two exact temporary trees. They were removed single-threaded:

```
/p/scratch/synthlaion/lee27/apptainer_tmp/prebuild_15476552
/p/scratch/synthlaion/lee27/apptainer_tmp/prebuild_15476640
```

After cleanup, `/p/scratch` reports 2,476,273 free inodes and 44% inode use (down from 100%).
The validated SIF cache was preserved.

### Current cluster state

At the final 2026-07-29 check:

- no jobs in `squeue` for the user;
- no bridge/proxy tmux server;
- no vLLM, worker, GPU, or RL job left running;
- both cancelled-build temporary trees are gone;
- all 100 cached SIFs pass `apptainer inspect`.

### Remaining work

The infrastructure, model-backed smoke, launcher integration, cache, and small validation are complete.
The only unresolved prerequisite for a claimed comparable 200-task run is authoritative provenance for
Marianna's alleged second 100-task manifest. The confirmed pinned 100-task evaluation can be launched now.
