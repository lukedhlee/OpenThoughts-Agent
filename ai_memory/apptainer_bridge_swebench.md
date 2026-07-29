# Apptainer bridge + SWE-bench as a validation set

Investigated 2026-07-28. Companion to [[jureca_marianna_setup_plan]] (her fork refs / JURECA layout)
and the eval-regime confound in `handoff.md`.

Visual explainer (2026-07-28): https://claude.ai/code/artifact/6447de1d-c124-4830-b2fc-165173afd06b

## The fact that drives everything

**SWE-bench task images are `x86_64` only. Jupiter is `aarch64` (GH200).**

```
FROM swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest
                        ^^^^^^
```

x86 containers cannot run natively on ARM, and each task runs a real test suite, so qemu emulation is
not viable. **SWE-bench containers cannot execute on Jupiter compute nodes at all.**

ARM variants (`swebench/sweb.eval.arm64.*`) DO exist on Docker Hub but coverage is partial —
**21/40 sampled (~53%)**, missing skewed toward django / sphinx / xarray. An ARM-only subset would be a
different, repo-biased benchmark and NOT comparable to Marianna's 200 or any published number. Do not
go down this road for headline numbers.

## Measured: SWE-bench is 1 snapshot per task (Daytona)

Hashed all 500 `environment/` dirs of `DCAgent/swebench-verified` the way harbor does
(content hash of the env dir):

```
env dirs: 500   UNIQUE ENV HASHES: 500   files per env dir: {1: 500}
500 unique FROM lines, zero duplicates
```

Irreducible 1:1 because each Dockerfile pins a distinct upstream base image. Contrast **r2egym: 3,328
tasks → 7 snapshots**. SWE-bench is the pathological case.

### Harbor NEVER reaps snapshots (verified in code)

`daytona.snapshot.delete` in `harbor/environments/daytona.py` is called ONLY on error paths:
- ERROR-state snapshot deleted before a retry (`:1223`, `:574`)
- an `existing` snapshot deleted before recreate (`:1266`, `:606`)

There is **no cleanup of successfully-created snapshots**. They accumulate monotonically until
`scripts/daytona/daytona_snapshot_manager.py --delete-stale` (14-day idle threshold) removes them. So a
200-task eval `SnapshotCapExceeded`s around task ~40. **Do not plan around lazy/rolling snapshot reuse.**

### The useful corollary

Snapshots are keyed by env hash and **shared across every run using those tasks**. So repeated evals of
the same val set cost the SAME snapshots, not N×. The binding constraint is **val-set size**, not eval
frequency.

Org state 2026-07-28 (`cli` org, `--api-key-env DAYTONA_API_KEY`): **20/60 used, 0 stale, headroom 40** —
and all 20 are Daytona's own base images (`daytonaio/sandbox:*`, `daytona-vm-*`, `windows-*`), i.e. there
are **zero `harbor__<hash>` task envs in the org**; the 7 r2egym envs from earlier were reclaimed.
⇒ **Max Daytona SWE-bench val set today = ~40 distinct tasks, evaluated as often as we like.**
n=40 gives ±7.7 pts at 95% vs ±3.5 at n=200 — enough to see Marianna's +10/+11.5, not a +2.5.

We hold only ONE Daytona key (`DAYTONA_API_KEY`); `DAYTONA_DATA_API_KEY` is not in our secrets, so we
cannot audit the other org.

## Her architecture (marianna13/harbor @ marianna/beam)

`src/harbor/environments/apptainer_bridge/` — a subsystem, NOT a backend flag. ~110 KB:
`worker.py` (72 KB), `server.py` (20 KB), `harbor_env.py` (17 KB), `vllm_router{,_async}.py`,
`start_bridge.sh`, `tunnel_monitor.sh`, `prebuild_sifs.sh`, `rebuild_r2egym_sifs.sh`,
per-cluster `{jureca,jusuf,juwels}_workers.sbatch`, `docs/workflow.md`, `CONNECTION_INFO.md`.

Also in that fork: `src/harbor/agents/terminus_structured/` — the agent behind all her
`laion/rl_*_terminus-structured` models (structured tool calls: bash/view/edit/create/search).
**She is migrating off terminus to OpenCode**, so treat terminus_structured as a fading target.

### Topology — two flavors

**(a) Cross-cluster** (`CONNECTION_INFO.md`, Jupiter + JUWELS):
```
Jupiter (bridge server :9920, harbor client, vLLM)
   ↑ reverse SSH tunnel
JUWELS login :9920
   ↑
JUWELS compute: 1 dispatcher/node → 16 workers/node → apptainer instances
```
Dispatcher layer exists to protect the tunnel: 32 nodes × 16 workers = 512 pollers → 32.
Workers never poll over HTTP; they only POST results back.

**(b) Single-cluster on JURECA** (per [[jureca_marianna_setup_plan]]): `dc-cpu` apptainer worker pool +
`dc-gpu` vLLM pool + login-node router. **JURECA is x86_64**, so this needs NO cross-cluster tunnel and
has NO ARM problem.

**The sandbox cluster needs NO GPUs.** All three of her worker sbatches are CPU-only — zero `--gres=gpu`:

| script | partition | account | shape |
|--------|-----------|---------|-------|
| `jureca_workers.sbatch` | `dc-cpu` | **synthlaion** ✅ ours | 1 node × 128 cores, 6h |
| `jusuf_workers.sbatch` | `batch` | **reformo** ✅ ours | 4 nodes × 128 cores, 12h |
| `juwels_workers.sbatch` | `batch` | `transfernetx` ❌ hers | 8 nodes × 48 cores, 24h |

⇒ **Use JURECA or JUSUF, NOT JUWELS** — she already wrote those two against projects we hold.

GPU choice is orthogonal and only matters if we ALSO move the model off Jupiter:
- keep model on Jupiter (GH200) → fast, but needs the cross-cluster tunnel
- move eval vLLM to JURECA `dc-gpu` (A100) → no tunnel; A100 is fine for *inference*, would only hurt
  for 30B-A3B *training* (40 GB/GPU vs 96)

### Two operational wrinkles

1. **JSC compute nodes have no internet.** `prebuild_sifs.sh` pulls base images through a SOCKS proxy
   (`start_proxy_tunnel.sh` + proxychains-ng). Any task whose tests `apt-get`/`uv add` needs it live.
2. **Her setup is hardwired to her identity** — project `jureap59` / `laionize`, JUWELS account
   `projectnucleus`, user `nezhurina1`, `/p/scratch/transfernetx/nezhurina1/`, keys `~/.ssh/docker_jusuf`
   and `~/.ssh/docker`. Re-pointing is mechanical but touches many files.

### The SIF trick that probably saves us (from `docs/workflow.md`)

| dataset | unique Dockerfiles | SIFs actually built |
|---------|-------------------:|--------------------:|
| R2EGym | 1,785 | 1,785 |
| **SweSmith** | **2,500** | **38** (base images only) |
| NL2Bash | 1,570 | 1 |

> "38 unique base images, 2500 unique Dockerfiles (differ by `git checkout` branch). Only 38 base SIFs
> pre-built. Worker does `git checkout` at runtime via `_extract_run_commands()`."

**VERIFIED 2026-07-28 — the trick does NOT transfer to SWE-bench.** All 500 SWE-bench Dockerfiles are
byte-identical except the `FROM` line:

```
FROM swebench/sweb.eval.x86_64.<instance>:latest   ← the ONLY difference
WORKDIR /testbed
RUN curl -LsSf https://astral.sh/uv/0.7.13/install.sh | sh
RUN mkdir -p /logs
```
```
Dockerfiles containing "git checkout"/"git reset":  0 / 500
distinct non-FROM command sets:                    1  (all 500 share it)
```

SweSmith's differentiation was **a Dockerfile command** (`git checkout <branch>`) so the worker could
hoist it to runtime via `_extract_run_commands()`. SWE-bench's is **baked into the image layers** —
`/testbed` already holds the repo at the buggy commit PLUS an era-matched dependency env. Nothing to
hoist. Reconstructing it at runtime = re-implementing SWE-bench's own image build (why they publish 500
images); slow, fragile, and any deviation changes test outcomes → destroys the comparability that was
the whole point.

**But we don't need it.** Daytona's 60 is a HARD QUOTA; apptainer's cost is disk + a one-time build:

| | Daytona | Apptainer |
|---|---------|-----------|
| 500 task envs | ✗ cap 60 | ✓ 500 SIF files |
| inodes | — | 500 (trivial) |
| bytes | — | ~1–2 TB of ~124 TiB free |
| build | — | one-time ~4–10h via SOCKS proxy, then cached |

⇒ **The bridge solves SWE-bench with or without the trick.** Don't spend effort chasing base-image reuse.

**Note her SIF docs cover r2egym / swesmith / nl2bash but NOT swe-bench**, yet all her published results
are SWE-bench evals ⇒ her SWE-bench eval likely runs through a DIFFERENT path than the training bridge.
Ask before copying the wrong subsystem.

## Recommended plan (two tracks)

1. **In-flight validation → r2egym held-out slice, every 10 steps.** Same 7 snapshots as training, zero
   cap pressure, zero new infra. Gives the learning curve + collapse detection, which is what in-flight
   val is FOR. Marianna's "I rarely use ID held-out sets" is about what she *reports*, not what a run
   steers by. **This keeps the bridge off the launch critical path.**
2. **Headline numbers → SWE-bench on the best 2–3 checkpoints only.** Start with the 40-task Daytona set
   (available now); upgrade to full 200 via the bridge when built.

Estimate for the bridge: **2–4 days** if JSC access + allocation check out. Risk is plumbing (tunnels,
keys, accounts, proxy), not code. Worth building independent of this experiment — it replaces a
rate-limited external service with compute we already own and removes the snapshot cap entirely.

## The `container` group gate (RESOLVED 2026-07-28)

`apptainer --version` on JURECA *and* Jupiter returned, verbatim:

```
User lee27 is not in group container and hence not authorized to use apptainer
Please visit JUDOOR to sign the Container Service Level Description, afterwards you have
the container group set and are allowed to use apptainer across our systems.
```

`/usr/bin/apptainer` + `/usr/bin/singularity` ARE installed (ELF binaries, group check compiled in —
not a wrapper script to bypass). `podman` / `docker` / `enroot` / `udocker` are all ABSENT, so there is
no alternative runtime; the group is the only path.

**Exact JuDoor path** (per [JSC docs](https://apps.fz-juelich.de/jsc/hps/jureca/container-runtime.html)):
Software → *Request access to restricted software* → **Access to other restricted software** →
**Container Runtime Engine** → **`Get Access`** → accept the SLD.
The card looks inert until clicked; the `Get Access` button is a second reveal. JSC's error text says
"Container Service Level Description" but the UI labels it "Container Runtime Engine" — that mismatch is
why searching for the document by name fails. **Accepted 2026-07-28.**

⚠ **Propagation takes hours** ("due to caching effects") — `id -Gn` lacking `container` right after
accepting is expected, not a failed signup.

**Side finding:** we have never been in this group, so every run to date used the venvs
(`envs/rl`, `envs/rl-megatron`), NEVER the SIF paths our ops docs describe.

**SLD item 2 — JSC Container Build System.** Building images *from scratch* on JURECA is forbidden
(needs root); JSC instead provides an external Dockerfile→image webservice with a CLI on the login
nodes. Note the distinction: `apptainer pull docker://...` is a *conversion* of a published image and
should need none of this — the build path matters only for Marianna's `Bootstrap: localimage`
tmux/asciinema overlay. Rootless `apptainer build --fakeroot` is the documented alternative "with
limitations". Untested; verify alongside the SOCKS-proxy pull.

## Access status (2026-07-28)

We have JSC projects: `ccstdl`, `mmlaion`, `reformo`, `laionize`, `synthlaion` (`jutil user projects`).
- **Jupiter** — works, but IP/key registration drops intermittently (twice in one session; error mutates
  `publickey` → `keyboard-interactive` when the key falls out of the accepted set). Operator can re-clear.
- **JURECA** — ssh config entry exists, `Permission denied (publickey)`; needs clearing. **Highest priority**,
  since it's the x86_64 single-cluster path.
- **JUWELS / JUSUF** — **not in `~/.ssh/config` at all**; need entries (`juwels01.fz-juelich.de`,
  `jusuf.fz-juelich.de`) + key registration.

## Open questions for Marianna

1. How does the 200-task SWE-bench **eval** run — apptainer_bridge, or a separate path? (Her SIF docs
   omit swe-bench.)
2. Does the SweSmith 38-base-SIF + runtime-`git checkout` trick extend to SWE-bench's ~12 repos?
3. Exact 200-task list (100 verified folders + 100 random SWE-bench) for comparability.
4. Which JSC account/cluster do the workers run under — would `ccstdl`/`laionize` work for us?
5. Training-side context length (we train 28,672 prompt / 32,768 total; her clean eval regime is 81,920+).
