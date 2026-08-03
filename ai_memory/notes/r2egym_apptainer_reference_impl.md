# r2egym on the apptainer bridge — the co-lead's reference implementation

Marianna already runs **r2egym agentic RL on the apptainer bridge, on JUPITER, with an 8B model** — i.e.
almost exactly our target experiment. This is the reference to copy rather than re-derive.
Read before building any r2egym sandbox path, and before trusting a raw-r2egym reward curve.

Discovered 2026-07-29 by reading her live JURECA tree read-only (it is world-readable):
`/p/project1/laionize/marianna/dc_agent`. Parent: [[handoff]]. Network facts + the Daytona blocker that
sent us here: [[gotchas]]. SWE-bench-side bridge state: [[apptainer_bridge_handoff]].

---

## ⚠ FLAG FIRST — leaked Docker Hub credential (not ours to fix)

`dc_agent/bash-scripts/prebuild_r2egym_worker.sh` exports a **plaintext Docker Hub PAT** in
`APPTAINER_DOCKER_PASSWORD` / `SINGULARITY_DOCKER_PASSWORD`, world-readable to anyone with
`/p/project1/laionize` access. **Marianna must revoke and reissue it.** Do NOT use it and do NOT copy the
value anywhere — the token value is deliberately not recorded in this file. It exists only to lift
Docker Hub anonymous-pull rate limits, so nothing we build requires it. Standing rule: a committed key
is a leak; rotate, don't fix-forward.

## The load-bearing insight: r2egym belongs on JUPITER, not JURECA

Her RL script is `deepswe_repro/run_rl_deepswe_8b_repro_apptainer_seqmean_r2egym_learnable.sh`
(419 lines) and it runs on **Jupiter**: `PARTITION="booster"`, `ACCOUNT="reformo"`, `/e/...` datasets,
`NUM_NODES=12`, `NUM_GPUS_PER_NODE=4`.

That works because r2egym's base image is `python:3.6-slim-buster` — **multi-arch, with arm64** — so it
runs natively on GH200. SWE-bench's `sweb.eval.x86_64.*` images are x86_64-only and genuinely cannot,
which is the *only* reason JURECA is involved at all. See [[gotchas]] for the build-vs-pull distinction.

## Exact harbor wiring (the three lines that matter)

```
-S +terminal_bench_config.harbor.environment_import_path=harbor.environments.apptainer_bridge.harbor_env:BridgeApptainerEnvironment
-S +terminal_bench_config.harbor.bridge_url=http://10.128.1.1:9920
-S +terminal_bench_config.harbor.sif_cache=<scratch>/sif_cache
```

`10.128.1.1` is a **Jupiter internal IB address** — the bridge server runs on a Jupiter login node and
compute nodes reach it over `10.128.x` (measured reachable; see [[gotchas]]). Note she uses the OLD
`apptainer_bridge` subsystem, whereas our SWE-bench port uses the newer maintained
`harbor.environments.apptainer` — reconcile deliberately, don't assume they're interchangeable.

Other config of note: `AGENT=terminus-structured` (⚠ our bridge was only ever validated with
**OpenCode**), `MAX_EPISODES=50`, `TIMEOUT_SEC=1800`, `TRAIN_BATCH_SIZE=64`, `LR=3e-6`,
`EPOCHS=4`, `NUM_PARALLEL_WORKERS=128`, `NUM_STALENESS_STEPS=1`, `CKPT_INTERVAL=5`, `EVAL_INTERVAL=5`,
`SSH_KEY=~/.ssh/docker`.

## How the SIFs get built

`bash-scripts/prebuild_r2egym_sifs.sbatch` → `srun` across **4 compute nodes** →
`prebuild_r2egym_worker.sh`, 48-way parallel per node, driven by a
`r2egym_build_list.txt` of `task_name df_hash docker_image` lines. SIFs are named
`build_${task_name}-${df_hash}.sif` and skipped if already present (idempotent).

`worker.py:_build_sif()` converts the Dockerfile to a `.def` (`Bootstrap: docker` / `From: <base>`,
handling only `FROM`/`RUN`/`ENV`/`WORKDIR` — sufficient for r2egym), then runs `apptainer build`, trying
`--fakeroot` first and falling back to non-fakeroot. Apptainer cache/tmp are redirected to scratch.

## ★ THE SCIENTIFIC FINDING — raw r2egym COLLAPSES

Her script states the goal outright: *"does filtered-r2egym train STABLY with non-zero reward (unlike
raw r2egym, which collapsed)"*.

So **training on the full 3,328-task r2egym is a configuration already known to collapse.** The fix she
built is a **learnable band**:

- Keep only tasks where the BASE model scores **`0 < pass@8 < 1`**, plus `c/n <= 0.6` for a harder band.
- Measured over a 3,000-task generation pass **under the exact training constraints** (40960 ctx,
  50 turns, temp 1.0, terminus-structured, no summarize, timeout/context masked as fail) — the same
  masking the training run uses, so the filter and the reward agree.
- Yield: **740 tasks** train + **100 held out** for in-distribution eval (`fast_band_split.py`,
  `build_learnable_dataset.py`, `merge_split_learnable.py`).
- ⚠ **The band is MODEL-SPECIFIC** — she warns "do not swap it." A band built for her GLM checkpoint does
  NOT transfer to Qwen3-8B; we would have to measure our own.

This is the same lesson as [[gsm8k_format_artifact]] arriving from a different direction: measure whether
reward can move at all before spending a run on it.

## Also worth copying: her eval design

Three val sets, which together separate capability from format:
1. `swebench-verified-random-100-folders` @ `a2e51e9e` — cross-benchmark transfer (the pinned set we
   already have 100 SIFs for).
2. `r2egym_learnable_heldout` — in-distribution, disjoint from train.
3. `swebench_verified_100_r2egymfmt` — **format control**: swebench tasks reframed in r2egym's
   instruction format (`reformat_swebench_to_r2egym.py`). Comparing 1 vs 3 for the same checkpoint
   isolates the format confound from real transfer.

`EVAL_BEFORE=true` — she evaluates the base model at step 0, which is exactly the base→RL Δ denominator
our own plan needs.

## Open questions before copying this

- Is `/p/scratch` (her `sif_cache`) visible from Jupiter, or does she keep a Jupiter-side copy? Her
  datasets are `/e/...` but the sif_cache path is `/p/...` — unresolved.
- Does the bridge support **terminus-2**? Ours is validated only with OpenCode; she uses
  `terminus-structured`, which our pinned harbor's `AgentName` enum does not even contain.
- Building needs internet (docker pull + apt + pip) but compute nodes have none — pre-pull base layers
  into `APPTAINER_CACHEDIR` from a login node, or proxy the compute nodes. Which does she do?

---

## ⚠ CORRECTION 2026-07-29 (later) — the reference impl is NOT Jupiter-local

The claim above ("r2egym belongs on JUPITER, not JURECA" / bridge + workers both on Jupiter) was an
INFERENCE from the RL script's `PARTITION="booster"`. It is **wrong**. Measured by reading her scripts
read-only on JURECA:

**Her actual architecture is split across two clusters:**
- **RL job on Jupiter** — `/e/...` paths, `PARTITION=booster`, `ACCOUNT=reformo`, `NUM_NODES=12`,
  `POLICY_NUM_NODES=4`, `NUM_INFERENCE_ENGINES=8`, `--reservation=reformo`.
- **Bridge server on a Jupiter LOGIN node** — `10.128.1.1:9920`.
- **Sandbox workers on JUWELS**, a separate CPU cluster —
  `apptainer_bridge/juwels_workers.sbatch`: `--nodes=8`, `--partition=batch`,
  `--account=transfernetx`, `--cpus-per-task=48`, `--mem=0`, `--time=24:00:00`, **no `--gres=gpu`**.
- **Joined by a reverse SSH tunnel from the JUWELS login node into Jupiter:9920.**
- SIF cache is JUWELS-side: `/p/scratch/transfernetx/nezhurina1/sif_cache` (x86_64 — does NOT transfer
  to Jupiter GH200).

Her sbatch header states the design verbatim: *"Each compute node: 1 dispatcher polls localhost:9920
(via login node); N workers consume from local queue (no HTTP polling). Only 32 dispatchers poll
through tunnel (not 512 workers)."* Plus auto-chain: the worker job submits its successor before
starting, so workers outlive the 24h limit.

### Three consequences — all of these reverse earlier conclusions

1. **Cross-cluster sandboxes are PROVEN for agentic RL training, not fatal.** The earlier
   latency objection to a remote sandbox cluster is refuted by her running exactly that in production
   for this experiment. The dispatcher/local-queue design is how it's made to work.
2. **The "Jupiter compute can't reach JURECA" probe result is IRRELEVANT to this design.** Traffic never
   flows that way. Jupiter compute → Jupiter login bridge (internal `10.128.x`, measured reachable);
   the CPU cluster's LOGIN node initiates the tunnel inbound. ⇒ This also dissolves the routing gap
   flagged against the SWE-bench eval plan in [[apptainer_bridge_handoff]].
3. **Option B ("Jupiter-local bridge") in [[handoff]] § Sandbox is NOT what the reference impl does,**
   and it is the worse option: Jupiter has **no CPU-only partition** (measured
   `sinfo`: only `booster` + `largebooster`, both `gpu:gh200:4`, 288 CPUs / 878 GB / node), so
   Jupiter-hosted sandboxes occupy GH200 nodes to run pytest. **JUWELS ≈ JURECA for our purposes
   (operator, 2026-07-29): both have CPU clusters** ⇒ port her design with **JURECA `dc-cpu`
   (`synthlaion`)** as the sandbox cluster, where our worker sbatch is ALREADY ported and smoke-passed.

### Also measured
- Her Jupiter eval datasets confirm the three-val-set design: `r2egym_learnable_heldout`,
  `swebench-verified-random-100-folders` @ `a2e51e9e`, `swebench_verified_100_r2egymfmt`.
- **Her learnable splits are permission-denied to us** —
  `/e/data1/datasets/playground/ot/hf_hub/r2egym_learnable_{train,heldout}` → `Permission denied`.
  So the band tooling must be requested or rebuilt; it cannot be read.
