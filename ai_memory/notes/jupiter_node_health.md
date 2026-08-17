# Jupiter node health: ghost GPUs, hangs, and the exclusion list

Consolidated picture of the JUPITER booster node situation as of **2026-08-17**
(the post-unclog incident storm: incidents 37–49). Read this before touching
`hpc/hpc.py` `node_exclusion_list` or talking to JSC support.

## 1. Ghost nodes (stranded GPU memory) — the main killer

**Phenomenon:** Slurm hands us a "free" node whose GPUs still hold 60–84GB of
memory belonging to a previous job. Our job then OOMs — at NCCL init, at model
load, or (worst) 20–50 min in at the first backward peak.

**Evidence pattern (from PyTorch OOM accounting):** 95GB GPU, MBs free, our own
processes holding only 5–35GB → the gap is memory torch cannot enumerate =
foreign residue. Incident table (2026-08-17):

| Node | Foreign residue | Died at | Job |
|---|---|---|---|
| jpbo-076-41/42 | (OOM at NCCL stream init) | init | 1394803 |
| jpbo-107-0x | large | resume load | 1392688 |
| jpbo-076-38 | ~82GB | resume load | 1392690 |
| jpbo-026-41 | ~84GB | resume load | 1392691 |
| jpbo-040-24 | ~62GB | gs22 backward (41 min in) | 1392692 |
| jpbo-074-02 | ~60GB | gs22 backward (49 min in) | 1397228 |
| jpbo-037-39 | ~84GB | init_model (15 min in) | 1396798 |

**Key discovery — the residue is TRANSIENT.** A probe sweep at 20:05 CEST found
ALL of the above nodes completely clean (2–127 MiB used, no processes, 80GiB
cuMemAlloc OK on every GPU). Freshest case cleared in <2.3h. But 1397228 hit
residue 49 min into its own run, so lifetimes range **minutes to hours** — tied
to how long the leaked process survives.

**Mechanism (working theory):** a cleanup race. Slurm returns nodes to the pool
before the previous job's GPU memory is actually released (leaked processes in
D-state / slow reaping). Freshly-vacated idle nodes are preferentially handed to
the next job → incoming jobs land inside the residue window. This is why the
hours right after a machine-wide "unclog" (big invisible jobs ending; idle pool
churning) are a minefield: 2026-08-17 had 7 ghost incidents in ~6 hours, all on
freshly-vacated nodes.

**NOT ghosts (misdiagnoses to avoid repeating):**
- Levanter/JAX `RESOURCE_EXHAUSTED` with an allocator stats line `Limit: 71.25GiB`
  (or 87.40GiB) = **our own `XLA_PYTHON_CLIENT_MEM_FRACTION` cap**, not residue.
- `Disk quota exceeded` in wandb staging threads = non-fatal background noise.
- 8-rank SIGABRT + "Shutdown barrier DEADLINE_EXCEEDED 5/8" = a cascade; find the
  FIRST dead task's error, the barrier is never the root cause.

## 2. Backward hangs (EP/FSDP2) — separate category

Training freezes mid-step: **driver log and trial-results mtimes freeze at the
same second**, job stays RUNNING, burns walltime. 2026-08-17: both LR-sweep arms
hung at gs5 (right at the first ckpt_interval=5 checkpoint boundary — suspicious
correlation, unproven), earlier incidents at s55/s61/s21. This is likely our
software stack (EP/FSDP2 + NCCL) interacting with something node-local; the
excluded nodesets here are weaker evidence of node fault than §1.

**Detection:** freshness watcher (no new trial-result json in 45–50 min AND log
mtime frozen); confirm over one extra ~13 min interval; then scancel + exclude
nodeset + chain rolls (resume loses ≤ckpt_interval steps).

## 3. The exclusion list — state and policy

- Lives in `hpc/hpc.py` jupiter `node_exclusion_list` (~line 1082) → baked into
  every RL sbatch as `--exclude`. As of 2026-08-17 it expands to **1,221 nodes
  (~20% of booster)**.
- It mixes: (a) transient ghost sets, (b) hang sets, (c) precautionary full-rack
  escalations (jpbo-030/026/076/081...). Since ghosts are transient, **most ghost
  entries are stale within ~a day** — the list chases yesterday's ghosts and
  starves scheduling. **Policy going forward: age out ghost entries after ~24h;
  keep hang entries; rack escalations only after 2+ distinct dirty zones.**
- SFT (Levanter) runs a separate MINIMAL list inside
  `hpc/sbatch_sft/levanter_sft_jupiter.sbatch` (same-day incidents only) — per
  Luke's "no SFT exclusions" directive, deliberately deviated only for
  confirmed same-day dirt.
- **Patching pending jobs without losing queue age:** `scontrol update JobId=N
  ExcNodeList=<full new list>` on every pending link (cancel+resubmit resets
  priority). Resubmitting a rendered RL sbatch requires the full env preamble
  (`cd repo && export DCFT=$PWD` + preset-SOCKS proxy vars) or it dies in 10s at
  the WORKDIR guard / with a dead proxy.

## 4. Tools

- **`scripts/jupiter/ghost_gpu_probe.sbatch`** — 1-node, ~15-second, dependency-
  free probe: nvidia-smi snapshot before touching GPUs (any usage = foreign) +
  orphan-PID /proc check + 80GiB cuMemAlloc per GPU via libcuda ctypes (clean
  GH200 → OK; ghost → CUDA_ERROR_OUT_OF_MEMORY).
- **`scripts/jupiter/ghost_probe_sweep.sh node1 node2 ...`** — one probe per
  currently-idle node. Results: `$F/experiments/ghost_probe/logs/`.
- **Live-capture protocol:** ghosts decay, so probe WITHIN MINUTES of an
  incident — the incident nodes have just freed, the probe backfills in seconds.
  A same-node, minutes-later probe output is the evidence package for JSC.
- Cost: probes bill ~15 node-SECONDS each (JSC accounts per-second × nodes;
  requested walltime is never billed, only elapsed). Short probes are free;
  frozen multi-node hangs are the real money sink.

## 5. The JSC support conversation (prepared, not yet sent)

Present: the incident table (§1), the transience finding, the probe script.
Frame as a **cleanup race**, not broken hardware. Asks:
1. Epilog/prolog GPU-memory gate: don't return a node to the pool until
   `nvidia-smi memory.used` < threshold; drain if it doesn't clear in time.
2. Attribution: they can see which jobs leaked (PrivateData hides it from us).
   Failed job IDs to trace: 1394803, 1392688, 1392690, 1392691, 1392692,
   1397228, 1396798.
3. Offer the probe for their verification; suggest THEY reproduce the leak on a
   drained node (we won't deliberately wedge shared nodes).

## 6. Open TODO

- **Prolog ghost-guard in `universal_rl.sbatch`**: per-node nvidia-smi check at
  job start → fail in ~30s instead of 15–50 min, optionally auto-append the bad
  node to the next pending link's ExcNodeList. Build after the current fleet is
  stable.
- Exclusion-list aging (see §3 policy) — needs a small tool or manual pruning.
