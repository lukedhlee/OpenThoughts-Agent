# ota10k Levanter SFT — Qwen3-30B-A3B-Base on OpenThoughts-Agent-SFT-10K

The production SFT run behind path-2 of the RL campaign (SFT ckpt → pass@8 probe →
GRPO). **This note is the source of truth for the run.** The session brief
(`session_brief_sft_levanter.md`) covers the babysitter role + live job IDs;
GRPO-fleet/node-health incidents live in `currease_pass8_probe_handoff.md`.

## What it is

- **Model/data**: Qwen3-30B-A3B-Base (MoE, 128 experts, 48 layers) on
  OpenThoughts-Agent-SFT-10K, paper recipe arxiv 2606.24855 — lr 4e-5 cosine
  (warmup 0.1), global batch 96, seq 32768, beta2 0.98, max_grad_norm 1e-4,
  328 steps, HF export at 328.
- **Stack**: raw `levanter.main.train_lm` — **PI directive**, not LLaMA-Factory
  and not marin's executor/harness (no iris, no marin credentials). This matters:
  most of the pain below comes from raw train_lm skipping setup marin's own
  harness does for you (see root cause 2).
- **Cluster**: JUPITER booster (JSC), 8 nodes × 4 GH200 (96GB), ssh `jupiter`
  (user lee27), `F=/e/fscratch/reformo/lee27`. Wall 11:59 per link.
- **Predecessor**: the LLaMA-Factory path (`sft/lf_configs/.../32k_base_30b_a3b_base_ota10k.yaml`)
  was validated end-to-end (shakedown 1393983 reached the step loop, ~26GB/GPU
  ZeRO-3) before the PI moved SFT to Levanter. **LF remains a working fallback.**

## Where things live

| Thing | Path |
|---|---|
| sbatch | `hpc/sbatch_sft/levanter_sft_jupiter.sbatch` (8 nodes, 11:59) |
| config | `sft/levanter_configs/ota10k_qwen3_30b_a3b_base.yaml` |
| logs | `$F/experiments/ota10k_lev_sft/logs/ota10k_lev_sft_<JOBID>.out` |
| checkpoints | `$F/checkpoints/ota10k_sft_30ba3b_levanter/ckpt/y3m02qj0/step-N` |
| HF export | `$F/checkpoints/ota10k_sft_30ba3b_levanter/hf` (written at step 328) |
| levanter/haliax source | `$F/repos/marin/lib/{levanter,haliax}` |

**Tokenizer gotcha**: the config points at an on-disk padded copy
`$F/data/qwen3_30b_a3b_base_tok_pad151936` (267 dummy tokens → 151936). Upstream
levanter bug: `pad_tokenizer_to_match_model` pads only the converter's tokenizer
(hf_checkpoints.py:633) while train_lm.py:245 sizes Vocab from the *data*
tokenizer → opt_state/model mismatch explodes on step 1. The padded copy
sidesteps it.

## The marin fork (load-bearing)

`$F/repos/marin` is NOT on upstream main. It tracks
**github.com/lukedhlee/marin branch `lukedhlee/qwen3moe-gpu-gmm`** = upstream
`ab07b1a` + 2 commits. Local clone: `/Users/lukedhlee/marin`.

- `f0c24253a` — thread `use_gmm=True` through `Qwen3MoeConfig`/`Qwen3MoeExperts`
  → `hnn.MoELinear` (root cause 1). Must stay a *default*, not a yaml value:
  `use_hf_model_config` replaces the yaml model config wholesale.
- `faa0bfcb9` — `defvjp(..., optimize_remat=True)` on haliax's triton ragged_dot
  custom_vjp. Was a no-op on prod (see below) but correct and kept.

Upstream marin PR for these: **candidate, NOT opened** — needs Luke's explicit
ask (marin-fork rule: a subagent never self-merges, and PRs need an ask).

## The memory-explosion saga (incidents 50–57) — READ THIS BEFORE TOUCHING SHARDING

21 launches died before the first loss line. Every failure surfaced as the *same*
`RESOURCE_EXHAUSTED` and looked like a tuning problem; it was actually **four
stacked causes**, each hiding the next. The `Can't reduce memory use below
87.29GiB … only reduced to X` remat warning is the fingerprint — read X first in
any failed log, it identifies which layer you're on:

| Attempt | Change | Post-remat demand |
|---|---|---|
| v8 1399676 | autotune 0, microbatch 3 | 21.15 TiB |
| v10 1399820 | microbatch 1 | 7.18 TiB |
| v12 1399959 | use_gmm triton MoE | 348.66 GiB |
| v14 1400086 | + optimize_remat | 348.66 GiB (byte-identical = no-op) |
| v18 1400728 | + token axis mapping | 212.60 GiB |
| v21 1401031 | + param sharding | fits — trains |

1. **XLA SPMD densifies the MoE ragged dot** (incident 54). levanter's
   `qwen3_moe` used global-view `jax.lax.ragged_dot_general`; XLA's partitioner/
   autodiff densifies its backward into `[token, experts, dim]` buffers — ~1.65TB
   per tensor class × 48 layers at 32k seq. Fix: `use_gmm=True` → haliax's
   shard_map + pallas-triton kernels with an explicit VJP, so XLA never
   autodiffs or partitions the ragged op.
   *This also retroactively explains the "XLA autotuner wants 3–4.5TiB" deaths
   (incidents 50/51) — those were real operand shapes, not an autotuner bug.*
2. **`token`/`token_repeat` axes unmapped → whole MoE block REPLICATED**
   (incident 56, the true root cause). `Qwen3MoeSparseMoeBlock` flattens to
   ad-hoc axes named `token`/`token_repeat`. Raw train_lm has no mapping for
   them, so `hax.auto_sharded` + the block's shard_map pspecs resolved to
   **replicated**: every device stored and computed the *entire global*
   microbatch MoE (`bf16[8388608,2048]` = 34GB, ×61 refs in the HLO dump).
   marin's own harness maps them (`lib/marin/src/marin/experiment/train.py:84`,
   `_TOKEN_AXES = (replica_dcn, replica, data)`); raw train_lm bypasses it.
   Fix = mirror it in `trainer.mesh.compute_mapping`.
3. **Params/Adam sharded only intra-node** (incident 57). levanter's MeshConfig
   treats each GH200 node as a slice: ICI `data` axis = 4 (intra-node), nodes on
   DCN `replica_dcn` = 8. The default `param_mapping {embed: data}` therefore
   FSDP-sharded fp32 masters + m + v **4-way and replicated them across the 8
   nodes** (`f32[48,128,512,768]`; 512 = 2048/4 is the tell) — ~87GB/device from
   optimizer state alone. Fix = `param_mapping: {embed: [replica_dcn, data]}`
   → 32-way FSDP over the same axes batch spans.
4. **Capacity** (incident 52): `per_device_parallelism: 1` (microbatch 1, 3-step
   grad accum). Global batch 96 and optimizer math unchanged — recipe intact.

Also in the sbatch and still required: `XLA_FLAGS=--xla_gpu_autotune_level=0`
(the 4→1→0 ladder; level 4/1 OOM'd during train_step compile),
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.92`, `JAX_COMPILATION_CACHE_DIR`, fused-CE
autotune sweep ON (OFF is worse — inferred CE block sizes OOM in 7 min),
`JAX_PJRT_CLIENT_CREATE_OPTIONS=allocator:cuda_async`, `ulimit -c 0`.

### The diagnostic method that cracked it (reuse this)

1. Relaunch with `XLA_FLAGS="--xla_dump_to=<dir> --xla_dump_hlo_pass_re=rematerialization"`
   **and `JAX_ENABLE_COMPILATION_CACHE=false`** — jax scrubs dump flags from the
   cache key, so a cache hit silently produces no dump (cost us one link).
2. Census the `*jit__train_step*after_rematerialization*` HLO (~90MB text) in
   python: regex every `dtype[dims]`, compute bytes, print everything >2GB.
3. Diff the census against the previous run's. The tensors that *didn't* shrink
   name the remaining replication layer. Dump is post-partitioning, so sharded
   tensors appear at local shapes — anything at global shape is replicated.

### Ruled out (don't re-litigate)

- `RAGGED_DOT_IMPL=xla` as a fallback: 1-GPU probe shows the XLA ragged-dot
  backward densifies to 48GiB `[262144,128,768]` buffers even at *local* shapes.
  The triton kernels are mandatory.
- shard_map opacity / custom_vjp residual pinning: a 1-GPU probe of checkpointed
  scan + shard_map + triton custom_vjp remats correctly (881MiB vs 738MiB dense
  control). That's why `optimize_remat` was a no-op.
- DenseMixer (`dense_router_gradient`) — defaults False, never involved.

## Operational runbook

**Launch / extend the chain** (self-contained; no proxy/DCFT needed for SFT):
```bash
ssh jupiter 'cd /e/fscratch/reformo/lee27/OpenThoughts-Agent && \
  export LEV_CONFIG=$PWD/sft/levanter_configs/ota10k_qwen3_30b_a3b_base.yaml && \
  h=$(sbatch --parsable hpc/sbatch_sft/levanter_sft_jupiter.sbatch) && echo head=$h && \
  prev=$h && for i in 1 2 3; do n=$(sbatch --parsable --dependency=afterany:$prev \
    hpc/sbatch_sft/levanter_sft_jupiter.sbatch); echo link$i=$n; prev=$n; done'
```

- **TIMEOUT at 11:59 is NORMAL.** Checkpointing is time-based every 15 min
  (`CheckpointerConfig.save_interval`, checkpoint.py:1149), so a timeout loses
  ≤15 min. Step-100 keeps are permanent.
- **VERIFY EVERY RESUME** (incident 58). A tqdm progress line proves nothing —
  the first one prints `-/328 elapsed:00:00` at startup. Grep the new link's log:
  - `No training checkpoint found` → **FATAL restart**, scancel the whole chain.
  - `Loading checkpoint from …/step-N` → good.
- **`trainer.id: y3m02qj0` must stay pinned.** Checkpoints save to *and* are
  searched under `base_path/<run_id>`; `trainer.id` defaults to a random string
  per launch, so offline-wandb links each minted a new id, found an empty dir and
  silently restarted from the HF weights (discarded a full 12h link). The pinned
  value is where step-80 lives.
- **Config edits reach a job at job start**, so a dependency-released spare that
  is already RUNNING carries the OLD config/code — scancel doomed spares
  explicitly rather than assuming they inherit a fix.
- Fixes flow local Mac → `git push fork lukedhlee/vista-moe-grpo-30b` →
  `git pull` on the cluster. Never hand-edit on the cluster. Luke keeps unstaged
  local changes (.gitignore, ssh notes) — commit only your own files.
- Ghost-OOM at job start (foreign GPU memory): add only the confirmed-dirty
  nodeset to the sbatch's targeted `--exclude` (SFT deliberately does NOT carry
  the full RL blacklist) — see `jupiter_node_health.md`.

## Misdiagnoses (cost real hours — do not repeat)

- `RESOURCE_EXHAUSTED` with allocator line `Limit: 87.40GiB`/`71.25GiB` = **our
  own mem-fraction cap**, not ghost-node residue.
- 9.6GB `cuMemAllocAsync failed` bursts early in the log = the **fused-CE autotune
  sweep probing candidates**. Benign; jobs run for many more minutes after them.
- 8-rank SIGABRT + "Shutdown barrier DEADLINE_EXCEEDED" = cascade. Find the FIRST
  dead task's error; never blame the barrier.
- wandb `Disk quota exceeded` staging-thread spam = non-fatal noise.
- `transformer_engine is not installed … falling back` = fine; levanter falls back
  to its own blockwise flash attention (O(N) memory), not to vanilla attention.

## Open: throughput

~500 s/step steady state = ~6.3k tok/s across 32 GH200s for a 3B-active MoE —
**low, likely several× off**. Not blocking; the run finishes in ~4 links. Do NOT
disturb a green run without Luke's ask. Candidate levers, cheapest first:
retry `xla_gpu_autotune_level=1` now that the HLO is sane; check pallas-triton
block sizes for GH200 (`_triton_default_block_sizes`, haliax `nn/ragged_dot.py`
— the CE sweep picks `b1024/h512/v64` and warns the 512×1024 weight tile exceeds
the 101376-byte shared-memory budget).

## Endgame

1. Step 328 → levanter writes the HF export (`hf_save_steps: 328`).
2. Verify the export loads (config + tokenizer 151936).
3. Report loss curve + export path to Luke. HF upload defaults PUBLIC to
   `laion/` per standing guardrail — but **ASK Luke first**: SFT-checkpoint
   naming/visibility was never decided.
4. The follow-on (pass@8 probe of the SFT ckpt → curriculum work) belongs to the
   supervisor session. Hand off, don't start it.

## Incident index

⚠️ **Numbers COLLIDE with the GRPO ledger.** Both workstreams appended to
`currease_pass8_probe_handoff.md` from the same counter and drifted apart, so
e.g. "Incident 56" is the token-axis root cause *here* and a lr5e6 ghost-OOM
*there*. Always qualify as **"SFT incident N"**. Stubs remain in the GRPO ledger
at the original chronological positions, pointing here.

SFT-owned: **41, 48** (fused-CE autotune ON/OFF reversal), **50, 51** (XLA GEMM
autotuner ladder 4→1→0), **52** (first successful train_step compile; microbatch),
**54** (use_gmm), **55** (optimize_remat; no-op), **56** (token axis mapping —
true root cause), **57** (param sharding), **58** (pinned run id / silent restart).
Future SFT incidents: number them **in this note** as SFT-59, SFT-60, … and stop
appending to the GRPO ledger.

## SFT-59 (2026-08-19 ~00:20 CEST) — 🔴 RESUME CORRUPTS THE MODEL. Run is NOT valid.

**Do not trust any checkpoint in the current lineage.** Discovered while the run
looked healthy: the resumed link's loss is **24**, not 0.42.

- Link 1 (v21 1401031, from HF weights): loss 0.50 → 0.42 over steps 0–80,
  monotonic. Saved temp `step-80` at 16:48:00; SIGTERM (wall limit) 16:50:43.
- Link 2 (1406651) restored `step-80` and its first loss was **24**, then
  recovered monotonically 24 → 13.4 → 9.3 → 7.9 → 6.0 → 4.28 by step 126.
- **24 is WORSE than random** (ln 151936 ≈ 11.9) → the restored weights are
  corrupted, not merely re-initialized. Recovery-from-garbage, not divergence.
- Not a config/code drift: marin pinned at `faa0bfc` (clean tree), the only
  yaml delta since step-80 is `trainer.id`. Load path logs no shape mismatch or
  partial restore (the `Unknown leaf type NoneType` spam is benign — optional
  bias leaves).
- **`step-80` is GONE**: it was a *temporary* (15-min time-based) checkpoint and
  link 2's checkpointer deleted it after writing step-100. Everything on disk
  (step-100, step-124) descends from the corrupted restore → **no clean
  checkpoint exists; a restart from step 0 is required regardless of cause.**
- Leading hypothesis: the step-80 temp checkpoint was **torn** — written 2m43s
  before the kill, tensorstore's async flush possibly incomplete on some shards
  despite the "Saved checkpoint"/commit-callback log lines. Alternative:
  restore is systematically lossy (e.g. optimizer state), which would corrupt
  the model at *every* 12h boundary and makes levanter checkpointing unusable
  here as configured.
- **Decisive test RUNNING**: 1406651 cancelled (its model is useless anyway) so
  1406652 restores `step-124` — a checkpoint written mid-run, far from any
  signal. First loss ≈4.3 ⇒ restore is sound and step-80 was torn. First loss
  ≫4.3 ⇒ restore is systematically broken.

### Consequences for the restart (do NOT relaunch before deciding these)

1. Guard the wall-clock boundary: `#SBATCH --signal=B:USR1@<sec>` + a graceful
   stop, or stop scheduling saves near the wall, so no checkpoint is written
   into a kill window.
2. Keep a **permanent** cadence dense enough to survive (`keep: [{every: 25}]`),
   so a temp-checkpoint deletion can never orphan the last good state.
3. Validate a restore ONCE, cheaply, before committing 46h: resume and confirm
   the first loss matches the pre-save loss. **A resumed link that merely runs
   is not evidence it resumed correctly** — this failure was invisible for 7h
   because the job was healthy, memory was fine, and the loss was *falling*.
4. Also re-examine `max_grad_norm: 1e-4` (paper/repo-truth value): with lr 4e-5
   this clips updates to near-nothing, and link 1's loss barely moved
   (0.50→0.42 in 80 steps). Worth confirming it is intended before spending
   another 46h.
