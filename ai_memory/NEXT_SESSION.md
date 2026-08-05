# Takeover — 2026-08-05 (supersedes the 04 Aug 10:30 KST note)

Report times in **KST**. Read `ai_memory/handoff.md` bottom-up (newest sections first), then `gotchas.md`.

## Mission
Agentic RL (GRPO) on `Qwen/Qwen3.6-35B-A3B` over r2egym. Jupiter = training + vLLM; JURECA = Apptainer
sandboxes reached via a reverse SSH forward from a Jupiter login node.

**Two goals, in order:**
1. **The milestone** — one finite optimizer update + checkpoint + HF export. Still never executed.
2. **The learnable band** — p@4 over 3,328 tasks on `g1_diverse_tezos_100k_8b` (the co-lead's own 8B).
   Every group we have ever scored had **zero within-group reward variance** ⇒ advantage 0 ⇒ no gradient.
   The band is a gradient prerequisite, not a compute saving.

## Establish reality first — do not trust the numbers below
```bash
ssh jupiter "squeue --me -o '%.10i %.9T %.6M %.6L %.4D %N'"
ssh jupiter "ssh -S ~/.ssh/cm_jureca/qwen36 jureca.fz-juelich.de 'squeue --me'"   # bare `ssh jureca` FAILS
ssh jupiter "curl -s -m 8 http://10.128.1.2:9921/status"                          # probe bridge
ssh jupiter "python3 /e/fscratch/reformo/lee27/trajcheck.py"                      # THE GATE
ssh jupiter "python3 /e/fscratch/reformo/lee27/rewardcheck.py"
```

## Live: job 1242066 — OpenCode 1.18.8 + thinking OFF, port 18180
2 nodes, 3 h wall. JURECA fleet **15498197** (32 x dc-cpu, 16 workers/node, tmpfs staging) has ~20 h — reuse it.

**Decide on the gate, nothing else:**
- **`median_steps > 1`** → it works. Launch shards 1-7: set the port base in
  `/e/fscratch/reformo/lee27/launch_band_shards.sh` to **18190**, then
  `bash launch_band_shards.sh 1 7 04:00:00` and `setsid nohup bash fwd_all.sh &`.
  First wait for the bridge's `stopping` count to reach ~0, or the new shards queue behind the drain.
- **`median_steps == 1`** → thinking was not the cause. Read `agent/opencode.txt` for what the model
  returned: a terminus-trained checkpoint may never emit OpenCode's Qwen3 XML tool calls, which would make
  terminus-2's request timeout the only path and the thing to fix.

## Four rules that each cost hours
1. **`scored=N` counts FILES, not rewards.** Gate ladder in order: non-null reward → **trajectory with
   >1 step** → trial duration in *minutes* → only then believe a band number. Four separate bugs produced
   null rewards while every progress counter looked healthy.
2. **Address the rollout endpoint by IP `10.14.0.46`, never `jrlogin05i`.** Inside the sandbox Python and
   curl resolve that hostname differently, and it varies by node.
3. **Sandbox staging must be node-local** (`STAGING_BASE=/tmp/apptainer_staging`): `apptainer overlay
   create` is 18.06 s on `/p/scratch` vs 3.57 s on tmpfs, against a **hardcoded 60 s** timeout.
4. **JURECA ports are single-use** — a forward left by a dead job cannot be reclaimed. Burned 18100-18180.
   **Never `pkill -f` over SSH** — it kills your own session mid-chain; use `tmux kill-session`.

## The milestone's remaining blocker is arithmetic, not infrastructure
`1229649` finished generation (16/16 groups, `avg_pass_at_4 0.375`), ran forward logprobs and advantages,
then died in **backward**: `torch.OutOfMemoryError: Tried to allocate 30.57 GiB` (95 GiB GPU, 3.97 free).
Shape is large-vocab logits: ~151k vocab x 28,672 tokens ~= 8.6 GiB/copy bf16, several live at once, and
gradient checkpointing does not touch the final projection. Try chunked/fused cross-entropy, a shorter
training-time sequence cap, or sequence/tensor parallelism.
⚠ `Finished: 'policy_train'` is **NOT** success — the timer logs it from `__exit__` as the exception propagates.

## Ask the co-lead (Marianna) — cheap, and both change what we run
1. Her **band task-ID list** (V1 indices or docker_images). Our 3,328 is a 73% subset of her 4,578 keyed by
   the same index, so it intersects directly (~1,160 expected) — one message vs ~10 h of generation.
2. **Which agent, and was thinking on?** The band belongs to the (model, agent, config) triple, so her
   ~35% is only comparable if those match.

## Standing constraints
Jupiter <=16 nodes / JURECA <=32 for this work · never `scancel` another user's job · never `find`/`du` on
GPFS or JURECA scratch · bind listeners to internal interfaces, never `0.0.0.0` · `enable_db_registration:
false` · local clones are ground truth: edit locally → push to `fork` → `git fetch` + hard reset on the
cluster, never hand-edit · no `Co-Authored-By` in commits.

## Do not re-derive
The **`0/197` band figure is RETRACTED** — an artifact of one-step trajectories. It is left in the handoff
deliberately so nobody trusts it from git history. Six accepted-but-ignored config keys so far (newest:
harbor `timeout`) — verify any key by **behaviour**, never by its presence in the materialized config.
