# Takeover — 2026-08-05 (late) — supersedes all earlier notes

Report times in **KST**. Read `ai_memory/handoff.md` bottom-up (newest last), then `gotchas.md`.

## Mission
Agentic RL (GRPO) on `Qwen/Qwen3.6-35B-A3B` over r2egym. Jupiter = training + vLLM;
JURECA = Apptainer sandboxes reached via reverse SSH forwards from a Jupiter login node.

## ✅ What changed today — the pipeline actually works now
**For the entire project until today, no rollout had ever produced a multi-step trajectory.** The cause was
never the model or the reward: **two reverse tunnels were dead** and every trial timed out. With both
restored:

| signal | before | now |
|---|---|---|
| trajectory steps | `median=1, max=1` | **`median=18.5, max=26`** |
| bridge timeouts | 32/32 trials | **0 of last 12** |
| reward spread | uniform (0 of 46 groups varied) | **8×0.0, 4×1.0 → 33% pass** |

**33% ≈ Marianna's ~35%.** Different model (that run is the 35B), but it means **a learnable band EXISTS on
this task set**. Variance was never absent — the agent was never running. That retires the biggest risk.

**Also banked:** the MILESTONE machinery. `1243377` COMPLETED (exit 0:0) with a real optimizer update,
`global_step_1` checkpoint (**235 GB**) and HF-format export (**65 GB**) — preserved as
`.../MILESTONE_1243377_{checkpoints,exports}`. Its gradient was **zero** (rollouts were still timing out),
so it proves the machinery, not learning.

## 🎯 THE GOAL NOW (decided by Luke)
**Do NOT attempt the full band.** Instead:
1. Get the learnable band **reliably on a SUBSET** of tasks,
2. **on the 8B `g1_diverse_tezos_100k_8b`** (the model Marianna used),
3. then **extrapolate** to answer: *is our system scalable to a full band in 7–10h?*

Rationale: scalable rollout infra pays off across every future experiment and debugs ~10× faster
(~9 min/trial vs ~1.5h/training-step), and staying inference-only removes all training-side concerns.

## ⚠️ The 8B needs the tool-call parser wired FIRST — non-negotiable
The 8B emits tool calls as **bare JSON** (`{"name":"bash","arguments":{…}}`), 182/182 steps, zero XML.
vLLM ran `--tool-call-parser qwen3_coder` (XML-only) → never matched, **never logged** → OpenCode saw text
→ 1 step → reward = f(task) → zero variance. Measuring a band on this = re-deriving the retracted `0/197`.

Fix is written, validated, pushed, and **NOT yet synced to the cluster**:
- `OpenThoughts-Agent 4add607c` — `rl/tool_parsers/bare_json_tool_parser.py`.
  Validated offline against **all 182 real captured outputs: 182 parsed, 0 missed**, +8 edge cases,
  +streaming replay. (Four stock parsers each miss by one detail — see the module docstring.)
- `MarinSkyRL d8bdc79` — `pop_openai_kwargs` now forwards `tool_parser_plugin`. A plugin must be
  **imported** (so `@ToolParserManager.register_module` runs), not passed as a kwarg.

To wire it:
```bash
cd /e/scratch/reformo/lee27/MarinSkyRL-apptainer-bridge && git fetch fork lukedhlee/apptainer-bridge-rl \
  && git reset --hard fork/lukedhlee/apptainer-bridge-rl     # ONLY when no job is running
# then in the RL yaml / overrides:
#   engine_init_kwargs.tool_parser_plugin: <repo>/rl/tool_parsers/bare_json_tool_parser.py
#   engine_init_kwargs.tool_call_parser: bare_json
```
⚠ **Never `git reset --hard` a clone while a job runs** — Ray can spawn workers against changed code.

## Next actions, in order
1. ~~Harvest `1244916`~~ — **it died of node failure; nothing to harvest. The fleet is already free.**
   Re-add the vLLM forward at whatever head your next job gets.
2. **Sync the cluster** (above) — the queue is empty, so this is safe now.
3. **Launch an 8B SUBSET band shard** with `bare_json`.
4. **Gate before scaling** — `median_steps > 1` AND ≥1 group with non-zero within-group variance.
   A passing trial count is NOT a gate; see the ladder below.
5. **Ramp concurrency** 32 → ~250 and find the next ceiling (likely vLLM).
6. **Extrapolate** measured throughput → full-band hours. That is the deliverable.

## Throughput math (measured, not assumed)
Full band = 3,328 tasks × p@4 = **13,312 trials**; 10h ⇒ **22 completions/min**.

| measured | value |
|---|---|
| real agent trial | **median 8.9 min**, p90 13.3 |
| trial that TIMES OUT | ~62 min (p90) — a failure costs **7× a success** in slot-time |
| peak achieved concurrency | **32** = exactly the config cap |
| sandboxes sitting READY while 32 run | **110** |

⇒ Need **~200 concurrent** (22.2 × 8.9), ~295 sizing on p90. Fleet capacity is **512 slots**, so
band-in-10h needs a **~6× concurrency bump, not more nodes**. Concurrency is provably the binding
constraint; nothing else is saturated. (An earlier "666 concurrent" estimate was wrong — it used the 1800s
agent *budget* as the trial duration.)

⚠ **Exclude the free-pass tasks:** 10 of 46 groups scored **1.0 while doing nothing** — those r2egym
verifiers pass on an unmodified repo. They put a ~22% floor under any band number and can never yield
gradient. Cheapest throughput win available.

## 🔧 TUNNEL RUNBOOK — the ControlMaster carries **TWO** forwards
Restoring only one gives a **PASSING route gate and 100% rollout timeouts**. This cost hours today.

1. **`-4` is MANDATORY.** `jureca.fz-juelich.de` resolves IPv6-first and the key's JuDoor `from=` clause
   rejects IPv6 → `Permission denied (publickey)` with TOTP never offered. With `-4`:
   `Authenticated ... partial success` → *then* JSC prompts for the TOTP.
   (The old note blaming "hostname resolution" is **WRONG**.)
2. **Pin the login node** to `jureca05.fz-juelich.de` (single A record 134.94.1.132), never the
   12-address round-robin alias. The tunnel must bind `10.14.0.46`, owned only by **jrlogin05**; land on
   another node and the forward fails with `NO_LISTENER` and nothing holding the port — looks like a burned
   port, is not.
3. **Add BOTH forwards:**
```bash
S=~/.ssh/cm_jureca/qwen36; H=jureca05.fz-juelich.de
ssh -S $S -O forward -R 10.14.0.46:18000:<jupiter-head-ip>:8000 $H   # sandboxes -> vLLM
ssh -S $S -O forward -R 10.14.0.46:9923:10.128.1.2:9920         $H   # workers   -> bridge
```
   The bridge is a **python3 process on the Jupiter login node** at `10.128.1.2:9920`; workers reach it as
   `jrlogin05i:9923`. That port pair is recorded ONLY in the fleet log
   (`/p/scratch/synthlaion/lee27/dc_agent_eval/logs/apptainer_workers_<fleet>.out` → "Bridge URL").
   `BRIDGE_LOGIN` defaults to `jrlogin03i:9920` in the sbatch and **is overridden** — read the log.
4. **`workers_alive: false` + live worker processes = a missing bridge forward**, not dead workers.
   `pgrep -fc worker.py` on a fleet node distinguishes them. Recovery is instant (`active_jobs` jumps in
   seconds).
5. **Then run the COMPUTE-NODE route gate** — a real `/v1/chat/completions` from a fleet node. A listener
   check (`ss | grep 18000`) passes while the route is dead; that is exactly how `1243377` lost 64 rollouts.

Full master restart (needs ONE interactive TOTP):
```bash
ssh -t jupiter 'S=~/.ssh/cm_jureca/qwen36; ssh -S $S -O exit jureca.fz-juelich.de 2>/dev/null; \
  ssh -4 -i ~/.ssh/id_ed25519_jupiter2jureca -M -S $S -fN lee27@jureca05.fz-juelich.de && \
  ssh -S $S jureca05.fz-juelich.de hostname'      # MUST print jrlogin05.jureca
```
⚠ **Capture a dying master's forwards BEFORE killing it:** `ps -u $USER -o args | grep 'ssh .*-R'`.

## Live state at handoff (re-probe, do not trust)
- **Jupiter `1244916` is DEAD — I `scancel`led it deliberately at ~17:51 KST.** `sacct` says
  `CANCELLED by 34902`, exit `0:0`. **NOT a node failure** — the outgoing session wrote that up at 17:52 from
  the `NodeDiedError` that Slurm teardown always produces, one minute after the cancel; that claim is
  RETRACTED (see handoff § 2026-08-05 late). **There is no known hardware failure mode on Jupiter; do not
  shorten walls for one.**
  Why cancelled rather than harvested: its generation buffer was stuck at **6/16 groups after 55 min** and
  wave 1 returned `0/4 successful × 8 batches` with `TIS mode: ALL 4 trajectories missing logprobs → this
  batch cannot be used for TIS training` (prompts constantly overflowing 32768 at `28673 + 4096`). Best case
  was ~40–80 min to the step **plus** a ~1.5h update — 2.5–3.5h of a 7h fleet window for a gradient TIS
  would likely reject. It produced **no `grad_norm` and no checkpoint**, so the first non-zero gradient is
  still UNMEASURED.
  ⇒ **The JURECA fleet was thereby freed for the 8B subset work, which is the actual deliverable.**
- **Tunnels left in a CLEAN state — you only need to add ONE forward:**
  - `10.14.0.46:18000` (vLLM) — **cancelled and released**; port is free, NOT burned. Add it at your new
    job's head IP once allocated.
  - `10.14.0.46:9923` (workers → bridge) — **still UP and verified**. It is Jupiter-side, so it survives job
    deaths. Do not re-add it; just confirm the listener exists.
- **Bridge** `10.128.1.2:9920`: `workers_alive: true`, `queue_size: 0` — healthy, draining leftover envs.
- **JURECA fleet `15498197`**: 32 nodes, **~7h left**. Enough for a subset band; NOT enough for a full
  13,312-trial band — submit a fresh 24h fleet before attempting that.
- **Jupiter queue: EMPTY.** Cluster clone sync is safe right now.
- **JURECA fleet `15498197`** RUNNING, 32 nodes, **~7h20m left** — the scarce resource; the 8B band needs it.
- **Bridge** `10.128.1.2:9920`: `workers_alive: true`, `queue_size: 0`, `active_jobs: 32`, `envs.ready: 32`.
- **Both forwards up** on the master pinned to jrlogin05 (pid was 4083409).
- Helper scripts now persisted: `/e/fscratch/reformo/lee27/gate.py` (gate ladder) and
  `.../throughput.py` (trial-duration + achieved-concurrency measurement).

## Establish reality first
```bash
ssh jupiter "squeue --me -o '%.10i %.9T %.6M %.6L %.4D %N'"
ssh jupiter "ssh -S ~/.ssh/cm_jureca/qwen36 jureca05.fz-juelich.de \"squeue --me -o '%.10i %.9T %.12L %.4D'\""
ssh jupiter "curl -s -m 8 http://10.128.1.2:9920/status"      # workers_alive + active_jobs + envs.ready
ssh jupiter "python3 /e/fscratch/reformo/lee27/gate.py"                            # median_steps + rewards, last 16 min
```

## The gate ladder — in this order, every time
1. non-null reward → 2. **trajectory with >1 step** → 3. trial duration in *minutes* → 4. only then believe
a band number. **`scored=N` counts FILES, not rewards.** Four separate bugs produced null rewards while
every progress counter looked healthy. I also nearly reported a false `grad_norm=1.0` — my regex had
matched **`max_grad_norm=1.0`** from the config echo. Anchor patterns; verify by behaviour.

## Launch recipe that works (three attempts died on these)
```bash
W=/e/fscratch/reformo/lee27/OpenThoughts-Agent-r2egym-bridge-next     # the REAL WORKDIR
ssh jupiter "set -a; source /e/scratch/reformo/lee27/keys/secrets.env; set +a;
  cd $W && source hpc/dotenv/jupiter.env >/dev/null 2>&1; export DCFT=$W;
  sbatch --chdir=$W --time=06:00:00 --export=ALL,DCFT=$W,SKYRL_CHUNKED_LOGPROBS=1 <generated>_rl.sbatch"
```
- `WORKDIR` is **`/e/fscratch/.../OpenThoughts-Agent-r2egym-bridge-next`**, NOT
  `/e/scratch/.../OpenThoughts-Agent` (which lacks `hpc/shell_utils/flashinfer_aot_cache.sh`).
  Wrong dir ⇒ instant FAIL (`1243248`, `1243289`).
- Secrets live at **`$SCRATCH/keys/secrets.env`**, not `~/secrets.env` as ops.md claims. Missing it ⇒
  `AssertionError: WANDB_API_KEY is required` after full Ray startup (`1243351`).
- `--time=` / `--export=ALL,VAR=v` override the generated sbatch without editing it.
- **Never run Jupiter ssh calls in parallel** — the ControlMaster refuses sessions and you get a spurious
  TOTP prompt (this is the channel exhaustion that broke the watcher on `1229649`). Serialize.

## MILESTONE OOM — fixed, keep it enabled
`MarinSkyRL 637a764` (on the cluster clone). `logprobs_from_logits_v2` bounds memory by looping the
**BATCH** dim — a no-op at `micro_train_batch_size_per_gpu=1` — so it ran one autocast-promoted fp32
`log_softmax` over the whole `[S, V]`. Real vocab is **248320** (nested in `config.json → text_config`;
the old note's 151936 was the 8B's) ⇒ `30.57 GiB / 4 B / 248320 = S≈33046 ≈ max_seq_len`, matching exactly.
Sequence-chunking alone does **not** help (log_softmax saves its output); the fix recomputes softmax per
chunk in backward, mirroring `_EntropyFromLogits`.
GPU-validated: chunked **33.41 GiB** vs stock **OOM at the identical "30.57 GiB"**; training-path parity
9.5e-07; and vs fp64 the chunked path is **2.4× MORE accurate** than the stock one.
**Enable with `SKYRL_CHUNKED_LOGPROBS=1`**; the call site logs `chunked gathered log-softmax ACTIVE` once —
verify by that line. It only fires during a TRAINING step (~after generation), so absence early is normal.

## Known-open
- **HF Hub push never verified.** `HF_HUB_CACHE=/e/data1/datasets/playground/ot-baf/hf_hub` is **feuer1's**
  dir (`Permission denied` for lee27) and the sbatch derives `HF_HOME` from it. No `HF_TOKEN` in env; the
  valid token (user `lukeleeai`) is at `$HOME/.ssh/../.cache/huggingface/token`. The 65 GB export is
  local-only — push manually from the login node per `rl-agentic-job-cleanup`, and set a lee27-writable
  `HF_HUB_CACHE` for future runs.
- `MarinSkyRL` (the non-bridge clone) has **5 files of uncommitted Megatron-checkpointing work** — on no
  cluster. Decide whether to commit or discard.
- `retarget_job.sh`'s `OLD_TARGETS` is stale and its route gate uses the hostname; prefer the runbook above.

## Standing constraints
Jupiter ≤16 nodes / JURECA ≤32 · never `scancel` another user's job (our own wedged jobs are pre-authorised)
· never `find`/`du` on GPFS or JURECA scratch · bind listeners to internal interfaces, never `0.0.0.0` ·
`enable_db_registration: false` · local clones are ground truth: edit locally → push to `fork` → `git fetch`
+ hard reset on the cluster, never hand-edit · no `Co-Authored-By` in commits · destructive cleanup is
operator-only.

## Do not re-derive (retractions)
- **`0/197` band figure: RETRACTED** (one-step-trajectory artifact).
- **"151k vocab × 28,672 tokens ≈ 8.6 GiB"**: RETRACTED — wrong model's vocab.
- **"bare `ssh jureca` fails on hostname resolution"**: WRONG — it is IPv6 vs the key's `from=` clause; `-4`.
- **"restoring the tunnel needs one TOTP"**: incomplete — it needs `-4`, jrlogin05 pinning, **and two
  forwards**. The TOTP was never the hard part.
- Six accepted-but-ignored config keys so far — verify any key by **behaviour**, never by presence in the
  materialized config.
