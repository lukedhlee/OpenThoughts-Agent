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

## 🎯 THE GOAL NOW — REFRAMED by Luke, 2026-08-05 evening (supersedes the version below)
**Build scalable, trustworthy, functioning infra by REPRODUCING Marianna's learnable-band work.**
Success = **matching her number**, not beating a clock. A number to hit is a far stronger correctness test
than a throughput estimate, and "get a band within 10h" invites a *false* band.


> ⚠ **RETRACTED 2026-08-05 23:5x KST — "358 of 4,578 ≈ 8%" IS WRONG.** Marianna, asked directly, said the
> band is **~1.6k of 4.5k ≈ 36%**, and that her filtering pass cost `18k rollouts` (= 4.5k x 4, confirming
> pass@4 over a 4.5k pool). Where 358 came from is unknown; treat every "8%" / "358" below as void. This
> matters because it flips a verdict: our measured 0-in-band was "consistent with 8%" but is
> ~1-in-800,000 against 36%. Her run also used **terminus-structured**, not OpenCode.

**Her result, from the script she shared:** `358 learnable tasks` out of the **4,578**-task r2egym pool,
band defined as **`0 < pass@4 < 1`** ⇒ **≈8%**. (Her script says `n_samples_per_prompt=8`, but that is her
TRAINING config — Luke confirmed the band itself was built at **p@4**.)
⚠ So the long-quoted **"~35%" is NOT a band rate** — it must have been a pass rate / `resolved@1`. Our own
0-of-16 groups on the 35B is *consistent* with an ~8% band, not a contradiction of it.

**Her config, and where ours differed** (all now knobs in `gen_band_yaml.py`, `b5dba99b` + `758fce66`):

| | hers | ours (before) |
|---|---|---|
| `max_episodes` | **50** | **UNBOUNDED** (`canary_ctx max_turns=999999`) |
| `override_cpus / memory_mb / storage_mb` | 1 / 1024 / 1024 | 2 / 4096 / 4096 |
| `max_model_len` | **40960** (g1's native window) | 32768 |
| tensor-parallel × engines | **4 × 8** | 1 × 4 |
| `max_num_seqs`, GMU | 1024, 0.92 | 64, 0.85 |
| thinking | **ON** (`interleaved_thinking=true`) | tried to force OFF |
| agent | `terminus-structured`, `use_fn_calling=False` | OpenCode (needs a tool parser) |

**The episode cap is the throughput story, not thinking.** With `max_turns` unbounded a trial can only end by
hitting the 1800s wall — measured: **63 of 64 trials still running at 38 min for 5 completions**
(0.17 trials/min). I spent hours treating thinking as the bottleneck; it was a contributor, not the cause.

⚠ **Her `use_fn_calling=False` means her agent parses actions from RAW TEXT and uses no vLLM tool parser at
all.** OpenCode requires OpenAI tool-calling, so that part of the pipeline can never be identical to hers.

**Blocked on Marianna / Luke:** her paths are `Permission denied` for `lee27` —
`/e/project1/jureap59/marianna/...` and `/e/data1/datasets/playground/ot/hf_hub/...`. Worth asking for:
1. the **358-task learnable set** (`r2egym_learnable_heldout`, `merge_split_learnable.py`) — reusing it skips
   the 13,312-trial sweep entirely,
2. her **band-generation** script (she sent the training one),
3. `qwen3_thinking_acc.jinja2`.

### Old goal, kept for context (superseded)
1. band on a SUBSET, 2. on the 8B, 3. extrapolate to a 7–10h full band.

## ✅ Tool-call parsing is WIRED and VERIFIED IN PRODUCTION (was the top blocker)
Both clones are synced; `bare_json` is live and confirmed **by behaviour**, not by config echo:
- offline on the cluster's real vLLM **0.22.0**: `get_tool_parser('bare_json')` resolves
  `BareJsonToolParser` and parses all 5 shapes, 0 calls on plain text (`/e/fscratch/.../parsergate.py`)
- live endpoint: `finish_reason: tool_calls`, one parsed `bash` call, 21 completion tokens
- live rollout: `r2egym-v1-00300` made **2 clean tool calls** in a real trajectory
- through the tunnel from JURECA compute nodes `jrc0554/0555/0556`

⚠ **`qwen3_coder` was the wrong DIALECT, not merely the wrong parser.** g1's own
`chat_template.jinja` (4 KB — note `tokenizer_config.json`'s `chat_template` is EMPTY, the template lives in
the separate `.jinja` file) instructs:
`<tool_call>\n{"name": ..., "arguments": ...}\n</tool_call>` — that is the **hermes** dialect.
`qwen3_coder` expects `<function=name><parameter=x>`, a different grammar, so it never matched and never logged.

**Measured `hermes` vs `bare_json` on the real shapes** (`/e/fscratch/.../hermestest.py`):

| shape | `hermes` | `bare_json` |
|---|---|---|
| `<tool_call>` XML (template-canonical) | ✅ | ✅ |
| XML after `</think>` | ✅ | ✅ |
| bare JSON, no XML (the 182 captures) | ❌ | ✅ |

⇒ **`bare_json` is a strict SUPERSET of `hermes`**, so keeping it is right: a *missed* tool call ends the
episode, while a spurious one only wastes a step. Asymmetric — prefer over-parsing. (Its one observed cost: a
1024-token runaway yielded 29 extracted calls.) `hermes` is the correct fallback if the custom plugin is ever
unavailable — do NOT go back to `qwen3_coder`.

### The parser was never the whole bug — thinking was the other half
`enable_thinking` IS supported, at `chat_template.jinja:86`:
`{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\n\n</think>\n\n' }}`
i.e. it prefills an EMPTY think block. But **harbor's `extra_body` never reaches OpenCode** (implemented for
`terminus_2` / `openhands` / `mini_swe_agent` only), so the harbor-level keys are inert for it — the 7th
accepted-but-ignored key here. `MarinSkyRL 9904058` adds the server-side lever
(`generator.engine_init_kwargs.default_chat_template_kwargs`), which is the only layer that reaches an
external agent. **It is committed but NOT NEEDED for band parity — Marianna runs thinking ON.**

## Next actions, in order
1. **Read `1247578`** (128 tasks × p@4 = 512 trials, 8 Jupiter nodes, Marianna-parity config) with
   `python3 /e/fscratch/reformo/lee27/band.py <trace_jobs>`. It reports three things at once:
   **band rate** (compare to her ~8%), **zero-tool-call %** (is OpenCode's formatting right?), and
   **per-trial duration** (the only honest input to an extrapolation).
2. **Gate on the LADDER, in order** — non-null reward → >1 step → duration in minutes → only then a band
   number. `band.py` walks it for you and flags suspected free-pass groups separately.
3. If the band rate is near ~8%: the infra is trustworthy ⇒ scale to the full 4,578-task pool
   (needs a fresh 24h JURECA fleet). If it is far off: fix the DIFFERENCE from her config, don't tune blindly.
4. **Ask Marianna for the 358-task set** — reusing it makes the whole 13,312-trial sweep unnecessary.

## Throughput math — MEASURED, and the old conclusion was WRONG
Full pool = **4,578** tasks × p@4 = **18,312 trials** (the old note said 3,328/13,312 — that is our
`r2egym-patched-full-oracle` subset, a 73% slice of her pool).

| measured | value |
|---|---|
| real agent trial (35B, thinking on) | median **9.7 min**, p90 15.1 |
| trial that TIMES OUT | ~62 min p90 — a failure costs **~7× a success** in slot-time |
| **8B @ conc 64, thinking on, UNBOUNDED episodes** | **0.17 trials/min — 63 of 64 still running at 38 min** |
| peak achieved concurrency | 32 (config cap), then **64**, then **128** — all reached immediately |
| fleet capacity | 32 nodes × 16 workers = **512 slots** |

⚠ **RETRACTED: "band-in-10h needs a ~6× concurrency bump, not more nodes."** Concurrency was never the
binding constraint — we hit the configured cap instantly every time (32, then 64, then 128 sandboxes live).
The wall was **per-trial latency**, from two things:
1. **`max_turns: 999999`** ⇒ a trial can only end by hitting the 1800s wall. Marianna caps `max_episodes=50`.
2. **heavy sandboxes** (2 CPU / 4 GB vs her 1 CPU / 1 GB) ⇒ far fewer fit per fleet node.

Node arithmetic, for sizing: at 20k context a TP=1 engine holds ~21 sequences, so 4 engines ≈ 84 concurrent —
but ~25 engines saturate the 512 fleet slots, i.e. **~8 Jupiter nodes, not 16**. Beyond that more Jupiter buys
nothing and JURECA becomes the lever. Jupiter cap is 16 nodes; we had been using **2**.

⚠ **Exclude the free-pass tasks:** trials that score >0 with **ZERO tool calls** — the verifier passes an
untouched repo. Seen at 40% of scored trials on `1246344`. They can never yield gradient and they inflate any
band/pass number. `band.py` counts them separately; exclude them from the denominator.

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
- **JURECA fleet `15498197`**: 32 nodes, expires **~00:40 KST 2026-08-06**. A full 18,312-trial run needs a
  **fresh 24h fleet** — operator action.
- **Bridge** `10.128.1.2:9920` healthy. ⚠ `workers_alive` is a bookkeeping artifact (batch `/worker/get_jobs`
  never updates it) — trust `envs.ready` climbing, never that flag.
- **Jupiter `1247578` RUNNING** — the parity run: 128 tasks × p@4, 8 nodes, `max_episodes=50`,
  1 CPU/1 GB sandboxes, 40960 window, TP=4 × 7 engines, thinking ON, `bare_json`. Endpoint port **18300**.
- Runs `1244916` / `1246344` / `1246702` / `1246853` are all cancelled by us. `1246344`'s traces are the
  **thinking-on, unbounded-episode baseline** worth keeping (0.17 trials/min).
- Helper scripts on the cluster (`/e/fscratch/reformo/lee27/`):
  `band.py <trace_jobs> [win_min]` — **use this one**: gate ladder + band as within-group variance + tool-call
  and free-pass accounting + concurrency. `gate.py` is the old 35B-hardcoded version. `throughput.py`
  (durations/concurrency), `parsergate.py` (parser registration), `hermestest.py` (parser dialect comparison),
  `nothinkgate.sh` / `routegate.sh` (compute-node route gates), `insandbox.py` (in-sandbox DNS/HTTP probes).

## Establish reality first
```bash
ssh -o BatchMode=yes jupiter "squeue --me -o '%.10i %.9T %.6M %.6L %.4D %N'"
ssh -o BatchMode=yes jupiter "ssh -S ~/.ssh/cm_jureca/qwen36 jureca05.fz-juelich.de \"squeue --me\""
ssh -o BatchMode=yes jupiter "curl -s -m 8 http://10.128.1.2:9920/status"
ssh -o BatchMode=yes jupiter "python3 /e/fscratch/reformo/lee27/band.py <trace_jobs_dir>"
```
⚠ **Always `-o BatchMode=yes` on Jupiter.** The remote sshd runs out of channel slots under concurrent use and
falls back to keyboard-interactive, which burns TOTP attempts (3 per try) and risks a JuDoor lockout. BatchMode
fails fast instead. The condition is transient and self-heals — wait, don't retry harder:
`until ssh -o BatchMode=yes jupiter true; do sleep 20; done`. **Serialize Jupiter calls; never run two at once.**

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
- **"33% pass ≈ Marianna's ~35% ⇒ a learnable band EXISTS / the risk is RETIRED"**: **RETRACTED.** A trial
  pass rate is not a band. Re-measured on the 35B's full trace tree: 31.6% pass rate and **0 of 16 groups**
  with within-group variance (5 always-solved, 11 never). GRPO's advantage is within-group, so an all-agreeing
  group gives exactly zero gradient. The comparison put GROUPS in the before-column and TRIALS in the after.
  Her real band is **358 / 4,578 ≈ 8% at `0 < pass@4 < 1`**, so ~35% was never a band number.
- **"~6× concurrency bump, not more nodes"**: **RETRACTED.** We hit the configured concurrency cap instantly
  every time (32 → 64 → 128 sandboxes live). The wall is per-trial latency: unbounded `max_turns` plus heavy
  sandboxes. See the throughput section.
- **"`1244916` died of a node failure"**: **RETRACTED** — `sacct`: `CANCELLED by 34902`, exit `0:0`, our own
  scancel. A job vanishing from `squeue` plus a `NodeDiedError` is not evidence of hardware failure; every
  teardown path produces both. **Distinguish by `sacct`.**
- **"full pool = 3,328 tasks / 13,312 trials"**: that is OUR `r2egym-patched-full-oracle` subset. Her pool is
  **4,578** ⇒ **18,312** trials at p@4.
- **SEVEN** accepted-but-ignored config keys now — verify any key by **behaviour**, never by presence in the
  materialized config. Newest: `extra_body` / `interleaved_thinking` are inert for **OpenCode** (harbor
  implements them for `terminus_2` / `openhands` / `mini_swe_agent` only).
- **A gate that passes DURING STARTUP proves nothing.** My "NO-THINK GATE: PASS" was taken on an engine
  mid-weight-reload; the identical probe later returned thinking-on output and a 500. Worse, probing during
  weight sync **killed an EngineCore and hung the driver** (params are on the meta device; `_C` ops are
  CUDA-only). Gate only after `Starting batch generation` / the first `trace_jobs/*/` dir.
