# Handoff — lukedhlee · rewritten 2026-07-29

## Objective

RL the general **`Qwen/Qwen3-30B-A3B`** (MoE, not Coder) and show a real post-RL lift on agentic coding
evals (SWE-Bench Verified, OT-TB Lite, Terminal-Bench 2.0). Primary repo: **OpenThoughts-Agent**,
branch `lukedhlee/vista-moe-grpo-30b`. Vista is concluded.

**Cluster roles (corrected 2026-07-29 — the old "JURECA = the sandbox cluster" framing was wrong):**
- **Jupiter (JSC)** — ALL model execution (training + vLLM), AND the r2egym sandboxes, because r2egym's
  `python:3.6-slim-buster` base is multi-arch/arm64 and runs natively on GH200.
- **JURECA (JSC)** — sandboxes for **SWE-bench only**, whose `sweb.eval.x86_64.*` images are x86_64-only
  and cannot run on aarch64 Jupiter at all. **No model is ever served on JURECA.**

**Current focus = the SANDBOX layer, which now blocks the RL run.** A small-model r2egym GRPO run
(Qwen3-8B) was brought all the way up on 2026-07-29 and everything worked except sandboxes. See
§ Sandbox below; it supersedes the older "the bridge is not on the RL critical path" framing.

## State

- **CLOSED — GSM8K:** the `pass_at_1 ≈ 0.45` plateau was a **format artifact**, not math ability
  (paired, n=1319); no headroom worth reporting. → [[gsm8k_format_artifact]]
- **CONCLUDED — Vista:** the 24-node stall was never made deterministic (it moved between
  `fwd_logprobs`, `policy_train`, `sync_weights`); work moved to Jupiter, where it didn't reproduce.
  → [[vista_24n_sync_stall]]
- **READY — Jupiter training backends:** FSDP2 6-node disagg (16 policy EP4×FSDP4 + 8 infer) at
  ~432s/step; Megatron 1.36× step / 1.66× policy_train faster, same-topology resume only.
  → [[jupiter_bringup_and_throughput]], [[jupiter_megatron_bringup]], [[megatron_vs_fsdp2]]
- **← ACTIVE FOCUS · Apptainer/OpenCode bridge on JURECA: plumbing DONE, no real run yet.** Passed a
  1-task infra smoke, a 1-task model-backed smoke (**0.6B**), a 3-task **pristine-oracle** run (3/3, no
  model reasoning), and a 100-SIF cache audit. Launcher wired; cluster left clean. **But no real model
  has ever been evaluated through it** — everything so far was toy scale.
  → [[apptainer_bridge_handoff]] § "2026-07-29 (later)"
- ⚠ **The bridge code is NOT on this checkout.** It lives only in
  `/Users/lukedhlee/OpenThoughts-Agent-apptainer-bridge` (`lukedhlee/apptainer-opencode-bridge` @
  `bc8378ed`) + `/Users/lukedhlee/harbor-apptainer-bridge` (`02b06bc5`), both clean. The current
  `vista-moe-grpo-30b` tree has no `--apptainer-bridge-url` wiring and no apptainer eval config.
- **← r2egym GRPO (Qwen3-8B): LAUNCHED and RAN 2026-07-29. Whole stack works; SANDBOXES do not.**
  Job `1087425`/`1087417` reached a full training step: 8 vLLM engines up (15.27 GiB each, **472,016
  KV tokens** per engine), harbor 0.8.1, transformers 5.8.1, FSDP2 policy init, `init_weight_sync_state`
  6.4s, `sync_weights_to_inference_engines` 5.9s, `fwd_logprobs` 5.1s, advantages, policy train.
  **But `avg_raw_reward: 0.0`, `zero_rewards: 1.0`** — 128/128 rollouts died on
  `DaytonaConnectionError`. Cancelled; no gradient was possible. See § Sandbox. → [[gotchas]]
  - **The old `envs/rl` blocker is RESOLVED but the fix was different than expected:** `envs/rl` is
    genuinely unusable (its `vllm._C` needs `libcudart.so.12`; Jupiter ships only CUDA/13), so the run
    uses **`envs/rl-megatron`** ("megatron" is only the dir name — `strategy: fsdp2`). Nine further
    launch blockers were fixed to get there — env completeness (daytona SDK), harbor too old
    (`normalize_message`), `LD_LIBRARY_PATH` pointing at the broken env, `train_data` as a parquet
    yielding ZERO tasks silently, the fully-async trainer's constraint set, and the
    `flash_attn: false` trio. All recorded in [[gotchas]]; all fixed on the branch.
- **★ r2egym itself may be the wrong dataset shape.** The co-lead's reference implementation states raw
  r2egym **collapsed** with zero reward; she trains on a **model-specific learnable band**
  (`0 < pass@8 < 1`, 740 tasks + 100 held out). Our config trains on all 3,328 — the configuration
  already known to fail. **Measure base Qwen3-8B `pass@k` on an r2egym sample before any long run.**
  → [[r2egym_apptainer_reference_impl]]
- **Open, gating the 200 but not the 100:** no authoritative manifest for a second 100-task SWE-bench
  set (only the pinned `DCAgent2/swebench-verified-random-100-folders` @ `a2e51e9e`, exactly 100 tasks).

## Sandbox — the current blocker (written 2026-07-29)

**The one-line version:** agentic RL needs a sandbox per rollout; the only sandbox we had wired was
**Daytona, which is a CLOUD API**, and **JSC compute nodes have no internet** — so every rollout fails.

### Why it wasn't obvious

The failure is **silent**. `mask_exceptions` includes `DaytonaError`/`NetworkError` with
`default_error_treatment: zero`, so connection failures are absorbed into **zero rewards** instead of
crashing. The run executes perfectly and reports `avg_raw_reward: 0.0` — which reads as "the model can't
do it" rather than "the sandbox was never reachable". Under GRPO, all-zero reward ⇒ zero advantage
variance ⇒ **no gradient**; it would have burned all 50 steps on 3 nodes reporting nothing wrong.
**Always grep an RL log for `Network is unreachable` / `DaytonaConnectionError` before believing a low
reward.**

`hpc.py` does set `needs_ssh_tunnel=True` for jupiter, but the machine-local edit set
`proxychains_binary=""`, and `hpc.py:752` gates the entire proxy-setup block on that being non-empty ⇒
compute nodes get **no outbound path at all**. For the old GSM8K (non-agentic) runs that was harmless,
which is why it went unnoticed.

### The measured network shape (this is what constrains every option)

From a real probe job on Jupiter compute `jpbo-040-12` — full table in [[gotchas]]:
- internet (`app.daytona.io:443`, `huggingface.co:443`): **BLOCKED**
- JURECA, any address/port: **BLOCKED**
- Jupiter login via **public** IP: **BLOCKED**
- Jupiter login via **internal** `10.128.1.2` / `10.99.0.2`, HTTP **and** `:22`: **REACHABLE**

⚠ **Never probe with public IPs** — doing so gives a false "totally isolated" reading. Compute↔login
works over `10.128.x`/`10.99.x` (same /16), which is why Ray works, and a **relay on a Jupiter login
node is the basis of every viable fix.**

### The three options

| | what it is | status |
|---|---|---|
| **A · login-node proxy → cloud Daytona** | build proxychains-ng (gcc+git present on login), fill in `proxychains_binary`; compute reaches Daytona via login. Reuses the **7 already-prebuilt** r2egym snapshots. No SIFs, no tunnel, no TOTP. | viable, smallest change; **kept as fallback** |
| **B · Jupiter-LOCAL apptainer bridge** ← chosen direction | bridge server on a Jupiter login node (`10.128.1.x:9920`), workers + SIFs on Jupiter compute. Works because r2egym is multi-arch/arm64. No cross-cluster anything. | **the reference implementation does exactly this** → [[r2egym_apptainer_reference_impl]] |
| **C · JURECA bridge for r2egym** | what was briefly chosen before reading the reference impl | **WRONG CLUSTER** — unnecessary for arm64-capable r2egym, and Jupiter compute cannot reach JURECA at all |

### What option B still needs

1. **SIFs must be BUILT, not pulled.** r2egym `environment/Dockerfile`s are recipes
   (`FROM python:3.6-slim-buster` + apt + pip), unlike SWE-bench's *published* images. The reference
   impl converts Dockerfile→`.def` and runs `apptainer build --fakeroot`.
2. **`--fakeroot` works on COMPUTE, not login.** Login fails with "No user namespaces available"; the
   reference prebuild runs on 4 compute nodes. This resolves the open question in
   [[apptainer_bridge_handoff]].
3. **Building needs internet** (docker pull + apt + pip) and compute has none ⇒ pre-pull base layers into
   `APPTAINER_CACHEDIR` from a login node, or proxy compute. **So option A's proxy work is likely
   required for B too — it is not wasted either way.**
4. **Agent compatibility unverified:** our bridge was validated only with **OpenCode**; the training
   config uses **terminus-2** and the reference impl uses **terminus-structured** (not even in our pinned
   harbor's `AgentName` enum).
5. **Latency risk:** agentic rollouts make many sequential tool calls. Generation already took ~46 min for
   one step's buffer against Daytona (mostly retry backoff), and 50 steps must fit a 12h walltime.

### ⚠ Leaked credential to report

The reference `prebuild_r2egym_worker.sh` contains a **plaintext Docker Hub PAT**, world-readable under
`/p/project1/laionize`. **Marianna must revoke/reissue.** Not used by us; value deliberately not recorded.
Details: [[r2egym_apptainer_reference_impl]].

## Goal (apptainer)

Turn the bridge from *plumbing that works* into **a SWE-bench number we can defend**: evaluate base
`Qwen/Qwen3-30B-A3B` on the pinned 100-task set at the specified regime (~81920 ctx, no summarization,
TP=4). That single run delivers the **base→RL Δ denominator** the headline needs, load-tests the bridge
at 100-task concurrency, and exercises the regime — without waiting on the RL checkpoint.

⚠ **Amended 2026-07-29:** the model for this eval must be served on **Jupiter**, not JURECA (no model runs
on JURECA), with JURECA `dc-cpu` supplying only the x86_64 sandboxes. That makes a **reverse tunnel**
(Jupiter → JURECA, initiated from Jupiter because JURECA→Jupiter has no route) a prerequisite, not an
optional extra. Script + measured routing facts: `eval/jureca/jupiter_to_jureca_tunnel.sh` on the bridge
branch, and § Sandbox above. The 100-task eval config already exists there too
(`eval_opencode_apptainer_swebench100_ctx80k.yaml`, 77824+4096=81920, compaction off, 1 attempt) plus a
clean `opencode_swebench_prompt.md.j2` — the pre-existing smoke template hardcodes a sentinel instruction
and an astropy path and would corrupt all 100 tasks.

## Next action

~~Serve `Qwen/Qwen3-30B-A3B` on JURECA `dc-gpu` at TP=4~~ — **SUPERSEDED 2026-07-29. No model runs on
JURECA**, so the "does 30B fit A100-40GB at TP=4" question is **moot**; it was the largest listed unknown
and it simply does not apply. Serving happens on Jupiter GH200 (96 GB), where an 8B measured 15.27 GiB +
472k KV tokens per engine, so headroom is not the issue it was feared to be.

**Actual next action — decide the sandbox path, then unblock RL** (§ Sandbox above):

1. **Highest leverage, human-blocked: message Marianna.** She already runs r2egym+apptainer+8B RL on
   Jupiter and owns the prebuild list, worker scripts, and learnable-band tooling — one reply could
   replace most of options A/B. She also **must be told to rotate the leaked Docker Hub PAT**. Ask
   additionally: does `/p/scratch` (her `sif_cache`) resolve from Jupiter, and how do air-gapped compute
   nodes get internet during `apptainer build`?
2. **Independent of ALL sandbox work — measure whether r2egym can move at all for our model:** base
   Qwen3-8B `pass@k` on an r2egym sample, to see whether a learnable band exists. Inference-only, no
   sandbox, no bridge. The reference impl says raw r2egym **collapses**, so without this a "fixed"
   sandbox just buys a flat reward curve. Same lesson as [[gsm8k_format_artifact]].
3. Then build the sandbox path (B, Jupiter-local, with A as fallback).

Eval submission needs no approval; the standing "do not launch an RL job unasked" guard still applies to
RL. Note the RL config itself is now **fully debugged and working** apart from sandboxes —
`hpc/skyrl_yaml/jupiter/3node_qwen3_8b_r2egym_grpo.yaml` + `run_r2egym_qwen3_8b_grpo.sh`.

In parallel (human-blocked, not on the critical path): **ask Marianna for the second 100-task
manifest's provenance.** At n=100 the 95% CI is **±9.0 pts**, which barely resolves her +10 and cannot
resolve a +5; at n=200 it tightens to ±6.4. That is the real reason the second manifest matters.

## Invariants & constraints

- **Never `scancel` a RUNNING job without explicit user OK.** Standing exception: experiments *we*
  spawned on Megatron/Jupiter may be cancelled. Never cancel anyone else's job.
- **Do not auto-relaunch after a cancel** unless asked. **Do not launch an RL job** without being asked.
- **Ordinary non-destructive experiment submission needs no approval** — state the key config, then go.
  Destructive actions still need explicit approval.
- **Autonomy:** on smoke/bring-up failures, keep fixing + relaunching without waiting for "go fix it".
  Report when blocked (e.g. MFA) or when a milestone lands.
- **Git (HARD):** never `git push`, open a PR, or merge without an explicit ask **in that turn**. Push
  only to `lukedhlee`-owned repos. Cluster-local patches are fine for unblocking; upstream PR is a
  separate opt-in step. An unmerged fork fix rides `--harbor-ref` / `--skyrl-ref`.
- **Local clones are ground truth; clusters never diverge.** Edit locally → push → `git pull` on the
  cluster. No hand-editing, no patch-by-rsync. ⚠ The Jupiter clone currently has ~6 uncommitted files —
  a pre-existing violation that needs reconciling.
- **Never `find`/`du` on Jupiter GPFS or JURECA scratch** — inode exhaustion can get the whole project
  banned.
- **Daytona snapshot caps are HARD** (10 new/launch, 60 org) — clean stale, never raise. Always pass
  `--api-key-env DAYTONA_API_KEY` to the snapshot manager (the default is a different org).
- `enable_db_registration: false` in YAMLs. **≤6 RUNNING RL jobs per cluster.**
- **Secrets** only from `$DC_AGENT_SECRET_ENV`; never commit; never source someone else's keys. A
  committed key is a leak — rotate, don't fix-forward. → [[vista_secrets]]
- **Compute nodes have no internet** on every JSC system — stage on login/JUDAC; WandB offline → sync
  from login. → [[jupiter_wandb]]
- **Don't call a big job wedged too early**, and don't "fix" MoE weight sync by disabling
  `SKYRL_W13_RELOAD_BRACKET` (token salad). Both, plus ~30 other failure modes → [[gotchas]].
- Full working agreement (how Luke plans, PR procedure, cluster index) → `notes/operating-preferences.md`.

## Decisions in force

One line each; full context, rejected options, and consequences in [[decisions]].

- GSM8K retired as an MoE-RL vehicle; keep the `strict` scorer if it's ever rerun. *(2026-07-29)*
- Don't claim a comparable 200-task SWE-bench eval until the second manifest's provenance exists.
  *(2026-07-29)*
- Validation = a ~200-task **held-out r2egym slice**, scored **offline on saved checkpoints**.
  *(2026-07-28)*
- Formal ID benchmarks are Mrinal's; we ship an HF model id + size and **specify the eval regime**
  (SWE-bench, ~81920 ctx, no summarization, TP=4, 200×1). *(2026-07-28)*
- Bridge agent is **OpenCode**, not Terminus; tooling injected by **bind-mounting static binaries**.
  *(2026-07-28)*
- **Training-side decisions, all settled — re-read [[decisions]] before touching an RL config, don't
  re-litigate here:** disaggregated only (colocation falsified); `use_tis: true` and no R3;
  `cpu_offload=false` when memory allows; never add inference nodes to speed eval; larger geometry starts
  from scratch (FSDP2 ckpts can't reshard); Megatron is the PI's default but GSM8K understates it; NCCL
  debug must be file-backed; GRPO-vs-RLOO is a non-lever. *(2026-07-13 → 07-26)*

## Open questions

**Apptainer (focus):**
- **Second 100-task SWE-bench manifest** — only the co-lead (Marianna) can resolve provenance. Also
  ask: how does her 200-task SWE-bench *eval* actually run (her SIF docs omit swe-bench); the exact
  200-task list; her training-side context length; and how she stages OpenCode on air-gapped workers.
- **Does 30B-A3B fit JURECA A100-40GB at TP=4 with ~81920 ctx?** Weights ~60 GB force TP≥2; KV headroom
  and YARN scaling at that context are unverified. This is the Next action.
- **`dc-cpu` 6h walltime vs a 100-task run** — port Marianna's `MAX_CHAIN=5` auto-requeue, or size the
  run to fit one allocation? Undecided.
- **Do we ever need the other 400 SIFs?** Only 100 are cached. A full 500 prebuild is ~1–2 TB and
  ~6h at `MAX_PARALLEL=4`, and is pointless unless the val set grows past the pinned set.
- **Should the bridge branch merge into the working branch, or stay isolated?** It is currently
  worktree-only, which is safe but means the launcher on `vista-moe-grpo-30b` cannot run a bridge eval.

**Other:**
- **What fraction of r2egym failures are non-substantive** (malformed patch, out-of-budget) rather than
  capability? Must be measured before `avg_raw_reward` is trusted — this is the transferable lesson
  from the GSM8K artifact.
- **TP > 1 in the RL loop on Jupiter** — believed possible, never run. It is the only thing that could
  reopen colocation.
- **Weight-sync cost** is still ~43% of a 6-node step. Can it be reduced?
- Do unsupported Megatron checkpoint modes fail early with a clear message yet? (They should.) And are
  there regression tests for the checkpoint fixes?

## Map

**How to write memory:** `memory-guide.md` — two-tier rule + triage test. Read when *writing* memory.

**Permanent, append-only:** `decisions.md` (every choice + context + rejected options — read before
re-litigating anything above) · `gotchas.md` (~30 symptom→cause→fix — **read before debugging anything
env-shaped**: CUDA/vLLM/flash-attn/apptainer/inodes/NCCL) · `logs/<date>_<topic>.md` (raw session
journals: 07-19/07-20 Vista, 07-22 Jupiter flash-attn + vLLM reload, 07-26 Megatron 6n + TIS/R3) ·
`weekly/2026-07-27.md` (plain-language Megatron week).

**Apptainer — the focus, read these first:**
- `notes/apptainer_bridge_handoff.md` — operational state, exact commits, measured smokes, and the
  **§ "2026-07-29 (later)"** run scoping. **Read before touching the bridge.**
- `notes/apptainer_bridge_swebench.md` — why the bridge exists: x86_64-vs-aarch64, 1-snapshot-per-task,
  her architecture. **Read before proposing any SWE-bench val set.**
- `notes/jureca_marianna_setup_plan.md` — her stack + fork refs. **Read before messaging her.**
- `notes/jureca_ssh.md` · `notes/judac_ssh.md` · `notes/jureca_what_goes_where.md` ·
  `notes/jsc_paths_hazards.md` — JURECA access (the `from=` IP trap) and where files may/may not go.
- `notes/vista_moe_30b_sizing.md` · `notes/vista_moe_parallelism.md` — MoE sizing + EP/FSDP/TP rules.
  **Read before setting TP for the 30B serving smoke.**

**Queued RL work:** `notes/r2egym_apptainer_reference_impl.md` (**READ FIRST for anything r2egym** — the
co-lead already runs r2egym+apptainer+8B RL on Jupiter; has the harbor wiring, the SIF build path, a
leaked-credential flag, and the finding that **raw r2egym collapses** and needs a model-specific
learnable band) · `notes/r2egym_grpo_plan.md` (design, val-set decision, eval handoff — **read before
launching**) · `notes/jupiter_bringup_and_throughput.md` · `notes/jupiter_megatron_bringup.md` ·
`notes/megatron_vs_fsdp2.md` (**read before choosing a geometry/backend**) · `notes/moe_grpo_tis_r3.md` ·
`notes/jupiter_cluster.md` · `notes/jupiter_ssh.md` · `notes/jupiter_wandb.md` · `notes/vista_secrets.md`.

**Reference / historical:** `notes/operating-preferences.md` (working agreement) ·
`notes/gsm8k_format_artifact.md` (**read before citing any GSM8K number**) ·
`notes/vista_24n_sync_stall.md` (full Vista job record; only if Vista revives) · `notes/tacc_vista.md` ·
`notes/vista_moe_gsm8k_grpo.md` · `notes/vista_moe_milestone_plan.md` ·
`notes/jureca_grpo_vs_rloo_plan.md` (completed A/B + fairness table) ·
`notes/jureca_agentic_daytona_plan.md` · `notes/jureca_agentic_smoke_recap.md` ·
`experiments/vista_moe_gsm8k_tracker.md` · `scripts/` (`jureca_from_clause.sh` = JuDoor `from=` line,
`explore_marianna_jureca.sh`, `jupiter_monitor_r10.sh`, `vista_import_loop.py`).
