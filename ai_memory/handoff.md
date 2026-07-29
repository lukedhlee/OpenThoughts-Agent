# Handoff — lukedhlee · rewritten 2026-07-29

## Objective

RL the general **`Qwen/Qwen3-30B-A3B`** (MoE, not Coder) and show a real post-RL lift on agentic coding
evals (SWE-Bench Verified, OT-TB Lite, Terminal-Bench 2.0). Primary repo: **OpenThoughts-Agent**,
branch `lukedhlee/vista-moe-grpo-30b`. Clusters: **Jupiter (JSC)** for training, **JURECA (JSC)** for
the x86_64 sandbox bridge; Vista is concluded.

**Current focus = the JURECA apptainer bridge** (the eval side), not the RL launch. See Goal + Next
action below.

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
- **r2egym FSDP2 GRPO: designed, NOT launched.** Still the headline experiment, now queued behind the
  bridge baseline. → [[r2egym_grpo_plan]]
  - Its own blocker when we return to it: `envs/rl` cannot import vLLM (`libcudart.so.12` vs the Jul-27
    cu130 torch swap), and `hpc/rl_launch_utils.py:780` still defaults `RL_ENV_DIR` there ⇒ **an FSDP2
    launch today reproduces the break.** Fix = rebuild vLLM in `envs/rl` from `src/vllm_fork` against
    cu130. → [[gotchas]]
- **Open, gating the 200 but not the 100:** no authoritative manifest for a second 100-task SWE-bench
  set (only the pinned `DCAgent2/swebench-verified-random-100-folders` @ `a2e51e9e`, exactly 100 tasks).

## Goal (apptainer)

Turn the bridge from *plumbing that works* into **a SWE-bench number we can defend**: evaluate base
`Qwen/Qwen3-30B-A3B` on the pinned 100-task set at the specified regime (~81920 ctx, no summarization,
TP=4). That single run delivers the **base→RL Δ denominator** the headline needs, load-tests the bridge
at 100-task concurrency, and exercises the regime — without waiting on the RL checkpoint.

## Next action

**Serve `Qwen/Qwen3-30B-A3B` on JURECA `dc-gpu` at TP=4 as a standalone vLLM smoke.** This is the gate
and it fails fast: bf16 weights are ~60 GB against 40 GB A100s, so TP≥2 is mandatory and TP=1 cannot
work — yet every model-backed bridge smoke to date was 0.6B at TP=1. Verify the engine comes up and that
KV fits at ~81920 ctx with YARN scaling before any 100-task config work. Then, in order: author the
81920-ctx eval config, size the worker pool against the `dc-cpu` 6h walltime, submit the 100-task run.
Full gap list and ordering: [[apptainer_bridge_handoff]] § "2026-07-29 (later)".

This is an **eval**, not RL, so the standing "do not launch an RL job" guard does not apply and ordinary
submission needs no approval — state the config, then go.

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
