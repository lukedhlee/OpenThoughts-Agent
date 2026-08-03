# r2egym FSDP2 GRPO — the plan, the validation-set decision, and the eval handoff

The design of the current headline experiment: train Qwen3-30B-A3B with FSDP2 + GRPO on r2egym and
measure agentic-eval lift vs the pre-RL baseline. Contains the validation-set decision and its
reasoning, the Daytona snapshot measurements that constrain it, the eval-regime confound that can turn
a real +10 into a fake +2.7, and exactly what we hand to the eval lead.
Read before launching the r2egym RL job, before choosing a val set, or before asking anyone to eval our
checkpoint.

Carved out of `handoff.md` 2026-07-29 (append-only from here; supersede with dated entries at the
bottom). Snapshot/SWE-bench analysis: [[apptainer_bridge_swebench]]. Why not GSM8K:
[[gsm8k_format_artifact]]. Decisions log: [[decisions]].

---

## Goal (opened 2026-07-27)

Train Qwen3-30B-A3B (MoE) with **FSDP2 + GRPO** on the **r2egym** task set and measure how our key
agentic eval metrics improve vs the pre-RL baseline.

- **Data:** `DCAgent/r2egym-patched-full-oracle` — **3,328** verifier-bearing SWE bug-fix tasks
  (Harbor task-binary format; binary test-pass reward). It's the RL-ready R2E-Gym member of TaskTrove.
- **Backend:** FSDP2 (not Megatron for this run). Setup = **disaggregated** (the only viable 30B-A3B
  layout on GH200s per the 2026-07-26 bake-off). Agentic RL path → `rl-agentic-launch-jupiter`
  (Harbor + Daytona), NOT the standard-gym gsm8k path. Keep **`use_tis: true`** (cheap MoE insurance).

## VALIDATION SET = held-out r2egym (decided 2026-07-28)

`r2egym-patched-full-oracle` has **NO built-in train/val split** (single 3,328-task `train`; confirmed
via HF splits API). Carve a random **~200-task held-out slice** as `val_data`, train on the remaining
~3,100.

Why held-out r2egym (not SWE-bench, not nothing):
- **In-distribution** → the correct signal for overfit detection + **best-checkpoint selection**.
- **Free on snapshots** — held-out tasks fall on the same **7 envs** → fits our 60-cap org.
  (SWE-bench / any ID benchmark greatly exceeds our cap, so it CANNOT be an in-loop val — that's the
  decider.)
  **MEASURED 2026-07-28 — the real number is far worse than the 85–100 originally estimated:
  SWE-bench is 1 snapshot PER TASK (500 tasks → 500 unique env hashes, each Dockerfile pins a distinct
  `swebench/sweb.eval.x86_64.*` base). Harbor also NEVER reaps snapshots (delete only on error paths),
  and org headroom is 40 ⇒ max Daytona SWE-bench val set ≈ 40 tasks. Full 200-task SWE-bench needs
  Marianna's apptainer bridge on an x86_64 cluster (Jupiter is aarch64 — those images cannot run there
  at all). Full analysis + options: [[apptainer_bridge_swebench]].**
- **Regime-clean:** run **no-summ at training context** (harbor already `enable_summarize: false`).
- **Mechanics:** run it **OFFLINE on the saved checkpoints**, NOT in-loop (agentic val is
  synchronous/blocking). Keep `eval_interval: 999999`, `ckpt_interval: 5`; use held-out reward to pick
  the ship ckpt.
- Live training signal meanwhile = `reward/avg_raw_reward` on the r2egym rollouts.

## FORMAL EVAL → HANDED OFF TO MRINAL (eval lead) (decided 2026-07-28)

We do **not** run the ID benchmarks ourselves — a single ID benchmark (85–100 snapshots) exceeds our
60-cap org; it's the eval team's infra + the DATA org. **Our deliverable = an HF model id + size;
Mrinal registers + evals.**

- **Register via `rl-agentic-job-cleanup`** → `laion/<job>-<step>-<size>` (PUBLIC, weights flattened at
  repo root, vLLM-servable). NOT the trainer's auto-pushed `lukedhlee/<job>` repo (wrong nested layout).
- **Size to state:** Qwen3-30B-A3B = **30B MoE (~32B-class serving)**, not 8B.
- Ask Mrinal to ALSO eval **base `Qwen/Qwen3-30B-A3B`** on the same set → the **base→RL Δ** is the
  headline.
- (DATA-org key `DAYTONA_DATA_API_KEY` is Mrinal's concern, NOT ours — RL uses only `DAYTONA_API_KEY`.)

### ⚠ EVAL-REGIME CONFOUND (RL co-lead finding, 2026-07-28) — LOAD-BEARING

r2egym RL teaches long-horizon persistence, which **summarization + short context ERASES → false
negative.** Co-lead's `DCAgent/g1_diverse_tezos_100k_8b` (r2egym→SWE-bench):
**40k / summ ON = +2.7 (looks dead)**; **81920 / no-summ / TP4 = +10 (clearly works).**

So when Mrinal evals our model, **SPECIFY THE REGIME: SWE-bench, ~81920 ctx (YARN), NO summarization,
TP=4, 200×1** (200 = 100 verified folders + 100 random SWE-bench; 200×1 < SE than 100×3 —
task-variance dominates trial-variance). "SWE-bench @40k/summ" ≠ "@80k/no-summ" — different benchmarks
in effect. Align the exact protocol with the co-lead so our 30B number is comparable to his g1-8b.

Our config trains at 32k/4k-gen — revisit whether to train longer-ctx (separate design Q; the eval
regime is the immediate lever).

### Mrinal's ID scorecard (reference)

SWE-Bench-Verified (in-dist headline), Terminal-Bench-2, dev_set_v2. ID mean + per-benchmark z
(`analyze-id-eval-ranking`). Traps: `swebench-verified-random-100-folders` (ID, N=100) ≠ full
`swebench-verified` (OOD, N=500); never rank on `dev_set_v2` (partial).

## Pre-launch checklist

Confirm r2egym prebuilds its **7 snapshots** under cap (below) + **≤6 RUNNING RL jobs/cluster**
guardrail before submitting.

## Daytona snapshot findings (verified 2026-07-27)

- `r2egym-patched-full-oracle` = **7 unique snapshots** for all 3,328 tasks (measured: distinct
  `environment/Dockerfile` hashes; runtime-clone design → 1 env shared by up to 1,286 tasks).
  Launcher auto-prebuilds these; **snapshots are a NON-issue for the RL run.**
- Our Daytona org (`DAYTONA_API_KEY`, in `/e/scratch/reformo/lee27/keys/secrets.env`):
  **22/60 used → 38 headroom** (20 of the 22 are Daytona base images, only 2 are `harbor__*`).
- **Eval is the opposite — ~1 snapshot PER TASK** (measured: tb2 89 tasks→85 envs, dev_set_v2 100→89;
  swebench-100 ~100). ID suite ≈ **~274 unique snapshots**, OOD >1,000 → exceeds any single org cap →
  why eval is a TWO-org build-on-demand + 2h-reclaim cycling process (Mrinal's infra).

## Open item carried from the GSM8K work

**Before trusting r2egym's `avg_raw_reward`: measure what fraction of its failures are non-substantive**
(malformed patch, out-of-budget) rather than capability. This is the transferable lesson from
[[gsm8k_format_artifact]] — the structurally identical failure there cost us a whole reward curve's
interpretation.
