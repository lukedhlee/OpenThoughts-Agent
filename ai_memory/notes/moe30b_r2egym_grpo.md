# MILESTONE — Qwen3-30B-A3B GRPO on r2egym via the apptainer bridge

The first MoE RL run on r2egym: 200 GRPO steps on `Qwen/Qwen3-30B-A3B` with Marianna's
apptainer×r2egym sandbox runtime, checkpointed and validated every 20 steps, to establish a
baseline reward curve + validation curve.
Read before launching, before editing the r2egym 30B config, or when deciding what "done" means
here. Opened 2026-08-27 (Luke). Append-only; supersede with dated entries at the bottom.

---

## What matters right now

**This milestone is gated on the RL-stack bug work, not on compute or on data.** Luke is auditing
the stack and recording findings in [[skyrl_git_issue_queue]]; those get fixed first, then this
launches. Nothing here should be launched ahead of that gate.

**The two things that have never been proven together are the whole point of the run.** We have a
validated apptainer×r2egym runtime (driven by the trainer on 8B, and trainer-free on Marianna's
sweep path), and we have a validated MoE GRPO stack (gsm8k, 4 arms × 80 steps). **The MoE has never
touched r2egym, and r2egym has never been driven by an MoE policy.** Everything below is about
closing that specific gap.

**The parked band sweep is NOT a dependency.** [[decisions]] 2026-08-04 settled it: the raw pool
reaches the same final quality as the band (~45.5 SWB p@1), 120 steps vs 60 — the band is a
convergence/compute lever worth ~20%, not a quality one. Train on the raw allowlist. Do not
re-open the sweep to unblock this. → [[NEXT_SESSION]] § 0.0-SWEEP-PARKED

---

## The milestone spec (as stated by Luke, 2026-08-27)

| | value | note |
|---|---|---|
| Model | `Qwen/Qwen3-30B-A3B` (MoE, 30B-A3B) | which variant — Base vs Instruct-2507 — is an OPEN decision, below |
| Task set | r2egym | via apptainer SIFs, not Daytona |
| Algorithm | GRPO | |
| Steps | **200** | ~2× Marianna's full raw arm (she ran 120) |
| Validation | every **20 steps** | → 10 validation points |
| Checkpointing | yes, weights | cadence to match val (20) unless a reason to differ |
| Success criterion | reward curve goes up + a validation score at each interval | baseline-establishing, not a beat-the-baseline run |

**Sizing consequence, flagged for Luke:** at Marianna's geometry (bs 64 × n=8 = 512 rollouts/step),
200 steps = **~102k rollouts**. Her full raw arm was ~60k. At ~10 min/trial and conc 128 that is
~130h of pure rollout wall-clock before any incident overhead. The knobs are `train_batch_size`,
`n_samples_per_prompt`, and concurrency — see the concurrency ceiling under Risks. Worth an
explicit decision rather than inheriting 200 as free.

---

## Gate — do not launch before this clears

1. **[[skyrl_git_issue_queue]] findings resolved or explicitly accepted.** Luke owns this. Both
   open findings touch this run, and finding **#2 bites it directly**: on the Harbor path the YAML's
   temperature never reaches vLLM, so **this arm's `temperature: 1.0` is inert** and it will actually
   sample at `Qwen3-30B-A3B`'s own `generation_config.json` default. Finding #1 lists r2egym as clean
   *for #1 only* — that says nothing about #2. Read the served checkpoint's `generation_config.json`
   on-cluster before launching, and do not cite this run's `tis/` metrics until #2 is fixed.
2. **A short MoE-over-apptainer smoke.** The 8B proved the bridge; the MoE has not. The failure
   modes that matter (EP/FSDP2 hang, ghost GPU memory) are MoE-specific and orthogonal to everything
   the bridge validation covered.

---

## What already exists vs what is stale — the delta that has to be closed

Scaffolding is real and was written for this run, but predates both the gsm8k campaign and the
Marianna replication, so it encodes superseded values. **Do not launch it as-is.**

- `hpc/skyrl_yaml/jupiter/6node_qwen3_30b_a3b_r2egym_grpo.yaml`
- `hpc/skyrl_standard/jupiter/run_r2egym_moe30b_grpo.sh`

| field | committed value | what the milestone / later evidence says |
|---|---|---|
| `trainer.max_steps` | 50 | **200** |
| `trainer.eval_interval` | 999999 (disabled) | **every 20** — but see the in-loop-vs-offline decision below |
| `ckpt_interval` / `hf_save_interval` | 5 / 5 | 40 checkpoints at 200 steps; likely **20** |
| `n_samples_per_prompt` | 4 | **8** — [[decisions]] 2026-08-04, matches Marianna |
| `train_batch_size` | 32 | Marianna 64 |
| `n_concurrent_trials` | 32 | Marianna trains at 128; our GPFS ceiling is ~48/shard, ~192 total |
| `harbor.name` | `terminus-2` | Marianna runs **`terminus-structured`**; OpenCode ruled non-viable by managers 2026-08-12 |
| `harbor.max_episodes` | 999999 | Marianna **50** (mirrors DeepSWE `agent.max_steps`) |
| `max_model_len` / in / out | 32768 / 28672 / 4096 | Marianna **40960**; her in/out 15384/24576 |
| `override_timeout_sec` | 1800 | ✅ matches her |
| `temperature` | 1.0 | ⚠️ **inert** — finding #2 drops it on the Harbor path; the arm is "clean" for finding #1 only |
| `loss_reduction` | `sequence_mean` | ✅ matches her seqmean |
| `use_kl_loss` | false | she **overrides to true**; our gsm8k nokl arm beat KL-on by +0.45 — genuine open choice |
| `policy.optimizer_config.lr` | 1.0e-6 | gsm8k found **1e-6 undertrained at 80 steps** (66.4% vs 89.4% at 8e-6); her provenance is 3e-6-vs-8e-6 unresolved ([[marianna_parity]]) |
| `generator.batched` | true | **fails at launch** — `colocate_all: false` forces the fully-async trainer, which asserts `not batched` |
| `environment:` block | absent | must declare `type: apptainer` or the run falls through to Daytona |
| `data.train_data` | `DCAgent/r2egym-patched-full-oracle` (3,328, HF) | apptainer path uses local task dirs — **allowlist v3, 4,469 of 4,568 gate-verified solvable** |
| paths (`/e/scratch/reformo/lee27`) | as committed | gsm8k campaign relocated to **`/e/fscratch/...`** on inode/quota exhaustion; PI directive 2026-08-15: `/e/` only, `/p/` obsoleting |
| `RL_VENV` | `$DCFT/envs/rl` | `envs/rl` is **unusable** (`vllm._C` wants libcudart.so.12, Jupiter ships CUDA/13). gsm8k used **`$F/envs/rl-fa`** |

**Corrected 2026-08-28 — an earlier version of this note said the bridge wiring was the big gap.
That was wrong.** Apptainer is ready: the SIFs, the bridge, the fleet, and harbor's apptainer
environment type are all built and exercised. Two smaller things are actually missing:

1. **Neither r2egym RL config declares `environment: type: apptainer`.** That block is what selects
   the apptainer backend, and it lives in the harbor config. Without it the run falls through to the
   Daytona default. This is a block to copy in, not infrastructure to build.
2. **The 30B config would fail at launch.** It pairs `batched: true` with `colocate_all: false`, and
   `colocate_all: false` auto-selects the fully-async trainer, which asserts `not batched`. The 8B
   config already documents this in a comment naming the 30B config as the offender. The 8B config
   also carries attention settings and a fully-async worker count that the 30B one never received.

---

## Assets we already hold (verified surviving the August cleanups)

| asset | where | state |
|---|---|---|
| r2egym SIFs | `/p/scratch/synthlaion/lee27/r2egym_sif` | 4,583 — **on `/p/`, which is obsoleting; plan a migration** |
| Task dirs | `/e/fscratch/reformo/lee27/tasks/r2egym-raw` | 4,578, harbor layout |
| Allowlist v3 | `allowlist_r2egym_v3.txt` (+ `gatefail_r2egym_v3.txt`, 99 names) | 4,469 solvable, model-free gate-verified |
| Marianna's harbor fork | `marianna_repro/harbor_patched_src` | terminus-structured + bridge env; copied because `/p/project1` is login-only on Jupiter compute |
| SWE-bench-100 SIFs | JURECA | oracle **100/100** — makes SWE-bench-100 a viable val set, see open decisions |
| Parked sweep state | `marianna_repro/slurm_logs/jobs/` | ~1,750 trials banked; not needed for this milestone |

---

## Known risks, in order of expected cost

1. **The EP/FSDP2 backward hang.** ~1 per 2 arm-hours on 30B-A3B, no upstream root fix. The gsm8k
   campaign survived **68 incidents over ~48 arm-hours** via kill → exclude nodeset → resume. A
   200-step agentic run is far longer in wall-clock than 80 gsm8k steps, so **budget for this as the
   dominant operational load, not as an exception.** Runbook: [[base30b_gsm8k_validation]].
2. **Ghost GPU memory.** Killed jobs strand ~75GB; the next job OOMs at engine load on a 95GiB GPU.
   Recipe is the same: add the nodeset to `node_exclusion_list` in `hpc/hpc.py`, relaunch.
   Exclusions age out (~24h ghost / ~48h hang) — they are quarantine, not a blacklist.
3. **GPFS in-doubt inflation from sandbox churn.** Killed the band sweep twice with EDQUOT at only
   ~17GB real usage. It follows the **create/delete churn rate**, not capacity: keep concurrency
   ≤~48/shard (~192 total), watch staging dir-count from minute 1, alarm >900. Moving filesets only
   resets a ~1h clock. → [[NEXT_SESSION]] § 0.0-SWEEP-PARKED, [[jsc_inode_quota_2026-08]]
4. **Two-cluster topology is a single point of failure.** SIFs are amd64 → they run only on the
   JURECA/JUWELS CPU fleet; Jupiter (GH200, aarch64) serves vLLM and trains. They are joined by
   reverse forwards on a ControlMaster living on the Jupiter *login* node. **A listener is not a
   route** — probe from a fleet compute node, not from the login node.
5. **Silent reward-plumbing failures.** The single most expensive class in this project's history
   (600s exec cap faked "r2egym is untrainable" for a week; a constant per-task reward faked a
   working pipeline). Before trusting step 1: confirm `len(set(rewards)) > 1` within at least one
   group, and that a 0.0 corresponds to a real failing test rather than an infrastructure zero.

---

## Open decisions — need Luke

1. **Which 30B-A3B?** `Base` (what the gsm8k validation used) vs `Instruct-2507`. The agentic path
   needs tool-use competence at step 0 or the band is empty; Base was fine for gsm8k but that is not
   evidence for agentic. A pass@8 probe before committing 200 steps is the cheap answer — that is
   exactly the pattern used for curriculum-easy ([[currease_pass8_probe_handoff]]).
2. **Validation set: held-out r2egym, or SWE-bench-100?** [[r2egym_grpo_plan]] chose held-out r2egym
   *because Daytona's snapshot cap made SWE-bench impossible*. **That constraint is gone on the
   apptainer path** — we hold oracle-validated SWE-bench-100 SIFs. SWE-bench-100 is the real
   scoreboard and directly comparable to Marianna's ~45.5 p@1. Held-out r2egym is in-distribution
   and better for overfit detection. Possibly both: r2egym in-loop, SWE-bench offline.
3. **In-loop val or offline-on-checkpoints?** Agentic validation is synchronous and blocking —
   in-loop val at 100 tasks every 20 steps stalls training each time. [[r2egym_grpo_plan]] chose
   offline-on-checkpoints for this reason. That gets the same 10 validation points without the
   training stall, at the cost of a second harvest job.
4. **LR, and KL on or off.** gsm8k says 1e-6 undertrains and 8e-6 was best; that was a different
   task and a different horizon, so it is a prior, not an answer.
5. **SIF migration off `/p/`.** Required eventually by the PI directive; decide whether it happens
   before this run or after.

---

## Pointers

- [[skyrl_git_issue_queue]] — **the gate.** Stack bugs found by Luke; fix before launching.
- [[r2egym_grpo_plan]] — the original 30B r2egym design: val-set reasoning, the eval-regime confound
  (+2.7 vs +10 on the same model), what to hand the eval lead. Read before choosing a val set.
- [[marianna_parity]] — her exact hyperparameters and the band-number disambiguation. Read before
  claiming any of her settings.
- [[base30b_gsm8k_validation]] — the MoE GRPO stack validation + the incident runbook this run will
  live inside.
- [[apptainer_bridge_handoff]] — the bridge itself: how it was built, what it proved.
- [[r2egym_apptainer_reference_impl]] — the r2egym-on-apptainer runtime details.
- [[decisions]] 2026-08-04 — raw-vs-band (the reason the sweep is not a dependency), group size n=8.
- [[gotchas]] — the accepted-but-ignored-key bug class. Verify by behaviour, never by config.

---

## Status log

- **2026-08-27 — opened.** Milestone defined by Luke. Gated on the stack-bug work in
  [[skyrl_git_issue_queue]]. Nothing launched.

- **2026-08-28 — corrections.** Apptainer readiness overstated as a gap; see the corrected block
  above. Finding #2 in [[skyrl_git_issue_queue]] confirmed to hit this arm directly: its
  `temperature: 1.0` is inert on the Harbor path. `enable_summarize: false` confirmed correct
  (Luke) and already set — summarization erases the long-horizon persistence this run trains.
