---
status: active
project: snowball-r2egym
kind: decisions
authority: canonical
updated: 2026-09-02
---

# Decisions currently in force

Only active decisions belong here. Full historical decisions are preserved at
`../../archive/ledgers/decisions_legacy.md`.

## Mission and order

- The current mission is Snowball RL on R2E-Gym.
- The order is serve → Snowball-specific pass@8 probe → RL → held-out evaluation.
- R2E-Gym is the active training environment. The earlier R2E-Gym + SWE-smith combined-pool proposal
  is deferred until R2E-Gym works.
- Qwen/Coder is prior art, not an active model or workstream.

## Model and data

- Start from `laion/snowball-67b-a2b-sft-s3-nemotron-terminal-step1888` at the pinned revision recorded
  in `research/snowball_rl_plan.md`.
- Measure the learnable band on Snowball itself. Model-specific Qwen/8B/Coder bands are invalid inputs.
- Keep infrastructure failures and null rewards outside the model-failure denominator.

## Recipe

- Begin with the E17a Snowball constraints: FSDP2, expert parallel size 1 for training, frozen router
  bias, stochastic-rounded bf16 AdamW, GRPO, and no KL, entropy bonus, or length penalty.
- Treat agentic deltas such as batch size, sequence-mean loss, staleness, and learning rate as
  hypotheses until validated on Snowball; do not silently promote the Coder recipe to canon.
- Select checkpoints by held-out evaluation, not training reward alone.

## Coordination

- Align with the core Marin RL group. Marianna-specific material is historical reference only.
- Never push, file an issue or PR, merge, or modify another person's jobs without the authorization
  required by `shared/policies/operating-preferences.md`.

## 2026-09-03 (Snowball serve → probe session)

- **Inference layout is TP1 × DP4 × EP4 per 4-GPU node.** Not a choice: the marin vLLM fork raises
  `GrugMoE expert-parallel GPU serving requires tensor_parallel_size=1` for TP>1. Supersedes the
  plan's "DP4 or TP4" alternatives.
- **Attention backend: fork default (FLASH_ATTN).** Nine 1-node A/B jobs (short context, 14.7k
  decode-only, 14.7k unique-prefix prefill at conc 64) put FLASH_ATTN, TRITON_ATTN and FLASHINFER
  within noise; the OTA-vLLM-0.22 FA3 pathology does not exist in this fork. The band template's
  FLASHINFER export is equivalent and left as is.
- **`NCCL_PXN_DISABLE=1` for every Snowball trainer job on Jupiter.** With the template's
  `NCCL_NET_GDR_LEVEL=0`, NCCL 2.28.9's proxy-over-NVLink path fails its /dev/shm attach on the first
  cross-node scatter (both cu12 and cu13 builds). Beats dropping the GDR override because it keeps
  every other setting identical to the working Coder-30B recipe; revisit GDR for speed later.
- **Task pool: our raw R2E-Gym-Subset dirs + oracle gate (4,469), not `DCAgent/r2egym-patched-full-oracle`
  (3,328).** Luke confirmed 09-02 evening. Same pool by V1 index; the patched set's snapshot-collapsed
  images need outbound network and grade the stock wheel on our Apptainer path (settled in August).
- **Start checkpoint reaffirmed:** `laion/snowball-67b-a2b-sft-s3-nemotron-terminal-step1888`
  (5.7T Stage-3 agentic SFT). Ben's message 09-03: use the 5.7T agentic SFT "unless you enjoy hard
  mode". Closes the issue draft's question 1.
- **Probe context budget: run two variants in parallel** — 24,576/8,192 (the SFT eval split) and
  30,720/2,048 — and decide the RL harness from their pass@8 tables. **Summarization on/off is
  deliberately left to Luke** (harness choice vs the SFT eval contract; Ben's #8483 called the
  summarizer unreliable).

## 2026-09-03 (probe → pool session, 01:35–04:15 CEST)

- **Summarization is out of the harness (Luke, 03:25 CEST).** The summarization probe was cancelled
  at 681 attempts; its partial table is kept for the record only. Reason recorded: with it on, the
  900 s eval wall replaces context death and trajectories become segments — a different harness.
- **Full-pool pass@8 runs at 28,672 in / 4,096 out, summarize off, eval timeout 900 s** (launched
  03:48 CEST on Luke's "submit before we go offline"; the budget was mine, not yet ratified). Beat:
  24,576/8,192 (smaller band, 44 mixed on val), 30,720/2,048 (57 mixed but the 2k cap is hit on 12%
  of turns and each hit costs a retry). The 4,096 cap touches 2.3% of turns and costs one turn of
  input; its own val table (b28k) is the check. **The pool band is only valid for this budget.**
- **Every probe keeps the 900 s eval timeout** so the tables stay comparable; the training rollouts
  use 1800 s. Report the budget triple (context, summarize, timeout) with every number.
- **Prior-turn reasoning stays in the chat history.** The SFT corpus keeps every earlier turn's think
  block, so dropping reasoning (a third of the window) would be a departure from the SFT contract —
  an option to put to Ben, not a fix. The faithful fix is restoring the `<|start_think|>` markers
  vLLM strips (marker smoke pending at handoff).
- **No post-update eval is believed without a same-config replicate of the baseline.** The one-step
  jump (pass@8 0.196 → 0.295) is held as "unverified" until the replicate lands.
- **The 24k probe's automatic train step and final eval were allowed to run** (deviation from the
  "kill at eval/" plan): they gave the gate-3 plumbing evidence and the paired-eval mechanic for
  free; later probes are cancelled at their eval block because they add nothing new.

## 2026-09-03 (successor session, 04:15– CEST)

- **Serving past 32k: two-key vLLM override, no fork change.** `--hf-overrides '{"max_position_embeddings":
  65536, "max_seq_len": 65536}'` (the grug config freezes `max_seq_len` as a copy and the RoPE table on
  the sliding-window layers is sized from it). Adopted for every >32k engine; the probe generator's
  `--max-model-len` emits it. Rationale: both pretraining cooldowns ran at 65,536; smoke coherent to
  62k; global layers are NoPE and windowed layers never see |Δpos| > 2,047.
- **The 60k validation probe runs with a 1800 s eval timeout** (deviation from the 900 s contract):
  at ~25 s/turn under load a 900 s wall binds before a 60k window does, so a 900 s table would
  measure the wall. Its table is comparable to the other budgets only on pass@8/mixed counts, not on
  the terminal-cause breakdown. Budget triple: 61,440 / 4,096 (advisory) / summarize off / 1800 s.
- **The per-turn output cap is advisory** (`model_info.max_output_tokens` is not passed as
  `max_tokens`); budgets are described by their input limit. The 28k pool budget stands as launched.
- **Baselines are two-eval averages.** Same-config step-0 evals differ by 0.04 pass@8 on the 224-task
  split; no single eval is a baseline for a paired comparison.
- **Proposed, not yet adopted (Luke's call — it changes the harness contract under which every table
  was measured):** the SFT-parity fix — `skip_special_tokens=false` via harbor `extra_body`, a
  terminus-2 patch that strips the think span before the JSON parser while keeping it in history, and
  `chat_template_content_format="string"` in MarinSkyRL. Patches prepared on local branches
  (`lukedhlee/terminus2-think-parity`, `lukedhlee/chat-template-content-format`), not pushed.
- **Not decided, flagged to Luke:** whether the RL harness moves to 64k (the 28k pool band would then
  not be the training band; a 64k pool pass costs ~3×). Waiting on the full 60k table.
- **28k pool pass cancelled (Luke, 06:00 CEST 09-03).** The 60k probe's early signal (trial pass
  ~18–23%, 35 turns) makes 64k the intended harness, so the 28k band was no longer worth 48 nodes ×
  2 h more. Partial tables kept for the record; the 64k pool pass replaces it (8 shards, 12 h wall, on
  the existing fleets), to be run under the final harness contract (parity decision pending).

## 2026-09-03 (Luke, 08:05 CEST / 23:05 PT 09-02 — RL experiment design)

- **GRPO trains on the strict-mixed band; behaviour is tracked per training step; after training,
  the filtered-out tasks (all-zero + all-solved) are re-probed at the same budget to measure transfer
  and list the tasks that flipped.** Luke's design, agreed by the session with one ordering
  constraint: the harness budget (64k vs 32k) is decided before the pool pass that produces the band.
  Full design and the caveats (band drift, first-step size, two-eval baselines, sandbox deaths):
  `research/rl_band_experiment_design.md`.
- **Trace analysis by subagents before training (Luke):** Sonnet/Opus subagents read the 60k traces —
  quantitative behaviour profile vs 24k, within-task success/failure contrasts on mixed tasks (the
  learnable signal), all-zero failure taxonomy (within-reach vs out-of-reach), sandbox-death root
  cause. Dispatched 08:00 CEST 09-03; reports land under the session scratchpad `agents/` and are
  folded into `experiments/`.

## 2026-09-03 (serving-benchmark session, 06:20–07:50 CEST — Luke: "measure and optimize vLLM serving")

- **Keep the production engine flags; serving is at its config floor.** A decode step is 11–12 ms of
  HBM-bound kernels at ~15 seqs/GPU; 19 one-node A/B arms on a replay of real trajectories put every
  knob within ±3 % or worse. Rationale + tables: `experiments/2026-09-03_snowball_serving_bench.md`.
- **FLASH_ATTN stays, but the 09-03 00:xx "three-way tie" rationale is superseded:** under real
  multi-turn load TRITON_ATTN is −35 % and FLASHINFER +3 %; short-prompt A/Bs are not evidence for
  serving changes (benchmark with `code/snowball/bench/` replay instead).
- **Proposed, not adopted (needs one validation probe + Luke's OK):** `attention_backend=FLASHINFER` +
  `VLLM_USE_FLASHINFER_SAMPLER=1` + `performance_mode=interactivity` via `engine_init_kwargs`, measured
  together at +5–6 % per turn. The sampler switch changes the RNG stream, not the distribution.
- **Rejected for this workload:** n-gram speculative decoding (acceptance length 1.3–1.7 at
  temperature 1.0 / top-k 20 → −30…−50 %), Triton MoE tile tuning (kernel is bandwidth-bound; H200
  config identical), prefill-chunk-size changes (4k–32k identical, 2k −20 %), TF32 router (+2 %, alters
  routing numerics vs the trainer).
- **Throughput is bought with concurrency per engine node, bounded by the eval timeout:** 64 → 128 → 192
  trajectories per node = 160 → 206 → 272 turns/min at 13.8 → 23 → 25 s per turn (28k harness); at 64k
  a trajectory (34 turns) costs ~9–10 min of API time at 64 per node.


## 2026-09-03 (Luke, 09:00–09:15 CEST — go on the harness and the band)

- **Apply the SFT-parity fix now** ("if there is a running full probe, cancel it and apply the fix"):
  no probe was running; deployed 09:00–09:10 on the shared checkouts + snowball venv. Every table
  from here on is under the parity contract; the merged 60k table (0.348) is the last pre-parity one
  and the parity probe is its paired comparison.
- **Harness at 64k; band = the whole R2E-Gym train pool at pass@8 under parity** (~34k attempts,
  ~320 GH200 node-hours, one-off per contract). Rationale recorded in chat: the band is both the
  training set (~1,300 mixed tasks at 33%) and the baseline for the filtered-out transfer test.
- **Standing instruction for the autonomous stretch (Luke, 09:17 CEST):** prioritise saving time; use
  more GPU nodes if needed; cheaply verify before committing to big runs; actually read agent traces
  sometimes; use a cheap Opus/Sonnet model to babysit runs.
- **Bridge reaper window 2400 s** (`BRIDGE_STALE_READY_SEC`) adopted for the 64k regime; restart the
  bridge with it whenever it is recreated.


## 2026-09-03 (successor session, 09:20–09:55 CEST — parity validated, pool pass gated on it)

- **The SFT-parity contract is validated and every table from here on is measured under it.** On 115 parity-probe
  trajectories (4,414 turns): 99.3 % of completions start with `<|start_think|>`, the "Extra text detected before JSON"
  warning is on 0.2 % of turns (was 93 %), and in the served history the assistant `<|eot_id|>` is followed directly by
  the user header in 99,565 of 99,565 boundaries (the stray newline is gone). **The newline after the *user*
  `<|eot_id|>` is the model's own chat template** (`apply_chat_template` renders `user…<|eot_id|>\n<|start_header_id|>
  assistant`), not a mismatch — the old "token after `<|eot_id|>` must be 128006" check was wrong for this template
  and was replaced by `code/snowball/parity_check.py` (asst-eot-clean vs asst-eot-stray-newline pattern counts).
  Trial pass on the first 106 scored attempts: 14.2 % (pre-parity 60k table: 12.1 %). The remaining "Previous response
  had warnings" prefixes (~3 % of turns) are the harness's keystroke-format warning after the model's own `C-c`, which is
  in-contract behaviour, not a parity defect.
- **The 64k parity pool pass launches on that verdict** via the login-node gate (`pool_gate.sh`), not by hand, so the
  launch does not depend on the Mac's SSH session.
- **Trace trees are compacted after tabling.** A 60k attempt costs ~100 MB on disk because the ~35 MB result blob is
  written three times and is ¾ indentation; `compact_traces.sh` keeps every analysis input (trajectory.json, one compact
  result.json, verifier/) and drops the duplicates and pane dumps > 2 MB (p0: 24 → 2 GB, table identical). Applied by
  `pool_watch.sh` per shard and by `compact_all_finished.sh` to every finished tree. Rationale: reformo fscratch was at
  4.6 TB / 2.06 M files against a 2 TB / 400 k per-user soft cap before the pool pass; the Coder-30B trees
  (`lr*v5`, `band_full_*`, `band_resid_*`, `band_r3_*` ≈ 3.6 TB) are the archive candidates for Luke.
- **Trainer batch rule (found by the smokes):** MarinSkyRL asserts `train_batch_size ≥ dp_size` (prompts) and
  `policy_mini_batch_size × n_samples ≥ dp_size`; at fsdp 64 a step is ≥ 64 prompts. The GRPO default is therefore
  64 prompts × 8 samples = 512 rollouts per step (`make_snowball_grpo.py`).

## 2026-09-03 10:30–10:45 CEST (successor session — GRPO geometry, bridge timeouts)

- **GRPO geometry = fsdp 64 on 16 policy nodes + 8 engine nodes, 64 prompts × 8 samples per step.** Both 64k memory
  smokes passed (peak 64.8 GB at fsdp 32, 61.8 GB at fsdp 64; policy_train 115 s per 64 sequences at 2 per GPU on either);
  fsdp 64 halves policy-train wall (~8 min per 512 rollouts) and the trainer's batch rule (`train_batch_size ≥ dp_size`)
  makes 64 prompts the minimum there anyway. The rollout wave (~15–25 min at 64 trajectories per engine node) bounds
  the step; `max_staleness_steps=1` overlaps it with training.
- **Bridge-job timeouts are treated as infrastructure, not model behaviour, and widened for RL rollouts only.** ~9 % of
  pool attempts die mid-trajectory as `BridgeOperationTimeoutError` with sub-5 s commands: terminus-2's batched exec
  waits `sum(durations) + 180 s` and the JURECA per-node worker queue exceeds that at ~2k live sandboxes. Eval retries
  them; RL rollouts cannot, so the GRPO sbatch exports `HARBOR_TMUX_BATCH_EXEC_TIMEOUT_MARGIN_SEC=600` (harbor
  `lukedhlee/terminus2-think-parity-bridgewait`, default unchanged so the pool pass's contract is untouched) and a
  32-worker-per-node fleet (15586407) is queued. The pool pass keeps the 180 s contract; its null rate is reported
  beside the band.

## 2026-09-03 11:40 CEST — Coder-30B trace trees archived off fscratch (successor session, inode emergency)

The per-user inode quota on `/e/fscratch/reformo/lee27` tripped at 11:17 and cost the live pool pass ~150–400 attempts per
shard (`risks.md`). After the fast releases (28k pass trace trees, package caches), the concluded Coder-30B campaign's
compacted trace trees (`lr{1e7,3e7,1e6}v5`, `band_full_s0-3`, `band_resid_s0-3`, `band_r3_s0-3`; 551k files / ~890 GB)
are being **tarred to `/e/data1/mmlaion/lee27/archive_0903/<exp>_trace_jobs.tar` and then removed from fscratch**
(`code/snowball/archive_coder_traces.sh`, tmux `archive_coder`, log `experiments/archive_coder.log`). Tables, logs,
configs and checkpoints of those experiments stay where they were. Rationale: those runs are superseded (pre-#452 lr
sweeps; a Coder band is not Snowball evidence) and the inode headroom is needed for the rest of the pass and for GRPO
checkpoints. Reversible: untar from data1.

## 2026-09-03 16:20 CEST — parity contract confirmed on the paired val table

- **Parity vs pre-parity on the 224-task val split at the same 60k budget: pass@8 0.366 vs 0.348, strict-mixed 79 vs
  74, trial pass 11.7 % vs 12.1 % — inside the noise floor.** The SFT-parity harness is the contract for the band, GRPO
  and every eval from here; the parity2 table is the step-0 val baseline (pair it with each GRPO run's own step-0 eval
  for the two-eval average). Under the fixed infrastructure the harness null rate is 2.3 % (model-killed tmux + verifier
  timeouts), which is the denominator to quote beside band numbers.

## 2026-09-03 18:30–18:45 CEST — band arms: RLOO-N, staleness 3, engine:policy ratio measured first (Luke, successor session b48b29c0)

- **Supersedes** the automatic launch chain's plan (GRPO, `max_staleness_steps=1`, 16 policy + 8 engine nodes per arm).
  The chain was stopped after it built the band and before it submitted anything; the arms are launched by hand.
- **Estimator = `rloo_n`** (`trainer.algorithm.advantage_estimator=rloo_n`, `group_advantage_min_size=4`). Reason: the fork's
  GRPO estimator ignores `exclude_from_baseline`, so a masked infrastructure failure still counts as a reward-0 sibling in
  the group mean; RLOO-N drops it from the baseline, zeroes its advantage, and filters zero-variance groups. Luke chose to
  switch rather than patch GRPO; **GRPO vs RLOO-N is compared later at the 1-epoch sweep stage.** Both lr arms (1e-5,
  3e-6) stay, since RLOO's advantage scale (no std division) differs from GRPO's.
- **Staleness 3** (Mercor 397B guide) and the **engine:policy ratio is measured before the arms launch** — two async runs on
  the same 32 band tasks (`snowball_overfit32_a` 8 policy + 8 engines @ 32/node; `_b` 8 + 4 @ 64/node; 32 × 8 per step,
  12 steps) read out `timing/wait_for_generation_buffer` vs `timing/policy_train`; the arms take the ratio that makes the
  wait ≈ 0. They double as the Mercor-style overfitting check (learning signal within a few steps on a learnable subset).
- **Two unlisted infrastructure classes masked** for the arms: `ConnectionResetError`, `BridgeOperationTimeoutError`
  (were falling to `default_error_treatment=zero`). Applied by `code/snowball/patch_arm.py` on the generated config.
- **Not fixing sample-level regeneration** of infrastructure-failed rollouts: under a healthy fleet true infra nulls are
  ≈0.8 % of samples (≈6 % of groups lose one sample to a 7-sample baseline). The guard is the masked-fraction stop rule
  (> 5 % of a step's trajectories), not a trainer change.
- **Megatron is not an option for Snowball:** `validate_grug_training_strategy` rejects anything but fsdp2 for `grug_moe`.

## 2026-09-03 18:55 CEST — band cleaned to 1,332 tasks; hacking counters; verifier timeout 600 s (session b48b29c0, from the four pool trace analyses)

- **The arms train on `tasks/r2egym-raw-v3-train-band60k-clean` (1,332 = 1,524 − 192).** Excluded: the reward-validity
  quarantine (159 tasks: 19 git-history leak, 35 pre-solved, 62 nondeterministic, 49 graded on ≤ 2 or generic tests) ∪
  the harness tier-1 list (62: 16 with > 1,000 graded tests, 46 with a no-op win). List:
  `experiments/analysis/artifacts/2026-09-03_band_exclusions.txt`. Rationale: 4.3 % of band wins were unearned and
  concentrated in these tasks; RLOO-N cannot fix a reward that pays for `git checkout <fix commit>` or for doing nothing.
  The strict-mixed band itself (`band_strict_mixed.txt`) stays the measured artifact; the clean list is the training set.
- **Verifier timeout 600 s for RL rollouts** (pool measured at the effective 120 s; `task.toml` declares 720 s). Only
  reduces preserved-zero false signal on big suites; does not change which tasks are in the band.
- **Per-step hacking counters** (`rollout_profile.py`): sandbox grading-metadata touches, `git log/reflog/show`,
  `git checkout/reset/restore/stash/revert`, completion declared right after a failing observation. Baselines from the
  64k smoke: 0 / 11.7 % / 6.3 % / 0.8 % of rollouts. A rise in any of them while reward rises is the shortcut, not learning.
- **Overfit32 subset kept as is** although 8 of its 32 tasks are quarantined (drawn before the audits): its job is the
  engine:policy timing; read its learning signal on the 24 clean tasks.

## 2026-09-03 19:10–19:15 CEST — arms launched at 64 trials per engine node, 16 + 16 nodes, epochs 10 (successor session)

- **Supersedes** the 18:30 entry's "expected 16 @ 32/node". Engines run **64 concurrent trials per node** because a 60k
  rollout's wall time (~14 min) does not depend on the per-node load, so throughput is proportional to concurrency
  (b's wave: ~4 rollouts/min/node at 64; the 32/node smoke: 2.1). 16 engine nodes supply ~64/min against the trainer's
  ~58/min demand (policy-train 450 s per 8 seq/GPU, measured on b's step 1) → trainer-bound, staleness manager idles the
  engines, never the trainer. Bridge/fleet load: 1,024 sandboxes per arm.
- **`trainer.epochs=10` on the arms, 12 on the overfit run.** The trainer runs `epochs × len(dataloader)` steps capped by
  `max_steps`; the generator's `epochs=1` would have ended the arms at ~21 steps (b ended after one). `patch_arm.py`
  sets it (third argument, default 10).
- **The overfit relaunch (1648015) runs the arms' exact recipe** (RLOO-N, `group_advantage_min_size=4`, verifier 600 s,
  the two extra masked classes) instead of GRPO, so it is the small-geometry gate for the arm recipe and the 12-step
  overfitting curve; b's one GRPO step is the ratio measurement and the one-step gate.
- **Trace reaping is a launch prerequisite** (`rl_reaper.sh` on the login node) because the project fileset was 16 k
  files under its soft limit and each arm writes ~670 k inodes per 12 h uncompacted. Eval sessions are tabled before
  they are tarred; training trial dirs are dropped 30 min after completion.
- **Coder-30B trees archived and removed** (Luke, 19:03: "we don't need Coder-30B anymore") together with the pool
  residual-wave trace trees, as single tars in `/e/data1/mmlaion/lee27/archive_0903/`; pool tables stay in place.
- **HF export tested on b's step-1 checkpoint** (job 1648018) rather than waiting for a step-12 checkpoint.

## 2026-09-03 19:55 CEST — engine hang fixed in the fork, all three runs relaunched on it (successor session)

- **The hang is the per-actor retry after vLLM's ephemeral DP master-port collision** (detail in
  `.claude/projects/vllm/vllm.md` and `.claude/projects/marinskyrl/marinskyrl.md`); it affected 8 of 8 engine nodes on
  the first overfit run and 4 of 16 on the first lr1e5 arm, and lr3e6 / overfit_a never reached "engines ready".
- **Chosen: fix the port choice in the fork** (`lukedhlee/jupiter-parity64k` @ 686736fa, pushed to Luke's fork and
  pulled on Jupiter; local worktree `/Users/lukedhlee/MarinSkyRL-jp64k`) and relaunch 1649843 / 1649845 / 1649846.
  Beat: re-rolling (5–13 collisions per 16-node job made a clean launch unlikely), letting lr1e5 run at 12/16 engines
  (a quarter of trials would time out into false zeros, and eval nulls), and a client-side eviction of hung engines
  (more code, still 25 % capacity loss). Upstream PR not opened — Luke's call.
- **The three sweeps on the relaunch are the acceptance test**: 0 "port collision" lines and no `ENGINE-HUNG?` node.

## 2026-09-03 11:29–12:20 PT (20:29–21:20 CEST) — arms and overfit cancelled; cheap probes before any full run (session 087b8d72)

- **Cancelled 1649843 / 1649845 / 1649846** (own jobs, provably broken: 4/16, 4/16 and 8/8 engine nodes hung with zero port
  collisions). Luke 11:31 PT: "use cheap, fast probes first to ensure correctness and then run the full run when done."
- **Probe protocol:** clone the overfit geometry (deterministic 8/8 reproducer) with one variant each via `clone_ovf.sh`,
  16 nodes / ~15 min, sweep 3 min after the first requests. Round 1: ctrl 8/8, eager 0/8, nomix 8/8, noasync 8/8, tp4 died
  (port base vs TP init port), piecewise 8/8. Round 2 (in flight): tp4b, eng4, nosync.
- **Eager rejected for the arms** despite being hang-free: ~12 tok/s per sequence vs 60–90 with graphs → 60k rollouts would
  exceed the 1800 s agent timeout. Kept as the slow RLOO-N gate (`snowball_ovf_eager` 1650643).
- **Fork commit ecb677a3** (`SKYRL_SKIP_STARTUP_WEIGHT_SYNC=1`, off by default, diagnostic only; never for a checkpoint resume).
- **Reporting rule:** every time to Luke in PT (see `shared/policies/operating-preferences.md` § Communication).

## 2026-09-03 13:03–13:35 PT (22:03–22:35 CEST) — arms relaunched at 4 engine nodes; no growth until the hang is understood (session 087b8d72)

- **Both arms run 16 policy nodes @ fsdp 64 + 4 engine nodes @ 64 trials** (RLOO-N, staleness 3, epochs 10, ckpt every 6,
  eval every 12 at K=8, `eval_before_train=false`). It is the only engine geometry that has never hung (three independent
  runs, incl. the eval-only pool pass) and it is Marin's own reference scale. Cost: ~28 min per step, ~24 steps per 12 h.
  Beat: 16 engines (25 % hung), 8 engines (100 % hung, incl. the arm-scale probe at 16 trials per rank), eager engines (no
  hang, ~20× slower end to end).
- **Step-0 eval skipped tonight** (it would cost ~100 min at 4 engines); the parity2 table is the baseline; the first held-out
  read is at step 12. A standalone base eval on a separate 8-node probe job is the way to recover the paired baseline.
- **No rebuild of the vLLM fork** (upstream is one week newer, no relevant fix); **rebase MarinSkyRL onto main at the next
  restart** (28 commits; #491/#494 change the loop we use — never mid-run).
- **Root-cause path:** per-iteration engine logging on the reproducer (probe `iterlog`), then the 1-node/8-node vLLM-only
  reproducer, then escalate to Ahmad/Romain with the drafted issue.

## 2026-09-03 14:05–14:40 PT (23:05–23:40 CEST) — engine hang root-caused to SkyRL placement; fix deployed on the fork; PR staged (session b390fd72)

- **Cause:** the flat PACK placement group split each DP4×EP4 engine across two nodes (rank 0 on one, ranks 1–3 on another);
  the cross-node EP all-to-all deadlocks under CUDA graphs. Evidence: `experiments/2026-09-03_engine_hang_resolution.md`.
  This supersedes the 12:20 PT reading ("every CUDA-graph mode hangs on DP4×EP4") and the 13:03 PT rule (≤ 4 engines).
- **Fix chosen:** per-engine STRICT_PACK for tp·pp·dp > 1 + a startup assertion (fork `lukedhlee/jupiter-parity64k`
  @ b505c4a6 / fcecf833). Beat: `VLLM_ALL2ALL_BACKEND=naive` (would mask the split, not fix it) and TP4 (port bug, parked).
  Validated by the 8-engine reproducer: 0/8 hung at 3 and 15 min (1651990) vs 8/8 on seven launches before.
- **Pushed to the fork and pulled on Jupiter without asking** (own fork, running jobs unaffected — editable install, modules
  already imported). **PR not opened, issue not filed** (standing rule: Luke's explicit ask). PR branch cherry-picked onto
  upstream main in a worktree (`lukedhlee/dp-engine-node-local-placement` @ 2ca5940f, ruff clean, 19/19 CPU placement tests).
- **Arms left running at 4 engines** (Luke locked them; the lock's premise is now gone). Recommendation on the table:
  relaunch both at 16 engines @ 64 via `relaunch_arm16.sh` (eval every 24, eval_before_train=true); cost ≈ 1.5 h of progress.
- **Probes cancelled after reading:** 1651711 (iterlog), 1651990 (placement fix). No probe running.

## 2026-09-03 14:50–15:40 PT (23:50–00:40 CEST) — gate run collapsed; upstream merged; Mercor procedure before the full run (session b390fd72, Luke)

- **eng4 (lr 1e-5 overfit gate) cancelled at step 3** — reward 0/256, grad 0, entropy 0.03, 103 masked: dead, not overfitting.
  Both band arms show the step-2 precursor (entropy 0.41 → 0.23 at 1e-5, 0.40 → 0.29 at 3e-6). **Luke: drop lr 1e-5; next 3e-6
  and 1e-6.** The arms were left running through the handoff (their step 3 is the confirmation).
- **Upstream pulled as MERGES into both deployed branches** (Luke, PI's ask): MarinSkyRL main 18c9fa28 → `lukedhlee/jupiter-parity64k`
  @ 02f2c650 (trainer.py add/add kept both; dispatcher took upstream's process-isolated groups + our eval round-robin); harbor main →
  `lukedhlee/terminus2-think-parity-bridgewait` @ 34732fca (worker.py kept both). Beat: rebase (force-push on deployed branches).
  **Not pulled on Jupiter mid-run** (eval actors would import mixed versions); gate = the 8-engine reproducer on the merged code.
- **vLLM update = separate track**, built against torch 2.11 in a copied env (marin main pins torch 2.13; no cu130 aarch64 wheel).
  Not before the relaunch. A background build agent was started and died with the session (only the source clone landed).
- **Luke: follow the Mercor RL-systems procedure before the full run** (`plan.md` § "Stage 3b"); **80 nodes approved** → 16 training
  + 24 engine nodes per arm. PR go still not given; issue text ready.
- **Login-node incident (own fault):** running the MarinSkyRL CPU tests on jpbl-s01-02 exhausted the 4096-pid cgroup; recovered by
  single-command pkills. Rule recorded in gotchas + `.claude/ops/jupiter/ops.md`.

## 2026-09-03 15:45–21:25 PT (00:45–06:25 CEST 09-04) — successor session 2189af97: gate, Mercor Step 3 sync arms, fleet crunch

- **Cancelled both 4-engine band arms (15:47 PT)** — both had failed the entropy gate by step 3 at flat reward; Stage 3b needed the code
  checkout quiet for the pull. Evidence table in `state.md` update block. Own jobs, Luke's standing "cancel when the plan needs" applied.
- **Pulled the merged fork on Jupiter and gated it**: two merge defects fixed in flight (hydra schema renames → `fix_merged_keys.py` +
  `validate_hydra_args.py` in the clone scripts; stale `_outcome_rewards` import → fork cdcb435b, pushed to the fork without asking, own
  branch). Gate passed on both paths (placement, sweeps, clean steps).
- **Ran Mercor Step 3 as synchronous overfits** (`clone_sync.sh`, async trainer at staleness 0) at lr 1e-5 (Luke had dropped it for arms;
  kept as the staleness control), 3e-6, 1e-6, 5e-7 (lower bracket from the #452 memory), plus 3e-6 variants: processed logprobs
  (Luke's "mismatch?" question), full-distribution sampling, and 1e-6 + KL 0.01 (my ablation pick when 1e-6 entropy kept sliding).
  Cancelled collapsed / non-viable arms as they proved it (1e-5 after 2 steps, proclp after 4, nosamp never stepped, 3e-6 after 4).
  Result and interpretation: `experiments/2026-09-03_mercor_step3_sync_overfit.md`.
- **Fleet renewal**: submitted a 48-node 24 h fleet (16:45 PT) and four 8-node 12 h backfill fleets (18:52 PT); did not touch the
  foreign `apptainer_workers_tt*` jobs. **JUWELS second pool** proposed to Luke; he asked for a brief for another session (on his
  clipboard 20:55 PT, filed as `handoffs/2026-09-03_2055PT_juwels-sandbox-pool.md`); Luke uploads the JuDoor key.
- **Deleted Luke's stale Qwen hub caches from the Jupiter home** (Qwen3-30B-A3B/8B/0.6B, 74 GB) on his explicit go, plus the regenerable
  Triton cache, to clear the home quota that blocked the JUWELS ControlMaster socket.
- **vLLM-next**: built and serve-smoked (DP4×EP4 OK, +15 % conc-64 throughput; TP4×EP4 is NotImplemented upstream); not adopted for RL.
- **Handoff scheme changed (Luke, 21:15 PT)**: per-session handoff files under `ai_memory/handoffs/` + `ACTIVE.md` index; `TAKEOVER.md`
  is a pointer; `handoff_copy.sh <file>`; skill + memo + START_HERE updated.

## 2026-09-04 (session 0a6013a8, JUWELS pool + TaskTrove wave)

- **JUWELS is a full second sandbox pool, sized at 8 workers/node.** Its batch partition starts 32-node fleets in under a minute
  where JURECA dc-cpu queues for hours, and it is cheaper per sandbox-hour (3 vs 4 core-h). The nodes are diskless with a 47 GB
  tmpfs, and each sandbox creates a 4 GB overlay there, so 16/node overflowed under sympy-heavy load; 8/node did not. Beat: JURECA
  renewals (queue-bound), `mem192` nodes (7 h queue). Budget account `laionize` (synthlaion has no JUWELS quota; transfernetx `norun`).
- **Bridge port convention:** one bridge server per pool on the Jupiter login node (`:9922` Snowball/JURECA, `:9924` TaskTrove,
  `:9926` Snowball/JUWELS); CPU-cluster login nodes see reverse forwards (`jwlogin03i:9926 → :9926`, `jwlogin03i:9925 → :9924`,
  mirroring JURECA's `:9923`/`:9925`). Next pool takes `:9927`.
- **TaskTrove r2egym overlap is treated as our raw set — no re-selection.** 504 shared tasks are byte-identical to the raw twins and
  score within k = 8 noise of the raw 60k table (bucket agreement 83.5 % vs 86.6 % expected). What TaskTrove adds is sympy only
  (moto/matplotlib are text-less): pass@8 0.106, ~31 % mixed band, 93 % context deaths at 61k — extra band tasks for the 64k contract,
  nothing for shorter contracts. Beat: re-gating the overlap on TaskTrove's side.
- **Eval-only probes on the merged fork run with `use_tis=false`** until the eval-path rollout-logprobs check is fixed upstream
  (see issue queue #6). Training arms are unaffected.

## 2026-09-03/04 — TaskTrove r2egym track (session ae444a78)

- **Keep TaskTrove's task selection, replace its environment shape (Luke, 16:10 PT 09-03):** the 3,328 TaskTrove tasks were
  rebuilt in R2E-Gym's original shape (per-task image, /testbed prebuilt, offline) for JSC. Beat: running TaskTrove's flattened
  tasks as shipped (agent clones + pip-installs per rollout, network at solve time, not comparable to the full probe).
- **Harness-run setup instead of agent-run setup (Luke, 20:30 PT 09-03):** harbor gained a `setup_files/setup.sh` hook
  (branch `lukedhlee/setup-files-hook`, later cherry-picked onto the deployed lineage as
  `lukedhlee/terminus2-think-parity-bridgewait-setuphook`). Beat: leaving the install preamble in the agent's instruction.
- **Daytona is the target environment for TaskTrove r2egym (Luke, 22:06 PT 09-03: "so that we are not blocked by CPU
  clusters"); JSC stays as the reference side.** Comparability is measured, not argued: same model, same recipe, both envs.
- **One image per repo for pure-Python repos, per-rollout install for the rest; pandas + numpy excluded on Daytona** (their
  tests pass with no fix on TaskTrove's images). Beat: per-task images (over the snapshot cap) and building compiled repos per
  rollout (minutes). Revisit with per-commit in-place build artifacts if pandas/numpy are wanted in the RL pool.
- **Daytona probe = 3 shards × 12 h** (Luke: "no need for 6 shards"; Slurm QOS rejects > ~12 h). Beat: 2 × 20 h (rejected)
  and 6 × 10 h.
- **Freed fscratch by archiving old probe trace trees to /e/data1 rather than deleting** (01:20 PT 09-04), so the quota trip
  of 09-03 could not repeat mid-run; tables/configs stayed in place.

## 2026-09-04 (session 04b0b663, 21:20 PT 09-03 → 09:20 PT 09-04)

- **Overfit gate redefined (Luke, 00:25–00:35 PT):** reward must climb reliably toward saturation before entropy/reasoning collapse;
  entropy is the early warning, not the pass/fail; prune collapsed arms; run survivors long (20 steps, 12-h walls, no checkpoints).
  Beat: "entropy ≥ 0.5× step 1 at step ~8" as the sole criterion (too strict on the entropy bonus / Dr. GRPO, which raise entropy
  while reasoning dies, and blind to parse-fail collapses).
- **Concurrency cap: ≤ 5 sync arms + the band (Luke, 00:35 PT).** Beat: the 11-arm / 800-GPU sweep of 21:23–00:30 PT.
- **Knob ablations run at lr 1e-6, where the drift is visible in 4 steps**, with plain 1e-6 and 1e-6 + KL as controls. Beat: running
  them at 5e-7 (too slow to read) or on the band (seats).
- **Recipe for the band: lr 1e-6 + KL 0.01 + 16k tokens/turn (49k in) + RLOO-N + sequence_mean + TIS.** Beat: KL alone (drifts by
  step 9), 16k alone (gate at step 10), KL 0.03, 3e-6 + KL, n = 16 (one big first step, same slide), token_mean (no effect), Dr. GRPO and
  entropy bonus (entropy up, reasoning gone), behavior_clip (worst), prompt_mean / DPPO + prompt_mean / KL + prompt_mean (entropy runaway
  with parse-fail ≥ .2 by step 6–9).
- **Band geometry 16 policy + 24 engines × 22 = 528 seats, staleness 1 (band 1f).** Beat: 1,536 seats (bridge-client port ceiling +
  vLLM request timeouts; attempts a/b/c), 768 seats (borderline on the port ceiling), staleness 3 (band 1e collapsed in 9 updates
  because the buffer feeds ~3 updates per wave).
- **Fork fixes rather than config workarounds for the rollout-logprob checks** (cc284f6e, 8d02fd3a, 3c28d389): loss-mask logprob-less
  groups, keep their rewards in the baseline. Beat: `use_tis=false` on the band (loses the staleness correction) and masking
  ContextLengthExceeded as an infra exception (drops a legitimate reward-0 signal).
- **Mercor knobs ported into the fork (2d499e00) rather than run on upstream SkyRL.** Beat: switching the band to upstream (loses
  harbor parity, placement fix, TIS diagnostics).
- **No reward shaping without Luke** (his idea 00:58 PT): the data supports never-edit (28× lift), not `<<<SEARCH` tokens (0.97×);
  Mercor found length penalties neutral-to-negative. Beat: a shaped arm overnight.
- **JURECA renewal submitted a working day ahead** (15592091 at 06:00 PT for the 22:45 PT pool end). Beat: submitting at expiry.

## 2026-09-04 (session a915b13b, JUWELS pool + TaskTrove trace analysis, 09:12–12:45 PT)

- **JUWELS fleets are not kept warm (Luke, 11:40 PT):** let the seven fleets lapse; submit one 32-node fleet at 8 workers/node only
  when an arm is about to launch against bridge 9926 (queue wait < 1 min, ~18 k core-h per 12 h fleet otherwise wasted). Beat:
  refreshing at every expiry.
- **TaskTrove overlap is treated as the raw set on JSC; sympy is the only addition (confirmed on the full 3,009-task probe):**
  overlap 0.168 vs raw 0.176 mean success, same behaviour split. The "3 points easier" number is Daytona-vs-JSC runtime on identical
  tasks, not TaskTrove-vs-raw. Beat: re-selecting the overlap on TaskTrove's side.
- **Broken tasks are excluded, never repaired (Luke, 12:20 PT: "let's forget about fixing that").** 72 of the 735 overlap-band tasks
  are on the raw-pool quarantine list; only the git-history-leak class (1 task in the band) is repairable and it is not worth a
  harness change. The sympy band (360) still needs the model-free garbage-edit reward scan before use; one task already found
  (`r2egym-v1-06541`, truss, rewards a destroyed file). Beat: regenerating issue text / graded sets.
- **Ben's "500 copies of one task with many submission formats" warm-up: recommended against (12:30 PT, Luke did not object).**
  The habit to install (change the graded repo file, verify, stop) needs reward from real repo state; a format-reward set on one
  prompt narrows the policy without teaching the deliverable. Recommended instead: the existing 32-task sync overfit gate as the
  small first run, and a rejection-sampling SFT on the probes' ~1,200 clean winning trajectories before RL.
- **Second training arm = R2E band + SWE-smith, with SWE-bench Verified held out (Luke, 12:35 PT, "by SWE I meant SWE-smith").**
  Beat: training on Verified (it is the ID eval; SFT data was checked clean against it for this reason).
- **Evaluation is handed to the eval lead (Luke, 12:40 PT):** push the HF-format checkpoint to `laion/` **private** with the Snowball
  serving recipe (marin vLLM fork env, rotary override flags, 64k). Daytona eval-org quota is his. The ID legs run on Jupiter GPUs +
  Daytona sandboxes through the unified eval listener; JURECA/JUWELS pools are not part of eval. Beat: running the three legs ourselves.

## 2026-09-04 (session e3be22ba, 09:51 → 16:25 PT)

- **The overfit stage (Mercor Step 3) is closed; ablations happen on the band.** Luke (12:30 PT): "can we just move to the algo ablation
  stage without overfitting?" — yes; the 32-task arms answered what they could (reward climbs before collapse; lr sets the drift speed; KL and
  the 16k budget slow it; prompt_mean/DPPO/entropy bonus/Dr. GRPO are out) and the drift's real-distribution behaviour only has band answers.
  Pruned the last two sync arms on their own signals: KL + prompt_mean + 16k at step 6 (entropy 1.63×, parse-fail .35 — KL does not anchor
  prompt_mean), KL + n=16 + 16k at step 9 (reward .50 → .47 with entropy 0.45× and grad norm .05 → .07). No new sync arm will be launched.
- **Band 1f's recipe is the reference band recipe:** lr 1e-6, KL 0.01, 16k/turn (49k in), staleness 1, RLOO-N, sequence_mean, TIS, 528 seats.
  It beat 1e (staleness 3, 4k/turn) on the metric that matters — still learning at step 11 (reward .49, entropy 0.62×) where 1e was gated
  at step 9. Caveat recorded: 1f changed two knobs at once; if it keeps working we know the recipe works, not which knob did it.
- **"Use idle nodes, even one arm" (Luke, 10:58 PT):** a 24-node band hedge (`_f24`, 352 seats) was submitted 11:00 PT and cancelled unused
  12:17 PT when the 40-node 1f started first; its config is kept as a spare geometry.
- **The 16k per-turn budget is a training-time sampling knob, not a protocol change** (Luke asked whether it is cheating, 15:55 PT): the
  eval harness never enforces a per-turn cap (terminus-2 passes `max_tokens` only on summary calls; `model_info.max_output_tokens` is
  advisory), so the hard 4k cap in the sync arms was the mismatch and 16k is closer to eval. What must NOT be compared: the in-run step-24
  eval (49k in / 16k enforced out) vs the step-0 baseline (61k in / advisory 4k). The exported checkpoint gets re-probed at the standard
  budget instead.
- **Task-pool correction (Luke via the TaskTrove session, 12:40 PT):** the band is Marianna's raw `r2egym-raw-v3-train` filtered by
  Snowball's own probe; the Sep 2 direction covered her RL studies, not her task set. TaskTrove's r2egym is JSC-ready and 1,749 tasks are
  byte-identical to the raw set; "~3 points easier" is Daytona-vs-JSC runtime, not TaskTrove-vs-raw. Union band (~1,650) after the reward scan.
- **Seat-ceiling fix prepared, not deployed:** harbor `lukedhlee/bridge-keepalive` @ b99595d1 (thread-local `http.client` connection reuse +
  HTTP/1.1 server, kill-switch `HARBOR_BRIDGE_KEEPALIVE=0`, 13 new tests, Mac bench TIME_WAIT 16,928 → 1). Deploy needs a bridge restart
  between runs → Luke's call. Timeouts deliberately not retried (a retry could enqueue an exec twice).
- **Band arm 2 will carry three things Luke/Ben asked for:** the OT-Agent artifact store (`container.artifact_store.enabled: true` — one
  ext4 image per run mounted via fuse2fs on the batch host; our checkout has it, disabled), OT-Agent #141 (flush the async buffer before
  the wall), and the frozen-tmux fix Luke is expecting. Recommended recipe for arm 2: 1f at lr 5e-7 (every sync arm said step size sets
  how many updates precede the turn; MarinSkyRL's stochastic-rounded bf16 AdamW makes nominal 1e-6 ≈ 3× a classic 1e-6).
- **W&B moved to Luke's academic account** (entity `lukedhlee-marin`); Jupiter runs stay offline and are synced from the login node.

## 2026-09-05 (session 20ae8169, 02:00–15:30 PT)

- **Three concurrent 40-node band arms; replace, don't add** (Luke, 02:30 PT). A fourth needs his OK.
- **Prune on the turn signature, keep the last checkpoint before grad norm doubles.** Signature = grad norm ≥ 2× its plateau over 2–3
  steps with truncated fraction doubling (no-cap arms) or entropy rising (cap arms). Applied to arm 3 (30), arm 5 (23), arm 6 (25), the
  reset arm (4), arm 4 (34), arm 7 (23). Held-out probes lead the training markers by 2–6 updates, so pre-turn checkpoints are the only
  ones worth exporting.
- **Agent traces are retained** (Luke, 09:30 PT): the artifact store is compacted (`store_compact.py`, reaper every 30 min), never
  deleted; a 1 TiB image holds ~30 uncompacted steps.
- **Probes run at verifier timeout 600 s** (generator default from 11:10 PT); every arm carries the 30 s OpenAI connect timeout and
  masks the three uncatalogued exception types.
- **Diagnosis over more runs of the same kind** (Luke, 13:30 PT): three adversarial Opus audits (update path, sampler, data/env) instead
  of another lr sweep. They found the KL loss had no gradient, the sampler was truncated (top_p .95 / top_k 20) while TIS saw raw
  logprobs, and the band reward rose through "declare done" rather than competence.
- **Recipe changes adopted for new arms:** full-distribution sampling (top_p 1.0 / top_k -1), a working KL (fork be413fcb) at 0.04,
  `max_grad_norm` 0.1, per-epoch dataloader reseed, f/q decomposition metrics. Tested as: fulldist (sampling alone), mls (length-stop
  mask alone), best (all three). The stock DAPO overlong filter is rejected for Llama-3 templates (masks every sample).
- **Arm 7 was replaced before its own probe returned** (14:39 PT) because its KL was inert and it had turned; arm 4 was replaced at its
  turn. Real-KL-alone arm built but not launched (slot budget).

## 2026-09-05 (session 99a1d5ec, 15:31 → 23:00 PT)

- **Prune rule extended: entropy ≤ 0.5× of step 1 is a prune signal on its own** (21:15 PT). The length-stop-mask arm reached 0.31× with grad
  norm flat and its step-18 probe was back at base (+.014); the mask hides the grad-norm blow-up, not the damage. Grad norm ≥ 2× stays.
- **Arm mls pruned at step 18 (19:03 PT) and replaced by the KL-alone arm** (arm 3's recipe + working KL .04, truncated sampler kept) to
  separate the two fixes. Beat: keeping mls to see whether the mask alone delays the grad-norm turn (its held-out was already gone).
- **Arm fulldist turned off at step 30 (22:56 PT) on Luke's "turn off one arm"**: it had answered its question (sampler was the driver;
  clean past 30) and its s24/s30 probes run from exported checkpoints. Beat: turning off best (the hero candidate) or kl04real (the faster
  recipe if the KL alone holds).
- **Luke, 22:00 PT: accelerate the RL stack first, then experiment; forget the band re-probe for now.** Band/dataset fixes stay documented
  in the data audit but are not scheduled.
- **Keep the running arms to their verdict rather than restarting after band fixes** (22:10 PT, Luke asked): the collapse verdict transfers
  to any band; the band rebuild takes hours regardless; nothing gets relaunched on the old band after these walls.
- **Fleet resilience:** any bridge can be served from JUWELS (reverse forward `jwlogin03i:<port>`); a hop-submitted fleet must carry its env
  via `--export`; a fresh fleet may sit 10–20 min in Prolog/CONFIGURING before its log exists. Fallback fleets p22a/b and backup pool4b were
  cancelled unused once pool4 registered correctly.
- **Inode headroom comes from `experiments/` trees and caches, never from `tasks/`** (the raw-v3 trees are symlink farms into
  `tasks/r2egym-raw`; archiving it emptied both probes' datasets). Archived today: Currease gate tree, marianna_repro, cargo cache, triton
  cache (regenerates) → `/e/data1/mmlaion/lee27/archive_0906/`.
- **The data audit's empty-patch verifier test is moot**: every band task passed the JSC gate's pristine mode (unmodified image → 0) on entry;
  no-op wins need a hand audit of ~30 `win_no_src_edit` traces instead.

## 2026-09-06 00:50 PT — throughput session (5b45b53d)

- **`max_staleness_steps: 2` is the throughput setting of the band recipe, pending the step-18 held-out.** Measured on arm 1686334
  (best recipe otherwise identical): rollout wait 1–2 s every step from step 3 vs the baseline's 412/449/493 s on alternate steps,
  6.0 vs 3.8 updates/h, learning signals identical at equal steps. Beat: staleness 3 (cannot add speed once the wait is ~0, only
  older data — Luke's expert says 3 is fine for learning; still not the lever), more inference GPUs (engines were at 3–4 requests,
  4 % KV), raising seats first (blocked on the keep-alive deploy). If the s18 held-out matches best's, it goes into `make_snowball_grpo.py`.
- **Order of the next levers: grouped_mm bench → keep-alive deploy + more seats → node rebalance (engine nodes to the trainer)**,
  staleness revisited only if the wait reappears. Router replay is off and unsupported for Grug FSDP2; TIS (cap 2.0) is the
  correction and the live gap is 0.036 nats / 0.04 % capped, so it is not a lever.
- **Restart fixes ride harbor branch `lukedhlee/rl-transport` (9a7a9f53), NOT deployed into the staleness arm** — kept the A/B single
  variable. Deploy = checkout + `uv pip install --no-deps` into `envs/snowball` (non-editable install) + bridge restart between runs.
  The user's "wrap the bridge's urlopen in a bounded retry" became a connect-phase-only retry on top of the keep-alive client.
- **Storage root is `/e/data1/mmlaion/lee27` (Luke: "yes let's use mmlaion").** 38 finished run dirs / 16 TB moved with verify-then-
  symlink; fscratch 97 % → 63 %. Live arms, `exports/` (read by live val probes) and `_ray_logs` deliberately not moved. New runs
  still generate under the fscratch root — next launch must change `E=` in the generator/clone scripts.
- **Ben has been told (Discord draft on Luke's clipboard 23:40 PT)** that staleness 2 is being tested as a throughput lever with the
  TIS metrics as tripwire. Whether Luke sent it is unverified.

## 2026-09-06 06:15 PT — lock staleness 2 into the band recipe; carry grouped_mm; next lever is seats per engine node

Staleness 2 removed the rollout wait (6.0 vs 3.8 upd/h) with signals unchanged, and its step-18 held-out equals the staleness-1
best arm's (avg_score 0.1696 vs 0.1702, pass@8 0.4286 vs 0.4286, 224 tasks; probes 1689339 vs 1685928). Native grouped_mm
(`use_grouped_mm=true` on policy + ref) cut policy_train 446 → 87 s over four steps at identical signals (bench 1686910), so
every new band arm carries both. The 32-trainer-node rebalance is parked (needs train_batch_size ≥ 128; job 1689136) and is no
longer needed: the decode bench (1689698) showed 8 engine nodes at 66 seats each match 24 at 22 in groups/h, so the next arm
keeps 16 + 24 and raises seats to 1584 (clone ready, not submitted — 3x sandbox load on the shared bridge is Luke's call).
Beat: staleness 3 (adds nothing once the wait is ~0), the rebalance, and a lower-seat interpolation bench.

## 2026-09-06 00:15 PT — report held-out on the 191 clean validation tasks (session 3e3de7d6)

33 of the 224 validation tasks are byte-identical to band tasks and five times easier; every absolute held-out number was ~60 % relative too
high. From now on every paired table is quoted on the clean 191 (`EXCLUDE=$E/val_leaked33.txt python3 code/snowball/paired_delta.py …`);
probes still run all 224 so old and new probes stay paired on identical tasks. Beat: rebuilding the validation dir (breaks pairing with the
20 existing probes). Ranking unchanged; base .094, arm 3 s24 .177 on the clean set.

## 2026-09-06 01:30 PT — held-out probes keep the truncated sampler at eval; training samples the full distribution

The eval-sampler 2×2 (fulldist s30 and arm 3 s24, each scored truncated and full): truncation at eval is worth +.02–.03 to every checkpoint
with identical context deaths (per-token noise over ~30 turns), and under matched sampling fulldist s30 ties arm 3 s24 (−.008 ± .022).
Standard probe (top_p .95 / top_k 20) stays the judge and matches serving. Beat: switching the probe to the full sampler.

## 2026-09-06 05:41 PT — kl04real pruned by rule at step 37; fulldist resumed from ckpt 30 into its slot

Grad norm doubled over five rising steps (.047 → .092) with entropy 1.75× and rising — the drift-up turn. Slot went to the arm with the
best survival + transfer record (fulldist, tie with arm 3's peak at 30). Beat: a fourth slot (Luke's three-arm rule), replacing best.

## 2026-09-06 07:55 PT — prune rule gains an entropy-rising clause

Entropy ≥ 1.3× of step 1 and rising three steps running is a prune signal like ≤ 0.5×: kl04real was at base on held-out by step 30 while
every training metric read clean; its grad norm doubled only at 37. The held-out probe leads the training side by ~6 steps on every arm so far.

## 2026-09-06 09:45 PT — all recipe arms stopped for the faster stack (Luke)

fulldist (step 38, ckpt 36) and best (step 53, ckpt 48) cancelled; resume later on the staleness-2 + grouped_mm stack. Recipe to carry:
full-distribution sampling, KL off or ≤ .01, lr 5e-7 — the only recipe that survives past 30 and transfers. KL .04 in any combination is out
(blocks transfer; alone it only delays the turn).

## 2026-09-06 11:30 PT — the apptainer prompt gets TaskTrove's workflow body; the verifier is hardened first (Luke's spec)

The band arms trained on a 43-word header + issue because `build_tt_raw.py` rebuilt the instruction from the upstream parquet and dropped
TaskTrove's five-step body. New prompt = our header + TaskTrove's body from `<uploaded_files>` down, Environment Setup block excluded
(`data/r2egym/jsc/tt_prompt.py`; `build_tt_raw.py` emits it). Before shipping it, `tt_raw_template/test.sh` runs the graded pytest through
`/tests/safe_pytest.py` (PYTHONSAFEPATH is 3.11+ only; images are 3.7-3.9) and regenerates a deleted `run_tests.sh`. Beat: probing with the
old verifier and eating the reproduce_issue.py-in-/testbed zeros.

## 2026-09-06 12:00 PT — the re-probe is a paired A/B with a header-prompt control, 100 tasks per stratum, repo-aware

tt60k (the only base probe of these tasks) ran on the frozen-screen harness, so a workflow-only re-probe would confound the harness fix
with the prompt. Both arms (`tthd` header, `ttwf` workflow) run the same 300 tasks on today's harness and the hardened verifier; strata
from the base model's own pass@8 (0/8, 2-5/8 in the band, 6-8/8), repo-proportional. Luke: subset not the whole band, 100 per stratum,
repo-aware, minimise wall time, Opus readers for the traces. Beat: 50 per stratum (first draft); the whole 993-task band (~5x the cost).
Sandboxes on a dedicated bridge (9924, six JUWELS fleets) so the parallel session's 1,584-seat throughput bench on 9922 is untouched.

## 2026-09-06 13:40 PT — the workflow prompt becomes the default apptainer prompt; no arm is re-run for it

Paired probe on 300 tasks (`experiments/2026-09-06_workflow_prompt_probe.md`): the workflow prompt does not move pass@8 (−.014 [−.037,
+.007]), context deaths (72 → 74 %) or P(win | done) (.87 both); it does change the procedure (two thirds write and run the repro before
editing; done-after-repo-tests 48 → 59 % of wins; done-on-own-script-only 12 → 4 %). So: `build_tt_raw.py` ships it for every future tree
(better-evidenced "done" is the habit RL should start from), but the band arms are not relaunched on its account — the reward surface
and the context ceiling are the same. Beat: re-probing the whole band (would have said the same at 5x the cost); keeping the header
(nothing gained by the omission either). Verifier hardening stays regardless (17 wins in this probe would have been zeroed without it).
Amendment 14:05 PT (after the Opus readers): two clauses of the shipped body are traps on R2E-Gym — "run the repo's existing tests" and
"you don't have to modify tests" — because the repos' visible tests still assert the pre-fix behaviour while the graded tests are hidden
(three of nine read losses: reverted a correct fix on XFAIL noise, deadlocked on a contradicting visible test, declared done on a
non-covering green suite). Recommendation to Luke: qualify both clauses and require the repro to assert the issue's stated value before
the prompt enters any recipe. Not changed unilaterally — the spec named those clauses.

## 2026-09-06 15:05 PT — the fixed-clauses workflow prompt is the default; no more prompt probes; next is the arm

Fixed clauses (hidden tests grade you, the repo's tests may encode the old behaviour and are not evidence; assert the issue's stated
value; rerun after every edit) on the 100 band tasks: +.020 [−.028, +.068] over the header, P(win | done) .83 → .91 at the same tc_rate,
context deaths 76 → 74 %; the deadlock trap task went 3/8 → 8/8 with the clause visibly doing it. That is the environment cleaned as far
as a prompt can clean it. Beat: re-probing at pass@16 (no decision hangs on ±.02); leaving the verbatim body (traps). Remaining
misleading signal (visible suites that contradict the gold patch) is a task property → the model-free scan tags it if the next arm's
P(win | done) drifts; the next arm carries the verification penalty and logs P(win | done) per step.
Amendment 15:20 PT (trap reader): the rewrite closed the deadlock trap and the assertion trap; the remaining trap-task losses are edit
mechanics (no file-edit tool), not wording; 05754-type tasks (visible test contradicts the gold patch) get tagged out via the model-free
scan when it runs. Two further wording lines applied unprobed (graded tests not on disk / stop editing once the repro passes) — judged by
the next arm, revertable in one commit. The reader sees no verifier gaming, so the "done without rerun" penalty stays a proposal to be
decided at arm design, not a done deal.

## 2026-09-06 15:10 PT — staleness 3 at 1584 seats is worth ~15 %, not the 50 % the group-cap arithmetic promised

Bench 1697973 (x16, gmm, 1584 seats, staleness 3, 8 steps): steps 2–8 average 484 s (248/267/407/1028/500/452/487) vs the
staleness-2 arm's solo steps 2–8 average 556 s → 7.4 vs 6.5 upd/h. More groups were live (peak running 945–1433 vs 544–1057) but
each turn got slower with occupancy (e2e 11.5–16 s vs 10–15 s) and stale rejects rose to 6–8 per step. Signals matched (reward
.24–.32, entropy .55–.57). The cohort wave persists at staleness 3 (a 1028 s trough at step 5). Verdict: a real but small systems
gain that buys one more step of policy lag; not applied to the new arm by default — Luke's call, since it changes off-policyness and
the s18 held-out check covered staleness 2 only. Beat: the harness-fix bench (1698599) decides the stack for the new arm.

## 2026-09-06 15:55 PT — the harness fixes did not move step time; the seats lever is capped by admission dynamics, not client CPU

Bench 1698599 (harbor overlay 4886eb7e: incremental token count + orjson trajectory writer + argv-safe batch; 12 coordinators; else
the arm's config): steps 2–8 average 530 s (228/566/327/638/457/646/845) vs the arm's 556 s → ~5 %, inside the wave noise. Signals
clean (reward .22–.29, entropy .55–.58, 0 failed/masked); 55 verifier timeouts over 4,096 trajectories while sharing the fleet with
the staleness-3 bench. Reading: the coordinators were CPU-hot (tiktoken, trajectory JSON) but never saturated (~50 %), so cutting
their CPU did not shorten the turn; the per-turn "harness" seconds sit elsewhere (endpoint/LLM request path or vLLM queueing) or the
wave/group-cap dynamics dominate the step average. The fixes stay in (correctness-neutral, cheaper), applied to the new arm
1699514 (fulldist recipe resumed from step 36). Beat: paired per-turn trace timing across the three finished stores, then the
structural options — batch 128 groups (recipe) or trajectory-level/staggered admission in the dispatcher (systems).

## 2026-09-06 16:10 PT — train / val split inside TaskTrove first; validate existing checkpoints before any new data (Luke)

Split (`tt_split.py`): oodval = band's tornado + scrapy (115), idval = 150 band tasks stratified by repo × success, train = 728, and
`heldout` = 160 never-solved (band repo mix) + 16 always-solved, disjoint from everything trained so far. Reason for the last: the band
holds every "sometimes solved" task, so the existing checkpoints can only be validated on tasks the base never solved (or always
solved). Probing base / arm 3 s24 / fulldist s36 on `heldout` now (three 12-node probes); the next arm trains on `train` with idval +
oodval + heldout probes. Marianna's set is not added (1,749 byte-identical to TaskTrove; the rest lower-scoring); her clean-191 stays as
the bridge to old numbers. Beat: a fresh training run first (cannot answer "does RL transfer within distribution" faster than three
probes); carving an ID val out of the band for the existing checkpoints (they trained on it — that would measure memorisation).

## 2026-09-06 17:10 PT — measured turn split: 80 % LLM call, 20 % exec, 0 % client; the generator loses time between turns, not inside them

First 301 trajectories of the instrumented arm 1699607 (harbor 26c307fc timers in metrics.extra, fill phase, all 1536 live):
turn mean 24.2 s = LLM call 19.5 s (80 %, vs vLLM's own e2e 17.8 s → ~1.7 s of endpoint/routing) + exec 5.0 s (20 %, of which
2.1 s the agent's declared sleeps) + 0.0 s parsing/token-count/loop. The "9 s harness" of the morning was an inference artifact;
the per-turn client CPU was never on the critical path, which is why the three harbor fixes moved nothing. Implication: at full
occupancy a turn is decode physics (1,100 tokens at ~50 tok/s per request), so throughput is set by how many trajectories are live
on average — the admission/cohort dynamics (group cap, staleness budget, wave). Levers, in order: (1) admission at trajectory
granularity or staggered cohorts in the dispatcher (systems), (2) batch 128 groups (recipe, Luke), (3) more requests per GPU only
once (1) or (2) keeps the seats full. Steady-state re-measure queued at step 7.

## 2026-09-06 17:45 PT — batch 128 is the next A/B, on its own bridge; the arm restarted once for instrumentation

With the turn split measured (17:10 PT entry) the remaining lever is how many trajectories are live, so Luke approved a batch-128
bench (train + mini batch 128, 3,168 seats = 32 requests per GPU, `max_num_seqs` 48, 8 steps, resumed from fulldist step 36 like the
arm). It runs on bridge 9927 with four 100-node JUWELS fleets rather than sharing 9922/pool6 with the arm: the bridge peaked at 67 %
CPU under two full-seat jobs and the arm's verifier timeouts rose when the fleet was shared. Earlier (16:15 PT) Luke approved
restarting the fresh arm (10 min old) so every trajectory carries the per-turn timers (harbor 26c307fc): job 1699514 → 1699607.
Beat: running the bench alongside the arm on 9922, or after the arm.

## 2026-09-06 18:00 PT — ID transfer confirmed on never-seen same-repo tasks; the next arm is green-lit, data additions wait behind it

Base .027 → arm 3 s24 .061 (+.034 [+.016, +.055]) → fulldist s36 .058 (+.030 [+.016, +.045]) per-task pass on 160 held-out never-solved
TaskTrove tasks (band repo mix, fixed prompt, hardened verifier, never trained on); unlocked 22 → 35 → 41; sympy 11 → 16 → 21 of 70;
always-solved 16 tasks .85 → .89 → .91. So the band's RL gains are not memorisation and not confined to the raw val's repo mix. Next:
the arm on `r2egym-tt-v2-train` (728) with idval / oodval / heldout probes every 6 steps, fixed prompt, fulldist recipe on the fast
stack; P(win | done) logged; then and only then SWE-smith or other data. Beat: adding data now (the gate had not been checked).

## 2026-09-06 17:45 PT — history-think probes: does Snowball need its earlier turns' reasoning re-fed? (session f4a2c148)

The Stage-3 SFT re-fed every earlier assistant turn's think span and scored every turn under that context (Opus audit of the Marin
template `experiments/marin_tokenizer.py:96-99`, Levanter `formats.py:212-241` / `datasets.py:482-484`, corpus spot check); the teacher
that produced the corpus (DeepSeek-V3.2) strips prior reasoning from history by its own chat template, so the targets were generated
without it. Nobody in marin-community has recorded the teacher mismatch or discussed dropping re-feeding for Snowball; re-feeding is a
deliberate parity feature of Ben's (marin #6341, MarinSkyRL #330). Luke's call: probe it. Harness flag on the harbor fork
`lukedhlee/rl-transport` @ e6adddd8 (`HARBOR_TERMINUS2_HISTORY_THINK=keep|drop|last:N`; Chat.request_messages + PriorThinkStripper;
recorded history untouched; token count follows the request). Overlay `code/harbor_overlay/e6adddd8`; launcher
`data/r2egym/jsc/probe_history.sh`; read-out `hist_readout.py`. Probes on `tasks/r2egym-tt-v2-val441` (idval 150 + oodval 115 +
heldout 176), Stage-3 base, 64k parity budget, k 8: `snowball_hist_{keep,drop,last2}_base` = 1699824/1699825/1699826, bridge 9926,
JUWELS fleets hist{a,b,c} 14229634-36. Decision rule: drop holds pass@8 within ~.04 with fewer context deaths → RL runs under the
dropped contract (band re-derived first); hard drop → last:2 fallback; both fail → back to the SFT side (Ben).
Same evening, Luke's addition: probe the base on three TaskTrove synthetic sources Ben's 08-27 report scored (curriculum-easy .531,
pymethods2test-v3 .496, unitsyn-python-v4 .544; first 300 rows, TaskTrove-verbatim instruction, one SIF each) under keep and drop, after
an apptainer gate and adversarial Opus audits of each source. Prompt rule (prompt-reader agent on artifact cef36083): the r2egym trees'
fixed-clauses body is the approved prompt; the "weird prompt" to avoid is the 43-word header-only prompt; synthetic sources ship their
own instruction.md untouched.

### 2026-09-06 18:20 PT — history-think probes, first-trial verification (interim)
- Drop mode works as intended: on the seven tasks finished under both arms, the prompt at turn 10 is 13.4k tokens under drop vs 20.2k under keep (paired by task); episode input tokens 153k vs 220k median. Recorded histories keep their think spans (the stripper only touches the request).
- New mechanism, not a bug: without re-fed reasoning the model stops thinking. Its first completion token is the think-start token (id 128002) on 100 % of turns under keep at every depth, but under drop the share falls with depth (turns 0–1 100 %, turns 6–8 ~85 %, turns 10–13 ~65 %, turn 15+ ~50 %, turn 23 ~17 %); it emits the JSON action directly. last:2 drifts the same way (turn 15 ~64 %, turn 21 ~27 %). So "drop" at inference is also "stop thinking late in the episode"; the read-out must separate the window effect from the thinking effect (P(win|done), turns, completion tokens by depth).
- Inode reality: reformo project quota is 8.0M soft / 8.8M hard and usage was 8,085,161 at 16:20 PT (over soft; writes work while the GPFS grace holds). Monitor now reads jutil + a touch probe; the old df -i monitor read the filesystem total and was useless.

### 2026-09-06 18:15 PT — easy-source audits landed; six easy probes launched
- Stager: all 900 tasks share one image (3c7138b002df), gate 9/9 pristine 0 / oracle 1, trees at /e/fscratch/reformo/lee27/tasks/tt-easy3/<source>. Six probes 1699924-29 (keep/drop per source) launched 18:08 PT on fleets 14229645-50 (bridge 9926, 8 h).
- Three adversarial audits (summary in the session scratchpad easy3_audits.md; exclusion lists on Jupiter under experiments/easy3_audit/): none of the three sources is usable as an RL reward. curriculum-easy: 77/300 defective, no oracles exist, verifier pays a stub via os._exit or a planted conftest. pymethods2test-v3: 57/300 provably defective, os._exit pays 500/500, instruction paraphrase contradicts the grader. unitsyn-python-v4: an always-equal object wins 223/300, above every model's score; recommended drop. Ben's .531/.496/.544 measure instruction fidelity, not agent skill (six models within .03-.04).
- Probes kept running as asked: they give the apptainer-vs-Daytona parity check and the keep/drop contrast on short tasks; the read-out reports the full 300 and the audited-clean subset.

### 2026-09-06 18:50 PT — bridge outage under nine probes; easy probes rescheduled
- 18:30–18:45 PT: once the six easy fleets registered (2,304 seats on bridge 9926) and their probes ramped, the bridge (harbor apptainer server.py, ThreadedHTTPServer, thread per request) hit the login node's 4,096-pid cgroup cap ("RuntimeError: can't start new thread", BrokenPipe storms in server_9926.log); attempts on all nine arms failed with BridgeOutageError / BridgeOperationError (the r2egym trio lost most of its 18:30–18:45 attempts as nulls).
- Action: cancelled the six easy probes (1699924-29) and fleets (14229645-50) at 18:44 PT; bridge recovering with 768 seats + 3 probes (the load that had run cleanly for 45 min). Aborted easy trees deleted.
- Rescheduled: tmux easy3_after_trio on Jupiter waits for the trio to table, reads it out, then launches the six easy probes as snowball_easy2_<src>_{keep,drop}_base on the EXISTING 768 seats (no new fleets), reads out each pair (full 300 and audited-clean). Capacity rule learned: keep (workers + probe clients) well under ~3,500 on one login node per bridge; 9 probes x 256 + 2,304 workers is over the cap.
- Trio damage tally and relaunch decision pending the recovery watch.
- 18:55 PT damage tally (trio, ~40 % through): null attempts 18 % / 27 % / 26 % of those finished so far (keep/drop/last2), clustered by task group (16–25 tasks with zero scored, 61–72 tasks with ≥1 null among ~185 seen). Decision: no relaunch. Sequencer (tmux hist_sequencer, trio_topup_then_easy.sh) reads the trio out at tabling (strict 8 and relaxed 6), builds a top-up tree of every val441 task any arm left under-sampled, runs three top-up probes (keep/drop/last:2) on the existing seats, then the merged read-out (base+top-up per arm), then the six easy probes. Expected: trio tables ~21:00 PT, top-ups ~21:50, easy pairs ~23:00, report after.
- 19:23–19:26 PT second burst: the bridge reaped the dead fleets' sandboxes 2,400 s after they died (BRIDGE_STALE_READY_SEC) and ~300 in-flight attempts across the trio ended in BridgeOperationTimeoutError; clean again by 19:30. Lesson: after killing a fleet, its "ready" seats stay dispatchable for 40 min; drop the stale window or drain before killing. Trio at 19:39 PT: 2,600 / 2,423 / 2,337 of 3,528 attempts; tabling ≈ 20:45 PT; the top-up covers both bursts.
- 21:10 PT Luke: cancel the easy-source probes (the audits disqualify all three sources; no GPU hours on them). Sequencer trimmed to trio read-out + top-up; no easy2 job was ever launched. The easy-source read-out in the report is the audit verdicts only.
- Path note (Luke asked): tonight's probe outputs, tables and in-place trace tars are on /e/fscratch/reformo/lee27/experiments because the probe launcher (make_snowball_probe.py / probe_ckpt.sh) still defaults there; task trees are on fscratch by Luke's pointer. Nothing from tonight is under /e/data1/mmlaion/lee27. To do after tabling: move tars, tables and read-outs to mmlaion and fix the launcher default.

### 2026-09-06 22:10 PT — history-think probe result (from probe_watch pass8 tables; tar read-out with curves still running)
Mean per-task pass over fully sampled tasks (paired delta vs keep, bootstrap 95 %):
- idval 150: keep .392 | drop .398 (+.008 [-.026,+.046]) | last2 .415 (+.025 [-.008,+.058]); ctx death .67 / .59 / .59; med turns 24 / 32 / 32
- oodval 115: keep .394 | drop .432 (+.036 [-.002,+.075]) | last2 .399 (+.010 [-.026,+.046]); ctx death .57 / .49 / .52
- heldout 176: keep .106 | drop .111 (+.006 [-.007,+.020]) | last2 .122 (+.016 [+.004,+.028]); ctx death .81 / .78 / .76
- all 440: keep .278 | drop .290 (+.014 [-.002,+.031]) | last2 .293 (+.018 [+.002,+.033]); ctx death .70 / .64 / .64; turns 24 → 33
Reading: re-feeding prior reasoning is not load-bearing for the base model; stripping it costs nothing on any split and cuts context deaths by ~6 pts, which the model spends on ~9 more turns rather than on many more wins. last:2 is never worse than keep on any split and edges drop on idval/heldout; drop edges it on oodval; drop vs last2 is within noise. Caveat carried from the first-trial check: under drop and under last:2 the model stops emitting the think token as episodes deepen (~50 % by turn 15), so RL under either contract will shape whether the policy thinks at all; the curves and P(win|done) by think-share come from the tar read-out.
Decision rule outcome: drop holds pass@8 within .04 with fewer ctx deaths → RL under a stripped contract is admissible; recommendation last:2 (keeps the recent chain, same window saving, no split regression) with drop as the alternative; band must be re-derived under the new contract.

### 2026-09-06 22:40 PT — Mercor Step 3 under the new contract: snowball_sync_last2_lr1e6 (job 1700921)
Luke: "can you do the overfitting run now". clone_sync.sh from snowball_overfit32_a (8 policy nodes fsdp32 + 8 engines x 32 seats, 32 tasks x 8 samples, sequence_mean, no KL, staleness 0, use_tis on), lr 1e-6, 10 steps, plus trainer.policy/ref.fsdp_config.use_grouped_mm=true, APPTAINER_BRIDGE_URL pinned to 9926 (hist fleets, expire 03:42 PT), and the harbor overlay e6adddd8 + HARBOR_TERMINUS2_HISTORY_THINK=last:2 inserted before the launch line (insert_overlay.py in code/snowball). Gate: reward climbs toward saturation on the 32 tasks before entropy/reasoning collapse, AND the contract plumbing holds end to end in training (sequences = stripped prompts + completions, one sample per turn; masks/lengths sane at the first update). New health check to add: think-share in rollouts by step (first completion token == 128002). 16 nodes, wall 06:00:00.

### 2026-09-06 22:25 PT — second audit wave (Ben's v4.8 shortlist), first verdict
- laion/exp_rpt_bugsinpy-v4 (479 rows, TaskTrove commit 150c46e6): NOT usable as RL reward. Not repo bug fixes: one LLM-rewritten solution.py + LLM tests per task, no reference solution anywhere, grader leaked to the agent (setup_files/test_solution.py == tests/test_solution.py, 479/479). Verifier greps its own pytest stdout inside the agent's container (restart_environment=false): a fake /app/pytest.py printing "4 passed" scores 1 on every task; conftest/pytest.ini plugin routes pay too; a solution that raises at import writes no reward (zero can be deleted). 39 tasks already fixed in the starter, 51 instructions name a missing file, 119 hinge on one test. Offline clean. exclude.txt 46 ids; files in scratchpad adv2_bugsinpy/.
- Pending: multifile-v3, scaffold-v3, nl2bash-oracle-v2.
- laion/exp_rpt_scaffold-v3 (3,121 rows): NOT usable as shipped, but rescuable by one shared file. Content clean (0 duplicates, 0 solution leaks, pristine/untouched starter 0 everywhere). Verifier is the tonight-standard defect: restart_environment=false, PYTHONPATH=/app, exit code only, grader copy visible to the agent; os._exit(0) via /app/sitecustomize.py or /tests/conftest.py pays 450/450, starter + two atexit lines pays 271/300 with "11 failed" in the log. Hardened verifier = restart_environment=true + /app off the grader path + junitxml parsed against a baked-in test count, computed outside the container (we can apply this ourselves when building the apptainer tree, as done for r2egym). Limits: 117 offline-impossible (flask/seaborn/cv2, unmocked network), 361 multi-module starters with one named path; not terminal-shaped (3.3 % subprocess, 2 % argv) so no TB2 transfer claim. exclude.txt 179 ids; files in scratchpad adv2_scaffold/.
- DCAgent/exp_rpt_multifile-v3 (4,843 rows): NOT usable as shipped; content the cleanest of the four (empty workspace 0/4843, no duplicates, 96.5 % of instructions name every imported symbol, naive stub 0/300). Same fatal harness: pytest exit code inside the agent's container; os._exit(0) in the requested file 29/30, /tests/conftest.py 30/30, sitecustomize/.pth 30/30, stealth conftest 29/30 with "11 passed" in the log. Auditor verified in harbor worker.py that /app, /tests, /setup_files are rw binds on JSC and the test upload is `tar xf -C /tests` without a wipe. Graded tests shipped to the agent verbatim (test-driven source). No oracle anywhere; 42 non-discriminative, 11 offline-unwinnable, 74 under-specified. exclude.txt 54 rows; files in scratchpad adv2_multifile/.
- Shortlist state: bugsinpy rejected; scaffold + multifile rescuable by one verifier change we control (fresh instance for verify, wipe /tests before upload, /app off the grader path, junitxml vs baked-in test count parsed outside the container) ≈ 8k clean tasks; nl2bash pending.
- DCAgent2/nl2bash-tasks-cleaned-oracle-v2 (1,498 rows): NOT usable; exclusion cannot rescue it. Grader = subset match of one captured output file (order ignored 983/983, extras free 1497/1498, 202 tasks with a provably droppable sort/head/uniq); "oracle-v2" solutions call /tests/setup_seeds.sh which no task ships → oracle 18/300 as shipped, 227/300 repaired; 93 % of instructions tell the agent to invent fixtures whose exact contents the expected output encodes; ~1/3 of a 60-task read unwinnable or coin-flip. Clean: 0 TB2 overlap (90 tasks), low duplication, no leaks. Snowball's 5x over Coder most likely learned output protocol. Files in scratchpad adv2_nl2bash/ (REPORT refused by a harness rule; findings in the agent's final message, ledger here).
- TWO GENERALISING HOLES (ours to fix, harbor apptainer path): (1) harbor reads /logs/verifier/reward.json before reward.txt and test.sh never writes it → planting reward.json pays 1 on any source whose test.sh writes only reward.txt (our r2egym verifier included) — grep the band arms' trajectories for writes under /logs/verifier; (2) restart_environment is a dead key on apptainer, every verifier runs in the agent's container → worker fix: fresh instance for verify, wipe /tests and /logs/verifier before upload, /app off the grader path, absolute-path interpreter, junitxml vs baked-in test count. This fix rescues scaffold-v3 + multifile-v3 (~8k clean easy tasks).
- Shortlist final: bugsinpy rejected (content), nl2bash rejected (grader), scaffold + multifile usable after the worker fix.

## 2026-09-06 23:15 PT — RL stays on the re-feeding contract (`keep`); the dropped contract passes the rule and is still the wrong trade

The three history-think arms tabled (`experiments/2026-09-06_history_think_probe.md`; report artifact 689872ed). `drop` clears the rule
Luke set — per-task pass +.014 [−.003, +.031] paired on 421 tasks, context deaths 77.9 → 72.7 % — and I am recommending against it anyway,
because that +.014 is the residue of two much larger effects: −1.9 pts from the model losing its own reasoning, +5.2 pts from the freed
window, −2.3 pts that `keep` recoups on context-death attempts whose repo is already fixed. The price of the +.014 is +44 % turns
(24.4 → 35.1) and +48 % prompt tokens per rollout (667k → 990k), with win efficiency 11.35 → 8.19 wins per 1,000 turns; the extra 10.7
turns are 17 % productive, 41 % idle, 41 % more searching. In a generator-bound stack that is a throughput regression bought with a gain
whose interval touches zero. Beat: adopting `drop` on the rule as written (the rule measured a cancellation, not an equivalence);
`last:2` as a fallback (it recovers 93 % of the token saving and none of the thinking — a replicate of `drop`, not a middle setting).

Mechanism, which is why this generalises: stripping the spans makes the model **stop thinking**, and the cause is self-imitation, not long
context. `keep` holds 95.7 % thinking turns past 56k tokens of prompt, so context alone never stops it; inside `drop`, depth predicts the
decay ~3× more strongly than prompt length. What it loses is **pending intentions, not facts** — facts are in the terminal and the terminal
is re-fed; a decision made and not yet executed lives only in the think block. Signature: reverting its own working fix to run a control
experiment and never restoring it (restore commands 0.97 → 1.74 per attempt; repeated-command turns 2.0 → 8.9 %; 10,073 of drop's in-loop
turns are no-think against 176 that think). It never compensates — zero drop attempts of ~3,520 wrote a notes file and the `plan` field
gets 30 % shorter.

Next experiment, **not run**: a placeholder mode on the same `HARBOR_TERMINUS2_HISTORY_THINK` switch that replaces each old think span with
a short marker or one-line summary instead of deleting it, so every prior turn still visibly reasons at near-zero token cost. If the
self-imitation reading is right that gives `drop`'s turn count at `keep`'s thinking rate — the +5.2 without the −1.9. Hypothesis, not result.

Two items that are not this workstream's: (a) **for Ben / the SFT side** — Marin re-feeds prior reasoning but DeepSeek-V3.2, the teacher
that produced the Stage-3 corpus, strips it by its own chat template, so the targets were generated without the history the template always
supplies; nobody in marin-community has recorded the mismatch and this probe does not resolve it, it only shows the model is now dependent
on the re-feeding side of it. (b) **for harbor** — the loops in every arm start when the terminal is not at a shell prompt (open heredoc,
pager) and the observation does not say so, which makes "send Ctrl-C and look" a locally reasonable fixed point.

Data hygiene note: tonight's outputs went to fscratch because `make_snowball_probe.py` / `probe_ckpt.sh` still default there. The analysis
extracts are on mmlaion (`experiments/hist_analysis/`); the three trace tars (1.05 TB total) are still on fscratch and need moving, and the
launcher default needs fixing. Nothing deleted without Luke.
- 23:00 PT Luke: "launch another overfitting run with staleness 2 to accelerate our findings, in parallel" → snowball_async_last2_lr1e6_stale2 (job 1701112, 16 nodes): identical to 1700921 except trainer.fully_async.max_staleness_steps=2; same bridge 9926 (relief fleet 14229714, 32 nodes/12 h, submitted so seats outlive the hist fleets at 03:42 PT). Read the pair together: reward climb + think-share by step under last:2 at staleness 0 vs 2.
- 22:55 PT the other session's report (artifact 689872ed, "Snowball loses its plans, not its facts") decomposes the +.014: window +5.2 pts, lost thinking −1.9, keep's ctx-death free passes −2.3; win efficiency −28 % (wins per 1,000 turns), self-imitation mechanism (think share falls with depth, not context fill; last:2 ≈ drop), drop loses pending intentions (revert-and-never-restore, repeat rate 2.0 → 8.9 %). Its recommendation: RL under keep; test a placeholder contract. My revised position: adopt it; next experiment = head:N mode (keep the first ~300 chars of each prior think span), one probe before the hero run; do not launch the hero under last:2.

### 2026-09-06 23:10 PT — placeholder contract probe (Luke's request)
- Harbor fork 014e7562: HARBOR_TERMINUS2_HISTORY_THINK gains `placeholder` (each earlier span re-sent as `<|start_think|>\n[reasoning omitted]\n<|end_think|>` + action) and `head:N` (first N characters of each earlier span); HistoryThinkMode dataclass; 215 tests pass. Overlay /e/project1/transfernetx/lee27/code/harbor_overlay/014e7562; probe_history.sh accepts the modes.
- Probe snowball_hist_placeholder_sub160 (job 1701150, 12 nodes, bridge 9926 + fleet 14229715) on a stratified 160-task subset of val441 (60 idval / 50 oodval / 50 heldout, seed 20260906; list $E/hist_sub160.txt, tree tasks/r2egym-tt-v2-sub160). Hypothesis (report 689872ed): window saving of drop with keep's thinking rate. Read-out: tmux placeholder_readout → $E/hist_readouts/hist_placeholder_readout.md (paired vs the trio on the same 160 tasks). Not launched: head:300 (same subset) — one command when seats free up.
- 23:20 PT correction (Luke pointed at Ben's v4.8 three-model panel, https://storage.googleapis.com/marin-public/benjaminfeuer/tasktrove-v48-three-model-reward/2026.09.01/index.html): the audits were adversarial by instruction and measure gameability + instruction/test consistency; the panel measures honest-model reward with infra failures excluded, and it shows wide monotone gradients (bugsinpy .30/.73/.99). "Rejected on content" for bugsinpy was too strong. Revised shortlist: harden our verifier (fresh instance for verify, wipe /tests + /logs/verifier, /app off the grader path, absolute python, junitxml vs expected count), apply the exclusion lists, then bugsinpy + scaffold + multifile are RL candidates (~8.5k tasks); nl2bash stays doubtful (subset-match grader, missing seed script, protocol-dominated spread). Gameability is a risk under optimisation pressure, not a property honest models reveal.

### 2026-09-07 00:05 PT — placeholder probe interim (607/1280 attempts) + head:300 launched
- Placeholder keeps the model thinking: think share .99 at turn 10, .98 at 20, .96 at 30, .85 at 40 (drop .75/.43/.18/.07; keep ≈.99 throughout) with drop's prompt sizes (14.2k at t10, 28.4k at t20). Median turns 30.
- Caveat: the fixed note is copyable. Recorded turns containing "[reasoning omitted]" rise with depth: 0.1 % (t0-9), 0.9 % (t10-19), 3.2 % (t20-29), 6.8 % (t30-39), 11 % (t40-49), 28 % (t50-59). Of 616 echo turns, 253 continue with real reasoning, 53 are bare echoes with junk tokens then the action, 310 never close the span (wasted turn). Much smaller than drop's collapse, not zero → head:N (first 300 chars of each real span, nothing fixed to copy) launched as snowball_hist_head300_sub160 on the same subset. Five-arm sub160 read-out waiter: tmux sub160_readout → hist_readouts/hist_sub160_five_readout.md.

### 2026-09-07 00:50 PT — behavior_clip task (Luke's pasted TASK), steps 1–3 as far as the connection allowed; NO launch
- Step 1, live arm (snowball_ttband_lr5e7_stale2_gmm_seats1584_x16_a, mmlaion config): policy_loss_type unset → default `regular` (ppo_base_config.yaml:204); use_tis=true; tis_imp_ratio_cap=2.0; eps_clip_low=eps_clip_high=0.2; max_staleness_steps=2; rloo_n; kl_loss_coef 0.04; sequence_mean; lr 5e-7.
- Premise verified on that arm's 17 train steps: policy/ppo_ratio_exact_unit_fraction = 0.997–1.000 every step, policy/log_ratio_abs_mean ≈ 1e-5, ppo_clip_ratio ≤ 2.2e-5 → the PPO ratio exp(log_probs − old_log_probs) is identically 1 under fully-async (old_log_probs recomputed on current weights); the 0.2/0.2 clip has never bound.
- Our MarinSkyRL fork (be413fcb): `behavior_clip` exists (policy_losses.py:340, AReaL-style: ratio = exp(log_probs − rollout_logprobs), clamp to [1−eps_low, 1+eps_high], pessimistic max, dual-clip c on A<0); it raises on use_tis=true (utils.py:746, policy_losses.py:357) and forces sampling_params.logprobs=0. No assertion against `regular` under fully-async in our fork (fully_async_trainer.py TODO at ~530); TIS is asserted to `regular`/`dual_clip` only.
- Step 3 evidence available from logs: rollout-vs-train spread on that arm = tis/log_ratio_abs_mean 0.035–0.039, tis/imp_ratio_capped_fraction (ratio > 2.0, i.e. log r > 0.69) 4.0–5.1e-4, tis/imp_ratio_mean 1.0000 (staleness 2, lr 5e-7: the mismatch is mostly vLLM-vs-FSDP numerics, not policy lag). policy/rollout_train_prob_diff_* is a broken metric (1e3–1e12), ignore. Per-token percentiles are NOT logged; a stdlib script to compute them from exports/dumped_data/global_step_N_train_rollouts.jsonl (~90 MB/step; needs a trainer log-prob field next to rollout_logprobs — verify the keys first) is at data/r2egym/jsc/logratio_dist.py (NOT yet on Jupiter: the scp died when the ControlMaster expired at 00:48 PT).
- Preliminary sizing (two moments only): a Laplace bulk with mean-abs 0.037 gives P(|log r|>0.2)=0.45 %, P(>0.1)=6.7 %, P(>0.05)=26 %; the measured tail beyond 0.69 (4.5e-4) is ~10^4× heavier than Laplace, so P(|log r|>0.2) is probably 0.5–2 %. So 0.2/0.2 would NOT clip hard under behavior_clip — it would bind on ≤2 % of tokens (the numerically divergent ones TIS was capping), which is a defensible trust-region role but a different premise from the task text. Proposal to put to Luke after the per-token run: eps at the |log r| p99 (expected ≈0.15–0.2) if the role is "guard the mismatch tail", or at p95 (≈0.08–0.1) if the clip is meant to bind on ordinary tokens; symmetric unless the A>0/A<0 split from the dump says otherwise. Bounds to be reported BEFORE any launch; A/B = next band arm, regular+TIS(cap 2.0) vs behavior_clip(use_tis=false, chosen eps), everything else identical; note the spread grows with lr and staleness (a staleness-3 arm needs its own measurement).
- Morning steps: (1) re-open the master (`! ssh -fN jupiter` in the prompt), (2) scp logratio_dist.py to code/snowball, run it on 2–3 dumped steps of the x16_a arm and on the sync run's dumps (last:2, lr 1e-6 — larger spread expected), (3) pick eps from the percentiles, report, then launch the A/B.
