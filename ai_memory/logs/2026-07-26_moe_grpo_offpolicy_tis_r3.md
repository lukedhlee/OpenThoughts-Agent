# MoE GRPO — "staleness-4, TIS/R3 off" mismatch finding (Jupiter 6n/8n fast gsm8k)

Audience: operator/PI who wants the full picture of *how on-policy our 30B-A3B GRPO runs
actually were*, why the corrections were off, and what to change. Written 2026-07-26 while
runs `1043206` (8n) and `1042857` (6n) were live at ~step 48/50.

## One-sentence summary

Our `_fast` gsm8k GRPO configs run with **`use_tis: false`** and **`moe_router_replay: false`**.
They also inherited **`max_staleness_steps: 4`**, but that turned out to be a **RED HERRING (§3):
it's a `fully_async`-trainer knob and these runs use the synchronous `main_base` trainer, where
staleness is hardwired to 0 — so the runs were fully on-policy on the data axis all along.** The
one real question was the **generator-trainer (vLLM↔FSDP2) mismatch**, which was NEVER measured with
TIS off (the small `policy/log_ratio ≈ 0.02` is the training update axis, NOT the generator gap;
§4). → We ran a TIS-on rerun (`1045840`, §9) and measured it: **~0.030 nats mean, routing tail ~0.035%
→ small; keep TIS on, don't build R3.** Runs were trustworthy; the config was under-specified, not broken.

---

## 1. The runs in question

| | 6-node | 8-node |
|---|---|---|
| Job | `1042857` | `1043206` |
| Config | `hpc/skyrl_yaml/jupiter/6node_qwen3_30b_a3b_gsm8k_grpo_fast.yaml` | `8node_..._fast.yaml` |
| Exp dir | `/e/scratch/reformo/lee27/experiments/jupiter_moe30b_gsm8k_grpo_6n_fast_50step_eval5/` | `..._8n_fast_eval5/` |

The two configs are **byte-identical except one line**: `num_inference_engines` (8 vs 16).
Same `seed: 42`, same sampling (`T=0.7, top_p=0.95, top_k=20`), same batch (`32 × 4`),
same policy placement (EP=4×FSDP=4, TP=1), same `update_epochs_per_batch: 1`,
same `train_batch_size == policy_mini_batch_size == 32` (one gradient step per batch).

---

## 2. What was found (the config side)

Both configs, in their `fsdp_config` blocks and `algorithm` block:

- `use_tis: false`            (TIS = Truncated Importance Sampling — the logprob-mismatch correction)
- `moe_router_replay: false`  (R3 = replay vLLM's expert-routing in the training forward)
- **`max_staleness_steps` NOT set** → inherits base default **`4`**.

Base defaults (`MarinSkyRL/skyrl-train/skyrl_train/config/ppo_base_config.yaml`):
- `use_tis: false` (L224), `tis_imp_ratio_cap: -1.0` (L223)
- `moe_router_replay: false` (L97/128/166 — policy/ref/critic)
- `max_staleness_steps: 4` (L319), with the comment: *"The larger the max_staleness_steps,
  the more off-policy the training is, and the more throughput we get."*

The launched config on the cluster confirms `max_staleness_steps: 4` (grepped from the live `.out`).

---

## 3. Off-policy sources — the staleness one is a RED HERRING (corrected)

Originally framed as two uncorrected sources; **correction: only ONE is real here.**

1. **Data staleness (`max_staleness_steps: 4`) — INERT on these runs.** This knob belongs to
   `trainer.fully_async` and is consumed ONLY by `skyrl_train/fully_async_trainer.py`. Our runs use
   the **synchronous** trainer (`trainer.py` via the `main_base` entrypoint), where staleness is
   hardwired to 0 — `trainer.py:1628`: *"For sync RL this is absent (always 0); for
   fully_async_trainer it is populated."* The sequential step timing (generate → train → sync summing
   to the step, no overlap) confirms it: fresh rollout every step, on-policy on the data axis. So the
   inherited `max_staleness_steps: 4` never did anything, and `max_staleness_steps: 0` is a no-op
   (see §7). **Setting it 0 costs no throughput — sync RL already regenerates every step.**
2. **Engine / routing mismatch — the ONLY real source.** vLLM (bf16, fused kernels, its own MoE
   routing) computes token logprobs slightly differently from the FSDP2 training forward. Measured at
   ~0.03 nats mean (§9) — small.

Efficiency note: the throughput win people associate with staleness (overlapping rollouts with
training) requires **switching to the fully-async trainer** — a mode change, not this knob. On the
sync trainer generate is always on the critical path (~105–130s/step here). Fully-async + TIS is a
coherent future throughput play (hide generate behind train+sync, ~+30%), now that §9 shows TIS is
cheap and the mismatch tiny.

---

## 4. What was measured — and the metric-conflation correction ⚠️

**Two different `log_ratio`s live in this codebase — do not confuse them:**

| metric | subtracts | axis it measures | on when? |
|---|---|---|---|
| **`policy/log_ratio_abs_*`** (`ppo_utils.py:577` `compute_log_ratio_diagnostics`) | π_new − π_old, **both on the FSDP2 trainer** | how far the optimizer moved the policy this step (update size / staleness) | always |
| **`tis/log_ratio_abs_*`** (`worker.py:1244`) | trainer forward − **vLLM rollout logprob** | **the generator-trainer (vLLM↔FSDP2) mismatch** | ONLY `if use_tis` |

Across all ~48 steps of the 8n run we observed the **`policy/`** metrics:

| metric | value | correct reading |
|---|---|---|
| `policy/log_ratio_abs_mean` | **~0.020 nats**, dead stable | optimizer step moved the policy ~2%/token — i.e. **staleness/update axis is small** |
| `policy/log_ratio_abs_max`  | ~3.0–5.6 nats | update-size outliers (NOT the generator gap) |
| `policy/ppo_clip_ratio`     | **~0.4–0.7%** | almost nothing hits the PPO clip → little off-policy pressure |

**Correction to an earlier framing in this doc's history:** these `policy/` numbers were initially
read as "the mismatch is small." That is wrong — they measure the *staleness/update* axis, which
IS legitimately small (so the staleness concern is mild). They say **nothing** about the
generator-trainer gap. That gap is `tis/log_ratio`, which is **not emitted with TIS off, and we
don't even capture `p_vllm` in these runs → the generator-trainer mismatch is UNMEASURED here.**

The only generator-trainer datapoint we have is from a *different* run (marin notes):
`tis/log_ratio_abs_mean ≈ 0.094` nats, `tis/imp_ratio_mean ≈ 0.79` at an on-policy step — and MoE
routing divergence can push the *tail* (`tis/log_ratio_abs_max`, `tis/imp_ratio_capped_fraction`)
much higher. So for a 30B **MoE** the real gap could be materially larger than 0.02 — hence the
TIS-on measurement run (§9).

### Eval curves (per-step `pass_at_1`, read from `exports/dumped_evals/global_step_*_evals`)

| step | 6n | 8n |
|---:|---:|---:|
| 0  | 0.4511 | 0.4534 |
| 20 | 0.4882 | 0.4579 |
| 40 (peak) | **0.5057 (+5.5)** | **0.4936 (+4.0)** |
| 45 | 0.5027 | 0.4807 |

Both learn, neither collapses ("noisy but stable" — matches Ben's report). 6n is the cleaner,
near-monotonic run; 8n is jaggy and ~1.5pt weaker. Since the only config diff is engine count
(8 vs 16), the 6n>8n gap is **most likely rollout-sampling variance reseeded by engine count**
(seed 42 does NOT make 8-engine and 16-engine generation identical) — i.e. luck at N=1 each, not a
real quality difference. To settle it: rerun 8n with `num_inference_engines: 8` (then byte-identical
to 6n) or multi-seed.

---

## 5. Why the corrections were off by default (not a considered decision here)

- **Vanilla path is the framework default.** Non-TIS runs are byte-identical to upstream SkyRL, so
  results don't silently depend on recent marin-fork features. TIS / TITO-full / R3 are opt-in.
- **They have costs.** TIS needs vLLM rollout logprobs captured — on THIS fork that requires
  `generator.batched:true` (see §9 fork gotcha; the general "non-batched" TIS recipe crashes here);
  R3 needs a routed-experts capture rail (Harbor-only, see §6). Also: R3's
  `enable_return_routed_experts` currently can't be
  combined with vLLM **DCP** (decode context parallelism, the long-context KV-cache sharding on the
  new torch-2.11 stack) — vLLM's `_validate_return_routed_experts` rejects the combo. It's a config
  guard, not fundamental (fork has a "DCP>1 R3 guard-lift" in progress), and irrelevant to these
  short-context torch-2.9 gsm8k runs — it only bites a future long-context MoE run that wants both.
- **They're recent** fork additions.
- **The tell:** our *production* configs DO opt in — `24GPU_qwen3_30b_a3b_thinking.yaml` and
  `64GPU_qwen3_32b.yaml` both set `use_tis: true, tis_imp_ratio_cap: 2.0` (and `batched: false`).
  The **`_fast` gsm8k configs were throughput-sweep YAMLs** that inherited the base defaults and
  never opted in; the staleness-4 default rode along unnoticed.

---

## 6. Enabling the corrections — what's a config flip vs a code change

### TIS (`use_tis`) — nearly config-only on the standard gym path
- Needs the ratio `r = p_train / p_vllm`. `p_train` is computed live each step (the
  `fwd_logprobs_values_reward` phase) — nothing to store. **`p_vllm` (the rollout logprob) is the
  thing that must be captured at generation time.**
- Good news: the **standard `skyrl_gym_generator.py` already captures rollout logprobs** — field
  `rollout_logprobs` (L41/196/354), read off `engine_output["response_logprobs"]` (L397), aligned
  to the response (L425-427); concat in `generators/utils.py` (L503-519, L600). No Harbor needed.
- Proven MoE recipe (copy from `24GPU_qwen3_30b_a3b_thinking.yaml`):
  `use_tis: true`, `tis_imp_ratio_cap: 2.0`, **`batched: false`**. In non-batched mode `p_vllm`
  flows via `response_logprobs` automatically; the generator *forbids* `sampling_params.logprobs`
  when `batched: false` (explicit guard, skyrl_gym_generator.py:111-112). TITO-full auto-enables
  with `use_tis` (utils.py:1306+), giving exact token-id alignment.
- **Cost:** `batched: false` is slower than batched generation → expect the `generate` phase to grow.

### R3 router-replay (`moe_router_replay`) — needs a code change on the standard path
- Training side is READY and validated: flag consumed at `fsdp_worker.py:656` →
  `model_wrapper.py:615` (FSDP2 replay hook; ran a full GRPO backprop on the 80B); `trainer.py:982-995`
  threads `rollout_routed_experts`; module `skyrl_train/models/router_replay.py` + `models/layers/moe.py`.
- **But capture is Harbor-only.** `extract_routed_experts_from_rollout_details()` (generators/utils.py:1014)
  reads Harbor's `RolloutDetail.extra["routed_experts"]`. The standard `skyrl_gym_generator.py`
  has **zero** routed-experts capture, and nothing in `generators/` requests
  `enable_return_routed_experts`. So flipping `moe_router_replay: true` on a standard gsm8k config
  gets sentinel-filled experts, not real ones — enabling R3 here requires wiring the gym generator
  to request + read `routed_experts` from vLLM (a code change).

---

## 7. Plan (measure-first, gated) — see §9 for live status

1. **MEASURE the generator-trainer gap first (in flight, §9):** TIS-on rerun of the same 6n
   experiment → reads `tis/log_ratio_abs_mean` (mean gap), `tis/imp_ratio_mean`, and
   `tis/imp_ratio_capped_fraction` (the **routing-divergence tail proxy** = R3 go/no-go). This is
   the number §4 says we never had.
2. **`max_staleness_steps: 0` — a NO-OP here, skip it (corrected §3).** The knob is `fully_async`-only
   and our runs use the sync trainer (staleness already hardwired to 0). Setting it changes nothing
   and costs nothing; only worth adding as defensive documentation. The runs were already on-policy on
   the data axis. (If we ever switch to the fully-async trainer for throughput, THEN staleness>0 + TIS
   becomes the relevant pairing.)
3. **R3 decision — DECIDED (§9): DO NOT build R3.** Measured tail is negligible
   (`tis/imp_ratio_capped_fraction` ≈ 0.035%, mean gap 0.030 nats). Keep TIS on; R3 only if the tail
   grows in a future long-context / larger-MoE / high-staleness regime.
4. **Settle 6n vs 8n:** rerun 8n with `num_inference_engines: 8` (byte-identical to 6n) or multi-seed
   before reading anything into the reward gap.

---

## 8. How to diagnose this yourself (next time)

1. Grep the launched `.out` for `max_staleness_steps` and check it against `use_tis` in the config —
   staleness > 0 with `use_tis: false` is the smell.
2. Distinguish the two log_ratios (§4): `policy/log_ratio_*` (always on) = training update/staleness
   axis — mean ≲ 0.05 nats + clip < 1% means the *staleness* is mild. It does NOT measure the
   generator gap. For the generator-trainer (vLLM↔FSDP2) gap you MUST turn `use_tis` on and read
   `tis/log_ratio_*` — with TIS off it is unmeasured (and `p_vllm` isn't even captured).
3. To know if a correction is even *possible* config-only: check whether the generator you use
   (`skyrl_gym_generator.py` = standard; Harbor = agentic) captures the needed field
   (`rollout_logprobs` for TIS = yes on gym; `routed_experts` for R3 = Harbor-only).

---

## Caveats

- The `log_ratio_*` numbers are the *realized* mismatch, not proof of low *data* staleness — the
  async scheduler's actual per-step staleness isn't cleanly surfaced with TIS off. The small,
  stable log-ratio is strong circumstantial evidence the pipeline ran near-synchronous, not a
  direct staleness readout.
- N=1 per config. The 6n>8n eval gap is within plausible small-batch (32) GRPO noise; don't
  over-read it without a controlled rerun.
- `batched: false` (needed for TIS) trades throughput for the correction — not a free lunch on top
  of the already-large `sync_weights` (~200s, the disaggregated `colocate_all: false` 30B weight
  broadcast; see the throughput/ colocate discussion — Ben's 14 steps/hr came from a colocated
  setup where that broadcast nearly vanishes).

---

## 9. Measurement run — TIS-on rerun (2026-07-26)

**⚠️ FORK GOTCHA (load-bearing): on this `lukedhlee/MarinSkyRL`, `use_tis:true` REQUIRES
`generator.batched:true`.** `use_tis` auto-sets `generator.sampling_params.logprobs=0`
(`utils.py:589-592`, to capture p_vllm), but the gym generator FORBIDS `logprobs` when
`batched:false` (`skyrl_gym_generator.py:112`) → hard contradiction → instant `ValueError` at
generator setup. So the "proven" batched:false TIS configs (`24GPU_qwen3_30b_a3b_thinking`,
`64GPU_qwen3_32b`) would crash here — they predate this auto-set/guard pairing (fork drift, §fork-hygiene).
The working combo is **`use_tis:true` + `batched:true`** — batched mode captures `rollout_logprobs`
via the auto-set logprobs (`skyrl_gym_generator.py:521-526`). Bonus: batched:true = the baseline's
value, so the only delta is `use_tis` itself.

- **Attempt 1 — job `1045833` CRASHED** (~5 min, at generator `_validate_cfg`):
  `ValueError: sampling_params.logprobs should be None if batched is False`. Config had `batched:false`
  (wrongly copied from the stale 24GPU recipe) + `inference_engine_mp_backend:true`. Cheap failure —
  died at validation before any eval/training.
- **Attempt 2 — job `1045840`** — `jupiter_moe30b_gsm8k_grpo_6n_fast_tis` (exp dir `..._6n_fast_tis_2`),
  6 nodes, open booster, RUNNING. **Corrected config:** `use_tis:true`, `tis_imp_ratio_cap:2.0`,
  **`batched:true`** (reverted), `mp_backend` line dropped. Everything else byte-identical to the
  6n_fast baseline (seed 42, EP=4×FSDP=4, TP=1 × 8 engines, batch 32×4, eval every 5).
  `max_staleness_steps` left at inherited **4** (kept "same experiment"; baseline `policy/log_ratio ~0.02`
  ⇒ realized staleness ~0, so `tis/log_ratio` should read ≈ pure engine/routing gap).
- **Launch:** `RESERVATION=none NUM_NODES=6 RL_CONFIG=./hpc/skyrl_yaml/jupiter/6node_qwen3_30b_a3b_gsm8k_grpo_fast_tis.yaml JOB_NAME=jupiter_moe30b_gsm8k_grpo_6n_fast_tis TIME_LIMIT=11:59:00 bash hpc/skyrl_standard/jupiter/run_gsm8k_moe30b_grpo.sh`
- **What to read** (per-step from training step 1, ~after step-0 eval): `tis/log_ratio_abs_mean`
  (mean generator-trainer gap, nats), `tis/imp_ratio_mean` (avg p_train/p_vllm),
  `tis/imp_ratio_capped_fraction` + `tis/log_ratio_abs_max` (the MoE-routing tail = R3 go/no-go).
- **RESULT (measured, steps 1–5, rock-stable):**
  - `tis/log_ratio_abs_mean` ≈ **0.030 nats** (0.028–0.031) → mean vLLM↔FSDP2 gap ≈ exp(0.03) ≈ 1.03 (~3%/token)
  - `tis/imp_ratio_mean` ≈ **1.009** → avg p_train/p_vllm within ~1% of on-policy
  - `tis/imp_ratio_capped_fraction` ≈ **0.00035** → ~1 token in 2,800 hits the 2.0 cap = **routing tail negligible**
  - (contrast: the training-axis `policy/log_ratio_abs_mean` ≈ 0.021 — separate, smaller update-size number.)
- **DECISION → DO NOT build R3.** The generator-trainer mismatch is real but small (0.030 nats, ~3×
  smaller than the 0.094 reference from the other run), and the routing-divergence tail (capped_fraction
  ~0.035%) is negligible — R3 would recover essentially nothing here. **Keep TIS on for MoE runs**
  (cheap, correct insurance: it absorbs the ~3% gap + caps the ~0.035% outliers). R3 stays a
  "revisit only if the tail grows" item for long-context / larger-MoE / high-staleness regimes.

---

## 10. Throughput bake-off — colocated vs disaggregated (2026-07-26)

Question: is disaggregated (role-split, current default) actually faster than colocated for 30B-A3B
gsm8k GRPO, or does colocation win by killing the ~192s/step weight broadcast? (My earlier
"colocation faster" was an UNVERIFIED hypothesis from Ben's 14 steps/hr — his config was never
confirmed.) Settling it empirically. Node budget 18 (user-set): 6 megatron (other workstream) +
6 + 6 bake-off; cancelled the TIS run `1045840` to free its 6.

- **Arm A — DISAGGREGATED (control):** job `1045982`, `6node_qwen3_30b_a3b_gsm8k_grpo_fast.yaml`
  (colocate_all:false, cpu_offload:false, EP4×FSDP4=16 policy GPU + 8 vLLM engines). Baseline ~432s/step.
- **Arm B — COLOCATED (test):** job `1045983`, `6node_qwen3_30b_a3b_gsm8k_grpo_coloc.yaml`
  (colocate_all:true, cpu_offload:true, EP4×FSDP6=24 GPU shared, 24 vLLM engines TP=1, util 0.70,
  max_model_len 1536 — mirrors the surviving vista colocated recipe). ⚠️ OOM-prone at weight-sync/wake.
- **Both:** MAX_STEPS=20, eval+ckpt OFF (`EVAL_INTERVAL=999 EVAL_BEFORE_TRAIN=false CKPT_INTERVAL=999`),
  TIS off, seed 42, batch 32×4 — throughput + `reward/avg_raw_reward` comparison, no 37-min evals.
- **Expectation:** reward ~equal (identical hypers); decisive metric = steps/hr. Key sub-question:
  does colocated's `sync_weights` (~192s on disagg, the 30B network broadcast) collapse to ~0, and does
  that outweigh losing train/gen overlap + paying cpu_offload?
- **Watch:** local `scratchpad/bakeoff_watch.sh` (bg) — compares step times + phase timings + reward,
  trips early if coloc OOMs.
- **⚠️ COLOCATED GOTCHA (attempt 1 `1045983` CRASHED at vLLM engine-core init):**
  `AssertionError: Expandable segments are not compatible with memory pool` (PyTorch #147851) — all 24
  engine cores. Cause: `PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True` (inherited from the disagg
  baseline's `container.extra_env`) is incompatible with the **colocate_all** vLLM sleep/wake **CUDA
  memory pool**. NOT OOM, NOT fundamental. **Fix: colocated configs must DROP `expandable_segments`**
  (disagg keeps it — disagg doesn't use the pool). Relaunched as **Arm B = `1045998`** (exp dir
  `..._coloc_bakeoff_2`) with it removed. (Note: the vista colocated yaml still carries
  expandable_segments — it would hit this on the jupiter vLLM 0.16.0 fork; treat as stale.)
- **⚠️ STRUCTURAL FINDING — 6-NODE (24-GPU) COLOCATION IS IMPOSSIBLE for Qwen3-30B-A3B**
  (attempt 2 `1045998` cleared engine-init but CRASHED at FSDP policy init):
  `AssertionError: fsdp_size=6 must divide num_experts//ep_size=32 (num_experts=128, ep_size=4)`.
  colocate_all needs ALL GPUs on the policy, so `ep_size × fsdp_size = total_GPUs` AND `fsdp_size` must
  divide `128/ep_size`. At 24 GPUs there is NO valid factorization — 24 = 8×3, and the factor of 3 never
  divides a power-of-2 expert count (128/64/32/16). Checked ep∈{1,2,4,8}: all fail. **This is exactly WHY
  the disagg 6n config puts only 16 GPUs on policy (EP4×FSDP4, 32%4=0 ✓) and 8 on inference — 16 is
  clean, 24 isn't.** Colocation is only expressible at power-of-2-friendly GPU counts: 16 GPU (4 nodes) or
  32 GPU (8 nodes). So a same-node 6n coloc-vs-disagg comparison is mathematically impossible.
- **PIVOT → Arm B = 4-NODE colocated** (`4node_qwen3_30b_a3b_gsm8k_grpo_coloc.yaml`, 16 GPU, EP4×FSDP4,
  16 engines), job **`1046007`**. Comparison is now disagg-6n (24 GPU) vs coloc-4n (16 GPU) — tests
  colocation's real pitch: same 16 policy GPUs, no separate inference nodes → do it in 4 nodes not 6.
- **RESULT — DISAGGREGATED WINS (decisive).**
  - **Arm B 4n coloc `1046007`: OOM'd** at `broadcast_to_inference_engines` (weight-sync/wake) —
    `torch.OutOfMemoryError: GPU 0 95GiB total, 69MiB free; vLLM engine holds 57.95GiB`. The
    fundamental vista `843025/843198` failure: 30B vLLM weights (~58GiB) + FSDP gather for the
    broadcast don't fit on 96GiB. The error suggests `expandable_segments:True` to defrag — but
    colocation FORBIDS that (trap #2) → hard bind. Lowering util won't help (weights are a fixed
    ~58GiB floor). This is why vista concluded "prefer disagg; colocate OOMs on weight sync/wake."
  - **Arm A 6n disagg `1045982`: runs clean** — ~430–455s/step (steady → ~8 steps/hr),
    reward noisy 0.37–0.57 (no-eval run). Cancelled at step 16/20 (verdict already settled; its
    throughput just reproduced the baseline and it had no eval → nothing new to learn). Bake-off CLOSED.
  - **Verdict:** disaggregated is the ONLY viable setup for 30B-A3B on these GH200s. Colocation failed
    4 distinct ways (TIS/batched → expandable_segments → 24-GPU expert-sharding → weight-sync OOM).
    Role-division (the user's intuition) wins. The earlier "colocation faster / Ben 14 steps/hr"
    hypothesis is **FALSIFIED for this setup** — colocation doesn't fit here, so whatever gave 14
    steps/hr was not colocation on this config. The 192s disagg weight-broadcast is real but it's the
    cost of a setup that runs. (Supersedes the §throughput-caveat line attributing Ben's number to colocation.)
- **Colocated launch traps found (all reusable):** (1) `use_tis`+`batched:false` contradiction (§9);
  (2) `expandable_segments` vs colocate memory pool; (3) 24-GPU expert-sharding impossibility. A clean
  colocated 30B recipe = power-of-2 GPU count, no expandable_segments, cpu_offload+util0.70+max_model_len1536.

### 10b. How colocation COULD work on 30B-A3B here — node count is NOT the lever

There were two independent walls; adding nodes touches only one, and not usefully:

- **Wall 1 — expert-sharding (structural).** colocate_all needs ALL GPUs on the policy, so
  `ep_size × fsdp_size = total_GPUs` AND `fsdp_size` must divide `128/ep_size`. This is about node-count
  **parity**, not quantity: **4 nodes (16 GPU, EP4×FSDP4) ✓, 8 (32, EP4×FSDP8) ✓, 16 (64) ✓ — but 6 or
  12 ✗** (factor of 3 never divides a power-of-2 expert count). Our 4n attempt cleared this wall.
- **Wall 2 — weight-sync/wake OOM (the binding one; INVARIANT to node count).** Engines ran **TP=1 /
  EP=1**, so **every GPU held the full ~58 GiB 30B model**, leaving no room for the FSDP weight-broadcast
  gather on a 96 GiB card. More nodes at TP=1 just adds more full-model engines — each GPU still carries
  58 GiB → **8 or 16 nodes would OOM identically.**

**The real fix = shard the INFERENCE engine so no GPU holds all 58 GiB:**
- **inference `tensor_parallel_size` > 1** (TP=2 → ~29 GiB/GPU, TP=4 → ~14.5 GiB/GPU), or
- **inference `expert_parallel_size` > 1** (`generator.inference_engine_expert_parallel_size`; shards the
  MoE experts — the MoE-native way).
This frees the headroom the broadcast needs. Combine with a power-of-2 node count (Wall 1).

**Two caveats:**
1. **Jupiter has a known TP>1 gotcha** — the disagg 6n config comment: *"TP=1 × 8 engines… Avoids the
   TP>1 c10d/IPv6 path."* So the memory fix (inference TP>1) runs into a networking problem they were
   deliberately dodging. Inference EP>1 might sidestep it but is untested here.
2. **Never verified Ben actually colocated on Jupiter** — his experiment dir was permission-denied, and
   the "14 steps/hr = colocation" attribution was FALSIFIED (§10). He may not have colocated at all.

**Verdict stands:** colocation is *achievable* (power-of-2 nodes + sharded inference engine), but it's
finicky on this stack; disaggregated remains the pragmatic choice unless someone specifically invests in
the TP>1/EP>1 inference path. Untested next step if pursued: 8n colocated, `inference_engine_expert_parallel_size≥2`.
