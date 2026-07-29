# Handoff — Vista MoE gsm8k GRPO (lukedhlee)

**Session logs:** `ai_memory/logs/2026-07-19_*.md`, `ai_memory/logs/2026-07-20_vista_moe_gsm8k_grpo.md`, `ai_memory/logs/2026-07-22_jupiter_flash_attn_cute.md`, `ai_memory/logs/2026-07-26_moe_grpo_offpolicy_tis_r3.md`, `ai_memory/logs/2026-07-26_jupiter_megatron_6n_gsm8k.md`

## RESOLVED (2026-07-29) — the GSM8K ~0.45 eval was a FORMAT artifact
→ **[[gsm8k_format_artifact]]**. Jupiter job `1086698`, n=1319, paired. `strict@1024=41.6%` (reproduces
step-0 `0.4511`) vs `flexible@4096=90.67%` ≈ Qwen's published 91.81%. Cause: the model answers in
`\boxed{N}` and never emits the `####` the strict scorer needs — `has ####: False`, `finish=stop`, so
neither scorer brittleness nor truncation. Format costs **+37.7 pts**, truncation +11.3.
**⇒ the 0.4511→0.5057 lift over 45 steps was format compliance, not reasoning — do not report it as
evidence MoE GRPO improves math.** GSM8K is a poor vehicle for that claim (fix the format and step-0 is
~85–90%, no headroom). Reinforces r2egym, whose binary test-pass reward can't be gamed by formatting.
**TODO before trusting r2egym's `avg_raw_reward`: measure what fraction of its failures are
non-substantive** (malformed patch, out-of-budget) rather than capability.

## ACTIVE SIDE TASK (2026-07-28) — enable Apptainer on JSC + port the apptainer bridge
→ **[[apptainer_bridge_handoff]]** is the operational state file; [[apptainer_bridge_swebench]] is the
background analysis. Goal: unlock SWE-bench as a **200-task** val set by replacing Daytona with apptainer
task containers on x86_64 JSC CPU nodes (Daytona caps us at ~40 tasks; Jupiter is aarch64 and cannot run
the images at all). **Status: `container` group GRANTED 2026-07-28, apptainer 1.4.5 verified working on
JURECA, x86_64 confirmed, login node has direct internet (no proxy needed), 1.0 GB/SIF @ ~2m47 measured.
Open blocker: no user namespaces ⇒ no `--fakeroot`/`apptainer build`, and SWE-bench images lack tmux +
asciinema which harbor's agents need.** NOT on the RL launch critical path — in-flight val stays the
held-out r2egym slice.

## NEW TASK (2026-07-27) — FSDP2 GRPO on r2egym, track eval lift
**Goal:** Train Qwen3-30B-A3B (MoE) with **FSDP2 + GRPO** on the **r2egym** task set and measure how
our key agentic eval metrics improve vs the pre-RL baseline.
- **Data:** `DCAgent/r2egym-patched-full-oracle` — **3,328** verifier-bearing SWE bug-fix tasks
  (Harbor task-binary format; binary test-pass reward). It's the RL-ready R2E-Gym member of TaskTrove.
- **Backend:** FSDP2 (not Megatron for this run). Setup = **disaggregated** (the only viable 30B-A3B
  layout on GH200s per the 2026-07-26 bake-off). Agentic RL path → `rl-agentic-launch-jupiter`
  (Harbor + Daytona), NOT the standard-gym gsm8k path. Keep **`use_tis: true`** (cheap MoE insurance).
- **VALIDATION SET = held-out r2egym (decided 2026-07-28).** r2egym-patched-full-oracle has **NO
  built-in train/val split** (single 3,328-task `train`; confirmed via HF splits API). Carve a random
  **~200-task held-out slice** as `val_data`, train on the remaining ~3,100. Why held-out r2egym (not
  SWE-bench, not nothing):
  - **In-distribution** → the correct signal for overfit detection + **best-checkpoint selection**.
  - **Free on snapshots** — held-out tasks fall on the same **7 envs** → fits our 60-cap org. (SWE-bench/
    any ID benchmark greatly exceeds our cap, so it CANNOT be an in-loop val — that's the decider.)
    **MEASURED 2026-07-28 — the real number is far worse than the 85–100 estimated here: SWE-bench is
    1 snapshot PER TASK (500 tasks → 500 unique env hashes, each Dockerfile pins a distinct
    `swebench/sweb.eval.x86_64.*` base). Harbor also NEVER reaps snapshots (delete only on error paths),
    and org headroom is 40 ⇒ max Daytona SWE-bench val set ≈ 40 tasks. Full 200-task SWE-bench needs
    Marianna's apptainer bridge on an x86_64 cluster (Jupiter is aarch64 — those images cannot run
    there at all). Full analysis + options: [[apptainer_bridge_swebench]].**
  - **Regime-clean:** run **no-summ at training context** (harbor already `enable_summarize: false`).
  - **Mechanics:** run it **OFFLINE on the saved checkpoints**, NOT in-loop (agentic val is synchronous/
    blocking). Keep `eval_interval: 999999`, `ckpt_interval: 5`; use held-out reward to pick the ship ckpt.
  - Live training signal meanwhile = `reward/avg_raw_reward` on the r2egym rollouts.
- **FORMAL EVAL → HANDED OFF TO MRINAL (eval lead) (decided 2026-07-28).** We do NOT run the ID
  benchmarks ourselves — a single ID benchmark (85–100 snapshots) exceeds our 60-cap org; it's the eval
  team's infra + the DATA org. **Our deliverable = an HF model id + size; Mrinal registers + evals.**
  - **Register via `rl-agentic-job-cleanup`** → `laion/<job>-<step>-<size>` (PUBLIC, weights flattened at
    repo root, vLLM-servable). NOT the trainer's auto-pushed `lukedhlee/<job>` repo (wrong nested layout).
  - **Size to state:** Qwen3-30B-A3B = **30B MoE (~32B-class serving)**, not 8B.
  - Ask Mrinal to ALSO eval **base `Qwen/Qwen3-30B-A3B`** on the same set → the **base→RL Δ** is the headline.
  - (DATA-org key `DAYTONA_DATA_API_KEY` is Mrinal's concern, NOT ours — RL uses only `DAYTONA_API_KEY`.)
- **⚠ EVAL-REGIME CONFOUND (RL co-lead finding, 2026-07-28) — LOAD-BEARING.** r2egym RL teaches
  long-horizon persistence, which **summarization + short context ERASES → false negative.** Co-lead's
  `DCAgent/g1_diverse_tezos_100k_8b` (r2egym→SWE-bench): **40k / summ ON = +2.7 (looks dead)**;
  **81920 / no-summ / TP4 = +10 (clearly works).** So when Mrinal evals our model, SPECIFY THE REGIME:
  **SWE-bench, ~81920 ctx (YARN), NO summarization, TP=4, 200×1** (200 = 100 verified folders + 100 random
  SWE-bench; 200×1 < SE than 100×3 — task-variance dominates trial-variance). "SWE-bench @40k/summ" ≠
  "@80k/no-summ" — different benchmarks in effect. Align the exact protocol with the co-lead so our 30B
  number is comparable to his g1-8b. Our config trains at 32k/4k-gen — revisit whether to train longer-ctx
  (separate design Q; the eval regime is the immediate lever).
- **Mrinal's ID scorecard (reference):** SWE-Bench-Verified (in-dist headline), Terminal-Bench-2,
  dev_set_v2. ID mean + per-benchmark z (`analyze-id-eval-ranking`). Traps: `swebench-verified-random-
  100-folders` (ID, N=100) ≠ full `swebench-verified` (OOD, N=500); never rank on `dev_set_v2` (partial).
- **Pre-launch:** confirm r2egym prebuilds its **7 snapshots** under cap (below) + **≤6 RUNNING RL
  jobs/cluster** guardrail before submitting.
- **Daytona snapshot findings (verified 2026-07-27):**
  - `r2egym-patched-full-oracle` = **7 unique snapshots** for all 3,328 tasks (measured: distinct
    `environment/Dockerfile` hashes; runtime-clone design → 1 env shared by up to 1,286 tasks).
    Launcher auto-prebuilds these; **snapshots are a NON-issue for the RL run.**
  - Our Daytona org (`DAYTONA_API_KEY`, in `/e/scratch/reformo/lee27/keys/secrets.env`):
    **22/60 used → 38 headroom** (20 of the 22 are Daytona base images, only 2 are `harbor__*`).
  - **Eval is the opposite — ~1 snapshot PER TASK** (measured: tb2 89 tasks→85 envs, dev_set_v2
    100→89; swebench-100 ~100). ID suite ≈ **~274 unique snapshots**, OOD >1,000 → exceeds any single
    org cap → why eval is a TWO-org build-on-demand + 2h-reclaim cycling process (Mrinal's infra).

## Finding (2026-07-26) — MoE GRPO off-policy / TIS / R3
The `_fast` gsm8k GRPO configs run `use_tis: false` + `moe_router_replay: false`. They inherit
`max_staleness_steps: 4` but that is a **RED HERRING** — it's a `fully_async`-trainer knob and these
runs use the sync `main_base` trainer (staleness hardwired to 0; `trainer.py:1628`), so they were
**on-policy on the data axis all along**; `max_staleness_steps: 0` is a no-op (don't bother). Runs
trustworthy, not broken. **Key nuance:** the small `policy/log_ratio ≈ 0.02` is the *update* axis —
NOT the generator-trainer (vLLM↔FSDP2) gap, which is `tis/log_ratio` and was **unmeasured** (TIS off). → **MEASURED** via job `1045840` (TIS-on 6n rerun): `tis/log_ratio_abs_mean` ≈ **0.030 nats**
(imp_ratio ≈ 1.009), routing tail `tis/imp_ratio_capped_fraction` ≈ **0.035%** → mismatch small,
routing tail negligible. **DECISION: keep TIS on for MoE (cheap insurance); DO NOT build R3** (would
recover ~nothing; revisit only for long-context / larger-MoE / high-staleness).
**Throughput bake-off (§10): DISAGGREGATED WINS decisively** — colocation is NOT viable for 30B-A3B on
these GH200s (4 distinct failures: TIS/batched → expandable_segments → 24-GPU expert-sharding → weight-sync
OOM). Disagg 6n runs clean ~430s/step (~8.3 steps/hr), reward 0.47→0.57 by step 3. The "colocation faster /
Ben 14 steps/hr" idea is FALSIFIED for this setup. **Fork gotcha:** TIS needs `generator.batched:true`
here (attempt 1 `1045833` crashed on auto-set-logprobs vs the batched:false guard); the "proven"
batched:false TIS configs are stale on this fork. **Full writeup:**
`ai_memory/logs/2026-07-26_moe_grpo_offpolicy_tis_r3.md` (§9).

## Autonomy
Keep fixing + relaunching without waiting for "go fix it".
**But:** never `scancel` a RUNNING job without explicit user OK.
**Megatron/Jupiter update:** user gave standing OK to cancel experiments spawned by us; do not cancel any other jobs.
**Also:** do **not** auto-relaunch after cancel unless user asks.

## Away mission (2026-07-22) — Jupiter GRPO 50-step
User away: **autonomously** run Qwen3-30B-A3B gsm8k GRPO → **50 steps**, **eval every 5**, track eval reward up.
- Config: `hpc/skyrl_yaml/jupiter/4node_qwen3_30b_a3b_gsm8k_grpo.yaml` (`max_steps=50`, `eval_interval=5`, `flash_attn=true`).
- For this mission only: fix+relaunch OK; `scancel` OK if job is clearly wedged/dead and blocks the pipeline.
- Goal evidence: step-0 baseline + evals at 5/10/…/50 with `pass_at_1` (or equiv) trend.
- **r10 1012192 FAILED** (~7m): missing `torchtitan` → installed `@a1fdd7e` + tyro.
- **r11 1012240 FAILED** (~7m): `_StridedShard` import from `_dtensor_spec` (gone in torch 2.9). Fixed in MarinSkyRL `fsdp_utils.py` → `placement_types` (local branch `lukedhlee/fix-stridedshard-torch29` @ `1f0c5b0`; same patch on Jupiter clone).
- **r12 1012293 FAILED** (~7m): stock `vllm==0.13` missing `model_loader.reload` (MoE weight-sync). feuer1 SIF unreadable.
- **Blocker:** lab vLLM fork build still compiling on **login** (`MAX_JOBS=2`, `ninja` on `_vllm_fa3_C`…). Mid-build stock vLLM is uninstalled (`import vllm` fails until install finishes). Log: `$SCRATCH/logs/build_vllm_fork_login_maxjobs2.log`. On `reload OK` → submit **r13** (no auto-PR).
- MarinSkyRL `_StridedShard` patch is **cluster-local only**; rushed PR #93 was closed by user — proper PR only when asked.
- Do **not** set `SKYRL_W13_RELOAD_BRACKET=0` on MoE (token salad).
- Queue empty; last train job **r12 1012293 FAILED**.

## Jupiter setup (2026-07-22) — bring-up in progress
- Fixed ladder: proxychains → upath → wandb offline → libcudart LD path → drop routed_experts kwarg → TP=1 (IPv6) → numpy≤2.2 → **FA2 restored**.
- **flash-attn fix:** CuTe ≠ FA2. FA2 kernels OK; `flash_attn/cute` disabled vs cutlass-dsl 4.6.1. Script: `hpc/skyrl_standard/jupiter/fix_flash_attn_cute.sh`. Log: `ai_memory/logs/2026-07-22_jupiter_flash_attn_cute.md`.

## Jupiter Megatron bring-up (2026-07-25)
- **Update 2026-07-26:** see `ai_memory/logs/2026-07-26_jupiter_megatron_6n_gsm8k.md`.
  - Faster 6-node no-save Megatron smoke completed: job `1045827`, exit `0:0`, elapsed `00:30:38`.
  - Shape: 4 vLLM rollout GPUs + 16 Megatron policy/ref GPUs (`20/24` Ray GPUs), batch size 16, `MAX_STEPS=1`.
  - Key markers: `InferenceEngineClient initialized with 4 engines`, `trainer/global_step: 1`, `Training done!`.
  - Model-only checkpoint save smoke completed after save-planning patch: job `1045872`, exit `0:0`, elapsed `00:30:46`; saved `global_step_1` plus end checkpoint `global_step_2`.
  - Resume smoke completed after disabling Megatron `FullyParallelLoadStrategyWrapper` on load: job `1045992`, exit `0:0`, elapsed `00:32:22`; loaded from `global_step_1` in `50.96s`, ran resumed step 2, saved `global_step_2` plus end checkpoint `global_step_3`.
  - Full Megatron distributed optimizer-state checkpoint save/load is now validated for same-topology resume using Megatron `dp_reshardable` optimizer checkpoint metadata.
    - Save: job `1046376`, exit `0:0`, elapsed `00:33:51`; saved full policy checkpoints at `global_step_1` and `global_step_2`, each with `.metadata` and 16 `distcp` shards.
    - Resume: job `1046602`, exit `0:0`, elapsed `00:33:40`; loaded model + optimizer + LR scheduler from `global_step_2`, ran resumed step 3, and finished `Training done!`.
  - Speed compare against FSDP2 6n fast: Megatron job `1047736`, 6 nodes, 16 policy GPUs + 8 vLLM engines, batch 32, no eval/ckpt, `MAX_STEPS=3`, completed `0:0` in `00:46:16`.
    - Megatron steps 1-3 avg: `step=313.6s`, `generate=101.9s`, `fwd=22.8s`, `policy_train=66.6s`, `sync_weights=122.1s`.
    - Megatron warm steps 2-3 avg: `step=301.9s`, `policy_train=59.1s`, `sync_weights=122.5s`.
    - FSDP2 baseline job `1042857` same 6n fast geometry avg all available train rows: `step=425.6s`, `generate=105.1s`, `fwd=25.3s`, `policy_train=110.7s`, `sync_weights=184.1s`.
    - Interpretation: Megatron is about `1.36x` faster on full step average (`425.6/313.6`) and `1.66x` faster on policy train (`110.7/66.6`) for GSM8K; warm-step comparison is about `1.41x` step and `1.87x` policy train. Startup was heavy but amortized per-step timings are clearly better.
  - Failed attempts worth remembering: `1045839` hit NCCL error during checkpoint save planning; `1045911` timed out in policy checkpoint load using the old fully-parallel load wrapper; `1046102` hit Megatron's default `fully_sharded_model_space` flattened-range unsupported error; `1046202` showed `fully_reshardable` optimizer save is not stable on Jupiter/Ray because its Gloo DP gather closed by peer; `1046522` loaded model then OOM'd during optimizer restore until the load path started releasing loaded state before restoring the next component.
  - Operational lesson: 30B Megatron can be quiet for 10+ minutes after vLLM `Model loading took ...` due to FlashInfer MoE autotuning and worker bring-up; do not call it wedged too early.
  - Remaining caveat: Megatron optimizer checkpoint portability across topology/TP/PP/EP changes is **not** validated. The supported full-state path is same-topology resume with `dp_reshardable`.
- Native env: `/e/scratch/reformo/lee27/OpenThoughts-Agent/envs/rl-megatron`.
- Added smoke configs:
  - `hpc/skyrl_yaml/jupiter/2node_qwen3_0p6b_gsm8k_grpo_megatron_smoke.yaml`
  - `hpc/skyrl_yaml/jupiter/6node_qwen3_30b_a3b_gsm8k_grpo_megatron_smoke.yaml`
  - launcher wrapper `hpc/skyrl_standard/jupiter/run_gsm8k_megatron_grpo.sh`
- Launcher fixes:
  - `hpc/sbatch_rl/universal_rl.sbatch` emits RL env overrides before activation, so `RL_ENV_DIR=.../envs/rl-megatron` takes effect.
  - `hpc/rl_config_utils.py` treats Megatron `model_config_kwargs` / `transformer_config_kwargs` as Hydra `++` optional overrides.
  - `hpc/rl_config_utils.py` respects YAML `hf_hub_repo_id: null` instead of auto-defaulting Hub upload.
- Verified jobs:
  - `1043401`: 2-node Qwen3-0.6B Megatron smoke completed, exit `0:0`.
  - `1043464`: 6-node Qwen3-30B-A3B trained both steps but failed in final Megatron optimizer checkpoint save (`gloo ... Connection closed by peer`).
  - `1043650`: 6-node Qwen3-30B-A3B no-save smoke completed, exit `0:0`, elapsed `00:39:11`; train loop reached `Training done!`.
- Current caveat: 30B Megatron full checkpoint/resume now works on the validated same-topology 6-node geometry with `SAVE_OPTIMIZER_STATES=true` for save and `LOAD_OPTIMIZER_STATES=true` for resume. Cross-topology optimizer resharding remains unsupported/unvalidated; do not use `fully_reshardable` on Jupiter/Ray unless specifically debugging it.

## Status — submitted (2026-07-20)
- **843437** cancelled (hung step-15; no val).
- **844374** submitted then **cancelled on user request** (do not relaunch).
- Tip ready: `00e40bd8`; local next-run config now has `eval_before_train: true`, `eval_interval: 10`, `ckpt_interval: 5`, `seed: 42`.
- **844625** submitted on `gh`×24 from commit `135ca0a5`, then **cancelled on user request** at ~`2026-07-20 09:18` TACC local.
  - Job name: `vista_moe30b_gsm8k_grpo_24n_eval1`
  - Slurm state: `COMPLETING` immediately after cancel; no checkpoint was saved because `ckpt_interval=20` and it had not reached step 20.
  - Log path: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_eval1/logs/vista_moe30b_gsm8k_grpo_24n_eval1_844625.out`
  - GPUs: 24 Vista GH200 GPUs = 24 nodes, 1 GPU/node; layout 16 policy/ref + 8 vLLM.
  - ⚠ **EVERY `pass_at_1 ≈ 0.44–0.47` BELOW IS A FORMAT ARTIFACT, NOT MATH ABILITY** (measured
    2026-07-29, [[gsm8k_format_artifact]]). The model answers in `\boxed{N}`, never emits the `####`
    the strict scorer requires. Paired re-scoring of identical rollouts: `strict@1024=41.6%` (reproduces
    these numbers) vs `flexible@4096=90.67%` (≈ Qwen's published 91.81%). Any upward trend in these
    curves is the model learning FORMAT COMPLIANCE, not reasoning. Do not cite as reasoning gains.
  - Step-0 eval: `pass_at_1=0.4412433662`, eval wall `1847.62s`.
  - Step 1: train step wall `400.03s` (~9.0 train-only steps/hour); train `avg_raw_reward=0.4765625`, `avg_pass_at_4=0.65625`; eval `pass_at_1=0.4579226687`, eval wall `1863.94s`.
  - Step 2: train step wall `390.61s`; train `avg_raw_reward=0.4296875`, `avg_pass_at_4=0.59375`; eval `pass_at_1=0.4427596664`, eval wall `1868.44s`.
  - Step 3: train step wall `384.85s`; train `avg_raw_reward=0.546875`, `avg_pass_at_4=0.71875`; eval `pass_at_1=0.4632297195`, eval wall `1866.20s`.
  - Step 4: train step wall `381.76s`; train `avg_raw_reward=0.546875`, `avg_pass_at_4=0.75`; eval `pass_at_1=0.4586808188`, eval wall `1872.24s`.
  - Step 5: train step wall `385.88s`; train `avg_raw_reward=0.5390625`, `avg_pass_at_4=0.71875`; eval `pass_at_1=0.4632297195`, eval wall `1873.57s`.
  - Step 6: train step wall `386.05s`; train `avg_raw_reward=0.578125`, `avg_pass_at_4=0.78125`; eval `pass_at_1=0.4586808188`, eval wall `1874.68s`.
  - Step 7: train step wall `385.14s`; train `avg_raw_reward=0.65625`, `avg_pass_at_4=0.78125`; eval `pass_at_1=0.4670204701`, eval wall `1880.29s`.
  - Step 8: train step wall `388.19s`; train `avg_raw_reward=0.421875`, `avg_pass_at_4=0.59375`; eval `pass_at_1=0.4601971190`, eval wall `1885.05s`.
  - Step 9: train step wall `388.82s`; train `avg_raw_reward=0.3203125`, `avg_pass_at_4=0.46875`; eval `pass_at_1=0.4442759666`, eval wall `1870.31s`.
  - Step 10: train step wall `381.94s`; train `avg_raw_reward=0.6171875`, `avg_pass_at_4=0.71875`; eval `pass_at_1=0.4730856710`, eval wall `1879.10s`.
  - Step 11: train step wall `385.64s`; train `avg_raw_reward=0.375`, `avg_pass_at_4=0.5`; eval started `2026-07-20 07:40:36` and was at 5/42 (12%) as of `07:43:35`.
  - With eval every step, effective cadence is ~1.6 steps/hour because each full eval costs ~31 min. Train-only throughput is now ~9.2-9.4 steps/hour.
  - Eval curve through step 10: `0.44124`, `0.45792`, `0.44276`, `0.46323`, `0.45868`, `0.46323`, `0.45868`, `0.46702`, `0.46020`, `0.44428`, `0.47309`.
- **846086** submitted on `gh`×24 from commit `00e40bd8`, then **TIMED OUT**.
  - Job name: `vista_moe30b_gsm8k_grpo_24n_eval10_ckpt5`
  - Slurm final state: `TIMEOUT`, elapsed `12:00:27`; Slurm killed it at `2026-07-20T21:20:14` CDT due to time limit.
  - Log path: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_eval10_ckpt5/logs/vista_moe30b_gsm8k_grpo_24n_eval10_ckpt5_846086.out`
  - Checkpoint path: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_eval10_ckpt5/vista_moe30b_gsm8k_grpo_24n_eval10_ckpt5/checkpoints`
  - Submitted config verified: `trainer.eval_interval=10`, `trainer.eval_before_train=true`, `trainer.ckpt_interval=5`, `policy_num_nodes=16`, `generator.num_inference_engines=8`.
  - Checkpoint load: `No checkpoint found, starting training from scratch`.
  - Baseline eval step 0: `pass_at_1=0.4397270660`, eval wall `1926.32s` (~32.1 min).
  - Step 1 generation done: `reward/avg_raw_reward=0.4765625`, `reward/avg_pass_at_4=0.65625`; policy_train started `2026-07-20 10:14:33` TACC local.
  - Step 1 completed at `10:19:20`: step wall `404.12s`, train-only throughput `8.91` steps/hour, no eval due `eval_interval=10`.
  - Step 2 generation completed at `10:20:51`: `reward/avg_raw_reward=0.4375`, `reward/avg_pass_at_4=0.5625`.
  - **Hard stall:** log did not advance past step-2 `Started: 'fwd_logprobs_values_reward'` / `MESH_DISPATCH method=forward issued forward.remote() to 16 actors` from `10:20:51` until Slurm timeout at `21:20`.
    - Ray status: 24 nodes active, 24/24 GPUs reserved; 16 `FSDPPolicyWorkerBase` + 8 `AsyncVLLMInferenceEngine` actors alive.
    - No trainer traceback/CUDA/NCCL fatal surfaced; several init-time `InfoActor` deaths are present but policy/vLLM actors are alive.
    - GPU sample: several ~89GB policy/ref GPUs idle at 0% while many ~21GB vLLM-sized GPUs are at 100%.
  - No checkpoint was saved. First checkpoint would have been step 5, but only step 1 completed and step 2 stalled.
  - Do not relaunch the exact same 24n recipe blindly; this looks like distributed forward/logprob/collective instability, not eval cadence.
- **848839** submitted on `gh`×24 from the same commit/config (`00e40bd8`) at user request to retry without fixes.
  - Job name: `vista_moe30b_gsm8k_grpo_24n_eval10_ckpt5_retry`
  - Log path: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_eval10_ckpt5_retry/logs/vista_moe30b_gsm8k_grpo_24n_eval10_ckpt5_retry_848839.out`
  - Checkpoint path: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_eval10_ckpt5_retry/vista_moe30b_gsm8k_grpo_24n_eval10_ckpt5_retry/checkpoints`
  - Actual SkyRL command verified same intended geometry: `trainer.eval_interval=10`, `trainer.ckpt_interval=5`, `trainer.eval_before_train=true`, `policy_num_nodes=16`, `generator.num_inference_engines=8`.
  - Baseline eval step 0 finished at `22:43:16` CDT: `pass_at_1=0.4397270660`, eval wall `1927.86s`.
  - Step 1 completed at `22:50:00`: step wall `403.85s`, `reward/avg_raw_reward=0.4765625`, `reward/avg_pass_at_4=0.65625`.
  - Step 2 generation completed at `22:51:30`: `reward/avg_raw_reward=0.4765625`, `reward/avg_pass_at_4=0.59375`.
  - Step 2 `fwd_logprobs_values_reward` **did not stall this time**; it finished in `17.80s`.
  - Step 2 `policy_train` finished in `137.46s`.
  - **Likely hard stall now at step-2 `sync_weights`:** last trainer progress is `Started: 'sync_weights'` at `22:54:06`; log/stat still frozen at `22:54:07` as of `23:07:11` while Slurm job remained `RUNNING`.
  - No checkpoint yet; first checkpoint is step 5.
  - Do not `scancel` without explicit user OK.
  - Debug update (2026-07-21 KST):
    - Still `RUNNING` as of `2026-07-21 13:51:12 KST`; log/stat still frozen at `2026-07-21 12:54:07 KST`.
    - Ray status: 24 active nodes, all 24 GPUs reserved, no pending demands/failures.
    - Ray actors: 16 `FSDPPolicyWorkerBase` alive, 8 `AsyncVLLMInferenceEngine` alive; `InfoActor` deaths are init-time leftovers.
    - Ray tasks retained only the last 100 tasks, all `AsyncVLLMInferenceEngine.update_named_weights` and all `FINISHED`.
    - Process names on all 16 policy workers are `ray::FSDPPolicyWorkerBase.broadcast_to_inference_engines`; vLLM actors are idle/alive.
    - `/proc`/thread sampling: policy actor main threads are in `ep_poll`; no `pstack`/`gdb` attach due ptrace denial; no NCCL trace files were dumped under `/scratch/11584/lukedhlee/nccl_trace_*`.
    - vLLM log shows step-1 weight reload/cache reset completed at `2026-07-21 12:50:00 KST`; no equivalent step-2 finish/reset line after `2026-07-21 12:54:06 KST`.
    - Interpretation: current stall is inside policy-side `broadcast_to_inference_engines`, likely at a policy barrier or Ray/vLLM reload/update await after the vLLM update tasks have mostly/completely returned.
    - Remote SkyRL source staged for next launch with `WEIGHT_SYNC_DEBUG` phase logging in `/scratch/11584/lukedhlee/MarinSkyRL/skyrl-train/skyrl_train/workers/fsdp/fsdp_worker.py`; syntax verified with `py_compile` and `git diff --check`. This does not affect the already-running job.
- **848839** cancelled on user approval at `2026-07-21 14:16 KST` after remaining stuck in step-2 `sync_weights`.
- **849329** submitted on `gh`x24 at `2026-07-21 14:17 KST` for targeted weight-sync debugging, then failed fast before training.
  - Job name: `vista_moe30b_gsm8k_grpo_24n_syncdbg_bs8_noeval`
  - Log path: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs8_noeval/logs/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs8_noeval_849329.out`
  - Config path: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs8_noeval/configs/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs8_noeval_rl_config.json`
  - Local config: `hpc/skyrl_yaml/vista/24node_qwen3_30b_a3b_gsm8k_grpo_syncdebug.yaml`
  - Purpose: reproduce the step-2 `sync_weights` stall quickly with `WEIGHT_SYNC_DEBUG=1`.
  - Actual submitted SkyRL args verified: `trainer.max_steps=3`, `trainer.train_batch_size=8`, `trainer.policy_mini_batch_size=8`, `trainer.eval_interval=999999`, `trainer.eval_before_train=false`, `trainer.ckpt_interval=999999`, `trainer.placement.policy_num_nodes=16`, `trainer.placement.ref_num_nodes=16`, `generator.num_inference_engines=8`, `generator.n_samples_per_prompt=4`.
  - Note: runner startup prints generic env `NUM_INFERENCE_ENGINES=24` / `POLICY_NUM_NODES=24`; generated SkyRL hydra args are the authoritative geometry and are correct.
  - Ray cleanup/startup was healthy; the long `Cleaning up existing Ray instances...` pause was a sequential 24-node `ray stop --force` cleanup loop.
  - Failed at `2026-07-21 14:23:50 KST` in SkyRL config validation: `train_batch_size (8)` must be at least the policy DP LCM (`policy_dp_size=16`, `lcm_dp_size=16`). No training step ran.
- **849358** submitted on `gh`x24 at `2026-07-21 14:26 KST` with corrected minimum batch size for the same targeted weight-sync debug.
  - Job name: `vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval`
  - Log path: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval/logs/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval_849358.out`
  - Config path: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval/configs/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval_rl_config.json`
  - Actual submitted SkyRL args verified: `trainer.max_steps=3`, `trainer.train_batch_size=16`, `trainer.policy_mini_batch_size=16`, `trainer.eval_interval=999999`, `trainer.eval_before_train=false`, `trainer.ckpt_interval=999999`, `trainer.placement.policy_num_nodes=16`, `trainer.placement.ref_num_nodes=16`, `generator.num_inference_engines=8`, `generator.n_samples_per_prompt=4`.
  - Final state: `FAILED`, elapsed `23:47`.
  - Progress before failure: Ray ready, vLLM loaded, FSDP policy weights loaded, initial `init_weight_sync_state` finished in `8.72s`, `load_checkpoints` found no checkpoint, initial `sync_weights` started at `2026-07-21 14:47:36 KST` and logged `Finished: 'sync_weights', time cost: 1.78s`.
  - Failure cause: instrumentation bug in remote `fsdp_worker.py`: `UnboundLocalError: cannot access local variable 'os'` because a later local `import os` shadowed module-level `os`. This is not evidence of the original weight-sync stall.
- **849442** submitted on `gh`x24 at `2026-07-21 14:52 KST` after fixing the instrumentation bug.
  - Job name: `vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval_dbg2`
  - Log path: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval_dbg2/logs/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval_dbg2_849442.out`
  - Config path: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval_dbg2/configs/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval_dbg2_rl_config.json`
  - Actual submitted SkyRL args verified: `trainer.max_steps=3`, `trainer.train_batch_size=16`, `trainer.policy_mini_batch_size=16`, `trainer.eval_interval=999999`, `trainer.eval_before_train=false`, `trainer.ckpt_interval=999999`, `trainer.placement.policy_num_nodes=16`, `trainer.placement.ref_num_nodes=16`, `generator.num_inference_engines=8`, `generator.n_samples_per_prompt=4`.
  - Slurm/job startup:
    - At `2026-07-21 15:04 KST`, one vLLM engine hit retryable FlashInfer JIT/FileLock `OSError: [Errno 116] Stale file handle`; vLLM retried and recovered.
    - `InferenceEngineClient initialized with 8 engines` by `2026-07-21 15:05:10 KST`.
    - `load_checkpoints`: no checkpoint found, started from scratch.
  - Initial `sync_weights` completed at `2026-07-21 15:18:17 KST`, cost `149.97s`, after streaming all `18866` tensors through `WEIGHT_SYNC_DEBUG` to `lm_head.weight`.
  - Step 1:
    - generate `48.68s`; `reward/avg_pass_at_4=0.5625`, `reward/avg_raw_reward=0.328125`
    - fwd/logprobs/reward `19.32s`; policy_train `84.64s`
    - post-step `sync_weights` completed at `2026-07-21 15:23:28 KST`, cost `157.85s`
    - step wall `310.61s`; train batches `1/3`
  - Step 2:
    - generate `45.95s`; `reward/avg_pass_at_4=0.8125`, `reward/avg_raw_reward=0.640625`
    - fwd/logprobs/reward `10.06s`; policy_train `75.54s`
    - post-step `sync_weights` completed at `2026-07-21 15:28:16 KST`, cost `156.42s`
    - step wall `288.03s`; train batches `2/3`
    - This explicitly passed the previous step-2 `sync_weights` hang point.
  - Step 3:
    - generate `45.08s`; `reward/avg_pass_at_4=0.75`, `reward/avg_raw_reward=0.5625`
    - fwd/logprobs/reward `10.18s`; policy_train `78.31s`
    - final `sync_weights` completed at `2026-07-21 15:33:06 KST`, cost `156.31s`
    - step wall `289.95s`; train batches `3/3`; trainer logged `Reached max training steps (3)`.
  - Debug conclusion: the 24-node targeted run did **not** reproduce the stall. All three post-train weight syncs completed with stable ~156-158s sync cost. The earlier step-2 sync hang is intermittent/environment-sensitive, not deterministic for this recipe.
  - End-of-run checkpoint saved despite `ckpt_interval=999999`:
    - `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval_dbg2/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval_dbg2/checkpoints/global_step_4`
    - verified files include `data.pt`, `trainer_state.pt`, `latest_ckpt_global_step.txt`, `policy/fsdp_config.json`, and all 16 policy model/optim/extra-state rank shards.
  - HF export completed at `2026-07-21 15:37:35 KST`:
    - `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval_dbg2/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval_dbg2/exports/global_step_4/policy`
    - verified files include two `.safetensors` shards, `model.safetensors.index.json`, config/generation config, tokenizer files, and chat template.
  - HF Hub upload failed with `401 Unauthorized` after training completed; not a training/sync failure.
  - Ray stopped and logs were preserved to `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_noeval_dbg2/ray_logs/`.
  - Final Slurm accounting at `2026-07-21 15:41:59 KST`: top-level job `COMPLETED`, elapsed `00:47:55`, exit `0:0`. `squeue` still briefly showed `COMPLETING` while cleanup drained, but `sacct` recorded successful completion.
- **849721** submitted on `gh`x24 at `2026-07-21 15:48 KST` for a longer instrumented no-eval sync-stall sample.
  - Job name: `vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_20step`
  - Log path: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_20step/logs/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_20step_849721.out`
  - Config path: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_20step/configs/vista_moe30b_gsm8k_grpo_24n_syncdbg_bs16_20step_rl_config.json`
  - Local config: `hpc/skyrl_yaml/vista/24node_qwen3_30b_a3b_gsm8k_grpo_syncdebug20.yaml`; remote copy created from the 3-step debug config with `trainer.max_steps=20`.
  - Actual generated Hydra args verified: `trainer.max_steps=20`, `trainer.train_batch_size=16`, `trainer.policy_mini_batch_size=16`, `trainer.eval_interval=999999`, `trainer.eval_before_train=false`, `trainer.ckpt_interval=999999`, `trainer.hf_save_interval=999999`, `trainer.placement.policy_num_nodes=16`, `trainer.placement.ref_num_nodes=16`, `generator.num_inference_engines=8`, `generator.n_samples_per_prompt=4`, `SKYRL_WEIGHTSYNC_DEBUG=1`.
  - Slurm allocation started immediately on 24 nodes; as of `2026-07-21 15:49:24 KST`, state `RUNNING`, elapsed `1:09`, still in expected Ray startup cleanup (`Cleaning up existing Ray instances...`).
  - Live status at `2026-07-21 16:40:54 KST`: Slurm job still `RUNNING`, elapsed `00:52:39`; `squeue -u lukedhlee` shows this is the only queued/running user job. Earlier Ray status had 24 active Ray nodes, all 24/24 GPUs reserved, no pending Ray resource demands/failures.
  - Startup passed the relevant distributed phases:
    - Ray cluster ready by `2026-07-21 15:55:09 KST`.
    - `InferenceEngineClient initialized with 8 engines` by `2026-07-21 15:59:13 KST`.
    - `init_weight_sync_state` finished in `8.21s`; no checkpoint found, started from scratch.
    - Initial `sync_weights` completed at `2026-07-21 16:11:27 KST`, cost `123.13s`.
  - Step 1 completed:
    - generate `47.62s`; raw reward `0.328125`, pass@4 `0.5625`
    - fwd/logprobs `16.98s`; policy_train `76.13s`; post-step `sync_weights` `127.41s`
    - step wall `268.26s`; `Training Batches Processed: 1/20`
  - Step 2 reached:
    - generate `46.72s`; raw reward `0.59375`, pass@4 `0.75`
    - fwd/logprobs `10.21s`, so this run passed the prior 846086 step-2 fwd/logprob stall point.
    - `policy_train` started at `2026-07-21 16:16:53 KST`.
  - **Current suspected hard stall:** no log advance after file timestamp `2026-07-21 16:18:23 KST`; last progress line is `Policy Train epoch [1/1]: 15/16 (94%)` at `2026-07-21 16:17:53 KST`. There is no `16/16`, no `Finished: 'policy_train'`, and no step-2 `sync_weights` start.
  - Process/thread sample at `2026-07-21 16:31 KST`: sampled policy worker GPUs remain at 100% utilization; policy main threads are in `ep_poll`; `pt_autograd_0` threads are running at ~81% CPU. This supports an in-training policy/FSDP/autograd/collective stall rather than eval/checkpoint/generation/fwd-logprob/sync.
  - Current interpretation: 849721 is not stalled in eval/checkpoint/generation/fwd-logprob/sync. It is wedged inside step-2 policy-side training, likely around the final mini-batch/optimizer/FSDP collective. Do not `scancel` without explicit user approval.
  - Cancelled on user approval at `2026-07-21 16:43 KST`.
  - Final Slurm accounting: `CANCELLED by 910114`, elapsed `00:55:12`, exit `0:0`.
- **850044** submitted on `gh`x24 at `2026-07-21 16:47 KST` for the next targeted policy-train debug run.
  - Job name: `vista_moe30b_gsm8k_grpo_24n_policydbg_bs16_10step`
  - Log path: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_policydbg_bs16_10step/logs/vista_moe30b_gsm8k_grpo_24n_policydbg_bs16_10step_850044.out`
  - Config path: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_policydbg_bs16_10step/configs/vista_moe30b_gsm8k_grpo_24n_policydbg_bs16_10step_rl_config.json`
  - Local config: `hpc/skyrl_yaml/vista/24node_qwen3_30b_a3b_gsm8k_grpo_policydebug10.yaml`
  - Remote SkyRL instrumentation active in `/scratch/11584/lukedhlee/MarinSkyRL/skyrl-train/skyrl_train/workers/worker.py`:
    - `POLICY_TRAIN_DEBUG` logs for ppo_train enter, each mini-batch before/after `training_step`, status all-reduce before/after, final policy barrier, and exit.
    - `POLICY_STEP_DEBUG` logs for per-rank `training_step` enter, `to_device`, forward, policy loss, backward, router replay teardown, optimizer step, entropy all-reduce, and return.
    - `faulthandler.dump_traceback_later(..., repeat=True)` enabled during policy_train with `SKYRL_POLICY_TRAIN_DUMP_AFTER_S=180`, so a stall should dump Python stacks into the Slurm log.
    - Existing `WEIGHT_SYNC_DEBUG` instrumentation in `fsdp_worker.py` remains active.
  - Validated with `py_compile` and `git diff --check` on Vista.
  - Generated Hydra args verified: `trainer.max_steps=10`, `trainer.train_batch_size=16`, `trainer.policy_mini_batch_size=16`, `trainer.eval_interval=999999`, `trainer.eval_before_train=false`, `trainer.ckpt_interval=999999`, `trainer.placement.policy_num_nodes=16`, `generator.num_inference_engines=8`.
  - Sbatch env verified: `SKYRL_WEIGHTSYNC_DEBUG=1`, `SKYRL_POLICY_TRAIN_DEBUG=1`, `SKYRL_POLICY_TRAIN_DUMP_AFTER_S=180`.
  - Initial Slurm status at `2026-07-21 16:47:26 KST`: `RUNNING`, elapsed `0:14`, 24 nodes allocated; log showed `Starting Ray cluster with 24 nodes, 1 GPUs/node` and `Cleaning up existing Ray instances...`.
  - Final status checked at `2026-07-21 18:52 KST`: job no longer in `squeue`; `sacct` top-level state `FAILED`, elapsed `01:17:22`, exit `1:0`.
  - Important outcome: the suspected training stall did **not** reproduce. The run reached `Training Batches Processed: 10/10 (100%)` at `2026-07-21 17:56:32 KST`; steps 1-10 all completed generate, fwd/logprobs, policy_train, and sync_weights.
    - Step 10 timings: generate `47.45s`, fwd/logprobs/reward `8.58s`, policy_train `70.75s`, sync_weights `153.67s`, full step wall `280.50s`.
    - Training throughput over steps 1-10: `2836s` for 10 steps = about `12.7` training steps/hour.
    - Step-10 reward metrics: `reward/avg_raw_reward=0.578125`, `reward/avg_pass_at_4=0.6875`.
  - Final checkpoint and export were created despite the failed Slurm state:
    - checkpoint: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_policydbg_bs16_10step/vista_moe30b_gsm8k_grpo_24n_policydbg_bs16_10step/checkpoints/global_step_11`
    - export: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_policydbg_bs16_10step/vista_moe30b_gsm8k_grpo_24n_policydbg_bs16_10step/exports/global_step_11/policy`
  - Failure mode: after training and after `Successfully saved model weights` at `2026-07-21 18:00:26 KST`, Ray raised `WorkerCrashedError` at `2026-07-21 18:02:33 KST`. The Raylet reported worker `skyrl_entrypoint` PID `1232885` on IP `129.114.17.45` exited with connection error code 2 / EOF.
  - Memory evidence near the crash: one policy worker rank on IP `129.114.18.199` reported node memory pressure during save/export, rising from `85.2%` to `91.9%` node memory used (`18.3` to `17.2 GiB` available), with monitor advice to reduce concurrent trials / parallel generation workers. This points to a post-training save/export memory crash, not the earlier policy-train hang.
  - Current Slurm queue at `2026-07-21 18:52:17 KST`: no jobs running or queued for `lukedhlee`.
- **850782** submitted on `gh`x24 at `2026-07-21 18:57 KST` for an unattended resumed long debug run.
  - Job name: `vista_moe30b_gsm8k_grpo_24n_resume11_eval10_policydbg60`
  - Purpose: continue from `850044`'s healthy `global_step_11` checkpoint while keeping policy-train and weight-sync instrumentation active; add validation/checkpoint cadence every 10 steps.
  - Local/remote config: `hpc/skyrl_yaml/vista/24node_qwen3_30b_a3b_gsm8k_grpo_resume11_eval10_policydebug60.yaml`
  - Generated config: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_resume11_eval10_policydbg60/configs/vista_moe30b_gsm8k_grpo_24n_resume11_eval10_policydbg60_rl_config.json`
  - Log path: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_resume11_eval10_policydbg60/logs/vista_moe30b_gsm8k_grpo_24n_resume11_eval10_policydbg60_850782.out`
  - Verified generated Hydra args:
    - `trainer.max_steps=60`
    - `trainer.resume_mode=latest`
    - `trainer.ckpt_path=/scratch/11584/lukedhlee/experiments/vista_moe30b_gsm8k_grpo_24n_policydbg_bs16_10step/vista_moe30b_gsm8k_grpo_24n_policydbg_bs16_10step/checkpoints`
    - `trainer.export_path=/scratch/11584/lukedhlee/experiments/vista_moe30b_gsm8k_grpo_24n_resume11_eval10_policydbg60/vista_moe30b_gsm8k_grpo_24n_resume11_eval10_policydbg60/exports`
    - `trainer.eval_interval=10`, `trainer.eval_before_train=false`, `trainer.ckpt_interval=10`, `trainer.hf_save_interval=999999`
    - `trainer.train_batch_size=16`, `trainer.policy_mini_batch_size=16`
    - `trainer.placement.policy_num_nodes=16`, `trainer.placement.ref_num_nodes=16`, `generator.num_inference_engines=8`
  - Verified sbatch env includes `SKYRL_WEIGHTSYNC_DEBUG=1`, `SKYRL_POLICY_TRAIN_DEBUG=1`, and `SKYRL_POLICY_TRAIN_DUMP_AFTER_S=180`.
  - Initial Slurm state at `2026-07-21 18:57:18 KST`: `PENDING`, reason `(Priority)`, 24 nodes requested, 12h wall limit.
  - Queue detail at `2026-07-21 18:59:45 KST`: still `PENDING (Priority)`. `scontrol` showed estimated start `2026-07-22 00:28:23 KST` and scheduled nodes listed; this is an estimate and may move.
  - Live check at `2026-07-22 00:26-00:27 KST`: job is `RUNNING`, elapsed `~4:10`, but appears hard-stalled. Log mtime is `2026-07-21 20:46:38 KST` (shown by `stat` as `2026-07-21 06:46:38 -0500`), so no log progress for ~3h40.
  - Resume worked: checkpoint global step 11 loaded, and training progressed.
  - Step 11 post-resume sync completed at `2026-07-21 20:37:34 KST`, cost `138.40s`, progress `Training Batches Processed: 11/60`.
  - Step 12 completed at `2026-07-21 20:42:04 KST`:
    - policy_train `77.93s`, sync_weights `130.31s`, step wall `270.57s`
    - reward `avg_raw_reward=0.5625`, train `avg_pass_at_4=0.6875`
  - Step 13 reached:
    - generate/fwd completed; `policy_train` started at `2026-07-21 20:43:00 KST` and finished at `20:44:17 KST`, cost `77.19s`.
    - Current stall is in the following `sync_weights`, not policy_train. Last log lines are `WEIGHT_SYNC_DEBUG rank=0 phase=chunk_update_task_await_before chunk=18666 name=model.layers.47.mlp.experts.62.down_proj.weight` at `2026-07-21 20:46:27 KST`; no corresponding `await_after`, no final barrier, no `Finished: 'sync_weights'`.
  - Interpretation update: stall timing is not deterministic. It has now appeared at prior step-2 policy/fwd/sync phases and at resumed step 13 sync; short runs can complete without reproducing it.
  - Final: **TIMEOUT** at `2026-07-21T18:17:31` CDT after wall `12:00:21`; never advanced past the step-13 sync hang. No new checkpoint beyond parent `global_step_11`.
- **NCCL (otagent env on Vista):** `nvidia-nccl-cu12` / `libnccl` **2.28.9** (`NCCL version 2.28.9+cuda12.9`); torch `2.11.0+cu128`. Prior debug YAMLs used `NCCL_DEBUG: WARN`.
- **854962** submitted on `gh`×24 at `2026-07-22 16:01 KST` for NCCL_DEBUG=INFO sync-stall capture.
  - Job name: `vista_moe30b_gsm8k_grpo_24n_syncdbg20_ncclinfo`
  - Config: `hpc/skyrl_yaml/vista/24node_qwen3_30b_a3b_gsm8k_grpo_syncdebug20.yaml` (`max_steps=20`, no eval, `NCCL_DEBUG: INFO`, `SKYRL_WEIGHTSYNC_DEBUG=1`)
  - Log: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_syncdbg20_ncclinfo/logs/vista_moe30b_gsm8k_grpo_24n_syncdbg20_ncclinfo_854962.out`
  - Sbatch verified: `export NCCL_DEBUG="INFO"` wins after an earlier WARN default; still has `SKYRL_WEIGHTSYNC_DEBUG=1`.
  - Cancelled at user request `2026-07-22 04:02 CDT` after ~1h18m / **~11–12/20** clean steps (no stall).
  - **NCCL_DEBUG=INFO verdict:** env was exported, but Slurm `.out` had **no real NCCL INFO library lines** (no Bootstrap/Channel/IB dumps). Log bloat (~235MB) was almost entirely `WEIGHT_SYNC_DEBUG`. Only useful side notes: P2P/SHM disabled (expected 1GPU/node); torch deprecation warns (`NCCL_BLOCKING_WAIT`, `TORCH_NCCL_TRACE_BUFFER_SIZE`); DTensor fsdp×ep double all-reduce warn. **No new stall root-cause from INFO.**
- **855070** cancelled before start (user: fix NCCL capture first).
- **NCCL debug fix (verified on gh-dev smoke 855073):**
  - MarinSkyRL `prepare_runtime_environment` now forwards `NCCL_DEBUG_FILE` + mkdir parent (Vista checkout `lukedhlee/fix-ray-255-pg-strategy`).
  - Set `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,NET,COLL`, `NCCL_DEBUG_FILE=/scratch/11584/lukedhlee/nccl_debug/nccl_%h_%p.log`.
  - Smoke produced real files e.g. `nccl_c642-002_*.log` with **~44k `NCCL INFO` lines** each (NCCL 2.28.9). Evidence kept in `$SCRATCH/nccl_debug/smoke_855073/`.
- **855080** submitted `2026-07-22 04:18 CDT` — 50 steps, eval every 5, file-backed NCCL INFO.
  - Job: `vista_moe30b_gsm8k_grpo_24n_50step_eval5` (exp dir `..._50step_eval5_2` due to name collision)
  - Config: `hpc/skyrl_yaml/vista/24node_qwen3_30b_a3b_gsm8k_grpo_50step_eval5.yaml`
  - `max_steps=50`, `eval_interval=5`, `eval_before_train=true`, `ckpt_interval=5`, `train_batch_size=16`
  - Log: `$SCRATCH/experiments/vista_moe30b_gsm8k_grpo_24n_50step_eval5_2/logs/`
  - NCCL files: `$SCRATCH/nccl_debug/nccl_%h_%p.log`; FR: `$SCRATCH/nccl_fr/nccl_trace_`


## Jupiter r13 (2026-07-25) — env ready, 50-step run LIVE
- **Blocker cleared:** lab vLLM fork built into `$DCFT/envs/rl` → `vllm 0.16.0 | reload OK`; `flash_attn 2.8.3` (cute disabled); `torchtitan`+`tyro 1.0.15` OK; `_StridedShard` patch present in `MarinSkyRL/.../distributed/fsdp_utils.py` (clone on `main` @ `36fdbc0`).
- **1041623** submitted `2026-07-25 ~09:29 CEST`, **RUNNING** on `booster` (open, no reservation), 4 nodes `jpbo-060-[17,29-31]`, wall `11:59:00`.
  - Job name: `jupiter_moe30b_gsm8k_grpo_4n_50step_eval5`
  - Config: `hpc/skyrl_yaml/jupiter/4node_qwen3_30b_a3b_gsm8k_grpo.yaml`; verified `max_steps=50`, `eval_interval=5`, `eval_before_train=true`, `ckpt_interval=5`.
  - Exp dir: `$SCRATCH/experiments/jupiter_moe30b_gsm8k_grpo_4n_50step_eval5/`; log under `.../logs/`.
  - WandB offline (`jupiter-moe-gsm8k-grpo`, entity `lukeleeai`) → sync from login after.
  - Watch: step-0 baseline `pass_at_1`, first `sync_weights` completing (r12's crash point), then eval curve at 0/5/10/…/50. Known risk: intermittent Vista-style early `sync_weights` hang.
  - **Nodes = open `booster`** (`jpbo-060-[17,29-31]`), NOT dev. `develbooster` is a reservation (8 nodes `jpbo-101-*`, 4-node/120-min cap) — same GH200 hardware, just short-wall policy; a 50-step run needs open booster's 12h wall. Fresh bigger run also goes on open booster.
  - **PROGRESS (2026-07-25):** step-0 baseline `pass_at_1=0.4556`; initial `sync_weights` 204.86s clean; steps 1–5 done (~880s/train-step, ~15min incl. amortized); **Vista-style hang did NOT reproduce** on 4-node Jupiter topology. `global_step_5` checkpoint written. `policy_train` ~525s/step (slow).
  - **Slow-step cause:** policy on only **8 GPUs** (EP4×FSDP2) + `cpu_offload=true` + `gradient_checkpointing` + `micro_train_batch=1` — all memory workarounds forced by the old 4-node (develbooster) cap. On open booster we can scale policy to 16 GPUs and drop `cpu_offload`.
  - **CHECKPOINT GOTCHA (verified in `fsdp_strategy.py`):** SkyRL FSDP2 ckpt is per-rank sharded `model_world_size_{WS}_rank_{R}.pt` via plain `torch.save`, loaded by exact `{WS}`+`{R}` name — **NOT DCP, cannot reshard.** Resume ONLY works into the SAME policy world_size. Cannot carry an 8-GPU ckpt into a 16-GPU mesh (→ FileNotFoundError). So a bigger run must start **from scratch** (step-5 weights are scientifically negligible anyway: 5 steps @ lr 1e-6).
  - **PLAN (executed):** 4-node was the sanity check → then scaled out on open booster.
- **4-node eval curve (1041623):** step0 `0.4556` → step5 `0.4625` → step10 `0.4420`. Noisy/non-monotonic, within Vista's 0.44–0.47 band; too few points for a trend. **Cancelled `1041623` on user OK (2026-07-25)** once 6-node was healthy (redundant, 2× slower).
- **6-node FAST (1042857):** `hpc/skyrl_yaml/jupiter/6node_..._fast.yaml`, 16 policy (EP4×FSDP4) + 8 infer TP=1, **`cpu_offload=false`**, 50 steps/eval5. **cpu_offload OFF WIN: policy_train 533s→113s (4.7×), step wall 891s→432s (2.06×), no OOM.** Baseline `pass_at_1=0.4511`.
  - **Bottleneck shifted:** now `sync_weights` 184s (43%) + `generate` 103s (24%); `policy_train` only 26%. Eval (~2191s/~36min every 5) is now the biggest wall-clock cost → eval-every-5 won't quite reach step 50 in 12h. **More POLICY GPUs (naive 12-node) ≈ no gain.**
- **8-node EVAL-OPT (1043206):** `hpc/skyrl_yaml/jupiter/8node_..._fast.yaml`, 16 policy + **16 infer** (TP=1), else same. Launched 2026-07-25, RUNNING on `jpbo-123-*`.
  - **NEGATIVE RESULT — doubling inference did NOT speed eval.** 8n eval ~49–55s/unit vs 6n ~52s/unit (~6%, projected ~2050s vs 2191s). **Eval is latency/decode-bound, not throughput-bound:** `eval_batch_size=32` doesn't even saturate 8 engines, so +8 more sit idle. **Real eval levers = bigger `eval_batch_size`, shorter eval `max_generate_length`, fewer eval samples, or `eval_interval=10` — NOT more inference nodes.** 8n is redundant/over-provisioned vs 6n; kill candidate (no scancel w/o user OK).
- **Vista vs Jupiter (matched 24-GPU / 16pol+8inf):** Vista step ~280s @ bs16; Jupiter 6n 432s @ bs32. Normalized per-sample: generate identical (~2× = batch), policy_train Jupiter ~20% FASTER/sample (NVLink EP locality), sync_weights Jupiter ~1.2× slower (fixed cost). Net: **Jupiter on par / better per-sample.**
- **Node limits:** booster ~5679 nodes (4071 idle at check); QOS `normal` + reformo/laionize assoc have NO hard per-user node/TRES cap — real governor is project node-hour budget + fair-share. 6–8 nodes trivial. Guardrail ≤6 running RL jobs.
- **Last verified jobs (2026-07-25 ~17:40 CEST):** `1042857` (6n fast) RUNNING; `1043206` (8n eval-opt) RUNNING; `1041623` (4n) cancelled. Re-check queue before acting.

## Megatron vs FSDP2 — PI switching RL defaults (see [[megatron_vs_fsdp2]])
- **PI Ben Feuer: "megatron curb stomps FSDP2 … switching our RL defaults!"** Megatron merged in MarinSkyRL (results in closed PR #7). Benchmark on TaskTrove `pymethods2test-large` (long-seq): total compute ~**9× faster** than fsdp2 (policy_train 6020s→600s ~10×). gsm8k UNDERSTATES the win (our policy_train small); fair test needs long-seq data.
- **SkyRL Megatron path is green on Jupiter** (`distributed/megatron/`, `workers/megatron/`, `config/megatron_config/`) using native env `/e/scratch/reformo/lee27/OpenThoughts-Agent/envs/rl-megatron`.
- **aarch64 blocker cleared:** built/installed Megatron Core 0.18.0, Megatron Bridge 0.5.0, TransformerEngine 2.11.0, torch 2.11.0+cu128, transformers 5.8.x, vLLM 0.22.0 in `envs/rl-megatron`.
- **Validation:** 30B Megatron train/generate/fwd/sync works on Jupiter (`1043650` completed 2 steps, exit `0:0`). Model-only checkpoint/resume is validated on the 6-node 4-vLLM geometry (`1045872` save smoke, `1045992` resume smoke). Full distributed optimizer-state checkpointing remains unvalidated.

## Relaunch only when asked
```bash
NUM_NODES=24 PARTITION=gh TIME_LIMIT=12:00:00 \
  RL_CONFIG=./hpc/skyrl_yaml/vista/24node_qwen3_30b_a3b_gsm8k_grpo.yaml \
  JOB_NAME=vista_moe30b_gsm8k_grpo_24n_eval10_ckpt5 \
  bash hpc/skyrl_standard/vista/run_gsm8k_moe30b_grpo.sh
```

## Working recipe
- Partition **`gh`**, `critic_num_gpus_per_node: 1`, `gpu_memory_utilization: 0.80`, `policy_strict_spread_pg: true`
- 16 policy EP=4×FSDP=4 + 8 vLLM
- **Eval/checkpoint:** `eval_before_train: true`, `eval_interval: 10`, `ckpt_interval: 5`, `seed: 42` — judge by **val**, not train `pass@4`

---

## JURECA Apptainer/OpenCode bridge — completed status (2026-07-29)

Full details are in `ai_memory/apptainer_bridge_handoff.md`.

- **Go/no-go passed:** cached `Qwen/Qwen3-0.6B` served by vLLM on one JURECA A100 drove OpenCode
  through the `dc-cpu` bridge for `astropy__astropy-12907`. OpenCode made successful Bash and Read
  tool calls and edited `/testbed`. No RL job was involved.
- **Forks pushed:** `lukedhlee/OpenThoughts-Agent` branch
  `lukedhlee/apptainer-opencode-bridge` at `bc8378ed`; `lukedhlee/harbor` on the same branch at
  `02b06bc5`. Both clean worktrees match their upstream refs.
- **Normal launcher wired:** bridge URL and exact Harbor ref are accepted, propagated, and validated
  fail-closed.
- **Small validation passed:** pristine oracle run was 3/3 reward 1 for Astropy, Django, and
  Matplotlib, with zero exceptions.
- **Confirmed subset cached:** Marianna's deployed pinned dataset is
  `DCAgent2/swebench-verified-random-100-folders` snapshot
  `a2e51e9e0e8029156ed340719eb8cc7ceee3ed1a`. All 100 corresponding SIFs pass
  `apptainer inspect`.
- **Inodes recovered:** prebuild now uses node-local temporary storage. The two exact cancelled-job
  GPFS temp trees were removed; scratch recovered from 100% to 44% inode use with 2,476,273 free.
- **Cluster left clean:** no user Slurm jobs or tmux bridge/proxy sessions remain.
- **Only unresolved item:** no authoritative second 100-task manifest was found. The public
  `sample100` overlaps the confirmed random100 by 18, so it cannot honestly be presented as an exact
  comparable 200. The pinned 100-task run is ready; do not claim a comparable 200 until provenance is
  obtained.
