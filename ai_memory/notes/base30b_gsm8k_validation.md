# Base-30B GSM8K RL-stack validation — state + runbook (2026-08-14, session 1)

SEPARATE WORKSTREAM from the r2egym/Marianna sweep (which runs in another session).
Operator: Luke. Goal: validate the RL stack end-to-end — GSM8K GRPO on
**Qwen/Qwen3-30B-A3B-Base**, ≥80 steps, clearly upward reward curve. This is INFRA
validation (decisions.md L141: gsm8k retired as MoE-science vehicle — do not report as science).
Mac plan file: `/Users/lukedhlee/.claude/plans/shimmying-juggling-turtle.md` (full staged plan).

## BINDING OPERATOR DECISIONS (do not re-litigate)

1. **Use Marianna's setup** = FSDP2 + her algorithm block verbatim: use_kl_loss true /
   kl_loss_coef 0.001, eps_clip 0.2/0.28, loss_reduction sequence_mean, wd 0.01,
   grad_norm 1.0, constant LR 0 warmup, n_samples_per_prompt 8, temp 1.0 / top_p 1.0 /
   top_k -1. Deliberate divergences (cited in plan): grpo not rloo_n; SYNC trainer
   (main_base); TIS ON (validated FSDP2 combo, job 1045840, needs generator.batched=true);
   greedy n=1 eval; gsm8k-domain ctx lengths (probe-decided), not her 40960/28000.
   Source of truth: her script copy `/e/fscratch/reformo/lee27/marianna_deepswe_repro_copy_0812/
   run_rl_deepswe_8b_repro_apptainer_seqmean_r2egym_learnable.sh` (line 321: strategy=fsdp2).
2. **FA gate RESOLVED 2026-08-14 — own env `$F/envs/rl-fa`; datasets-group ask NOT needed.**
   The "no torch2.11 aarch64 FA wheel" claim was WRONG (only the last 5 mjun0812 releases
   were checked): release **v0.9.22** ships `flash_attn-2.8.3+cu128torch2.11-cp312`
   manylinux_2_34 aarch64 — exact tag match for the validated stack, and Ben's otagent
   conda (ENVIRONMENT_MAP §2a) is standing proof FA-2.8.3-wheel + torch 2.11 works on
   Jupiter GH200. Built `$F/envs/rl-fa` (13G, uv venv py3.12.13, builder script + logs at
   `$F/envs/build_rl_fa_env.sh` / `build_rl_fa.log`): exact `--no-deps` replication of the
   rl-megatron freeze (`$F/envs/rl-megatron.freeze`; vLLM 0.22.0 official aarch64 PyPI
   wheel, torch 2.11.0+cu128) + git pins (torchtitan a1fdd7e, harbor 725fc069,
   dynamic-semaphore 4d5f49f) + skyrl-train/skyrl-gym editables re-pointed at
   `$F/repos/MarinSkyRL` + the FA wheel. Freeze diff vs rl-megatron = conda machinery
   dropped, `transformer_engine_torch` absent (Megatron-only, FSDP2 path never imports
   it), flash_attn added — otherwise identical.
   **Smokes PASSED:** login import smoke (flash_attn, vllm._C, fully_shard, _StridedShard,
   torchtitan, skyrl editables, ray — needs `module load GCC/14.3.0
   nvidia-compilers/25.9-CUDA-13` for vllm._C's libcudart.so.13, same as rl-megatron;
   hpc.py:862 already loads it in jobs) + GPU smoke job **1362688** COMPLETED on GH200
   (fa2 fwd/bwd, fa2-vs-sdpa max|diff| 0.004, varlen/sample-packing path, HF
   is_flash_attn_2_available, vllm._C — `$F/envs/rl_fa_smoke_1362688.out`).
   **Use:** `RL_VENV=$F/envs/rl-fa` +
   `RL_CONFIG=./hpc/skyrl_yaml/jupiter/6node_qwen3_30b_a3b_base_gsm8k_grpo_fsdp2_arms_fa.yaml`
   (committed; 3-knob diff: flash_attn true, use_sample_packing true, attn_backend auto).
   SDPA yaml kept as fallback. Her `/e/data1/datasets` env: no longer on the critical path.
3. **4 racing arms × 6 nodes** (PI approved up to 64 nodes; sweep coexistence fine):
   | JOB_NAME | POLICY_LR | extra | basis |
   |---|---|---|---|
   | base30b_gsm8k_lr1e6 | 1e-6 | — | DeepSWE-seqmean reference |
   | base30b_gsm8k_lr3e6 | 3e-6 | — | her script default |
   | base30b_gsm8k_lr8e6 | 8e-6 | — | her other paste = PI production value |
   | base30b_gsm8k_lr3e6_nokl | 3e-6 | USE_KL_LOSS=false | Luke: single-lever KL test vs arm B |
   Racing gate at step ~30: kill ONLY pathological arms (reward→0, entropy→~0.01, grad
   explosion); healthy-but-trailing arms run to ≥80. Never alter hparams mid-series.
   Same seed 42 across arms. Verdict: EMA-5 train reward + eval strict (n=1319, CI ±2.7)
   ≥ +10 pts by step ≥80 in ≥1 arm.

## Repo/branch state (all pushed to lukedhlee forks)

- **OT-Agent** `lukedhlee/vista-moe-grpo-30b` @ `d4b07e90`:
  `0e0489fc` arms YAML + launcher knobs + sidecar + probe band mode + July gsm8k assets
  finally tracked; `bed3cd1c` cluster hpc.py/jupiter.env tracked verbatim (+ ray_spill &
  OT_AGENT_RAY_LOG_DIR → fscratch delta); `d4b07e90` cache redirects + USE_KL_LOSS knob.
- **MarinSkyRL** `lukedhlee/jupiter-worktree-0814` @ `4f62f79`:
  `69f6d74` = verbatim snapshot of the Jupiter July-validated dirty tree (megatron ckpt +
  offline-tracking patches) on 36fdbc0; `4f62f79` = diag_utils + trainer hook + config keys
  (`trainer.diag_group_metrics`, `trainer.dump_train_rollouts`, default off, ON in arms yaml).
  Mac worktree: `/Users/lukedhlee/MarinSkyRL-jupiter-snapshot`.

## QUOTA CRISIS → fscratch relocation (THE central operational fact)

- `/e/scratch/reformo` at HARD inode limit (8.796M/8.8M) — ALL writes EDQUOT. Treat as
  **READ-ONLY** (venv, secrets, old caches still readable). Do NOT git-pull the old
  /e/scratch clones — abandoned as deploy targets. Project-level cleanup = meeting ask
  (mostly not our files; nezhurina1 exports etc.).
- `$HOME` (lee27) also over quota — vLLM torch-compile cache to ~/.cache killed probe
  attempt 1362190. Redirects now REQUIRED everywhere (baked into arms yaml; export for
  any manual job): VLLM_CACHE_ROOT / XDG_CACHE_HOME / TRITON_CACHE_DIR /
  TORCHINDUCTOR_CACHE_DIR / FLASHINFER_WORKSPACE_BASE → /e/fscratch/reformo/lee27/cache/*.
- Canonical layout `F=/e/fscratch/reformo/lee27`: repos/{OpenThoughts-Agent,MarinSkyRL}
  (fresh clones, the deploy targets), symlinks $F/OpenThoughts-Agent + $F/MarinSkyRL →
  repos/*, repos/OpenThoughts-Agent/envs → /e/scratch/.../OpenThoughts-Agent/envs
  (read-only venv), data/gsm8k/{train,validation}.parquet (copied), hf_hub (Base model),
  cache/*, wandb, experiments, ray_spill.

## Model + probe state

- Base snapshot: `$F/hf_hub/models--Qwen--Qwen3-30B-A3B-Base/snapshots/
  1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9` — 57G, 16/16 shards, chat_template ✓ (G1 PASSED).
- **Stage-2 probe job `1362199` RUNNING at handoff** (attempt 2; attempt 1 = 1362190 died
  to the $HOME cache quota). 1 node; full-val budget sweep (T=0.7, 4096 gen, rescored
  1024/2048) + BAND: pass@4 @ 20% subset @ T=1.0. Results JSON:
  `$F/experiments/jupiter_gsm8k_budget_probe/results/probe_1362199.json` — band key:
  aggregates{frac_all_wrong,frac_mixed,frac_all_correct,num_correct_histogram},
  token_lengths{mean,p50,p90,p99,max,frac_over_1024,frac_over_2048}.
- **G2 gates on the results:** G2a headroom — strict step-0 in [2%,60%]; ~0 everywhere incl
  flexible → HALT, inspect samples. G2b gen length — ≤10% over 1024 → keep 1024/2048;
  else 2048/3072 (MAX_GENERATE_LENGTH/MAX_MODEL_LEN launcher envs); ambiguous → round UP.
  G2c signal — band frac ≥10% good; 2–10% launch-but-expect-slow-start; <2% → add 1-shot
  `####` exemplar to the prompt (data prep change) + re-probe.

## Observability stack (built + deployed; validated by the smoke, not yet by a real run)

- Trainer-native (already logged): policy_loss/lr/entropy, ppo_clip_ratio, policy_kl,
  raw_grad_norm, avg_raw_reward, pass@8, tis/*, KL terms, timings.
- diag_utils (our MarinSkyRL commit): per-step `diag/*` — frac_groups_all_wrong/mixed/
  all_correct, group_reward_std_mean, phat_frac_k_of_8, length percentiles, truncated_frac;
  plus `<export_path>/diag_rollouts/step_N.jsonl` (uid, prompt, response, reward,
  stop_reason, len, tool_calls placeholder; oracle deliberately omitted — join via parquet).
- Native eval dumps (`dumped_evals/global_step_N_evals/*.jsonl`) carry prompt/response/
  score/stop_reason/env_extras(oracle) every eval_interval.
- Sidecar `scripts/analysis/gsm8k_diag_sidecar.py` (44-assertion self-test passed):
  run per arm on a LOGIN node tmux with the rl-megatron python (has wandb; login has
  internet → ONLINE wandb run `<arm>-diag`, group base30b_gsm8k_arms_r1, x-axis
  global_step). Computes parser-vs-model split (strict/flexible EM, malformed), lengths
  by correctness, acc-by-length buckets, rolling-window p̂ migration; writes
  `<experiment_dir>/diag_summary.json` — the cheap AI polling endpoint.
- WandB: project `jupiter-base30b-gsm8k-grpo`; trainer runs offline (sync loop from login
  ~10-15 min, `scripts/wandb/jupiter_sync_offline.sh`); AI access = fscratch JSONLs over
  ssh + wandb API from the Mac + diag_summary.json.

## Runbook — remaining stages in order

1. **Read probe** → apply G2a/b/c (above). **DONE 2026-08-14 — probe 1362199 COMPLETED (42:50):**
   strict@full = strict@1024 = strict@2048 = **60.50%** (798/1319, zero paired flips — answers land
   before token 1024; overage is post-answer rambling), flexible@full 49.96% (< strict: Base keeps
   generating after `####`, last-number extraction grabs ramble). Tokens: mean 1135, 20.1% at 4096
   cap (Base never EOSes), 28% >1024. Band (264 probs, pass@4 @ T=1.0): all_wrong 20.5% / **mixed
   74.6%** / all_correct 4.9%, grp_std 0.341; lengths p50 332 / p90 2133 / >2048 10.5%.
   **Gates:** G2c PASS (mixed ≫10%). G2b → T=1.0 lengths + round-UP rule ⇒
   **MAX_GENERATE_LENGTH=2048 / MAX_MODEL_LEN=3072**. G2a borderline: 60.5% sits AT the 60% ceiling
   — signal is strong (74.6% mixed groups), recommendation = proceed; flagged to Luke.
   JSON also copied to Mac scratchpad; full copy must ride the campaign harvest tar.
2. **FA gate: DONE (see item 2 above).** Env `$F/envs/rl-fa` + `..._arms_fa.yaml` smoked
   and committed. Stages 3-4 run with `RL_VENV=$F/envs/rl-fa` + the `_fa.yaml` config.
3. **Stage-3 smoke (2 steps, 6n, ~1.5-2h):** from `$F/repos/OpenThoughts-Agent`:
   `JSC_SCRATCH=$F RL_VENV=$F/envs/rl-fa DATA_DIR=$F/data/gsm8k EXPERIMENTS_DIR=$F/experiments \
    RL_REPO_DIR=$F/repos/MarinSkyRL WANDB_DIR=$F/wandb RESERVATION=none NUM_NODES=6 \
    RL_CONFIG=./hpc/skyrl_yaml/jupiter/6node_qwen3_30b_a3b_base_gsm8k_grpo_fsdp2_arms_fa.yaml \
    JOB_NAME=base30b_gsm8k_smoke MAX_STEPS=2 EVAL_BEFORE_TRAIN=true CKPT_INTERVAL=1 \
    TIME_LIMIT=01:45:00 bash hpc/skyrl_standard/jupiter/run_gsm8k_moe30b_grpo.sh`
   G3 gates: 8 engines up (≥12-min post-load patience — FlashInfer autotune silence);
   step-0 eval ≈ probe ±5 pts; raw_grad_norm finite/nonzero (ANCHORED grep, exclude
   max_grad_norm echo); avg_pass_at_8 > avg_raw_reward (group variance); tis/* emit sanely
   (imp_ratio≈1); KL finite; **diag/* metrics in wandb + diag_rollouts JSONL written +
   sidecar --once produces panels**; ckpt written; warm step time recorded → final wall
   math (>480s/step ⇒ eval_interval 20 and/or MAX_STEPS 80).
3a. **Smoke LAUNCHED 2026-08-14: job 1363397** (attempts 1363394/95 self-canceled during
   config debugging — 95 was actually fine). Verified rendered config: rl-fa python,
   Base snapshot, max_steps=2, ckpt_interval=1, eval_interval=5, gen 2048 / model_len
   3072 (G2b decision), FA on, lr 3e-6. **Launch-cmd gotchas fixed this session:**
   (i) `JSC_SCRATCH=$F` is MANDATORY — run script line ~125 forces SCRATCH=$JSC_SCRATCH
   (default /e/scratch) AFTER dotenvs; (ii) `hpc/dotenv/jupiter.env` + `jupiter.lee27.env`
   (sourced preferentially!) both relocated to fscratch (commits b3a4fd89, +lee27 sync);
   (iii) arms_fa.yaml env block had stale rl-megatron/old-scratch pins — fixed cb8cd2e9
   (RL_PYTHON/RL_ENV_DIR/DCFT_RL_ENV/LD_LIBRARY_PATH → $F/envs/rl-fa, secrets → $F/keys);
   (iv) secrets copied to $F/keys/secrets.env; (v) MODEL_PATH must be passed explicitly
   (auto-search looks for Instruct under HF_HUB_CACHE). Launcher relaunches rename the
   experiment dir (_2, _3…) — always re-resolve the dir from smoke_launch.log before
   inspecting sbatch/config.
3a-cont. **Smoke crash-fix ladder (all committed):** 1363397 died — FlashInfer fused-MoE
   JIT .so needs GLIBCXX_3.4.32 (GCC-14 module) but dlopen resolves old system
   /lib64/libstdc++ → engine crash-loop. Fix: `engine_init_kwargs.moe_backend: triton`
   in arms_fa.yaml (same call as the probe; FA attention unaffected) + wiped
   $F/cache/flashinfer. 1363512 died at 9 min — MarinSkyRL RefWorker built its
   HFModelWrapper WITHOUT moe_grouped_gemm/use_grouped_mm kwargs → KL ref model unswapped
   → EP=4 assert "no grouped MoE experts". July runs never exercised KL-with-ref.
   Fix: MarinSkyRL d9946bd on lukedhlee/jupiter-worktree-0814 (pass both kwargs from
   ref.fsdp_config). **Current smoke attempt: 1363765.**
   Provenance audit (2 subagents, 08-14): ALL prior MoE-30B runs (Ben's June yamls, July
   vista/jupiter campaigns) ran use_kl_loss=false — KL+ref+EP first combined in our
   08-13 arms yamls. Marianna = dense-only (32B/8B; her 8B DID use KL, no MoE ever).
   **Upstream marin-community main already fixed this same bug 08-07: a906145 (#339,
   Ben) via a shared `_fsdp_moe_model_kwargs()` helper that ALSO passes
   moe_router_replay to the ref.** Our branch diverged 07-21 (36fdbc0), predating it;
   d9946bd is an independent re-fix, behaviorally identical while
   moe_router_replay=false (our arms). TODO post-smoke: reconcile to the upstream shape
   (cherry-pick a906145 or add moe_router_replay passthrough) before the arms.
3b. Also start per-arm sidecar tmux + the wandb offline sync loop at arms launch.
4. **Stage-4 arms:** 4 launches, same command as smoke minus smoke overrides, plus
   JOB_NAME/POLICY_LR per the arms table (arm D adds USE_KL_LOSS=false),
   TIME_LIMIT=11:59:00. Gate at step ~30; prune ckpts keep-last-2/arm (~430GB each);
   babysit per [[babysit|monitor conventions]]: 15-min semantic check, entropy watch,
   EDQUOT scan, `sacct -j <id> -X` before any cause-of-death claim.
5. **Verdict + report** (frame INFRA), update this note + NEXT_SESSION + tracker; cleanup
   per rl-standard-job-cleanup; NO HF uploads.
6. **Harvest before the purge clock** (agent task, `crud-archive-run` + the standing rule in
   [[jsc_storage_map]] §Purge exposure): wandb sync confirmed → tar each run dir as ONE file →
   durable tier → checksum → rm. Hard deadlines from 08-14 mtimes: hf_hub snapshot purge-eligible
   ~09-10; experiments/results ~09-13 (30d fscratch); rl-megatron venv dies ~10-23 (90d scratch,
   not rebuildable in place — scratch inode-locked).

## 2026-08-14 overnight: G3 PASS + arms LIVE

**Re-smoke 1365265 (base30b_gsm8k_smoke2_2) = G3 PASS, all gates.** COMPLETED 0:0 in
1:01:47 on MarinSkyRL c5ca635 (surgical upstream-shape `_fsdp_moe_model_kwargs()` —
the a906145 reconciliation TODO above is DONE; full cherry-pick 8706320 was abandoned
after a NameError from bad conflict resolution killed 1364558 in 4 min).
Numbers: steps 469/521/452s (sync_weights fixed ~178s/step, NOT one-time);
reward 0.336→0.359→0.398; **eval@3 64.5% vs 57.8% step-0** (probe strict 60.5%);
KL 0.004–0.005, entropy 1.0→0.85, grad_norm 2.1–4.4, TIS ≈1.00.
**eval_batch_size 256 ⇒ eval 1183s (~20 min), 4× faster than bs=32.**
**Diag sanity PASS:** step-3 recompute from diag_rollouts JSONL == trainer diag/*
EXACTLY (mixed .9375 / all-wrong .0625 / p50 292 / trunc .0938 / full phat);
sidecar acc 0.6447 vs trainer 0.6452 (≤1 row). Quirks: end-of-training saves an extra
ckpt as global_step_N+1 (no extra train step); trainer offline wandb run lives in
`<exp_dir>/wandb/wandb/offline-run-*` (NOT $WANDB_DIR); `WANDB_MIRROR kind=eval` line
absent for end-of-run eval (dumped_evals JSONL is authoritative).

**wandb project override gotcha:** run script line 178 passes
`trainer.project_name=$WANDB_PROJECT` (default jupiter-moe-gsm8k-grpo) as a hydra
override that BEATS the yaml — WANDB_PROJECT must be set in the launch env. Smoke2
trainer run re-synced into the right project as id `smoke2-trainer`
(`wandb sync --no-skip-synced -p jupiter-base30b-gsm8k-grpo --id ...`).

**QOS cap:** partition MaxTime=UNLIMITED is a lie for scheduling — QOS `part_booster`
MaxWall=12:00:00 rejects >12h (`QOSMaxWallDurationPerJobLimit`, first arm volley
bounced at 16h). 11:59:00 is the real ceiling.

**Ckpt sizing:** 197GB/ckpt (6-node FSDP2 30B + optimizer). fscratch reformo at
30/42.9TB soft → CKPT_INTERVAL=40 for arms (~0.6TB/arm). Smoke2 left 2×197GB
(global_step_3/4) — reclaim at harvest.

**Arms LAUNCHED 11:0x, all RUNNING immediately (verified in rendered sbatch):**
| arm | job | POLICY_LR | kl |
|---|---|---|---|
| lr1e6 | 1366671 | 1.0e-6 | on |
| lr3e6 | 1366672 | 3.0e-6 | on |
| lr8e6 | 1366674 | 8.0e-6 | on |
| lr3e6_nokl | 1366675 | 3.0e-6 | **use_kl_loss=false** |

Common: 6 nodes, TIME_LIMIT=11:59:00, MAX_STEPS=80, **EVAL_INTERVAL=40,
EVAL_BEFORE_TRAIN=false** (wall math at 481s/step: 80 steps + evals@{0,40,80} does not
fit 12h; step-0 baseline already measured twice at 57.77%), CKPT_INTERVAL=40,
gen 2048/model_len 3072, WANDB_PROJECT=jupiter-base30b-gsm8k-grpo. Expected ~11.8h;
ckpt-before-eval ordering means worst-case walltime kill costs only the final eval.
Experiment dirs: `$F/experiments/base30b_gsm8k_<arm>_2` (the `_2` is real — the bounced
16h volley consumed the bare names). Launch logs: `$F/experiments/launch_logs/`.
Login-node tmux: `arms_sync` (10-min wandb sync loop, --no-skip-synced) +
`sidecar_<arm>` ×4 (watch mode, online). Per-arm 60s death-watchers running Mac-side.
Racing gate step ~30: kill ONLY pathological (reward→0, entropy→~0.01, grad explosion).

**12:20 incident — lr1e6 (1366671) backward-pass deadlock, py-spy'd live, killed, relaunched
as 1368006.** Log froze at 11:41 mid-step-4 (siblings at s8-9). py-spy per [[pyspy]] method:
driver idle in `ray.get`; **all 16 policy ranks in `_engine_run_backward`
(fsdp_strategy.py:186 backward) with GPUs pinned 100%** for 45+ min — symmetric NCCL/EP
collective deadlock (EP=4 all-to-all / FSDP2 reduce-scatter ordering race is the suspect;
routing-dependent). **Neither 30-min NCCL watchdog fired** (TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC
/ BLOCKING_WAIT both 1800s) — spinning collective kernels apparently keep the heartbeat
alive, so this hang class does NOT self-abort; only the log-silence + py-spy caught it.
Nodes jpbo-014-[18,24-25,27,31-32] (same set ran smoke2 cleanly → race > fabric; also
25/27 are the vLLM engine nodes — 0% util 85.9GB during train phase is NORMAL, don't
misread). SBATCH_EXCLUDE was silently dropped by the launcher pipeline — 1368006 landed
on the SAME nodes; accepted deliberately (recurrence would be diagnostic). Watch item:
if backward-hang recurs, hand-edit the rendered sbatch with --exclude and consider
ep_comm_backend/nccl-env mitigations. Sidecar lr1e6 restarted → dir `_3`, run
`base30b_gsm8k_lr1e6_v2`; arms_sync loop rewritten to resolve latest dir per arm each
cycle (no more stale `_2` pins).

**12:30 incident #2 — lr8e6 (1366674) SAME backward deadlock on a DIFFERENT nodeset**
(jpbo-115), frozen since 12:13 entering step-9 policy_train; py-spy identical (all ranks
`_engine_run_backward`, GPUs 100%). **Fabric theory dead → routing-dependent NCCL/EP race
in the grouped-GEMM/FSDP2 backward, ~1 hang per 2-3 arm-hours.** Caught in 17 min by the
new stall-watcher (log-mtime silence >15 min; threshold now tightened to 10 min).
Killed → relaunched as **1368098** (dir `_3`, sidecar run base30b_gsm8k_lr8e6_v2) with
**CKPT_INTERVAL=10** so future hangs resume instead of restarting (RESUME_MODE/RESUME_PATH
launcher knobs exist, run script lines 188-192). Standing relaunch policy for any arm that
hangs: kill, relaunch with CKPT_INTERVAL=10 (+RESUME_MODE=latest RESUME_PATH=<prev dir>/
checkpoints if a ckpt exists); healthy arms are NOT touched. Root-cause hunt (upstream
MarinSkyRL diff on EP backward, NCCL env mitigations) is a daylight task — flagged for
Luke/Ben, not an overnight experiment.

**13:12 incident #3 — lr3e6 (1366672) stalled at s15** (log frozen 12:59:58 entering
policy_train). Mixed cause: head node jpbo-060-05 refused srun task launch
("Communication connection failure") — sick node, likely triggered/accompanied the freeze.
Killed → relaunched **1368822** on fresh nodes jpbo-044-[07,11-15], CKPT_INTERVAL=10,
sidecar run base30b_gsm8k_lr3e6_v2 (dir `_3`). Stall-watcher caught it in 11 min.
Sweep-glob lesson (also fixed in sync loop + stall-watcher): `base30b_gsm8k_lr3e6*`
matches the NOKL dir — always resolve arm dirs with exact-name + `_[0-9]` globs.
Background subagent auditing upstream MarinSkyRL for post-07-21 EP/backward hang fixes.
Current fleet: lr1e6→1368006, lr3e6→1368822, lr8e6→1368098, nokl→1366675.

**13:2x — upstream audit verdict + watchdog root cause FIXED (729704dd).** Subagent
compared our snapshot vs marin-community main (e1c2dff, 08-13): EP all_to_all
dispatch/combine code is byte-identical — **upstream has NO root-cause fix for the hang
class** (their 449d1eb: "the initiating fault in natural training remains unknown"; they
hit the SAME 16-rank spin on the SAME Jupiter GH200s). But 449d1eb found WHY nothing
aborts: the legacy **`NCCL_BLOCKING_WAIT=1` alias selects a wait mode whose deadline
never fires**. WE inject it — `universal_rl.sbatch:167` (old Perlmutter debugging) +
`TORCH_NCCL_BLOCKING_WAIT_TIMEOUT_MS` in hpc.py's jupiter env. Removed both in
**729704dd** (pushed + cluster pulled): future (re)launches get functional watchdogs —
hang → abort ≤30 min (heartbeat 1800s) with FlightRecorder dumps (TRACE_BUFFER=2000
already on). Currently-running arms keep the old env; watchers remain their only cover.
Verify next rendered sbatch has no NCCL_BLOCKING_WAIT export.
Daylight port candidates from the audit: **eb1e229** (real numerics bug in our exact
moe_grouped_gemm path — `torch._grouped_mm` leaves ALIGN_SIZE_M pad rows uninitialized
→ eval/train forward divergence; our moe.py lacks the routed_rows zeroing) and
**ebed3f4** (collective phase diagnostics — localizes the diverging rank).

**14:17 incident #4 — lr1e6 (1368006) second backward hang at ~s11** (same py-spy
signature, jpbo-014 nodeset again — that arm/nodeset has now hung twice, but jpbo-115
also hung, so still race-not-fabric). Killed → **1369951** (dir `_4`, sidecar
base30b_gsm8k_lr1e6_v3), CKPT_INTERVAL=10, **first launch through the fixed template —
rendered sbatch verified clean of blocking-wait exports**, so this arm's watchdogs
should actually fire on the next hang (expect abort ≤30 min + FlightRecorder dump in
the job log — grep for "Flight Recorder" / ProcessGroupNCCL abort on its next death).
Hang tally: 4 incidents / ~7 arm-hours (lr1e6 ×2, lr8e6 ×1, lr3e6 ×1 sick-node).

**14:50 incident #5 — lr8e6 (1368098) froze at s15 → FIRST RESUME-FROM-CKPT relaunch:
1370390** resumes from `_3/checkpoints/global_step_10` (RESUME_MODE=from_path; only
s11-15 lost). Signature unconfirmed (I py-spy'd the WRONG nodeset — 1368098 ran on
jpbo-100-[43-48], not jpbo-115; evidence = 17-min log freeze). Verify resumes via
`configs/*_rl_config.json` hydra list, NOT the sbatch (overrides live there).
Tally: 5 incidents / ~8 arm-hours.

**15:0x incident #6 — lr3e6 (1368822) backward hang at ~s11** (py-spy confirmed same
signature, jpbo-044). Resume-relaunched as **1370635** from `_3/.../global_step_10`
(verified in configs json; dir `_4`, sidecar v3). Gotcha: the combined
confirm+cancel+relaunch heredoc DIED silently after its srun probe (ssh channel) —
1368822 kept running unnoticed until a state re-check; ALWAYS verify scancel took
effect with a separate squeue call before relaunching. Tally: 6 incidents / ~9 arm-hours.

**15:2x incident #7 — nokl (1366675) hung at s32/33** (frozen 14:59:55, jpbo-034), the
last original arm — CKPT_INTERVAL=40 meant **no ckpt, full restart** as **1370717**
(dir `_3`, sidecar v2, CKPT_INTERVAL=10, use_kl_loss=false verified). Its s32 gate-pass
data survives in wandb/diag exports. Ops lesson (hit twice): srun --overlap probes into
a hung job's nodes can WEDGE the whole ssh session — the combined heredoc dies silently
mid-script. Confirm hangs by log mtime + step-stop alone; py-spy only as a separate,
expendable call; ALWAYS verify scancel with an independent squeue check.
Tally: 7 incidents / ~10 arm-hours; every arm has now hung ≥1×.

**15:31 incident #8 — lr3e6-v3 (1370635) FAILED 21 min in, exit 1** — NOT a hang: all 4
vLLM engine processes on ONE engine node (10.128.32.158, jpbo-044 set) were killed at
OS level (OOM-killer/segfault per raylet) during `broadcast_to_inference_engines`, which
took the job down cleanly. Resume itself is EXONERATED — lr8e6-v3 did the identical
global_step_10 resume and is training (s12+). Relaunched **1371287** (dir `_5`, sidecar
v4, same resume). If jpbo-044's engine node kills it again → exclude that node.
Tally: 8 incidents / ~10 arm-hours (6 hang-class, 1 sick-node, 1 engine-node OOM/segv).

**15:5x incident #9 — lr8e6-v3 (1370390) CUDA OOM at ~s13**: trained s11-12 post-resume,
then a 384 MiB alloc failed inside the weight-sync `ProcessGroupNCCL::allgather` (96 MiB
free / 95 GiB GPU — policy+ref colocated + fragmentation). Both resumed jobs have now
died memory-flavored (lr3e6-v3 engine-node OS kill incl. possible host OOM; lr8e6-v3
device OOM) → **resume-memory-watermark hypothesis, 1 more data point wanted**: lr8e6
relaunched **1371443** (dir `_5`, sidecar v4) with the SAME resume. Second identical OOM
⇒ pattern confirmed ⇒ switch failed-arm relaunches to FRESH starts (no resume) and flag
resume memory behavior as a daylight bug. lr3e6-v4 (1371287, same resume) is the
parallel test. Tally: 9 incidents / ~11 arm-hours.

**16:1x — OOM ROOT CAUSE: ghost GPU memory on scancel'd deadlock nodes (fix b132d7aa).**
lr1e6-v3's fresh-start OOM broke the resume hypothesis; full OOM message shows 95GB at
capacity with only ~20GB attributed to live processes — **~75GB is ghost memory left by
the predecessor job killed mid-NCCL-spin**, and Slurm reallocated each relaunch onto its
predecessor's nodes (all 3 OOM'd jobs; nokl-v2 on fresh jpbo-026 runs clean). Fix:
appended the 6 dirty nodesets (jpbo-014/034/044/060/100/115 subsets, ~36 nodes) to
jupiter `node_exclusion_list` in hpc.py (b132d7aa, cluster pulled) — prune after JSC
resets them; consider reporting to JSC. AT-RISK: 1371287 (lr3e6-v4, on nokl's dirty
jpbo-034) and 1371443 (lr8e6-v4, on dirty jpbo-044) predate the fix — expected to OOM;
death-watchers armed, relaunch-with-exclusions + resume on death.
Tally: 10 incidents / ~11 arm-hours (6 hang, 1 sick-node, 3 ghost-memory OOM).

**16:0x incidents #11-12 + hypothesis shake-out.** nokl-v2 (1370717) OOM'd at s2 on
FRESH jpbo-026 nodes → ghost-memory can't be the whole story (unless another user's dead
job dirtied them — unverifiable). lr3e6-v4 (1371287) died 2 min into STARTUP
(WorkerCrashedError) on the wedged jpbo-034-18 (node refuses srun steps), then hung 25
min in Ray teardown — sick node, not race/OOM. **Counter-evidence that CLEARS the env
change: lr8e6-v4 (1371443, post-fix env, "dirty" jpbo-044) trains healthily past s11.**
Working model: random hang-race + several genuinely sick/ghost nodes; env change kept.
Fleet rebuilt with exclusions VERIFIED in rendered sbatch: lr3e6-v5 **1372782**
(resume@10, dir _6), lr1e6-v4 **1372785** (fresh, dir _5), nokl-v3 **1372789** (fresh,
dir _4), lr8e6-v4 **1371443** (s11+, dir _5). Tally: 12 incidents / ~12 arm-hours.

**16:4x incident #13 — WATCHDOG FIX VALIDATED IN ANGER.** lr8e6-v4 (1371443) hit the
hang-race at ~s16 and the fixed env **self-aborted it** (abort stacks show
`ProcessGroupNCCL::HeartbeatMonitor::runLoop()` — heartbeat monitor teardown), FAILED
1:03:25, death-watcher fired within a minute. Hang detection is now fully automatic
(was: 45-min manual py-spy hunt). lr8e6 keeps dying pre-s20 (s13, s16) so it never banks
a new ckpt → relaunched **1374136** (dir _6, sidecar v5) resume@10 with
**CKPT_INTERVAL=5** to bank progress inside the groundhog window.
Tally: 13 incidents / ~13 arm-hours.

**14:55 sweep — nokl PASSES the step-30 racing gate** (s32: rew 0.805 rising, ent ~0.35
stable, trunc ~1-3%). lr8e6-v3 resume VERIFIED in-log (resume_mode from_path,
global_step_10). No arm pathological; no kills at the gate.

## Open/parked items

- Sweep-shards mystery RESOLVED: the other session's sweep was operator-PARKED 08-14
  (~18:15 PT) after 3 failed rounds — see NEXT_SESSION `0.0-SWEEP-PARKED`. Not our concern.
- Optional: wandb-workspaces saved view (Reward/Learnability/Parser/Lengths/Stability).
- ~~FA long-term: torch-2.9/cu130 coherent env rebuild~~ MOOT — `$F/envs/rl-fa` (torch
  2.11+cu128 + exact-tag FA wheel) resolved it 2026-08-14; no torch downgrade needed.
- /e/scratch/reformo + lee27 $HOME quota cleanups (meeting asks; nothing for this run).
