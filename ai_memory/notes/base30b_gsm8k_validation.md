# Base-30B GSM8K RL-stack validation — state + runbook (2026-08-14, session 1)

SEPARATE WORKSTREAM from the r2egym/Marianna sweep (which runs in another session).
Operator: Luke. Goal: validate the RL stack end-to-end — GSM8K GRPO on
**Qwen/Qwen3-30B-A3B-Base**, ≥80 steps, clearly upward reward curve. This is INFRA
validation (decisions.md L141: gsm8k retired as MoE-science vehicle — do not report as science).
Mac plan file: `/Users/lukedhlee/.claude/plans/shimmying-juggling-turtle.md` (full staged plan).

## RESUME-HERE — new-session bootstrap (updated 2026-08-14 ~22:40 CEST, session 2)

**Task**: babysit 4 GRPO arms to ≥80 steps each, then verdict (INFRA framing) + harvest.
Racing gate (s30) DONE — all arms passed, zero kills. Reward curves clearly upward:
lr1e6 0.34→0.38 (s10), lr3e6 →0.70 (s29), lr8e6 →0.85-0.95 (s39), nokl →0.77 (s32).
**lr8e6 eval@40 = 88.40% greedy pass@1 (n=1319) vs 57.77% step-0 = +30.6 pts — the
verdict criterion (≥+10 pts in ≥1 arm) is already exceeded at s40** (formal verdict
still waits for s80 per BINDING §3). Eval took 564s (~9.4 min — short generations),
ckpt@40 banked (gs25/30/35/40). Session-2 bootstrap DONE 22:30: watchers restarted
(4 death + 1 stall, Mac-side), login tmux verified (arms_sync + 4 sidecars alive).
Eval facts: dumps at `<dir>/<job_name>/exports/dumped_evals/global_step_N_evals/`
(aggregated_results.jsonl has the score; openai_gsm8k.jsonl has n=1319 rows); NO
`WANDB_MIRROR kind=eval` line — the step-N train mirror logs right after eval ends.
WandB (all runs, group by name, take newest per arm):
https://wandb.ai/lukeleeai/jupiter-base30b-gsm8k-grpo

**Fleet as of 23:30 CEST** (job / newest dir under `$F/experiments/` / resume source):
| arm | job | dir | resumed from |
|---|---|---|---|
| lr1e6 | 1381632 | base30b_gsm8k_lr1e6_25 | _13/ckpt gs20 |
| lr3e6 | 1381636 | base30b_gsm8k_lr3e6_18 | _16/ckpt gs60 |
| lr8e6 | 1381493 | base30b_gsm8k_lr8e6_17 | _15/ckpt gs60 |
| lr3e6_nokl | 1380985 | base30b_gsm8k_lr3e6_nokl_9 | _8/ckpt gs75 |

ETAs to s80 at ~7.5 min/step (if no more incidents): lr8e6 ~02:30, nokl ~04:30,
lr3e6 ~05:00, lr1e6 ~07:00+. Eval@40 + ckpt@40 land automatically (EVAL_INTERVAL=40,
step-0 baseline 57.77% greedy; lr8e6's eval@40 was in-flight at handoff).

**FIRST ACTIONS in a new session** (Mac-side watchers die with the old session):
1. Restart per-job 60s death-watchers (background `squeue -h -j <id>` poll → sacct on
   exit) + ONE stall-watcher (per-arm log-mtime >600s while RUNNING → alert; resolve
   dirs with EXACT globs `..._$a`, `..._${a}_[0-9]`, `..._${a}_[0-9][0-9]` — a bare
   `lr3e6*` glob matches the NOKL dir).
2. Sidecars + sync loop live in LOGIN-NODE tmux (survive sessions): `arms_sync` +
   `sidecar_<arm>` ×4. Verify with `tmux ls`; sync loop pushes offline runs every 10 min
   (`<dir>/wandb/wandb/offline-run-*`, `wandb sync --no-skip-synced`; live-file EOF
   errors are cosmetic).

**Incident runbook (the whole night was this loop, 27 incidents):**
- STALL alert → confirm log mtime + last `WANDB_MIRROR kind=train step=` → `scancel`
  → **verify with separate squeue** (heredocs died mid-script twice) → **immediately
  add the job's nodeset to `node_exclusion_list` in hpc.py** (jupiter cluster, ~line
  1005; commit+push+`git pull` on cluster — kills/aborts strand ~75GB ghost GPU memory
  and Slurm re-hands the nodes) → relaunch resuming from the NEWEST `global_step_N`
  across the arm's dirs → restart that arm's sidecar tmux (new dir, bump -vN run name)
  → new death-watcher + rebuilt stall-watcher.
- Death (job left queue, FAILED): same minus scancel. OOM-at-load = ghost node
  (exclude); `HeartbeatMonitor::runLoop` frames = watchdog abort of the EP hang.
- The EP/FSDP2 backward hang race (~1 per 2 arm-hr, no root fix upstream) + sick/ghost
  nodes are the only real enemies. NEVER py-spy/srun-probe INTO a hung job's nodes from
  a combined script — it wedges the ssh session. All current jobs run blocking-wait env
  (revert 5f9bcb44) → hangs do NOT self-abort → stall-watcher is the only hang net.

**Relaunch command template** (from `$F/repos/OpenThoughts-Agent`, F=/e/fscratch/reformo/lee27;
drop RESUME_* if no ckpt; POLICY_LR/JOB_NAME per arm; nokl adds USE_KL_LOSS=false):
```
env JSC_SCRATCH=$F SCRATCH=$F DCFT=$F/repos/OpenThoughts-Agent \
  MODEL_PATH=$F/hf_hub/models--Qwen--Qwen3-30B-A3B-Base/snapshots/1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9 \
  RL_VENV=$F/envs/rl-fa RESERVATION=none NUM_NODES=6 \
  RL_CONFIG=$F/repos/OpenThoughts-Agent/hpc/skyrl_yaml/jupiter/6node_qwen3_30b_a3b_base_gsm8k_grpo_fsdp2_arms_fa.yaml \
  WANDB_PROJECT=jupiter-base30b-gsm8k-grpo \
  JOB_NAME=base30b_gsm8k_<arm> POLICY_LR=<lr> \
  MAX_STEPS=80 EVAL_INTERVAL=40 EVAL_BEFORE_TRAIN=false CKPT_INTERVAL=5 \
  RESUME_MODE=from_path RESUME_PATH=<.../checkpoints/global_step_N> \
  TIME_LIMIT=11:59:00 MAX_GENERATE_LENGTH=2048 MAX_MODEL_LEN=3072 \
  bash hpc/skyrl_standard/jupiter/run_gsm8k_moe30b_grpo.sh
```
Verify a launch via `<dir>/configs/*_rl_config.json` (hydra args incl. resume/lr) — NOT
the sbatch. Launcher renames dirs `_N`; always re-resolve newest.

**After all arms hit 80 (or walltime)**: final `wandb sync` per arm; verdict per
BINDING DECISIONS §3 (EMA-5 train reward + eval strict ≥+10pts by s80 in ≥1 arm —
lr8e6/nokl already far past this on train reward, confirm with eval@80); write verdict
into this note; **prune ckpts keep-last-2 per arm** (interval-5 banking left many
197GB ckpts — check `du`); cleanup per rl-standard-job-cleanup; NO HF uploads; harvest
per §Stage 6 / [[jsc_storage_map]]; report the sick-node/ghost-memory saga to JSC
(node list = the 2026-08-14 additions in hpc.py node_exclusion_list; commits b132d7aa,
c6986883..). Daylight code tasks: port upstream eb1e229 (grouped-GEMM pad-row numerics
bug) + ebed3f4 (collective phase diagnostics); memory-headroom work to re-enable
watchdog env (449d1eb analysis).

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

**17:0x incidents #14 + THE BLOCKING-WAIT VERDICT (revert 5f9bcb44).** lr1e6-v4
(1372785) OOM'd at s2 on definitively FRESH nodes — 5th OOM, ghost-node theory dead.
Final pattern: **every post-729704dd OOM victim is a LONG-generation arm**
(lr1e6/lr3e6/nokl, p50 300-450 tok); the only survivor lr8e6 generates short (ent~0.3).
Mechanism: without NCCL_BLOCKING_WAIT the CPU thread races ahead and FSDP2's allgather
pipeline holds more concurrent buffers → +few GB peak → long-gen arms (near the 95GB
ceiling) OOM at s1-13; short-gen fits. GH200 NVML per-process attribution is garbage
(20GB attributed on a genuinely full device) — trust cudaMemGetInfo, not the process
list. **REVERTED 5f9bcb44** (blocking-wait restored, exclusions kept): watchdog
self-abort is sacrificed; stall-watchers resume hang duty. Daylight task: buy memory
headroom (micro-batching, gpu_memory_utilization, activation offload) before retrying
watchdog mode. Fleet: lr1e6-v5 **1374467** (reverted env, verified in sbatch),
lr3e6-v5 1372782 (s12+, condemned env but surviving), lr8e6-v5 1374136 (startup),
nokl-v3 1372789 (s2+). Condemned-env jobs stay up while healthy; watchers decide.
Tally: 14 incidents / ~13 arm-hours.

**17:10 sweep + incidents #15-16.** lr1e6-v5 (1374467, reverted env) died 10 min in —
worker OS-kill during engine bring-up on jpbo-012-[34-36,38,41,45] (logged, NOT excluded
yet — one strike). Relaunched lr1e6-v6 **1374955** (dir _7, CKPT_INTERVAL=5). nokl-v3
(1372789) hung at s3 on the condemned env and **the watchdog self-aborted it — 2nd
in-anger validation** — deliberately NOT scancel'd (kills on spinning ranks are what
strand ghost memory). lr3e6-v6 **1374664** (dir _7) + lr1e6-v6 in clean startup on
reverted env. lr8e6-v5 (1374136, condemned env, short-gen) thriving: s11 rew 0.930,
ent 0.20, KL 0.077 — near train ceiling, gate call at s30. Standing rule: hung jobs on
the condemned env get ~35 min for watchdog self-abort before any scancel; reverted-env
hangs need scancel (watchdog blind) — accept the ghost risk, exclusions absorb it.
Tally: 16 incidents / ~14 arm-hours.

**17:5x — SICK-NODE STORM confirmed, blocking-wait theory DEAD, fleet cycled again.**
lr1e6-v6 OOM'd at s1 ON THE REVERTED ENV → the NCCL_BLOCKING_WAIT revert did NOT cause/
fix the OOMs (revert 5f9bcb44 stays anyway — harmless, pre-fix-proven). Real story:
**a pool of sick/ghost-memory GPUs circulating in the idle pool** (any user's crashed EP
job seeds them); relaunches roll dice. Direct evidence: jpbo-009-03 (killed lr1e6-v6)
went DOWN/DRAINED minutes later — JSC health automation drains them one victim late.
Attrition 15:00-18:00 ≈ 70% infant mortality; expect it to subside as the pool drains.
lr3e6-v5/v6 deaths = same worker-OS-kill class (ActorDiedError, no OOM/heartbeat lines).
**lr8e6-v5 (1374136) broke through: s15 done, ckpt global_step_15 BANKED** (future lr8e6
resumes from 15). The "resume jobs die s12-16" pattern was storm coincidence.
Fleet: lr1e6-v7 1375864 (fresh, dir _8), lr3e6-v7 1375865 (resume@10, dir _8),
lr8e6-v5 1374136 (s15+, jpbo-003), nokl-v4 1375221 (startup, jpbo-068).
Tally: 18 incidents / ~15 arm-hours.

**19:0x incident #19 — lr8e6-v5 hang at s23-24, watchdog abort #3, RESUME LADDER WORKS.**
Hung after s23 (18:05 silence), heartbeat monitor self-aborted at 2:24:23 elapsed
(~30 min, on schedule). Ckpt ladder paid off: gs15 AND gs20 banked → relaunched
**1376820** (dir _7, sidecar v6) resuming from **global_step_20** — net loss only s21-23.
lr8e6 cumulative: s1-16 (v1) + 10-23 (v3/v4/v5 replays) → now monotone from s20 with
5-step banking. 19:00 fleet: lr1e6-v7 s6 (0.375), lr3e6-v7 s16 (0.656), nokl-v4 s11
(0.555) — post-storm calm, all healthy. Tally: 19 incidents / ~17 arm-hours.

**19:2x incident #20 — WATCHDOG ABORTS GHOST NODES TOO.** lr8e6-v6 (1376820) OOM'd at
MODEL LOAD (20MiB alloc, 39MiB free — GPU full at job start) after landing on 5/6 of its
watchdog-aborted predecessor's nodes. So BOTH scancel and heartbeat self-abort strand
GPU memory when ranks die inside spinning NCCL kernels; only JSC drain/reset clears it.
**New protocol: exclude every hang site immediately at death, BEFORE relaunching**
(c6986883 adds jpbo-003 + nokl-v3's jpbo-102/103/122 sites; exclusion list now ~100
nodes — prune when JSC resets). lr8e6-v7 **1376990** (dir _8, sidecar v7, resume@gs20).
Tally: 20 incidents / ~17 arm-hours.

**19:30-20:05 — attrition cycles #21-24, lr1e6 PARKED.** lr1e6 suffered 4 straight
infant deaths (v7 hang s9 → banked its FIRST ckpt gs5; v8 ghost-OOM at load jpbo-020;
v9 ghost-OOM jpbo-022) → **parked ~90 min** (triage: lr3e6/lr8e6/nokl carry the
campaign; unpark timer set). lr8e6-v7 hung in startup (jpbo-069) → v8 **1377151**
resume@gs20 now s21+ on jpbo-112. lr3e6-v7 reached **s24, banked gs15+gs20** before
hanging (jpbo-105) → v8 **1377427** resume@its own gs20. nokl-v4 healthy **s21,
banked gs5+gs20** (one clipped grad spike 40 — watching). Exclusions grown per
protocol: jpbo-046/069/020/105/022 sites (c6986883..). 20:05 fleet: 3 arms active
(nokl s21 the from-scratch leader), lr1e6 parked. Cumulative best steps: nokl 21,
lr3e6 24, lr8e6 23, lr1e6 9. Tally: 24 incidents / ~19 arm-hours.

**21:05 sweep 11 — lr8e6 PASSES the s30 racing gate; ALL ARMS PASS.** lr8e6-v8 at s30:
rew 0.891, ent 0.166 (plateaued, not collapsing), KL 0.072 bounded → healthy fast-
converger, no kill. Gate summary: nokl PASSED (s32, morning run), lr8e6 PASSED (s30),
lr3e6 healthy s21 (0.688) — gate moot for it and lr1e6 (no pathology). **Zero arms
killed at the gate.** Attrition cycles #25-27: lr3e6-v8/v9 ghost-OOMs (jpbo-020 AGAIN
+ jpbo-100 disjoint subset → BOTH racks now excluded rack-level), nokl-v4 hang at
**s28** with gs25 banked → v5 1378378 resume@25. lr1e6 UNPARKED as v10 1378384
(resume@gs5). Fleet 21:05: lr1e6 1378384, lr3e6 1378002 (s21+), lr8e6 1377151 (s30+),
nokl 1378378. Cumulative best: lr8e6 30, nokl 28, lr3e6 24, lr1e6 9.
Tally: 27 incidents / ~21 arm-hours.

**22:04 sweep 12 — QUIET HOUR, all arms at new high-water marks.** lr1e6 s10 (0.383),
lr3e6 s29 (0.703), lr8e6 s39 (0.852, eval@40 imminent — first greedy-acc readout vs
57.8% baseline), nokl s32 (0.773). Zero incidents in the hour — the rack-level
exclusions + drained pool are holding. Upward reward curves now unambiguous on all
four arms. Tally still 27 / ~22 arm-hours.

**22:30 sweep 13 (session 2 start) — bootstrap done, lr8e6 eval@40 = 88.40% (+30.6).**
All 4 arms RUNNING and fresh (lr1e6 s12, lr3e6 s31, lr8e6 s41 rew 0.875/ent 0.165/
KL 0.083, nokl s35). Watchers rebuilt Mac-side (4 death + 1 stall, scripts in session
scratchpad); login tmux intact. lr8e6 eval@40: 88.40% pass@1 vs 57.77% baseline —
+30.6 pts at half-way; ckpts gs25-40 banked. Quiet since 21:05 — zero new incidents.
Tally holds 27 / ~22 arm-hours.

**22:3x incidents #28-29 — DOUBLE backward hang (lr1e6 + lr3e6, ~simultaneous).**
lr1e6 (1378384) froze 22:21:51 mid-s13 policy_train (jpbo-122-[20,23,26-29]); lr3e6
(1378002) froze 22:20:24 at s32 batch 10/32 (jpbo-067-[06,08-12]). Both on reverted
env (no self-abort); stall-watcher caught both in ~11-13 min. lr8e6/nokl unaffected
(fs fine). scancel'd + verified via separate squeue → both nodesets excluded
(4749bce3, cluster pulled) → relaunched: **lr1e6 1379097 (dir _12, resume@gs10),
lr3e6 1379098 (dir _12, resume@its gs30 — only s31 lost)**. Configs verified (resume
paths, LR overrides, exclusions in sbatch, blocking-wait present). Sidecars restarted
(v11 ×2); death-watchers + stall-watcher rebuilt. New fleet: lr1e6 1379097,
lr3e6 1379098, lr8e6 1377151 (s41+), nokl 1378378 (s35+).
Tally: 29 incidents / ~23 arm-hours.

**22:5x-23:2x incidents #30-32 — restart storm on the relaunches; nokl the lone survivor.**
#30: lr3e6-v11 (1379098) ghost-OOM at ckpt load 14:45 in (jpbo-030, 37MiB free) →
excluded (585764dc) → v12 **1379307** resume@gs30 (dir _13, sidecar v12).
#31: lr8e6-v8 (1377151) backward hang mid-s44 (last mirror s43, froze 22:42:45,
jpbo-112) → scancel+verified → excluded (50b42140) → v9 **1379329** resume@gs40
(dir _10, sidecar v9; only s41-43 lost, eval@40 + gs40 safely banked).
#32: lr1e6-v11 (1379097) ghost-OOM at first ppo_train 21 min in (jpbo-025, 58MiB
free) → excluded (f8929615) → v12 **1379349** resume@gs10 (dir _13, sidecar v12).
Fleet 23:25: lr1e6 1379349, lr3e6 1379307, lr8e6 1379329 (all PENDING→starting),
nokl 1378378 RUNNING s39. Cumulative best: lr8e6 43, nokl 39, lr3e6 31, lr1e6 12.
Tally: 32 incidents / ~24 arm-hours (16 hang / 11 OOM / 6 node per attempt table).
**23:05 sweep 13: queue turned BUSY** — booster 4354 alloc / 1236 planned / 13 idle;
our 3 relaunches are TOP of pending (prio 350057, nobody ahead) with Slurm start
estimate **01:36**. NOT a config problem — evening instant-starts are over. Wall
math from 01:36 still fits all arms (worst lr1e6: 70 steps ≈ 8.8h → ~10:45).
nokl healthy s39 rew 0.75 ent 0.39. Expect a ~2.5h quiet gap; death-watchers idle
on PENDING (normal), stall-watcher ignores PENDING dirs.
**23:2x — nokl eval@40 = 85.22% greedy pass@1 (+27.4 vs 57.77 baseline), ckpt@40
banked (gs30/35/40), s40 train mirror logged, job healthy.** Second arm past the
+10 verdict bar — and it's the NO-KL arm. Its KL-on comparator is lr3e6's eval@40
(fires after 1379307 starts and passes s40). Scoreboard: eval@40 lr8e6 88.40,
nokl 85.22, baseline 57.77.
**00:2x-00:4x incident #33 — nokl-v5 (1378378) backward hang mid-s53** (progress
emitter froze 00:18:04 at policy_train 8/32, log silent 786s+, jpbo-081-[10-15];
gs50 banked → ≤3 steps lost). scancel+verified → excluded jpbo-081-[10-15]
(d974514e; note: push goes to remote `fork`, not `origin`) → v6 **1379844**
resume@gs50 (dir _7, sidecar v6; verified in config: from_path gs50, lr 3e-6,
use_kl_loss=false, exclusion in sbatch). 1379844 PENDING behind the same busy
queue as the other three (start est was 02:32 for those). Death-watcher b47nablpt,
stall-watcher bthrf4lde rebuilt. Cumulative best: lr8e6 43, nokl 52, lr3e6 31,
lr1e6 12. Tally: 33 incidents / ~25 arm-hours.
**00:42 — queue broke early: ALL FOUR arms started** (est was 02:33). New nodesets:
lr3e6 jpbo-003-[20,...], lr8e6 jpbo-054, lr1e6 jpbo-066, nokl jpbo-107.
**00:56 incident #34 — lr3e6-v12 (1379307) ghost-OOM at engine load 13:55 in**
(jpbo-003-[20,23-24,26-27,30], GPU had 18.88 MiB free of 95 GiB pre-load; different
sub-range of already-partially-excluded jpbo-003 rack) → excluded (8c18896e) →
v13 **1380047** resume@_11/gs30 (dir _14, sidecar v13; config verified: from_path
gs30, lr 3e-6, exclusion in sbatch). Death-watcher blgvr1n46. Other three arms
survived their load window (RUNNING 15+ min). Tally: 34 / ~25 arm-hours.
**02:30 incident #35 — nokl-v6 (1379844) backward hang at s61** (froze 02:19 entering
policy_train, jpbo-107-[07-12]; **gs60 banked** — net +10 steps that attempt) →
excluded (ccecd14e) → v7 **1380775** resume@gs60 (dir _8, sidecar v7, verified).
nokl needs only 20 more steps. Death-watcher booaycjnb, stall-watcher b24r64uhc.
Cumulative best: nokl 61, lr8e6 43+ (running 1:49), lr3e6 31+, lr1e6 12+.
Tally: 35 / ~26 arm-hours.
**02:45 incident #36 — lr8e6-v9 (1379329) backward hang at s55** (last mirror s54,
froze 02:24:40 in ref-forward dispatch, jpbo-054-[01-03,09,12-13]; gs50 banked, s51-54
lost — that attempt still netted +10) → excluded (37f49f0e) → v10 **1380786**
resume@gs50 (dir _11, sidecar v10, verified: from_path gs50, lr 8e-6, exclusion in
sbatch). nokl-v7 1380775 RUNNING (started 02:4x, booting). Death-watcher bwah069ja,
stall-watcher bibbltqgh. Cumulative best: nokl 61, lr8e6 54, lr3e6 31+, lr1e6 12+.
Tally: 36 / ~26 arm-hours. Probe HANDED OFF to second session —
ai_memory/notes/currease_pass8_probe_handoff.md (865335c5); this session does NOT
relaunch it.
**02:55 incident #37 — lr1e6-v12 (1379349) backward hang at s23** (last mirror s22,
froze 02:35:50 in policy_train, jpbo-066-[17-18,20-22,24]; gs20 banked — that attempt
netted +10) → excluded (203f6e01) → v13 **1380812** resume@gs20 (dir _14, sidecar v13,
verified: from_path gs20, lr 1e-6, exclusion in sbatch). Death-watcher besvlmrdt,
stall-watcher bjr5pzg4f. Fleet: lr3e6 RUNNING 1:50 (approaching s45+), lr8e6-v10
RUNNING 10 min, nokl-v7 RUNNING 16 min, lr1e6-v13 PENDING. Cumulative best:
nokl 61, lr8e6 54, lr3e6 31+, lr1e6 22. Tally: 37 / ~27 arm-hours.
**03:1x OPERATOR-APPROVED mid-campaign config change (2f5ad088): arms yaml
`ref.fsdp_config.cpu_offload: false→true`** (e9fc5e8 semantics; placement-only, loss
math identical; policy offload stays false; nokl has no ref → unaffected). Takes
effect at each KL arm's NEXT incident relaunch — do NOT scancel healthy runs for it.
First relaunch with it = first-time combo on our branch (ref offload never smoke-tested
here): watch ref init closely; if it crashes at ref build, revert yaml + relaunch.
JSC ticket draft for the ghost-memory nodes: ai_memory/notes/jsc_ghost_gpu_memory_ticket.md.
**03:0x incident #38 — lr8e6-v10 (1380786) ghost-OOM at ckpt load 14:12 in** (8×
OutOfMemoryError in FSDPPolicyWorkerBase.load_checkpoint, jpbo-014-[17,19,23,26,28-29]
— rack 014's OTHER nodes; second disjoint hit → **jpbo-014 escalated to rack-level
[01-48]** (72f1252d), like jpbo-100 precedent) → v11 **1380813** resume@gs50 (dir _12,
sidecar v11). **v11 is the FIRST job carrying ref.cpu_offload=true** (verified in
config: policy false + ref true) — watch ref init at start. Death-watcher bxbxg0fja.
Fleet 03:1x: lr3e6 1:56 (longest current streak), lr1e6-v13 RUNNING 7 min,
nokl-v7 RUNNING 23 min, lr8e6-v11 PENDING. Tally: 38 / ~27 arm-hours.
**03:2x incidents #39+#40 (double)**: #39 lr1e6-v13 (1380812) ghost-OOM at ckpt load
19:43 in (jpbo-033-[33-36,45,48], 666MiB free; never resumed — 0 steps lost) →
excluded → v14 **1380821** resume@_13/gs20 (dir _15). #40 lr3e6-v13 (1380047)
backward hang at s43 (froze 03:00:25 mid-policy_train; **gs40 + eval@40 banked this
attempt**, s41-42 lost) → scancel+verified → excluded jpbo-013-[17-18,20,23,25-26]
→ v14 **1380822** resume@_14/gs40 (dir _15). Both exclusions in 023a48c1. Both v14
configs verified incl. **ref cpu_offload=true** (jobs 2-3 carrying the flip; lr8e6-v11
was 1st). Sidecars v14 ×2; death-watchers bph7acgof (lr1e6), bd8t5f978 (lr3e6).
lr3e6 cumulative best now 42 (+eval@40 dump exists in _14 exports — HARVEST IT).
Tally: 40 / ~28 arm-hours.
**PEER-SESSION FINDING (probe session, 03:2x): compute→login ssh DEMANDS TOTP even
from internal 10.x source** (tested from jpbo-013-17; JSC docs: account-level TOTP
fires after pubkey regardless of source; only FIDO/ssh-certs exempt) → the sbatch
SOCKS tunnel is dead-on-arrival for lee27 as-is. Peer's fallback: login-side listener
(login→compute reverse ssh works, no TOTP on compute sshd) + tiny additive hpc.py
_setup_proxy change to honor an existing listener. Luke checking JuDoor TOTP disable
(feuer1-parity). My bridge tmux (harbor apptainer server on 10.128.1.2:9920) PROVES
compute→login-listener TCP works without ssh — simplest fix is a login-side SOCKS5
server + preset PROXYCHAINS_SOCKS5_HOST/PORT.
**03:5x incident #41 — lr8e6-v11 (1380813) OOM at FIRST ppo_train, 19 min in**
(8 OOMs, 607MiB free, jpbo-008-[18,23,26-27,29,32]) — the first ref-offload run;
it loaded clean, generated, computed rewards (0.836), then OOM'd in the first
policy_train backward. **AMBIGUOUS: partially-dirty node (cf. #32 same signature,
pre-offload) vs the offload flip itself. A/B in flight**: lr1e6-v14 (1380821) +
lr3e6-v14 (1380822) carry the same flip on other nodes — if their first train steps
pass, it was the node; if they OOM too, REVERT yaml cpu_offload→false and relaunch
all three (flip needs upstream code, not just config). Excluded (0435c0c0) →
v12 **1380847** resume@gs50 (dir _13, sidecar v12, watcher b1y0p4xcq).
Tally: 41 / ~28 arm-hours.
**04:0x incident #42 — lr1e6-v14 (1380821) ghost-OOM at ckpt load 14:49 in**
(jpbo-005-[34,37-38,43,46-47], 14.25MiB free; died AT LOAD → says nothing about
offload; dirty-node prior strengthens: 18/666/607/14 MiB tonight) → excluded
(476cca69) → v15 **1380859** resume@_13/gs20 (dir _16, sidecar v15, watcher live).
Peer's proxy fix (fff9333d etc.) now interleaved on the branch — env-gated, harmless;
both sessions share this Mac clone, commits stack. Tally: 42 / ~28 arm-hours.
**04:2x A/B CLOSED — ref cpu_offload=true EXONERATED**: lr3e6-v14 (1380822) logged
mirror step 41 + 2 reward batches, 0 OOMs, with the identical offload config v11 died
under → #41 was a partially-dirty jpbo-008 node. Offload flip validated through a full
train step; keep it for all future relaunches. Fleet 04:2x: ALL FOUR RUNNING —
nokl s67 (1:04h streak), lr3e6 past s41, lr8e6-v12 17min (loading), lr1e6-v15 7min.
**04:3x incident #44(=tally 43) — lr1e6-v15 (1380859) partial-ghost OOM at first
ppo_train 21:29 in** (jpbo-078-[01-02,04,14-16], 146MiB free; loaded+generated then
OOM at train peak — third dirty nodeset in a row for lr1e6) → excluded (801aca9c) →
v16 **1380911** resume@_13/gs20 (dir _17, sidecar v16, watcher live).
Tally: 43 / ~29 arm-hours.
**04:5x incident #44 — lr8e6-v12 (1380847) backward hang at s52** (completed s51 w/
offload config fine — more offload validation — then froze in s52 policy_train,
jpbo-079-[03,05,10-13], silent 784s; only s51 lost, gs50 still the resume point) →
excluded (22dc69a1) → v13 **1380912** resume@gs50 (dir _14, sidecar v13). Watchers:
death bur8v9heg, stall b61f4xbhy. lr8e6 stuck at cumulative 54 since 02:24 — 4th
attempt at the 50s. Tally: 44 / ~29 arm-hours.
**05:0x incident #45 — lr1e6-v16 (1380911) ghost-OOM at load 13:18 in**
(jpbo-015-[17,19,21,23,29-30], 8.12MiB free; lr1e6's 4th dirty nodeset in a row —
requeue-often → get-dirtiest-nodes vicious cycle; rack 013/014/015 corner is filthy)
→ excluded (ad5b3637) → v17 **1380924** resume@_13/gs20 (dir _18, sidecar v17,
watcher live). Tally: 45 / ~29 arm-hours.
**05:1x incident #46 — nokl-v7 (1380775) backward hang at s76, FIVE steps from done**
(2:09h streak, gs75 BANKED, froze 04:30:06 mid-s76 policy_train,
jpbo-016-[02,05,08-09,11,13]) → excluded (ad8b9a2c) → v8 **1380985** resume@gs75
(dir _9, sidecar v8, watchers beommtupb/b79elmt3o). nokl needs 5 steps + eval@80 +
final ckpt once it starts. Cumulative: nokl 75, lr8e6 54, lr3e6 45+, lr1e6 22.
Tally: 46 / ~30 arm-hours.
**05:2x incident #47 — lr3e6-v14 (1380822) backward hang at s51** (1:41h streak,
+10 net: gs45+gs50 banked this attempt; froze 04:40:48 mid-s51 policy_train on
jpbo-060-[01-02,07,09,11,13] — OTHER half of rack 060, second disjoint hit →
**rack-level [01-48]** (85a1bcc1)) → v15 **1381010** resume@_15/gs50 (dir _16,
sidecar v15, watchers bzoiwzs0h/bsga1gc1j). Cumulative: nokl 75, lr8e6 54,
lr3e6 50, lr1e6 22. Tally: 47 / ~30 arm-hours.
**05:3x incident #48 — lr8e6-v13 (1380912) backward hang at s56** (broke the 50s
barrier: gs55 BANKED, new cumulative high; froze 04:55:13 mid-s56,
jpbo-083-[05-06,09,13,15-16]) → excluded (d4755a0c) → v14 **1381037** resume@gs55
(dir _15, sidecar v14, watchers bd11ut9x7/b2va176rl). Cumulative: nokl 75, lr8e6 55,
lr3e6 50, lr1e6 22. Tally: 48 / ~30 arm-hours. Hang cadence has TIGHTENED (~25-40
min/hang since 04:30 vs 2-3h earlier) — pre-dawn queue churn handing us more
marginal nodesets; resume machinery keeping net progress positive regardless.
**05:5x incident #49 — lr1e6-v17 (1380924) backward hang at s25** (trained s21-24,
no new ckpt — interval 5, so s21-24 lost; jpbo-093-[35-38,44,47]) → excluded
(e379deca) → v18 **1381101** resume@_13/gs20 (dir _19, sidecar v18). Cumulative:
nokl 75, lr8e6 55, lr3e6 50, lr1e6 24. Tally: 49 / ~31 arm-hours.
**06:1x incident #50 — lr1e6-v18 (1381101) ghost-OOM at load** (jpbo-111-[09-12,15-16],
36MiB free; lr1e6's 5th ghost nodeset) → excluded (067310f4) → v19 **1381172**
resume@gs20 (dir _20, sidecar v19, watcher live). Tally: 50 / ~31 arm-hours.
**06:1x — nokl-v8 at s79. ONE step from campaign-first finish**; eval@80 + final
ckpt follow s80; finish-check scheduled (ba62qoalb).
**★ 06:4x — nokl COMPLETE (first arm to finish): 1380985 COMPLETED 0:0, 80/80 steps,
eval@80 = 87.49% greedy pass@1 (+29.7 vs 57.77 baseline; eval@40 was 85.22 → still
improving 40→80). gs80+gs81 banked in _9. BINDING §3 VERDICT: formally MET at s80
(bar was +10; nokl delivered +29.7 at completion). Campaign verdict = PASS regardless
of the three stragglers; they continue for completeness (lr8e6 55, lr3e6 50, lr1e6 24).
nokl endgame remaining: final wandb sync (arms_sync auto), ckpt prune at harvest,
NO further relaunches for this arm.**
**06:5x incident #51 — lr1e6-v19 (1381172) partial-ghost OOM at first ppo_train**
(jpbo-079-[01,04,07-08,15-16], 713MiB free — rest of rack 079, 2nd disjoint hit →
**rack-level [01-48]** (a49f237e)) → v20 **1381350** resume@gs20 (dir _21, sidecar
v20, watcher live). lr1e6: 6 nodeset casualties, 8th attempt tonight, still gs20.
Tally: 51 / ~31 arm-hours.
**07:1x incident #52 — lr1e6-v20 (1381350) ghost-OOM at load** (jpbo-022-[41-46],
87MiB free; rack 022 2nd disjoint hit → **rack-level [01-48]** (7acf9f57)) → v21
**1381409** resume@gs20 (dir _22, sidecar v21, watcher live). lr1e6: 7 nodeset
casualties, 9th attempt. lr8e6 1:11h streak (~s60+), lr3e6 1:23h (~s57+).
Tally: 52 / ~32 arm-hours.
**07:2x incident #53 — lr8e6-v14 (1381037) backward hang at s65** (1:22h streak,
gs60 banked, +5 net; jpbo-019-[18-21,27,30]) → excluded (e22334b0) → v15 **1381457**
resume@_15/gs60 (dir _16, sidecar v15). Cumulative: nokl DONE 80, lr8e6 64, lr3e6
~57+, lr1e6 24. Tally: 53 / ~32 arm-hours.
**07:4x incident #54 — lr1e6-v21 (1381409) partial-ghost OOM at first ppo_train**
(jpbo-053-[01-04,13-14], 572MiB free; 8th ghost nodeset for lr1e6) → excluded
(4e85f8bb) → v22 **1381465** resume@gs20 (dir _23, sidecar v22, watcher live).
lr1e6 wall check: 60 steps from ~08:00 start ≈ 7.5h → ~15:30, fits 11:59 walltime.
Tally: 54 / ~33 arm-hours.
**08:0x incident #55 — lr8e6-v15 (1381457) partial-ghost OOM at first ppo_train**
(jpbo-029-[22-23,25,29,31-32], 88MiB free) → excluded (a2ea6fa2) → v16 **1381493**
resume@gs60 (dir _17, sidecar v16, watcher live). lr3e6 1:58h streak (~s63+),
lr1e6-v22 RUNNING 9min. Tally: 55 / ~33 arm-hours.
**08:2x incident #56 — lr3e6-v15 (1381010) backward hang at s64** (2:09h streak,
+13 net: gs55+gs60 banked; jpbo-009-[21,23-25,27-28]) → excluded (37e78765) →
v16 **1381529** resume@_16/gs60 (dir _17, sidecar v16). Stall-watcher patched:
missing-log epoch ages no longer false-alert. Cumulative: nokl DONE, lr8e6 64,
lr3e6 63, lr1e6 24. Tally: 56 / ~34 arm-hours.
**08:4x incident #57 — lr1e6-v22 (1381465) partial-ghost OOM** (jpbo-112-[25-30],
64MiB free; rack 112 2nd disjoint hit → **rack-level [01-48]** (760c0330)) → v23
**1381546** resume@gs20 (dir _24, sidecar v23, watcher live). lr1e6: 9 ghost
nodesets, 11th attempt. Tally: 57 / ~34 arm-hours.
**09:0x incident #58 — lr1e6-v23 (1381546) ghost-OOM at load** (jpbo-016-[01,03-04,
06-07,14], 27MiB free; rack 016 2nd disjoint hit (nokl's s76 crime scene) →
**rack-level [01-48]** (0140573a)) → v24 **1381632** resume@gs20 (dir _25, sidecar
v24, watcher live). lr1e6: 10 ghost nodesets, 12th attempt. Wall check: 60 steps
from ~09:20 ≈ 7.5h → ~16:50, still fits. Tally: 58 / ~35 arm-hours.
**09:2x incident #59 — lr3e6-v16 (1381529) NCCL "unhandled cuda error" in EP
all_to_all_single during FIRST forward** (jpbo-001-[17-19,25,27,29]; NEW failure
mode for this campaign — the loud fail-fast cousin of the silent EP hang; sick GPU
suspected; 0 steps lost) → excluded (a59a4251) → v17 **1381636** resume@gs60
(dir _18, sidecar v17, watcher live). Tally: 59 / ~35 arm-hours.
**Probe 1380770 postmortem (FAILED 3:52, exit 127)**: two env gaps for lee27 —
(a) tracegen sbatch sources conda.sh but never activates → bare `python` 127;
fix = export `DCFT_ACTIVATE_ENV='source $F/envs/rl-fa/bin/activate'` at submit;
(b) proxy skipped: PROXYCHAINS_BIN hardcoded to feuer1's build (perm-denied) →
added `PROXYCHAINS_BIN_OVERRIDE` hook (40648803), built own aarch64 binary at
`$F/tools/proxychains-ng-install/bin/proxychains4`; ALSO needs SSH_KEY for the
compute→login SOCKS tunnel — local authorized_keys is IGNORED (JuDoor-only sshd);
**BLOCKED on operator: JuDoor paste of intra key with from="10.0.0.0/8,134.94.0.0/16"**
(reuse id_ed25519_jupiter2jureca). $F/OpenThoughts-Agent is a SYMLINK to
repos/OpenThoughts-Agent (workdir was never wrong). Runner honors
PROXYCHAINS_CONF_FILE (datagen_launch_utils L868+). PI note 08-15: /p/ going
obsolete — store under /e/ (datasets → /e/data1/datasets/playground/mmlaion/).

**NEW TOOL: `scripts/dashboard/` (Runboard, commits c8fbcaa9+f78d6c82)** — local
self-contained analysis website (operator-requested). `pull_wandb.py specs/<exp>.json`
→ `build.py` → open `dist/dashboard.html` (dist/ is gitignored; rebuild anywhere).
Spec-driven + drag-drop JSON loading; gsm8k arms campaign is the first spec, incl.
the full per-attempt fate table (keep it updated on incidents). Also published once
as artifact https://claude.ai/code/artifact/c192387b-b103-4c6d-b2d4-070fee769430
(operator prefers the LOCAL site; artifact optional).

**14:55 sweep — nokl PASSES the step-30 racing gate** (s32: rew 0.805 rising, ent ~0.35
stable, trunc ~1-3%). lr8e6-v3 resume VERIFIED in-log (resume_mode from_path,
global_step_10). No arm pathological; no kills at the gate.

## SIDE CAMPAIGN 01:0x-01:4x — curriculum-easy pass@8 probe (operator-approved, Daytona route)

**Job 1380770** `currease_base_pass8_traces`, 1 node, submitted ~01:40 CEST. The binding
pre-flight for the agentic RL campaign: Qwen3-30B-A3B-Base × TaskTrove
`DCAgent__exp_rpt_curriculum-easy` (514 tasks) × 8 attempts = 4,112 trials, verifier ON.
- Tasks: HF `open-thoughts/TaskTrove` subset parquet → extracted to
  `$F/tasks/tasktrove_curriculum_easy` (514 dirs; 504 share the 3-line python:3.10-slim
  Dockerfile, 10 share one heavier sci-py image → exactly 2 Daytona snapshots, both built
  at submit). NO Jureca/JUWELS involvement — Daytona cloud runs the terminals.
- Launch: `hpc.launch --job_type datagen`, serve config
  `hpc/datagen_yaml/qwen3_30b_a3b_base_vllm_serve_32k_4xGH200.yaml` (NEW, commit 8cbe1334:
  tp2/dp2, 32k, temp 0.7/top_p 0.95, `--moe-backend triton` for the GH200 FlashInfer JIT
  crash), harbor config `ctx32k_verified.yaml` (verifier enabled), `--trace-n-attempts 8`
  (NOT `--n_attempts` — that flag is rejected), `--trace-n-concurrent 96`,
  target repo `laion/qwen3-30b-a3b-base-currease-pass8`.
- Daytona: lee27's `DAYTONA_API_KEY` audits as a 60-cap GENERAL org (23/60 used) — OK for
  datagen per iris ops interchangeability; lee27 has NO `DAYTONA_DATA_API_KEY`.
- **STALE-NOTE FIX: Qwen3-30B-A3B-Base DOES ship a tools-aware chat_template** (4,116
  chars, in tokenizer_config.json) — NEXT_SESSION's "base has NO chat template" is wrong.
- Verify-after-boot: `_vllm.log` serve OK, then trial dirs under the run's `trace_jobs/`
  accumulating multi-turn trajectories (avg turns > 1). pass@8 = per-task any-success over
  8 rewards at consolidation. Death-watcher bndiqtrzv.

## Open/parked items

- Sweep-shards mystery RESOLVED: the other session's sweep was operator-PARKED 08-14
  (~18:15 PT) after 3 failed rounds — see NEXT_SESSION `0.0-SWEEP-PARKED`. Not our concern.
- Optional: wandb-workspaces saved view (Reward/Learnability/Parser/Lengths/Stability).
- ~~FA long-term: torch-2.9/cu130 coherent env rebuild~~ MOOT — `$F/envs/rl-fa` (torch
  2.11+cu128 + exact-tag FA wheel) resolved it 2026-08-14; no torch downgrade needed.
- /e/scratch/reformo + lee27 $HOME quota cleanups (meeting asks; nothing for this run).
