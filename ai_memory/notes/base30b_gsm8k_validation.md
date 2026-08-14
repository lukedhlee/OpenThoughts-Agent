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
2. **USE FLASHATTENTION for the trainer — Luke INSISTS (2026-08-14). Smoke/arms GATED on FA.**
   Path: get lee27 into unix group **`datasets`** and use HER env:
   `source /e/data1/datasets/playground/ot/envs/mamba/bin/activate
    /e/data1/datasets/playground/ot/envs/py3.12` (her script lines 166-172). lee27 NOT in
   the group today (verified). Luke fires the ask (Marianna/Jennia/JuDoor).
   Facts behind this: mjun0812 has NO torch2.11 aarch64 FA wheel (checked 08-14, last 5
   releases); source build quasi-blocked (Jupiter has no CUDA-12 module vs rl-megatron's
   torch 2.11+cu128; plus FA-cute/cutlass-dsl breakage — fix_flash_attn_cute.sh era).
   The committed arms YAML currently carries the SDPA trio as FALLBACK ONLY.
   ⚠ When her env lands: it was built for HER torch/SkyRL — before arms, smoke OUR
   MarinSkyRL branch inside it (import vllm._C, flash_attn, torchtitan EP path, ray) and
   build a `..._fa.yaml` variant (flash_attn true, attn_backend auto, use_sample_packing
   true restored, RL_PYTHON/LD_LIBRARY_PATH rebuilt for her env layout). Keep SDPA yaml.
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
2. **FA gate (BLOCKING per Luke):** datasets-group ask → env import smoke → `..._fa.yaml`
   variant → only then stages 3-4. (If the ask stalls, Luke decides: wait vs SDPA fallback.)
3. **Stage-3 smoke (2 steps, 6n, ~1.5-2h):** from `$F/repos/OpenThoughts-Agent`:
   `JSC_SCRATCH=$F RL_VENV=<env> DATA_DIR=$F/data/gsm8k EXPERIMENTS_DIR=$F/experiments \
    RL_REPO_DIR=$F/repos/MarinSkyRL WANDB_DIR=$F/wandb RESERVATION=none NUM_NODES=6 \
    RL_CONFIG=./hpc/skyrl_yaml/jupiter/6node_qwen3_30b_a3b_base_gsm8k_grpo_fsdp2_arms.yaml \
    JOB_NAME=base30b_gsm8k_smoke MAX_STEPS=2 EVAL_BEFORE_TRAIN=true CKPT_INTERVAL=1 \
    TIME_LIMIT=01:45:00 bash hpc/skyrl_standard/jupiter/run_gsm8k_moe30b_grpo.sh`
   G3 gates: 8 engines up (≥12-min post-load patience — FlashInfer autotune silence);
   step-0 eval ≈ probe ±5 pts; raw_grad_norm finite/nonzero (ANCHORED grep, exclude
   max_grad_norm echo); avg_pass_at_8 > avg_raw_reward (group variance); tis/* emit sanely
   (imp_ratio≈1); KL finite; **diag/* metrics in wandb + diag_rollouts JSONL written +
   sidecar --once produces panels**; ckpt written; warm step time recorded → final wall
   math (>480s/step ⇒ eval_interval 20 and/or MAX_STEPS 80).
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

## Open/parked items

- Sweep-shards mystery RESOLVED: the other session's sweep was operator-PARKED 08-14
  (~18:15 PT) after 3 failed rounds — see NEXT_SESSION `0.0-SWEEP-PARKED`. Not our concern.
- Optional: wandb-workspaces saved view (Reward/Learnability/Parser/Lengths/Stability).
- FA long-term: torch-2.9/cu130 coherent env rebuild with FA + source vLLM (own gated task).
- /e/scratch/reformo + lee27 $HOME quota cleanups (meeting asks; nothing for this run).
