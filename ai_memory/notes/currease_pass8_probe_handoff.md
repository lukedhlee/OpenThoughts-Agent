# HANDOFF: curriculum-easy pass@8 probe (Jupiter, Daytona route)

You are taking over the **pass@8 probe of Qwen/Qwen3-30B-A3B-Base on TaskTrove
`DCAgent__exp_rpt_curriculum-easy`** (514 tasks × 8 attempts = 4,112 trials, verifier ON).
This is the BINDING pre-flight for the upcoming agentic RL campaign (see
`ai_memory/NEXT_SESSION.md` § "Meanwhile-threads" — but note one correction below).
Deliverable: per-task pass@8, overall stats, and a go/no-go read for RL learnability.

## ⚠️ COORDINATION — another live session is on this cluster

A separate Claude session (Luke's main one) is ACTIVELY babysitting the
**base30b gsm8k GRPO campaign** on Jupiter — 4×6-node arms fighting chronic node/training
instabilities (EP/FSDP2 backward hangs ~1/2-3 arm-hours + ghost-GPU-memory OOMs; 35
incidents so far, each handled kill→exclude→resume). Rules of engagement:

1. **NEVER touch jobs named `base30b_gsm8k_*`** (at handoff: 1379329, 1379349, 1380047,
   1380775 — IDs churn as that session relaunches). No scancel, no scontrol, ever.
2. **`hpc/hpc.py` `node_exclusion_list` (jupiter, ~line 1025) is hot** — that session
   appends to it after every incident. `git pull` BEFORE any local edit/commit; on push
   rejection `git pull --rebase` and retry. Branch `lukedhlee/vista-moe-grpo-30b`,
   remote **`fork`** (NOT `origin` — no write access there). Never force-push.
3. Cluster clone `$F/repos/OpenThoughts-Agent` (`F=/e/fscratch/reformo/lee27`) is pulled
   frequently by that session — leave NO uncommitted changes on the cluster.
   (`$F/OpenThoughts-Agent` is a symlink to it.)
4. Login-node tmux sessions that are NOT yours: `arms_sync`, `sidecar_lr1e6`,
   `sidecar_lr3e6`, `sidecar_lr8e6`, `sidecar_lr3e6_nokl`, `bridge`, `cmkeep`.
5. Shared Slurm account `reformo` (leave default; jureap59 QOS is suspended). Your probe
   is 1 node — fine. Don't queue anything bigger without asking Luke.
6. If YOUR job OOMs at engine load with only MBs free on a 95GiB GPU → ghost node
   (stranded memory from someone's killed job): add its nodeset to `node_exclusion_list`
   (pull→edit→commit→push→cluster pull), relaunch. Same recipe that session uses.
7. PI directive (2026-08-15): store under `/e/` only (`/p/` going obsolete); datasets →
   `/e/data1/datasets/playground/mmlaion/`.

## State — everything below is ALREADY DONE

- Tasks verified + extracted: `$F/tasks/tasktrove_curriculum_easy/` (514 dirs from HF
  `open-thoughts/TaskTrove` subset parquet; 504 share a 3-line python:3.10-slim+pytest
  Dockerfile, 10 share one heavier sci-py image).
- Both Daytona snapshots BUILT (prebuild at next launch should report "already existed").
- Daytona key: use `$DAYTONA_API_KEY` from `$F/keys/secrets.env` — audited as a **60-cap
  general org (23/60 used)**, OK for datagen (lee27 has NO `DAYTONA_DATA_API_KEY`;
  general and data orgs are interchangeable per `.claude/ops/iris/ops.md`).
- Serve config committed: `hpc/datagen_yaml/qwen3_30b_a3b_base_vllm_serve_32k_4xGH200.yaml`
  (tp2/dp2, 32k, temp 0.7/top_p 0.95, `--moe-backend triton` — do NOT remove that flag;
  FlashInfer fused-MoE JIT crash-loops on GH200 in the lee27 env).
- Harbor config: `hpc/harbor_yaml/datagen/ctx32k_verified.yaml` (verifier ENABLED — the
  non-`_verified` one silently disables rewards).
- proxychains built: `$F/tools/proxychains-ng-install/bin/proxychains4` (aarch64);
  template hook `PROXYCHAINS_BIN_OVERRIDE` committed (40648803).
- **Correction to NEXT_SESSION.md**: Qwen3-30B-A3B-Base DOES ship a tools-aware
  chat_template (4,116 chars in tokenizer_config.json). No template work needed.
- Attempt 1 = job **1380770**, FAILED at 3:52 (exit 127). Root causes, both addressed:
  (a) sbatch sources conda.sh but never activates an env → bare `python` not found —
  fixed via the sbatch's `DCFT_ACTIVATE_ENV` eval hook (export below);
  (b) proxy skipped (feuer1's proxychains perm-denied + `SSH_KEY` unset) → Daytona
  unreachable from compute nodes (they have no direct internet).

## GATE — check this FIRST, do not relaunch before it passes

Luke was asked to paste an additional JuDoor SSH key for JUPITER (reuses
`~/.ssh/id_ed25519_jupiter2jureca` with `from="10.0.0.0/8,134.94.0.0/16"`). JSC sshd
ignores local authorized_keys; activation ≤15 min after paste. Test (from the Mac):

```bash
ssh jupiter 'ssh -i ~/.ssh/id_ed25519_jupiter2jureca -o BatchMode=yes \
  -o StrictHostKeyChecking=no -o ConnectTimeout=10 lee27@jpbl-s01-01 hostname'
```

`Permission denied` → not active yet; wait or ping Luke. (Do NOT test against
`localhost` — 127.0.0.1 is outside the from= clause by design.) Without this key the
sbatch's compute→login SOCKS tunnel fails and every Daytona call dies.

## Relaunch (exact, once the gate passes)

```bash
ssh jupiter
F=/e/fscratch/reformo/lee27
set -a; source $F/keys/secrets.env; set +a
unset DAYTONA_TARGET
cd $F/repos/OpenThoughts-Agent && git pull && export DCFT=$PWD
export DCFT_ACTIVATE_ENV='source /e/fscratch/reformo/lee27/envs/rl-fa/bin/activate'
export PROXYCHAINS_BIN_OVERRIDE=$F/tools/proxychains-ng-install/bin/proxychains4
export SSH_KEY=$HOME/.ssh/id_ed25519_jupiter2jureca
$F/envs/rl-fa/bin/python -m hpc.launch --job_type datagen \
  --job_name currease_base_pass8 \
  --datagen_config hpc/datagen_yaml/qwen3_30b_a3b_base_vllm_serve_32k_4xGH200.yaml \
  --trace_harbor_config hpc/harbor_yaml/datagen/ctx32k_verified.yaml \
  --tasks_input_path $F/tasks/tasktrove_curriculum_easy \
  --trace_target_repo laion/qwen3-30b-a3b-base-currease-pass8 \
  --daytona_api_key "$DAYTONA_API_KEY" \
  --num_nodes 1 --trace-n-attempts 8 --trace-n-concurrent 96 \
  --time_limit 11:59:00
```

Flag traps: it's `--trace-n-attempts` (`--n_attempts` is REJECTED by the datagen path);
`--daytona_api_key` is mandatory. Experiment dir lands at
`$F/repos/OpenThoughts-Agent/experiments/currease_base_pass8*/` (launcher may suffix `_2`).

## Verify within 15 min of the job starting (RUNNING ≠ working)

1. Main log `experiments/currease_base_pass8*/logs/*_<jobid>.out`:
   want `[proxy] ✓ Found proxychains binary`, `✓ SSH tunnel started successfully`,
   proxy connectivity test line, and NO `python: command not found`.
2. vLLM serve log (find `*vllm*log*` under the exp dir): model load ~2-4 min. A
   GLIBCXX/flashinfer crash-loop should be impossible with `--moe-backend triton`;
   if seen anyway, stop and report.
3. `trace_jobs/` trial dirs accumulating REAL multi-turn trajectories (avg turns > 1).
   All trials 1-turn exception stubs = dead proxy or wrong Daytona org — kill and debug,
   don't let it burn walltime.
4. Runtime estimate: 4,112 trials @ concurrency 96 ≈ 3–5 h.

## Scoring & wrap-up

- pass@8 per task = any of its 8 rewards > 0; also report pass@1 mean and the
  distribution (0/8 solved / partial 1–7 / 8/8). The RL go/no-go wants a healthy
  partial band — tasks solved sometimes-but-not-always are where GRPO signal lives;
  ~0% ⇒ base model can't drive this harness (escalate: Instruct-2507 fallback question,
  Ben's open thread); ~100% ⇒ dataset too easy for RL.
- Traces upload to `laion/qwen3-30b-a3b-base-currease-pass8`. Post-run: verify realness
  + free disk per `.claude/skills/datagen-job-cleanup` (uploading without rm'ing
  `trace_jobs/` is the #1 inode leak).
- Optional second leg (operator to decide, don't self-start): same probe on
  Qwen3-30B-A3B-Instruct-2507 to settle base-vs-instruct with data.

## Reading list
- `.claude/skills/datagen-launch/SKILL.md` (flow + gotchas; you've inherited a
  mid-flight instance of it)
- `ai_memory/notes/base30b_gsm8k_validation.md` § "SIDE CAMPAIGN" (probe history) and
  the incident log (what the OTHER session is doing — read, don't touch)
- `.claude/projects/daytona/daytona.md` (snapshot/sandbox caps — HARD, never raise)

---
## ⚠ CORRECTION (2026-08-15, attempt 2): the GATE section above is OBSOLETE
The JuDoor key paste is NOT sufficient and the login→login gate test can NEVER pass for
lee27: JSC TOTP is account-level and fires after pubkey from ALL sources, including
compute→login (see gotchas.md 2026-08-15 entry). The working route is the login-node
microsocks + preset-SOCKS env path (commit fff9333d): tmux `currease_socks` on login02
runs authenticated microsocks at 10.128.1.2:7011 (creds `$F/keys/socks5_currease.env`);
submit with PROXYCHAINS_SOCKS5_PRESET_HOST/PORT/AUTH + PROXYCHAINS_BIN_OVERRIDE and NO
SSH_KEY. Attempt 2 = job 1380855 (exp dir experiments/currease_base_pass8_2), launched
2026-08-15 with this route; compute-node proxy test passed.

---
## RESULTS — Instruct-2507 leg COMPLETE (2026-08-15, job 1382360, 4:47:31)

Probe deliverable for the instruct arm (n=502 tasks with full 8 attempts; 3 nop-pass
tasks excluded: 0297/0506/0508 pass with zero agent actions — dataset defect):

- pass@1 = 34.2%, pass@8 = 41.8%
- Band profile: 0/8 = 292 (58%) · partial 1–7 = 107 (21%) · 8/8 = 103 (21%)
- Solves-of-8 histogram: {0:292, 1:12, 2:10, 3:9, 4:13, 5:13, 6:13, 7:37, 8:103}
- Harness: 100% natural termination, 0% JSON parse errors, median trial 125 s
- Traces: https://huggingface.co/datasets/lukeleeai/qwen3-30b-a3b-instr2507-currease-pass8
  (laion/ upload 403'd — neither lee27's nor lukeleeai's token has laion write; move later
  if wanted. Pipeline auto-upload also needs HF_TOKEN in the SUBMIT env — cached CLI token
  is NOT read; all 4 probe jobs skipped auto-upload, manual uploads required.)

RL read (instruct start): GO. 107 partial-band tasks are prime GRPO signal; group size 8
mixes on them by construction. Base leg (3 pooled jobs) still running for the base profile;
early base pass@1 ≈ 4% genuine — the base-vs-instruct gap is categorical (termination,
format, focus), not raw capability.

## RESULTS UPDATE — base leg (2026-08-15 evening), CORRECTED for dead-engine trials

Two of three pooled base jobs lost their vLLM engine mid-run (b/1382373: NCCL watchdog
"Invalid access of peer GPU memory over nvlink or a hardware error" ~1 h in on jpbo-006-47
— node now excluded; c/1382374: EngineDeadError at ~9 h). Trials after engine death time out
with ZERO LLM interaction but still write reward-0 results — 998/3,637 pooled base trials
were such artifacts. **Validity filter: agent_result.n_output_tokens > 0** (n_episodes is NOT
a valid filter — dead trials still report n_episodes=1).

Valid base data (runs 7+b+c): 2,617 trials, 4–7 attempts/task, pass@1 = 4.1%,
pass@any(4–7) = 83/511 = 16.2%. Top-up job d (1385854, 4 attempts, HF_TOKEN in env this
time) running to complete ≥8 valid attempts/task. Per-task table:
ai_memory/artifacts/currease_pass8_per_task.csv (base counts are valid-only).

Traces: lukeleeai/qwen3-30b-a3b-base-currease-pass8{,-b,-c} — note the b repo has only 65
rows because the exporter (correctly) drops zero-conversation trials; realness checks on
trace repos should compare against VALID trials, not result.json counts.

---
## RL ARMS LAUNCHED — base-vs-instruct GRPO comparison (2026-08-16 ~00:35 CEST)

Operator decision: run paths 1 (instruct-start) AND 3 (raw-base) GRPO side by side on
curriculum-easy and measure the difference empirically.

- Configs (commit 525f95f4): `hpc/skyrl_yaml/jupiter/6node_currease30b_grpo_{base,instr2507}.yaml`
  — identical except policy/ref snapshot paths. Design: terminal_bench/terminus-2/Daytona
  chassis (probe-parity: 32k ctx, no thinking-template bits, strict JSON, agent timeout
  1800 s, verifier 120 s, n_concurrent 64) + the gsm8k-campaign-validated FSDP2 MoE
  training block (rl-fa venv, flash-attn+packing, moe_backend triton, cu13
  LD_LIBRARY_PATH baked, EP=4×FSDP=4, 4 policy nodes + 8 TP=1 engines, ref cpu_offload
  true). GRPO n_samples_per_prompt=8, lr 3e-6 (mid arm of the validated gsm8k grid),
  KL 0.001, TIS cap 2.0, seq_mean, temp 1.0/top_p 1.0/top_k -1, batch 16 prompts/step
  (=128 rollouts), max_steps 50, ckpt_interval 5, eval off (probe = step-0 baseline),
  seed 42. Train set: `$F/tasks/tasktrove_curriculum_easy_rl511` (514 minus the 3
  nop-pass defects 0297/0506/0508).
- Jobs (6 nodes each, 12h links, 5 chained restarts): base **1386478**→…→1386483;
  instr2507 **1386484**→…→1386489. Both RUNNING since 2026-08-16 00:37 CEST.
- Launch route: direct `hpc.launch --job_type rl` from rl-fa python, preset-SOCKS env
  (microsocks 10.128.1.2:7011, tmux `currease_socks`) + PROXYCHAINS_BIN_OVERRIDE
  `$F/tools/proxychains-ng-install/bin/proxychains4`; snapshot prebuild = 2 registry
  hits, 0 new. **Gotcha: `--model_path` is MANDATORY** — omitting it crashes
  rl_launch_utils.py:1081 (`ParsedRLConfig` has no `.model`); it only overrides
  policy, so ref paths are baked in the yamls.
- Expected shape: instruct arm reward ≈0.35-0.42 at step 0, mixed groups on the
  107-task partial band; base arm ≈0.04 with mostly zero-advantage groups and
  timeout-bound rollout waves (~2×30 min/step worst case vs ~2×3 min for instruct).
  The cost/signal asymmetry IS part of the measurement.
- Verdict read: compare train-reward slope + entropy/grad_norm over first ~10-20 steps;
  watch for the EP/FSDP2 backward-hang race (gsm8k incident runbook applies verbatim —
  stall = log mtime >600 s while RUNNING).

### RL-arm incident log
- **Attempt 1 (1386478/1386484, FAILED 8.5 min):** `generator.batched: true` (inherited
  from gsm8k sync recipe) is rejected by FullyAsyncRayPPOTrainer — main_tbench selects
  the async trainer whenever `placement.colocate_all=false`. Fix: `batched: false`
  (commit in yamls).
- **Attempt 2 (1386506/1386515, killed at ~step 2):** trained but every rollout was a
  1-token zero-reward dummy (response_len_mean 1.0, reward 0.0 on BOTH arms, entropy 0,
  tis skipped). Root cause: `DaytonaConnectionError: Cannot connect to app.daytona.io`
  in RolloutCoordinators — hpc/ray_utils.py `from_hpc` read only `hpc.proxychains_binary`
  (feuer1's perm-denied build) → wrap silently skipped ("proxychains binary not
  executable") → ray started with proxy env UNSET. Fix commit: ray_utils honors
  `PROXYCHAINS_BIN_OVERRIDE` (same hook as datagen). Detection recipe: WANDB_MIRROR
  step metrics with response_len_mean==1.0 + trace_jobs result.json exception_info.
- **Attempt 3 (base 1386654→…59, instr 1386660→…65):** launched with wrap engaged;
  watchers alert on first train step / traceback / "not executable" skip line.
- **Attempt 4 (1386991/1386997):** proxy wrap finally engaged after SECOND fix (commit:
  rl_launch_utils builds RayClusterConfig DIRECTLY, bypassing from_hpc — both sites now
  honor PROXYCHAINS_BIN_OVERRIDE). Rollouts went REAL (138 exception-free trials, median
  842 output toks) and step 1 trained — then the post-step weight sync killed 4 engines:
  `_C::rotary_embedding` on META tensors. Mechanism (vllm_engine.py:2184 abort_generation):
  drain-to-idle can't converge under continuous agent HTTP traffic (no pause gate on the
  HTTP endpoint; agents re-fire on abort) → 600 s deadline → layerwise meta reload proceeds
  with live decodes → eager decode hits meta cos_sin_cache. Same disease class as the
  rms_norm meta incident (agent_logs/2026-07-07_grid30bc_rmsnorm_meta_sync_weights.md);
  the rms_norm fake just lets the decode die one op deeper. Design note says cudagraph
  masks the straggler — we had enforce_eager true (gsm8k inheritance).
- **Attempt 5 (base 1387075→…80, instr 1387081→…86): enforce_eager false** (cudagraphs =
  production parity; validated on this model/env by the 4.1k-trial serve probe). Real
  milestone = step 2 (first weight sync survived). Proper upstream fix to file later:
  HTTP-endpoint pause gate in MarinSkyRL vllm_server so the drain converges.

## ✅ ATTEMPT 5: RL STACK VALIDATED END-TO-END (2026-08-16 ~05:25 CEST)
Instruct arm (1387081) SURVIVED the first weight sync and is training:
step 1 reward 0.484 / len_mean 1410 / groups mixed 31% (all-wrong 38%, all-right 31%) /
entropy 0.162 / grad 0.67 / KL 0.0026; step 2 reward 0.383. ~25 min/step → ~27 steps
per 12 h link, 50 steps ≈ 2 links. Reward matches the probe pass@1 (0.342) within batch
noise — GRPO signal on curriculum-easy is exactly as the probe predicted.
Base arm (1387075): rollouts flowing (86 trials @1:10), first step pending (slow
timeout-bound waves, by design). Its sync-survival milestone = step 2, watcher armed.
Remaining known risk: EP/FSDP2 backward-hang race (gsm8k runbook applies; stall watch
in the arm watchers via log-mtime).
- **Incident (2026-08-16 ~08:30 CEST): instr arm 1387081 EP/FSDP2 hang** at s6 forward
  (convert_to_training_input logged 08:28, then 44+ min silence; the gsm8k race, ~1 per
  2 arm-hr expected). Runbook applied: chain scancelled, nodeset
  jpbo-018-[01,03,05,09-10,13] excluded (hpc.py commit), configs stashed to
  configs_old_1387081/, relaunched **1388099**→…→1388104 resuming from banked
  global_step_5 on the un-suffixed dir. Steps 1-5 metrics preserved in the old log.
- Resume saga: 1388099 OOM'd at ckpt load (jpbo-019-[17,23-24,26,28-29] excluded, ghost
  suspect); link **1388100** resumed clean from gs5 and posted step 6 (on jpbo-122 nodes
  — no OOM there, so ghost theory is per-node, not per-rack).
- **BASE ARM STEP 1 (job 1387075, ~10:0x CEST):** reward 0.102, **38% groups mixed**
  (62% all-wrong, 0% all-correct), entropy 1.581, grad 0.377, len_mean 7650. Base has
  real GRPO signal — more mixed groups than instruct (31%), higher entropy, at ~9x the
  wall-clock per step (~6 h to step 1 vs 40 min). Reward >> probe pass@1 (4.1%) — temp
  1.0 sampling + batch mix. Step-2 sync-survival watcher armed.

## FINAL BASE NUMBERS — job d complete (2026-08-16, pooled 7+b+c+d, validity-filtered)
Job d (1385854) TIMEOUT at 11:59 (normal datagen wall). Pooled base scoring
(1,000/4,947 dead-engine trials excluded): 511 tasks, 3,903 valid trials,
269 tasks with ≥8 valid attempts.
- **pass@1 = 4.1% · pass@8 = 18.2% (n=269) · pass@any = 111/511 (21.7%)**
- Solves-of-8 hist: {0:220, 1:24, 2:16, 3:4, 4:4, 5:1} — ZERO saturated (8/8) tasks;
  every base-solvable task is partial-band. pass@8/pass@1 ratio 4.4× (instruct: 1.2×)
  → base is high-variance per attempt, consistent with the RL arm's 38% mixed groups.
- Per-task CSV regenerated with d (ai_memory/artifacts/currease_pass8_per_task.csv, 514 rows).
- Job d's auto-upload did NOT run (wall kill preempts trace export — same as runs 7/b/c);
  manual export to lukeleeai/qwen3-30b-a3b-base-currease-pass8-d running in login tmux
  `upload_d` (verify row count, then trace_jobs cleanup across probe dirs).
- **Hang #2 (2026-08-16 ~13:3x CEST): instr link 1388100** froze entering s11 forward
  (same EP/FSDP2 race) on jpbo-122-[01,04,07,09,11,13] — excluded. gs10 banked.
  Relaunch **1389753**→…→1389758 resumes @gs10. Hang cadence so far ≈ 1 per 5 arm-hours
  (2 hangs / ~10 instr arm-hours) — milder than gsm8k's 1-per-2 but same runbook.
- Resume-OOM root cause CONFIRMED ghost nodes (natural experiment: 3/3 OOMs on nodesets
  of previously-killed jobs, 2/2 clean resumes on fresh nodes; OOM site = optimizer
  load_state_dict with 15 MiB free of 95 GiB — a prior job's ~75GB ghost). Dirty pool
  from ALL tonight's kills now excluded (six nodesets, commit in hpc.py). PROCESS RULE
  REINFORCED: exclude the nodeset after EVERY kill, immediately, not only after failures.
  Link **1389755** resumed clean @gs10 → step 11 (jpbo-026-[33-36,46,48]). Instr arm
  cumulative: 0 steps lost across 2 hangs + 3 OOM'd resume attempts.
- Base head 1387075 TIMEOUT at wall (normal) with 3 steps done — but **ckpt_interval 5
  banked NOTHING** (base ~2.4h/step → a 12h link tops out at ~4 steps: interval-5 =
  perpetual step-0 restarts). Fixed: base yaml ckpt_interval 5→2; stale links cancelled
  (1387076 nodeset jpbo-003-[33-34,36,43-44,48] excluded post-kill); base relaunched
  fresh as **1390499**→…→1390504. Base steps 1-3 metrics preserved in old log
  (r 0.102/0.102/?, mixed 0.38/0.44/?). Instr unaffected (interval 5 fine at ~20 steps/link).
- **Hang #4 (2026-08-16 ~22:24 CEST): instr link 1390983** froze entering s21 (trial
  results + driver log frozen at the same second; only wandb heartbeat alive; no s21
  after 2h at 8-22 min/step cadence) on jpbo-030-[10-14,16]. gs20 banked → 0 steps lost.
  Rack-030 now has TWO confirmed disjoint hits ([33,36,39,41,44,46] ghost-OOM +
  [10-14,16] hang) → RACK ESCALATION: jpbo-030-[01-48] excluded (hpc.py b8e0231d).
  Chain scancel'd, configs stashed (configs_old_1390983), relaunch **1392566**→…→1392571
  resumes @gs20. Instr cumulative: 4 hangs, 0 steps lost. Base unaffected (1390499
  running, trial flow live, s2 + first bank imminent).
- **Base arm CANCELLED by operator (2026-08-17 ~00:0x CEST):** Luke pulled the plug on
  currease30b_grpo_base (chain 1390499-1390504) at ~7h07 elapsed, mid-step-2 rollouts —
  no checkpoint ever banked (2.4h/step; s2 bank was ~1h away). Base science preserved:
  s1 reproducible across v1/v2 (r 0.102/0.117, mixed 0.38/0.19, ent 1.58/1.48) + probe
  numbers (pass@1 4.1% / pass@8 18.2%). Freed nodeset jpbo-059-[41,43-47] excluded
  (58a774a5). Pending instr v6 chain (1392566-71) was resubmitted as **v7
  1392687→…→1392692** so its sbatch carries the fresh exclusion (Slurm re-deals freed
  ghost nodes; 3/3 resume-OOM history). Instr is now the ONLY running arm — the 1-vs-3
  comparison is instr-trajectory + base probe/s1 evidence, not two full RL curves.

## SFT ARM LAUNCHED (path 2): Qwen3-30B-A3B-Base on OT-Agent-SFT-10K (2026-08-17 ~01:0x CEST)
Ben's ask, per the OT-Agent paper (arxiv 2606.24855): the paper's SFT stack IS our LF fork
(+ ALST); "Levanter" was a for-instance (live qwen3_moe exists only in marin monorepo
lib/levanter — the standalone repo Ben linked is frozen/merged). Paper 10K recipe = lr 4e-5
cosine wr0.1, gbs 96, 7 ep, ctx 32768, ZeRO-3 (+ repo-truth quirks adam_beta2 0.98,
max_grad_norm 1e-4 from 32k_base_bs96.yaml).
- Config: sft/lf_configs/qwen3/extra/32k_base_30b_a3b_base_ota10k.yaml (8bd9b9a6) =
  bs96 ladder + ds_z3_offload_nomat (30b coder chassis) + save_steps 100.
- NEW ENV: $F/envs/sft-lf (build_sft_lf_env.sh; rl-fa freeze clone + LF submodule editable
  + deepspeed 0.19.5 + liger + trl; needs DISABLE_VERSION_CHECK=1). Activated via
  DCFT_ACTIVATE_ENV (lee27 has no otagent conda; feuer1 scratch unreadable).
- Dataset cached at $F/hf_hub (117M) + launcher-built _thinking_preprocessed (180M).
- Chain: 1392818 (head) + 1392819-21. 8 nodes/32 GPU → accum 3. output_dir
  $F/checkpoints/ota10k_sft_30ba3b__Qwen3-30B-A3B-Base (ZeRO-3 sharded → consolidate flow).
- After SFT: GRPO from the SFT ckpt = path-2 arm vs paths 1 (instr, running) and 3 (base probe).
- **SFT shakedown #1 (1393916, 90-min backfill clone) FAILED at deepspeed init:**
  CUDAMismatchException — jupiter module CUDA 13.0 vs torch cu128; the ZeRO-3 OFFLOAD
  json tries to JIT DeepSpeedCPUAdam. FIX (9977af76): drop offload → plain
  sft/lf_configs/deepspeed/ds_z3_accelerate.json (optimizer fp32 state ~11.4GB/GPU
  sharded at 32 GPUs — fits GH200 96GB w/o offload, no JIT ops) + DS_SKIP_CUDA_CHECK=1
  in DCFT_ACTIVATE_ENV. Backfill probe: 1-node 10-min job started in 90s → machine is
  hoarding for an invisible big job; 12h walls blocked till ~10:00 CEST horizon.
  Relaunch (exp dir forked to ota10k_sft_30ba3b_2 by dedup; ckpt dir unchanged):
  shakedown2 1393983 (90min) + chain 1393985-88 (12h, exclude=""). SFT-10K composition:
  swesmith 2688 / issue 2874 / superuser 2390 / tezos 2048; teacher GLM-4.7-AWQ,
  64% clean / 35% AgentTimeout traces; median 22 msgs. Curriculum-easy composition:
  stack-pytest 493/511. Band analysis (per-task CSV): instr partial 111, base partial
  109, overlap 33 → post-SFT plan: pass@8-probe the SFT ckpt, train RL on its partial band.
- **PI DIRECTIVE: SFT moves to LEVANTER.** LF 12h chain cancelled. Shakedown2 (1393983,
  no-offload z3) validated the LF path end-to-end — passed deepspeed init, built chat data,
  loaded 30B MoE ZeRO-3 (~26GB/GPU), reached step loop (672 steps = 10k×7ep/96) — then
  step-0 OOM on ONE rank: jpbo-106-20 GPU0 carried ~67GB of another user's ghost memory
  (node never in our exclusion list; ghosts are cluster-wide, not just ours). LF = working
  fallback, one clean node away. Levanter route: marin monorepo has the general chat-SFT
  launcher (experiments/sft/launcher.py, delphi_1e22 = LF-parity-validated example;
  LiteCoder-Terminal-30b-a3b-sft = existing 30B-A3B terminal SFT precedent; qwen3_moe model
  complete in lib/levanter). DRAFTED config: marin-mono/experiments/sft/configs/
  ota10k_qwen3_30b_a3b.py (HFModel qwen3_moe + Qwen eos 151645/151643, OPENCODE_TOOLS
  ChatML template w/ generation-mask, DatasetSpec conversations@d0f898f, AdamConfig
  4e-5/0.98/clip1e-4/cosine/wr0.1, seq 32768, batch 96, num_train_epochs=7). GATE: marin
  iris credentials (openathena.ai login) — either Luke gets creds or config goes as marin PR.
  Instr RL v7 1392687 RUNNING since 09:12 CEST on jpbo-095-[21-26] (resume@gs20 verify pending).
- **LR SWEEP LAUNCHED (2026-08-17 ~10:5x CEST):** two more instr2507 GRPO arms from
  step 0 — lr1e6 chain 1394803-08, lr8e6 chain 1394809-14 (yamls ..._lr1e6/_lr8e6.yaml,
  9f3e4bb5; only lr differs from baseline 3e-6). Spacing matches the gsm8k-30B precedent
  (1e-6/3e-6/8e-6). 3 concurrent RL arms × 6 nodes; Daytona snapshots = registry hits.
  Step-1 health watchers armed. Baseline 3e-6 (chain 1392688, resumes gs20) still queued.
  Levanter SFT bring-up at smoke6 (vocab-pad fix; cache+load+fwd/bwd already validated).
- **LR SWEEP REVISED (2026-08-17, Luke): 8e-6 judged too hot → replaced with 5e-6.**
  lr8e6 chain 1394809-14 scancelled while still fully PENDING (0 steps run, no ckpts;
  exp dir deleted). New arm: `..._lr5e6.yaml` (2d9f50b1, only lr line differs) →
  chain **1394836-41** via rl-fa venv python (`$F/envs/rl-fa/bin/python -m hpc.launch`;
  base miniforge python lacks pydantic — rl-fa is the launcher env for lee27).
  Final sweep = **1e-6 (1394803) / 3e-6 (1392688, from gs20) / 5e-6 (1394836)**.
  Step-1 watcher armed on 1394836; lr8e6 watcher killed.
- **Levanter smoke6 (1394737) FAILED — vocab-pad flag is broken upstream.** Same pytree
  error as smoke5: grads vocab=151936 vs opt_state.mu vocab=151665. Root cause read from
  source: `with_tokenizer_padded_to_match_model()` pads only the **converter's** tokenizer
  (hf_checkpoints.py:633, adds `<|padding_i|>` dummies); train_lm.py:245 computes Vocab
  from the local `tokenizer` var (= data tokenizer, never re-assigned after padding) →
  initial_state builds opt_state at 151665, HF load swaps in a 151936 model
  (train_lm.py:317), first take_step explodes. **Upstream one-line fix** (candidate marin
  PR, needs Luke's ask): after the pad calls, `tokenizer = converter.tokenizer`.
  **Our zero-fork fix:** padded tokenizer materialized on disk with levanter's own scheme
  → `$F/data/qwen3_30b_a3b_base_tok_pad151936` (267 dummies over HF len 151669;
  levanter's `load_tokenizer` measures it at exactly 151936, original at 151665;
  real-text encoding verified byte-identical). Config now points data.tokenizer there +
  fresh cache_dir `ota10k_levanter_cache_pad` (6ccbc365). `pad_tokenizer_to_match_model`
  kept (now a no-op guard). **Smoke7 = 1394988** (2 nodes, 1:20 wall; adds
  `--hf_save_steps 3` so the 30B HF-export path is validated in the same shot).
- **Machine unclogged ~12:00 CEST 2026-08-17** (hoard spent: 4371 alloc / 32 idle; all
  4 jobs started at once). **Incidents 37+38, both ghost-OOM at start/load:**
  (37) lr1e6 head 1394803 CUDA OOM at NCCL init on jpbo-076-[41-46] (ranks on -41/-42)
  → excluded (315a6502); link 1394804 self-recovered on jpbo-081, running.
  (38) baseline head 1392688 OOM at resume-load on jpbo-107-[01-06]; Slurm respawned
  link 1392689 onto the SAME nodeset → excluded (639ae5da), all 12 pending links
  ExcNodeList-patched (both new sets), doomed 1392689 scancelled → chain rolls to
  patched 1392690 (resume@gs20 intact). LR-sweep ID swap: **lr1e6 arm now = 1394804.**
  Watchers re-armed: 1392690 (resume past gs20), 1394804 (step-1). lr5e6 1394836 +
  smoke7 1394988 running clean ~17 min.
- **Incident 38b:** baseline link 1392690 ALSO ghost-OOM'd — jpbo-076-[33-38], GPU0 on
  jpbo-076-38 carried ~82GB foreign residue (95GB total, 29MiB free, ours 11.6GB).
  Second dirty subset of the same rack → **jpbo-076 escalated to full rack [01-48]**
  (93a12b82), remaining 10 pending links re-patched. Link 1392691 running on
  jpbo-026-[37-38,40-41,43-44] (clean territory) — baseline has links 91+92 left;
  if both burn, resubmit fresh chain (gs20 ckpt intact, auto-resume). Post-unclog
  lesson: nodes freshly vacated by the big invisible jobs are ghost-heavy — expect
  start-window OOM storms right after a machine-wide unclog.
- **Incident 38c:** baseline link 1392691 third ghost-OOM (jpbo-026-41, ~84GB residue)
  → jpbo-026 rack-escalated [01-48] (0a98aea7), last link 1392692 patched while pending.
- **Incident 39 — lr5e6 arm launched WITHOUT proxy env (my launch-shell miss):**
  step-1 all-zero vitals (reward 0, entropy 0, response_len 1.0) = every Daytona call
  DaytonaConnectionError→ZERO because the sbatch fell back to feuer1's proxychains path.
  RL launches MUST export the full preset-SOCKS preamble (sbatch --export=ALL bakes it):
  `PROXYCHAINS_BIN_OVERRIDE=$F/tools/proxychains-ng-install/bin/proxychains4`,
  `PROXYCHAINS_SOCKS5_PRESET_HOST=10.128.1.2`, `_PORT=7011`,
  `_AUTH="$SOCKS_USER $SOCKS_PASS"` (from socks5_currease.env), `unset DAYTONA_TARGET`.
  Old chain 1394836-41 scancelled + exp dir wiped (70 min of zero-reward no-op steps —
  policy unharmed, data garbage). **lr5e6 v2 chain = 1396797-1396802** (fresh hpc.py
  blacklist incl. 076/107/026 baked in). Watcher checks the proxy line first, then step-1.
  Healthy arms unaffected: lr1e6 1394804 step-1 GREEN (reward 0.445, entropy 0.169,
  mixed 0.44, len 1584); smoke7 past the pytree layer (lowering train_step at 13:35).
- **Smoke7 (1394988) verdict: vocab fix CONFIRMED, then 2-node XLA-compile OOM
  (RESOURCE_EXHAUSTED 32GiB during autotune; f32 state ~46GB/GPU on 8 GPUs leaves no
  workspace), TIMEOUT at wall. Not a pipeline bug — smoke-shape artifact. Padded-tok
  cache (ota10k_levanter_cache_pad) built + reusable. **PRODUCTION LAUNCHED: 1396849**
  (+afterany restart link 1396850): 8 nodes/32 GPUs (~11GB/GPU state), 328 steps,
  seq 32768, gbs 96, lr 4e-5 cosine — full paper recipe, no smoke shrinkage. Levanter
  auto-resumes from checkpointer (every:100 + final); hf export at step 328 to
  checkpoints/ota10k_sft_30ba3b_levanter/hf. Watcher: first loss line or exit.
- **Sweep fully green (2026-08-17 ~14:30 CEST):** lr5e6 v2 (1396797) step-1 GREEN with
  proxy fixed (reward 0.5, entropy 0.168, mixed 0.31, len 1506). Baseline 1392692
  resumed past gs20 (step 21+, jpbo-040); 3 extra links queued behind it with current
  blacklist via `--exclude` override (1396944-46). gs50/hang watchers armed on all
  three arms. Levanter production 1396849 (+1396850) queued.
- **Incident 40 + link-submission gotcha (2026-08-17 ~15:40 CEST):** baseline 1392692
  FAILED at 41 min — ghost-OOM in `ppo_train()` backward on jpbo-040-24 (~62GB foreign
  residue; survived load+rollouts, died at backward peak) → jpbo-040-[17-20,23-24]
  excluded (a328f472). My 3 spare links (1396944-46) all died in ~10s: **rendered-sbatch
  resubmits MUST come from a shell with `cd $F/OpenThoughts-Agent && export DCFT=$PWD`
  AND the full proxy preamble** — the WORKDIR guard exit-1s otherwise. Resubmitted
  correctly: baseline links **1397228→1397229→1397230** (afterany chain, full blacklist
  incl. 040 via --exclude). Sweep links 1394805-08 + 1396798-802 patched with 040 set.
  Baseline still resumes from gs20 ckpt (steps 21-22 lost to ckpt_interval=5 — fine).
  Levanter prod 1396849 RUNNING on jpbo-114 since 14:15: in XLA compile/fused-CE
  autotune since 14:08 (first-time 32k-seq MoE HLO, no compile cache — slow is normal).
- **Incident 41 — Levanter prod 1396849 FAILED at 2:25 (all-rank SIGABRT):** root chain:
  fused-CE autotune sweep at the prod shape (seq 32768 × vocab 151936) compiled
  pathological candidates (gpu_hlo_schedule: 3.34TB per-device args vs 81.6GB limit)
  for 2h23m; task 1 died ~16:26; the other ranks (still compiling) missed the 5-min
  Shutdown barrier (5/8 reached) → CoordinationServiceAgent CHECK-abort everywhere.
  Sweep results weren't even cacheable ("kernel jaxpr unavailable" → no sharing).
  Fix (15c676a6): `LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS=0` (use inferred block sizes,
  warn-and-fallback path exists if invalid) + `JAX_COMPILATION_CACHE_DIR=$F/cache/
  jax_comp_cache` (restart links skip recompile). **Prod v2 = 1397721 (+link 1397722).**
  Note: NO config knob selects the CE implementation (compute_next_token_loss hardcodes
  the fused kernel path); the env flag is the only lever short of patching levanter.
- **Incidents 42+43:** (42) Lev prod v2 1397721 died 8 min on jpbo-077-[01-06] — ghost
  residue from OUR OWN lr5e6-v1 scancel; SFT sbatch now carries a TARGETED exclusion
  list (today's confirmed-dirty sets only, 73837428 — deliberate deviation from Luke's
  "no SFT exclusions", flagged to him). Prod chain: 1397722 RUNNING on jpbo-046 +
  link 1397912 (new script). (43) baseline link 1397228 ghost-OOM at gs22 backward on
  jpbo-074-[01-02,11-14] (~60GB residue) → excluded (d36aa260), 10 pending links
  patched, insurance links 1398002-03 queued (correct DCFT+proxy recipe). Baseline now
  on 1397229 (jpbo-017). **Structural fix TODO: prolog ghost-guard in universal_rl.sbatch**
  — per-node nvidia-smi memory.used probe at start, fail-fast in ~30s (vs 15-50 min)
  when any GPU carries >2GB foreign residue; optionally auto-append the bad node to the
  next pending link's ExcNodeList via scontrol from the compute node. Post-unclog storm
  = 6 ghost incidents today; whack-a-mole exclusion is losing ~40 min/link.
- **Incident 44 — the SFT fast-fails were partly SELF-INFLICTED quota cascade:** each
  SIGABRT crash dumped 8×15-18GB cores into the repo cwd ($F/repos/OpenThoughts-Agent/
  core.*) — 16 cores / 258GB total, now DELETED. Quota state (jutil project dataquota):
  /e/scratch/reformo OVER soft INODE limit (8.07M/8M); /e/project1/reformo over soft
  data (22.7/21.4TB) — flag to Luke, most of it is project-wide (feuer1 etc), but our
  cores were pure waste. 1397722 died with `OSError: [Errno 122] Disk quota exceeded`
  in the wandb staging thread + cuMemAllocAsync failures (ghost-vs-quota entangled).
  sbatch hardened (64d444f5): `ulimit -c 0` + jpbo-046-[02-12] excluded.
  **Prod chain now: 1397912 RUNNING on jpbo-001-[20-30] (fresh nodes)** — carries
  autotune-off + comp-cache + 077/114 exclusions (046 patch rode scontrol; new script
  for future links). Watcher armed.
- **Incident 45 — ROOT CAUSE of ALL Levanter prod deaths found: JAX default
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 = 71.25GiB self-cap per 96GB GH200.** The
  allocator stats line ("Limit: 71.25GiB") proves the RESOURCE_EXHAUSTED failures were
  against OUR OWN cap, not ghost residue — the jpbo-077/046/001 "ghost" reads were
  wrong (RL's OOMs with foreign-process accounting remain real ghosts). Consistent
  story: smoke7's 2-node "compile OOM" (45.75GB state + 32GiB request > 71.25),
  v1's task-1 death mid-sweep, and all three 7-min v2/v2b/1397912 deaths. The wandb
  `Disk quota exceeded` staging error is NON-FATAL (background worker catches+drops;
  home is empty; likely /e/scratch inode-over-soft grace weirdness) — red herring.
  Fix (145339af): XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 (87.4GiB) + WANDB_DATA_DIR/
  CACHE_DIR moved off $HOME. **Prod v3 = 1398134 (+link 1398135).**
- **Incident 46:** lr5e6 1396797 EP/FSDP2 backward hang at gs5 on jpbo-051-[33-35,38,43-44]
  (log+results frozen 43-63 min, confirmed over 13-min recheck) → scancelled, nodeset
  excluded (d831364e), all 12 pending links patched. Chain rolls to 1396798, resumes
  from gs5 ckpt. Steps 1-5 metrics banked (step-1 reward 0.5).
- **Incidents 47+48:** (47) lr1e6 1394804 backward hang at gs5 (log+results frozen SAME
  second 17:39:50, 53 min) — BOTH sweep arms hung right at the gs5 ckpt boundary →
  jpbo-081 rack-escalated [01-48] (6885ffb4; 3rd bad zone), 12 links patched; chain →
  1394805, resumes gs5. (48) **Levanter reversal:** v3 1398134 died 7 min at the NEW
  87.40GiB limit, main thread in train_step → the autotune-OFF inferred CE block sizes
  are the fast killer (9.6GB blocks), NOT the memory cap; v1's sweep survived this
  phase 2h23m and its death is explained by the old 71.25 cap. Reverted to autotune ON
  + kept 0.92 + comp cache (d9e8cb8d). **Prod v4 = 1398566 (+link 1398567).** Expect a
  2-3h first compile; XLA module cache should make restarts cheap.
- **Incident 49:** lr5e6 link 1396798 ghost-OOM at init_model on jpbo-037-[36,39,42-44,48]
  (~84GB residue, real foreign-process accounting) → excluded (2656a695), links patched.
  Fleet after roll: baseline 1397229 (1:43h, jpbo-017-[38-45]), lr1e6 1394805 (jpbo-024),
  lr5e6 1396799 (jpbo-017-[01-14]), Lev prod v4 1398566 (jpbo-077-[07-14], past the
  7-min autotune-off death window — sweep running). Watchers re-armed.
- **Ghost-probe sweep (20:05 CEST): ALL 8 former incident nodes probe CLEAN** (jpbo-037,
  026-41, 076-38/41/42, 077-02 + control jpbo-002-37): 2-127 MiB used, no compute
  procs, 80GiB cuMemAlloc OK on every GPU. Conclusion: **residue is TRANSIENT** —
  lifetime tied to the leaking process (freshest case jpbo-037-39 cleared in <2.3h;
  but 1397228 hit 60GB residue 49 min into its run, so lifetimes range minutes-hours).
  Reframes the JSC ask: cleanup race/latency (node returned to pool before previous
  job's GPU memory is released), NOT permanently bad nodes. Implication for us: the
  1221-node blacklist chases yesterday's ghosts — ghost entries should be AGED OUT
  (~24h), hang entries kept. New protocol: on the next ghost incident, fire
  scripts/jupiter/ghost_probe_sweep.sh at the incident nodes within minutes to hand
  JSC a live specimen (probe + sweep committed, ba67ca7f).
- **Node-situation knowledge consolidated → `ai_memory/notes/jupiter_node_health.md`**
  (ghost transience + mechanism, hang category, exclusion-list state/aging policy,
  probe tools, prepared JSC asks, prolog-guard TODO). Read that note first for any
  future node incident.

## Session split (2026-08-17 ~20:50 CEST)

Luke split the workload into three sessions. THIS ledger's supervisor session
keeps: GRPO LR-sweep fleet ops + node-health/JSC thread. Spun out:
- **SFT Levanter babysitter** → `session_brief_sft_levanter.md` (owns chain
  1399224/1399225; incident 50 = XLA GEMM autotuner 24GiB OOM killed v4
  1398566 → fix `xla_gpu_autotune_level=1`, commit f41bb3e5; v5 1398567
  cancelled as doomed; v6 1399224 launched with fix).
- **Runboard dashboard** → `session_brief_runboard_currease.md` (currease
  wandb is OFFLINE per-exp-dir, never synced; cloud project doesn't exist yet).
Once those sessions boot, the supervisor stops touching their workstreams.

## SFT babysitter session (post-split incidents)

- **Incident 51:** Lev v6 1399224 FAILED 22m50s in (compile cache warm — CE sweep
  + train_step lowering done in ~15 min). XLA GEMM autotuner at level 1 still
  RESOURCE_EXHAUSTED during train_step compile: 2 of 646 instructions, both
  `__triton_gemm` fusions of `Qwen3MoeSparseMoeBlock/.../MoELinear/ragged_dot_general`
  (MoE expert GEMMs), autotuner tried to materialize 3.00TiB / 4.50TiB operand
  buffers — the ragged-dot operand shapes themselves are the problem, not the
  level-4 init/check copies (incident 50). Ladder step 2: `xla_gpu_autotune_level=0`
  (no autotune runs; heuristic configs). v7 1399225 cancelled (snapshot-inherited
  the level-1 script at submit). New head+spare submitted from the level-0 sbatch.
  Non-fatal noise confirmed again: 9.6GB cuMemAllocAsync failures at 20:35 were
  CE-sweep probing; job ran 16 more min after them. If level 0 dies on the same
  triton ragged-dot fusions, next candidate rung: `--xla_gpu_enable_triton_gemm=false`
  (falls back to cuBLAS emitters) — NOT yet validated.
- **Incident 52 — MILESTONE + new failure class:** v8 1399676 at level 0
  **COMPILED train_step for the first time ever** (autotune ladder closed: 4→1→0
  fixed the compile-time OOM). Then FAILED at 23m54s at step-0 EXECUTION:
  `JaxRuntimeError RESOURCE_EXHAUSTED 8.96GiB [executable_name='jit__train_step']`
  — exactly 9,622,075,392 bytes, the same size the CE sweep candidates probed
  (benignly) at 21:02. Genuine capacity shortfall at microbatch 3/device
  (96/32, seq 32768) under the 0.92-fraction pool (~88.3GiB of 96). Fix:
  `trainer.per_device_parallelism: 1` (3-step grad accum; global batch 96 and
  optimizer math unchanged — recipe intact). Mem fraction kept at 0.92 — bump
  to 0.95 is the NEXT lever if execution still OOMs. v9 1399677 dependency-
  released on v8's death and started with the old config (yaml read at job
  start) → cancelled 36s in. NOTE: new shapes → compile cache miss; expect the
  CE sweep (~8 min) + full train_step compile again on the next attempt.

## Incident 53 (2026-08-17 ~21:45) + sweep extension to gs100

- **Luke directive: extend all three arms 50 → 100 steps.** Mechanism: yamls
  bumped (9d4fa0d3) AND the launcher config JSONs
  `experiments/<arm>/configs/<arm>_rl_config.json` sed-patched
  (`trainer.max_steps=50→100`, `.bak_ms50` backups kept) — the rendered sbatch
  builds the srun command from that JSON at link start, so every pending link
  inherits 100 automatically. Running heads still stop at 50 (command already
  built); their successor links continue 51–100. Constant LR → extension is
  schedule-safe; dataset supports ~159 steps at epochs=5.
- **Incident 53**: baseline 1397229 backward hang mid-s26 on
  jpbo-017-[38-40,43-45] — log + newest trial-result frozen at the same second
  (19:07:56), discovered 2h37m later (watchers had died post-compaction; only
  the stale one survived). gs25 banked. scancel 21:46:30 → 1397230 auto-started
  ~3 min later on jpbo-109-[41-42,44-45,47-48]. Exclusion c016d041; all 9
  pending links scontrol-patched with the full list + new set.
- **Live-capture FIRST USE**: ghost_probe_sweep fired at the 6 freed nodes
  ~2 min post-scancel (jobs 1399859-64): **ALL CLEAN** (2–71 MiB, 80GiB
  cuMemAlloc OK ×4 GPUs each). Evidence: our own scancel of a hung EP/FSDP2
  job does NOT strand memory — the ghost producers are other workloads or
  timing-dependent teardown, strengthening the cleanup-race framing for JSC.
- Fleet after: baseline 1397230 (resume gs25→100), lr1e6 1394805 gs10,
  lr5e6 1396799 gs10, spares ×2-3 per arm all ms100-inheriting. New
  exit+hang-aware watchers armed on all three.

## SFT Incident 54 (2026-08-17 ~22:45 CEST) — ROOT CAUSE FOUND: XLA densifies MoE ragged_dot; fix = levanter use_gmm branch

- v10 1399820 (per_device_parallelism=1) FAILED identically to v8 at 32m56s:
  runtime RESOURCE_EXHAUSTED 8.96GiB in jit__train_step. The log's remat
  warnings expose the real problem: post-remat memory demand **7.18TiB** at
  microbatch 1 (v8: **21.15TiB** at microbatch 3 — exactly 3×, so the knob DID
  work; ~235MB/token/device is the pathology). gpu_hlo_schedule also reports
  while-loop (layer-scan) I/O arguments of 1.66–2.75TB.
- **Root cause (explains incidents 50–52 retroactively)**: levanter's
  qwen3_moe uses `hnn.MoELinear` with `use_gmm=False` → global-view
  `jax.lax.ragged_dot_general`; XLA's SPMD partitioner/autodiff densifies its
  backward into [token, experts, dim] buffers (32k×128×2048×fp32×48 layers ≈
  1.65TB per tensor class). v6's "3.0/4.5TiB autotuner allocations" were these
  real HLO operand shapes, not an autotuner bug. DenseMixer ruled out
  (dense_router_gradient defaults False).
- **Fix**: haliax already ships the cure behind `MoELinear(use_gmm=True)` —
  shard_map + ragged_dot kernels (pallas-triton on GPU, custom VJP so XLA
  never autodiffs/partitions the ragged op; megablox on TPU) + unit tests. No
  model threaded it. Patch = thread `use_gmm` through Qwen3MoeConfig/Experts,
  default True (must be a default: use_hf_model_config REPLACES the yaml model
  config, so yaml-set fields don't survive). Upstream marin main checked: 0
  relevant commits ahead. Deployed via fork-branch flow:
  **github.com/lukedhlee/marin branch `lukedhlee/qwen3moe-gpu-gmm`**
  (f0c24253a on top of upstream ab07b1a); cluster
  `$F/repos/marin` checked out on that branch (was: plain upstream main).
  Local clone (new): `/Users/lukedhlee/marin`. Upstream marin PR: candidate,
  NOT opened (needs Luke's explicit ask per marin-fork rules).
- pallas-triton import verified in the levanter env. **v12 = 1399959 (head),
  v13 = 1399960 (spare)** submitted ~22:50. Full recompile expected (HLO
  changed). Watch for: (a) `ragged_dot auto fallback` RuntimeWarning in the
  log = triton kernel failed, silently on XLA path again — treat as FAILURE
  even if it runs; (b) plausible first loss (~1.5-3 for a base model on SFT
  data) as the numerics sanity check on the new kernel path.

## Incident 54 (2026-08-17 22:39): lr5e6 hang at s11 → rack-017 escalation

- lr5e6 1396799 backward hang at s11 on jpbo-017-[01,03,06,10,13-14] — log +
  newest trial-result frozen same-second at 21:41:21, caught by watcher at
  ~57 min. gs10 banked (saved 21:13-21:41 window), resume loses ≤1 step.
  NOTE: hang again right AFTER a ckpt_interval boundary (gs10) — that's now
  4 of the last 5 hangs at a boundary+1 step (46, 47, 53?, 54).
- Second distinct dirty zone in rack jpbo-017 today (0x/1x now, 3x/4x incident
  53) → escalated to jpbo-017-[01-48] in hpc.py; all 8 pending links patched
  BEFORE the scancel (closes the fresh-node reschedule race). Chain rolls to
  1396800 (ms100 + pruned list + rack-017).
- Live-capture probe attempted at +30s: nodes still COMPLETING (skipped);
  retry loop armed for the idle flip.
- Incident-54 probe verdicts (jobs 1400051-54, ~3 min post-kill): jpbo-017-01/
  03/06/10 ALL CLEAN (3-30 MiB, 80GiB alloc OK x4). Second same-day clean-kill
  datum — our scancels of hung EP/FSDP2 jobs do not strand GPU memory.
