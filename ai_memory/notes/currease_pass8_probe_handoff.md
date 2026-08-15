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
