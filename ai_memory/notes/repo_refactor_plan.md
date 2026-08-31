# Repo refactor plan — one clean OTA, one clean SkyRL, harbor on the shelf

Opened 2026-08-30, pruned same day. Status (2026-08-31): **steps 1–3 done, step 4 (gate): both stacks LAUNCH AND TRAIN from the clean branches (1556228 GSM8K, 1557218 currease); numeric parity with the old lr3e6 arm FAILS at step 2 — upstream MarinSkyRL main takes a much larger effective step; control run 1557828 PROVED it is upstream MarinSkyRL (old snapshot on the new OTA+venv reproduces the old small-step numbers); step 5 awaits Luke.** See "Status log" at the bottom. Companion: the OTA Code Atlas
(https://claude.ai/code/artifact/e633ab79-c602-4ca1-a5d7-5b52ce4e3aa2) holds the full inventory this plan
was made from; this note holds only the decision and the port list.

## What matters

We start from scratch: a fresh clone of each upstream, one branch per fork, and we **port only the code
the GSM8K and curriculum-easy runs need** — six commits in total. Nothing is rebased or squashed out of
the old 460-commit branch; the old tips are frozen as tags. The target is forks whose every commit could be
a PR, so onboarding is "clone this, check out this branch, run this script".

Forks, not upstream branches: we have **no write access** to any of the three upstreams (verified 2026-08-30
by dry-run push as `lukedhlee`). Ask Ben for write access; if granted, the same branch moves upstream
unchanged. For a newcomer it is one URL and one branch name either way.

Scope cuts, all decided 2026-08-30: **no Levanter SFT**, **no Megatron** (live path is FSDP2), **no
OpenCode** (agent is stock terminus-2), **nothing from Marianna's forks** (her SkyRL is a dead lineage;
her harbor Apptainer bridge is redundant with ours; only her band/filter scripts are useful and are already
copied to `$F/marianna_deepswe_repro_copy_0812`). `ai_memory/` leaves the repo: gitignored, Mac-only.

## The three repos

| Layer | Upstream · branch | Our fork · branch | Commits | When |
|---|---|---|---|---|
| Launcher | `open-thoughts/OpenThoughts-Agent` · `penfever/working` | `lukedhlee/OpenThoughts-Agent` · `lukedhlee/jupiter` | 4 | now |
| Trainer | `marin-community/MarinSkyRL` · `main` | `lukedhlee/MarinSkyRL` · `lukedhlee/jupiter` | 2 | now |
| Sandbox | `marin-community/harbor` · `main` | `lukedhlee/harbor` · (existing bridge branch, untouched) | 2 | r2egym |

Production today uses Ben's stock harbor (pinned by MarinSkyRL's lock file) with Daytona sandboxes; harbor
needs no work for GSM8K/currease.

## The six commits

**OTA** (source: `lukedhlee/vista-moe-grpo-30b`; ~540 changed lines of code in the whole branch)

| # | Commit | What it does for a run | Cleanup while porting |
|---|---|---|---|
| 1 | Jupiter cluster entry | `hpc.launch` knows Jupiter: account/partition, venv activation, Ray log + spill dirs, FlashInfer workspace | Read every path from `hpc/dotenv/jupiter.env`; no personal `/e/…/lee27` constants in code |
| 2 | Sandbox egress proxy | Compute nodes reach Daytona through a preset SOCKS5 proxy (`PROXYCHAINS_BIN_OVERRIDE`, host/port/auth), skipping the SSH tunnel; non-executable binary is skipped; `secrets.env` comments stripped | — |
| 3 | MoE RL configs + run scripts | base30b GSM8K arms YAML; 4 currease YAMLs; `run_gsm8k_moe30b_grpo.sh`; `prep_gsm8k_parquet.sh`; sbatch template: `NCCL_BLOCKING_WAIT=1`, `ulimit -c 0` | Drop `vista/*`, `24node_*` debug variants, stale `*_r2egym_*` |
| 4 | Housekeeping | `.gitignore` for `ai_memory/`; `build_rl_fa_env.sh` venv recipe; `jupiter.env` template without secrets; W&B offline sync scripts | Dashboard (`scripts/dashboard`) and node-exclusion list stay out of the repo |

**MarinSkyRL** (source: `lukedhlee/jupiter-worktree-0814`; check each against upstream `main` first)

| # | Commit | What it does for a run |
|---|---|---|
| 5 | MoE reference-worker fixes | Reference model gets the policy's MoE kwargs (grouped GEMM, router replay); without it KL-with-reference on EP>1 asserts at init |
| 6 | Per-group GRPO diagnostics + rollout dumps | Opt-in `diag/*` metrics (frac groups all-wrong / mixed / all-correct, reward spread, response-length percentiles, truncation) + decoded rollout JSONL per step. `frac_groups_mixed` is the "is the environment measuring the model" signal and the dead-proxy tell |

Also carried inside #5/#6 only if upstream still lacks it: the `WANDB_MODE=offline` guard in tracking (JSC
compute has no internet).

**harbor, at r2egym** (source: `lukedhlee/apptainer-opencode-bridge`, 30 commits → 2)

| # | Commit | What it does for a run |
|---|---|---|
| 7 | Apptainer environment backend | Runs a Harbor task inside a SIF on a JURECA CPU node, fully offline: binds repo mirrors + tools, mounts artifacts, grades r2egym / SWE-bench without network |
| 8 | JURECA worker fleet | Slurm worker chain that keeps scale + walltime across restarts; reaper that spares sandboxes with a live agent; staging sweep that spares live sandboxes |

Dropped from harbor: all OpenCode agent fixes.

## Order of work

1. Freeze: tag old tips (`archive/vista-20260830`, `archive/jupiter-worktree-0814`, `archive/engine-init-batch`, `archive/apptainer-opencode-bridge`). Commit the Mac's pending changes first.
2. Fresh clones on the Mac; one branch each; commits 1–6, one purpose per commit, amend rather than fixup.
3. Jupiter: new clones + venv on a **no-purge tier** — `/e/data1/mmlaion/lee27/code/` (fallback
   `/e/project1/transfernetx`). Not fscratch: its per-file 30-day purge silently deletes untouched source
   files; the current `$F/repos/` clones start decaying ~09-13. Verify compute sees the path and imports
   aren't slower.
4. Gate: base30b GSM8K smoke + one currease arm for ~10 steps from the new clones; first-signal numbers in the
   Atlas "Run it" section must match.
5. Switch the Jupiter symlinks, delete old clones, open the first PRs (one per commit, cherry-picked onto
   upstream `main`). Rebase the working branch onto upstream before each campaign, never mid-run.

## Open decisions

- Branch name for the single working branch (`lukedhlee/jupiter` proposed).
- Where `ai_memory/` and the dashboard live once out of the repo.
- Ask Ben: write access to OTA + MarinSkyRL.

## Source anchors — where each commit's code is lifted from

Old OTA branch `lukedhlee/vista-moe-grpo-30b` (tip `5c6096cc`), merge-base with `origin/main` =
`3772221c`. `git diff 3772221c...lukedhlee/vista-moe-grpo-30b -- <files>` prints exactly the code to port;
`git log 3772221c..lukedhlee/vista-moe-grpo-30b -- <file>` shows why each line exists.

| # | Commit | Lift from (files on the old branch) | Leave behind |
|---|---|---|---|
| 1 | Jupiter cluster entry | `hpc/hpc.py` — the Jupiter `HPC(...)` entry, env exports (FlashInfer workspace, Ray log/spill dirs, `LD_LIBRARY_PATH`), `RL_PYTHON` resolution block in `hpc/sbatch_rl/universal_rl.sbatch`; `hpc/dotenv/jupiter.env` (template) | the 91 `node_exclusion_list` edits (operational state → dotenv/gitignored); Vista entry, `gpu_gres` flag, TACC conda pin in `rl_launch_utils.py`/`launch_utils.py`; hard-coded `/e/scratch/reformo/lee27/...` paths (commit `bed3cd1c`) → read from dotenv |
| 2 | Sandbox egress proxy | `hpc/hpc.py` sbatch proxy block (`PROXYCHAINS_BIN_OVERRIDE`, `PROXYCHAINS_SOCKS5_PRESET_{HOST,PORT,AUTH}`, `chmod 600` on the conf); `hpc/ray_utils.py` + `hpc/rl_launch_utils.py` (override + non-executable skip, commits `a8fc4a24` `cfa78591` `af39808e`); `hpc/datagen_launch_utils.py` `_resolve_proxychains_binary` (`fff9333d`); `hpc/launch_utils.py` secrets.env comment strip | `DCFT_POST_ENV_EXPORTS` tracegen hook (`422815e8`, datagen-only) |
| 3 | MoE RL configs + run scripts | `hpc/skyrl_yaml/jupiter/6node_qwen3_30b_a3b_base_gsm8k_grpo_fsdp2_arms_fa.yaml`; `6node_currease30b_grpo_{base,instr2507,instr2507_lr1e6,instr2507_lr5e6,instr2507_lr8e6}.yaml`; `hpc/skyrl_standard/jupiter/{run_gsm8k_moe30b_grpo.sh,prep_gsm8k_parquet.sh}`; `universal_rl.sbatch` `NCCL_BLOCKING_WAIT=1` (`5f9bcb44`) + `ulimit -c 0` (`c7d92d20`); `hpc/rl_config_utils.py` ref-model path + `hf_hub_repo_id` guard | `run_r2egym_*.sh`, `vista/*`, `24node_*`, `*megatron*`, `fix_flash_attn_cute.sh`; the `save/load_optimizer_states` optional-pattern lines (Megatron) |
| 4 | Housekeeping | `.gitignore` (+`ai_memory/`); `scripts/wandb/{jupiter,jureca}_sync_offline.sh`; `build_rl_fa_env.sh` + `smoke_rl_fa.sbatch` (cluster-only today: `$F/envs/`); `hpc/dotenv/jupiter.env` template with secrets removed | `hpc/dotenv/jupiter.lee27.env` (personal), `scripts/dashboard/*`, `scripts/analysis/gsm8k_*`, `scripts/jupiter/ghost_*` |
| 5 | MoE reference-worker fixes | MarinSkyRL `d9946bd5`, `c5ca6352` (`skyrl-train/skyrl_train/workers/...` `_fsdp_moe_model_kwargs`) — check upstream `main` for `a906145` first | — |
| 6 | Per-group GRPO diagnostics | MarinSkyRL `4f62f796` (`diag_utils.py`, trainer hook, `ppo_base_config.yaml` keys `diag_group_metrics`, `dump_train_rollouts`) — compare with upstream #280/#138 | `69f6d74c` (Megatron ckpt + offline tracking snapshot) except the `WANDB_MODE=offline` guard in `utils/tracking.py` if upstream still lacks it |
| 7–8 | harbor (r2egym) | `lukedhlee/harbor` branch `lukedhlee/apptainer-opencode-bridge` (tip `a12ca6bc`, 30 commits over `marin-community/harbor` main): keep `apptainer` environment + `jureca` worker commits; drop every `fix(opencode)` and `fix(terminus)` commit | OpenCode agent fixes |

## Status log

**2026-08-31 — steps 1–3 done, gate jobs queued.**

- Frozen: `archive/vista-20260830` (OTA), `archive/{jupiter-worktree-0814,engine-init-batch,fix-stridedshard-torch29-20260830}` (MarinSkyRL), `archive/apptainer-opencode-bridge` (harbor). Pushed to the forks.
- New branches (both `lukedhlee/jupiter`, on the forks; Mac worktrees under `~/refactor/`):
  - OTA on `origin/main@2ee2eb61` — 4 commits: `859ad6e0` dotenv-driven Jupiter paths + **gitignored `hpc/dotenv/<cluster>.local.env` overlay** (loader + all 5 sbatch templates; `jupiter.local.env.example` documents it) · `4e2aa3eb` preset-SOCKS5 proxy · `c9a0b8b0` MoE configs (6 YAMLs now declare `context_budget:` — upstream rejects hand-set `max_prompt_length`/`max_generate_length`/`max_model_len`; run script's `MAX_GENERATE_LENGTH`/`MAX_MODEL_LEN` map to `context_budget.*` overrides) · `87d6ad13` housekeeping (venv recipe; see venv notes).
  - MarinSkyRL on `upstream/main@bdfef12b` — 3 commits: torch-2.9 `_StridedShard` import fix · `WANDB_MODE=offline` guard · opt-in `diag/*` per-group metrics + rollout dumps (+6 CPU tests). **Commit 5 (MoE ref-worker) not needed — upstream a906145 is byte-identical.**
- Surprises vs the plan: upstream already had the Jupiter `HPC(...)` entry, `ulimit -c 0`, `NCCL_BLOCKING_WAIT`, the `hf_hub_repo_id` auto-default was removed upstream (guard moot), and MarinSkyRL is now ONE root distribution (PR #284) whose hatch `force-include` copies `skyrl_train`/`skyrl_gym` into site-packages even for `-e` → the venv recipe installs the root then swaps the copies for a `.pth` onto the checkout so `git pull` is live. `reasoning-gym` is a new upstream dep not in our freeze (installed on top; torch/vllm/transformers unchanged: 2.11.0+cu128 / 0.22.0 / 5.8.1).
- Jupiter home is **`/e/project1/transfernetx/lee27/code/`** (`OpenThoughts-Agent`, `MarinSkyRL`, `envs/rl-fa`, `envs/uv-python`), NOT `/e/data1/mmlaion` as planned: /e/data1 creates small files at ~0.75 s each (200 files = 150 s vs 0.06 s on fscratch) — a clone never finished. /e/project1 is no-purge, fast, and an `/e/` path. lee27's overlay: `hpc/dotenv/jupiter.local.env` there (copy kept at `~/refactor/jupiter.local.env.lee27`).
- Gate reference (Atlas "Run it"): GSM8K step-0 reward ≈0.45, rising by step 10; currease reward 0.35–0.45 at step 0–1, `diag/frac_groups_mixed` 0.31–0.44, entropy ~0.17; dead-proxy tell = reward 0.0 + entropy 0 + response_len 1.0. microsocks restarted in tmux `currease_socks` on login02 (10.128.1.2:7011; Daytona health 200 through it).
- Launch commands used: `/e/project1/transfernetx/lee27/code/gate_gsm8k.sh`, `gate_currease.sh` (logs alongside). Note: upstream launcher turns `trainer.logger=wandb` into console when `WANDB_MODE=offline` — metrics are `WANDB_MIRROR` lines in the job log.
- Still open for Luke: (a) step 5 — delete old Jupiter clones under `$F/`, switch any symlinks, open the PRs (one per commit onto upstream main); (b) ask Ben for write access; (c) final home of `ai_memory/` + dashboard (now: stays in the old checkout on the Mac).

**2026-08-31 (later) — first gate attempts died on two launch defects; relaunched as 1553446 (GSM8K) + 1553468 (currease).**

- 1552473 FAILED at 12 s: upstream `jupiter.env` hardcodes `SCRATCH=/e/scratch/jureap59/$USER` and `DCFT=$SCRATCH/OpenThoughts-Agent` then `source "$DCFT/hpc/shell_utils/resolve_rl_repo.sh"` — for lee27 that path is unreadable, `set -e` kills the sbatch before the overlay runs. Fix (in commit 1, `34950b0b`): source the helper relative to the dotenv file (`$(dirname "${BASH_SOURCE[0]}")/../shell_utils`). Upstream-worthy on its own.
- 1552502 cancelled (mine, demonstrably broken): my overlay had `export PROXYCHAINS_BIN_OVERRIDE=<path>  # comment` — **the dotenv loader does not strip inline comments**, the binary path became "path  # comment", proxy setup was skipped → would have been the dead-proxy case. Rule: no inline comments in `*.local.env`. Also `hpc.launch` without `--experiments_dir` writes `experiments/<job>` INSIDE the checkout; pass `--experiments_dir $SCRATCH/experiments/<job>` (the GSM8K run script already does).
- OTA branch now: `34950b0b` · `118f5de7` · `563232b5` · `898423fb` (force-pushed; Jupiter clone reset to it).

**2026-08-31 (attempt 3/4) — two more upstream-drift defects fixed; gates relaunched as 1555904 (GSM8K) + 1555897 (currease).**

- 1553446 (GSM8K) crashed in the trainer on `/e/scratch/jureap59/...` + `/e/data1/datasets/playground` paths: `jupiter.env` line 3 clobbers `$DCFT`, so the sbatch's very next line `[ -f "$DCFT/hpc/dotenv/jupiter.local.env" ]` looked under Ben's checkout and **silently skipped the overlay**. Fix (commit 1, `77521dc5`): every sbatch template captures `_OT_DOTENV_DIR="$DCFT/hpc/dotenv"` BEFORE sourcing `<cluster>.env`. (The Python launcher side was already right — verified by importing `hpc.set_environment` on Jupiter.)
- 1553468 (currease) died on `ModuleNotFoundError: examples.terminal_bench.entrypoints.main_tbench`: MarinSkyRL main deleted it (#386, 08-15) → YAML `entrypoint: skyrl_train.entrypoints.terminal_bench`. 1555059 then died on `Key 'enable_ray_prometheus_stats' is not in struct` → key removed from the 5 currease YAMLs. Both in commit 3 (`a222ab92`). **Upstream OTA main's own Jupiter YAMLs still carry both dead settings — OTA main and MarinSkyRL main are currently not co-runnable; our commit 3 is the fix.**
- Method that found the remaining incompatibilities in one shot: flatten our YAML keys vs `ppo_base_config.yaml` in the Jupiter venv (`NOT IN BASE` list). Remaining non-base keys are OTA-side (`config_groups`, `trainer.enable_db_registration`, `trainer.hf_hub_*`, `engine_init_kwargs.moe_backend`) and are accepted.
- OTA branch now: `77521dc5` · `1774a89c` · `a222ab92` · `4350a0fe`.

**2026-08-31 (GSM8K attempt 4 = 1556228).** 1555904 died at Ray-head start: with `PROXYCHAINS_BIN_OVERRIDE` set (needed for currease) the launcher wrapped the head in `proxychains4 -f "$PROXYCHAINS_CONF_FILE"` even though no proxy was configured, so the conf did not exist → exit 1 "couldnt find configuration file". Fix in commit 2 (`123269fe`): `ray_utils.py` + `rl_launch_utils.py` only wrap when `PROXYCHAINS_CONF_FILE` names an existing file. OTA branch: `77521dc5` · `123269fe` · `02b96d95` · `75c48903`. Cosmetic: the "Check log file:" path in that error points at `$DCFT/experiments/logs/` while the head log is really under `OT_AGENT_RAY_LOG_DIR` (`$SCRATCH/experiments/_ray_logs/`).

**2026-08-31 (currease attempt 5 = 1557218).** 1555897 got past config + proxy and died in the Harbor runner: `ModuleNotFoundError: harbor_config.errors` — MarinSkyRL main pins harbor `df866b30` (uv.lock) + the `harbor-config` release wheel of the same sha; our freeze had harbor `725fc069` (pre-#57 error taxonomy). Installed both into the venv (torch/vllm/transformers unchanged); recipe updated in commit 4 (`185dd0bc`, `HARBOR_REF`). Lesson: **when bumping MarinSkyRL, re-pin harbor from its uv.lock** — the freeze alone is not a complete spec.

## Gate results (2026-08-31, ~11:40 CEST) — read this first

**Both stacks run end-to-end from the clean branches.** GSM8K (1556228) and curriculum-easy (1557218) both reach training steps with the ported diag metrics live. **Numeric parity with the old lr3e6 arm does NOT hold, and the cause is upstream MarinSkyRL, not the port:**

| | step 1 | step 2 | step 3 |
|---|---|---|---|
| old lr3e6 (W&B, 08-14) | 0.336 / ent 0.99 / KL 0.0041 | 0.344 / 1.02 / 0.0048 | 0.383 / 0.84 / 0.0049 |
| old lr8e6 (W&B, 08-14) | 0.336 / 0.99 / 0.0046 | 0.742 / 0.50 / 0.037 | 0.742 / 0.45 / 0.046 |
| **new lr3e6 (1556228, MarinSkyRL main bdfef12b)** | **0.336 / 0.99 / 0.0043** | **0.820 / 0.34 / 0.069** | **0.891 / 0.21 / 0.103** |

Step 1 is identical to the digit (same data order, same rollouts, same ref) → the port is faithful. From step 2 the new run at lr 3e-6 moves faster than the old lr **8e-6**. The launched config is the old one (lr 3e-6, 16×8, 1 epoch, `policy_update_steps=1`, grad-norm clipped 3.2→1.0, AdamW) — so the update *count* and nominal size are unchanged; the policy simply moves ~14× further in KL per step. AdamW's first step is scale-invariant, so this is about which parameters actually receive gradient, not loss scaling. Suspects in upstream `36fdbc0a..bdfef12b` (330 commits): the MoE gradient-correctness cluster — #363 "Zero uninitialized tail rows in grouped-GEMM MoE path" (08-12), #373 "Average FSDP2 expert-parallel gradients" (08-13), #399/#422 "Replay MoE routes / keep checkpoint-replay autograd tapes aligned" (08-16/20). If expert gradients were partly wrong or missing on the 08-14 snapshot, **the new fast learning is the correct behaviour and the old lr sweep (lr8e6 winner) was tuned against a bug.** Not yet proven.

**Control run 1557828** = second checkout `$C/OpenThoughts-Agent-ctl` whose gitignored overlay pins `SKYRL_HOME`/`PYTHONPATH` at the frozen 08-14 MarinSkyRL (`$C/MarinSkyRL-0814`); same venv. (1557785 was cancelled: exporting `SKYRL_HOME` in the launch shell is not enough — the sbatch re-sources the overlay, so a per-run trainer pin needs its own checkout+overlay.) If it reproduces old lr3e6 numbers (0.34 at step 2), the difference is 100 % upstream MarinSkyRL; then bisect the four suspects.

**Currease 1557218 step 1:** reward 0.367 (band 0.35–0.45 ✓), entropy 0.16 (≈0.17 ✓), 8.0 min/step (8–22 ✓), proxy path healthy, 8/8 trajectories per batch. `diag/frac_groups_mixed` = 0.06 vs old 0.31–0.44 — **not comparable**: the ported metric counts groups mixed on the *unshaped* outcome, the old one on reward std; std-based twin here = 1−0.9375 = 0.06 too, i.e. 15/16 groups have identical rewards across their 8 samples (62.5 % all-wrong, 31 % all-correct). Worth a look once the run has 5+ steps (low within-group diversity = little GRPO signal), but it is a step-1 reading of a 16-group batch.

Decision needed from Luke (nothing blocks on it tonight): whether "gate = numbers match" (then pin MarinSkyRL to 08-14 for the currease campaign and treat main as a separate migration) or "gate = runs and learns" (then re-tune lr on main; lr3e6 already behaves like the old lr8e6).

**Control run 1557828 (MarinSkyRL 08-14, same OTA branch + venv) — first readings.** Step 1: reward 0.336, entropy 0.99, KL 0.0042 (identical rollouts) but **`policy/raw_grad_norm` 2.52 vs 3.22 on main** → same data, different backward pass. Step 2 reward **0.42** (old W&B lr3e6: 0.34; main: 0.82). So the step-size change lives in MarinSkyRL `36fdbc0a..bdfef12b`, not in the OTA port or the venv. The MoE gradient fixes (#363 grouped-GEMM tail rows, #373 EP gradient averaging, #399/#422 checkpoint-replay autograd alignment) are the prime suspects; next step if Luke wants certainty is a 4-point bisect on those merges with the same 12-step GSM8K gate (each point ≈ 90 min on 6 nodes).

**VERDICT (2026-08-31 12:25 CEST).** Control 1557828 step 2: reward 0.42, entropy 0.95, KL 0.0057, grad-norm 2.56 — the old lr3e6 regime (W&B 08-14: 0.34 / 1.02 / 0.0048). Main (1556228) at the same lr, same rollouts: 0.82 / 0.34 / 0.069. **The clean OTA branch + new venv reproduce the old behaviour when MarinSkyRL is pinned at 08-14; MarinSkyRL `36fdbc0a..bdfef12b` alone changes the effective policy update by ~14× in KL.** Gate status: *runs-and-learns* PASSED for both stacks; *numeric parity* PASSED with MarinSkyRL pinned at `c5ca6352`, FAILED on `main` — and that failure is upstream behaviour, not the port.

Practical consequence for the currease campaign: either pin `SKYRL_HOME` at `$C/MarinSkyRL-0814` (known lr sweep stays valid) or move to main and re-sweep lr (lr 3e-6 on main already behaves like the old 8e-6). Bisect probe launched at the parent of #399 to split {#363, #373} from {#399, #422, #454}.

**Bisect log (GSM8K lr3e6, 6 nodes; "small" = old regime ≈0.34–0.42 / KL≈0.005 at step 2, "big" = ≈0.82 / KL≈0.07).**

| MarinSkyRL commit | date | has | step-2 reward / entropy / KL | grad-norm@1 | regime |
|---|---|---|---|---|---|
| `c5ca6352` (our 08-14 snapshot, control 1557828) | 07-21 base + our picks | — | 0.42 / 0.95 / 0.0057 | 2.52 | small |
| `0fdb50f6` (probe 1, 1558819) | 08-16 | #276 #329 #339 #363 #373 | 0.42 / 0.88 / 0.0051 | 2.06 | **small** |
| `20cdcf7b` (probe 2, 1559326) | 08-20 | + #399 #405 #422 | 0.445 / 0.82 / 0.0045 | 2.35 | **small** |
| `30c49275^` (probe 4, parent of #452) | 08-23 | + #441 #442 #448 | pending | | |
| `7f43cf9b` (probe 3, 1560170) | 08-23 | + **#452 stochastic-round bf16 AdamW** #453 #454 | pending | | |
| `bdfef12b` main (1556228) | 08-30 | + #441 #442 #448 #454 #468 #474 … | 0.82 / 0.34 / 0.069 | 3.22 | **big** |

So #363/#373 (the grouped-GEMM tail-row and EP-gradient-averaging fixes) are NOT the cause. Probe checkouts: `$C/MarinSkyRL-bisectN` worktrees (+ our tracking.py offline guard applied uncommitted, else wandb.init hangs on compute) launched from `$C/OpenThoughts-Agent-bN` whose overlay pins `SKYRL_HOME` and whose GSM8K YAML has the two `diag_*` keys stripped (they don't exist upstream). Job 1556228 (main) hung at step 12 in the policy update (12/32 for 30 min) — cancelled; nodes `jpbo-046-[17,19-20,23,27,31]` in `$F/_hang_log.txt`. Currease gate 1557218 COMPLETED 10/10 steps (1h25m, exit 0).

**Prime suspect (2026-08-31 13:40): MarinSkyRL #452 "Stochastically round BF16 AdamW updates" (`30c49275`, 08-23).** Mechanism: the policy keeps bf16 master weights; an AdamW step of lr·sign ≈ 3e-6 on weights of magnitude 1e-2…1e-1 is far below bf16's ~4e-3 relative resolution, so with round-to-nearest most per-parameter updates were silently discarded. Stochastic rounding applies them in expectation. Consequence: **the old lr sweep (lr1e6/3e6/8e6) was measuring "how much of each update survives bf16 rounding", not the algorithm.** Under #452, lr 3e-6 already behaves like the old lr 8e-6, so a fresh lr sweep on main (start ≈1e-6) is needed; the old 08-14 pin would keep training against the rounding artefact. Probes 3 (with #452) and 4 (its parent) are running in parallel to confirm.
