# Repo refactor plan — one clean OTA, one clean SkyRL, harbor on the shelf

Opened 2026-08-30, pruned same day. Status (2026-08-31): **steps 1–3 done, step 4 (gate) running as Jupiter jobs 1552473 (GSM8K lr3e6, 12 steps) + 1552502 (currease instr2507, 10 steps); step 5 awaits Luke.** See "Status log" at the bottom. Companion: the OTA Code Atlas
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
