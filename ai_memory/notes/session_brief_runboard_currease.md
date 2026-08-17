# Session brief: Runboard dashboard for the curriculum-easy GRPO sweep

You are a dedicated Claude Code session whose ONLY job is to stand up and
maintain the **Runboard** dashboard for the curriculum-easy GRPO LR sweep on
JUPITER, so Luke can watch reward/diagnostics per arm. A separate "supervisor"
session babysits the RL jobs themselves; another owns the Levanter SFT run.
You never launch/cancel cluster jobs — you only read logs/metrics and build
the dashboard.

## The experiment you are charting

Three GRPO arms, identical except learning rate, RL-training
**Qwen3-30B-A3B-Instruct-2507** on the terminal-agent task set
`tasktrove_curriculum_easy_rl511` (Harbor episodes in Daytona sandboxes,
Terminus-2 harness; binary episode reward). SkyRL/GRPO, FSDP2+EP, 6 GH200
nodes per arm, 8 samples/prompt, batch 16 groups × 8 = 128 samples/step,
**max_steps 50**, ckpt every 5, seed 42, KL on (coef 0.001), TIS cap 2.0.

| arm | lr | job chain (head RUNNING → spares) | wandb run_name |
|---|---|---|---|
| baseline | 3e-6 | 1397229 → 1397230, 1398002, 1398003 | `currease30b_grpo_instr2507` |
| low | 1e-6 | 1394805 → 1394806-08 | `currease30b_grpo_instr_lr1e6` |
| high | 5e-6 | 1396799 → 1396800-02 | `currease30b_grpo_instr_lr5e6` |

State at 2026-08-17 ~20:30 CEST: baseline gs25 (reward 0.375, pass@8 0.44),
lr1e6 gs9 (reward 0.23), lr5e6 gs10 (reward 0.29, pass@8 0.50). All healthy.
(lr8e6 was cancelled permanently by Luke — sweep is 1e-6/3e-6/5e-6.)

## Where the data lives (the crux)

The trainer logs wandb **OFFLINE** (`WANDB_MODE=offline`), one offline run per
Slurm attempt, under EACH experiment dir on Jupiter (ssh alias `jupiter`,
user lee27, `F=/e/fscratch/reformo/lee27`):

```
$F/experiments/<arm>/wandb/wandb/offline-run-*        # arm ∈ the 3 run_names above
```

(~40MB total. `trainer.project_name=jupiter-currease30b-grpo`.) The cloud
project does NOT exist yet — nothing was ever synced. Additionally, every
per-step metrics dict is mirrored to the driver log as a parseable line:
`WANDB_MIRROR kind=train step=N metrics={json}` in
`$F/experiments/<arm>/logs/<arm>_<JOBID>.out` — a full fallback data source.

Recon already done (2026-08-17):
- offline runs rsync fine; `fchmodat: Operation not permitted` warnings are
  cosmetic GPFS noise (use `rsync -a --no-perms` to silence).
- The offline runs appear to carry **no wandb group** (SkyRL sets run_name
  only). `pull_wandb.py` stitches attempts by GROUP — see task 2.
- Precedent: the finished gsm8k campaign's cloud data
  (`lukeleeai/jupiter-base30b-gsm8k-grpo`) came from
  `scripts/analysis/gsm8k_diag_sidecar.py` creating `<run_name>-diag` runs,
  NOT from syncing trainer offline runs. The sidecar itself is gsm8k-specific
  (parses GSM8K answers) — don't reuse it blindly for terminal-bench arms.
- A 4th exp `currease30b_grpo_base` (base-model arm, earlier phase) also has
  offline runs — not part of this sweep; ask Luke before adding it.

## Your tasks

1. **Get the metrics into WandB cloud** (project `jupiter-currease30b-grpo`,
   entity = default for this Mac's wandb auth, expected `lukeleeai`).
   Path A (preferred): rsync the offline runs to your scratchpad, then
   `wandb sync --project jupiter-currease30b-grpo <offline-run-dir>` each.
   Runs from RUNNING attempts sync partially — re-sync later to append.
   Path B (if sync misbehaves): write a small uploader that parses the
   `WANDB_MIRROR` lines from all attempts' driver logs and logs them into one
   cloud run per attempt (this also solves grouping cleanly).
2. **Ensure per-arm grouping**: after the first sync, check via the API
   whether runs carry a group. If not, set it post-hoc
   (`api.run(...).group = "<arm run_name>"; run.update()`) or fall back to
   Path B with `group=` set at init. Spec `series[].group` must match.
3. **Write the spec**: copy `scripts/dashboard/specs/base30b_gsm8k_arms.json`
   → `specs/currease30b_grpo_arms.json`. id/title/wandb_project/series (3
   arms, color_slots 1–3), `step_target: 50`; keep
   metrics/charts/glossary = "rl_default". Attempts/timeline material: the
   incident ledger `ai_memory/notes/currease_pass8_probe_handoff.md`
   (incidents 37–49 cover this sweep's job war: ghost-OOMs, backward hangs,
   the all-zero proxy-env incident 39). Degrades gracefully if you omit them
   on the first pass — ship charts first, gantt second.
4. **Build + open**:
   `/Users/lukedhlee/miniforge3/bin/python scripts/dashboard/pull_wandb.py scripts/dashboard/specs/currease30b_grpo_arms.json`
   then `.../build.py`, then `open scripts/dashboard/dist/dashboard.html`.
5. **Refresh cadence**: re-rsync + re-sync + pull + build after each incident
   or every few hours while arms run (offline mode means NOTHING updates in
   the cloud without your sync). Report reward trends to Luke each refresh:
   `reward/avg_raw_reward`, `reward/avg_pass_at_8`, `policy/policy_entropy`,
   `diag/frac_groups_mixed`, response-length percentiles.

## Gotchas (from Luke, verbatim intent)

- Never hand-write `series_data` — `pull_wandb.py` generates it.
- `dist/` is gitignored; commit `specs/` + `data/` only.
- Shared Mac clone: `git pull -q --rebase` before every commit, push to
  remote **`fork`** (branch `lukedhlee/vista-moe-grpo-30b`). Luke has
  unstaged changes (.gitignore, ssh notes) — leave them; if rebase refuses,
  commit only your files and push.
- **Leave `base30b_gsm8k_arms.*` spec/data alone** — finished campaign.
- No Co-Authored-By lines in commits. Play
  `afplay /System/Library/Sounds/Funk.aiff` after long tasks.
- Read-only on the cluster: never scancel/sbatch anything; never touch other
  users' files or peer tmux sessions.

## Pointers

- `scripts/dashboard/README.md` — full Runboard docs.
- `ai_memory/notes/currease_pass8_probe_handoff.md` — sweep history/incidents.
- `ai_memory/notes/jupiter_node_health.md` — what the ghost/hang incidents in
  the timeline actually were.
