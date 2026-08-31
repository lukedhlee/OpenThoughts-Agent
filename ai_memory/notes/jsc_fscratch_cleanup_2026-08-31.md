# JSC fscratch cleanup (2026-08-31) — per-user cap alert: 7.4 TB → <2 TB, 1.77M → <400k files

Jenia's two emails (08-30 15:09/15:10 KST) introduced a **per-user cap on `/e/fscratch/reformo`: 5% of
the project soft quota = 2 TB and 400k files per user**. lee27 was at 7.4 TB (18.6%) and 1.77M files
(22.1%). This note records what was where, what was archived where, and how to get it back.
Companion: [[jsc_inode_quota_2026-08]] (the 08-22 alert + the bytes≠inodes lesson), [[jsc_storage_map]].

## What matters

Almost all the bytes were **checkpoints of two RL campaigns** (curriculum-easy fleet 4.7 TB, GSM8K
validation 2.0 TB) and almost all the files were the **curriculum-easy fleet's rollout traces** (960k).
Everything with reuse value was archived to no-purge `/e/data1/mmlaion/lee27/` as single tars before
deletion; the working environment (`envs/`, `repos/`, `marianna_repro/datasets`) was left intact.

End state, **measured 08:51 CEST 08-31** (`_inode_survey/fscratch_d3_0831_0851.tsv`; `jutil` lags hourly):
**43,701 files / 236 GB** on `/e/fscratch/reformo/lee27`, from 1,865,469 / 8,190 GB at 08:12. What remains:
hf_hub model caches 139 GB, `models/` 88 GB (Qwen3.6-35B + g1 8B), and ~40k files across `repos/`,
`marianna_repro/`, `tasks/`, `cache/`. The three uv venvs (268k files, 19 GB) were **relocated to
`/e/fscratch/transfernetx/lee27/envs`** with a symlink left at `$F/envs`, so every YAML/sbatch path still
resolves (verified: `torch 2.11.0+cu128` imports through the symlink). They remain on a 30-day-purge tier
and purge ~09-13..16 by mtime — rebuild recipes are committed under `hpc/env_builds/jupiter/`.

Jenia's stated destination `/e/data1/datasets/playground/mmlaion/` is **not writable by lee27** (no
`datasets` group, probed 08-31). **Luke decided 08-31 not to pursue `datasets` for now — mmlaion is the
archive home.** A reply draft to Jenia was handed to Luke.

## Archive index — `/e/data1/mmlaion/lee27/`

| tar | what | restore |
|---|---|---|
| `currease30b_grpo/currease30b_grpo_instr2507_ckpt_gs45.tar` (197G) | baseline arm (lr 3e-6) latest FSDP ckpt + `latest_ckpt_global_step.txt` | `tar -xf … -C <exp>/<arm>/checkpoints/` then the resume recipe in [[currease_pass8_probe_handoff]] §FLEET PAUSED |
| `currease30b_grpo/currease30b_grpo_instr_lr1e6_ckpt_gs30.tar` (197G) | lr1e6 arm latest ckpt | same |
| `currease30b_grpo/currease30b_grpo_instr_lr5e6_ckpt_gs30.tar` (197G) | lr5e6 arm latest ckpt | same |
| `currease30b_grpo/currease30b_grpo_instr2507_model_gs40.tar` (57G) | loadable HF model, baseline gs40 (pass@8 46.0 vs 41.8) | `tar -xf` → `policy/` |
| `currease30b_grpo/<arm>_run.tar.zst` ×5 (0.4–1.1G each) | per-arm run dir minus weights: trace_jobs, ray_logs, **never-synced wandb offline runs**, driver logs, configs, sbatch, diag exports | `tar -I zstd -xf … -C <experiments>/` |
| `currease30b_grpo/currease_pass8_probes__repo_experiments.tar.zst` | the pass@8 probe datagen runs (base/gs40/instr2507) that lived in `repos/OpenThoughts-Agent/experiments/` | `tar -I zstd -xf … -C repos/OpenThoughts-Agent/` |
| `parked/marianna_repro_slurm_logs__parked_r2egym_sweep_resume_state.tar.zst` | the ~1,750 banked r7 trials + job dirs the sweep resume needs | `tar -I zstd -xf … -C $F/marianna_repro/` **before** the resume recipe in NEXT_SESSION `0.0-SWEEP-PARKED` |
| `parked/tasks_r2egym-raw.tar.zst` | raw r2egym task set (41k files; also rebuildable from HF) | `tar -I zstd -xf … -C $F/tasks/` |
| `base30b_gsm8k/*` (08-28/29) | GSM8K campaign results + 4 gs81 models | see [[base30b_gsm8k_validation]] |

`/e/project1/transfernetx/lee27_archive/fscratch_experiments_stale_0831.tar.zst` — every other
superseded run dir under `experiments/` (STALE_*/INVALID_*/FAILED_*/band pilots/smokes/canaries/ota10k
SFT logs), weights excluded. Plus `marianna_deepswe_repro_copy_0812.tar.zst` and the stale
`OpenThoughts-Agent-r2egym-bridge-next` clone.

## Deleted without archive (pre-approved by Luke 08-31)

- Curriculum-easy **older** checkpoints (gs5…gs40 / gs5…gs25 ×2) — 3.6 TB; superseded by the latest.
- `currease30b_export_gs40/{checkpoints,checkpoints_out}` — redundant FSDP copies of baseline gs40 (132 GB).
- GSM8K campaign: all 85 dirs incl. 1.57 TB FSDP resume state + 285 GB exports (archived 08-28/29).
- `experiments/_weights_kept` (262 GB) + all `checkpoints/exports` of the stale set (123 GB, 200 dirs) —
  Aug 5–11 band/sweep pilot weights.
- `checkpoints/ota10k_sft_30ba3b_levanter` (37 GB) — corrupted lineage (SFT-59).
- **Aug-4 Qwen3.6-35B "canary" MILESTONE weights, 600 GB** (`jupiter_qwen36_35b_r2egym_grpo_canary_n4t1800_20260804h`:
  gs1/gs2 FSDP state 470 GB + HF exports 130 GB). A canary = short 4-node pre-flight that dies first if the
  pipeline is broken; this one proved the first end-to-end GRPO update+ckpt+export on Jupiter. Proof of
  pipeline, not a model — deleted on Luke's call 08-31. Its logs/configs/traces are in the stale tar.
- **8 core dumps, 157 GB**, in `repos/OpenThoughts-Agent/core.jpbo-001-*` — incident-44 leftovers.
- `cache/jax_comp_cache`.

## Method (reusable)

- **Survey in one metadata pass**: `find $F -printf '%b\t%p\n' | awk` aggregating inodes *and*
  allocated bytes per depth-3 prefix (`_inode_survey/survey_d3_bytes_inodes.sh`, 6 min for 1.87M files).
  `du` needs two passes for the same answer.
- **Pull weights out first, by rename** (`_ckpt_pending_delete/`), so zstd never sees a checkpoint and the
  arm dirs can be tarred wholesale with no exclude logic. Watch for non-standard names — the canary's
  weights were `MILESTONE_*_checkpoints`, missed by the `-name checkpoints` sweep; and a glob like
  `base30b_gsm8k_lr1e6_*` misses the un-suffixed first attempt `base30b_gsm8k_lr1e6`.
- **Big files → plain `tar -cf` straight to the destination** (~1 GB/s per stream, 2 parallel);
  **small files → `tar -I zstd`** (429k JSON files → 1.0 GB in 2 min). Never tar locally then copy.
- **Delete only from a verified list**: each tar appends its source to `verified_delete.list` after
  `tar -tf | wc -l ≥ find | wc -l`; the delete stream waits for A/B/C OK flags and reads that list.
- Detach with `tmux new-session -d`; `ssh '… nohup … &'` was SIGHUP'd before it wrote a byte.
- Never `pkill -f <pattern>` from an ssh one-liner whose own command line contains the pattern — it kills
  the shell running it (happened here; the rest of the one-liner silently never ran).
- Wall clock for the whole thing: 08:12 survey start → 08:51 final survey = **39 min**, ~7 TB archived or
  deleted, 1.82M files removed, zero failed verifications.
