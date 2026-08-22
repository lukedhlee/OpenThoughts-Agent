# JSC inode quota alert (2026-08-22) — `reformo` over soft on 3 filesets

What the alert was, what `lee27` actually owned, and the two reusable lessons: **how to measure your
own footprint without a `du` walk**, and **why an inode alert is never solved by deleting big files**.
Read before any JSC cleanup, and before assuming a quota email is about disk space.

Companion: [[jsc_storage_map]] (which tier for what), [[jsc_paths_hazards]] (the ban-risk rules).

## The alert (Jenia, 2026-08-22 01:29)

| fileset | used / soft / hard | our share |
|---|---|---|
| `/p/scratch/reformo` | 4,054,263 / 4M / 4.4M | 3,441 (**0.1%**) |
| `/e/project1/reformo` | 4,301,837 / 4M / 4.4M | **4** (0.0%) |
| `/e/scratch/reformo` | 8,609,320 / 8M / 8.8M | **586,914** (6.8%) |

Stated target: **<100k files per user**, up to 250k only when strictly necessary. A soft-limit breach
can block filesystem access for **every user of the budget** — this is a shared-fate resource.

Top user across all three was `blanchon1` (~6.9M files total); `wijngaard1` and `singhvi1` next.
**Of the three flagged filesets we could only move one** — but `/e/scratch` was over soft by just
127,179 files, so clearing our tree alone brought it back under.

## ★ Lesson 1 — `jutil user dataquota` is the right tool, NOT `du`

`jutil user dataquota` (no `-p`) prints **every project × fileset row for you**, on every tier, in one
call. That is how you find your own footprint without touching GPFS metadata. `jutil project
dataquota -p <proj>` gives project totals. Both are cached (hourly-or-worse), so they lag deletions.

`du --inodes` is still needed to find *which subtree* — but see [[gotchas]] "never `find`/`du` on
Jupiter GPFS". Cost measured 2026-08-22: `/e/fscratch` (3.28M files, flash) ≈ 12 min; `/e/scratch`
(587k files, spinning) was **slower than the 3.28M flash walk**. Run it `nohup`'d, single-threaded,
with `TMPDIR` and output **off `$HOME`** (see Lesson 3).

Our real footprint, which the email only partly revealed:

| fileset | files | bytes |
|---|---|---|
| `/e/fscratch/reformo` | **3,282,680** | 8.6 TB |
| `/e/scratch/reformo` | 586,914 | 5.35 TB |
| `/p/scratch/synthlaion` | 205,731 | 1.69 TB |
| `$HOME` (exa_home) | 6,544 | 21 GB (**at hard limit**) |

The email flagged our 587k on `exa_scratch` but was blind to the bigger number: we owned **54% of the
entire `exa_fscratch` fileset** (3.28M of its 6.09M), which was not yet over soft. *The fileset that
emails you is not necessarily the one you are worst on.*

## ★ Lesson 2 — bytes and inodes are DECOUPLED; an inode alert is not a disk-space problem

Measured on one run dir (`experiments/band512r_s0`):

| subtree | bytes | inodes |
|---|---|---|
| `checkpoints/` | 46 GB | **25** |
| `exports/` | 16 GB | **9** |
| `ray_logs/` | 1.4 GB | **23,043** |
| `trace_jobs/` | 274 MB | **14,706** |

62 GB of model weights costs **34 inodes**. 1.7 GB of traces and ray logs costs **37,749**.
Meanwhile `/e/scratch/reformo` was over its **inode** soft limit while sitting at **2.6% of its data
limit** (5.35 TB of 205 TB).

**Therefore:** to fix an inode alert, delete the many-small-file trees (`trace_jobs`, `ray_logs`,
XLA/compiler dumps, `uv`/`hf` caches, venvs, git checkouts) and **keep every model artifact** — moving
weights aside costs ~34 inodes per run and is an instant same-filesystem rename. Do **not** tar
hundreds of GB of checkpoints "to be safe": it buys nothing on the metric that is actually breached,
takes hours, and the standing harvest rule already says winner checkpoints belong on HF, not in tars.

The converse holds too: a **block**-quota alert is never fixed by deleting trace directories.

## ★ Lesson 3 — the `$HOME` breach was core dumps, for the second time

`ota10k_lev_sft` job **1396849** SIGABRT'd on 8 nodes (`jpbo-114-[01-03,05-06,08,11,13]`) at
2026-08-17 16:31 and wrote **8 × ~4.7 GB cores into `$HOME`** → `exa_home` at **21 GB / 21 GB hard**.

- **Symptom is not "disk full"** — it is any command that needs a *new* file failing with
  `Disk quota exceeded`. A plain `sort` in the survey pipeline died this way. Overwrites still work,
  so the account looks healthy until something creates.
- Cores land in the **crashing process's CWD**, named `core.<node>.<pid>`. Incident 44 put 258 GB in
  the repo cwd; this one put 21 GB in `$HOME`. Same bug, different cwd.
- `levanter_sft_jupiter.sbatch` got `ulimit -c 0` in `64d444f5` — committed **~1 h after** this crash.

**Fixes applied 2026-08-22:** `ulimit -c 0` appended to `~/.bashrc` on Jupiter (JSC's skel `.bashrc`
has **no interactive guard**, so it applies to non-interactive shells too), and added to the four
remaining JSC-targeting sbatch templates (`c7d92d20`). `hpc.py`'s jupiter `ClusterConfig` already
carried `pre_run_commands=["ulimit -c 0"]`, so `hpc.launch` jobs were always covered — the gap was
only in hand-written sbatch files.

## Latent issues found while surveying (not yet fixed)

- **`hpc.py:878`** — the jupiter `ClusterConfig` default `LD_LIBRARY_PATH` points into
  `/e/scratch/reformo/lee27/OpenThoughts-Agent/envs/rl`, which **no longer exists**. Harmless today
  (every live yaml overrides `LD_LIBRARY_PATH`), but it is a dead cluster-wide default.
- **`hpc.py:1001`** — `conda_activate` sources
  `/e/scratch/reformo/lee27/miniforge3/etc/profile.d/conda.sh` and then does `&& true` (it never
  activates an env; the real interpreter is `RL_PYTHON` from the fscratch venv). This is the **only
  live dependency of the entire `/e/scratch` tree** besides `envs/rl-megatron` (146,460 inodes).
  `/e/scratch` is a **90-day purge** tier and that env's mtime is 2026-07-25 → **purges ~2026-10-23**.
  Repoint both defaults to fscratch (or make `conda_activate` a no-op) before then.

## Never delete without checking

- `/e/fscratch/reformo/lee27/marianna_repro/slurm_logs/jobs` — **parked r2egym sweep resume state**,
  ~1,750 banked trials, exact resume recipe in `NEXT_SESSION.md:55`. It *looks* like 90k inodes of
  stale logs. It is not.
- `experiments/currease30b_*` — the paused curriculum-easy fleet's traces.
- `experiments/base30b_gsm8k_*` — 85 dirs, 180,587 inodes, the GSM8K validation campaign.

## Standing prophylaxis

- `ulimit -c 0` in every sbatch **and** `~/.bashrc` on every JSC system.
- Point `UV_CACHE_DIR`/`HF_HOME` at a purge tier and clean them routinely — `cache/uv/archive-v0`
  alone was 147,816 inodes on fscratch and 99,178 on scratch.
- XLA/compiler dumps (`--xla_dump_to`) are an **inode bomb**: one Levanter debugging session left
  909,357 files. Delete them the moment the investigation closes; the method to regenerate is in
  [[ota10k_levanter_sft]].
- Nested `experiments/` inside a git checkout (`repos/OpenThoughts-Agent/experiments`, 71,906 inodes)
  — runs should never write into the repo tree.
