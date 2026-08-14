# JSC storage map for lee27 — memo for agents (surveyed 2026-08-14, jutil snapshots 01:21 CEST)

Purpose: which project/path to use for what, on JUPITER/JUWELS/JURECA. Re-verify hot numbers with
`jutil project dataquota -p <proj>` (it's an HOURLY-OR-WORSE stale cache) and probe-write seconds
before relying on any path. Percentages below are vs SOFT limits.

## Non-negotiable platform rules
- **JUPITER compute nodes see ONLY `/e/*` paths.** Every `/p/*` path is login-node-only there.
  JUWELS/JURECA compute CAN read `/p/*` (incl. /p/data1).
- **Purge clocks (per-file mtime; atime unreliable):** fscratch tiers = 30 days; scratch tiers = 90
  days; empty dirs = 3 days. project1/data1 = no purge (quota only). $ARCHIVE = tape, login-only.
- **Inode policy is punitive**: ~3–4M/project typical; crossing a hard limit locks the project for
  ALL users and can get a user banned. `ulimit -c 0` in every sbatch. Prefer few-large-tars.
- **GPFS in-doubt inflation**: high create/delete churn (agentic staging at conc ≳384) EDQUOTs any
  fileset within ~1h regardless of real usage; reconciles only when quiet. Size concurrency to churn.
- Group-quota billing: files bill the fileset+group of where they LAND (setgid dirs make this sane);
  symlink redirects work for relocating staging without touching worker config.

## Membership (lee27): reformo, laionize, synthlaion, transfernetx, ccstdl, mmlaion
Pending/recommended joins: datasets, cstdl (data projects → /p/data1/…), open-sci-mm (extra JUPITER
budget; Marianna=nezhurina1 is a member). jutil refuses quota queries for non-member projects.

## Health snapshot (2026-08-14; RE-CHECK before use)
| Path | Free (data / inodes vs soft) | Notes |
|---|---|---|
| /p/scratch/reformo | OVER-soft data 🔴 / 1.5M | blanchon1 87TB; unusable till they clean |
| /e/scratch/reformo | 108TB / **AT 8.8M HARD** 🔴 | venv lives here; effectively unwritable |
| /e/project1/reformo | over-soft / at-soft 🔴 | full both ways |
| /e/fscratch/reformo | 13TB / 4.7M | main working tier (marianna_repro); 30d purge |
| /p/scratch/laionize | 30TB / 1.75M | in-doubt-sensitive (r7 lesson) |
| /e/scratch/laionize | 176TB / 4.8M 🟢 | good overflow |
| /p/scratch/synthlaion | 75TB / 3.4M 🟢 | shared with 18 members; probe first |
| /p/project1/synthlaion | 11TB / **160k** ⚠️ | inode-tight; tars only |
| /p/scratch/transfernetx | 76TB / 1.77M | fine |
| /p/project1/transfernetx | 2.7TB / **8k** 🔴 | avoid |
| /e/scratch/transfernetx | **187TB / 7.5M** 🟢🟢 | best scratch headroom anywhere |
| /e/project1/transfernetx | 18TB / 3.2M 🟢 | durable model library candidate (verify write+bandwidth) |
| /e/fscratch/transfernetx | 43TB / 8M, empty 🟢 | pristine fast tier |
| /p/data1/mmlaion | 147TB / **150k** 🔴-inodes | no purge; SINGLE-TAR archives only, never file trees |
| /e/data1/mmlaion | **~11PB / 40M, EMPTY** 🟢🟢🟢 | no purge; UNVERIFIED write perms + compute mount — probe before adopting as archive/model home |

## Placement policy
- Code/configs/recipes → git (OpenThoughts-Agent repo). The battle-tested marianna_repro sbatch/lists
  should be committed, not scratch-only.
- Rebuildables (venv, SIFs, HF caches, staged datasets) → scratch/fscratch, accepted disposable;
  rebuild recipes in git. Venvs die ~90d after install mtime — expected.
- Trained checkpoints / results → HF `laion/` (+ Supabase register) = system of record. Cluster-side
  archives → single tar into /p/data1/mmlaion (today) or /e/data1/mmlaion (after probe).
- Big training data (>1TB) → /p/data1 projects (datasets/mmlaion/cstdl), never scratch. On JUPITER,
  stage login-side to /e/fscratch before jobs.
- Models: durable copy in /e/project1/transfernetx (or /e/data1/mmlaion if verified) → prologue rsync
  to /e/fscratch for serving I/O.
- Staging for agentic rollouts: fileset with ≥2M free inodes, conc sized to churn (≤~192 total),
  dir-count watchdog from minute 1 (alarm >900), probe-write seconds before submit.

## Compute budgets (Aug 2026)
reformo@JUPITER 66M core-h left (33%; expires 2026-12-31 — USE IT). laionize@JUPITER 37.5M (90%).
laionize@JUWELS 4.1M (83%; CPU fleets). synthlaion@JURECA CPU 2.0M / GPU 6.8M. transfernetx@JUPITER
unmeasured — run `jutil project cpuquota -p transfernetx`. lee27 lifetime usage ≈ 1.3k node-h Jupiter
(~0.2% of reformo) + ~1k node-h JUWELS. Jupiter: 1 node = 4×GH200 = 288 cores; core-h = node-h×288.
JUPITER stability: >64-node jobs "high chance to hang/crash"; scaling ladder + short probes first.

## Sources
LAION JSC ops doc: https://iffmd.fz-juelich.de/WPHnb-DoTAKN3BN0afc9MQ (append /download for raw md).
JUPITER FAQ (purge windows): https://apps.fz-juelich.de/jsc/hps/jupiter/faq.html
Session postmortems: ai_memory/NEXT_SESSION.md §0.0-SWEEP-PARKED, ai_memory/gotchas.md (2026-08-14).
