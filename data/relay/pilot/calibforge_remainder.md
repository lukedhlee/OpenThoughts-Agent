# CalibForge remainder: 2,613 tasks, and the 1,000-task pilot drawn from them

`calibforge_remainder.txt` holds every CalibForge task that no earlier run used and that runs on the three CalibForge
snapshots we already hold on Daytona: 2,613 tasks. `calibforge_remainder_pilot1000.txt` is a 1,000-task subset drawn to
match the remainder's mix, for the first Horizon relay run. Both lists are written longest agent budget first, because
harbor starts trials in list order. Neither list shares a task, or an instruction text, with `full_pool.txt` (2,043) or
`calibforge_heldout300.txt` (300). The plan's estimate of ~2,660 came out at 2,613 once every image's manifest was read.

## How the remainder was cut (2026-09-29)

| step | tasks |
|---|---|
| CalibForge, all tasks (`hamishivi/agent-task-calibforge`) | 5,431 |
| minus `full_pool.txt` and `calibforge_heldout300.txt` | 3,088 |
| minus the 86 TB2-templated tasks | 3,002 |
| minus agent budget above 1,800 s (266) | 2,736 |
| minus 1 task whose whole instruction appears inside 21 used tasks' instructions (5 held-out); the router matches episodes by the first 400 characters, so it could not tell them apart | 2,735 |
| minus 122 tasks the three snapshots cannot serve (`build_tree.py` coverage): 52 on `python:3.11-slim`, 4 on other bases, 46 whose working directory is not `/app`, 20 whose own layers exceed 1.5 GB | **2,613** |

No task in the remainder duplicates another task's instruction, in the remainder or in the used lists.

The remainder includes 326 tasks (12.5 %) in which the agent needs internet. `recommended_2500` left those out only because
JSC compute nodes have no internet, but Daytona sandboxes do. The earlier pool never had tasks like these, so check their
pass rate separately in the pilot readout (column `agent_net_strong` in the strata file).

## How the pilot was drawn

The pilot uses the same stratified draw as `pilot_tasks.py pick`. Quotas go to subset x `tb2_gap` by largest
remainder, then to categories inside each stratum the same way, with a seeded draw per cell (`random.Random(20260929)`).
`calibforge_remainder_pilot1000_strata.tsv` gives each pilot task's stratum, category, difficulty, budget and base.

- **Match to the remainder.** Strata match exactly. Categories are within 0.1 point, difficulty within 0.4 and base
  within 1.1. Agent-internet tasks are 12.7 % of the pilot vs 12.5 % of the remainder.
- **Budget, the one visible drift.** 33.2 % of pilot tasks have the 1,800 s budget, vs 36.1 % of the remainder. So the
  pilot's wall time per episode will read slightly low for the full remainder.
- **No cache preference.** The brief asked for pilot tasks whose manifests were already cached, to save Docker Hub pulls.
  I dropped that because all 2,735 candidate manifests were fetched before the draw, so every task was equally cheap.

## Where things are

- **Tree.** Built by `data/calibforge/daytona/build_tree.py --all --only <2,735 candidates>` from
  `~/.local/share/otagent/calibforge-work/tree-remainder-20260929/` on the Mac. The build names the three snapshots we
  hold: `harbor__313e69c036ad__snapshot` (ubuntu2404, 1,355 tasks), `harbor__13bf95818376__snapshot` (tb2verifier,
  1,067) and `harbor__239bee0dfdf6__snapshot` (bookworm, 191).
- **Copy on Horizon.** `/scratch/11584/lukedhlee/relay/calibforge_tree/` holds the 2,613 task dirs and `TASKS.txt` (=
  this remainder). It also has `TASKS_pilot1000.txt`, `router_tasks.json` (all 2,613), `router_tasks_pilot1000.json`,
  `pool.json` and `coverage.tsv`. Its file checksum matches the Mac tree. For the pilot, pass
  `TASK_LIST=$TREE/TASKS_pilot1000.txt`.
- **Layers.** setup.sh reads each layer from the HF mirror `laion/calibforge-daytona-layers` first, and falls back to
  Docker Hub's blob endpoint. Blob downloads are not metered pulls, and every layer's sha256 is checked either way.
  Almost none of these layers were on the mirror yet: 9,692 of 9,905 are new (110.7 GB), and the pilot's share is
  3,764 (42.3 GB).
  - **Pilot.** All 3,892 of the pilot's layers were on the mirror at 13:10 PT on 09-29 (`mirror_upload.py` verify).
  - **Rest of the remainder.** 5,928 more layers (68.4 GB) started uploading at 13:10 PT. Until a layer lands, setup.sh
    fetches it from Docker Hub.
  - **To finish or check it.** `mirror_upload.py --tree <tree> --repo laion/calibforge-daytona-layers --work <dir>`
    resumes, and it ends with a `verify: N/N` line.
- **Manifests.** The manifests and configs are cached in the Mac's calibforge-work dir.
  - They were read by digest from Docker Hub, spread over the Mac (IPv4 and IPv6), Vista, Jupiter and JUWELS logins.
  - About 250 came from `mirror.gcr.io` (`registry.py fetch --registry`), each sha256-checked against its digest.
  - About 2,600 by-digest manifest GETs never moved any source's `ratelimit-remaining` counter off 100, and none got a 429.
  - `registry_cache.tar.gz` in the HF dataset now holds all 5,146 manifests and configs, so the tree can be rebuilt
    without Docker Hub.
