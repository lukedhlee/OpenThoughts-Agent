# TMax on Daytona: one shared ubuntu 22.04 snapshot, each task's own Dockerfile replayed at start

**The conversion is built and its replay machinery has been tested. It has not run on Daytona yet. TMax becomes
runnable after four things: two snapshot slots are freed (the eval org holds 41 against its cap of 40), the one TMax
snapshot is built, the no-op gate passes, and the relay runs on the gate's pass list.** The task lists and how they
were cut are in `../tmax_pool.md` (3,119 strict, 8,068 lenient, with TB-Hard removed).

## How a TMax task runs on Daytona

Every TMax task is a small recipe: a Dockerfile `FROM ubuntu:22.04`, a `post_install.sh` that apt/pip-installs what the
task needs and writes its files, and sometimes fixtures. The v2 tasks also have a shared `base_install.sh` (a large
toolchain plus a CPU torch stack). There are only four Dockerfile shapes. So one snapshot can serve every task, and
each task's own steps run when its sandbox starts. This is the CalibForge setup-files pattern
(`data/calibforge/daytona`), except that the task's steps are replayed rather than its image layers stacked.

- **One snapshot**, `harbor__05b5bff9013d__snapshot`: `ubuntu:22.04` pinned to jammy-20260410 (`tmax_base.py`)
  plus harbor's agent tooling. Its name depends only on that recipe, so every task and every rebuilt tree maps to it.
- **`setup_files/setup.sh`** replays the task's Dockerfile after `FROM` with Docker's build semantics
  (`setup_template.sh`):
  - **ENV** values reach every later step, and the same values go into `task.toml` so the agent's shell sees them.
  - **COPY** copies from the task's own `environment/` dir, shipped as `setup_files/context/`. The file keeps its
    original mode (1,635 fixtures are executable oracles) and is owned by root.
  - **RUN** runs as root under `/bin/sh -c` from `/`, with only the image's environment. A non-zero exit fails the
    trial before the agent starts.
  - **Leftover processes.** After each RUN, setup.sh kills every process the step started. A Docker build step leaves
    nothing running, and 5 % of post_install scripts start a background process or a service.
  - **Cleanup.** It puts back `/etc/hosts`, `/etc/hostname` and `/etc/resolv.conf` (a build never persists edits to
    them), and it deletes `/setup_files`, so the agent never sees the recipe.
- **`task.toml`** keeps TMax's own file, adding `workdir = "/"` (the image sets no WORKDIR), the image's ENV, and the
  relay budgets. TMax's own budgets stay under `[metadata]`.
- **Why replay rather than stacking prebuilt layers.**
  - TMax's prebuilt images (`hamishi740/swerl-tmax-v3:<tag>`, one per task) exist, but they are the swerl variant,
    not this recipe: the v2 task I sized is 172 MB in all, with no `base_install` stack.
  - They sit on at least two different ubuntu bases, so they would need two or more slots.
  - They would need about 14,500 metered tag-manifest pulls and a mirror of 50–170 MB per task.
  - Replay needs one slot and no mirror. Its costs are setup time (apt/pip at start) and archive drift since May, and
    the gate measures both.
  - If the gate fails badly on either, layer stacking from those images is the fallback.
- **v2 tasks** (74 lenient, 43 strict) replay `base_install.sh` at start too, so they need no second slot.
  - That install is slow, an estimated 5–15 minutes.
  - If the gate shows it is too slow, drop these tasks (under 1 %) rather than spend a slot.
- `data/tmax15k/generate.py` (July) predates the setup-files hook. It bakes a union package set into the base, which
  changes every task's environment, and it pastes the setup script into the agent's instruction. It is not used.

## Budgets

**TMax gives every task 600 s for the agent and 120 s for the verifier. The relay should use 1,800 s and 600 s, which
are `build_tree.py`'s defaults.**

- **Agent budget.** The relay's `CLOCK=repair` charges the student's serving time to the budget.
  - On CalibForge, timeouts followed the budget, not the task: in the Qwen-alone baseline, 27 % of 600 s tasks timed
    out, 13 % of 900 s and 2 % of 1,800 s.
  - Timeouts from latency are weak rows that the matcher drops.
  - 1,800 s is the CalibForge relay's longest budget, the regime where the relay is known to behave.
- **Verifier budget.** Many TMax verifiers run a fuzz loop against an oracle, thousands of executions. In R2E-Gym
  arms, a 150 s verifier budget scored 4–14 % of trials as false zeros under load.

## Internet

- **Every TMax task sets `allow_internet = true`, and every setup needs it.**
  - All 14,601 post_install scripts run apt-get and pip, 1,850 also curl or wget, and 64 run git clone.
  - The tests use the network in 1,823 tasks, and 114 reach outside hosts.
  - Daytona sandboxes have internet, and the Horizon driver reaches Daytona through its tunnels as it does for
    CalibForge. Nothing else changes.
- **The load risk is new.** Hundreds of sandboxes starting at once each pull about 50–150 MB from `archive.ubuntu.com`
  and PyPI. The gate's staging (20, then 200, then all) checks that before a relay run depends on it.

## Snapshot slots (read-only census, 2026-09-29 15:50 PT; nothing created or deleted)

**TMax needs one slot. The org holds 41 custom snapshots against a cap of 40, so two must go before the TMax build,
which then brings it back to 40/40. Build right after they are freed: eval runs keep adding per-task snapshots (12
appeared at 15:20 PT today).**

- **Idle, with an owner who has to agree:** the 14 env-gen campaign snapshots of key `…ded` (another workstream; see
  `.claude/projects/daytona/daytona.md`). They have 0 sandboxes, and were last used on 09-22 between 08:58 and
  23:33 PT (created 09-22).
  - `cap-d14-comb-slot7-tools-v4`, `cap-d14-comb-slot3-eval-v4`, `cap-d14-comp8-slot4-eval`,
    `cap-d14-comp8-slot4-eval-v2`, `cap-d14-fxunit-slot9-solver`, `cap-d14-fxunit-slot9-verifier`,
    `cap-d14-thermo-slot6-cell`, `cap-d14-slot5-s2-toolchain`, `cap-d14-slot5-verifier`,
    `cap-d14-slot10-verifier-py312`, `d14s9-toolchain-s1`
  - `cap-verifier-e8c921bc0d1f6ecd5b06`, `cap-verifier-9bc09c38bb8f6e936d5d`, `cap-verifier-9712dfa9da683904d4cc`
  - Snapshots from the same campaign were deleted on 09-24 (with Luke's go) and on 09-27.
- **Not candidates:**
  - CalibForge's three: in use by the remainder run, 251 live sandboxes.
  - Every other `harbor__*` snapshot is a terminal-bench eval image (its recipe carries the TB canary string).
    - `harbor__5e39a065f612` has 271 live sandboxes, and `harbor__9d3e262109f7` (TB2 `train-fasttext`) has 5.
    - 12 were built today at 15:20 PT.
    - Ten built between 09-28 17:20 PT and 09-29 12:04 PT have had no use since 12:04–14:28 PT today. Their owner is
      unknown; they are likely TB2.1/TBLite evals, which Luke said to keep.

Build, once two slots are free (about 2 minutes; the one create this plan needs):

```bash
PYTHONPATH=/Users/lukedhlee/harbor-wt/snowball-r2egym/src /Users/lukedhlee/harbor/.venv/bin/python \
  data/calibforge/daytona/build_snapshots.py --tree <tmax tree>      # reads Dockerfile.jammy; --dry-run prints the plan
```

## No-op gate

**One no-op sandbox per task.** The task passes when three things hold:
- setup.sh ran;
- the task's own `test_initial_state.py` collects tests and all of them pass;
- the untouched sandbox fails `test_final_state.py` for a real reason: reward 0, with tests collected and at least one
  failing.

TMax's Harbor tree dropped `test_initial_state.py`, so `build_tree.py --gate` restores it from `allenai/TMax-15K`
(present for 14,600 of 14,601 tasks) and adds a gate `test.sh`. Every other file is the relay tree's, so the gate tests
the exact setup the relay will use. `gate.py report` sorts failures into five classes:
- `setup_failed`;
- `initial_failed`;
- `final_passes_untouched` (a verifier that gives reward for nothing);
- `final_broken` (collection errors);
- `no_verifier_output`.

It writes `gate_pass.txt`, and the relay runs on that list.

```bash
export PYTHONPATH=/Users/lukedhlee/harbor-wt/snowball-r2egym/src; PY=/Users/lukedhlee/harbor/.venv/bin/python
$PY data/tmax/daytona/build_tree.py --src <TMax harbor/> --only data/tmax/tmax_strict.txt --out <gate tree> \
    --gate <allenai TMax-15K train parquet>
$PY data/tmax/daytona/gate.py sample --tree <gate tree> --n 20 --out pilot20.txt
$PY data/tmax/daytona/gate.py run --tree <gate tree> --tasks pilot20.txt --jobs <dir> --conc 20      # then 200, then all
$PY data/tmax/daytona/gate.py report --jobs <dir>/tmax_nop
```

**Cost estimate (to be replaced by the pilot's measured setup times).**
- **Per sandbox.** A legacy task should live about 1.5–4 min: 5 s to create, 1–3 min of apt/pip, and 0.2–1 min of
  tests. A v2 task should live 10–20 min.
- **Strict (3,119).** About 90–220 sandbox-hours, 20–45 min of wall time at 300 concurrent sandboxes.
- **Lenient (8,068).** About 210–560 sandbox-hours, 45 min to 2 h at 300.
- **Staging.** The pilot of 20 costs about 1 sandbox-hour and 15 min. A 200-sandbox burst then checks apt and PyPI
  under load.

**To start it:**
1. Two slots are freed.
2. The snapshot is built.
3. `gate.py run`. It refuses to start unless the snapshot is ACTIVE, because with `auto_snapshot` harbor would
   otherwise build it on the first trial.

**After the gate**, the relay follows the CalibForge steps:
1. `build_tree.py --only gate_pass.txt` for the relay tree, and `pilot_tasks.py router-json` for `router_tasks.json`.
2. Copy the tree to Horizon, then run `TREE=<it> SHUFFLE_SEED=<n> launch_driver.sh`.
3. Set `CONC` about 15 % higher than for CalibForge. Each trial holds its slot through minutes of setup.

## What has been tested

- **Prototype tree, 20 tasks.** The sample covers v2 with and without fixtures, executable oracles, background
  services, `/etc/hosts` edits and outside-host tests.
  - All 20 share one environment hash and parse as harbor `TaskConfig`.
  - Every script passes `bash -n`, and `router-json` builds.
  - The relay tree and the gate tree differ only in `tests/`.
- **Replay machinery, synthetic task.** It ran on a Horizon compute node in apptainer, with fakeroot, its own PID
  namespace and arm64 jammy-20260410.
  - ENV and `$PATH` expansion, `HOME=/root`, cwd `/` and the COPY modes (755 oracle, 640 data) were all right.
  - Both background processes were killed, and a process started before setup survived.
  - `/setup_files` and `/tmp/post_install.sh` were gone, and a failing RUN exited 10 with its output tail.
- **Not yet tested:**
  - a real post_install (apt/pip);
  - the `/etc/hosts` restore (apptainer mounts it read-only);
  - harbor's upload and exec on Daytona.

  The gate pilot is the first run that exercises them.

## Files

- `select_tasks.py`: the task lists (`../tmax_*.txt`).
- `build_tree.py`: the relay or gate tree.
- `tmax_base.py`: the base recipe.
- `setup_template.sh`: the replay.
- `gate.py` and `gate_nop.yaml`: the no-op gate.
- Reused unchanged: `data/calibforge/daytona/build_snapshots.py` (creates the snapshot from `Dockerfile.jammy`),
  `snapshot_census.py`, and `data/relay/pilot/pilot_tasks.py router-json`.
