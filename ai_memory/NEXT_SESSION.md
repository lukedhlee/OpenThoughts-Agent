# ⚠ TEMPORARY TAKEOVER NOTE — rewritten 2026-08-04 10:30 KST · DELETE ONCE ABSORBED

Supersedes the 01:00 KST version, whose central call ("hold the band") has since been **reversed by
data**. Read `handoff.md` § "Live state — 2026-08-04 10:30 KST" first, then `gotchas.md` and
`decisions.md` § 2026-08-04.

## Two things you must not get wrong

**1. The band IS required.** The 600s agent-timeout defect was real and is fixed — 39% of trials now
run past 600s and score honestly, times span 140–862s with nothing at the 1800s ceiling. But fixing it
did **not** produce variance. `1229343` gave 18 honest rewards: **0 of 6 groups varied**, and five
tasks scored *identically* to the capped run (`04114` 1,1,1,1 → 1,1,1,1,1; `06360` 0,0,0,0 → 0,0,0).
Both are true: the old measurement was invalid, and its conclusion survives anyway.
⇒ Next action is a **~384-task `p@4` probe** (1,536 rollouts, ~1.7h at 4 rollout + 8 JURECA nodes).
Caveat: 6 *partial* groups on the same 6 task ids — evidence about *these* tasks, not a population.

**2. `/e/scratch` cannot create any new file.** Project-wide 8M inode cap, shared by 26 users,
exhausted. `touch` → `Errno 122`; overwrites still work, so it looks healthy until something needs a
new inode. It killed `1221005` and `1229343` mid-run and **breaks `git fetch`**, so you cannot deploy
into it. Everything writable moved:

⛔ **CORRECTED 2026-08-04 (later) — `/p/scratch` IS NOT MOUNTED ON JUPITER COMPUTE.** The `/p/scratch`
rows below were wrong and would fail on compute. `45572f93` moved the tree there; **`ae180c37`
reverted it to `/e/fscratch`**, and `31cd646d` moved `WANDB_DIR` off `/e/scratch` too (offline W&B
creates a new run dir per job). Verified live on `1229488`.

| what | where |
|---|---|
| experiments tree, checkpoints, HF exports, `WANDB_DIR` | `/e/fscratch/reformo/lee27/experiments` |
| execution checkout (`DCFT`) | `/e/fscratch/reformo/lee27/OpenThoughts-Agent-r2egym-bridge-next` |
| read-only 67 GiB model | `/e/fscratch/reformo/lee27/models/Qwen3.6-35B-A3B/995ad96e…` |

Still on `/e/scratch` and fragile: the `rl-megatron` venv, harbor, MarinSkyRL, FlashQLA,
`TILELANG_CACHE_DIR`. Reads work; writes through them can fail.

## What is running right now

- **Jupiter `1229446`** — PD `(Priority)`, 5 nodes, **4h** wall, 16 × 8 = 128 trajectories.
- **JURECA `15494122`** R (~13h left) **+ `15495516`** PD (24h). Two fleets deliberately overlap so any
  start between now and ~10:15 KST tomorrow has sandboxes.
- **Watcher** `/p/scratch/reformo/lee27/retarget.sh` → `retarget_d.log`. On allocation it resolves the
  head node, retargets the reverse listener, and runs the compute-node route gate **automatically**.
  This automates the step whose omission killed `1225422`.
- Bridge clean (`ready: 0`). Node budget 9/16.

## Scheduling trap you will hit

`ReqNodeNotAvail,_Reserved_for_maintenance` appears while **3,848 nodes are idle** and
`scontrol show res` shows no MAINT reservation. The cause is a window you cannot see that your
**walltime cannot finish before**. Diagnose free with `sbatch --test-only -t <T> <sbatch>`:
**6h → Aug 5 12:00 CEST; ≤4h → same evening.** Every walltime ≤4h gave the same start, so going
shorter than 4h buys nothing.

## If you launch anything

1. `DCFT=/e/fscratch/reformo/lee27/OpenThoughts-Agent-r2egym-bridge-next` — **not** `/e/scratch`
   (stale at `8ac43278`, cannot be fetched into) and **not** `/p/scratch` (not mounted on compute).
2. `TIME_LIMIT=04:00:00` or less until maintenance passes.
3. Confirm a JURECA fleet has **more time left than your walltime plus startup**. Preflight does NOT
   check this — see Open #2 below.
4. Let the watcher do the tunnel retarget, or do it by hand:
   `ssh -S ~/.ssh/cm_jureca/qwen36 -O cancel -R 10.14.0.46:18000:<old>:8000 jureca.fz-juelich.de`
   then `-O forward` to the new head. Internal bind only. Then prove `/health` **and** a real
   `/v1/chat/completions` **from a compute node**, not the login node.
5. Reward lives at **`verifier_result.rewards.reward`**, never top-level `reward` (that key does not
   exist and returns `None` for every trial). Tooling: `python3 /e/scratch/reformo/lee27/bench/tput2.py
   <trace_jobs>` — reads fine, `/e/scratch` reads still work.

## Open

1. **The band probe.** Full set = 3,328 × `p@4` = 13,312 rollouts ≈ **14.5h** at 4+8 nodes (~58h at
   today's 32-way), so it needs chunking across ~4 allocations plus a done-task manifest. The 384-task
   probe fits a single 4h slot. The probe needs **no policy/ref nodes** — pure rollout.
2. **Fleet↔job dependency is not robust.** `qwen36_grpo_preflight.py:34` checks `workers_alive` — a
   **liveness** check doing duty as a **sufficiency** check. Cheap fix: assert fleet `TIME_LEFT >
   walltime + margin`. Durable fix: validate and enable the **auto-chain** (`MAX_CHAIN`,
   `dependency=afterany:` in `jureca_workers.sbatch`) — it exists but has **never crossed a real
   walltime boundary**.
3. **Startup is now ~95% vision-tower profiling** (~920s, *"1 image items of the maximum feature
   size"*) for a tower SkyRL unwraps and no rollout uses. `MM_LIMIT=0` staged in `8ac43278`,
   defaults unchanged, **UNMEASURED**. Bigger than the 11 min the fscratch fix already saved.
4. **Rollouts are prefill-dominated** — 150,223 in vs 3,377 out per trial (44:1). `max_num_seqs: 8`
   against measured KV headroom 17× (32k) to ~36× (real 15k). Prefix caching is ON but vLLM warns it is
   **experimental for Mamba `align` mode**; effectiveness unmeasured, and `n_cache_tokens: 0` is not
   evidence either way. Settle it by reading vLLM `/metrics` during rollouts.
5. **Operator decision — DAPO `dynamic_sampling: filter`** targets the blocker as a flag but needs
   `colocate_all: true`, colliding with disaggregated-only.
6. **The 110-minute** shards→policy-init on `1221005` — not reproduced on `1229343` (~35 min).
   Plausibly GPFS contention while `/e/scratch` hit the inode wall. Unexplained.
7. **The 08-03 dead-tree cleanup is still UNAPPROVED.** `/p/scratch` routes around `/e/scratch` rather
   than fixing it.
8. **The optimizer update, checkpoint and HF export have STILL never executed.** The milestone.

## Traps worth not repeating

- **A config key in the materialized TrialConfig proves NOTHING about whether the consumer reads it.**
  Four keys have now failed this way (`strict_json_parser`, `compaction.reserved`,
  `store_all_messages`, `override_timeout_sec`). Verify by BEHAVIOUR.
- **Checking a thing EXISTS is not checking it WORKS for the duration.** Same error twice: the route
  gate (endpoint reachable ≠ reachable from a compute node) and `workers_alive` (fleet alive now ≠
  alive at the end).
- **`MODEL_PATH` overrides the YAML** — it reaches the launcher as `--model_path`. Editing the YAML's
  model path alone is silently reverted.
- **Never infer filesystem bandwidth from `cp`** — page cache + readahead flattered `/e/scratch` 30×
  versus `O_DIRECT`. Measure from a **compute** node at the concurrency the real consumer uses.
- **`pkill -f <pattern>` over SSH matches your own command line** and kills the shell mid-command.
- **One long-lived background `ssh` locks the whole Jupiter connection** (~one session channel), and
  each locked-out attempt burns a TOTP try. `nohup` remotely and poll with short calls.
- **`mask` = infrastructure ⇒ ABORTS the whole batch.** Only `passthrough` survives.
- **Never `find`/`du` on GPFS.** Use `jutil project dataquota -p reformo` — but note its counters can be
  **11 days stale** and read 77% while the filesystem is actually full.
