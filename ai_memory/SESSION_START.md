# Fresh-session kickoff prompt — paste the block below

Volatile numbers go stale fast. Re-probe before trusting them; the durable state is in `handoff.md`.
Last refreshed **2026-08-04 10:55 KST**.

---

```
Read ai_memory/handoff.md § "Live state — 2026-08-04 10:30 KST" first, then
gotchas.md and decisions.md § 2026-08-04. NEXT_SESSION.md is the takeover note.
Report times in KST.

We're doing agentic RL (GRPO) on Qwen/Qwen3.6-35B-A3B over r2egym: Jupiter for
training + vLLM, JURECA for Apptainer sandboxes via a bridge on a Jupiter login
node. THE MILESTONE IS ONE FINITE OPTIMIZER UPDATE — it has never executed with
this model. Pipeline validation, not model quality.

Start by establishing reality, don't trust the numbers below:
  squeue on both clusters (JURECA only via
    ssh -S ~/.ssh/cm_jureca/qwen36 jureca.fz-juelich.de — a bare `ssh jureca`
    fails on hostname resolution and looks like a dead tunnel when it isn't)
  curl http://10.128.1.2:9920/status
  tail /e/fscratch/reformo/lee27/retarget_f.log

As of 10:55 KST: Jupiter 1229488 RUNNING (5 nodes, 3h47m left, 16x8 = 128
trajectories, agent budget 1800s); JURECA fleets 15495516 (23h) + 15494122
(12h); bridge clean, workers alive; a watcher has already retargeted the
reverse tunnel to head 10.128.25.20 and is waiting on vLLM to answer.

FOUR THINGS NOT TO GET WRONG
1. The learnable band IS REQUIRED. The 600s agent-timeout defect was real and
   is fixed, but fixing it did NOT produce reward variance: 1229343 gave 18
   honest rewards, 0 of 6 groups varied, and five tasks scored identically to
   the capped run. Next action is a ~384-task p@4 probe (~1.7h at 4 rollout +
   8 JURECA nodes). Earlier notes saying "hold the band" are reversed.
2. Filesystems — measured FROM A COMPUTE NODE, the login node lies:
     /e/fscratch  visible + WRITABLE  <- everything the job touches lives here
     /e/scratch   inode cap SHARED by 26 users, fluctuates, counter 11 days
                  stale; killed two runs mid-rollout. Reads fine.
     /p/scratch   NOT MOUNTED on Jupiter compute (login-node staging only) —
                  BUT sif_cache must stay on it, JURECA workers do mount it.
3. Verify by BEHAVIOUR, never by config. Five keys have now been accepted,
   written, and ignored by their consumer. And checking a thing EXISTS is not
   checking it WORKS where and when it's needed — that mistake has recurred
   three times (route gate, workers_alive, /p/scratch).
4. Reward is at verifier_result.rewards.reward — NOT top-level `reward`, which
   does not exist and returns None for every trial. The number that matters is
   per-group len(set(rewards)) > 1, never avg_raw_reward.

BEFORE ANY LAUNCH
  DCFT=/e/fscratch/reformo/lee27/OpenThoughts-Agent-r2egym-bridge-next
  TIME_LIMIT=04:00:00 or less — a 6h job is deferred a full day by a hidden
    maintenance window. Probe free with: sbatch --test-only -t <T> <sbatch>
  Confirm a JURECA fleet has more time left than walltime + startup. Preflight
    does NOT check this (qwen36_grpo_preflight.py:34 checks workers_alive, a
    liveness check doing duty as a sufficiency check).
  After allocation the reverse listener MUST be retargeted at the new head and
    proven with /health AND a real /v1/chat/completions FROM A COMPUTE NODE.
    /e/fscratch/reformo/lee27/retarget.sh automates it; omitting it wasted a
    5-node 3-hour allocation.

OPERATOR-ONLY — surface, don't act:
  (a) DAPO dynamic_sampling: filter — targets the blocker as a flag but needs
      colocate_all: true, colliding with the settled disaggregated-only call.
  (b) Any destructive cleanup (the 08-03 /e/scratch dead-tree deletion is still
      unapproved).
  Cancelling our OWN wedged jobs is pre-authorised. Never anyone else's.

Autonomy: on bring-up failures, keep root-causing and relaunching. Report when
blocked on something only Luke can do (the Jupiter->JURECA ControlMaster
carries the reverse tunnel; restoring it needs one interactive TOTP) or when a
milestone lands. Push to lukedhlee forks freely; PRs need an explicit ask.
```

---

## Keeping this current

Refresh only the "As of …" paragraph and the four job/fleet ids. Everything else is stable guidance
that belongs here rather than in a paste — if a *rule* changes, change `handoff.md` and let this file
keep pointing at it.
