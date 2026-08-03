# ⚠ TEMPORARY TAKEOVER NOTE — rewritten 2026-08-04 ~01:00 KST · DELETE ONCE ABSORBED

Supersedes the 2026-08-03 version, whose central claim ("the DATA is the blocker") was **wrong**.
Read `handoff.md` § "Live state — 2026-08-04" first, then `gotchas.md` § 2026-08-04 and
`decisions.md` § 2026-08-04.

## The one thing to understand

**The blocker was a 600-second agent timeout, not the data.**

`agent.override_timeout_sec = 1800` was materialized but unreachable. OpenCode issues its
long-running `run` through `exec_as_agent()` with no `timeout_sec`, so it inherited
`ApptainerEnvironment.exec()`'s hardcoded `timeout_sec or 600`. The trial layer's
`asyncio.wait_for(agent.run(), 1800)` is an OUTER guard that can only fire if it is *shorter*, so the
worker always won. On `1221005`: **19 of 25 trials ended at exactly 600–601s**, and **19 of 19**
exceptions carry the literal `Command timed out after 600s` / `return_code -1`. 76% of trials were
cut off mid-task, which flattens the within-group variance GRPO consumes.

**Fourth instance of the signature bug** — a key accepted, deep-merged, written to disk, and ignored
by its consumer (`strict_json_parser`, `compaction.reserved`, `store_all_messages`, now this).

## Two conclusions that were carried for a week and are now RETIRED

1. **"Raw r2egym collapses."** It does not. The co-lead's RAW arm reaches the **same ~45.5 p@1** on
   SWE-bench as her filtered band, in 120 steps instead of 60. The band is `p@4` (not `p@8`), yields
   **~1.6k of 4.5k (~36%)**, is built for an **8B**, and gives **no performance boost** — convergence
   speed and ~20% compute only (60k → 48k rollouts). ⇒ For a *pipeline-validation* goal, a band's only
   value is guaranteeing a nonzero gradient exists. We would need enough in-band tasks for ~16 groups
   for ONE step, not her 1.6k-task set.
2. **"Qwen3.6 is essentially deterministic per r2egym task."** Trajectories within a uniform group
   differ wildly — distinct hashes, 26–60 messages, 1.7k–6.1k output tokens. Sampling works fine at
   temp 1.0. Only the *outcomes* were invariant, and under a truncated budget.

## Shipped, pushed, and deployed this session

| commit | repo | change |
|---|---|---|
| `f44a1170` | harbor | `BRIDGE_EXEC_TIMEOUT` knob; default still 600; 4 tests, Ruff clean |
| `6b76270b` | OT-Agent | vLLM canary reads shards from a chosen filesystem |
| `2b6defd1` | OT-Agent | exec timeout 2100, n=8, fscratch model, `-next` checkout, 6h wall |
| `05518eed` | ai_memory | the correction set (**local only, not pushed**) |

Deployed on Jupiter by fetch + hard reset: harbor `f44a1170`, OT-Agent `-next` `6b76270b`
(⚠ `2b6defd1` deploy was INTERRUPTED — see Next actions).

## Measured this session

- **`Loading weights` 701.12s → 38.03s (18.4×)** by staging the checkpoint on `/e/fscratch`
  (`1217900` vs `1224804`, same 1-GPU canary).
- `O_DIRECT`, compute `jpbo-018-14`, job `1224678`: `/e/scratch` **0.056 GiB/s per stream**,
  `/e/fscratch` **2.51**. vLLM reads the 26 shards **sequentially in one stream**, so it sits exactly
  on that per-stream floor — which is why aggregate bandwidth never helped.
- ⚠ **Never infer filesystem bandwidth from `cp`.** A 4-stream `cp` gave 1.67 GB/s off `/e/scratch`
  (page cache + readahead) and led me to wrongly report the 22× claim refuted. `O_DIRECT` from a
  compute node, at the concurrency the real consumer uses, is the only valid measurement.

## Live resources

- **Jupiter: 0 nodes.** `1224678` and `1224804` COMPLETED and were freed.
- **JURECA workers `15494122`**, 2 nodes × 16, ~22h left from 00:30 KST. Predecessor `15489111`
  expired on schedule; continuity held.
- **Bridge** `10.128.1.2:9920` pristine, `workers_alive: true`.
- **ControlMaster** pid 185890 alive 28h+. Reach JURECA ONLY as
  `ssh -S ~/.ssh/cm_jureca/qwen36 jureca.fz-juelich.de` — there is no `~/.ssh/config` on Jupiter, so a
  bare `ssh jureca` fails on hostname resolution and **looks like a dead tunnel when it isn't**.
- ⚠ `Session open refused by peer` on the Mac→Jupiter master is **MaxSessions**, not auth expiry.
  Do NOT retry in a loop — the fallback prompts for TOTP and repeated failures risk an account lock.
  Let long-running background SSH sessions finish first.

## Next actions, in order

1. **Finish deploying `2b6defd1`** to `/e/scratch/reformo/lee27/OpenThoughts-Agent-r2egym-bridge-next`
   (`git fetch fork lukedhlee/qwen3-6-r2egym-grpo && git reset --hard 2b6defd1`). The deploy was
   interrupted by MaxSessions exhaustion, so the tree may still be at `6b76270b`. **Verify before
   launching** — without it the canary runs at n=4 with the 600s cap and reproduces the old result.
2. **Read `/e/scratch/reformo/lee27/bench/timeout_proof.py` output** (job launched ~00:40 KST). It
   isolates ONLY the changed path — one Apptainer env, `sleep 700` with `timeout_sec=1500` — and PASSES
   iff `return_code 0` after >600s. If it FAILS, the fix did not take and step 3 must not run.
3. **Allocation-free preflight**, then launch the 5-node canary:
   `DCFT=/e/scratch/reformo/lee27/OpenThoughts-Agent-r2egym-bridge-next PREFLIGHT_ONLY=1 \
    REQUIRE_CLEAN_BRIDGE=1 bash hpc/skyrl_standard/jupiter/run_r2egym_qwen3_6_35b_grpo_canary.sh`
   Expect `OpenCode limit=15360+4096, proactive_compaction_at=11264`.
4. **The number that decides everything:** group `trace_jobs/*/result.json` by task prefix and check
   `len(set(rewards)) > 1` per group. Reward lives at **`verifier_result.rewards.reward`**, NOT at
   top-level `reward` (that key does not exist; reading it returns `None` for every trial).
   Tooling already on Jupiter: `/e/scratch/reformo/lee27/bench/tput2.py <trace_jobs>`.
   **Never** read a healthy `avg_raw_reward` as evidence GRPO can learn.

## Still open

- **The 110-minute** shards-loaded→policy-init on `1221005` vs ~28 min on `1219434`. After the bytes
  are in memory; fscratch does not touch it. Unexplained.
- **Operator decision:** DAPO `dynamic_sampling: filter` drops all-same-reward groups and resamples —
  targets the blocker as a flag — but needs `colocate_all: true`, colliding with the settled
  disaggregated-only decision. → `decisions.md` 2026-08-04.
- **Token fidelity** (`literal.jsonl`) still never enabled; required before trusting TIS ratios.
- **The optimizer update, checkpoint, and HF export have STILL never executed** with this model. That
  remains the milestone.
- The **6-node** production YAML and `run_qwen36_live_canary.sh` may still carry the stale
  `0f04b250` checkout path; only the canary path was collapsed onto `-next`.

## Traps worth not repeating

- **A config key in the materialized TrialConfig proves NOTHING about whether the consumer reads it.**
  Verify by BEHAVIOUR. Four keys have now failed this way.
- **`mask` = infrastructure ⇒ ABORTS the whole batch** under `fail_on_infrastructure_error`. Only
  `passthrough` survives. Adding an exception to `mask_exceptions` makes it MORE fatal.
- **`MODEL_PATH` overrides the YAML** — it reaches the launcher as `--model_path`. Editing the YAML's
  model path alone is silently reverted.
- **Never grep a trial tree for success markers** — the rendered config contains the instruction text,
  so markers match before the agent has done anything.
- **Harbor is a COPIED install**; the fix only lands because the sbatch puts
  `harbor-apptainer-bridge/src` first on `PYTHONPATH`. Verify with the job's REAL `PYTHONPATH`.
- **Never `find`/`du` on GPFS.** Use `jutil project dataquota -p reformo`.
- Piping a remote script through `tail` buffers its output until EOF; `flush=True` buys nothing.
