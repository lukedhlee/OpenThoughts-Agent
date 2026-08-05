# Fresh-session kickoff prompt — paste the block below

Volatile numbers go stale fast. Re-probe before trusting them; the durable state is in `handoff.md`.
Last refreshed **2026-08-05 (late)**.

---

```
Read ai_memory/NEXT_SESSION.md FIRST — it is the takeover note and it contains the
tunnel runbook and the launch recipe. Then handoff.md § "2026-08-05 (late)" and
gotchas.md. Report times in KST.

Agentic RL (GRPO) on Qwen/Qwen3.6-35B-A3B over r2egym: Jupiter for training +
vLLM, JURECA for Apptainer sandboxes via reverse forwards from a Jupiter login
node.

THE PIPELINE NOW WORKS. For the whole project until 08-05 no rollout had ever
produced a multi-step trajectory; the cause was TWO dead reverse tunnels, not the
model or the reward. With both up: median_steps 18.5 (was 1), 0 bridge timeouts
(was 32/32), rewards 8x0.0 + 4x1.0 = 33% pass, which matches Marianna's ~35%. So
A LEARNABLE BAND EXISTS. The milestone machinery is also banked (1243377: update
+ 235GB ckpt + 65GB export, preserved as MILESTONE_1243377_*), though with
grad_norm=0 because rollouts were still timing out then.

THE GOAL NOW (Luke's call): do NOT chase the full band. Get the learnable band
RELIABLY ON A SUBSET, on the 8B g1_diverse_tezos_100k_8b that Marianna used, then
EXTRAPOLATE to answer whether the system scales to a full band in 7-10h. Stay
inference-only for now: it removes training-side concerns and debugs ~10x faster
(~9 min/trial vs ~1.5h/training-step).

FIRST, non-negotiable: the 8B emits BARE-JSON tool calls (182/182 steps, zero
XML) while vLLM ran --tool-call-parser qwen3_coder (XML-only), so it never
matched and never logged -> 1-step no-ops -> zero variance. Wire the fix before
measuring anything, or you re-derive the retracted 0/197:
  OT-Agent 4add607c  rl/tool_parsers/bare_json_tool_parser.py (validated 182/182)
  MarinSkyRL d8bdc79 pop_openai_kwargs forwards tool_parser_plugin
Neither is synced to the cluster. Sync ONLY when no job is running.

Start by establishing reality, don't trust numbers:
  ssh jupiter "squeue --me -o '%.10i %.9T %.6M %.6L %.4D %N'"
  ssh jupiter "ssh -S ~/.ssh/cm_jureca/qwen36 jureca05.fz-juelich.de 'squeue --me'"
  ssh jupiter "curl -s -m 8 http://10.128.1.2:9920/status"   # workers_alive, active_jobs, envs.ready
  ssh jupiter "python3 /e/fscratch/reformo/lee27/gate.py"     # median_steps + rewards, last 16 min

FIVE THINGS NOT TO GET WRONG
1. THE CONTROLMASTER CARRIES TWO FORWARDS. Restoring only one gives a PASSING
   route gate and 100% rollout timeouts.
     -R 10.14.0.46:18000 -> <jupiter-head>:8000   sandboxes -> vLLM
     -R 10.14.0.46:9923  -> 10.128.1.2:9920       workers   -> bridge
   The bridge is a python3 process on the JUPITER login node; workers address it
   as jrlogin05i:9923. That port pair lives ONLY in the fleet log ("Bridge URL").
   workers_alive:false WITH live worker procs = missing bridge forward, not dead
   workers (pgrep -fc worker.py). Rebuilding the master needs `-4` (IPv6 fails the
   key's from= clause) and MUST pin jureca05.fz-juelich.de (only jrlogin05 owns
   10.14.0.46). Capture a dying master's forwards BEFORE killing it.
2. Filesystems — measured FROM A COMPUTE NODE, the login node lies:
     /e/fscratch  visible + WRITABLE  <- everything the job touches lives here
     /e/scratch   inode cap SHARED by 26 users, fluctuates, counter 11 days
                  stale; killed two runs mid-rollout. Reads fine.
     /p/scratch   NOT MOUNTED on Jupiter compute (login-node staging only) —
                  BUT sif_cache must stay on it, JURECA workers do mount it.
3. Verify by BEHAVIOUR, never by config. Six keys have now been accepted,
   written, and ignored by their consumer. And checking a thing EXISTS is not
   checking it WORKS where and when it's needed — that mistake has now recurred
   four times (route gate, workers_alive, /p/scratch, listener-vs-route).
   Anchor log patterns too: a `grad_norm=1.0` "success" was really the substring
   inside max_grad_norm=1.0 from the config echo.
4. Reward is at verifier_result.rewards.reward — NOT top-level `reward`, which
   does not exist and returns None for every trial. The number that matters is
   per-group len(set(rewards)) > 1, never avg_raw_reward.
5. Concurrency is the throughput bottleneck, not nodes. Measured: real trials
   take ~8.9 min (median), peak achieved concurrency is 32 = exactly the config
   cap, and 110 sandboxes sat READY while only 32 ran. Band-in-10h needs ~200
   concurrent against 512 fleet slots => a ~6x bump, no new nodes. A timed-out
   trial holds a slot ~62 min, so failures cost 7x a success.

BEFORE ANY LAUNCH — this exact recipe; three jobs died one per line
  W=/e/fscratch/reformo/lee27/OpenThoughts-Agent-r2egym-bridge-next   # the REAL WORKDIR
  source /e/scratch/reformo/lee27/keys/secrets.env    # NOT ~/secrets.env (ops.md is wrong)
  cd $W && source hpc/dotenv/jupiter.env && export DCFT=$W
  sbatch --chdir=$W --time=<T> --export=ALL,DCFT=$W,SKYRL_CHUNKED_LOGPROBS=1 <generated>_rl.sbatch
  TIME_LIMIT 04:00:00 or less — a 6h job can be deferred a day by a hidden
    maintenance window. Probe free: sbatch --test-only -t <T> <sbatch>
  Confirm the JURECA fleet has MORE time left than walltime + startup. Preflight
    does NOT check this (it checks workers_alive, a liveness check doing duty as
    a sufficiency check).
  After allocation: retarget BOTH forwards at the new head, then prove the route
    with a real /v1/chat/completions FROM A COMPUTE NODE. A listener check
    (ss | grep 18000) passes while the route is dead — that is exactly how
    1243377 lost all 64 rollouts.
  NEVER run Jupiter ssh calls in parallel — the ControlMaster refuses sessions and
    you get a spurious TOTP prompt. Serialize.

KEEP ENABLED: SKYRL_CHUNKED_LOGPROBS=1 (MarinSkyRL 637a764) — fixes the large-vocab
backward OOM. Validated: 33.41 GiB peak vs stock OOM at the identical 30.57 GiB,
and 2.4x MORE accurate than the stock path vs fp64. Confirm by the log line
"chunked gathered log-softmax ACTIVE", which only fires during a training step.

OPERATOR-ONLY — surface, don't act:
  (a) DAPO dynamic_sampling: filter — targets the blocker as a flag but needs
      colocate_all: true, colliding with the settled disaggregated-only call.
  (b) Any destructive cleanup.
  (c) HF Hub push: HF_HUB_CACHE points at feuer1's dir (Permission denied) and
      there is no HF_TOKEN in env; the 65GB export is local-only.
  Cancelling our OWN wedged jobs is pre-authorised. Never anyone else's.

Autonomy: on bring-up failures, keep root-causing and relaunching. Report when
blocked on something only Luke can do (a JURECA ControlMaster restart needs one
interactive TOTP — but see #1, the TOTP was never the hard part) or when a
milestone lands. Push to lukedhlee forks freely; PRs need an explicit ask.
```

---

## Keeping this current

Refresh the "Last refreshed" line and any job/fleet ids. Everything else is stable guidance that belongs
here rather than in a paste — if a *rule* changes, change `handoff.md`/`NEXT_SESSION.md` and let this file
keep pointing at it.

⚠ **Read `NEXT_SESSION.md` first.** On 08-05 two jobs were burned rediscovering the `DCFT` value that was
already written on line 51 of this file — because the takeover note pointed at `handoff.md` and
`gotchas.md` but not here. The launch recipe now lives in both.
