# Fresh-session kickoff prompt — paste the block below

Volatile numbers go stale fast. Re-probe before trusting them; the durable state is in `handoff.md`.
Last refreshed **2026-08-05 23:00 KST**.

⚠ **This file is the bootstrap, so a stale claim here costs more than anywhere else.** On 08-05 it sat for
three hours still asserting "A LEARNABLE BAND EXISTS" and the "~6x concurrency bump" *after* both had been
retracted in `NEXT_SESSION.md` — i.e. it was seeding every new session with the project's two most expensive
wrong beliefs. **When you retract something, grep for it across all of `ai_memory/` and fix this file in the
same commit.** Corrections are not done until they are propagated.

---

```
Read ai_memory/NEXT_SESSION.md FIRST — it is the takeover note and it contains the
tunnel runbook and the launch recipe. Then handoff.md § "2026-08-05 (late)" and
gotchas.md. Report times in KST.

Agentic RL (GRPO) on Qwen/Qwen3.6-35B-A3B over r2egym: Jupiter for training +
vLLM, JURECA for Apptainer sandboxes via reverse forwards from a Jupiter login
node.

THE TRANSPORT NOW WORKS. For the whole project until 08-05 no rollout had ever
produced a multi-step trajectory; the cause was TWO dead reverse tunnels, not the
model or the reward. With both up: median_steps 18.5 (was 1), 0 bridge timeouts
(was 32/32). The milestone machinery is also banked (1243377: update + 235GB ckpt
+ 65GB export, preserved as MILESTONE_1243377_*), though with grad_norm=0 because
rollouts were still timing out then.

RETRACTED — do not re-derive: "33% pass matches Marianna's ~35%, so A LEARNABLE
BAND EXISTS." That compared GROUPS against TRIALS. Re-measurement of the same
35B data found a 31.6% pass rate and **0 of 16 groups in band**. A pass rate is
NOT a band. The band is per-group: 0 < passes < n_samples. Compute it only with
band_report.py / band.py, never by eye.

THE GOAL NOW (Luke's call, 08-05 evening): do NOT chase a fast band number.
Build scalable, trustworthy infra by REPRODUCING Marianna's band measurement.
HER NUMBERS, from her own words 08-05 late: band = ~1.6k of 4.5k r2egym tasks
=> ~36%, at 0 < pass@4 < 1, and the filtering pass cost "18k rollouts"
(= 4.5k x 4, so pass@4 over a 4.5k pool is confirmed). She used the agent
terminus-structured, NOT OpenCode.
  RETRACTED: "358 of 4,578 ~= 8%". Never verified, source unknown, void. It
  flipped a verdict once already: our measured 0-in-band reads as "consistent
  with 8%" (p~0.47) but is ~1-in-800,000 against 36%. If you see 358 or 8%
  anywhere, it is stale.
Model: the 8B g1_diverse_tezos_100k_8b she used. Agent: OpenCode (Luke's call —
its transport is the one that works; her band % came from terminus-structured, so
our absolute % may legitimately differ. The pass/fail criterion is therefore
"is there real within-group variance at all", not "does it equal 36%").
Stay inference-only: it removes training-side concerns and debugs ~10x faster.

STOP-THE-LINE, 08-05 23:40 KST: THE REWARD IS A PER-TASK CONSTANT. On 1248713
(OpenCode, 8B, r2egym-patched-full-oracle) 62.6% of trials scored 1.0 while
0 of 30 fully-sampled groups showed ANY within-group variance. At a 62.6% pass
rate ~83% of 4-sample groups should be in band, so 0/30 has probability
~0.17^30 ~ 1e-23. The reward is not measuring the model. Corroborating: a trial
scored 1.0 with ZERO agent steps, and trials that never edited a file passed MORE
often (76% of passes made no edit) than trials that did. Prime suspect: the
"-patched-" in r2egym-patched-full-oracle means the fix ships pre-applied, so the
tests already pass and editing only breaks them. DO NOT tune throughput, scale
nodes, or believe any band number until this is resolved.

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

SEVEN THINGS NOT TO GET WRONG
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
3. Verify by BEHAVIOUR, never by config. SEVEN keys have now been accepted,
   written, and ignored by their consumer. And checking a thing EXISTS is not
   checking it WORKS where and when it's needed — that mistake has now recurred
   four times (route gate, workers_alive, /p/scratch, listener-vs-route).
   Anchor log patterns too: a `grad_norm=1.0` "success" was really the substring
   inside max_grad_norm=1.0 from the config echo.
4. Reward is at verifier_result.rewards.reward — NOT top-level `reward`, which
   does not exist and returns None for every trial. The number that matters is
   per-group len(set(rewards)) > 1, never avg_raw_reward.
5. RETRACTED: "concurrency is the bottleneck => a ~6x bump, no new nodes." We
   hit the configured cap instantly at every setting tried (32 -> 64 -> 128 all
   went live), so the cap was never the binding constraint. What actually made
   trials slow: max_turns: 999999 (Marianna caps 50) plus heavy sandboxes
   (2 CPU/4GB vs her 1 CPU/1GB). A timed-out trial holds a slot ~62 min, so
   failures cost ~7x a success — that part stands.
6. TP=1 ONLY. inference_engine_tensor_parallel_size > 1 makes vLLM build a
   distributed executor that copies the parent actor's os.environ into a nested
   Ray runtime_env; Ray then asserts on its own
   __RAY_WORKER_PROCESS_SETUP_HOOK_ENV_VAR and NO engine ever starts. SkyRL
   mislabels this as "port collision (EADDRINUSE)" and burns 5x120s retries on a
   deterministic assertion, so gate on
   `grep -c RAY_WORKER_PROCESS_SETUP_HOOK <joblog>`, not the EADDRINUSE string.
   Cost: 8-node job 1247578, whole allocation. TP=1 is also simply better for an
   8B on 96GB GH200 — 28 engines on 8 nodes instead of 7, no collectives.
7. Route-gate at max_tokens=4096, the REAL per-turn budget. At 256 the 8B loops
   <think> blocks and returns finish_reason=length with zero tool_calls, which
   reads exactly like a dead parser. At 4096 it emits a clean tool call in ~79
   tokens. A too-small gate manufactures the failure it is testing for.

BEFORE ANY LAUNCH — this exact recipe; three jobs died one per line
  W=/e/fscratch/reformo/lee27/OpenThoughts-Agent-r2egym-bridge-next   # the REAL WORKDIR
  source /e/scratch/reformo/lee27/keys/secrets.env    # NOT ~/secrets.env (ops.md is wrong)
  cd $W && source hpc/dotenv/jupiter.env && export DCFT=$W
  sbatch --chdir=$W --time=<T> --export=ALL,DCFT=$W,SKYRL_CHUNKED_LOGPROBS=1 <generated>_rl.sbatch
  TIME_LIMIT 04:00:00 or less — a 6h job can be deferred a day by a hidden
    maintenance window. Probe free: sbatch --test-only -t <T> <sbatch>
  RUN THE GATE FIRST — it now enforces most of the rules above, so you do not
    have to remember them:
      bash hpc/skyrl_standard/jupiter/band_preflight.sh <config.yaml> <walltime> <fleet_jobid>
    Non-zero exit = DO NOT SUBMIT. It checks TP=1, derived context fields, dead
    keys, n_samples>=2, sane max_turns, the ControlMaster, BOTH forwards, that the
    fleet outlives walltime+30min, and bridge capacity.
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
