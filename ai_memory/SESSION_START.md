# Fresh-session kickoff prompt — paste the block below

> ## ⛔ 2026-08-30 — THE CURRENT TASK IS THE REPO REFACTOR. Read `notes/repo_refactor_plan.md` first.
> Everything below this banner is the r2egym / learnable-band / currease era and is **frozen context**, not
> the next action. Scope cuts already decided: no Levanter SFT, no Megatron, no OpenCode, nothing merged
> from Marianna's forks. Inventory: the OTA Code Atlas
> (https://claude.ai/code/artifact/e633ab79-c602-4ca1-a5d7-5b52ce4e3aa2). Clock: fscratch purge of the
> base30b/currease checkpoints + `rl-fa` venv starts ~09-10 — decide keep/drop before starting.


Volatile numbers go stale fast. Re-probe before trusting them; the durable state is in `NEXT_SESSION.md`.
Last refreshed **2026-08-06 04:00 KST**.

⚠ **This file is the bootstrap, so a stale claim here costs more than anywhere else.** On 08-05 it sat for
three hours still asserting "A LEARNABLE BAND EXISTS" and the "~6x concurrency bump" *after* both had been
retracted in `NEXT_SESSION.md` — i.e. it was seeding every new session with the project's two most expensive
wrong beliefs. **When you retract something, grep for it across all of `ai_memory/` and fix this file in the
same commit.** Corrections are not done until they are propagated.

---

```
Read ai_memory/NEXT_SESSION.md FIRST -- sections 0 and 1 before touching anything.
Then gotchas.md. Report times in KST (cluster clocks are CEST = KST - 7h).

CONTEXT. Agentic RL (GRPO) over r2egym, INFERENCE-ONLY: Jupiter serves vLLM, JURECA
runs Apptainer sandboxes reached via reverse forwards on a ControlMaster that lives
ON the Jupiter login node (not the laptop). Model = the 8B g1_diverse_tezos_100k_8b
(Qwen3-family, Marianna's checkpoint). Agent = OpenCode. 35B-A3B is CONCLUDED.
GOAL = reproduce Marianna's learnable band (~1.6k of 4.5k tasks at 0 < pass@4 < 1,
~36%) as end-to-end proof the infra is trustworthy.

*** ENVIRONMENTS ARE VALIDATED MODEL-FREE, 2026-08-06 03:50 KST. Do not re-litigate. ***
  r2egym    : 25/32 tasks give pristine 0.0 AND oracle 1.0. Harness clean
              (64/64 records, 0 timeouts, 0 nulls). CEILING 78.1%.
              The 7 failures are BROKEN TASKS, diagnosed: aiohttp dies on
              `asyncio.async(` -> SyntaxError (async reserved since Py3.7);
              pandas dies on pytest `unrecognized arguments: --strict-data-files`.
              Unsolvable by any model => real dead weight, not a harness artefact.
  SWE-bench : 8/8 pilot tasks give nop 0.0 AND oracle 1.0 -- all 8 largest repos,
              = 479/500 = 96% of the benchmark, incl. django (231/500 = 46%).
              500 task dirs staged; full pull-only SIF build running (~625GB, ~6h).
  Getting here meant fixing FOUR bugs in MY OWN GATE, each of which faked
  "the environment is broken": missing agent_tools bind + offline-pip patch;
  missing --no-home; missing --cleanenv on start AND every exec (runs went from
  hanging past 25min to ~2min); no PATH / wrong cwd / wrong exec form.
  RULE: a gate that diverges from worker.py measures the gate. Diff argv line by line.

*** THE IN-JOB GATE PASSED 2026-08-06 02:30 KST -- also settled. ***
Yesterday: "the r2egym reward is a per-task CONSTANT and never measures the model."
True then, FIXED AND VERIFIED now. A group finally holds BOTH values:
  task r2egym-2514 (task_name confirmed in BOTH result.json files), 2 of 4 samples
    dCmGsKH  reward 1.0  "3 passed"           4x edit tool calls + read/ls/bash
    uGPuwvX  reward 0.0  "2 failed, 1 passed" bash + glob only -- NO edits
The sample that EDITED turned 2 failing tests into passing; the one that didn't,
didn't. That is the INVERSE of the old pathology (passing trials used to edit LESS:
24% vs 49%). THE ENVIRONMENT MEASURES THE MODEL. Go get a band number.
BE PRECISE OR YOU REPEAT THE PASS-RATE-VS-BAND ERROR: within-group variance is
PROVEN, but the band PERCENTAGE is NOT yet measured -- band.py counts only
FULLY-SAMPLED groups (>=4 trials) and r2egym-2514 had 2 of 4 in, so it still prints
"0 fully sampled". Different claims; do not upgrade one into the other.
Rest of the ladder passes: non-null reward 7/7, steps median 3 max 8, tool calls
median 2 max 11 with ZERO zero-tool trajectories, duration median 3.4 min.
THROUGHPUT is the next real constraint: 0.29 trials/min, peak concurrency 5 =>
the full 13,312-trial band projects to ~759 h. Scaling is now sanctioned.
What remains is a RATE problem, not a correctness one: only 1 of 7 trials made any
edit. The two causes below are throughput/quality levers, NOT blockers.

ENVIRONMENT = VALIDATED, model-independently (do not redo):
  /testbed populated  -- real repo inside her SIF, no clone, no network
  HEAD == base_commit^ -- exactly the buggy parent (inverse of the old bug)
  editable install     -- Orange.__file__ -> /testbed via Orange3.egg-link, so agent
                          edits DO reach the tests, even for the "hard repos"
  /testbed writable    -- WRITE_OK on fleet node, fakeroot unavailable and NOT needed
  chardet safe offline -- "Audited 1 package", exit 0. NEXT_SESSION 4.1 RETIRED
REWARD IS NOW REAL: across trials on different repos each verifier ran a genuine
pytest with EXACTLY ONE failing test = the bug that task names, rest passing, so a
0.0 is correct rather than free -- and fixing it flips the reward to 1.0 (see above).

THE RATE PROBLEM (measured with agentstall.py over 7 trials: 7/7 ripgrep errors,
5/7 unparsed JSON, 1/7 made an edit -- and that one scored 1.0):
  ripgrep error 7/7 (100%) -- rg is NOT in the sandbox, so OpenCode's glob AND grep
    fail on the agent's first action. harbor worker.py ~line 91 has a HARDCODED tool
    dict {opencode,tmux,asciinema,uv}. Fix = stage a static musl rg into
    $BRIDGE_AGENT_TOOLS/bin/rg AND add "rg" to that dict. Needs a FLEET RESTART.
  thinking is ON -- config.json shows interleaved_thinking=True AND
    enable_thinking=True. NB: BAND_HARBOR_THINK is INERT for OpenCode. The parser is NOT broken; real
    tool_use events appear. The model emits <think> prose plus MALFORMED JSON
    (unterminated string, two concatenated objects, stray </think>), so vLLM cannot
    parse tool_calls, OpenCode stores it as TEXT and finishes reason=stop after ~2
    steps. 5 of 7 trials. Only 1 of 7 made any edit at all.
  => CHEAP TEST FIRST: BAND_SERVER_NO_THINK=1 (NOT BAND_HARBOR_THINK, which is a
     var, no fleet restart. ("thinking-off (DONE)" in the task list was WRONG.)

DO THIS FIRST (operator's call): VALIDATE ENVIRONMENTS WITH NO MODEL.
The oracle is free here -- /testbed is at base_commit^, so
`git checkout <base_commit> -- .` IS the gold patch (no solve.sh needed; that's the
patched dataset).
  bash /p/scratch/synthlaion/lee27/envgate.sh <task> pristine   # expect 0.0
  bash /p/scratch/synthlaion/lee27/envgate.sh <task> oracle     # expect 1.0
A pristine 0.0 AND an oracle 1.0 on the same task IS THE GATE, no GPU involved.
32 tasks are staged at /p/scratch/synthlaion/lee27/envgate/<task>/. A sweep was
left running in tmux 'envgate' on JURECA -> envgate_results.jsonl (RE-CHECK IT;
an earlier attempt died to a 12-min srun limit -- Orange3 setup is slow, use 40min).
/e/fscratch is NOT visible from JURECA, hence the staging step.

LIVE STATE (re-probe, do not trust):
  1251403   8B x 32 tasks x pass@4, 2 nodes TP=1. Ends ~02:56 KST. 7 results,
            {0.0: 6, 1.0: 1}, GATE PASSED on r2egym-2514. It LOST ITS FIRST 64
            ROLLOUTS to a dead forward (below), so it only had ~35 min of real run.
  15500584  JURECA fleet, 32 nodes, ~21h left, SIF_CACHE correct.
Its band % will be small but NON-ZERO -- that is the 1-in-7 edit rate, not a defect.
Tell the modes apart by the verifier output: constant 1.0 with NO edits was the OLD
free-pass bug; 0.0 with exactly one failing test is a correct read on an idle agent;
1.0 next to 4 edit calls is the pipeline working.

FIVE MISTAKES THAT COST REAL TIME TONIGHT -- do not repeat them
1. A LISTENER IS NOT A ROUTE. 1251403 froze at 868 log lines with NO error for 20
   min. arm_rollout_forward.sh logged a clean install and `ss` showed
   10.14.0.46:18300 LISTENING, but curl FROM A FLEET COMPUTE NODE gave HTTP=000 in
   1.5ms. The bridge forward (9923) was 200 the whole time, so everything looked
   healthy -- that's rule #1: one forward up, one down => 100% rollout timeouts.
   A cancel + reinstall of the BYTE-IDENTICAL spec fixed it. Now enforced:
     arm_rollout_forward.sh <job> 18300 <fleet_jobid>   # arg 3 = the real probe
   then grep -E 'ROUTE OK|FATAL' ~/arm_forward_<JOBID>.log. A 400
   "Model name mismatch: loaded model name g1_diverse_tezos_100k_8b" is a PASS.
2. jrc USED TO RUN HALF YOUR COMMAND ON THE WRONG CLUSTER and exit 0 (ssh joins all
   args into one remote string). Fixed 42bd89e9 via base64. A one-word payload was
   the only case it got right, so DISTRUST any pre-08-06 cross-cluster claim that
   was checked with a single-word jrc.
3. THE "AUTH OUTAGE" IS YOU. "Session open refused by peer" then "Permission denied
   (keyboard-interactive)" = you exhausted the ControlMaster's session cap with
   background pollers. NEVER kill the Jupiter master (TOTP re-auth risk, and it is
   the documented false "JuDoor key rejected"). Kill the pollers; it recovered on
   the FIRST retry. Run AT MOST ONE poller, always via jup/jrc, never raw ssh.
4. HER SIFs ARE amd64 -> JURECA ONLY. From Jupiter you get "FATAL: While checking
   container encryption ... architecture (amd64) ... host's (arm64)", which reads
   like a corrupt image but is just arch. The JURECA dc-cpu fleet is the ONLY place
   they run.
5. mirrorgate.py's CLONE verdict is INVERTED on the raw path: it prints FAIL when
   attempted=0, but 0 is CORRECT (the repo ships in the SIF). At tiny n its other
   verdicts print FAIL off n/a denominators -- FAIL at n=1 is NOT a gate reading.

STANDING RULES
  TP=1 ONLY. TP>1 makes vLLM build a distributed executor that trips Ray's own
    __RAY_WORKER_PROCESS_SETUP_HOOK assertion and NO engine starts; SkyRL mislabels
    it "EADDRINUSE" and burns 5x120s. Gate on
    grep -c RAY_WORKER_PROCESS_SETUP_HOOK <joblog>. Cost: all 8 nodes of 1247578.
  VERIFY BY BEHAVIOUR, NEVER CONFIG. Seven keys have been accepted, echoed and
    ignored (ai_memory/DEAD_KEYS.md); strict_json_parser=False is in the live config
    right now. A key is not set until a LOG LINE or a TRAJECTORY proves it fired.
  ANCHOR YOUR GREPS. The log echoes the whole hydra command line, so
    "FileNotFoundError" matches RewardFileNotFoundError inside mask_exceptions=[...]
    and "grad_norm=1.0" matches max_grad_norm=1.0. Both produced phantom findings.
  Never probe the vLLM endpoint DURING WEIGHT SYNC (kills an EngineCore, hangs the
    driver -- cost 1246702). After "Starting batch generation" it is safe.
  sacct -j <id> -X before writing any cause of death ("node failure" was really
    CANCELLED by 34902).
  Reward is at verifier_result.rewards.reward, NOT top-level reward. What matters is
    per-group len(set(rewards)) > 1, never avg_raw_reward.
  Do NOT rename a task dir or touch a Dockerfile under tasks/r2egym-raw -- SIFs are
    keyed build_${task}-${sha256(Dockerfile):12}.sif; 4568/4578 resolve for free.
  /e/fscratch for anything a job touches. /p/scratch is NOT mounted on Jupiter
    compute but IS visible from the Jupiter LOGIN node and JURECA -- the login node
    is the only host that sees both, so cross-cluster copies run there.
  source hpc/skyrl_standard/jupiter/jup.sh -> use jup / jrc. Serialise everything.

RETRACTED, do not re-derive: "358/4578 ~ 8%" (it is ~1.6k/4.5k ~ 36%); "33% pass
matches her 35% so a band EXISTS" (compared GROUPS to TRIALS); "concurrency is the
bottleneck => 6x free"; "1244916 died to a node failure"; "our JuDoor key is
rejected"; "the bare_json parser is validated" (it fires, but median 2 tool calls is
the 2-step stall, not health); "thinking-off (DONE)" (it is ON).

AUTONOMY: keep root-causing and relaunching on bring-up failures. Cancelling OUR OWN
wedged jobs is pre-authorised, never anyone else's. Push to lukedhlee forks freely
(origin is open-thoughts, no push rights); PRs need an explicit ask. Report when a
milestone lands or when blocked on something only Luke can do. Do NOT tune
throughput or scale nodes for its own sake -- but note the gate that gated it HAS
now passed, so scaling toward the full 4,568 (band_raw_all_names.txt) to compare
against her ~36% is the sanctioned next step once the edit rate is raised.
```

---

## Keeping this current

Refresh the "Last refreshed" line and any job/fleet ids. Everything else is stable guidance that belongs
here rather than in a paste — if a *rule* changes, change `NEXT_SESSION.md` and let this file keep pointing
at it.

⚠ **Read `NEXT_SESSION.md` first.** On 08-05 two jobs were burned rediscovering the `DCFT` value that was
already written in this file — because the takeover note pointed at `handoff.md` and `gotchas.md` but not
here. The launch recipe now lives in `NEXT_SESSION.md` §8.
