# Full Terminus-2 relay run on CalibForge

**What it is.** The data run that follows the 100-task pilot. It uses run 3's exact settings on all usable CalibForge
tasks:
- the 09-21 student, thinking on;
- parse_error repairs, plus sticky takeovers on done_claim, the exact-repeat loop and the no-progress wait;
- the student's thinking stripped from the teacher's view;
- summarization off, 64k input;
- the student's clock paused during repairs.

It produces two SFT arms of about 2,000 kept traces each, 1:1 pass:fail:
- **control**: Qwen3.8-27B from scratch;
- **relay_repair**: the student with teacher repairs and takeovers.

**Decided (Luke, 18:10 PT).**
- **Layout: 12 nodes.** Core is 4 × 09-21 + 4 × Qwen (8 nodes); burst is 4 × Qwen, released when control drains.
- **Cost.** About 34 node-hours expected (32–39 across the relay-episode range), 3.2–4.1 h of wall time. The hard
  ceiling is **42 node-hours**, from the two jobs' `--time` and the driver's node-hour stop.
- **Every Qwen reply is capped at 16,384 tokens,** reasoning plus content, in both arms.
  - The router adds `max_tokens` to every teacher request; harbor's chat path sends none of its own.
  - A reply cut at the cap reaches harbor as served (`finish_reason: length`). Terminus-2 then salvages it, or asks
    again ("NONE of the actions … were performed").
  - The router logs it as `teacher_cut_at_cap`. The readout counts cut replies and episodes with one per arm, and
    `select_kept.py` records `teacher_cut_turns` on each kept row.
  - When the cap does not fit what the context has left, vLLM answers 400. The router then asks once more for what
    the context has left, which is what the uncapped request would have done, and logs
    `teacher_max_tokens_lowered_to`.
  - Run 3 had one reply of 64,208 tokens.
- **Not changed.** No parse-after-thinking, so autofix stays at about 36 %.
- **Still with Luke:** the separate 100-task check run (about 2.4 node-hours) or the in-run check (cancel on relay
  overflow above 20 % after 200 episodes, which costs about 7–9 node-hours if it fires).
- **Nothing is submitted.**

## Re-check (Luke's go, 21:15 PT): five fixes, pass rule pre-registered before its job starts

**Rollout fixes, tested by the re-check.**
1. **Qwen serves `--max-model-len 131072`** for relay (`TEACHER_MAXLEN`; Qwen3.8's native length is 262k).
   - 09-21 stays at 65,536, exactly as in its eval.
   - Harbor's `max_input_tokens` is 131,072 for relay runs, and the router reports 131,072 on `/v1/models`.
   - Harbor's guard counts the owner's view. So a student turn whose view outgrows 65,536 ends with the student's own
     context 400. Harbor scores that as `ContextLengthExceededError`, an overflow.
   - The teacher's view (full reasoning) gets up to 128k. The student's view keeps the 1k cap on older Qwen reasoning.
2. **Every model call pauses the episode's budget clock** (`--pause-model-calls`), student and teacher alike, in both
   arms from now on.
   - The clock stops from the router sending a request (retries included) to its answer.
   - The student's clock runs from the episode's first request, and the teacher's from its takeover. Control's also
     runs from the first request.
   - **How the agent timeout is enforced:** the router ends an episode at 1× the task's budget on the owner's clock,
     with a synthetic `task_complete` (owner `router`, never trained on).
   - Harbor's own agent timeout, `agent_timeout_multiplier` 8.0, is only a backstop.
   - Logged per turn: `paused_sec`, `paused_this_turn_sec`, `student_clock_sec`.
3. **The Qwen reply cap goes from 16,384 to 32,768 tokens.** The fallback is kept: when the cap does not fit, the
   router retries without `max_tokens`.

**Re-check.** relay_repair only, on the same 100 pilot tasks, 1 × 09-21 + 1 × Qwen (128k), `-A transfernetx`. About
1.6 node-hours expected, ceiling 2.5 (2 nodes × 75 min). Decided by `check_decide.py --rule recheck`.

**It PASSES only if all five hold:**
- **C1, harness gate:** H0–H5, as before. Overflow is a scored model failure, never a harness error.
- **C2:** relay context overflow in at most 20 % of scored relay episodes.
- **C3:** zero Qwen context-length 400s. That means final 400s on teacher requests; a cap that did not fit and was
  retried without `max_tokens` does not count, but is reported.
- **C4:** the student owns at least 50 % of executed turns.
- **C5:** recovery after a takeover is at least 0.20.

**Informational, not a gate:** relay pass vs run 3's control, paired by task. The clock change makes it not
like-for-like.

**PASS** → `launch_relay.sh` (RULE=recheck, ROLLOUTS=4) starts the relay arm automatically.
- 945 baseline-solvable tasks × 4 rollouts, on 4 × 09-21 + 4 × Qwen (128k), all fixes on.
- Ceiling = 42 − (baseline 8.88 + the re-check's node-hours), so the whole full run stays within 42. About 24
  expected.
- The in-run overflow cancel stays as a backstop: more than 20 % of the first 200 episodes overflowing cancels the run.

**FAIL** → no launch, and a report of the failing check with its numbers.

**Training side (CPU, existing traces).**
4. **Kept failures:** a timeout failure is kept only if the episode had stalled.
   - Stalled means a loop or no-progress trigger fired (as a takeover or, in control, as `would_fire`), or its last 3
     executed turns produced no new terminal output (`relay_triggers`' own rule).
   - The rest are re-balanced to 1:1 with passes (`select_kept.py`).
5. **The same thinking mask in both arms** (`sft/render.py`).
   - A Qwen turn's thinking gets loss only if it was never cut in the rendered view AND is at most 8,192 tokens of
     09-21's tokenizer. That is 09-21's eval per-turn output limit.
   - Visible analysis, plan and commands are always trained.
   - 09-21's own turns are context only. Autofixed turns train the rewritten action only.

## Training-side results on the baseline (CPU, 21:30 PT)

**Stalled-only timeout failures shrink control's kept set from 1,890 to 440** (220 passes + 220 failures).
- Only 6 of the baseline's 848 timeout failures had stalled: 4 by a loop or no-progress trigger, 2 by no new output
  in the last 3 turns.
- The eligible failures are now 220: false done 155, context overflow 56, timeout (stalled) 6, format loop 2, tests
  failed 1.
- Most baseline timeouts ran out of wall-clock budget while waiting on the teacher. Latency was p50 52 s per call
  under the old clock, which counted serving time.
- **To get failures back toward 1,000**, the baseline has to be rerun under the new clock (paused on every model
  call), or the stall definition loosened. **Luke's call.**

**Same thinking mask, baseline rows:** all 2,042 render within 64k (p50 15.0k, p90 27.1k, max 54.3k).
- The 1k cap cut older Qwen turns 3,601 times.
- 105 uncut turns had thinking over 8,192 tokens, masked.
- 11,119 of 14,827 teacher turns keep their thinking trained.
- 11.3M trained tokens in total.

## Baseline arm result (job 2037808, 18:44–20:56 PT, 8.88 node-hours of its 13 ceiling)

**Qwen alone passes 945 of 2,007 scored tasks: 0.471, CI [0.449, 0.493].**
- **Harness.** Every check passed:
  - 34 harness errors (1.7 %): TmuxBatchProtocolError 32, TmuxCommandError 1, SetupScriptError 1.
  - Owner labels joined 17,207 of 17,207 turns.
  - Harbor re-fed the teacher's reasoning on 62,147 of 62,152 agent turns, with none restored.
  - The start-up burst's `EnvironmentStartTimeoutError` trials were retried, and all resolved.
- **Failure causes:** timeout 848, false done 155, context overflow 56, format loop 2, tests failed 1.
  - Overflow ended 71 episodes in all (3.5 %), so 15 of them passed anyway.
  - Turns per episode: p50 8, p90 13.
- **The 16k reply cap:** 621 replies were cut at the cap, in 528 episodes. On 332 requests the cap did not fit the
  context, and the router dropped `max_tokens` (the 88a4b3b8 fix working).
- **Latency.** Teacher p50 52 s, p90 361 s, at 100 agents per node.
- **Kept (`select_kept.py`): 1,890 traces**, 945 passes and 945 failures.
  - The kept failures are timeout 758, false done 133, overflow 51, format loop 2, tests failed 1.
  - There are no same-task pairs: one rollout per task.
- **Solvable list:** `runs/relay_full_baseline_20260925/solvable_tasks.txt`, 945 tasks.

**Relay arm, still HOLD (check run C2).** Projection on the 945 solvable tasks, from the check run's pass rate on
solvable tasks (0.67) and its episode times:
- **4 rollouts per task:** about 3,780 episodes. Kept about 2,000: 1,000 passes plus 1,000 failures, **every failure
  an overflow**. About 3.0 h of wall time and about 24 node-hours, inside the 33.1 left of 42.
- **3 rollouts per task:** kept about 1,680. About 2.3 h, about 19 node-hours.

## Check run result, corrected (19:30 PT): FAIL on C2, overflow 49 %; the relay stays on HOLD

**The first decision was wrong.** It scored the 40 episodes that ended on the cap-retry 400 as harness errors. That
failed C1 and dropped them from C2's denominator. Scored as the overflows they are:
- **C1 passes:** 8 harness errors in 100 trials; 99.35 % of the executed trace is valid format.
- **C2 fails: 45 of 92 scored episodes overflowed (49 %).**
- **C3 passes:** the student owns 72 % of executed turns.
- **C4 passes:** recovery after takeover 0.65.
- **C5 passes:** relay − control = −0.06, CI [−0.17, +0.06], paired over 89 tasks.
- **Relay pass rate:** 0.50, CI [0.40, 0.60].

**What fills the 64k.** Measured at the last request, overflowed vs other episodes.

| | overflowed (45) | other (47) |
|---|---|---|
| turns per episode | 25.6 | 16.0 |
| repairs per episode | 5.4 | 4.2 |
| autofixes per episode | 2.6 | 1.9 |
| takeover share | 0.22 | 0.51 |
| student view total | 41.1k | 18.3k |
| · terminal output | 18.0k | 8.5k |
| · 09-21 visible | 8.5k | 3.5k |
| · 09-21 reasoning | 5.7k | 1.7k |
| · Qwen reasoning, as shown (capped) | 5.4k | 2.2k |
| · Qwen visible | 2.6k | 1.5k |
| · prompt | 1.0k | 0.9k |
| teacher view total | 50.6k | 27.7k |
| · Qwen reasoning, older (full) | 11.7k | 7.5k |
| · Qwen reasoning, latest | 4.8k | 0.5k |

**Where the overflowed episodes ended.** 40 of 45 ended on a teacher request, which vLLM rejected because prompt +
the 16,384 reply cap > 65,536. The teacher's prompt was at least 49,154 tokens there. Harbor's own guard ended 5.

**Why the fixes did not move it** (run 3 was 44 %).
- **The cap only shrinks the student's view.** Qwen's older reasoning drops from 11.7k to 5.4k there. But the
  teacher's own request is not capped, and it is the one that fails.
- **Harbor's guard counts the owner's view.** During repairs that is the student's view, about 41k. So the guard does
  not trip before the teacher's larger request (about 50k) meets the 16k reply cap.
- **Terminal output is the largest component,** about 44 % of the context. No fix touches it.
- **The overflowed episodes are long student episodes that never hand off:** 25.6 turns, and only 22 % reach a
  takeover. Repairs keep the student going until the context is full.
- With the fixed router (88a4b3b8, `max_tokens` dropped when the cap does not fit), those 40 would have run on with
  about 15k of room. How many would then finish is unmeasured.

**Yield if overflow is accepted.**
- On the tasks run 3's control solved (the stand-in for "baseline-solvable"), relay episodes pass 33/49 = 0.67.
  **Every one of the 16 failures is an overflow.** Kept 1:1 failures would therefore all be truncated,
  context-full episodes.
- Projection, assuming about 1,100 solvable tasks from the baseline (it has passed 92 of 144 scored so far, and 272
  tasks lost to start timeouts are not in it):
  - 3 rollouts per task, about 3,300 relay episodes;
  - about 1,960 kept (980 passes + 980 overflow failures);
  - about 2.7 h of wall time on 8 nodes, about 22 node-hours;
  - **about 90 kept relay traces per relay node-hour;**
  - about 32–33 node-hours in total with the baseline, inside 42.

## Check run result (job 2037164, 18:19–19:10 PT, 1.62 node-hours): FAIL on C1, so the relay was not launched

**Verdict: FAIL.** The only failing check is C1: harness errors on 48 % of trials, against a limit of 10 %. 40 of the
48 come from the cap-retry bug (`RouterCapRetry400`). It was fixed in 88a4b3b8, after this run's router had started.
The other 8 are TmuxBatchProtocolError 6 and TmuxSessionEndedError 2. `launch_relay.sh` held as designed.

**The other checks passed, on the 52 scored episodes:**
- **C2:** overflow 5 of 52 (9.6 %).
  - **Caveat.** The 40 bug-hit episodes were all at a context of at least 49k tokens, so they are the ones most likely
    to overflow.
  - If all of them had overflowed, the rate would be 45 of 92 (49 %).
  - So C2 is not really measured by this run.
- **C3:** the student owns 72 % of executed turns.
- **C4:** recovery after takeover 0.64, CI [0.45, 0.80]. There were 34 takeovers (30 done_claim, 3 loop, 1
  no-progress wait).
- **C5:** relay − run-3 control = −0.04, CI [−0.20, +0.12], paired over 51 tasks. The relay pass rate is 0.63, CI
  [0.50, 0.75].

**The fixes worked.**
- 222 autofixes against 436 teacher repairs. Of all failing replies, 34 % were autofixed.
- 99.35 % of the executed trace was valid format.
- 18 teacher replies were cut at the 16k cap.
- The reasoning cap cut older teacher turns 891 times.
- Harbor re-fed the teacher's reasoning on 2,210 of 2,210 turns.
- The student's clock paused a mean 427 s per episode.
- Teacher latency p50 29 s, p90 236 s. Student latency p50 2.9 s.

**Spend.** 1.62 node-hours.

**Sandbox start-up.** The full-run baseline started 400 sandboxes at once and lost 272 trials to
`EnvironmentStartTimeoutError` in its first 5 minutes; none after. The retry list names this exception, but these
trials were not retried. Future runs set `environment_build_timeout_multiplier: 4.0`. The 272 tasks are not in the
baseline and would need a rerun (about 1.5 node-hours).

## Plan change (Luke, 18:40 PT): baseline first, relay only on solved tasks

- **Baseline arm, running now** (job 2037808, submitted 18:41 PT; `tmux relay_full_base`):
  - Qwen alone, 1 rollout per task on the 2,043-task pool, on 4 Qwen nodes at 400 concurrent (100 per node).
  - Settings as the full-run baseline: no summarization, 64k, 16k Qwen reply cap, harbor 89098635, and the same keep
    and labelling (`select_kept.py`).
  - Cost: about 10 node-hours expected, hard ceiling 13 (`--time 3:15`). This is carved out of the 42 node-hour
    full-run budget.
- **Relay arm, launched automatically** by `launch_relay.sh` (`tmux relay_launcher`, started 18:50 PT). It launches
  when the check run PASSES its pre-registered rule AND the baseline has **finished**.
  - Waiting for the whole baseline is deliberate. Its tail is the long-budget tasks, the ones most worth relaying, and
    a complete solvable list keeps the plan exact.
  - The relay runs **only on the tasks the baseline solved**, on 4 × 09-21 + 4 × Qwen (8 nodes, 400 relay agents),
    with the overflow backstop: cancel if more than 20 % of the first 200 relay episodes overflow.
- **Rollouts per task** (`relay_plan.py`): the smallest r of 1–4 whose expected kept traces reach 2,000.
  - Otherwise r = 3, and the shortfall is reported.
  - The expectation uses the check run's relay pass rate on tasks run 3's control solved.
  - Kept traces = 2 × min(passes capped at 2 per task, real failures, 1,000).
- **Cost rule.** Relay node-hours are projected as 8 × (startup + n·r·D/400 + p90 tail), with D and the tail taken
  from the check run. The relay launches only if that fits 42 − the baseline's actual node-hours. Its `--time` is
  that remainder ÷ 8, so baseline + relay stays ≤ 42 by construction. If it does not fit, no launch, and a report.
- **Expected**, from run 3 numbers:
  - about 1,140 solvable tasks × 3 rollouts ≈ 3,400 relay episodes;
  - about 2.6–2.9 h of wall time, about 21–23 node-hours;
  - about 31–33 node-hours in total with the baseline.

## Check run (Luke's go, 18:25 PT): pass rule, pre-registered before its job starts

**What.** relay_repair only, on the pilot's 100 tasks, with every fix on: autofix, the reasoning cap, the 16k Qwen
reply cap, the paused student clock, no summarization, 64k. It uses 1 × 09-21 + 1 × Qwen on `-A transfernetx`,
paired with run 3's control. Expected about 2.4 node-hours; hard ceiling 3.0 (`--time 1:30` × 2 nodes). Decided by
`check_decide.py`.

**It PASSES only if all five hold:**
- **C1, harness gate as before:** H0, H1 (router and trials), H2, H3 (agent turns), H4, and H5 (the executed trace is
  at least 99 % valid format).
- **C2:** relay context overflow in at most 20 % of scored relay episodes.
- **C3 (S2):** the student owns at least 50 % of executed turns. Autofixed turns count as the student's.
- **C4 (S4):** recovery after a sticky takeover is at least 0.20.
- **C5:** the relay pass rate is not below run 3's control by more than 15 points, paired by task.

**Amendment, 18:50 PT, before the check run's result.** A router bug is fixed in 88a4b3b8.
- **The bug.** When the 16k cap did not fit the context left, the router lowered `max_tokens` using vLLM's "prompt
  contains at least N input tokens". That N is only a lower bound, so the retry failed again with a context 400.
  Harbor then ended the episode as a context overflow.
- **The fix.** Retry without `max_tokens`. What the context has left is below the cap anyway.
- ~~How the running check run is scored: episodes ended by that 400 are harness errors (`RouterCapRetry400`).~~
  **Withdrawn at 19:30 PT.** Every one of those episodes was at a prompt of at least 49,154 tokens, at the context's
  edge. Harbor recorded them as ContextLengthExceededError, and they are scored as overflow, a model failure.
  Treating them as harness errors dropped them from C2's denominator and hid the overflow rate. They are now only
  labelled `cap_retry_400`.
- The thresholds are unchanged. The baseline and relay routers start with the fix.

**PASS** → the 12-node full run launches at once from the staged kit: ceiling 42 node-hours, with the in-run overflow
cancel kept as a backstop. **FAIL on any check** → no launch, and a report of the failing check with its numbers.

**Pool: 2,043 tasks** (`data/relay/pilot/full_pool.txt`). That is the 2,457 Daytona-covered CalibForge tasks, minus
the 300 held-out tasks, minus 114 tasks with agent budgets above 1,800 s. There are no duplicate instructions. The
pilot's 100 tasks are included.

**Phases.**
- **Phase 1**, all 12 nodes busy: control, 1 rollout per task (400 concurrent, 50 per Qwen node), and relay_repair,
  1 rollout per task (400 concurrent, 100 per student node).
- **Phase 2**: relay_repair, 2 more rollouts on every task control solved in phase 1. It starts once phase 1's relay
  job has at most 20 trials running, so the student nodes stay at about 100 agents.
- **Releasing nodes.**
  - The Qwen servers run as two jobs: *core* (4 × 09-21 + 4 × Qwen, 8 nodes) and *burst* (4 × Qwen).
  - When control finishes, the burst servers are dropped from the routers' teacher list and drained (no requests in
    flight, or 10 min). Then their job is cancelled.
  - Episodes pinned to a dropped or dead server re-pin to a live one.
  - The core job is released when the relay jobs finish, or when traffic stops after the deadline.

**Keep rule** (`select_kept.py`).
- Per arm, 1:1 pass:fail, target 2,000.
- At most 2 kept passes per task.
- Failures are only real model failures. Harness errors, verifier timeouts and deadline-censored episodes are
  dropped. Each failure is labelled by cause: context overflow, format loop, false done, timeout, tests failed.
- For each kept pass, a failure from the same task is preferred.
- relay_repair episodes count only if the teacher wrote at least one turn.
- **Limit:** with one control rollout per task, control can keep at most 2 × min(passes, failures). At a 0.57 pass
  rate that is about 1,760, short of 2,000.

**Cost.** Expected about 20–25 node-hours with a hard ceiling of 35, from the two jobs' `--time`. `full_decide.py`
sets those from run 3's measured episode times, so the two jobs' `--time` sum to at most 35 node-hours. The driver
also stops at 35 node-hours across both jobs. Wall time is about 4 h. Daytona sandboxes are free under the deal, at
most about 600 open at once.

**Launch rule** (Luke's conditional go, 15:35 PT; amended 17:40 PT: S5 is now a keep filter, and the S3 floor is
0.25). `full_decide.py` must say LAUNCH:
- run 3 passed every harness check (H0–H5);
- run 3 passed every scale-up check (S1–S4, S6);
- recomputed for this 10-node layout from run 3's measured episode wall times, pass rates and startup, the projected
  node-hours are at most 0.85 × 35 = 29.75, so nothing borderline launches;
- the core job's share of the ceiling covers the projected wall time;
- at least 1,500 kept traces are projected per arm.

Anything else is HOLD: no submission, and a report with the failing check.

**Server start.** vLLM's DP workers race for torch.distributed ports, and a start can die with `EADDRINUSE`. In run 3
this took 5 of 7 teacher start attempts and cost the first submission (job 2033358). `serve_node.sh` restarts a
server that dies before `/health`, up to 6 times on a 10-node run. Each retry costs about 2 min.

**Gates while it runs**, as in run 3:
- death check: DEAD / not RUNNING / 15 min with no traffic → cancel both jobs and clean up;
- latency gate at 15 min: teacher p50 above 30 s → warning; above 90 s → abort;
- early harness gate at 25 min (`readout.py --gate early`);
- node-hour cap.


## Changes after run 3 (Luke, 17:40 PT), built and tested, not run

**1. Format autofix before a Qwen repair** (`router/autofix.py`, router flag `--autofix`).
- When a student reply fails Terminus-2's strict parser but its action can be recovered without guessing, the reply
  is rewritten into valid Terminus-2 JSON and the student's own action runs.
- **Recoverable shapes:**
  - `<tool_call>` with keystrokes, `{"name": .., "arguments": {"command"|"cmd": ..}}`, `{"command": ..}`, a commands
    list, or a Terminus object;
  - several command objects, which become the commands list;
  - a JSON object missing analysis or plan, parsed leniently;
  - exactly one ```bash block;
  - `{"name": "finish"}`, which becomes `task_complete: true`.
- **How the rewrite is built:**
  - The student's thinking is kept verbatim, up to its last `<|end_think|>`.
  - analysis = the visible prose before the action; plan = "" unless the student wrote one; task_complete is kept.
  - The rewrite is checked with the same parser, and its commands must equal the recovered ones.
  - The triggers (done_claim etc.) judge the rewrite.
- **Logged per turn:** owner student, `autofix=True`, `autofix_kind`, and `original_student_reply`.
- **Still sent to Qwen:** no action, ended inside thinking, truncated, unparseable JSON, several bash blocks, and
  `thinking_holds_json`.
- **Measured on run 3's 528 discarded replies (CPU replay): 191 autofixable, 36 %.**

  | outcome | replies |
  |---|---|
  | autofixed: Terminus object parsed leniently | 40 |
  | autofixed: several objects → list | 36 |
  | autofixed: tool_call keystrokes | 31 |
  | autofixed: several tool_calls → list | 21 |
  | autofixed: bare keystrokes object | 19 |
  | autofixed: tool_call with a Terminus object | 17 |
  | autofixed: tool_call command | 12 |
  | autofixed: bash block | 12 |
  | Qwen: prose with no action | 125 |
  | Qwen: thinking_holds_json | 78 |
  | Qwen: unparseable tool_call JSON | 52 |
  | Qwen: ended inside thinking | 28 |
  | Qwen: truncated | 21 |
  | Qwen: other | 33 |

  - On a 3,000-reply sample of the held-out run's parse failures (09-21 alone), 65 % are autofixable.
  - `thinking_holds_json` means the student's own thinking contains a `{...}` that Terminus-2's parser reads first,
    because at this harbor base the parser runs on the raw reply, thinking included. Keeping the thinking verbatim
    makes these 15 % unfixable. They become fixable only if harbor parses after the think span (branch
    `lukedhlee/terminus2-think-parity`), and that changes the eval harness. **That is a decision for Luke, not taken.**

**2. The cap on older Qwen reasoning in the student's view** (`router/reasoning_cap.py`, router flag
`--student-tokenizer`).
- The most recent teacher turn keeps its reasoning in full.
- Every older teacher turn's reasoning is cut to its first 1,000 tokens of the 09-21 tokenizer, at the last sentence
  or line boundary, with no marker.
- The teacher's own view is unchanged. The same function serves the router and the SFT renderer.
- **Logged per request:** `student_view.cuts` with each cut's `content_sha`, `reasoning_chars` and `cut_at` (char
  offset). The exact student-bound body is logged too.
- **Run 3 replayed through the renderer:**
  - Cut in 80 of 100 relay episodes (182 teacher turns).
  - The rendered episode shrinks to a median 0.88 of its uncapped length.
  - All 100 fit in 64k (98 without the cap).
  - Of the 44 episodes that overflowed, the capped history leaves at least 16k tokens of room in 18.
  - Some overflows are on the teacher's side: a single Qwen reply ran to 64,208 completion tokens. A cap on Qwen's
    output (e.g. `max_tokens` 16k) would address that. **Not built; a decision for Luke.**

**3. Training layout, Luke's option 1** (`sft/render.py`).
- **One sequence per episode**, rendered with 09-21's own chat template (HF jinja settings; the token ids match HF
  `apply_chat_template` on run 3 episodes) and with exactly the capped history the student saw.
  - Teacher turns appear inline as `<|start_think|>{reasoning}<|end_think|>{content}`.
  - The final teacher turn keeps its reasoning whole, and every older one is cut.
- **Loss:**
  - Teacher content plus `<|eot_id|>` is always trained.
  - Teacher reasoning is trained only if it was never cut: it is at most 1,000 tokens, or it is the final teacher
    turn.
  - A cut turn's whole think span is masked, markers included.
  - An autofixed student turn is labelled. By default its rewritten action and format (content plus `<|eot_id|>`) is
    trained, never its reasoning; `autofix_loss='none'` masks it.
  - Everything else is masked: student turns, observations, router endings and headers.
- **`check_row` enforces these rules.** On run 3's relay arm:
  - 100 of 100 episodes rendered and passed the mask checks.
  - All fit in 65,536 tokens: p50 30.8k, p90 60.6k, max 65,155.
  - 787k trained tokens, against 1.76M uncapped.

**4. S5 becomes a keep filter; S3 at 0.28 passes.**
- `select_kept.py` keeps a done_claim takeover episode only if Qwen ran at least one command before confirming. On run
  3 that is 12 of 28.
- The readout now gates S3 at ≥ 0.25, with a target of 0.30. Run 3's 0.28 is inside noise, per Luke.
- Under the amended rule, run 3 passes every check (S1–S4, S6).
- Run 3 kept-trace replay: relay 56 (73 candidates after the S5 filter), control 86.

## Cost lines (from run 3's timings; relay episode 820–1,100 s after autofix and the cap, 900 s central)

- **(a) 100-task check run** of relay_repair with the fixes: 1 × 09-21 + 1 × Qwen, paired with run 3's control, which
  the fixes do not change.
  - Expected about 2.4 node-hours, 65–75 min of wall time.
  - Ceiling 3.0 node-hours, `--time 1:30` × 2.
- **(b) Full T2 run, 10 nodes** (2 × 09-21 + 8 × Qwen, 4 of them released after control):
  - 40–51 node-hours (43 central), 5.5–7.4 h of wall time.
  - The relay arm is bound by the 2 student nodes.
  - Over the 35 ceiling in every case.
- **(b) Full T2 run, 12 nodes** (4 × 09-21 + 8 × Qwen, 4 released after control):
  - 32–39 node-hours (33.7 central), 3.2–4.1 h of wall time.
  - Faster and cheaper than 10 nodes.
  - Inside 35 only at the optimistic end; `full_decide.py` HOLDs it under its 0.85 × 35 borderline rule.
  - A ceiling of 42 node-hours covers the pessimistic case.
  - Releasing 2 more Qwen nodes in phase 2 would save about 3 node-hours. The teacher's load there is only repairs
    and takeovers.
- **Kept traces (both layouts).** Relay reaches 2,000 (S5 filter included). Control reaches about 1,810, because one
  rollout per task at a 0.557 pass rate gives about 900 real failures.

**Auto-cancel for doing the check inside the full run** (`run_full.sh`, `OVF_AFTER=200`, `OVF_MAX=0.20`).
- Once 200 relay_repair episodes have finished, if more than 20 % of them ended in context overflow, the driver
  cancels both jobs, cleans up and stops.
- Run 3 was at 44 %.
- On the 12-node layout the first 200 relay episodes finish about 30–40 min after the servers are up. A cancel there
  costs about 7–9 node-hours, against 2.4 for the separate check run.

## How to launch (Jupiter login node; only on LAUNCH)

```bash
C=/e/project1/transfernetx/lee27/code
git -C $C/OpenThoughts-Agent fetch fork lukedhlee/vista-moe-grpo-30b
git -C $C/OpenThoughts-Agent worktree add --detach $C/ota-relay-full FETCH_HEAD   # never re-checkout a worktree a live driver reads
P=$C/ota-relay-full/data/relay/pilot
LIST=$P/full_pool.txt bash $P/stage_tree.sh /e/fscratch/reformo/lee27/tasks/calibforge_full2043
R3=/e/fscratch/reformo/lee27/experiments/relay/pilot/runs/relay_run3b_20260925
$C/envs/snowball-v2/bin/python $P/full_decide.py --run3 $R3 --readout $R3/readout_amended.json --relay-episode-s 900 > /tmp/decision.json   # 12 nodes, ceiling 42
# LAUNCH -> core_time_h / burst_time_h from the decision
CORE=$(RELAY_PILOT_DIR=$P N_STUDENT=4 sbatch --parsable --export=ALL --nodes=8 --time=<core_time> --job-name=relay_full_core $P/serve_relay.sbatch)
BURST=$(RELAY_PILOT_DIR=$P N_STUDENT=0 sbatch --parsable --export=ALL --nodes=4 --time=<burst_time> --job-name=relay_full_burst $P/serve_relay.sbatch)
tmux new -d -s relay_full "bash $P/run_full.sh $CORE $BURST relay_full_20260925"
```

Outputs:
- `runs/<name>/`: `readout.json`, `kept_manifest.jsonl`, and the router logs per arm;
- the harbor jobs, in `/e/data1/mmlaion/lee27/experiments/relay_full_jobs/<name>_{control,relay_repair,relay_repair_p2}`.

## Held-out reference (the "before" measurement)

Job `heldout_0921` is 09-21 alone on the 300 held-out tasks, 1 rollout each, 1 node, 2.0 node-hour ceiling. It is the
before reference for the later SFT arms: the relay-vs-control SFTs get evaluated on the same 300 tasks under the same
settings.
- Settings are run 3's student: thinking on, summarization off, 64k, Terminus-2 strict parser.
- There is no teacher and no triggers. The router runs in pass-through `--mode student` and logs owners and turns.
- The student budget is 1× the task's budget.
- All 300 tasks are included, the 15 over 1,800 s too, longest first.
- The readout gives the pass rate with a CI, the parse-error rate per turn, failure causes (format loop, false done,
  timeout, overflow, tests failed), turns, and the overflow rate.
