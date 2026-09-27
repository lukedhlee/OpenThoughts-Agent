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

## Full relay run launched on the context-budget check (2026-09-26 15:00 PT)

**The check failed only on overflow, inside the band the plan had already said to launch on.** The context-budget
check (`relay_ctxb2_20260926`, job 2076715, 1.97 node-hours) failed C2 only: 22 of 94 scored relay episodes overflowed
(23.4 %, limit 20 %). All 22 are hard ends after a context_budget takeover. The other four checks passed: harness gate
clean, 99.95 % valid format, the student owns 51.6 % of executed turns, recovery after a takeover 0.56. Before the
result the plan said: overflow 20–25 % → launch the full run anyway, since overflowed episodes are scored failures that
the keep filter drops (they cost compute, not data quality); above 25 % → lower the trigger. So the relay launches.

**What runs.** The check's exact driver and settings (`run_pilot.sh`, `CLOCK=repair`, context_budget 32k, the 64k row
hard end, harbor input 131,072, Qwen reply cap 32,768, `--balance active`, the start wave staggered 3 min, the latency
and KV gates, the teacher format guard, the verify note) on the 945 baseline-solvable tasks × 4 rollouts (3,780
episodes). 4 × 09-21 + 4 × Qwen (one server per GPU, 128k), 192 concurrent (12 per Qwen GPU).
- `run_full.sh` is superseded: it passes none of the check's settings. `launch_relay.sh` now calls `run_pilot.sh`
  directly.

**Cost: a flat 30 node-hour ceiling** (`--time` 3 h 45 × 8 nodes). It replaces the old 42 / 47 accounting.
- Projected from the check's episode times (`relay_plan.py`: mean 707 s, p90 1,239 s): about 35 node-hours to finish
  all 3,780 episodes at 192 concurrent. The earlier estimate of about 24 assumed 400 concurrent.
- So the cap probably binds. The router deadline ends the last episodes 5 min before it, and the readout drops them as
  censored. Harbor runs rollout 1 of every task before rollout 2, so a cut tail loses late rollouts, not whole tasks.
- Solvable-task episodes may be shorter than the check's 100-task mix, which would bring the finish inside the cap.
- Expected kept traces are about 1,400, not 2,000: at a relay pass rate of about 0.80 on solvable tasks, failures
  are the scarce side of 1:1.

**In-run stop rule** (`stop_rule.py`, checked every 15 min from 300 scored relay episodes on): cancel if overflow is
above 30 % of scored episodes, valid format below 98 %, or harness errors above 10 % of finished trials.

**Attempt 1 stopped by the KV gate at 15:29 PT; relaunched at 9 episodes per Qwen GPU.**
- `relay_full_relay_20260926` (job 2077230, ecffa68b, 192 concurrent) ran 25 min. Latency was fine (teacher p50
  14–18 s), and no engine died. But the load was uneven: 3–5 of the 16 Qwen engines sat at KV 0.90–0.99 with 1–6
  requests waiting, while others were at 0.1–0.5. One engine stayed saturated for 5 min, and the gate cancelled the
  run. The cause is the context_budget takeovers: they hand the teacher whole 32k–57k episodes (157 fires in the first
  25 min), and `--balance active` counts episodes, not their size.
- Cost 4.42 node-hours. About 250 scored episodes finished; they stay on disk and join the relaunch's at the readout.
- **Relaunch** `relay_full_relay2_20260926` (job 2077750, 90c03166): the same settings at **144 concurrent (9 per Qwen
  GPU)**, plus `--failover-5xx`, so a crashed engine's 500s go to another server (baseline 6's failure). Cap 25.5
  node-hours (`--time` 3 h 11), so both attempts stay within 30.

**Attempt 2 stopped by the in-run stop rule at 16:13 PT (harness errors 10.3 % > 10 %). Nothing relaunched; Luke's
call.**
- At 313 scored episodes: overflow 15.0 % (limit 30 %), valid format 100 % (limit 98 %), harness errors 10.3 %
  (limit 10 %). The errors are TmuxBatchProtocolError (return code 143 or 137) and TmuxSessionEndedError.
- They came in a burst: 14–16 % of trials finishing per 10 min between 15:40 and 16:00 PT, then 5–6 % after. The
  burst overlapped baseline 6b's 400 sandboxes on the same Daytona org, and early finishers are enriched for errors.
  The steady 5–6 % matches the ctxb2 check's 6 %. No engine died in either attempt.
- The KV gate was also close at 9 per GPU: one or two engines at 0.93–0.99 with up to 4 requests waiting for
  10 min or more.
- **What the two attempts produced** (pass rate among scored; early finishers, so biased high):

  | | attempt 1 (192) | attempt 2 (144) |
  |---|---|---|
  | scored episodes | 225 | 313 |
  | pass rate | 0.87 [0.82, 0.91] | 0.80 [0.75, 0.84] |
  | real failures | 29 (overflow 16, false done 12, tests 1) | 63 (false done 40, overflow 21, tests 2) |
  | recovery after context_budget | 0.80 (n 121) | 0.71 (n 161) |
  | recovery after done_claim | 0.95 (n 62) | 0.89 (n 87) |
  | teacher guard: first sample failed | 7.1 % of 2,970 turns | 7.1 % of 3,474 turns |
  | kept 1:1 | 58 | 126 |
  | rendered rows over 64k | 0 of 430 | 3 of 485 (max 71,211) |
  | node-hours | 4.42 | 5.46 |

- **Spend so far: 9.88 of the 30 node-hours.** Run dirs `runs/relay_full_relay_20260926`,
  `runs/relay_full_relay2_20260926` (readout.json, kept_manifest.jsonl, render.txt).

**Attempt 3 ran to the node-hour cap (16:24–18:56 PT); the three attempts merged give 740 kept traces, not 2,000.**
The stop rule and the gates never fired. The 30 node-hours are spent, and the relay arm ends with 2,147 scored
episodes, 740 kept at 1:1. Failures are the scarce side: 377 real failures in 2,147 scored.
- **What ran.** `relay_full_relay3_20260926` (job 2078270, dd709f19, worktree `ota-relay-v8`) had the attempt-2
  settings at **128 concurrent (8 per Qwen GPU)**, with a 20.1 node-hour cap. It ran only the 3,242 (task, rollout)
  slots that attempts 1 and 2 had not scored (`merge_runs.py --per-task 4 --remaining`), so no slot was paid for
  twice. `run_pilot.sh` ran the repeated task lines round by round, longest budget first; `launch_relay.sh TOPUP=`
  sets this up. Baseline 6b had already left the queue (early gate at 16:01 PT), so nothing else of ours shared Daytona.
- **Health across the run.**
  - The early gate passed at 16:56 PT. The KV gate never fired. At most one engine at a time reached 0.95–0.99
    with 1–4 requests waiting.
  - The stop rule read overflow 11.8–17 %, format 99.99 % and harness errors 7.0–8.7 % at every evaluation. The last
    in-run value was 8.1 % at 18:39 PT.
  - Every context_budget fire was at 32,002 tokens or more. The view count matched the student's `prompt_tokens`
    on all 15,833 compared requests.
  - The teacher guard resampled 1.6 % of Qwen turns. The verify note fired exactly once per done_claim takeover.
- **How it ended.**
  - The router deadline (18:50 PT) ended 1,483 queued or in-flight episodes, which the readout drops as censored.
  - The driver released the serve job at the cap at 18:55 PT (20.19 node-hours). Harbor was stopped at 19:41 PT,
    after the 45 min verify window.
  - The final readout's H1 "router clean" fails on one unrecovered 502 at 18:49:52 PT: engine jpbo-066-42:8002
    refused a connection 15 s before the deadline. That was the only upstream error in the run.
- **Merged arm** `runs/relay_full_relaym_20260926` (attempts 1 + 2 + 3, `merge_runs.py --per-task 4`): 3,780
  slots, 2,147 scored (225 + 313 + 1,609), no slot scored twice. Pass rate among scored and failures are below.
  Early finishers dominate, so the pass rate reads high.

  | | merged 1 + 2 + 3 |
  |---|---|
  | scored episodes | 2,147 (910 of 945 tasks) |
  | pass rate | 0.824 [0.808, 0.840] |
  | real failures | 377: false done 225, overflow 136, tests failed 8, timeout 8 |
  | overflow (hard end included) | 268 of 2,147 scored (12.5 %), 263 of them hard ends |
  | recovery after context_budget | 0.78 [0.75, 0.80] (n 1,087) |
  | recovery after done_claim | 0.88 [0.85, 0.90] (n 643); the teacher ran a command before confirming in 98.8 % |
  | teacher guard: first sample failed | 6.9 % of 20,836 turns |
  | kept 1:1 (`select_kept.py`) | 740 (370 + 370; 60 same-task pairs) |
  | rendered rows over 64k (shared mask) | 15 of 3,780; 12 of the 740 kept (max 72,214) |
  | node-hours | 30.07 (4.42 + 5.46 + 20.19) |

- **Projection against 2,000.**
  - The 1,633 unscored slots would add about 1,500 scored episodes and about 260 failures, for about 1,260 kept.
    That costs about 19 more node-hours at attempt 3's rate.
  - 2,000 kept needs about 1,000 real failures, which is about 5,700 scored episodes at this failure rate. That is
    more than the 3,780 slots of the 945 × 4 plan.
- **Harness errors come from inside the sandbox and cluster on one task family** (diagnosis at 18:10 PT, 105 errors).
  - **What fails.** 84 % are TmuxBatchProtocolError: harbor's per-turn tmux script is killed in the sandbox before
    printing, by SIGTERM (143) or SIGKILL (137). 10 % are TmuxSessionEnded. Only 18 of the 88 follow an agent
    command containing kill or pkill.
  - **Where.** contrastive_solver / system-administration tasks (supervisord, nginx, gunicorn, Flask servers) error
    in 40 % of their trials (35 of 88); every other family is at 0–8 %. 12 tasks errored on every trial they had.
    The errors do not cluster on any student server or owner.
  - **Why it bursts.** The Daytona org held a steady ~758 other sandboxes the whole time. The 10-min rate (2.6–13.6 %)
    swings with when system-administration tasks come up in the queue, which likely explains attempt 2's burst as
    well.
  - **Fix before any further top-up.** Quarantine the repeat-offender system-administration tasks, or make harbor's
    batch exec survive in-sandbox kills (keep its argv from `pkill -f`, run it in its own process group). Either
    brings the rate to about 5–6 %.
- **Top-up 4 launched (Luke's go, 20:28 PT).**
  - **What runs.** `relay_full_relay4_20260926` (job 2080079, 128 concurrent, cap 20 node-hours) covers the 1,493
    slots still unscored after attempts 1–3. It was launched after baseline 6c left the queue.
  - **Quarantine.** The 35 tasks that errored on every trial so far are dropped, in
    `runs/relay_full_relaym_20260926/quarantine_tasks.txt`. They are exactly the 35 solvable tasks with no scored
    trial: 7 had only one trial, 18 had two, 10 had three or four. The baseline arm drops the same list when both arms
    are trimmed to equal size.
- **Stop-rule change for top-up 4 (Luke 20:55 PT).**
  - **What changed.** The harness-error part is judged only on tasks with no harness error in attempts 1–3. That
    list is 133 tasks, in `runs/relay_full_relaym_20260926/harness_error_tasks_1to3.txt`, linked as the run dir's
    `harness_exclude_tasks.txt`.
  - **Why.** A top-up re-runs the slots whose trials errored, so it is error-rich by construction. At 106 finished
    trials (20:51 PT), tasks with a prior harness error errored 12 of 34 times (35.3 %). Tasks that never errored
    errored 3 of 72 times (4.2 %), a normal baseline. The rule exists to catch a broken harness, and the clean tasks
    measure exactly that. The error-prone tasks carry the diagnosed in-sandbox tmux kill, and their errored trials
    are dropped from the data anyway.
  - **What did not change.** The 10 % limit stays, and overflow and format are still judged on all scored episodes.
  - **How.** `stop_rule.py --harness-exclude-tasks` (d0ddd237) falls back to `<run dir>/harness_exclude_tasks.txt`
    when that file exists, so the live driver picked it up. Only `stop_rule.py` changed in `ota-relay-v8`, and the
    driver's `run_pilot.sh` is byte-identical. The rule was live at 20:53 PT, at 113 scored.
  - **Logging.** Every check appends the judged rate, the all-task rate and the old rule's verdict
    (`all_task_rule`) to `stop_rule.log`. The first line (after 1) is a manual verification: judged 3.5 % (85
    trials), all tasks 12.4 % (129 trials), so the old rule would have read stop.
- **Files.**
  - Run dir `runs/relay_full_relay3_20260926`. The three-attempt merge described above now lives in
    `runs/relay_full_relaym123_20260926` (`MERGED.json`, `readout.json`, `kept_manifest.jsonl`, `select_kept.txt`,
    `render.txt`, and `rendered.jsonl` with all 3,780 rows, 696 MB). `relaym` is the four-attempt merge below.
  - Top-up list `runs/relay_full_relay3_20260926.topup.txt`.

**Top-up 4 finished every slot (20:28–23:40 PT, 19.36 node-hours); the relay arm ends at 1,182 kept traces.**
The stop rule and the gates never fired. The rule judged harness errors on clean tasks: 3.5–5.2 % at every check.
The old all-task rule would have stopped the run once, at 21:38 PT, when the all-task rate read 10.3 % over 777
trials. At every other check it read 9.0–9.9 %.
- **Attempt 4.**
  - All 1,493 slots ran, and none were censored. 1,354 were scored.
  - Pass rate 0.833 [0.812, 0.852]. The 226 real failures were false done 130, overflow 63, timeout 27, tests
    failed 6.
  - Harness errors were 9.2 % of trials: 111 tmux kills, 15 session ends, 5 Daytona 502s, 4 tmux command errors.
  - Router clean, with no upstream errors. The early gate passed at 21:00 PT.
  - The serve job was released at 22:53 PT, past the deadline with no LLM traffic. Harbor was stopped at 23:38 PT,
    after the 45 min verify window.
- **Merged arm, all four attempts** (`runs/relay_full_relaym_20260926`, `merge_runs.py --per-task 4`): 3,501 of
  3,780 slots scored (225 + 313 + 1,609 + 1,354), none scored twice. The 35 quarantined tasks have no scored trial.
  The quarantine and prior-error lists were copied into this dir.

  | | merged 1 + 2 + 3 + 4 |
  |---|---|
  | scored episodes | 3,501 (910 of 945 tasks) |
  | pass rate | 0.828 [0.815, 0.840] |
  | real failures | 603: false done 355, overflow 199, timeout 35, tests failed 14 |
  | overflow (hard end included) | 391 of 3,501 scored (11.2 %), 383 of them hard ends |
  | recovery after context_budget | 0.78 [0.76, 0.80] (n 1,737) |
  | recovery after done_claim | 0.88 [0.86, 0.90] (n 1,073); the teacher ran a command before confirming in 98.6 % |
  | recovery after loop / no_progress_wait | 0.83 (n 36) / 0.70 (n 23) |
  | teacher guard: first sample failed | 6.6 % of 33,041 turns |
  | kept 1:1 (`select_kept.py`) | 1,182 (591 + 591; 185 same-task pairs) |
  | rendered rows over 64k (shared mask) | 21 of 3,780; 15 of the 1,182 kept (max 72,214) |
  | node-hours | 49.43 (4.42 + 5.46 + 20.19 + 19.36) |

- **Where this leaves the 2,000 target.**
  - The 945 × 4 plan is exhausted: every slot outside the 35 quarantined tasks has been scored.
  - Failures cap the kept set. There are 591 usable real failures against 2,871 candidate passes.
  - 2,000 kept would need about 1,000 real failures. At the 17 % failure rate that is about 5,800 scored episodes,
    either more rollouts per task or harder tasks.
- **Files.**
  - Run dirs `runs/relay_full_relay4_20260926` (`stop_rule.log` has both harness-error rates per check) and
    `runs/relay_full_relaym_20260926`.
  - The merged dir holds `MERGED.json`, `readout.json`, `kept_manifest.jsonl`, `select_kept.txt`, `render.txt`,
    `rendered.jsonl` (all 3,780 rows), `quarantine_tasks.txt` and `harness_error_tasks_1to3.txt`.
  - That four-attempt merge now lives in `runs/relay_full_relaym1234_20260926`. `relaym` is the five-attempt merge
    below.

**Attempt 5 (Luke: 2,000 kept per arm) stopped on the number: 1,003 real failures, 1,958 kept, 26.39 node-hours.**
The merged relay arm reached its failure target at 03:58 PT, 3 h 17 min into the run and within its 32 node-hour cap.
No gate or stop rule fired. After select_kept the relay arm has 979 usable failures, fewer than the baseline's
1,000 strict, so both arms match at 979 + 979.
- **What ran.**
  - `relay_full_relay5_20260927` (job 2086476, 12ad6991, worktree `ota-relay-v10`) was the 910 non-quarantined tasks
    with tries 5–7: 2,869 slots from `merge_runs.py --per-task 7 --remaining`, round by round.
  - Settings as attempt 4: 128 concurrent, `CLOCK=repair`. The handoff said "clock paused during model calls", but
    attempt 4 ran `repair`, so it was kept identical for comparability.
  - Baseline 6d ran in parallel on the same Daytona org.
- **Stop rule.** Harness errors are judged on clean tasks via `HERR_EXCLUDE`: the 1–3 list plus the 10 tasks that
  errored on every trial in attempt 4, 143 tasks in `harness_error_tasks_1to4.txt`.
  - Judged rate 1.0–3.1 % at every check. The all-task rate was 3.9–6.5 %.
  - One burst at 01:40–01:50 PT (13.7 % over 10 min) was the known in-sandbox tmux kill, mostly on
    system-administration tasks.
- **How it ended.**
  - `run_pilot.sh TARGET_FAIL=1000 TARGET_BASE=603` (the 603 real failures of attempts 1–4) checked with the stop
    rule every 15 min. It read 1,003 at 03:58 PT.
  - It then stopped harbor, released the servers (26.39 node-hours) and ran the final readout. The 128 in-flight
    trials became CancelledError, which is why the attempt's own readout shows 10.3 % harness errors.
- **One unfixed router gap.**
  - Qwen engine jpbo-006-24:8001 died at about 03:43 PT. The main-request failover moved its traffic, but the teacher
    format guard's resample went back to the dead engine and returned 502.
  - 39 requests hit this (`teacher_guard.outcome = http_502`); those episodes end as harness errors, not data.
  - The guard's resample path needs the same connection-error failover. Not built.
- **Merged arm, all five attempts** (`runs/relay_full_relaym_20260926`, `merge_runs.py --per-task 7`): 5,780 scored
  slots (225 + 313 + 1,609 + 1,354 + 2,279) on 910 tasks, none scored twice.

  | | merged 1–5 |
  |---|---|
  | scored episodes | 5,780 (910 of 945 tasks) |
  | pass rate | 0.827 [0.817, 0.836] |
  | real failures | 1,003: false done 599, overflow 323, timeout 59, tests failed 22 |
  | overflow (hard end included) | 634 of 5,780 scored (11.0 %) |
  | recovery after context_budget | 0.78 [0.77, 0.80] (n 2,897) |
  | recovery after done_claim | 0.88 [0.87, 0.90] (n 1,722); the teacher ran a command before confirming in 98.9 % |
  | recovery after loop / no_progress_wait | 0.74 (n 66) / 0.81 (n 42) |
  | teacher guard: first sample failed | 6.6 % of 55,417 turns |
  | kept 1:1 (`select_kept.py`) | 1,958 (979 + 979; 349 same-task pairs). 24 real failures drop out: episodes where the teacher wrote no turn, and the S5 filter |
  | rendered rows over 64k (shared mask) | 35 of 6,526; 25 of the 1,958 kept (max 76,515) |
  | node-hours | 75.82 (4.42 + 5.46 + 20.19 + 19.36 + 26.39) |

- **Matching.** `match_kept.py --passes 979 --failures 979 --weak-timeouts fill` over the quarantine list writes
  `kept_manifest_matched.jsonl` (1,958 rows, identical to the kept set). The baseline arm is trimmed to the same
  979 + 979.
- **Files** in `runs/relay_full_relaym_20260926`: `merge.txt`, `MERGED.json`, `readout.json`,
  `kept_manifest.jsonl`, `kept_manifest_matched.jsonl`, `select_kept.txt`, `render.txt`, and `rendered.jsonl` (all
  6,526 rows; `render.py` identical to d0ddd237). The quarantine and prior-error lists are there too.
  Attempt 5's `stop_rule.log` has both rates and the real-failure count per check.

## Context-budget check (Luke 2026-09-26 12:50 PT): pass rule pre-registered, not submitted

**Why.** In the last relay check 09-21 wandered slowly: about 21 turns per episode against about 9 for Qwen alone.
Between 23 % and 49 % of relay episodes overflowed 09-21's 64k, mostly before any takeover. A CPU replay of the 300
logged relay episodes tried a takeover once 09-21's view reached 32k. It would have caught 96 of 132 overflows before
any other takeover. It fired in 52 % of episodes, at a median turn of 14. On 27 of those episodes 09-21 would have
passed alone. Luke chose 32k.

**What the router now does** (`relay_router.py`, tested on CPU, off unless the flags are set).
1. **context_budget, a sticky decision trigger** (`--context-budget-tokens 32000`).
   - Before the student is asked, the router counts 09-21's view of the request on the student server's `/tokenize`.
     This is the body the student would get, with older teacher reasoning cut, so the count is the `prompt_tokens`
     the student's request would report.
   - At 32,000 tokens or more, the student is not called. The teacher answers this request and keeps the episode.
   - It never fires before agent turn 2 (`--context-budget-min-turn`). It cannot fire again after a takeover, because
     the router only counts while the student owns the episode.
   - Logged like the other takeovers: the takeover record and event carry the trigger, the turn and `prompt_tokens`.
     Every counted request carries `student_view_tokens`.
   - The rule itself is `relay_triggers.context_budget_fire`.
2. **The trainability hard end** (`--student-row-max-tokens 65536 --student-row-reserve 8192`).
   - Once the teacher owns the episode, the router counts 09-21's view of every request the same way.
   - Above 65,536 − 8,192 = 57,344 tokens, it ends the episode. It answers that request, and any later one, with
     vLLM's own context-length 400 ("This model's maximum context length is …").
   - Harbor maps that 400 to `ContextLengthExceededError`, exactly as when 09-21's own server rejects a prompt. With
     summarization off, Terminus-2 re-raises it and the verifier scores the sandbox. So a hard end is a scored model
     failure, never a harness error.
   - Logged as owner `router`, ending `context_hard_end`, `upstream_status` 400, with the view size, the limit and
     the takeover trigger. The readout counts it as a context 400, not as an upstream error.
   - It applies after any takeover, done_claim included.
   - A final teacher reply longer than 8,192 tokens can still push the last row over 65,536. `sft/render.py` marks
     such a row `fits=False`.
3. **Count failures.** If `/tokenize` fails, neither rule acts on that request. The error is logged and the readout
   counts `view_count_errors`.

**Interplay** (all covered by the CPU tests).
- A parse-error repair turn counts toward the student's view, because teacher turns are rendered inline. Repairs
  stay non-sticky below the threshold. Once context_budget fires, no student call and no repair follow.
- done_claim below the threshold keeps done_claim. Terminus-2's confirmation is not a student request. When the
  budget is reached first, the teacher answers before the student can claim.

**The check.** relay_repair only, on the same 100 pilot tasks.
- **Settings are the re-check's:**
  - Qwen serves 131,072 tokens and harbor's `max_input_tokens` is 131,072;
  - the Qwen reply cap is 32,768, with the drop-`max_tokens` fallback;
  - autofix, the reasoning cap, summarization off.
- **One change to the clock (`CLOCK=repair`).**
  - The student's clock pauses only while a teacher repair is in flight. Otherwise the task's budget runs on wall
    time.
  - The router ends the student at 1× the task budget, and the teacher at 1× from its takeover. This is the check
    run's and run 3's clock.
  - Harbor's `agent_timeout_multiplier` of 8 is only a backstop.
- **Serving follows the healthy-serving rules.**
  - 1 × 09-21, served exactly as in its TB2 eval (TP1 × DP4 × EP, 65,536).
  - 2 × Qwen3.8 with one vLLM server per GPU (`PER_GPU=1`: 8 servers, TP1, MTP 2, 131,072).
  - The router pins each episode to the Qwen server with the fewest active episodes (`--balance active`).
  - 96 concurrent episodes, which is at most 12 per Qwen GPU.
  - The start wave is staggered in two halves 3 min apart.
- **Auto-cancel from 15 min after harbor starts:** the median Qwen reply latency over the last 5 min is above 30 s,
  or any engine (09-21's included) stays above 90 % KV with waiting requests for 5 min.
- **Cost:** 3 nodes, about 2.5–3 node-hours expected. The hard ceiling is **3.75 node-hours** (`--time 1:15`,
  `CAP_NODE_H=3.75`). The router's deadline ends episodes 5 min before that.

**It PASSES only if all five hold** (`check_decide.py --rule ctxbudget`):
- **C1, harness gate:** H0–H4 of the readout.
- **C2:** relay context overflow, hard ends included, in at most 20 % of scored relay episodes.
- **C3:** the executed trace is at least 98 % valid format.
- **C4:** the student owns at least 50 % of executed turns.
- **C5:** recovery after a sticky takeover (every trigger, context_budget included) is at least 0.20.

**Informational, not gates:**
- the context_budget fire rate, with the turn and view size at the fire;
- recovery after context_budget vs after the other takeovers;
- the hard ends and where the overflows came from;
- Qwen context 400s;
- the pass rate paired against run 3's control.

**PASS** → report to Luke. Nothing launches automatically. **FAIL** → report the failing check with its numbers.

**Launch (Jupiter login node; only after Luke's go, once the Qwen-alone baseline confirms the per-GPU serving):**

```bash
C=/e/project1/transfernetx/lee27/code; P=$C/ota-relay-v5/data/relay/pilot
JOB=$(RELAY_PILOT_DIR=$P N_STUDENT=1 PER_GPU=1 TEACHER_MAXLEN=131072 sbatch --parsable --export=ALL --nodes=3 --time=1:15:00 --job-name=relay_ctxb $P/serve_relay.sbatch)
tmux new -d -s relay_ctxb "ARMS=relay_repair CLOCK=repair CTX_BUDGET=32000 ROW_MAX=65536 ROW_RESERVE=8192 MAX_INPUT=131072 TEACHER_MAX_TOKENS=32768 BALANCE=active STAGGER_SEC=180 GATE_MIN=15 GATE_LAT=30 GATE_KV=0.90 CONC=96 NODES=3 CAP_NODE_H=3.75 bash $P/run_pilot.sh $JOB relay_ctxb_20260926"
# after RUN_DONE:
R=/e/fscratch/reformo/lee27/experiments/relay/pilot/runs
$C/envs/snowball-v2/bin/python $P/check_decide.py --check $R/relay_ctxb_20260926 --run3 $R/relay_run3b_20260925 --rule ctxbudget
```

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

## Overnight results (2026-09-25 21:07 PT → 2026-09-26 00:20 PT)

- **Re-check** (job 2041132, 2.52 node-hours, hit its 2.5 cap): **FAIL on C1 and C2.** The relay was not launched
  (`launch_relay.sh` HOLD).
- **Baseline rerun** (job 2041230, 10.02 node-hours, hit its 10.0 cap before finishing): partial.
  - The clock paused on model calls removed the only bound on an episode, and the Qwen engines saturated (the
    coordinator's diagnosis below). It is marked SUPERSEDED.
- **Spend on the full run so far:** baseline 1 8.88 + check 1.62 + re-check 2.52 + rerun 10.02 = **23.04
  node-hours**.

## Clean Qwen-alone baseline (Luke, 2026-09-26 11:30 PT; relay paused)

**Diagnosis (coordinator, from the rerun's vLLM logs).**
- Each GPU holds 440,878 tokens of KV cache, and CalibForge conversations reach about 34k tokens. So capacity is about
  12 agents per GPU, about 50 per node. We ran 100 per node.
- Busy engines sat at 85–95 % KV with 10–19 requests waiting and 0 % prefix-cache hits, while sibling engines on the
  same node idled: DP spread the load unevenly.
- Baseline 1's 848 timeouts were queue waits.

**The run.**
- All 2,043 training tasks, 1 rollout each, on 8 Qwen nodes on `-A transfernetx`.
- **50 agents per node** (400 in total).
- **One vLLM server per GPU**: TP1, DP1, 32 endpoints, Qwen3.8 with MTP2 (`PER_GPU=1`).
  - The router pins each episode to an endpoint, choosing the one with the fewest ACTIVE episodes (`--balance
    active`: pinned there and seen in the last 10 min).
  - It logs every endpoint's KV usage, running and waiting requests every minute (`engines.jsonl`).
- **Normal wall-clock budgets, as in the TB2 eval.** Harbor's own 1× agent timeout ends an episode. There is no paused
  clock and no router budget.
- Qwen `--max-model-len` 65,536; reply cap 32,768 with the drop-to-remainder fallback; no summarization.
- Sandbox starts: a 4× start window, and the start wave is staggered in two halves 3 min apart
  (`STAGGER_SEC=180`).
- **Auto-cancel from 15 min after harbor starts, if either holds:**
  - the median Qwen reply latency over the last 5 min is above 30 s;
  - any engine stays above 90 % KV with waiting requests for 5 min.
- **Cost:** about 8–10 node-hours expected, including the start-up wave (32 servers loading, about 8 min). Hard
  ceiling **12 node-hours** (8 nodes × 1.5 h `--time`). The router's deadline ends episodes 5 min before that.

**Readout** (`readout.py`, plus `select_kept.py --timeouts all` and `sft/render.py`):
- the pass rate;
- failure causes, with stalled vs not for timeouts;
- teacher latency p50/p90 per 10 min;
- turns per node-hour against baseline 1's about 2,300;
- per-engine KV maximum;
- kept 1:1 counts;
- the number of rows over 64k under the shared thinking mask.

## Baseline rerun (Luke's option 1, 21:30 PT) and the 47 node-hour ceiling

**What.** Qwen alone on the same 2,043 tasks, 1 rollout each, under exactly the relay settings:
- the clock paused on every model call;
- Qwen at 131,072 context;
- a 32,768-token reply cap;
- the same harbor and router refs;
- 4 Qwen nodes on `-A transfernetx`;
- the sandbox start window widened 4× (`environment_build_timeout_multiplier`), so no start-up burst loss.

It launches as soon as the re-check's early health gate confirms the paused clock and the 128k Qwen serve. It does
not wait for the re-check's result. About 9–10 node-hours expected.

**The whole full run's ceiling is 47 node-hours.** That covers the old baseline 8.88, the check run 1.62, the
re-check, the baseline rerun, and the relay.
- The rerun gets `--time` 2 h 30 × 4 nodes = 10.0 node-hours: 47 − 8.88 − 1.62 − 2.5 (re-check ceiling) − about 24
  (relay expected).
- The relay still launches only on a re-check PASS, using the existing 945-task solvable list, at 4 rollouts per task.
- `launch_relay.sh` charges the finished runs plus the rerun's full 10.0 node-hour ceiling as a reserve. The relay's
  `--time` is what is left of 47 ÷ 8 nodes.

**Training.**
- The rerun's traces replace the old baseline arm. The old arm stays on disk, marked `SUPERSEDED`.
- Keep filter and 1:1 rules as before. **Timeouts count as real failures now** (`select_kept.py --timeouts all`),
  since the clock no longer charges serving time.
- Stalled vs not is still reported for every kept timeout. The old baseline used `--timeouts stalled`.

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

## Clean baseline result (2026-09-26, 12:20–12:58 PT): serving fixed, run stopped by the early gate

**Per-GPU serving fixed the queueing.** With one Qwen server per GPU, no reply queued, KV never stayed saturated, and
not one trial timed out. The run still stopped after 25 min: one of the 32 engines crashed, and the early gate
(H1, "router clean") aborts on any upstream error, even when the retry succeeds. The 510 scored episodes are the
quick ones, so their pass rate is biased high and is not comparable to baseline 1's 0.47.

- **Serving:** Qwen reply latency p50 8.6 → 14.1 → 16.9 s and p90 about 95–100 s per 10 min. The latency gate never
  fired (worst 5-min p50 17 s). 12 of 32 engines reached KV > 90 % at some point, but at most 7 requests waited and
  the KV gate never fired. Throughput was 2,392 turns per node-hour, the same as baseline 1's about 2,300.
- **The abort:** engine `jpbo-104-46:8001` died at 12:50 PT from a CUDA device-side assert (a vectorized gather
  index out of bounds, then a flash-attn TMA descriptor failure). The router moved its episodes to the other
  engines, and all 14 of its 500s were retried. H1 counts upstream errors with no tolerance, so the early gate
  aborted at 12:58 PT (`early_gate.txt`).
- **Outcomes so far:** 947 trials. 399 were cancelled by the abort and 38 hit harness errors (7 % of the uncancelled:
  20 sandbox start timeouts, 17 tmux protocol, 1 setup). 510 were scored: 331 passes, pass rate 0.649, CI
  [0.607, 0.689]. The 179 real failures were 93 false done and 86 context overflow. Timeouts were 0. Overflow ended
  111 episodes (12 %) at 64k.
- **Kept 1:1:** 179 + 179 (`kept.jsonl`). All 919 rendered rows fit 64k under the shared thinking mask
  (p90 34k, max 54k tokens).
- **Cost:** 5.27 node-hours in total: 5.11 for job 2074704 plus 0.17 for the failed start 2074696. Commit
  bc0b2f59 had dropped `serve_node.sh`'s environment block, so every server exited at start; fixed in 276de47c.

Next decision: before any rerun, either let H1 tolerate retried 500s from a dead engine, or find the gather-index
crash (Qwen3.8 with MTP 2 under per-GPU TP1). Otherwise the next run can stop the same way.
Run dir: `/e/fscratch/reformo/lee27/experiments/relay/pilot/runs/relay_full_baseline4_20260926`.

## Baseline 5 stopped by hand: the verify note on every confirmation made Qwen re-verify forever (2026-09-26 14:38 PT)

**Why it was stopped.** Baseline 5 (job 2076765; commit 51cacd25; format guard on, verify note
on) was cancelled after 29 min and 3.86 node-hours, with its run dir kept
(`runs/relay_full_baseline5_20260926`). Serving and the format guard were healthy. The problem was the note.
- In Qwen-alone mode the router put the note on **every** Terminus-2 confirmation. Qwen ran checks instead of
  confirming 87 % of the time, then claimed done again, got the note again, and so on.
- By 14:35 PT, 41 of the 221 episodes that had reached a confirmation had 8 or more of them; the worst had 35 in 79
  turns, rebuilding and re-testing the same thing. Such episodes end only at the agent timeout or the context limit,
  so outcomes, the failure split and the kept traces would all have been shaped by the loop.
- The replay that justified the note only ever showed it once, and the relay arm shows it once (the done_claim
  takeover's confirmation). The fix (ac025a7a) puts the note on the episode's **first** confirmation only in
  Qwen-alone mode too. The relay path is unchanged.

**Relaunched** as `relay_full_baseline6_20260926` (job 2077047), same command, on ac025a7a, ceiling 11.7 node-hours.
The babysitter checks confirmations per episode (expected at most 2 for nearly all) and the share of first
confirmations where Qwen runs commands before confirming.

## Baseline 6 stopped by the early gate after an engine crash; top-up 6b on the tasks it did not score (2026-09-26 15:12 PT)

**What happened.** Baseline 6 (job 2077047, ac025a7a) ran cleanly for 25 min: the verify note sat on first
confirmations only (97 % of episodes at 2 or fewer confirmations, Qwen ran commands first at 81 % of first
confirmations), and the format guard held Qwen's first-sample parse failures at about 2.5 %. Then one Qwen engine
crashed at about 15:05 PT. Its API server stayed up and answered "EngineCore encountered an issue" 500s. Harbor
re-sent 4 of those requests (they got a 200 on another server) but not the 5th; that trial ended, and the early
gate's H1 aborted the run at 15:12:42 PT on the one unrecovered 500. 4.41 node-hours; 403 of 2,043 tasks scored
(268 passes); 400 trials were cancelled in flight.

**Fix (router `--failover-5xx`, `FAILOVER_5XX=1` by default in `run_pilot.sh`).** An upstream 5xx is now handled
like a connection error: that server is marked down for 5 min and the same request goes to another server, so a
crashed engine costs no request and never reaches harbor or H1. Model-facing settings are unchanged.

**Episode order was not shuffled.** `run_pilot.sh` orders tasks longest agent budget first, then splits them
alternately into the two staggered halves. So a run cut early has scored mostly the long-budget tasks that happened
to finish quickly; the tasks it did not reach are the shorter-budget ones. The cut therefore biases which tasks were
done, which is why the top-up covers every task without a scored trial rather than a sample.

**Top-up 6b** (`relay_full_baseline6b_20260926`): exactly the 1,640 tasks with no scored trial in baseline 6
(never started, cancelled, or harness error), identical settings (8 Qwen nodes per-GPU, 50 agents per node, 64k,
32k reply cap, CLOCK=wall, TEACHER_GUARD=1, VERIFY_NOTE=1) plus the 5xx failover. Ceiling 15.3 node-hours, about
1.3 × the projected 11–12.

**One baseline arm from 6 + 6b.** `merge_runs.py --out runs/relay_full_baseline6m_20260926 run6 run6b` builds a run
dir of symlinks: per task the first scored trial (6 first, then 6b), a task scored nowhere keeps 6b's trial, the
router records follow the kept sessions, node-hours add up, and `MERGED.json` lists the source of every task.
`readout.py`, `select_kept.py` and `render.py` read the merged dir as one run; no task is counted twice.

**6b stopped by the KV gate (2026-09-26 16:01 PT).** One of 32 engines (jpbo-057-08:8003) stayed above 90 % KV
with requests waiting for 5 min, 31 min after harbor started; the driver aborted as designed. 5.30 node-hours; 595
more tasks scored, no crash, no unrecovered upstream error. Baseline 6 + 6b together (merged read-only in
/tmp/lee27_b6m on the Jupiter login node, not yet the final `relay_full_baseline6m_20260926`): 998 of 2,043 tasks
scored, pass 655 / 998 = 0.656 [0.626, 0.685], 9.71 node-hours (plus 3.86 for the stopped baseline 5). 1,045 tasks
remain unscored.

## Qwen-alone baseline complete: 6 + 6b + 6c = `relay_full_baseline6m_20260926` (2026-09-26 20:10 PT)

**Result.** Every one of the 2,043 tasks has one trial; 2,001 are scored. Qwen3.8 alone passes **1,245 / 2,001 =
0.622, 95 % CI [0.601, 0.643]**, with the format guard and the verify note (first confirmation only) on. That gives
**1,512 kept traces at 1:1** (756 + 756), short of the 2,000 target because only 756 real failures exist.
- Top-up 6c (job 2079799, launched 19:11 PT once the relay job left the queue, 40 agents per node, 1,045 tasks)
  ran to completion: 1,003 scored, 7.70 node-hours, no gate stop.
- **Failures (756):** false done 312, timeout 224, context overflow 217 (316 episodes overflowed at 64k in all),
  tests failed 3. Excluded: 40 harness errors (37 tmux protocol), 2 verifier timeouts, 0 censored.
- **Format guard:** Qwen's first sample failed Terminus-2's parser on 3.0 % of 23,055 turns (0.1 % on plain
  confirmations, 1.4 % on noted ones, 3.6 % on work turns). 323 were autofixed, 251 recovered on a resample (547
  extra samples), 122 went to harbor unparseable; 114 replies were cut at the 32k cap. Executed steps are 99.98 %
  valid format.
- **Verify note:** 95.6 % of the 1,449 episodes that reached a confirmation stopped at 2 or fewer (max 4); Qwen ran
  commands before confirming at 77.8 % of first confirmations.
- **Harness:** H1 "router clean" fails on one 502, a request that arrived after 6c's servers were released at the
  end of the run; every engine error during the runs was retried to a 200.
- **Node-hours:** 17.41 for 6 + 6b + 6c (4.41 + 5.30 + 7.70), 21.27 with the stopped baseline 5.
- Files: `runs/relay_full_baseline6m_20260926/` (symlinked trials, `MERGED.json`, `readout.json`, `readout.txt`,
  `kept_manifest.jsonl`, `kept_summary.json`).

## Matched SFT arms: baseline trimmed to 591 + 591 and rendered (2026-09-27 00:15 PT)

**The baseline's 224 timeouts are slow Qwen replies, not stuck agents.** The baseline ran with `CLOCK=wall`, so the
agent budget also paid for Qwen's reply time, and these episodes spent their budget waiting on Qwen. (Corrected
2026-09-27: the relay arm runs `CLOCK=repair`, not a clock paused on every model call. Only teacher repair turns pause
it; the student's turns and sticky Qwen turns are charged their reply time, as the relay routers' logs show.)
Baseline top-up 6d runs `CLOCK=paused` on purpose, for yield: for Qwen alone `CLOCK=repair` is effectively
`CLOCK=wall`, whose latency timeouts are weak and dropped by `match_kept.py` anyway, so pausing turns that compute into
real passes and failures; both arms' kept sets exclude weak timeouts.
- **Timing.** Timeouts took a median 7 turns (passes 10) at 121 s of Qwen time per turn (passes 24 s). The median
  longest single reply was 445 s, and 56 % had a format-guard resample (passes 8 %). In 185 of 224 the last reply was
  still being generated when the clock ran out.
- **The rate follows the budget, not the task.** 27 % of 600 s tasks timed out, 13 % of 900 s and 2 % of 1,800 s.
  That is why baseline 6 (long budgets first) had none, 6b had 30 and 6c (the short-budget remainder) had 194.
- **Hand-read of 8.** Six were cut mid-work after 1 to 7 turns while one reply ran 5 to 20 min. One (BIND/DNSSEC,
  18 turns) had named answering queries. One was rewriting a broken test script. None looped.
- **Not the verify note.** Only 9 of 224 had reached a confirmation, and only 3 had stalled (no new output in the
  last 3 turns).

**Trim** (`match_kept.py`, 792421dc, seed 20260927; the original `kept_manifest.jsonl` is unchanged).
- The 35 quarantined tasks drop 21 rows (14 passes, 7 failures).
- Without the 218 unstalled timeouts only 531 real failures remain, 60 short of 591.
- **`kept_manifest_matched.jsonl`, 591 + 591.** It keeps all 531 and fills the 60 with uniformly sampled unstalled
  timeouts, marked `weak_timeout_fill`. Failures are false done 310, overflow 215, timeout 63 (3 stalled + 60 fill)
  and tests failed 3. The 591 passes are sampled from 742.
- **`kept_manifest_matched_strict.jsonl`, 531 + 531 in both run dirs.** No weak timeouts. The relay side is a
  uniform sample of its 1,182.
- Which pair trains is open.

**Render.** `ota-relay-v8` at d0ddd237, with the tokenizer and template from `grug-datakit-sft-20260921` and default
cap and limits. `render.py`, `reasoning_cap.py` and the readout code the renderer uses are unchanged from 6b3b21e0
through 792421dc. Re-rendering 5 relay rows reproduced their stored ids and masks exactly. The run rendered 2,042 of
2,043 episodes, with no errors and no row over 64k.

| kept rows (shared mask) | relay 1,182 | baseline matched 1,182 | relay strict 1,062 | baseline strict 1,062 |
|---|---|---|---|---|
| rows over 64k (max) | 15 (72,214) | 0 (55,381) | 13 (72,214) | 0 (55,381) |
| row tokens p50 / p90 | 37.1k / 57.7k | 22.2k / 38.9k | 37.4k / 57.7k | 22.5k / 39.5k |
| trained tokens, total (per row) | 9.40 M (7,957) | 9.78 M (8,273) | 8.48 M (7,986) | 9.00 M (8,478) |
| turns per row, mean (p50) | 21.1 (19) | 11.3 (10) | 21.0 (19) | 11.6 (11) |
| Qwen turns per row | 9.7 | 11.3 | 9.7 | 11.6 |
| Qwen thinking tokens per Qwen turn, as rendered | 619 | 515 | 619 | 511 |
| thinking share of trained tokens | 25.8 % | 28.4 % | 25.8 % | 27.9 % |

- Both arms train about the same number of tokens: the baseline gets 4 % more in the matched pair and 6 % more in
  the strict pair. Relay rows are longer because the student's turns are in context but masked.
- The thinking rows skip 18 relay rows. Their observations contain literal chat-header strings, which the tokenizer
  turns into special tokens. Those rows are in the user turns and never trained.

**Files.**
- Baseline run dir `runs/relay_full_baseline6m_20260926/`:
  - the two matched manifests, `match_kept.txt` and `match_kept_strict.txt`;
  - `rendered.jsonl` with all 2,042 rows, `render.txt` and `render.sha`;
  - `matched_stats.json`;
  - `timeout_probe.json` (per-episode timing) and `timeout_handread.txt`.
- Relay run dir `runs/relay_full_relaym_20260926/`: `kept_manifest_matched_strict.jsonl` and `match_kept_strict.txt`.

## Baseline arm at 2,000 kept: top-ups 6d + 6e, merged 6 + 6b + 6c + 6d + 6e (2026-09-27 02:46 PT)

**Result.** The Qwen-alone arm now has **1,038 real failures** (quarantine dropped, weak timeouts excluded, as in
`match_kept.py`), so its strict kept set is **1,000 passes + 1,000 real failures**, rendered with the relay arm's
`render.py` (d0ddd237) and settings: all 2,000 rows fit 64k.
- **6d** (job 2086439, fa6edbab): a second try on the 2,043 tasks minus the 35 quarantined, in a uniform shuffled
  order (`SHUFFLE_SEED=20260927`), `CLOCK=paused` on purpose for yield, 40 agents per node. The KV gate stopped it at
  01:39 PT (one engine above 90 % KV with requests waiting for 5 min): 7.87 node-hours, +330 real failures (861).
- **6e** (job 2091008, 403bf053): launched automatically at 01:42 PT on the 999 tasks 6d had not scored, in 6d's
  order, then a second pass; the target watcher stopped it at 02:22 PT at 1,038: 5.43 node-hours, +177.
  The 6f follow-up (32 agents per node) was armed but not needed. 6d + 6e = 13.30 of the 25 node-hour budget.
- **Merge** (`merge_runs.py --per-task 2`): 2,026 of 2,043 tasks scored, 3,546 scored trials; no task got a third
  scored trial, none counted twice. The 6 + 6b + 6c merge moved to `relay_full_baseline6m_20260926_v1` intact.
- **Pass rate** 2,282 / 3,546 = **0.644 [0.628, 0.659]**. Failures 1,264: false done 601, context overflow 435,
  timeout 225 (222 weak, nearly all from the CLOCK=wall runs; 6d + 6e added about one), tests failed 3. Harness
  errors 445, mostly 363 in-flight trials cancelled by gate aborts (so H1's 10 % trial check fails), 72 tmux protocol.
- **Format guard:** first sample failed Terminus-2's parser on 3.0 % of 44,022 Qwen turns; 598 autofixed, 1,031 extra
  samples, 223 passed unparseable, 191 cut at the 32k cap; executed steps 99.97 % valid format.
- **Verify note:** 95.7 % of the 2,717 episodes with a confirmation had at most 2 (max 4); Qwen ran commands before
  confirming at 77.7 % of noted first confirmations.
- **Strict kept set** (`kept_manifest_strict.jsonl`, seed 20260927): passes sampled from 1,254, failures from 1,038:
  false done 581, overflow 415, tests failed 2, stalled timeout 2. Rendered rows: 0 over 64k (max 58,152), tokens
  p50 22.6k / p90 39.7k, 16.93 M trained tokens (8,467 per row), 11.9 Qwen turns per row, 335 autofixed Qwen turns
  (action trained, reasoning masked).
- **Node-hours:** the arm 30.71 (6 4.41, 6b 5.30, 6c 7.70, 6d 7.87, 6e 5.43); 34.57 with the stopped baseline 5.
- Files in `runs/relay_full_baseline6m_20260926/`: `MERGED.json`, `readout.json`, `kept_manifest.jsonl`,
  `kept_manifest_all.jsonl`, `kept_manifest_strict.jsonl`, `match_kept_strict.txt`, `rendered.jsonl` (3,966 rows),
  `render.txt`, `render.sha`, `strict_stats.json`.

## Final matched SFT arms: 907 + 907 each, every row within 65,536 tokens (2026-09-27 04:41 PT)

**Result.** Both arms are trimmed to **N = 907 passes + 907 real failures** (`final_match.py`, fc2507db; seed
20260927). Each run dir has `final_manifest.jsonl`, `final_rendered.jsonl` (1,814 rows, the existing renders
subset by session, which is exact because a row is rendered per episode) and `final_match.json`.
- **Relay step 1.** 25 of the relay's 979 + 979 kept rows were over 65,536 tokens, all failures, so no pass needed
  replacing. That left 954 failures. I also held the relay's failures to the baseline's strict definition (no
  quarantined tasks, no weak timeouts, as `match_kept.py` defines them): 47 were unstalled timeouts, so **N = 907**.
  Keeping those weak timeouts would give N = 954 instead.
- **Trim.** `match_kept.py --weak-timeouts drop`: the relay's passes sampled from its 979, the baseline's from
  `kept_manifest_strict.jsonl` (1,000 + 1,000).

| per arm, 907 + 907 | relay | baseline (Qwen alone) |
|---|---|---|
| failure mix | false done 594, overflow 286, tests failed 16, timeout 11 | false done 526, overflow 377, timeout 2, tests failed 2 |
| trained tokens (per row) | 14.42 M (7,950) | 15.40 M (8,489) |
| row tokens p50 / p90 | 36.4k / 57.6k | 22.6k / 39.8k |
| max row tokens | 65,439 | 58,152 |
| Qwen turns per row | 9.7 | 12.0 |

- The baseline trains 7 % more tokens. Relay rows are longer because the student's turns are in context but masked.
- The relay's 11 remaining timeouts are stalled ones (a loop / no-progress trigger or no new output), the baseline's 2
  likewise; every kept row fits 09-21's 65,536.

## Final arms re-rendered with a 16,384-token thinking-loss limit (65k/16k TB policy, 2026-09-27 04:58 PT)

**Result.** Both 907 + 907 final sets are re-rendered with the per-turn thinking-loss limit raised from 8,192 to
16,384 tokens, as `final_rendered_think16k.jsonl` in each run dir (`final_rendered.jsonl` is kept). Only the loss mask
changes: every row's token ids are identical to the 8k render and every row's trained tokens are a superset. No row
is over 65,536 in either arm. `render_think_limit.py` (7e1e0c55) imports `render.py` from `ota-relay-v8` (d0ddd237),
so the template, tokenizer, capped history, mask rules and row limit are unchanged; only `think_limit` differs.

| per arm, 907 + 907 | relay | baseline (Qwen alone) |
|---|---|---|
| rows over 64k | 0 | 0 |
| trained tokens, 8k → 16k limit | 14.42 M → 14.97 M (+0.55 M) | 15.40 M → 15.77 M (+0.38 M) |
| Qwen turns with uncut reasoning masked for length, 8k → 16k | 81 → 34 (of 17,524) | 52 → 19 (of 21,796) |
| max reasoning tokens in one Qwen turn | 29,555 | 29,928 |

- "Masked for length" means a Qwen turn whose reasoning was never cut in the rendered view but is over the limit, so
  its whole thinking span is masked (the text itself is never truncated by this limit; older turns' 1,000-token cut
  in the student's view is separate and unchanged).
- Stats: `render_think16k.json` in each run dir.

## SFT arms: quality comparison (2026-09-27 10:00 PT)

**The relay's trained tokens sit where 09-21 goes wrong, but a fifth of them copy 09-21's format quirks and its
failures come from fewer tasks.** Measured on the two final training sets (`final_rendered_think16k.jsonl`, 907 + 907
each) with `data/relay/report/arm_quality.py` (read-only, login node, 6 single-threaded processes, about 1 min). The
gist section "SFT arms: quality comparison" is generated from its output by `status_figs.py`.
- **Where the relay's 14.97 M trained tokens sit.** Qwen after a context-budget takeover 49 %, after a done-claim
  takeover 11 %, after loop / no-progress 3 %, Qwen's one-turn repairs 29 %, 09-21's own autofixed actions 8 %. So
  92 % are Qwen working inside 09-21's episodes. Median context at the takeover is 32.3k tokens after the context
  budget (11 student turns) and 18.2k after a done claim (8).
- **Repair-only rows are not empty.** 299 rows (16 %) have no takeover; they carry 1.34 M trained tokens (9 %),
  median 3,990 per row. Only 17 relay rows (0.9 %) train under 1,000 tokens. In 338 relay rows the first Qwen done
  claim is a repair turn (the router lets a repair turn's claim through without the verify note); 142 of them failed.
- **Coverage.** Relay 790 tasks, 56 % of rows from tasks with 3 or more rows, up to 7 rows per task; its 907 failures
  come from 437 tasks. Baseline 1,342 tasks, at most 2 rows per task, failures from 680 tasks. Family mix is the same
  (16 families, software engineering 37–38 %).
- **Behaviour.** Before Qwen's first done claim, a check (tests, running the program, or reading the output) sits in
  the last 3 turns in 99 % of relay passes and 100 % of baseline passes (tests or the program 90 % and 94 %). Thinking
  per Qwen turn is 1,600 tokens mean (345 median) against 1,473 (274). Failure rows with a repeated command are 4.4 %
  and 4.5 %.
- **Format residue (the main cost).** Qwen copies 09-21's quirks from the context. 3,782 of 17,524 trained Qwen turns
  (22 %, in 615 rows) put `<tool_call>` (3,364) or `<|end_think|>` / `<|start_think|>` (486) into the reply content,
  against 10 of 21,796 in the baseline. They hold 20 % of the relay's trained tokens. 58 % of relay Qwen turns have
  prose before the JSON (baseline 22 %). Terminus-2 accepts these with warnings, so the executed-step format rate
  does not show them. Stripping the markers from teacher content at render time would remove them.
- **Hand-read of 13 failed rows** (7 relay, 6 baseline; sampled false done ×3, overflow ×2, other ×1 per arm, plus
  one relay repair-only false done). Neither arm rubber-stamps a done claim; every false done followed a check.
  Both miss the same way: verifying against their own tests, or explaining away a warning ("The Flask module error
  is expected"). Both arms have one row hunting for the grader, and both have a trained thinking span of about 10k
  tokens. The one baseline timeout row spends about 13 trained turns waiting on a frozen screen. The relay rows add
  the copied markers and 09-21's messy turns in the masked context.
