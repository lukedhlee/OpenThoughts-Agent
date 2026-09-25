# Relay pilot on 100 CalibForge tasks (Terminus-2)

**What it tests (run 2, Luke's design of 2026-09-25 13:45 PT).** Whether a student-to-teacher relay produces usable
SFT data on CalibForge when the student is the 09-21 Datakit SFT with thinking on, served exactly as in its TB2 eval.
That student often answers in its own `<tool_call>` format, which Terminus-2 rejects (run 1, below). So the relay now
has two kinds of hand-off:
- **Repair (`parse_error`, non-sticky).** Before harbor sees a student reply, the router runs Terminus-2's own parser
  on it (harbor's `terminus_json_plain_parser.py`, loaded from the harness's own file). If it would be rejected, the
  reply is discarded and logged, the teacher answers the same request for one full turn, and the student keeps the
  episode. No "fix your JSON" re-prompt enters the trace.
- **Takeover (sticky).** done_claim, the exact-repeat loop and the no-progress wait hand the rest of the episode to the
  teacher, as before.

Two arms, same 100 tasks, one rollout each, one 2-node allocation:
- **control**: Qwen3.8-27B from scratch. This is the baseline.
- **relay_repair**: the student with repairs and takeovers. The student's thinking is stripped from what the teacher
  sees.

A repair turn in the student's later history is shown to the student the way the SFT data will render a teacher turn
for 09-21: `<|start_think|>{teacher reasoning}<|end_think|>{teacher content}`, with no newlines around the span and no
separate reasoning field. The router logs it per turn (`student_view`).

**Cost.** The ceiling is 3.0 node-hours on `-A transfernetx`: 2 GH200 nodes × 1.5 h, with Slurm `--time` as the
backstop. I expect about 2.6 node-hours, which is about 75 min of wall time plus about 5 min of server start-up.
Queue time is extra.
- Each arm runs as one wave: 100 concurrent per arm, 200 Daytona sandboxes.
- A relay episode lasts at most 2× its task budget, and 37 of the 100 tasks have a 1,800 s budget.
- The router's deadline is 10 min before the cap. After it, every episode ends at its next request, so the run
  finishes with its verifiers run. The readout counts those episodes as censored.
- One Qwen node serves up to about 200 agents here. That is past the bench's sweet spot of 128 at short contexts, so
  expect queueing on the teacher. The readout reports latency by owner.

**Stop rule, rewritten 2026-09-25 before run 2 started.**

*Harness gate.* Every check must hold in both arms. If any fails, we fix the harness and re-run the pilot, with no
scale-up.
- H0: at least 90 % of student replies, discarded ones included, carry 09-21's think markers. This shows
  skip_special_tokens=false is working.
- H1, router side: the router stays clean. No FATAL, and no upstream error other than a context-length 400. No
  detector errors. Every episode matches its task, and every upstream call has its body logged.
- H1, trial side: at most 10 % of trials end in a harness error. Verifier timeouts and deadline-censored episodes are
  counted separately.
- H2: 100 % of trajectory agent turns join a router record. The join is by content hash, and the served model must
  match the record's owner.
- H3: at least 95 % of teacher replies carry reasoning. Harbor re-feeds 100 % of the teacher's earlier turns with
  their reasoning on main-chat agent requests, and the router restores none there.
  - Amended after run 2's abort: summarization requests are counted separately, because at this harbor base they
    rebuild history without reasoning (upstream #223 fixed it later) and the router restores it.
- H4: after every repair, the student answers the next request, unless a sticky trigger or an ending takes over at
  that request.
- H5: the executed relay_repair trace is at least 99 % valid format, measured as trajectory turns not followed by a
  parse-error re-prompt. What remains would be teacher replies that failed the parser twice.

*Scale-up to 2,000 kept traces per arm (1:1 pass:fail).* These count only if the harness gate passed. All six must
hold.
- S1: the control pass rate is between 0.25 and 0.75.
- S2: the student owns at least 50 % of the executed relay_repair turns. Below that, the trace is mostly the
  teacher's, and the arm is a noisier copy of control.
- S3: the sticky takeover rate is at least 0.30 of relay_repair episodes.
- S4: recovery after a sticky takeover, P(pass | takeover), is at least 0.20.
- S5: at done_claim takeovers, the teacher runs at least one command before confirming in at least 50 % of cases.
- S6: at the pilot's measured yield and node-hours per episode, 2,000 kept traces cost at most 60 node-hours per arm.
  - A kept relay_repair trace is an episode with at least one teacher turn (repair or takeover): its passes, plus as
    many real failures.
  - For control it is passes plus as many real failures.

If all six hold, the next wave runs relay_repair and control, each sized to 2,000 kept traces. It goes to Luke with a
cost line before any submission. If any check fails, there is no scale-up, and the readout names the failing check.

*Reported, not deciding.*
- The relay_repair pass rate against control, paired by task, with a bootstrap CI and the both / relay-only /
  control-only / neither table.
- Repairs per episode, and the share of executed turns by student, repair and sticky teacher.
- Teacher and student turns per episode.
- Teacher repair replies that failed the parser.
- Takeovers by trigger, with recovery and CIs.
- Kept traces per node-hour.

With 100 tasks, a paired difference in pass rate smaller than about 12–15 points is not distinguishable from zero.

The strip-vs-keep comparison of run 1 is dropped from this run. The code still supports it (`--student-think keep`,
arm `relay_keep`).

## How to launch (Jupiter login node, in tmux)

```bash
C=/e/project1/transfernetx/lee27/code
# code: the relay harbor branch, and this repo's relay files in a detached worktree of the existing clone
git clone -b lukedhlee/terminus2-relay --single-branch https://github.com/marin-community/harbor $C/harbor-terminus2-relay
git -C $C/harbor-terminus2-relay rev-parse --short=8 HEAD          # 89098635
git -C $C/OpenThoughts-Agent fetch fork lukedhlee/vista-moe-grpo-30b
git -C $C/OpenThoughts-Agent worktree add --detach $C/ota-relay FETCH_HEAD
P=$C/ota-relay/data/relay/pilot
# tasks: the 100 from the public bundle (sha256-pinned), plus the router's task file
bash $P/stage_tree.sh                                             # -> /e/fscratch/reformo/lee27/tasks/calibforge_relay100
# servers (2 nodes, 1.5 h wall = the 3 node-h ceiling), then the driver
mkdir -p /e/fscratch/reformo/lee27/experiments/relay/pilot/logs
JOB=$(RELAY_PILOT_DIR=$P sbatch --parsable --export=ALL $P/serve_relay.sbatch)
tmux new -d -s relay_pilot "bash $P/run_pilot.sh $JOB relay_repair_20260925"
```

Arms are set by `ARMS` (default `control relay_repair`). Everything a run writes is in `/e/fscratch/reformo/lee27/experiments/relay/pilot/runs/<name>/`:
- `driver.log`
- `router_<arm>/`: `turns.jsonl`, `events.jsonl`, `bodies/`, `health.json`
- `early_gate.*`, `readout.json` and `readout.txt`
- `run.meta`: node-hours and commits
- The harbor jobs, under `jobs/` → `/e/data1/mmlaion/lee27/experiments/relay_pilot_jobs/<name>_<arm>`

If the run is interrupted, clean up the sandboxes left behind by id with `python $P/cleanup_sandboxes.py --delete
<job dirs>`. The driver does this on every exit.

## What the kit does, in order

1. **Pre-flight, before any GPU time.** It checks:
   - the harbor clone is at 89098635;
   - the tree has 100 tasks, and none of them is in the held-out split;
   - the Daytona key works;
   - the three CalibForge snapshots are ACTIVE (`snapshot_census.py`). A missing snapshot would make harbor build one
     into an org at 39/40, so the kit stops and asks for `RECOVERY.md`'s restore.
2. **Servers.** `serve_relay.sbatch` starts the student on node 1 and Qwen on node 2, gives each one real completion,
   then writes the endpoint files.
   - The student completion must keep its think markers. The teacher's reasoning must come back split off.
   - If either server dies, the job writes `DEAD` and cancels itself.
3. **Routers.** One per arm. Each checks both served names and one completion, and exits 3 on a mismatch. A failed
   check cancels the job.
4. **Harbor.** One job per arm, 100 concurrent (one wave), longest agent budget first.
5. **Watch.** Every 60 s the driver checks:
   - router FATAL, the serve job DEAD or gone, and 15 min with no traffic → cancel and clean up;
   - at 25 min, the **early gate** (`readout.py --gate early`): H0, H1 router side and H3 on the logs so far, plus at
     least 20 turns per arm. Otherwise the driver aborts.
6. **Finish.** Stop the routers, release the nodes, delete leftover sandboxes, write `run.meta`, and run the final
   readout.

## Readout (`readout.py`)

Per arm:
- pass rate with a Wilson CI;
- harness errors by type, verifier timeouts and deadline-censored episodes;
- owner-label join rate, and teacher reasoning present and re-fed;
- summarizations per episode;
- per-turn latency by owner;
- teacher completion tokens.

Relay arms add:
- takeover rate overall and by trigger;
- takeover turn (quartiles) and budget used at the takeover;
- recovery by trigger with CIs;
- at done_claim, whether the teacher worked or confirmed without a command;
- episodes with no takeover, and how they ended.

Control adds how often the triggers would have fired on the teacher's own passes and failures (false fires).

Across arms: kept-trace yield (passes + an equal number of real failures), the projection to 2,000 kept traces,
strip vs keep, and the verdict under the rule above.

## Details and decisions

- **Student serve** (`serve_node.sh student`). This is `serve_snowball.sbatch` exactly as serve 1967259 ran it:
  model `grug-datakit-sft-20260921`, `POLICY=trained` (temperature 1.0), draft `probe_adapt_20260911/checkpoints/3`,
  its own `chat_template.jinja`, no reasoning parser, TP1×DP4×EP, 65,536 context, served name `snowball`. Harbor
  sends `skip_special_tokens=false` in every request, as the TB2 policy does. The router drops it for the teacher.
- **Teacher serve** (`serve_node.sh teacher`). This is `serve_qwen38.sbatch`, the bench's pick: TP1×DP4, MTP 2, 64k,
  prefix caching, `--reasoning-parser qwen3`, the model's own sampling defaults, served name `qwen38`.
- **Harbor.** `marin-community/harbor` `lukedhlee/terminus2-relay` @ 89098635. This is the v0.1 pin 7b18505a, plus
  three upstream commits (#150, #151, #153), the setup-files hook that CalibForge on Daytona needs, and two relay
  commits:
  - d006935e: Chat re-sends a turn's reasoning under both `reasoning` (the key vLLM's chat parser reads) and
    `reasoning_content`.
  - 8dfd835a + 89098635: the opt-in `llm_session_header`, which carries the agent's per-attempt session id.
  - The policy is the TB2 v0.1 one (32,768 in / 8,192 out, summarization on), plus three settings:
    `interleaved_thinking`, the session header, and `trajectory_config.raw_content`.
- **Episode identity.** Terminus-2 requests carry no per-trial id. LiteLLM sends `session_id` only when the job config
  sets one, and then it is the same for every trial. The summarization question request also starts a new message
  list. A per-trial `api_base` is impossible in one job config, and a hash of the first message collides on retried
  attempts. So harbor sends the agent's own session id as `X-Harbor-Session-Id`, the same id `trajectory.json`
  records. `/tokenize` calls carry no header and are routed by their first message.
- **Budgets.** The router ends an episode at the task's own budget: the student at 1× without a takeover, the teacher
  at 1× from its takeover, control at 1×. It ends one by answering `task_complete: true` with no commands, twice,
  because Terminus-2 confirms. Those turns are owned by `router` and never trained on. Harbor's own hard stop is set
  at 2.1×. This gives the teacher its own budget, as the design's guard asks, and ends both arms the same way.
- **Tasks.**
  - The 100 tasks (`relay100_tasks.txt`, strata in `relay100_tasks_strata.tsv`) come from the 2,457 Daytona-covered
    CalibForge tasks.
    - They exclude the 8 mini-swe-agent smoke tasks, the 130 tasks with budgets above 1,800 s (these bound the wall),
      and duplicate instructions.
    - They are stratified by subset × TB2-gap: 43 contrastive/Python, 35 contrastive/gap, 15 multi/Python and 7
      multi/gap. The category mix follows the pool.
  - The held-out eval split, `calibforge_heldout300.txt`, has 300 tasks under the same stratification, with all
    budgets included. It is disjoint from the 100, from the smoke tasks, and from their instruction texts. Never
    generate relay or SFT data from it.
- **parse_error repair, the details.**
  - The check is Terminus-2's own parser, `TerminusJSONPlainParser.parse_response` on the raw reply, as
    `_handle_llm_interaction` calls it. It is loaded from the harbor clone's file, and its sha256 is logged in the
    router's start event. The parser file is identical at the v0.1 pin.
  - Only `result.error` triggers a repair. Warnings pass through, as in the eval.
  - A repair turn is a full teacher turn (analysis, plan, commands). If the teacher's own reply fails the parser, it
    is asked once more (`--repair-attempts 2`); a second failure is returned and counted.
  - If a repair turn claims done, the teacher also answers Terminus-2's confirmation, and then the student resumes.
  - The student's own done claim is judged directly, not by the scanner's "first claim in the episode" rule, because a
    teacher repair turn may have claimed first.
  - Discarded replies stay in the router log (`discarded_student_reply`) for the readout and as preference pairs.
- **Retries** cover sandbox and upload infrastructure only. LLM-path errors are left to surface as trial exceptions.
- **Known effects to read the numbers against.**
  - With Qwen's reasoning re-fed under the 32k input cap, summarization will fire often. The readout counts it.
  - Terminus-2's "Extra text detected before JSON object" warning appears after every student turn, as it does in the
    TB2 eval. The teacher sees these warnings.
  - The student never sees its own rejected replies or the "fix your JSON" prompts it would get in the eval. That is
    the point of the repair, and it is a deliberate departure from the eval's distribution.
  - At 100 concurrent per arm, one Qwen node serves up to about 200 agents with long contexts. The KV cache, not decode
    speed, is the limit, so expect queueing. The readout reports latency by owner.

## Tests run before launch (2026-09-25)

- **CPU.** `data/relay/router/tests` has 23 tests, all passing, five runs in a row. They drive real Terminus-2 from
  the harbor branch, with a fake tmux session and scripted fake servers.
  - Run 2 adds:
    - a repair, after which the student resumes and a later done_claim takes over sticky;
    - two repairs in a row, where the teacher's second repair request re-feeds the first repair's reasoning;
    - the discarded reply never reaches harbor: it is not in `trajectory.json` or any later request, and there is no
      parse-error re-prompt anywhere;
    - the executed trace parses 100 %;
    - the student sees the repair turn as `<|start_think|>teacher reasoning<|end_think|>{json}` with no reasoning
      field;
    - the teacher answers the exact request the student failed.
  - Earlier tests cover no trigger, the budget, done_claim, the loop, no-progress wait, reasoning re-feed, concurrent
    episodes, summarization, a sticky decision trigger, the control arm, strip/keep rendering, the deadline, the
    health-check and FATAL paths, and the false-done guard.
  - The trigger tests pass, and so do the 200 harbor unit tests on the branch.
- **Wiring on real Daytona.** `wiring_check.sh` ran control plus relay_repair, with a fake student that sends one
  unparseable reply, on 2 CalibForge tasks (4 sandboxes).
  - H0–H5 all passed.
  - There were 2 repairs, and both returned to the student.
  - The executed trace was 100 % valid format.
  - Owner labels joined 10 of 10 turns.
  - BADFORMAT appears in the router log only, in 0 of 4 trajectories.
  - No sandboxes were left behind.
  - The science checks fail there by construction, since fake models solve nothing.

## Run 1: job 2028189, cancelled after 8 min (2026-09-25)

**I cancelled it because the thinking-on student cannot drive Terminus-2.** That is the eval's behaviour too, not a
harness fault.
- About 90 % of the student's replies in both relay arms were Datakit-style `<tool_call>{"keystrokes": …}` blocks,
  which Terminus-2 rejects. Of 191 relay-arm requests at the time of the check, 165 were parse-error re-prompts.
- In parse-error turns nothing runs, so loop and no-progress wait cannot fire. done_claim rarely comes. The relay arms
  would mostly have produced student timeouts with no hand-off.
- The 09-21 thinking-on TB2 eval (`grugdk0921think_t1_v01_20260923`, the serve this pilot copies) shows the same:
  - 67 % of its 14,420 agent turns contain `<tool_call>`, and 81 % draw a parse-error re-prompt.
  - Only 11 of 89 episodes stay under 20 % parse errors.
  - Our first prompt is byte-identical to the eval's apart from the task text.
- The thinking-off eval (`grugdk0921nt_t1_v01_20260922`) is much cleaner:
  - 19 % `<tool_call>`, 38 % parse errors;
  - 69 of 89 episodes under 20 % parse errors.
  - Datakit trained Terminus rows under `/nothink` only (`pipeline_findings.md`).

**What worked before the cancel.**
- Both servers came up in 270 s and 300 s with the right served names and smokes.
- All three routers passed their health checks.
- 59 episodes started, with 473 requests logged: 258 relay, 161 relay_keep, 54 control.
- Control's teacher replies carried reasoning, and harbor re-fed 38 of 38 earlier teacher turns.
- No router errors until the cancel. The 58 502s all come after it, when the servers died mid-request.
- The abort path released the nodes, stopped harbor, and left 0 sandboxes behind (150 trials checked).

**Spend.** 0.267 node-hours (481 s × 2 nodes).

**What Luke decided (13:45 PT).** Keep thinking on, as in the eval, and repair format failures with one teacher turn
(run 2, above). The option I had put to him was to relaunch with the student in thinking-off mode, `chat_template_kwargs.enable_thinking=false`
for student requests only (the TB2 nothink policy)? Then the relay_keep arm has nothing to keep, so the pilot drops to
2 arms at the 3 node-hour ceiling. The alternative is to keep thinking on and add a format-failure takeover trigger,
but that trigger is uncalibrated, and its relay data would teach recovery from format loops.

## Run 2: job 2030212, stopped by its early gate at 25 min (2026-09-25 13:57–14:29 PT)

**The gate stopped the run on a false alarm, but the run showed the real problem: one Qwen node cannot serve 200
CalibForge agents.**
- **The false alarm.** H3 counted 21 teacher turns the router had restored. All 21 were in Terminus-2's
  summarization requests (answers 19, summary 1, handoff 1). At this harbor base those requests rebuild history
  without reasoning; upstream #223 fixed it later. On agent turns, harbor re-fed 2,236 of 2,236 in control and 648 of
  648 in relay_repair. H3 is now judged on agent turns only (39c25def).
- **The teacher queue.** Teacher latency per request was p50 70 s (control) and 92 s (relay_repair), p90 about 330 s.
  Some requests waited 19 min.
  - Budgets are wall-clock, so this biases both arms.
  - In relay_repair the teacher's repair time also counts against the student's budget. 31 episodes ended at the
    student budget within 25 min.
- **The repair mechanism itself worked.**
  - 376 repairs in 98 episodes: mean 3.8, p90 7. Every episode needed at least one.
  - The student resumed after all 327 repairs that had a next turn.
  - The student owned 58 % of executed turns.
  - The executed trace was 99.5 % valid format. The 5 remaining parse errors are from 13 teacher repair replies that
    failed the parser twice.
  - 917 of 917 student replies kept their think markers.
  - Owner labels joined 100 % in both arms.
  - 6 done_claim takeovers, 5 of 5 recovered where scored.
- **Other failures, all from the cancel or infrastructure.**
  - Relay_repair's 8 teacher 500s and its FATAL are all from the cancel at 14:28:39 PT. vLLM rejected the queued
    requests at shutdown, and a connection reset reached the router's catch-all. Every aiohttp connection error is now
    an upstream error, not fatal.
  - TmuxBatchProtocolError hit 3 trials per arm, which is harness infrastructure.
- **Partial numbers.** Only 40 and 41 trials were scored, and they skew to short budgets, so they are not results.
  - Control: 23/40 passed.
  - Relay_repair: 22/41 passed.
  - Paired over 27 tasks: difference 0.00, CI [−0.19, +0.19].

**Spend.** 1.084 node-hours. With run 1, the pilot total is 1.35 of Luke's 3.0.
