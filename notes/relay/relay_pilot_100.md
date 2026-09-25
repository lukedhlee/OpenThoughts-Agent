# Relay pilot on 100 CalibForge tasks (Terminus-2)

**What it tests.** Whether a student-to-teacher relay produces usable SFT data on CalibForge. The student is the 09-21
Datakit SFT, served as in its thinking-on TB2 evals. When a calibrated trigger fires, Qwen3.8-27B takes over the live
sandbox for the rest of the episode. The triggers are done_claim, exact-repeat loop and no-progress wait. The pilot
first checks that the harness records what we need: an owner label on every turn, the teacher's reasoning kept and
re-fed, and no harness errors. It then measures the numbers that decide whether to scale to 2,000 kept traces per arm:
how often the teacher takes over, how often it recovers, and what a kept trace costs.

It runs three arms on the same 100 tasks, one rollout each, all in one 2-node allocation:
- **control**: the teacher from scratch. This is the baseline and the ceiling for recovery.
- **relay** (relay_strip): the student's think spans are removed from the history the teacher sees. Its Terminus-2
  JSON, including analysis and plan, stays. This is the design's default.
- **relay_keep**: the student's think spans are sent as each earlier student turn's reasoning, so Qwen3.8's template
  shows them inside `<think>`.

**Cost.** The ceiling is 4.0 node-hours on `-A transfernetx`: 2 GH200 nodes × 2 h, with Slurm `--time` as the
backstop. I expect about 3.2 node-hours and about 1.6 h of wall time after the job starts. Queue time is extra.
- Three mechanisms hold the cap:
  - After a deadline set 10 min before the cap, the router ends every episode at its next request.
  - The driver releases both nodes once the LLM traffic stops.
  - Harbor then finishes its verifiers on the login node with no GPU.
- The expected figure comes from these inputs:
  - The servers take about 8 min to come up. Snowball measured 200–330 s. The Qwen serve is the same layout as the
    bench.
  - After that, about 85–95 min of episodes at 50 concurrent per arm, longest budgets first.
  - A relay episode lasts at most 2× its task budget: the student gets 1×, then the teacher gets its own 1×.
  - Of the 100 tasks, 60 have budgets ≤ 900 s and 37 have 1,800 s.
- Daytona sandboxes are free under the deal. At most 150 are open at once.

**Stop rule, written 2026-09-25 before any result.**

*Harness gate.* Every check must hold in every arm. If any fails, we fix the harness and re-run the pilot, with no
scale-up.
- H0: at least 90 % of student replies keep their think markers.
- H1, router side: the router stays clean. No FATAL, and no upstream error other than a context-length 400. No
  detector errors. Every episode matches its task, and every upstream call has its body logged.
- H1, trial side: at most 10 % of trials end in a harness error.
  - Verifier timeouts are counted separately. One pilot task's verifier hangs on an unsolved sandbox.
  - Episodes cut by the deadline are also counted separately.
- H2: 100 % of trajectory agent turns join a router record. The join is by content hash, and the served model must
  match the record's owner.
- H3: at least 95 % of teacher replies carry reasoning. Harbor re-feeds 100 % of the teacher's earlier turns with
  their reasoning, and the router restores none.

*Scale-up to 2,000 kept traces per arm (1:1 pass:fail).* These use relay_strip against control and count only if the
harness gate passed. All five must hold.
- S1: the control pass rate is between 0.25 and 0.75.
- S2: the relay takeover rate is at least 0.40.
- S3: recovery, P(pass | takeover), is at least 0.20.
- S4: at done_claim takeovers, the teacher runs at least one command before confirming in at least 50 % of cases. A
  rubber-stamp confirmation teaches nothing.
- S5: at the pilot's measured yield and node-hours per episode, 2,000 kept traces cost at most 60 node-hours per arm.

If all five hold, the next wave runs the relay variant (strip or keep, per the rule below) and control, each sized to
2,000 kept traces from the pilot's yields. S5 caps each at 60 node-hours, and it goes to Luke with a cost line before
any submission. If any check fails, there is no scale-up: the readout names the failing check and the variant it
points to (K-turn relay, earlier triggers, a different pool).

*Strip vs keep (pre-registered 2026-09-25 13:25 PT, before the job started).*
- **Primary metric.** Recovery after takeover, P(verified pass | takeover), in each relay arm, with a Wilson CI. The
  difference is keep − strip, with Newcombe's 95 % interval. The readout also reports the task-paired version, a
  bootstrap over the tasks where both arms took over.
- **Keep is chosen only if all four hold:**
  - recovery_keep − recovery_strip ≥ +15 points;
  - the 95 % CI of that difference excludes 0;
  - keep does not worsen guard g1 by more than 5 points;
  - keep does not worsen guard g2 by more than 5 points.
- **Otherwise strip stays**, the default.
- **The guards**, as rates over takeover episodes:
  - g1, teacher false-done: the teacher's first `task_complete` comes within 2 of its turns after the takeover, with
    no verification command in its turns before it, and the task fails. A verification command is a test runner,
    script or build run that does not edit files (`is_check` and not `is_modify` in `relay_triggers.py`).
  - g2, context exceeded after the takeover.
- **Cost as a tie-breaker, not a decider.** Teacher turns and tokens (completion and prompt) per recovery, for both
  arms.
- **The pilot can only detect differences of about 15 points or more.** With roughly 50–70 takeovers per arm, the
  95 % interval on a difference in recovery is about ±15–18 points wide. A smaller real difference will read as
  "strip".

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
# servers (2 nodes, 2 h wall = the 4 node-h ceiling), then the driver
mkdir -p /e/fscratch/reformo/lee27/experiments/relay/pilot/logs
JOB=$(RELAY_PILOT_DIR=$P sbatch --parsable --export=ALL $P/serve_relay.sbatch)
tmux new -d -s relay_pilot "bash $P/run_pilot.sh $JOB relay_pilot_20260925"
```

Everything a run writes is in `/e/fscratch/reformo/lee27/experiments/relay/pilot/runs/<name>/`:
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
4. **Harbor.** One job per arm, 50 concurrent, longest agent budget first.
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
- **Retries** cover sandbox and upload infrastructure only. LLM-path errors are left to surface as trial exceptions.
- **Known effects to read the numbers against.**
  - With Qwen's reasoning re-fed under the 32k input cap, summarization will fire often. The readout counts it.
  - Terminus-2's "Extra text detected before JSON object" warning appears after every student turn, as it does in the
    TB2 eval. The teacher sees these warnings in both relay arms.
  - At 50 concurrent per arm, one Qwen node serves up to about 150 agents with long contexts. The KV cache, not decode
    speed, is the limit, so expect queueing. The readout reports latency by owner.

## Tests run before launch (2026-09-25)

- **CPU.** `data/relay/router/tests` has 20 tests: real Terminus-2 from the harbor branch, a fake tmux session, and
  scripted fake servers.
  - It covers no trigger, where the student completes; the student budget; done_claim; the exact-repeat loop;
    no-progress wait; teacher reasoning re-fed (three variants, plus the rendered Qwen3.8 prompt); concurrent
    episodes; summarization mid-episode; a discarding decision trigger; the control arm; strip vs keep rendering; the
    deadline; and the health-check and FATAL paths.
  - The rendered hand-off prompts for strip and keep are in `data/relay/router/tests/rendered/`.
  - The trigger tests pass, and so do the harbor unit tests on the branch (200 in `tests/unit/agents/terminus_2` and
    `tests/unit/llms`).
- **Wiring on real Daytona.** `wiring_check.sh` ran with fake models, 2 tasks × 3 arms, on real sandboxes.
  - Every harness check passed.
  - Owner labels joined 14/14, 8/8 and 14/14 trajectory turns.
  - All teacher reasoning was re-fed by harbor.
  - The keep arm sent the student's thinking as reasoning.
  - No sandboxes were left behind.
