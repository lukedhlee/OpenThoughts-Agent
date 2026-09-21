---
name: gepa-prompt-loop
description: >-
  Run a GEPA-shaped prompt-evolution loop for Snowball on R2E-Gym, agentically — THIS session is the
  reflection LLM, there is no GEPA framework. A population of general terminal-agent guidance blocks
  (≤ 400 tokens, procedure only) is appended to each task's instruction.md as a strict suffix; each
  candidate is scored on a 500-task dev split that is SCORES-ONLY, and reflection reads a separate
  64-task feedback sample drawn from train; selection is a per-task Pareto front over pass plus eight
  deterministic behaviour axes. Compute is one standing serve job plus a candidate queue. Use when
  asked to evolve / optimise / search the agent prompt, run a GEPA wave, reflect on a candidate's
  feedback traces, or gate a child block. Scripts:
  data/r2egym/jsc/gepa/. Reference: ai_memory/active/snowball-r2egym/research/2026-09-21_gepa_agentic_loop.md.
---

> ⚠ **The loop runs on ONE standing serve job and a candidate queue.** State its cost line once, at
> session start, and wait for the go. After that, enqueueing candidates needs no new go — the
> allocation is already paid for and the queue only decides whether it idles. What you owe instead is
> a node-hours-burned figure in **every** status line. `gepa_serve.sh` never submits without
> `SUBMIT=1`, and the test set never runs without `CONFIRM=1`.

# gepa-prompt-loop

GEPA without the GEPA library. The population is a set of **guidance blocks**: short, general
procedure for a terminal agent — how to explore, reproduce, edit, verify, decide it is done. A block
is appended to every task's `instruction.md` behind a fixed delimiter, so the control arm is the same
file with nothing added. The **reflection LLM is this session**: it reads the traces the parent lost
on and writes the children by hand. Nothing about the loop is automatic except the scoring.

Access boilerplate (ssh, login-node limits, paths) → `.claude/ops/jupiter/ops.md`. The probe recipe
this loop rides on → `.claude/skills/rl-agentic-launch-jupiter/SKILL.md` §4 for the Daytona
conventions. The behaviour detectors and their provenance → `data/r2egym/jsc/gepa/gepa_feat.py`.

## 1. Hard rules

- **One cost line, at session start, for the serve job.** `gepa_serve.sh up` prints it: nodes × wall.
  Show it, get the go, then `SUBMIT=1`. Enqueueing a candidate afterwards is free of a new go.
- **Never let the servers idle.** Keep **at least two candidates queued at all times**. Score and read
  finished candidates while the next ones run. The whole point of the standing job is that the GPUs
  do not wait for a reflection step.
- **Report node-hours burned in every status line.** `gepa_serve.sh status` prints it. A standing
  allocation is invisible unless you say what it has cost.
- **Dev is scores-only. You never read a dev trace.** Selection runs on dev numbers — per-task reward,
  the behaviour axes, the Pareto front, paired wins and losses — and `gepa_score.py` gives you all of
  them. The trajectories are closed. Reading the traces of the very tasks selection scores is how a
  prompt gets fitted to 500 particular tasks instead of to the job. `gepa_dump.py` and `gepa_worst.py`
  refuse a dev id, exit 2.
- **Reflection reads this wave's feedback batch, and nothing else.** Every wave draws a **fresh 64**
  train tasks, seeded by the wave name, into `experiments/gepa/<wave>/feedback.txt`. Every candidate
  is rolled out on it (k=1) alongside dev, and it is the only trace the session ever opens. Nothing
  selects on train, so reading it costs no generalisation — and because the batch changes each wave,
  the search cannot start fitting one fixed set of 64. `gepa_dump.py` and `gepa_worst.py` refuse a
  task from another wave's batch as firmly as they refuse a dev id.
- **The cheap gate also uses a fresh minibatch.** Each gate wave draws 32 fresh dev tasks
  (`<wave>/gate.txt`), and the parent and control are re-run on exactly those 32, so the comparison
  stays paired. `split_dev_mini.txt` is kept only for the pilot. **The full dev-500 stays fixed** —
  that is what makes candidates comparable across waves.
- **Every full-dev candidate also runs OODMINI**, 32 train tasks from the held-out repos, scores only.
  A candidate whose dev delta is positive while its OODMINI delta is below −0.05 is a **specialist**:
  it bought dev with the band's own repos. The ledger marks it and parent sampling skips it, so the
  trick cannot breed.
- **Test is sealed.** Scored exactly once, by `gepa_final.sh`. `gepa_tree.py` refuses to build a tree
  over it without `FINAL=1`, which only `gepa_final.sh` sets.
- **SWE-bench and Terminal-Bench 2 are outside the loop entirely.** Never rolled out by it, and their
  traces and per-task results are off limits to the session — not "prefer not to", off limits. They
  are the final base-vs-prompt comparison, run on Luke's go after the loop ends. A block tuned against
  the thing that is supposed to judge it tells us nothing. What the session is allowed to carry from
  them is the general prior in §2.1, which names behaviours and no tasks.
- **Blocks are ≤ 400 tokens and general procedure only.** Never a task id, repo name, file name, test
  name or dataset name. `gepa_tree.py` lints this and refuses to build the tree if a block fails.
  The point is a procedure that would help on any terminal task, not knowledge about these tasks.
- **The ledger is the source of truth.** `experiments/gepa/ledger.json` holds every candidate, its
  parent, its block text and its dev scores. A candidate that is not in the ledger does not exist.
- **Every wave's `summary.md` is written before the next reflection.** Reflect from the scored
  summary and the dumped traces, never from memory of what a run looked like.
- **Write the pass/fail rule before the result lands.** The gate thresholds are flags on
  `gepa_score.py`; state them in the cost-line message, not after seeing the numbers.
- **Standing cluster rules still apply.** Login node only for the scripts (no `find`, no `du`,
  `OMP_NUM_THREADS=1`, ≤ 4 python processes). Local clone is ground truth; `code/snowball` on Jupiter
  is a plain copy, so changes go out with `gepa_sync.sh --go`, not `git pull`.
- **The runner lives on the login node and so do the harbor runs.** Each shard is a tmux running
  `harbor` at concurrency 32; several at once is hundreds of threads against a 4,096 pid ceiling that
  was hit once and took Jupiter down for everyone. `gepa_runner.py` refuses to admit while the thread
  count is above `PID_GUARD`. Do not raise `SHARDS_PER_CAND`, `CONC` or `MAX_INFLIGHT` to route
  around that message — the guard is the thing standing between this loop and an outage.

## 2. What the loop is

| piece | what it is here |
|---|---|
| **candidate** | one guidance block, ≤ 400 tokens, appended to `instruction.md` behind `\n\n---\nWorking guidance:\n` |
| **control** | the same task with nothing appended (`-pctl`); every wave carries it |
| **score** | per task: the verifier reward, plus eight deterministic behaviour features from the trajectory |
| **dev (500)** | FIXED and SCORES ONLY. Selection runs on its numbers; its trajectories are closed to the session |
| **feedback (64)** | a FRESH draw from train each wave. The only traces the session ever reads |
| **gate (32)** | a FRESH draw from dev each gate wave; parent and control re-run on the same 32 |
| **oodmini (32)** | fixed OOD-repo train tasks, scores only: the specialist check on every full-dev candidate |
| **selection** | per-task Pareto front over (pass, the eight axes) on DEV; parents sampled by how many tasks they win |
| **reflection** | this session reads the parent's worst FEEDBACK traces and writes 2–3 children with the lesson added |
| **cheap gate** | child vs parent on the wave's fresh 32, paired wins plus the predicted axis — pass rate alone cannot decide at this n |
| **judge** | 8 feedback traces hand-graded per accepted child; an axis below 6/8 agreement is FROZEN and leaves the front |
| **confirm** | the winner re-run vs control on dev at k=2 before the test set is spent |
| **full eval** | an accepted child is re-scored on all 500 dev tasks and entered in the ledger |
| **final** | the best block and the control, once, on the 500-task test split, paired, plus the OOD-repo check |
| **the compute** | one standing serve job (8 nodes, one vLLM server each) plus a login-node queue. A candidate is N `harbor run`s against those servers on Daytona, never its own Slurm job |

## 2.1 Prior knowledge: the failure modes, as behaviours

Carried over from the behaviour work on other suites. **Deliberately general.** No task names, no repo
names, no per-task numbers, and no claim about which suite showed what — those results are off limits
to this loop (§1), and a block that encoded them would be fitting to the judge. Treat this as a list
of things worth writing a procedure against, not as evidence:

- declaring the task done without ever checking the stated outcome from the outside;
- spending the budget on repetition: the same command again, a wedged screen, turns lost to responses
  the harness rejects;
- edit, break, restore, retry — churning with in-place stream edits and multi-line heredocs instead of
  one exact replacement;
- reverting a correct fix after misreading the evidence;
- deleting or emptying the inputs the task was given;
- opening with a large batch of commands and wedging the terminal before anything has been read.

Every seed block in `seed_blocks.json` targets one of these. When a reflection finds a new one, it
belongs here as a behaviour, phrased the same way.

## 2.2 The axes

The eight axes, all oriented so higher is better: `self_check`, `ran_test_after_edit`, `in_place`,
`no_sed_patch`, `no_repeat3`, `no_json_reject`, `no_input_delete`, `no_ctx_death`. Seven are binary
per trial; `no_json_reject` is a rate, because a binary version is constant-0 on this model (the
think-span preamble warning fires on essentially every turn and carries no signal, so only
schema-level rejections count).

## 3. The split (built once, already done)

`data/r2egym/jsc/gepa/gepa_split.py`, seed 20260921, over the 2,476-task Daytona tree
`tasks/r2egym-tt-daytona`. Stratified by base-pass@8 bucket (0 / 1–3 / 4–5 / 6–8) and repo.
Constraints: the surviving v2 `idval` tasks are forced into dev, `oodval` + `heldout` into test.

```
train 1476   dev 500 (127 forced)   test 500 (258 forced)   dev_mini 32   feedback 64
-> experiments/gepa/split_{train,dev,test,dev_mini,feedback}.txt, split.tsv, split_strata.md
```

`feedback` is 64 **train** tasks, stratified the same way and disjoint from dev and test by
construction. It is the reading set: the only trajectories the session ever opens. Dev supplies
numbers, feedback supplies traces, and the two never overlap.

Re-run it only to rebuild from scratch; it overwrites the lists and every ledger number becomes
incomparable. **Test's base pass@8 (0.223) is higher than dev's (0.173)** because the forced
`oodval` and always-solved `heldout` tasks land there. That shifts the absolute rate, not the
verdict: the final run is paired block-vs-control on the same tasks.

## 4. The loop, step by step

### 4.0a Pilot — the first live action, before any of this has ever run

Nothing below §4.0a has executed against a live server. The serve job has never started, no `harbor
run` has ever gone against a GEPA tree, and the runner's reap path has only run over an empty set.
**Do not open with 8 held nodes.** Prove the chain end to end on one.

```bash
ssh jupiter
export OMP_NUM_THREADS=1; cd /e/project1/transfernetx/lee27/code/snowball/gepa
NODES=1 HOURS=6 bash gepa_serve.sh up          # cost line: 1 node x up to 6 h
#   -> show it, get the go, then:
SUBMIT=1 NODES=1 HOURS=6 bash gepa_serve.sh up
bash gepa_serve.sh status                       # wait for 1 of 1 endpoints
bash gepa_queue.sh start

python3 gepa_tree.py --wave w0p --candidates seed_blocks.json --split feedback,dev_mini
bash gepa_queue.sh add-all w0p 1                # ctl + c000..c003
```

**Cost.** 5 arms × 96 tasks (64 feedback + 32 dev_mini) = **480 attempts ≈ 4.8 node-hours**, plus
~0.5 node-hours of engine load. On **one** node that is node-hours = wall-hours, so **about 5 hours**,
inside the 6-hour wall with little to spare. If it looks like overrunning, cut to `ctl` + `c000` (192
attempts, ~2 hours) and pilot with two arms. The runner needs no reconfiguring: shards per candidate
are auto, so one live endpoint means one shard.

**What the pilot has to prove**, in order, before §4.0 applies:

1. the serve job comes up and publishes its endpoint file;
2. `gepa_run.sh`'s preflight passes against a real server — served name, think markers, Daytona key;
3. a `harbor run` completes on the R2E-Gym Daytona tasks, which have never gone through this harbor
   path: the `setup_files/setup.sh` hook, snapshot resolution under `auto_snapshot`, the verifier;
4. the runner **reaps** — `RUN_DONE exit=` parsed, the candidate moved to `done/`, `runs.jsonl`
   appended — and then starts the second leg on the freed endpoint;
5. `gepa_score.py w0p --leg feedback` and `--leg dev_mini` produce sane numbers, and
   `gepa_ledger.py add` records the candidates;
6. `gepa_worst.py` / `gepa_dump.py` open a feedback trace and refuse a dev id;
7. **the runner survives a restart**: kill the `gepa_runner` tmux mid-wave and start it again with
   `gepa_queue.sh start`. The wave must resume — finished shards stay finished, the running ones keep
   going in their own tmuxes, and no shard is launched twice. (`gepa_run.sh` refuses a run name whose
   job dir exists, so a double launch fails loudly rather than corrupting the dir, but the point is
   that the runner should not try.)

Only when all seven hold does the 8-node job in §4.0 make sense. If the pilot dies, it cost one node.

### 4.0 Bring the servers up, once (only after the pilot passes)

```bash
ssh jupiter
export OMP_NUM_THREADS=1; cd /e/project1/transfernetx/lee27/code/snowball/gepa
bash gepa_serve.sh up                     # prints the COST line; submits nothing
#   -> show it to Luke, get the go, then:
SUBMIT=1 bash gepa_serve.sh up            # 8 nodes, one vLLM server each, EAGLE-3 draft
bash gepa_queue.sh start                  # the runner tmux on the login node
bash gepa_serve.sh status                 # endpoints up, queue depth, node-hours burned
```

The serve job self-cancels after `IDLE_MIN=20` minutes with an empty queue, so an abandoned session
releases its nodes by itself. That is also the failure mode to watch: **let the queue run dry for
20 minutes and the servers are gone**, and the next candidate pays the engine load again. Keep two
queued.

### 4.1 Seed the population (wave `w0`)

```bash
python3 gepa_tree.py --wave w0 --candidates seed_blocks.json --split feedback,dev,oodmini
bash gepa_queue.sh add-all w0 1           # control + c000..c003, no new go needed
bash gepa_queue.sh list
```

Each candidate runs as two ordered **legs**: `feedback` (64 train tasks) then `dev` (500). Feedback
lands first because it is small and it is the only thing you may read, so reflection can start while
the dev leg is still running. The runner admits `MAX_INFLIGHT=2` candidates, splits the current leg
into `SHARDS_PER_CAND=4` `harbor run`s (one per server, concurrency 32 each), and starts the next leg
on the endpoints the finished one just freed. Peak load is 2 × 4 × 32 = 256 concurrent Daytona
sandboxes, well under the ~1,000-per-user ceiling; the runner refuses a layout exceeding `SANDBOX_CAP`.

`seed_blocks.json` is the seed population: `c000` is **block A of P2O wave 0 verbatim**, the only
block with a confirmed paired lift (+0.144 [+0.096, +0.194] on dev120, 2026-09-13); `c001`–`c003`
each target one thing the wave-0 traces showed missing — checking the stated deliverable before
declaring done, a small first command batch and a time-boxed hypothesis, and never deleting inputs
or reverting blind.

### 4.2 Score the wave

Score each candidate **as it finishes**, while the others are still running. Do not wait for the
whole wave.

```bash
python3 gepa_score.py w0 --leg feedback    # what reflection ranks on (this wave's fresh 64)
python3 gepa_score.py w0 --leg oodmini     # the specialist check; run it BEFORE the ledger add
python3 gepa_score.py w0                   # the DEV leg: scores.csv + summary.md, what selection runs on
python3 gepa_ledger.py add --wave w0 --all --parent c000 \
  --candidates /e/fscratch/reformo/lee27/experiments/gepa/w0/cands.json \
  --scores /e/fscratch/reformo/lee27/experiments/gepa/w0/scores.csv \
  --oodmini-delta <from the specialist table> --accepted
```

The two legs are scored separately and never merged. The ledger records the **dev** numbers; the
feedback numbers exist only to point reflection at the right traces.

Read `experiments/gepa/w0/summary.md` before anything else. **Check the dropped-sample rate per arm
first** — if one arm dropped far more trials than another, the wave is not paired and its deltas are
void; say so and re-run rather than reading the numbers.

### 4.3 Pick a parent

```bash
python3 gepa_ledger.py front            # front membership and the sampling weights
python3 gepa_ledger.py sample --n 1 --seed <wave number>
```

The front is the candidates on at least one task's Pareto front. Sampling is proportional to
`front_wins`, which is GEPA's selection rule: a block that is best on a few tasks nobody else wins
stays in the gene pool even if its mean is unremarkable.

### 4.4 Reflect (this is the session's job)

```bash
python3 gepa_worst.py w0 c000 10                 # the FEEDBACK tasks to read, ranked, with dump commands
python3 gepa_dump.py w0 c000 <feedback task>     # one readable transcript
python3 gepa_dump.py w0 ctl <feedback task>      # the control on the same task, for the contrast
```

All of this is feedback-set only; a dev or test id is refused. Read at least six traces, and always
read the control on the same task. Then write 2–3 children into a new `<wave>_cands.json` — the
parent's block **plus one lesson**, never a rewrite. The reflection meta-prompt is §5.

The split of roles is the point: you form the hypothesis from feedback traces, and dev tells you
whether it was right. If a candidate looks better in the feedback traces but dev's numbers disagree,
**dev wins** — that is exactly the disagreement the two sets exist to surface.

### 4.5 Cheap gate on `dev_mini`

```bash
python3 gepa_tree.py --wave w1g --candidates .../w1_cands.json --split feedback,gate
bash gepa_queue.sh add-all w1g 1                 # no new go; keep the queue two deep
python3 gepa_score.py w1g --leg gate --parent c000 \
  --gate-axis-name c010=self_check --gate-axis-name c011=no_repeat3
```

The gate rule, fixed before the run and printed in the summary: **accept if paired (wins − losses)
≥ 2, or if the child's predicted axis gains ≥ +0.10 while the paired pass delta stays ≥ −0.02 and
pass does not fall on the tasks where that axis actually moved.** That last clause is the one that
catches detector-gaming: an axis can rise while pass drops on exactly the trials where the new
behaviour appeared, which means the behaviour made things worse. The summary prints that restricted
delta in its own column.
Pass rate alone does not decide — at 32 tasks it cannot. A gate showing no movement on the predicted
axis means the lesson did not land in behaviour, whatever pass did.

**Always pass `--gate-axis-name <cand>=<axis>`** — the axis §5 made you predict for that child. Without
it the gate takes the best of eight axes, which is a multiple-comparisons trap: with eight axes
something usually clears +0.10 by chance or by a detector artifact, and those rows come back marked
`weak` rather than `ACCEPT`. The `p2o6all` fixture shows the failure mode live — every arm appears to
gain ≈ +0.17 on `in_place` over block A, purely because A's replace-script style opens a *variable*
and the literal-path detector cannot see the write. A `weak` verdict is not an acceptance; either
name the predicted axis or judge the child on paired wins.

### 4.6 Full dev eval of the accepted children

```bash
python3 gepa_tree.py --wave w1 --candidates .../w1_accepted.json --split feedback,dev,oodmini
bash gepa_queue.sh add-all w1 1
python3 gepa_score.py w1 --parent c000
python3 gepa_ledger.py add --wave w1 --all --parent c000 --candidates .../w1/cands.json --scores .../w1/scores.csv --accepted
```

Then back to §4.3 for the next wave. Build and enqueue the **next** wave's gate before reading this
one's traces, so the servers have work while you read.

### 4.6a Judge the axis you just relied on

An accepted child was accepted partly because a detector said its predicted axis moved. Check that
the detector is telling the truth, on 8 traces, by hand:

```bash
python3 gepa_judge.py w1 c010 --axis self_check     # 8 feedback trials, balanced yes/no, with dump commands
#   ... read all eight, decide for yourself, then:
python3 gepa_ledger.py judge --cand c010 --axis self_check --agree 7 --of 8
```

Below 6 of 8 the ledger **freezes** that axis and `gepa_score.py` drops it from the Pareto front
until a later judge clears it. This is not ceremony: `in_place` cannot see a write through a
variable, so it reported the opposite of the truth for a whole wave. An axis that decides selection
and has never been read is a detector deciding the search.

### 4.6b Merge two front members

When two candidates sit on the front by winning **disjoint** dev tasks, they are likely teaching
different lessons, and a merge is worth one slot in the next wave. Do it by hand, by clause:

1. Confirm the disjointness: from each one's `scores.csv`, take the tasks where it is on the front
   and the other is not. Overlapping winners are the same lesson twice — do not merge those.
2. Write the union **clause by clause**, not paragraph by paragraph: take each block's distinct
   instructions and drop anything the two say twice in different words.
3. Cut to fit 400 tokens by dropping the weakest clause of each parent, never by compressing both
   into denser prose — a block the model cannot follow is worse than a shorter one.
4. Enter it as a child of the stronger parent, and predict the axis it should move like any other
   child. A merge that clears the gate on neither parent's axis is not a merge, it is a third block.

There is no script for this. A mechanical union produces contradictory instructions, which is worse
than either parent.

### 4.7 Final, once

```bash
# 1. CONFIRM first: the winner vs ctl on dev at k=2, fresh samples on the same tasks
bash gepa_final.sh confirm <best cand> .../<wave>/cands.json
bash gepa_final.sh verdict        # needs paired delta >= +0.05 with the 95 % CI strictly above 0

# 2. only then the sealed test set
bash gepa_final.sh build <best cand> .../<wave>/cands.json
#   -> confirm with Luke, then: CONFIRM=1 bash gepa_final.sh build ...
bash gepa_final.sh readout
bash gepa_serve.sh down                          # release the nodes when the loop is over
```

The confirm step exists because the winner was chosen by looking at dev many times: taking a maximum
over a dozen candidates on one 500-task set inflates it, and part of the margin is whichever
candidate drew friendlier noise. `gepa_final.sh build` **refuses** without a passing confirm marker.
A winner that cannot reproduce its own dev margin on fresh samples would not have survived the test
set either, and finding that out costs 2,000 dev attempts instead of the one held-out number.

## 5. The reflection meta-prompt

When the session reflects, it does exactly this:

**Read — feedback only.** `gepa_worst.py <wave> <parent> 10` ranks feedback tasks; `gepa_dump.py` on
at least six of them, plus the control's trace on the same tasks. Look for the *mechanism* of the
loss, not the outcome: which turn did the run go wrong, what did the model believe at that turn, and
what in the screen output it had already seen should have told it otherwise. Dev's numbers are
available and dev's traces are not; if you catch yourself wanting a particular dev trace to explain a
number, that is the fitting this rule exists to prevent.

**Write.** Two or three children. Each one is **the parent's block with one lesson added or one
clause sharpened**. Say in one sentence, in the wave note, which trace taught the lesson and what
behaviour you expect to move — that sentence is the prediction the gate tests.

**Forbidden in a block.** Any task id, repo name, file name, test name, dataset or benchmark name.
Any instruction to run or not run the repository's own test suite (on R2E-Gym those suites encode
the old behaviour; "run the existing tests" is a trap and the 2026-09-06 workflow probe showed the
clause changes how the model works without changing pass@8). Any claim about the grader. Any
lengthening past 400 tokens — if the lesson does not fit, drop a weaker clause to make room.

**Forbidden in the reflection.** Writing a child from memory of what the parent said. Writing a child
without reading a trace. Reading a dev or test trace, or trying to get at one some other way —
there is no override, by design. Reading a SWE-bench or Terminal-Bench 2 trace or per-task result.
Concluding from pass rate alone on `dev_mini`.

## 6. Selection and stop rules

**Selection.** Per task, a candidate is on the front if no other candidate is ≥ it on pass and on
every LIVE axis with at least one strict >. Frozen axes (§4.6a) are excluded. `front_wins` = tasks
whose front it is on. Parents are sampled ∝ `front_wins`, **excluding specialists**. A candidate is
removed from the pool only by being dominated everywhere or by being flagged specialist, never by a
mean.

**Stop when any of these holds:**
- the node-hour budget agreed for the loop is spent;
- two consecutive waves add no candidate to the front (no child wins a task no parent already won);
- every candidate on the front is flagged specialist — the search is buying dev with the band's
  repos and the prompt channel has nothing general left to give.

Then run §4.7 once and write the readout. A loop that stops on the second rule with the seed still
winning is a real result: it says the prompt channel is exhausted at this size, and it is worth
saying plainly rather than spending another wave.

## 7. Cost model

Measured off the two P2O probes on this exact layout (2026-09-20), not assumed:

- **≈ 100 attempts per GPU node-hour** steady state — 93 for `p2oAc_s0` (960 trials, first to last trial
  1.29 h on 8 nodes) and 111 for `p2o6all_s0` (2,880 trials, 3.24 h).
- **plus ≈ 0.5 h × nodes of engine startup** before the first trial — `p2oAc_s0` elapsed 1:48 against
  1:19 of eval. On 8 nodes that is a flat **4 node-hours per job**.

**On the standing serve job the startup is paid once, at session start, not per candidate.** That is the
whole reason for the redesign: it is 4 node-hours each time, and a loop that reflects between candidates
used to pay it over and over.

| work | attempts | node-hours (eval only) |
|---|---|---|
| feedback leg, one candidate | 64 | 0.7 |
| oodmini leg, one candidate | 32 | 0.3 |
| dev leg, one candidate | 500 | 5.0 |
| **one full-dev candidate, all three legs** | **596** | **6.0** |
| gate wave, parent + 2 children + ctl (feedback + gate) | 384 | 3.8 |
| a 4-arm full-dev wave, all legs | 2,384 | 23.8 |
| confirm, winner + ctl on dev at k=2 | 2,000 | 20.0 |
| final test-500, best + ctl | 1,000 | 10.0 |
| **the serve job itself** | — | **4.0 once**, then 8 per wall-hour held |

Marginal cost of one more full-dev candidate is **6.0 node-hours** across its three legs. The confirm
step is the single most expensive item in the loop at 20 node-hours; budget for it from the start
rather than discovering it at the end. Raise `k` only to settle a
comparison the axes already call close: `k=2` doubles the bill for about a 1.4× tightening of the interval.

The figure that actually governs spend is now **wall-clock held**, not attempts: 8 nodes cost 8 node-hours
for every hour the job is up, busy or not. So an idle queue is pure waste, and `gepa_serve.sh status` prints
the running total. Quote that number, not an estimate.

> Two earlier figures in this loop's history were wrong and are worth not repeating: 200 attempts per
> node-hour (it is ~100), and "one probe job per wave is fine" (the per-job engine load made it the
> dominant cost). Quote from this section.

## 8. Readout format

Status goes out in the `/news` shape (good / bad / resolved / what's left). The wave readout itself
is: the paired pass delta vs control with its bootstrap 95 % interval and the paired win/loss count;
the axis table with deltas; the front win counts; the dropped-sample rate per arm; and the spend in
node-hours. Lead with what changed about the decision, not with the method. The final readout adds
the test-set paired delta and the OOD-repo row — a block that only helps the in-distribution repos is
a band-specific trick, not a procedure, and should be reported as one.

## 9. Anchors

- scripts → `data/r2egym/jsc/gepa/` (mirror to Jupiter with `bash data/r2egym/jsc/gepa/gepa_sync.sh --go`;
  `code/snowball` is **not** a git checkout, `git pull` does nothing for it)
- split + ledger + queue + waves → `/e/fscratch/reformo/lee27/experiments/gepa/`
  (`queue/`, `running/`, `done/`, `runs.jsonl`, `endpoints/`, `logs/runner.log`)
- wave trees → `/e/fscratch/reformo/lee27/tasks/gepa-<wave>/`
- harbor run dirs → `/e/data1/mmlaion/lee27/experiments/gepa_jobs/<wave>_<cand>_s<i>/`
- the serving stack, reused not rewritten → `data/tb2/jupiter/serve_snowball.sbatch` (the per-node server),
  `run_tb2.sh` (the harbor-run preflight and tmux convention), `rst_policy.yaml` (the policy shape)
- the setup-files-hook harbor, which R2E-Gym Daytona tasks need → `code/harbor-hook/src`
- the suffix method and the byte-identity check → `data/r2egym/jsc/p2o/p2o_trees.py`
- the FALLBACK launcher (one SkyRL probe job per wave) → `gepa_wave.sh`, plus
  `data/r2egym/jsc/make_tt_wave.py --daytona` and the cluster's `code/snowball/draftify_probe.sh`
  (EAGLE-3; policy 2026-09-12 is that every job uses the draft, so numbers from a no-draft run are not
  paired with anything here)
- detector provenance → `p2o/p2o_adherence.py` (message parser), `behaviour_scans_20260919/swe_evals/scan.py`
  (write targets, repeats, rm, warnings), `behaviour_scans_20260919/tb2_why/selfcheck.py` (self-check)
- note → `ai_memory/active/snowball-r2egym/research/2026-09-21_gepa_agentic_loop.md`
