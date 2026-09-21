---
name: gepa-prompt-loop
description: >-
  Run a GEPA-shaped prompt-evolution loop for Snowball on R2E-Gym, agentically — THIS session is the
  reflection LLM, there is no GEPA framework. A population of general terminal-agent guidance blocks
  (≤ 400 tokens, procedure only) is appended to each task's instruction.md as a strict suffix; each
  candidate is scored by a Daytona probe of the Stage-3 base on a 500-task dev split; selection is a
  per-task Pareto front over pass plus eight deterministic behaviour axes; the session reads the
  losing traces and writes children. Use when asked to evolve / optimise / search the agent prompt,
  run a GEPA wave, reflect on a candidate's dev traces, or gate a child block. Scripts:
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
- **Dev is open, test is sealed.** The session may read ANY dev trace, as many as it likes. The test
  split is scored exactly once, by `gepa_final.sh`, at the end. Looking at a test trace, or scoring
  the test set twice, destroys the only held-out number the loop produces.
- **TB2 is never touched by this loop.** Terminal-Bench 2 stays the external check on whether any of
  this transferred. A block tuned against TB2 tells us nothing.
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
| **selection** | per-task Pareto front over (pass, the eight axes); parents sampled by how many tasks they win |
| **reflection** | this session reads the parent's worst dev traces and writes 2–3 children with the lesson added |
| **cheap gate** | child vs parent on `dev_mini` (32 tasks), paired wins plus axis deltas — pass rate alone cannot decide at this n |
| **full eval** | an accepted child is re-scored on all 500 dev tasks and entered in the ledger |
| **final** | the best block and the control, once, on the 500-task test split, paired, plus the OOD-repo check |
| **the compute** | one standing serve job (8 nodes, one vLLM server each) plus a login-node queue. A candidate is N `harbor run`s against those servers on Daytona, never its own Slurm job |

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
train 1476   dev 500 (127 forced)   test 500 (258 forced)   dev_mini 32
-> experiments/gepa/split_{train,dev,test,dev_mini}.txt, split.tsv, split_strata.md
```

Re-run it only to rebuild from scratch; it overwrites the lists and every ledger number becomes
incomparable. **Test's base pass@8 (0.223) is higher than dev's (0.173)** because the forced
`oodval` and always-solved `heldout` tasks land there. That shifts the absolute rate, not the
verdict: the final run is paired block-vs-control on the same tasks.

## 4. The loop, step by step

### 4.0 Bring the servers up, once

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
python3 gepa_tree.py --wave w0 --candidates seed_blocks.json --split dev
bash gepa_queue.sh add-all w0 1           # control + c000..c003, no new go needed
bash gepa_queue.sh list
```

The runner admits `MAX_INFLIGHT=2` candidates at a time, splits each into `SHARDS_PER_CAND=4`
`harbor run`s (one per server, concurrency 32 each), and moves each finished candidate to `done/`
with its run dirs. Peak load is 2 × 4 × 32 = 256 concurrent Daytona sandboxes, well under the
~1,000-per-user ceiling; the runner refuses a layout that would exceed `SANDBOX_CAP`.

`seed_blocks.json` is the seed population: `c000` is **block A of P2O wave 0 verbatim**, the only
block with a confirmed paired lift (+0.144 [+0.096, +0.194] on dev120, 2026-09-13); `c001`–`c003`
each target one thing the wave-0 traces showed missing — checking the stated deliverable before
declaring done, a small first command batch and a time-boxed hypothesis, and never deleting inputs
or reverting blind.

### 4.2 Score the wave

Score each candidate **as it finishes**, while the others are still running. Do not wait for the
whole wave.

```bash
python3 gepa_score.py w0
python3 gepa_ledger.py add --wave w0 --all --parent c000 \
  --candidates /e/fscratch/reformo/lee27/experiments/gepa/w0/cands.json \
  --scores /e/fscratch/reformo/lee27/experiments/gepa/w0/scores.csv --accepted
```

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
python3 gepa_worst.py w0 c000 10                 # the tasks to read, ranked, with the dump commands
python3 gepa_dump.py w0 c000 <task>              # one readable transcript
python3 gepa_dump.py w0 ctl <task>               # the control on the same task, for the contrast
```

Read at least six traces, and always read the control on the same task. Then write 2–3 children into
a new `<wave>_cands.json` — the parent's block **plus one lesson**, never a rewrite. The reflection
meta-prompt is §5.

### 4.5 Cheap gate on `dev_mini`

```bash
python3 gepa_tree.py --wave w1g --candidates .../w1_cands.json --split dev_mini
bash gepa_queue.sh add-all w1g 1                 # no new go; keep the queue two deep
python3 gepa_score.py w1g --parent c000 \
  --gate-axis-name c010=self_check --gate-axis-name c011=no_repeat3
```

The gate rule, fixed before the run and printed in the summary: **accept if paired (wins − losses)
≥ 2, or if the child's predicted axis gains ≥ +0.10 while the paired pass delta stays ≥ −0.02.**
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
python3 gepa_tree.py --wave w1 --candidates .../w1_accepted.json --split dev
bash gepa_queue.sh add-all w1 1
python3 gepa_score.py w1 --parent c000
python3 gepa_ledger.py add --wave w1 --all --parent c000 --candidates .../w1/cands.json --scores .../w1/scores.csv --accepted
```

Then back to §4.3 for the next wave. Build and enqueue the **next** wave's gate before reading this
one's traces, so the servers have work while you read.

### 4.7 Final, once

```bash
bash gepa_final.sh build <best cand> /e/fscratch/reformo/lee27/experiments/gepa/<wave>/cands.json
#   -> confirm with Luke, then: CONFIRM=1 bash gepa_final.sh build ...
bash gepa_final.sh readout
bash gepa_serve.sh down                          # release the nodes when the loop is over
```

## 5. The reflection meta-prompt

When the session reflects, it does exactly this:

**Read.** `gepa_worst.py <wave> <parent> 10`, then `gepa_dump.py` on at least six of those tasks, and
the control's trace on the same tasks. Look for the *mechanism* of the loss, not the outcome: which
turn did the run go wrong, what did the model believe at that turn, and what in the screen output it
had already seen should have told it otherwise.

**Write.** Two or three children. Each one is **the parent's block with one lesson added or one
clause sharpened**. Say in one sentence, in the wave note, which trace taught the lesson and what
behaviour you expect to move — that sentence is the prediction the gate tests.

**Forbidden in a block.** Any task id, repo name, file name, test name, dataset or benchmark name.
Any instruction to run or not run the repository's own test suite (on R2E-Gym those suites encode
the old behaviour; "run the existing tests" is a trap and the 2026-09-06 workflow probe showed the
clause changes how the model works without changing pass@8). Any claim about the grader. Any
lengthening past 400 tokens — if the lesson does not fit, drop a weaker clause to make room.

**Forbidden in the reflection.** Writing a child from memory of what the parent said. Writing a child
without reading a trace. Reading a test trace. Concluding from pass rate alone on `dev_mini`.

## 6. Selection and stop rules

**Selection.** Per task, a candidate is on the front if no other candidate is ≥ it on pass and on all
eight axes with at least one strict >. `front_wins` = tasks whose front it is on. Parents are sampled
∝ `front_wins`. A candidate is removed from the pool only by being dominated everywhere, never by a
mean.

**Stop when either holds:**
- the node-hour budget agreed for the loop is spent, or
- two consecutive waves add no candidate to the front (no child wins a task no parent already won).

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
| `dev_mini` gate, parent + 2 children + ctl | 128 | 1.3 |
| dev-500, one candidate | 500 | 5.0 |
| dev-500, 3 candidates + ctl | 2,000 | 20.0 |
| final test-500, best + ctl | 1,000 | 10.0 |
| **the serve job itself** | — | **4.0 once**, then 8 per wall-hour held |

Marginal cost of one more candidate on dev at k=1 is **5 node-hours**. Raise `k` only to settle a
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
