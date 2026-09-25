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

**Pool: 2,043 tasks** (`data/relay/pilot/full_pool.txt`). That is the 2,457 Daytona-covered CalibForge tasks, minus
the 300 held-out tasks, minus 114 tasks with agent budgets above 1,800 s. There are no duplicate instructions. The
pilot's 100 tasks are included.

**Phases.**
- **Phase 1**, all 10 nodes busy: control, 1 rollout per task (400 concurrent, 50 per Qwen node), and relay_repair,
  1 rollout per task (200 concurrent, 100 per student node).
- **Phase 2**: relay_repair, 2 more rollouts on every task control solved in phase 1. It starts once phase 1's relay
  job has at most 20 trials running, so the student nodes stay at about 100 agents.
- **Releasing nodes.**
  - The Qwen servers run as two jobs: *core* (2 × 09-21 + 4 × Qwen, 6 nodes) and *burst* (4 × Qwen).
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

**Launch rule** (Luke's conditional go, 15:35 PT). `full_decide.py` must say LAUNCH:
- run 3 passed every harness check (H0–H5);
- run 3 passed every scale-up check (S1–S6);
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

## How to launch (Jupiter login node; only on LAUNCH)

```bash
C=/e/project1/transfernetx/lee27/code
git -C $C/OpenThoughts-Agent fetch fork lukedhlee/vista-moe-grpo-30b
git -C $C/OpenThoughts-Agent worktree add --detach $C/ota-relay-full FETCH_HEAD   # never re-checkout a worktree a live driver reads
P=$C/ota-relay-full/data/relay/pilot
LIST=$P/full_pool.txt bash $P/stage_tree.sh /e/fscratch/reformo/lee27/tasks/calibforge_full2043
$C/envs/snowball-v2/bin/python $P/full_decide.py --run3 /e/fscratch/reformo/lee27/experiments/relay/pilot/runs/relay_run3_20260925 > /tmp/decision.json
# LAUNCH -> core_time_h / burst_time_h from the decision
CORE=$(RELAY_PILOT_DIR=$P N_STUDENT=2 sbatch --parsable --export=ALL --nodes=6 --time=<core_time> --job-name=relay_full_core $P/serve_relay.sbatch)
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
