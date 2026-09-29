# Relay SFT on 09-21: results (three arms + 09-21)

## Result (2026-09-28, 17:58 PT)

**Under the pre-registered rule, A does not win: the held-out A − B gain is +0.011 with a 95 % CI of [−0.029, +0.051],
which includes 0.** The rule then says to scale Qwen-alone traces. Read plainly, the data give no evidence that
Qwen-alone data is better, and a lean toward the relay on TB2.1 (A − B +0.064 [+0.000, +0.128]: A solved 6 tasks B
missed, B solved 1 that A missed). The rule's other conditions: TB2.1 is not worse for A, and the SFT did not fail,
because A beats 09-21 on held-out (+0.043 [+0.004, +0.083]); B (+0.028) and C (+0.025) point the same way without
clearing 0.

- **Training on the autofixed 09-21 actions (C) does not help.** C − A is −0.015 [−0.058, +0.029] on held-out and
  −0.013 [−0.089, +0.063] on TB2.1.
- **Why pass rates stay near 10 %.** SFT changed how the models fail more than how often they finish. On held-out,
  context overflow went from 38 to 118–127 of 300 runs, timeouts from 186 to 117–123, and wrong "done" claims from 49 to
  16–22. The SFT models think far longer per turn (thinking is 63–72 % of their output, against 27 % for 09-21). Earlier
  turns' thinking stays whole in the history at eval, while the training rows cut it to 1,000 tokens, so about 20 turns
  fill 65k. Generation is the clock's bottleneck (about 80 % of each run's wall time is waiting on the model, at about
  60 tokens/s per task), so 30 minutes buys about 15 turns.
- **Diagnostics running (outside this pre-registration, so they do not change the verdict).** TB2.1 with the 1,000-token
  history cut at eval (A and B), and A at 128k context. SWE-bench Verified random-100 for A and B was added the same
  evening. Their results go below when they land.

| model | held-out 300: pass rate [95 % CI] (n) | TB2.1: pass rate [95 % CI] (n) |
|---|---|---|
| 09-21 (before) | 0.062 [0.040, 0.096] (291) | 0.059 [0.025, 0.130] (85) |
| A relay, 09-21 turns masked | 0.101 [0.071, 0.141] (287) | 0.113 [0.060, 0.200] (80) |
| B Qwen alone | 0.090 [0.062, 0.129] (289) | 0.047 [0.018, 0.115] (85) |
| C relay + autofix loss | 0.091 [0.063, 0.130] (286) | 0.093 [0.048, 0.173] (86) |

| paired (X - Y, tasks both usable) | held-out diff [95 % CI] (tasks) | TB2.1 diff [95 % CI] (tasks) |
|---|---|---|
| A-B | +0.011 [-0.029, +0.051] (277) | +0.064 [+0.000, +0.128] (78) |
| C-A | -0.015 [-0.058, +0.029] (274) | -0.013 [-0.089, +0.063] (79) |
| A-0921 | +0.043 [+0.004, +0.083] (278) | +0.063 [-0.013, +0.139] (79) |
| B-0921 | +0.028 [-0.011, +0.067] (282) | -0.012 [-0.072, +0.048] (83) |
| C-0921 | +0.025 [-0.014, +0.065] (279) | +0.036 [-0.036, +0.119] (84) |

**Decision rule:** A does not win: scale Qwen-alone traces

TB2.1 CIs here are Wilson (paired_eval.py); summarize_tb2.py prints a normal-approximation CI for the same counts.
Unscored (infrastructure) trials on TB2.1: 09-21 3, A 8, B 3, C 2 of 88 (B and C after one recovery pass).
Held-out unusable: 09-21 9, A 13, B 11, C 14 of 300.

**Cost.** 29.75 node-h for the pre-registered job: data prep and cache checks 0.68, training 9.44 (A 3.49, B 2.31,
C 3.64), 9 exports 2.84 and 9 held-out NLL scores 1.01, held-out evals 4.84, TB2.1 10.94 (09-21 1.81 plus a
0.12 cancelled TB2.0 start; A 2.65, B 3.19, C 3.16, B and C with a recovery pass). Diagnostics and SWE-bench are extra,
within the 48 cap.

**Artifacts.** Final exports: `/e/data1/mmlaion/lee27/snowball-sft/experiments/snowball-relay-sft/relay_{relay,qwen,relayaf}/lr3e-4-sched3/export-step{246,147,246}-hf-bf16`
(A, B, C). Held-out runs: `relay/pilot/runs/heldout6516_{0921,A,B,C}_20260928`. TB2.1 job dirs:
`tb2_jobs/tb21_6516_{0921,A,B,C}_20260928`. Comparison: `sft_v2/results.json`. Running log:
`/e/fscratch/reformo/lee27/experiments/relay/sft_v2/STATUS.md`.

The pre-registration below was written and committed on 2026-09-28 ~11:53 PT,
before any training job or eval of this comparison was submitted (only the two data-prep jobs had been).

## Pre-registration (fixed before any result)

**Models under test.** Four models, each served the same way.
- **09-21.** `grug-datakit-sft-20260921` itself, the before point.
- **A, relay, 09-21 turns masked (main arm).** The relay final_v2 manifest re-rendered with `--autofix-loss none`
  (think cap 16k, copied markers stripped, render.py at d0ddd237). Marin stage `relay_relay`.
- **B, Qwen alone.** The baseline final_v2 file as it is. Stage `relay_qwen`.
- **C, relay + autofix loss.** The current relay final_v2 file as it is (autofixed 09-21 actions trained). Stage
  `relay_relayaf`.
- Every arm has the same recipe: the 09-21 import init, router bias frozen at 09-21's non-zero value, LR 3e-4 with the
  Bespoke schedule (cosine to 10 %, 5 % warmup), seed 0, 3 real passes (steps = ceil(packs / 16) × 3), 16 × 65,536 on
  4 nodes, packing with cross-row attention blocked, loss taken from the rendered rows.
- The checkpoint under test is the **final pass-3 export** of each arm. Earlier passes are exported only to watch for a
  late regression; they do not enter the decision.

**Held-out 300 CalibForge.** The same driver as `heldout0921_20260925`.
- `run_pilot.sh` with `RUN_KIND=heldout`, tree `calibforge_heldout300`, 300 tasks, `ARMS=student_only`, 1 serve node
  (`serve_relay.sbatch`, `N_STUDENT=1`), `CONC=100`, `CAP_NODE_H=2.5`.
- 65,536 input and 16,384 output tokens. `CLOCK=wall`, which is harbor's own 1× agent budget per task.
- Terminus-2 strict. Harbor `harbor-terminus2-relay` at 89098635 for all four models.
- One trial per task.

**TB2.1.** The current Marin TB2 policy with only the token limits changed, on the TB2.1 tasks (Luke 2026-09-28, set
at 12:03 PT before any TB2 result).
- TB2.1 is laude-institute/terminal-bench-2 at 53ff2b8 ("Various task fixes for TB2.1"). It changes 7 files over 2.0 and no
  task image. Local tree: `/e/fscratch/reformo/lee27/tasks/terminal_bench_2_1` (`TASKS=` in the chain).
- `tb2_marin_policy_0924_65k16k.yaml`: harbor-p0924 at 761fb516, 65,536 input and 16,384 output tokens, 1,800 s agent
  budget, Daytona, 16 concurrent.
- 89 tasks (train-fasttext excluded, as in every 0924 run: its image is gone; counted as an infra loss), one trial per task. `serve_snowball.sbatch` on 1 node (EAGLE-3 draft, DP4), thinking at the model default.

**Analysis.** `data/relay/sft/paired_eval.py` as committed in f9aa523d, unchanged.
- Pass rates carry a Wilson 95 % CI. Paired differences use the tasks both models scored, with a bootstrap 95 % CI over
  tasks (10,000 resamples, seed 20260928).
- **Decision.** A wins if the held-out paired A − B CI lies above 0 and TB2 is not worse (the paired TB2 A − B CI does
  not lie entirely below 0). Then the relay is worth scaling. Otherwise, scale Qwen-alone traces.
- **Sanity.** An arm beats 09-21 if its held-out paired gain over 09-21 has a CI above 0. If neither A nor B beats 09-21,
  the SFT failed and A vs B says nothing.
- **Also reported.** C − A (does training the autofixed 09-21 actions help or hurt), C − B, and each arm vs 09-21.

**Runs and retries.**
- A failed step (job crash, serve death, harness errors over 10 % of trials) is retried once. A retry of an eval is merged
  per task, the later run winning, as `paired_eval.py` does. A second failure stops that step and is reported as FAILED.
- The A, B and C evals run at the same time, so the decision is not confounded by load drift. The 09-21 evals may run
  earlier.
- Budget ~43 node-hours, hard cap 48.
