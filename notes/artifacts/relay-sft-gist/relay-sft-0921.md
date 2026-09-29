# Relay SFT on Grug Datakit 09-21

**Training 09-21 on relay traces beats training it on Qwen-alone traces on all three agent benchmarks, most clearly on SWE-bench, and ties on held-out CalibForge.**

- SWE-bench Verified: relay 31 % vs Qwen-alone 18 % vs 09-21 9 % (relay gains 13 tasks and loses 2 against Qwen-alone)
- TB2.1 11 % vs 5 % and OpenThoughts-TBLite 18 % vs 12 %, the same direction; held-out CalibForge 10 % vs 9 %
- SFT fixed 09-21's reply format (the harness rejected 89 % of its turns, 15–20 % after SFT), and the models now run out of time because they write twice as many tokens

**Outline**
1. [Relay traces make a better SFT set than Qwen-alone traces on the agent benchmarks](#1)
2. [Training on 09-21's own autofixed actions adds nothing](#2)
3. [SFT fixed the reply format, and the new limit is the clock](#3)
4. [Next](#4)

![Pass rates](FIG_RESULTS)

<a id="1"></a>
## 1. Relay traces make a better SFT set than Qwen-alone traces on the agent benchmarks

**The relay arm solves more on both agent benchmarks, and on held-out CalibForge the two arms tie.** Both arms train 09-21 on the same recipe and the same number of rows: 808 passes and 808 failures on CalibForge tasks. In the relay rows 09-21 plays the early turns and Qwen3.8 takes over (09-21's turns are masked, only Qwen's are trained). In the Qwen-alone rows Qwen plays every turn.

| | held-out CalibForge (300) | TB2.1 (88) | SWE-bench Verified (100) | OpenThoughts-TBLite (100) |
|---|---|---|---|---|
| 09-21, before SFT | 6.2 % | 5.9 % | 9.4 % | not run |
| A, relay | **10.1 %** | **11.3 %** | **30.9 %** | **18.1 %** |
| B, Qwen alone | 9.0 % | 4.7 % | 17.9 % | 12.4 % |
| C, relay + autofix loss | 9.1 % | 9.3 % | not run | not run |

- **SWE-bench, same 93 tasks.** Relay gains 13 tasks and loses 2, +11.8 points (the range consistent with the data runs +4.3 to +19.4). Against 09-21 the relay arm gains 24 tasks and loses 4 (+21.3 points), the Qwen-alone arm 11 against 4 (+7.6).
- **OpenThoughts-TBLite, same 92 tasks.** Relay gains 11 and loses 4, +7.6 points (+0.0 to +16.3).
- **TB2.1, same 78 tasks.** Relay gains 6 and loses 1, +6.4 points (+0.0 to +12.8). A second TB2.1 run of both models, made for a different test, gives +7.3 points (+1.2 to +14.6, 7 gained against 1 lost).
- **Held-out CalibForge, same 277 tasks.** +1.1 points (−2.9 to +5.1), a tie. It is the only benchmark where the relay arm clearly beats 09-21 (+4.3 points, +0.4 to +8.3).

> I fixed the comparison before any result: the relay arm would win only if its held-out gain over Qwen-alone cleared zero. It does not, so under that rule the relay does not win. The three agent benchmarks all point the other way, SWE-bench most clearly. SWE-bench and TBLite were added after the pre-registered comparison and have no C run. 09-21 on SWE-bench scores 18 % under the older v0.1 policy (32k/8k tokens, 50 minutes per task) and 9.4 % here (65k/16k, 30 minutes), so the old numbers are not comparable with these.

<a id="2"></a>
## 2. Training on 09-21's own autofixed actions adds nothing

**Arm C trains the 09-21 actions the harness had auto-corrected (1.08 M extra trained tokens), and it scores no better than A.** C − A is −1.5 points on held-out (−5.8 to +2.9) and −1.3 on TB2.1 (−8.9 to +6.3). The relay arm should keep every 09-21 turn masked.

<a id="3"></a>
## 3. SFT fixed the reply format, and the new limit is the clock

**Before SFT the harness rejected almost every 09-21 reply. After SFT the models reply correctly but write so much that most tasks hit the 30-minute limit.**

![Why pass rates stay low](FIG_WHY)

- **Format.** Terminus-2 rejected 89 % of 09-21's turns on TB2.1 (mostly "missing required fields" and "no valid JSON"). After SFT it rejects 15–20 %, mostly malformed JSON keys. Each rejected turn is a turn spent doing nothing.
- **Length.** The SFT models write a median of 120–146k tokens per TB2.1 task against 56k for 09-21, and 62–73 % of it is thinking (33 % for 09-21). Generation runs at about 150 tokens/s per task, so that is 13–16 minutes of pure writing, before any command runs.
- **Timeouts.** 60–68 of 88 TB2.1 tasks hit the 30-minute clock after SFT, against 29 for 09-21.
- **Where the thinking comes from is open.** The trained thinking in the rows is short (median about 150 tokens per turn, 90 % under 700), yet the models think about 2,500 tokens per turn. A lower thinking cap in training would change under 1 % of the trained turns.

> On held-out CalibForge, which runs without Terminus-2's summarization, 40 % of SFT runs end by filling the 65k context instead: every earlier turn's thinking stays in the history, while the training rows cut it to 1,000 tokens. On TB2.1 summarization keeps the context in bounds, and cutting the history or allowing 128k changed nothing (within ±1.3 points).

<a id="4"></a>
## 4. Next

- **Scale the relay set.** Each arm used about half of its eligible rows: failures bind (808 of 919 relay failures used), about 1,400 passes per arm are unused, and the relay pool dropped 3,809 rows by allowing at most two per task.
- **Train on relay and Qwen-alone rows together** and compare with relay alone. This is running: 3,232 rows, same recipe, trained 2026-09-28, evals on 09-29.
- **Find where the long thinking comes from** before scaling, since the clock now decides most failures.

---

<sub>**Setup.** Base Grug Datakit 09-21 (the 09-21 import, router bias frozen). LR 3e-4, cosine to 10 % with 5 % warmup, 3 passes over the packed rows (147 steps for B, 246 for A and C), 16 × 65,536 on 4 Jupiter nodes, loss only on the rows' trained tokens (checked token for token against the cache the trainer reads). Evals at 65,536 input and 16,384 output tokens, Terminus-2, harbor 761fb516, one try per task, 16 concurrent per GH200 node for TB2.1, SWE-bench and TBLite (each served on its own node; per-task generation speed within 15 % across compared models) (the 09-24 Marin policy with only the token limits changed), 100 concurrent on one node for held-out CalibForge. TB2.1 is terminal-bench-2 @ 53ff2b8. Ranges are 95 % bootstrap intervals over the tasks both models scored; pass-rate bars show Wilson intervals. About 30 node-hours for training and the pre-registered evals.</sub>
