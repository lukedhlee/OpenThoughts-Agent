# Relay SFT: handoff (2026-10-01)

**What this is.** A recipe that lifts Grug 67B-A2B 09-21 on agent benchmarks by fine-tuning it on *relay* traces: the
student model works a terminal task until it hands over, and Qwen3.8-27B finishes it; only Qwen's turns are trained.
Relay traces beat Qwen-alone traces of the same tasks by +6 (TB2.1), +22 (SWE-bench) and +10 (TB-lite) points in a
matched test, and they are the only data we found that moves TB2.1. Write-up with every arm:
https://gist.github.com/lukedhlee/8669f5b6f97e4e3bae72344a8fd51f4b

## Artifacts

| what | where |
|---|---|
| base model | `open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21` |
| best models | `laion/snowball-67b-a2b-relay-sft-acont-step999` (H9), `laion/snowball-67b-a2b-relay-sft-allkimi-step1203` (H8) |
| relay traces (readable, labelled) | `laion/calibforge-relay-traces`: splits `used` (H8's rows), `unused_clean`, `unused_filtered`; `kimi_trials.txt` |
| Kimi SWE-smith traces | `open-athena/Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k` (we use the 4,392 trials in `kimi_trials.txt`) |

## Code (pinned)

| piece | repo, branch, commit |
|---|---|
| relay generation, rendering, matching, export, evals | `lukedhlee/OpenThoughts-Agent` `lukedhlee/vista-moe-grpo-30b` (this repo), `data/relay/` + `data/r2egym/horizon/sft/` |
| SFT training (Levanter / marin), stages, HF import with frozen router bias | `lukedhlee/marin` `lukedhlee/horizon-snowball-sft-0921` @ 8afe028c7, `experiments/june_tpu_67b_a2b/moe/` |
| harbor with the relay hooks (Terminus-2 relay, setup_files) | `marin-community/harbor` `lukedhlee/terminus2-relay` @ 89098635 |
| RL (GRPO on R2E-Gym, separate track) | `marin-community/MarinSkyRL` @ a03b2773; Horizon launcher in `data/r2egym/horizon/rl/` |

## The loop, end to end

### 1. Generate relay rollouts (Daytona sandboxes + vLLM serve job)

`data/relay/pilot/run_pilot.sh` drives everything: one router per arm (`relay_repair` = student then Qwen; `control` =
Qwen alone), harbor jobs on Daytona, gates, readout. On a Slurm cluster without internet on compute nodes
(Horizon), `data/relay/horizon/` wraps it: `serve_submit.sh` (N nodes: student + Qwen servers), `launch_driver.sh`
(a 1-node driver job reaching Daytona through login-side ssh tunnels). Full doc: `data/relay/horizon/DRIVER.md`,
`SERVING.md`.

```bash
S=$(STUDENT_MODEL=<student HF dir> bash data/relay/horizon/serve_submit.sh 8 3 08:00:00 srv_<name>)   # 3 student + 5 Qwen nodes
ARMS="relay_repair" N_ATTEMPTS=2 CONC=56 NODES=8 CAP_NODE_H=110 TREE=<harbor task tree> TASK_LIST=<ids> SHUFFLE_SEED=1 \
  bash data/relay/horizon/launch_driver.sh $S <run name>
```

Relay settings we use: `CLOCK=repair`, 32k student context budget, 64k row limit, Qwen at 128k, teacher format guard,
verify note. A 6-shard run (6 x (8 serve + 1 driver) nodes, ~770 concurrent sandboxes) does ~10k episodes in 2-3 h.
Task pools: CalibForge (3 Daytona snapshots; used up), TMax strict (`data/tmax/`, one shared snapshot + setup replay,
no-op gate `data/tmax/daytona/gate.py`).

### 2. Turn rollouts into SFT rows

```bash
python data/relay/sft/hz_pool_census.py --arm relay_repair --runs <run dirs> --out census.jsonl   # labels + QA tags
python data/relay/sft/render_think_limit.py --render-dir <OTA@d0ddd237>/data/relay/sft --run-dir <run> --arm relay_repair \
  --manifest <eligible rows> --tokenizer-dir <09-21> --think-limit 16384 --strip-copied-markers \
  --terminus-parser <harbor>/harbor/agents/terminus_2/terminus_json_plain_parser.py --autofix-loss none --out rows.jsonl
```

A row is `{sid, ids, loss, n_tokens, fits, turns}`: token ids (09-21 tokenizer) and a 0/1 loss mask; only Qwen's turns
carry loss. Filters (identical for every arm): drop weak timeouts and episodes whose verifier never ran, drop `leak`
(Qwen mentions the verify note / "previous agent"), `hunt` (commands read grader files) and `canary`, keep rows within
65,536 tokens, at most 2 rows per task. `build_hz_arms.py` / `build_matched_arms.py` assemble arms;
`export_relay_dataset.py` + `split_relay_dataset.py` produce the readable HF dataset.

### 3. SFT (marin / Levanter)

A stage per arm in `vista_snowball_chat.py` (`_relay_stage`: prerendered ids + loss, no chat template, frozen router
bias, 09-21 tokenizer pinned). The driver for one arm on Horizon, rows to export:

```bash
ARM=<stage suffix> ROWS=<rows.jsonl> DATASET_REVISION=<stage pin> [SNOWBALL_INIT=<native init>] WALL=08:00:00 \
  bash data/r2egym/horizon/sft/relay_sft_arm.sh
```

It converts rows to parquet and counts packs (1 node), builds the cache, trains (4 nodes, 16 x 65,536 tokens per step,
~7 s/step), and exports one HF checkpoint per pass. Recipe: LR 3e-4, warmup 5 %, one cosine over 3 passes, seed 0,
init = 09-21 imported with `pending_qb_betas = -router_bias` and the router bias frozen. To continue from a fine-tuned
export, import it first (`horizon_snowball_import.sbatch` with `SNOWBALL_PENDING_FROM_BIAS=true`) and pass the init.

### 4. Evaluate

```bash
MODELS='<tag>=<HF export dir>' REPS=1 GROUPED=1 bash data/r2egym/horizon/sft/eval_sft.sh   # one 5-node unit per rep
python data/r2egym/horizon/sft/eval_readout.py --tags <tag> [<other>] --day '2026*'         # pass@1 ± SE, paired diffs
```

One unit runs TB2.1 (88), SWE-bench Verified random-100 and OpenThoughts-TBLite (100) once each under the 65k / 16k
policy (Terminus-2, 16 concurrent trials, one try per task); we run 3 units (reps) per model. `eval_dispatch.sh` starts
units as arms finish, within a Daytona sandbox cap.

## What we learned (for the next person)

- Relay rows > Qwen-alone rows, matched per task (gist "Relay itself is what helps"). Qwen-alone rows barely move 09-21.
- Kimi SWE-smith carries SWE-bench and TB-lite; on SWE it substitutes for relay, on TB2.1 it does not.
- More relay rows help TB2.1 (A 12.1 -> H1 15.7 -> H8 18.8 %); extra epochs do not (H8 epoch 3 vs 2: +1-3 points, n.s.).
- Starting point matters little: H8 (from 09-21) and H9 (continued from arm A) tie.
- 30 GRPO steps on R2E-Gym on top of H9 (RL track): TB2.1 +4.2, TB-lite +5.4, SWE -3.5 (none significant yet).
- In flight on 10-01: H10 (every clean CalibForge relay row + Kimi) and TMax relay vs Qwen-alone (3,031 gate-passing
  tasks), SFT arms T1 / T1q / T2; results in the gist when they land.

## Gotchas

- Daytona: the eval org allows ~1,000 concurrent sandboxes and 40 custom snapshots (hard cap). Never build snapshots
  casually; `auto_snapshot` mints one for any task whose environment hash has none.
- Relay drivers can stall-abort after every trial has finished (a harbor process that never exits): count results
  before treating `ABORT` as a failure.
- TACC Horizon's submit filter sums nodes x time limit over all queued + running jobs against the SU balance: keep
  wall limits tight.
- Rendering fixes the 09-21 chat format; a row's `turns[k]` is the k-th assistant message.
