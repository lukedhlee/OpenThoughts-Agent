# Relay on Horizon: rollouts, SFT rows, training and evals

Relay makes SFT data for the Grug / Snowball student (67B-A2B). The student works a terminal task until it hands the
episode over (it claims it is done, loops, stops making progress, or its context passes a budget), and Qwen3.8-27B
finishes the episode. Each finished episode becomes one training row in which only Qwen's turns carry loss. The control
arm runs Qwen alone on the same tasks, so relay rows and Qwen-alone rows can be matched task for task. This folder runs
the whole loop on TACC Horizon: serve both models, drive episodes on Daytona sandboxes, turn them into rows, train with
marin / Levanter, and evaluate on TB2.1, SWE-bench Verified random-100 and OpenThoughts-TBLite. Results so far, with every
arm, are in `HANDOFF.md`, which links the write-up.

Do the steps in order: prerequisites (section 1), a 4-task smoke run (section 2), then the full loop (section 3).
Variants of the loop are knobs on the same scripts (section 4).

## 1. Prerequisites

Paths below use `$SCRATCH` and `$HOME` as TACC sets them. Run everything on the Horizon login node unless a step says
it is a job.

| | what | how |
|---|---|---|
| 1 | Horizon onboarding: login, tmux, `hf`, `tunnel.sh`, storage rules | `docs/horizon/ONBOARDING.md`. It ends with this repo cloned at `~/snowball/ota` (tag `relay-v1`). The scripts expect that path. |
| 2 | The snowball venv (`~/snowball/envs/snowball`, vLLM for GB200) | the commands below; about 5 min on the login node, then about 35 min in one job (Luke's 09-29 times) |
| 3 | The relay additions: python-socks for the driver's bridge, `envs/relay-extra` (torchvision for the teacher), `envs/msa-2.4.6` (mini-swe-agent), and the teacher, EAGLE-3 draft and Stage-3 models in `$SCRATCH/hf_hub` (about 150 GB) | `bash ~/snowball/ota/data/relay/horizon/build_relay_env.sh` on the login node, after step 2 |
| 4 | The 09-21 base, used as the student and as the SFT base | the `hf download` below. Step 3 does not fetch it. |
| 5 | Three harbor clones, each at a pinned commit | the `git clone` loop below |
| 6 | A Daytona key for Luke's eval org, which holds the task snapshots | ask Luke. **(you)** Put it in `~/.config/otagent/daytona_eval.env` as one line `DAYTONA_API_KEY=<key>`, then `chmod 600` it. |
| 7 | A staged task tree with `TASKS.txt` and `router_tasks.json` | section 2 (smoke tree) and section 3.1 (full tree) |

Step 2, the venv. The two SHAs must match `build_env.sbatch`'s values, and `SNOWBALL_RUSTUP=1` installs the cargo the
vLLM build needs. This sequence was run end to end by a fresh user on 2026-10-16 (login steps 4 min, the job 31 min,
then the serve smoke below passed):

```bash
ROOT=$HOME/snowball SNOWBALL_RUSTUP=1 MSRL_SHA=a03b2773 HARBOR_SHA=dcf609bc \
  bash ~/snowball/ota/data/r2egym/jsc/build_snowball_env.sh tools python clones torch      # login node, downloads only
cd ~/snowball && J=$(sbatch --parsable ~/snowball/ota/data/r2egym/horizon/build_env.sbatch | tail -1)
setsid nohup bash ~/snowball/ota/data/r2egym/horizon/tunnel.sh $J > ~/tunnel.$J.log 2>&1 < /dev/null &
```

The job writes `sb_build.<J>.out` in the submit dir. Its `eagle3` step builds `~/snowball/src/marin_vllm_eagle3`, the
serve overlay `serve_env.sh` points at; the job ends with `smoke: PASS` and `END ok`. If you build somewhere other than
`~/snowball`, pass the same dir as `ROOT=` to the build scripts and as `SNOWBALL_ROOT=` to everything that serves or
drives.

Step 4, the 09-21 base. `relay_sft_arm.sh`, `nr_window.py` and `msa_select.py` read it from this exact directory:

```bash
hf download open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21 --revision b8c07f7df1df65525abbfdbcd1572318ba11c42f \
  --local-dir $SCRATCH/models/grug-datakit-sft-20260921
```

A fine-tuned student goes the same way, for example
`hf download laion/snowball-67b-a2b-relay-sft-acont-step999 --local-dir $SCRATCH/models/relay-sft-acont-step999`.

Step 5, the harbor clones. Harbor is never installed; the scripts put a clone's `src` on `PYTHONPATH`, and each pre-flight
checks the clone's commit:

```bash
for c in harbor-relay:89098635 harbor-msa-relay:96fd911b harbor-p0924:761fb516; do
  git clone -q --filter=blob:none https://github.com/marin-community/harbor.git ~/snowball/${c%%:*} && git -C ~/snowball/${c%%:*} checkout -q ${c#*:}
done
```

| clone | commit | used for |
|---|---|---|
| `~/snowball/harbor-relay` | `89098635` (branch `lukedhlee/terminus2-relay`) | Terminus-2 relay rollouts and the held-out 300 eval (the driver's default) |
| `~/snowball/harbor-msa-relay` | `96fd911b` (branch `lukedhlee/mini-swe-relay`) | relay rollouts under mini-swe-agent tool mode (`AGENT=msa`) |
| `~/snowball/harbor-p0924` | `761fb516` (tag `harbor-config-761fb516...`) | TB2.1, SWE-bench and TB-lite evals (`eval_sft.sh`'s default) |

## 2. Smoke run first (4 tasks per arm, at most 4 node-hours)

The smoke runs both arms on the 4 shortest tasks of a 100-task tree, with 1 student node, 1 teacher node and a 1-node
driver job. It proves serving, the tunnels, Daytona, the routers, harbor and the readout before anything large. These
exact settings passed on a fresh build on 2026-10-16 in 22 minutes and about 1.1 node-hours.

Stage the smoke tree from the public CalibForge bundle (2,457 tasks, checked by sha256), keeping the 100 tasks in
`pilot/relay100_tasks.txt`:

```bash
mkdir -p $SCRATCH/relay/dl && cd $SCRATCH/relay/dl
hf download laion/calibforge-daytona-layers bundle/task_tree.tar.gz --repo-type dataset --local-dir .
echo "54e907f47434a3b731879cc1a6699aa16c24a5f9c0d0c8e522a69459852beafb  bundle/task_tree.tar.gz" | sha256sum -c -
L=$HOME/snowball/ota/data/relay/pilot/relay100_tasks.txt; T=$SCRATCH/relay/calibforge_relay100
sed 's#^#task_tree/#' $L > members.txt && tar -xzf bundle/task_tree.tar.gz -T members.txt && mv task_tree $T && cp $L $T/TASKS.txt
$HOME/snowball/envs/snowball/bin/python $HOME/snowball/ota/data/relay/pilot/pilot_tasks.py router-json \
  --tree $T --list $T/TASKS.txt --out $T/router_tasks.json
```

Then start the serve job and the driver:

```bash
cd ~/snowball/ota/data/relay/horizon
S=$(STUDENT_MODEL=$SCRATCH/models/grug-datakit-sft-20260921 bash serve_submit.sh 2 1 02:00:00 relay_srv_smoke); echo $S
N=relay_smoke_$(date +%Y%m%d_%H%M)
ARMS="control relay_repair" NODES=2 CAP_NODE_H=4 GATE_LAT=120 TREE=$SCRATCH/relay/calibforge_relay100 \
  bash launch_driver.sh $S $N smoke
```

- `serve_submit.sh NODES N_STUDENT TIME NAME` prints the serve job id. Here 2 nodes, the first serving the student.
- `launch_driver.sh <serve job> <run name> smoke` submits the 1-node driver job and starts two login-side tunnels.
  `smoke` means 4 tasks per arm, 4 concurrent, early gate at 10 minutes.
- `GATE_LAT=120` is the small-pilot latency gate of `horizon/rlvr1_pilot.sh`; the default 30 s is the full runs'. The
  latency gate acts only when its 5-minute window holds at least 10 teacher replies. Small runs should still set
  `GATE_MIN=0` (gate off): a few long teacher replies there say nothing about the servers.

Watch `$SCRATCH/experiments/relay/pilot/runs/$N/driver.log`. The run passed when that log ends with `RUN_DONE`, the run
dir has no `ABORT` file, and every `H` line in `readout.txt` there says `PASS`. The `S` lines judge pass and takeover
rates, which mean nothing on 4 tasks, so the final verdict line says `NO SCALE` on a smoke. If the driver job fails before `run_pilot.sh` starts, its own log is
`$SCRATCH/experiments/relay/pilot/logs/relay_drv_${N}_<job>.out`, and the bridge log is in `logs/driver_<job>/`. The
usual early stops:

| log line | meaning |
|---|---|
| `python_socks missing` | the venv lacks python-socks; run `build_relay_env.sh socks` |
| `no tunnel on any of 18080,18081` | the login-side `tunnel.sh` processes did not reach the driver node |
| `Daytona unreachable` (HTTP 000) | the bridge or the tunnels are down, not the key |
| `Daytona key rejected` | the key in `~/.config/otagent/daytona_eval.env` is wrong |
| `CalibForge snapshots not all ACTIVE` | the org lacks a task snapshot; ask Luke |
| `tree ... does not hold N staged tasks` | the tree's dir count differs from `TASKS.txt`, or `router_tasks.json` is missing |

## 3. The full loop

| stage | script | output |
|---|---|---|
| 3.1 stage the task tree | `../calibforge/daytona/build_tree.py`, then `pilot/pilot_tasks.py router-json` | `$SCRATCH/relay/calibforge_tree` |
| 3.2 serve | `horizon/serve_submit.sh` | endpoint files in `$SCRATCH/experiments/relay/pilot/endpoints/` |
| 3.3 drive the rollouts | `horizon/launch_driver.sh` -> `horizon/driver.sbatch` -> `pilot/run_pilot.sh` | run dir `$SCRATCH/experiments/relay/pilot/runs/<name>`, harbor trials in `$SCRATCH/experiments/relay/jobs/<name>_<arm>` |
| 3.3 gate and read out | `pilot/readout.py` (the driver runs it) | `readout.txt`, `readout.json` in the run dir |
| 3.4 census | `sft/hz_pool_census.py` | one JSON line per candidate episode, with QA tags |
| 3.4 render | `sft/render_think_limit.py` (Kimi rows: `sft/kimi_relay_rows.py`) | `<dir>/<run>.rendered.jsonl` |
| 3.4 build arms | `sft/build_matched_arms.py` | `<arm>_rows.jsonl` |
| 3.5 SFT and export | `../r2egym/horizon/sft/relay_sft_arm.sh` | one HF export per pass |
| 3.6 eval | `../r2egym/horizon/sft/eval_sft.sh` | harbor job dirs in `$SCRATCH/experiments/sft_eval/tb2_jobs/` |
| 3.6 readout | `../r2egym/horizon/sft/eval_readout.py` | pass@1 per set and paired differences |

### 3.1 Stage the task tree

The full runs use the CalibForge remainder: 2,613 tasks (`pilot/calibforge_remainder.txt`) that run on the three
CalibForge snapshots in the eval org and are disjoint from the held-out 300. It is not in the public bundle, so build it
from public sources (the CalibForge release, the Docker Hub manifests cache, the committed task table):

```bash
cd $SCRATCH/relay/dl
hf download hamishivi/agent-task-calibforge task-data.tar.gz --repo-type dataset \
  --revision f605550ac21c37cecc34e6767a753219e0ca58d3 --local-dir .
echo "79672cdec29ec9160b663e99dd0bb17be656c74756dea59c68a4fe2d285abb80  task-data.tar.gz" | sha256sum -c -
hf download laion/calibforge-daytona-layers registry_cache.tar.gz --repo-type dataset --local-dir .
mkdir -p registry_cache && tar -xzf registry_cache.tar.gz -C registry_cache
cd ~/snowball/ota
PYTHONPATH=$HOME/snowball/harbor-relay/src $HOME/snowball/envs/snowball/bin/python data/calibforge/daytona/build_tree.py \
  --parquet data/calibforge/calibforge_tasks.parquet --source $SCRATCH/relay/dl/task-data.tar.gz \
  --cache $SCRATCH/relay/dl/registry_cache --out $SCRATCH/relay/calibforge_tree --all --only data/relay/pilot/calibforge_remainder.txt
$HOME/snowball/envs/snowball/bin/python data/relay/pilot/pilot_tasks.py router-json --tree $SCRATCH/relay/calibforge_tree \
  --list $SCRATCH/relay/calibforge_tree/TASKS.txt --out $SCRATCH/relay/calibforge_tree/router_tasks.json
```

- `build_tree.py` prints `"covered": 2613` and writes `TASKS.txt`, `pool.json` and `coverage.tsv`. It reads pandas and
  pyarrow, which the snowball venv has (checked on a fresh build, 2026-10-16). The task images are public on Docker Hub (`aweaiteam/calibforge`), and each task's layers come from the
  public HF mirror `laion/calibforge-daytona-layers` when a sandbox starts.
- Run only a subset with `TASK_LIST=<file>`, for example `data/relay/pilot/calibforge_remainder_pilot1000.txt`
  (a stratified 1,000). The tree must hold every task in `TASKS.txt`; the pre-flight counts its dirs.
- The pre-flight refuses a tree whose `TASKS.txt` overlaps `pilot/calibforge_heldout300.txt`, and it refuses to start
  unless all three CalibForge snapshots are ACTIVE in the org. If they are not, ask Luke; never create or delete
  snapshots in the eval org yourself.
- Instead of rebuilding, Luke can share a tarball of his staged tree.

### 3.2 and 3.3 Serve, then drive the rollouts

```bash
cd ~/snowball/ota/data/relay/horizon
S=$(STUDENT_MODEL=$SCRATCH/models/grug-datakit-sft-20260921 bash serve_submit.sh 8 3 08:00:00 relay_srv_<name>)
ARMS=relay_repair NODES=8 CAP_NODE_H=110 CONC=56 N_ATTEMPTS=2 SHUFFLE_SEED=1 TREE=$SCRATCH/relay/calibforge_tree \
  TASK_LIST=$HOME/snowball/ota/data/relay/pilot/calibforge_remainder_pilot1000.txt bash launch_driver.sh $S <name>
```

- **Always set `STUDENT_MODEL`.** Without it, `serve_env.sh` serves the old Stage-3 model that `build_relay_env.sh`
  downloads, and the router reads the tokenizer of whatever was served, so the run completes with the wrong student and
  no error. Check `student_model=` in `endpoints/<serve job>.meta`.
- The serve job above has 3 student nodes and 5 teacher nodes (4 single-GPU Qwen servers each). The first student
  server answers in about 3 to 7 minutes with warm caches.
- `NODES` is the serve job's node count, and `CAP_NODE_H` caps serve and driver node-hours together. The router ends
  every episode at a deadline computed from that cap, so the run finishes inside it with its results written. The
  driver cancels the serve job when harbor is done, at the cap, or on an abort.
- `IN_SERVE=1` runs the driver as a step inside the serve job instead of a 1-node job of its own (one job fewer under
  Horizon's 20-running-jobs cap).
- `launch_driver.sh` fills every relay knob that is not set in the environment (section 4 lists them). Pick a new run
  name each time; `run_pilot.sh` refuses an existing one.

Watch the same files as the smoke, plus `FLAGS` and `arm_gates.log` in the run dir (the finetuned-student gates, which
flag but never cancel by default). `horizon/DRIVER.md` explains the driver, tunnels and gates, and `horizon/SERVING.md`
the serve job. A driver can end in `ABORT` after every trial has finished (a harbor process that never exits); count the
`result.json` files in the harbor job dirs before treating it as a failure.

### 3.4 Turn rollouts into SFT rows

Census, render and arm building are CPU work on files, so run them as a 1-node job, not on the login node. Put the run
names in, then submit:

```bash
mkdir -p $SCRATCH/relay/rows && cat > $SCRATCH/relay/rows/rows.sh <<'EOF'
#!/bin/bash
set -euo pipefail
PY=$HOME/snowball/envs/snowball/bin/python; SFT=$HOME/snowball/ota/data/relay/sft; O=$SCRATCH/relay/rows
E=$SCRATCH/experiments/relay/pilot; BASE=$SCRATCH/models/grug-datakit-sft-20260921
PARSER=$HOME/snowball/harbor-relay/src/harbor/agents/terminus_2/terminus_json_plain_parser.py
RUNS_relay_repair="$E/runs/<relay run 1> $E/runs/<relay run 2>"
RUNS_control="$E/runs/<control run 1>"
for ARM in relay_repair control; do
  RUNS=RUNS_$ARM; mkdir -p $O/rendered_$ARM
  $PY $SFT/hz_pool_census.py --arm $ARM --runs ${!RUNS} --out $O/census_$ARM.jsonl > $O/census_$ARM.json
  for R in ${!RUNS}; do
    $PY $SFT/render_think_limit.py --render-dir $SFT --run-dir $R --arm $ARM --manifest $O/census_$ARM.jsonl \
      --tokenizer-dir $BASE --think-limit 16384 --strip-copied-markers --terminus-parser $PARSER --autofix-loss none \
      --out $O/rendered_$ARM/$(basename $R).rendered.jsonl > $O/rendered_$ARM/$(basename $R).json
  done
done
$PY $SFT/kimi_relay_rows.py --parquet $SCRATCH/relay/kimi/kimi_swesmith_v1/train-00000-of-00001.parquet \
  --render-dir $SFT --tokenizer-dir $BASE --out $O/kimi_rows.jsonl
$PY $SFT/build_matched_arms.py --relay-census $O/census_relay_repair.jsonl --relay-rendered $O/rendered_relay_repair \
  --qwen-census $O/census_control.jsonl --qwen-rendered $O/rendered_control --kimi-from $O/kimi_rows.jsonl --out-dir $O/arms
EOF
sbatch -A CCR24067 -p debug -N 1 -t 02:00:00 -o $SCRATCH/relay/rows/rows.%j.out $SCRATCH/relay/rows/rows.sh
```

- **The rendered file names matter.** `build_matched_arms.py` and `build_hz_arms.py` read only
  `<dir>/*.rendered.jsonl`. Any other name gives empty arms and no error.
- The census output is the render's manifest: `render_think_limit.py` reads only each line's `sid` and renders the
  episodes of that one `--run-dir`. Its `missing` count is the other runs' episodes; ignore it.
- `build_matched_arms.py` applies the filters, identical for both arms: no weak timeout, the verifier ran, no `leak`
  (Qwen names the verify note or a previous agent), `hunt` (a command reads grader files) or `canary` tag, the row fits
  65,536 tokens and has a trained token. Then it matches per task: at most 2 relay rows per task, and the Qwen part gets
  the same passes and failures on that task. It writes `mrel_rows.jsonl`, `mqwen_rows.jsonl`, and the same with the
  Kimi rows added (`mrelk`, `mqwenk`), plus `match.json` with the counts.
- A row is `{sid, n_tokens, fits, ids, loss, turns}`: 09-21 token ids and a 0/1 loss mask.

The Kimi SWE-smith rows need the Kimi parquet, built once before `rows.sh` (download and convert on the login node,
about 4 minutes):

```bash
mkdir -p $SCRATCH/relay/kimi && cd $SCRATCH/relay/kimi
hf download --repo-type dataset open-athena/Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k \
  --revision 1b87a9cf78b897a2355105d47afd44d818cad25d --local-dir src_kimi
K=$HOME/snowball/ota/data/swesmith/kimi_ids
$HOME/snowball/envs/snowball/bin/python $HOME/snowball/ota/data/swesmith/kimi_sft_convert.py --shards 'src_kimi/data/*.parquet' \
  --tokenizer-dir $(ls -d $SCRATCH/hf_hub/models--laion--snowball-67b-a2b-sft-s3-nemotron-terminal-step1888/snapshots/*) \
  --idmap $K/taskindex_to_instance.json --train-ids $K/kimi_train_ids.txt --heldout-ids $K/kimi_heldout_ids.txt \
  --out kimi_swesmith_v1 --jupiter-root $SCRATCH/relay/kimi/kimi_swesmith_v1
```

Expect 4,398 train rows (66,269,429 tokens) and 628 held-out rows.

### 3.5 SFT and export

One-time setup: the marin branch with the relay stages, its env, and the 09-21 init. `data/r2egym/horizon/SFT.md` has
the env build; the relay recipe needs the `-0921` branch, not the plain one. Clone the tag `relay-v1` on Luke's marin
fork, the commit this runbook was checked against (branch `lukedhlee/horizon-snowball-sft-0921` @ `ec484450`); the
branch itself keeps moving:

```bash
git clone --filter=blob:none -b relay-v1 https://github.com/lukedhlee/marin.git ~/snowball/marin-sft
```

Import 09-21 into the native init the stages start from (4 CPU nodes, 45-minute limit). It is done when
`init-dk0921-step0/snowball_base.json` exists; if the job fails, remove the partial `init-dk0921-step0` dir before
retrying:

```bash
mkdir -p $SCRATCH/snowball-sft/logs $SCRATCH/snowball-sft/experiments/snowball-base-inits
sbatch -o $SCRATCH/snowball-sft/logs/import.%j.log \
  --export=ALL,MARIN_ROOT=$HOME/snowball/marin-sft,MARIN_PYTHON=$HOME/snowball/envs/marin-grug-sft/bin/python,SNOWBALL_HF_CHECKPOINT=$SCRATCH/models/grug-datakit-sft-20260921,SNOWBALL_INIT=$SCRATCH/snowball-sft/experiments/snowball-base-inits/init-dk0921-step0,SNOWBALL_BASE_CONFIG_FROM_HF=true,SNOWBALL_PENDING_FROM_BIAS=true \
  $HOME/snowball/marin-sft/experiments/june_tpu_67b_a2b/moe/horizon_snowball_import.sbatch
```

Then train one arm per tmux session. `ARM` must name a marin stage `relay_<ARM>`, and `DATASET_REVISION` must equal that
stage's pinned revision. The matched 2x2's stages are `relay_mrel`, `relay_mqwen`, `relay_mrelk` and `relay_mqwenk`,
all pinned to `m20261001`:

```bash
cd ~/snowball/ota
tmux new -d -s sft_mrel "ARM=mrel ROWS=$SCRATCH/relay/rows/arms/mrel_rows.jsonl DATASET_REVISION=m20261001 WALL=08:00:00 \
  bash data/r2egym/horizon/sft/relay_sft_arm.sh"
```

- The script submits a 1-node rows job (parquet and pack count), a 1-node cache job, the 4-node training job and one
  4-node export job per pass, each skipped when its output exists. The log is
  `$SCRATCH/snowball-sft/logs/relay_<ARM>/arm.log`; `ARM_DONE` is the end.
- Exports land in `$SCRATCH/snowball-sft/experiments/snowball-relay-sft/relay_<ARM>/lr3e-4-sched3/export-step<N>-hf-bf16`,
  one per pass. The recipe: LR 3e-4 (`LR`), 3 passes on one cosine (`EPOCHS`), warmup 5 %, seed 0, 16 x 65,536 tokens
  per step on 4 nodes, router bias frozen at 09-21's.
- `WALL` is the training job's time limit (default 02:30:00); raise it for arms of more than a few hundred steps.
- A new arm name needs a new stage: add one line to `STAGES` in `experiments/june_tpu_67b_a2b/moe/vista_snowball_chat.py`
  on your marin branch (copy the `relay_mrel` line, give it a new revision string and a `max_steps` above 3 passes), then
  pass that revision as `DATASET_REVISION`.
- To start from a fine-tuned export instead of 09-21, import that export the same way into a new init dir and pass
  `SNOWBALL_INIT=<that dir>`.
- Each arm keeps one native checkpoint per pass, about 625 GiB each, in `$SCRATCH`. Delete them once the exports are
  checked.

### 3.6 Evaluate and read out

Stage the three eval trees once. Each must hold exactly its task count; `eval_sft.sh` checks it:

```bash
mkdir -p $SCRATCH/tasks && cd $SCRATCH/tasks
git clone -q https://github.com/harbor-framework/terminal-bench-2.git terminal_bench_2_1 \
  && git -C terminal_bench_2_1 checkout -q 53ff2b87d621bdb97b455671f2bd9728b7d86c11
git clone -q https://github.com/open-thoughts/OpenThoughts-TBLite.git openthoughts_tblite_2_0 \
  && git -C openthoughts_tblite_2_0 checkout -q f075e463472c7790b85793b392dff1fff20cc0e3
hf download --repo-type dataset DCAgent2/swebench-verified-random-100-folders \
  --revision a2e51e9e0e8029156ed340719eb8cc7ceee3ed1a --local-dir swebench_verified_random100
for d in terminal_bench_2_1 swebench_verified_random100 openthoughts_tblite_2_0; do
  echo "$d $(find $d -mindepth 2 -maxdepth 2 -name task.toml | wc -l)"; done                # 89, 100, 100
```

TB2.1 is terminal-bench-2 at `53ff2b8` ("Various task fixes for TB2.1"; the repo used to be
`laude-institute/terminal-bench-2`), with `train-fasttext` excluded at run time. TB2.1 and TB-lite match the provenance
files of Luke's Horizon trees; the SWE-bench revision is the documented pinned snapshot, not checked against his
Horizon copy (unverified). `$SCRATCH` deletes files unread for
10 days, so re-stage a tree that is gone.

Evaluate every model you will compare in one queue, base included:

```bash
D=$(date +%Y%m%d)
tmux new -d -s eval_m "DAY=$D GROUPED=1 REPS='1 2 3' MODELS='base=$SCRATCH/models/grug-datakit-sft-20260921 mrel=<mrel export> mqwen=<mqwen export>' \
  bash ~/snowball/ota/data/r2egym/horizon/sft/eval_sft.sh"
```

- One unit is one (model, rep): a 5-node job that runs TB2.1, SWE-bench random-100 (2 shards) and TB-lite (2 shards)
  once each, one student server and 16 concurrent trials per node, under the 65,536 / 16,384 token policy
  (`data/tb2/jupiter/tb2_marin_policy_0924_65k16k.yaml`). A unit takes about 2 to 2.5 hours.
- The queue keeps at most `MAXJOBS` (34) of your jobs submitted, and each serve job asks for `SERVE_TIME` (04:00:00).
  The log is `$SCRATCH/experiments/sft_eval/eval.log`; `EVAL_QUEUE_DONE` is the end.

Then read out with the same day:

```bash
$HOME/snowball/envs/snowball/bin/python ~/snowball/ota/data/r2egym/horizon/sft/eval_readout.py --tags base mrel mqwen --day $D
```

It prints pass@1 per set and model over the reps (a task's score is its mean over reps, 95 % bootstrap over tasks), and
every paired difference over the tasks both models scored. `--day` defaults to `'*'`, which pools every run dir of a
tag on any day into the same reps, so always pass the day of the queue you mean.

## 4. Variants

**Your own intervention rules.** When the teacher steps in, where (rewrite the student's turn or take the next one) and for how
long are set by a trigger config: `TRIGGER_CONFIG=<yaml>` on `launch_driver.sh` replaces the intervention knobs below.
Rules, LLM judges and learned scores plug in as small Python functions or YAML entries, and `triggers/replay.py`
scores a config on recorded episodes before any GPU is used. The guide is `triggers/README.md`. Without
`TRIGGER_CONFIG`, everything below behaves exactly as before.

All variants run on the same scripts. The knobs live in three layers, and the layer you launch from decides the defaults:

| knob | `pilot/run_pilot.sh` default | `horizon/launch_driver.sh` default | `horizon/rlvr1_pilot.sh` (msa) |
|---|---|---|---|
| `ARMS` | `control relay_repair` | `relay_repair` | `relay_repair` |
| `CLOCK` | `paused` | `repair` | `repair` |
| `CTX_BUDGET` (context_budget takeover) | off | 32000 | 40000 |
| `ROW_MAX` / `ROW_RESERVE` | off / 8192 | 65536 / 8192 | 65536 / 8192 |
| `MAX_INPUT` / `MAX_OUTPUT` | 65536 / 8192 | 131072 / 8192 | 131072 / 8192 |
| `VERIFY_NOTE` | 0 | 1 | 1 |
| `TEACHER_GUARD` | 1 | 1 | 1 |
| `BALANCE` / `STAGGER_SEC` | `pinned` / 0 | `active` / 180 | `active` / 0 |
| `GATE_MIN` / `GATE_LAT` | 0 (off) / 30 | 15 / 30 | 15 / 120 |
| `STOP_AFTER` (stop rule) | 0 (off) | 300 | 16 |
| `ARM_GATES` | 0 | 1 | 1 |
| `CONC` / `N_ATTEMPTS` | 100 / 1 | 128 / 2 | 8 / 1 |
| `TIME_TAKEOVER`, `ASSIST` | off | off | 0.6, 1 (`ASSIST_CTX=20000,35000`, `ASSIST_ERR=4`, `ASSIST_GAP=3`, `ASSIST_MAX=6`) plus `TEACHER_HINT_TEXT` |

`launch_driver.sh` fills a knob with `${X:-default}`, so an empty value cannot turn it off there; set the value you want
(for example `CTX_BUDGET` cannot be emptied, and `launch_eval_heldout.sh` unsets the relay knobs for that reason).

| variant | launcher | env / flags | notes |
|---|---|---|---|
| Relay, Terminus-2 (the main recipe) | `launch_driver.sh` | defaults above, `ARMS=relay_repair` | student, then Qwen on parse errors (one turn) and on sticky takeovers |
| Qwen alone (control) | `launch_driver.sh` | `ARMS=control`, or `ARMS="control relay_repair"` for both in one run | render with `--arm control`. A control-only serve job can skip the student (`serve_submit.sh <n> 0 ...`); still set `STUDENT_MODEL`, because the driver reads its tokenizer from the serve job |
| Sticky takeovers only | `launch_driver.sh` | `ARMS=relay` (student thinking stripped) or `ARMS=relay_keep` (kept) | no parse-error repairs |
| Student alone | `launch_driver.sh`, or `launch_eval_heldout.sh` for the held-out 300 | `ARMS=student_only` | held-out eval, and self-distillation rows (`sft/selfd_select.py`) |
| mini-swe-agent tool mode (MSA) | `launch_driver.sh` | `AGENT=msa HARBOR_SRC=$HOME/snowball/harbor-msa-relay/src HARBOR_SHA=96fd911b`; `MSA_DIR` defaults to `$HOME/snowball/envs/msa-2.4.6`; `MSA_INTERLEAVED=false` stops harbor re-sending earlier reasoning | rows: `sft/render_msa.py`, then `sft/msa_select.py` (below) |
| Guided relay (MSA, from RLVR1 step 12) | `horizon/rlvr1_pilot.sh` | `AGENT_MODE=msa|terminus2`, `TASK_COUNT`, `TASK_OFFSET`, `SERVE_NODES`, `STUDENT_NODES`, `CAP_NODE_H`, `ARMS` | downloads and serves `open-athena/Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02`; `IN_SERVE=1`; `rlvr1_full.sh`, `rlvr1_paired_full.sh` and `rlvr1_msa_control_full.sh` shard it over the 2,613 tasks |
| Verify-note text | any | `VERIFY_NOTE=1 VERIFY_NOTE_TEXT='<text>'` | the default Terminus-2 note says "another agent did the previous work"; Qwen sometimes repeats it, and those rows are tagged `leak` and dropped. The MSA default note has no such framing |
| Student's thinking after a takeover (MSA) | any | `STUDENT_VIEW_STRIP=1` | the student's turns in the hard-end count and the row drop their thinking |
| Context hand-off point | any | `CTX_BUDGET=<tokens>`, `TIME_TAKEOVER=<fraction>` (MSA) | |

Rendering variants:

| rows | script | notes |
|---|---|---|
| Terminus-2 relay or control rows | `sft/render_think_limit.py` (wraps `sft/render.py`) | `--think-limit`, `--autofix-loss`, `--strip-copied-markers`, `--cap` |
| MSA relay or control rows | `sft/render_msa.py --runs <dirs> --tokenizer-dir <09-21> --out-dir <dir> [--arm control]` | writes `rows.jsonl` with QA tags; no `--cap` flag (always 1,000) |
| MSA arms | `sft/msa_select.py --relay <rows> [--control <rows>] --out-dir <dir> [--outcome pass|no_overflow] [--ends-submit]` | `msaall_rows.jsonl`, and with `--control` the matched `msarel` / `msaqwen` |
| No-refeed windows (MSA) | `sft/nr_window.py --rows <rows> --out <rows> [--k 8]` | rows whose context matches the no-refeed eval (`tb2_msa_toolmode_0924_norefeed.yaml`) |
| Self-distillation (MSA) | `sft/selfd_select.py --rows <student_only rows> --out <rows>` | |
| No loss on long commands (MSA) | `sft/cmd_mask.py --rows <rows> --out <rows> [--max-chars 2000]` | |
| SWE replay rows (MSA) | `sft/render_swe_traces.py`, `sft/render_si2ca.py`, `sft/render_orchard.py` | Open-SWE-Traces, SI2CA, Orchard |
| Kimi SWE-smith rows (Terminus-2) | `sft/kimi_relay_rows.py` | `--cap`, `--think-limit` |
| OpenThoughts-Agent-SFT rows | `sft/ota_relay_rows.py` | |
| Lower trained-thinking limit, no re-render | `sft/think_cap_rows.py` | |

**The reasoning cap.** `router/reasoning_cap.py` sets `CAP_TOKENS = 1000`. In every rendered row, each teacher turn but
the last shows only its first 1,000 reasoning tokens, and a cut reasoning span carries no loss, so most of the
teacher's older-turn reasoning is not trained. The router shows the student the same capped history during the episode.
`--cap 1000000000` on `render_think_limit.py` or `kimi_relay_rows.py` renders every turn's reasoning whole; it is then
trained when it is within `--think-limit`. Uncut rows are longer, so more of them exceed 65,536 tokens, and
`build_matched_arms.py` drops those.

MSA evals use the same `eval_sft.sh` with the no-refeed tool-mode policy and the harbor that policy names
(`lukedhlee/mini-swe-host-exec-isolation` @ `6feb3759`), cloned like the others. Which clone Luke's Horizon MSA evals
used is unverified; `harbor-msa-relay` @ `96fd911b` contains `6feb3759`.

```bash
git clone -q --filter=blob:none https://github.com/marin-community/harbor.git ~/snowball/harbor-msa-eval && git -C ~/snowball/harbor-msa-eval checkout -q 6feb3759
POLICY_FILE=tb2_msa_toolmode_0924_norefeed.yaml MSA_DIR=$HOME/snowball/envs/msa-2.4.6 \
  HARBOR_SRC=$HOME/snowball/harbor-msa-eval/src HARBOR_SHA=6feb3759 MODELS='...' GROUPED=1 bash ~/snowball/ota/data/r2egym/horizon/sft/eval_sft.sh
```

## 5. The comparison: relay vs Qwen alone

- **Same tasks, same pipeline.** Run the control arm on the same task list and settings, ideally in the same run
  (`ARMS="control relay_repair"`). Both arms go through the same census, render options and filters.
- **Matched arms.** `build_matched_arms.py` (Terminus-2) or `msa_select.py --control` (MSA) give both arms the same
  (task, outcome) multiset, so only who played the early turns differs. Trained tokens cannot be matched (a relay row
  trains only Qwen's turns after the hand-over); `match.json` reports them.
- **Same training recipe** for every arm: `relay_sft_arm.sh` with the same `LR`, `EPOCHS` and init.
- **Same eval window.** Put every compared model, base included, in one `eval_sft.sh` queue with one `DAY`, and pass that
  `--day` to `eval_readout.py`.
- **Paired readout.** Read the paired difference and its bootstrap interval, not two separate pass rates. One run's
  paired difference has a rough SD of 0.05 on TB2.1 and 0.065 on SWE-bench and TB-lite; three reps shrink it.

## 6. File map

Paths are relative to `data/relay/`.

**Main line**

| stage | files |
|---|---|
| serve | `horizon/serve_submit.sh`, `horizon/serve_env.sh`, `pilot/serve_relay.sbatch`, `pilot/serve_node.sh` |
| drive | `horizon/launch_driver.sh`, `horizon/driver.sbatch`, `horizon/socks_connect_bridge.py`, `horizon/cleanup_sandboxes_async.py`, `horizon/arm_gates.py`, `pilot/run_pilot.sh`, `pilot/relay_pilot.yaml`, `pilot/relay_msa.yaml`, `pilot/stop_rule.py` |
| route | `router/relay_router.py`, `router/autofix.py`, `router/msa_tool.py`, `router/reasoning_cap.py`, `triggers/relay_triggers.py` |
| read out | `pilot/readout.py`, `pilot/select_kept.py`, `msa/gate.py` |
| tasks | `pilot/pilot_tasks.py`, `pilot/calibforge_remainder.txt` (+ `_pilot1000`, `.md`), `pilot/calibforge_heldout300.txt`, `pilot/relay100_tasks.txt`, `pilot/full_pool.txt` |
| rows | `sft/hz_pool_census.py`, `sft/render.py`, `sft/render_think_limit.py`, `sft/kimi_relay_rows.py`, `sft/build_matched_arms.py`, `sft/render_msa.py`, `sft/msa_select.py`, `sft/nr_window.py` |
| SFT input | `sft/relay_rows_to_parquet.py`, `sft/pack_epoch_steps.py` (both called by `relay_sft_arm.sh`) |
| eval readout | `sft/paired_eval.py` (`eval_readout.py` imports it) |

**Support**

| files | what |
|---|---|
| `horizon/serve_check.py` | real completions against a serve job's endpoints |
| `horizon/launch_eval_heldout.sh` | held-out 300 CalibForge eval of one checkpoint |
| `horizon/push_rollouts.sh`, `horizon/pull_rollouts.sh`, `horizon/rollout_transfer.py` | Horizon to Jupiter transfer through a private HF repo (not needed when you render on Horizon) |
| `horizon/daytona_probe.py` | Daytona load test through the tunnels |
| `pilot/merge_runs.py` | merge a run and its top-up runs into one arm |
| `pilot/cleanup_sandboxes.py` | sandbox cleanup with the sync SDK (Jupiter; Horizon uses the async copy) |
| `sft/build_hz_arms.py` | the H1 / H4 arms; needs arm A's Jupiter rows (private) |
| `sft/export_relay_dataset.py`, `sft/split_relay_dataset.py`, `sft/build_relay_sft_tab.py`, `sft/build_relay_traces_msa2.py` | the public `laion/calibforge-relay-traces` dataset |
| `sft/contamination_check.py` | n-gram contamination check of an SFT set against the eval trees |
| `sft/verify_relay_cache.py` | proves the trainer's cache holds exactly the rendered ids and loss (written for the Jupiter arms) |
| `router/tests/`, `triggers/tests/` | CPU tests |

**Variants**: `horizon/rlvr1_pilot.sh`, `horizon/rlvr1_full.sh`, `horizon/rlvr1_paired_full.sh`,
`horizon/rlvr1_msa_control_full.sh`, `horizon/rlvr1_snapshots.py` (create-only CalibForge snapshot restore),
`sft/selfd_select.py`, `sft/cmd_mask.py`, `sft/render_swe_traces.py`, `sft/render_si2ca.py`, `sft/render_orchard.py`,
`sft/ota_relay_rows.py`, `sft/think_cap_rows.py`.

**History (Jupiter era, superseded)**: `pilot/launch_relay.sh` (replaced by `horizon/launch_driver.sh`),
`pilot/run_full.sh` (marked SUPERSEDED), `pilot/check_decide.py`, `pilot/relay_plan.py`, `pilot/full_decide.py`,
`pilot/final_v2.py`, `pilot/final_match.py`, `pilot/match_kept.py`, `pilot/real_failures.py`, `pilot/wiring_check.sh`,
`pilot/stage_tree.sh` (Jupiter paths; section 2 has the Horizon steps), `sft/relay_sft_arm.sh` (Jupiter's; Horizon's is
`data/r2egym/horizon/sft/relay_sft_arm.sh`), `sft/qa_final.py`, `sft/mix_driver.sh`, `sft/check_autofix_none.py`.

**One-off analyses**: `bench/*`, `report/*`, `verify_note/*`, `triggers/calibrate.py`, `triggers/adapters.py`,
`horizon/gate4_paired.py`, `sft/attempt_counts.py`, `sft/diag_compare.py`, `sft/full_counts.py`, `sft/harness_errors.py`,
`sft/ho_turns.py`, `sft/long_turn_check.py`, `sft/think_cap_check.py`, `sft/think_cut_proxy.py`, `sft/timeout_anatomy.py`.

## 7. Gotchas

**Silent failures**
- An unset `STUDENT_MODEL` serves the Stage-3 model with no error (section 3.3).
- Rendered files not named `*.rendered.jsonl` give empty arms with no error (section 3.4).
- `eval_readout.py` without `--day` pools every day's runs of a tag (section 3.6).

**Inputs you cannot read without Luke**
- The Daytona eval org: its key, the three CalibForge snapshots and the custom-snapshot cap (40). The org also allows
  about 1,000 concurrent sandboxes, shared with Luke's runs, so agree on `CONC` with him.
- `lukeleeai/relay-rollouts-horizon` (private): `push_rollouts.sh`'s default `REPO`. Set `REPO` to your own repo if you
  use the transfer at all.
- `lukeleeai/relay-calibforge-jupiter-0928` (private): the rows behind `relay_sft_arm.sh`'s `ARM=relay|qwen` defaults
  and `build_hz_arms.py`'s arm A part. Pass your own `ROWS`.
- `lukeleeai/snowball-kimi0921-pair` (private): the data step of marin's `pair_kimi0921.sh`, which is why section 3.5
  imports 09-21 with the import job directly.
- `laion/TMax-15K-Harbor` (laion members only): the source of the TMax task tree (`data/tmax/`).
- The per-arm commands of the matched relay vs Qwen-alone comparison were not committed; section 3.4 rebuilds them
  from `build_matched_arms.py` and the stage pins.
- The step that put a capped row in place of an uncut row over 65,536 tokens (the marin `*u` stages) is not a committed
  script.

**Horizon**
- TACC's submit filter sums nodes x time limit over all your queued and running jobs against the SU balance. Keep
  wall limits tight.
- `$SCRATCH` deletes files unread for 10 days: task trees, rollouts, rows and models included. Push what you keep.
- A run or rows name that already exists is refused; pick a new one.
