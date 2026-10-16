# Snowball evals on Horizon

**To evaluate checkpoints, run `data/r2egym/horizon/sft/eval_sft.sh` and read out with `eval_readout.py`.** One
`eval_sft.sh` queue runs TB2.1, SWE-bench Verified random-100 and OpenThoughts-TBLite for every model you list, several
times each, and `eval_readout.py` gives pass@1 per set and the paired difference between models. Horizon runs these
evals exactly as Jupiter ran them: the same harness revision, policy file, task trees (task checksums match Jupiter's
trials) and Daytona snapshots, at the same concurrency per server. The only structural change is where harbor runs.
Horizon compute nodes have no internet and the login node is shared, so harbor runs on a compute node and reaches
Daytona through login-side ssh tunnels (`tunnel.sh`) and a local CONNECT bridge, as the relay driver does
(`data/relay/horizon/DRIVER.md`).

| eval | tasks | harness | policy | concurrency |
|---|---|---|---|---|
| TB2.1 | 88 (89 minus train-fasttext) | harbor-p0924 @ 761fb516 | `tb2_marin_policy_0924_65k16k.yaml` | 16 on 1 server |
| SWE-bench Verified random-100 | 100, 2 shards of 50 | harbor-p0924 @ 761fb516 | same file | 16 per server, 2 servers |
| OpenThoughts-TBLite 2.0 | 100, 2 shards of 50 | harbor-p0924 @ 761fb516 | same file | 16 per server, 2 servers |
| Held-out 300 CalibForge | 300 | harbor-relay @ 89098635 | student_only, Terminus-2 strict, 65k/16k, CLOCK=wall | 100 on 1 server |

## What you need on Horizon

- **The snowball venv and the relay serve setup**: `data/relay/README.md` section 1 (venv, `build_relay_env.sh`,
  Daytona key). The model to evaluate is any HF export dir with a `config.json`.
- **Harbor clones** (standalone, never installed, imported through `PYTHONPATH`): `~/snowball/harbor-p0924` @
  761fb516 for TB2.1 / SWE-bench / TB-lite and `~/snowball/harbor-relay` @ 89098635 for the held-out 300. Both are
  `marin-community/harbor`; `data/relay/README.md` section 1 has the clone commands.
- **Task trees** in `$SCRATCH/tasks/`. Each must hold exactly its task count (`eval_sft.sh` checks it). `$SCRATCH`
  deletes files unread for 10 days, so re-stage a tree that is gone.

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

  - TB2.1 is terminal-bench-2 at 53ff2b8, "Various task fixes for TB2.1" (the repo used to be
    `laude-institute/terminal-bench-2`).
  - TB-lite is `open-thoughts/OpenThoughts-TBLite` at f075e463, the commit harbor's registry pins; the HF dataset has no
    such commit.
  - SWE-bench random-100 is the pinned 100-task snapshot of `DCAgent2/swebench-verified-random-100-folders`.
  - The held-out 300 tree is cut from the public CalibForge bundle with `data/relay/pilot/calibforge_heldout300.txt`,
    the same way `data/relay/README.md` section 2 stages the smoke tree, into `$SCRATCH/tasks/calibforge_heldout300`.
- **Snapshots.** Nothing to build. Check read-only before a run; every task must resolve to an ACTIVE snapshot:
  `PYTHONPATH=~/snowball/harbor-p0924/src ~/snowball/envs/snowball/bin/python ~/snowball/ota/data/tb2/horizon/snapshot_check.py --key-file ~/.config/otagent/daytona_eval.env --tree <tree>`.
  TB2.1 and SWE-bench use Daytona's global snapshots. TB-lite has 8 tasks, and TB2.1 has train-fasttext, whose
  snapshots are the eval org's own. Never create or delete snapshots.

## Evaluate checkpoints (the usual path)

From the login node, in tmux, with one `DAY` for every model you will compare:

```bash
D=$(date +%Y%m%d)
tmux new -d -s eval "DAY=$D GROUPED=1 REPS='1 2 3' MODELS='base=$SCRATCH/models/grug-datakit-sft-20260921 a=<export dir> b=<export dir>' \
  bash ~/snowball/ota/data/r2egym/horizon/sft/eval_sft.sh"
~/snowball/envs/snowball/bin/python ~/snowball/ota/data/r2egym/horizon/sft/eval_readout.py --tags base a b --day $D   # after EVAL_QUEUE_DONE
```

- One unit is one (model, rep). With `GROUPED=1` it is a single 5-node job: one student server per node and one set
  per node (TB2.1, SWE s0 / s1, TB-lite s0 / s1). A unit takes about 2 to 2.5 hours; TB2.1 is the long one.
- The harbor driver runs as an overlapping step inside the serve job, because Horizon allows 20 running jobs per user.
  The queue keeps at most `MAXJOBS` (34) of your jobs submitted, and each serve job asks for `SERVE_TIME` (04:00:00).
- Runs are named `<set>_<tag>_r<rep>_<DAY>`. Results land in `$SCRATCH/experiments/sft_eval/tb2_jobs/<run>`, per-run
  state in `$SCRATCH/experiments/sft_eval/state/<run>/`, and the queue's log in `$SCRATCH/experiments/sft_eval/eval.log`.
- `eval_readout.py` scores each trial with `paired_eval.py`'s TB2 rule (usable = the verifier returned a reward, pass =
  reward >= 1), merges a set's shards, takes each task's mean over reps, and bootstraps over tasks. Pairs use the tasks
  both models scored. `--day` defaults to `'*'`, which pools every day's runs of a tag into the same reps, so pass
  the queue's day. `--ref TAG=SET:<job dir>` adds a single-run reference.
- Other harnesses: `POLICY_FILE`, `HARBOR_SRC`, `HARBOR_SHA` and `MSA_DIR` pass through the environment (the MSA
  tool-mode command is in `data/relay/README.md` section 4). `eval_dispatch.sh` starts units as SFT arms finish.

## One run by hand

`run_tb2.sh` runs one set against one serve job, with its own 1-node driver job. `E=$SCRATCH/experiments/relay/pilot`
is where the serve endpoints are written.

```bash
cd ~/snowball/ota/data/relay/horizon
# held-out 300 (driver job + tunnels; ~1.2 h, ~2.5 node-h)
S=$(STUDENT_MODEL=$M bash serve_submit.sh 1 1 03:00:00 eval_srv_heldout) && bash launch_eval_heldout.sh $S heldout_<model>_$D
# TB2.1 (~2 h). Wait until $E/endpoints/$S.student exists, then:
S=$(STUDENT_MODEL=$M bash serve_submit.sh 1 1 04:00:00 eval_srv_tb21)
cd ../../tb2/horizon
HARBOR_SRC=$HOME/snowball/harbor-p0924/src HARBOR_SHA=761fb516 POLICY_FILE=tb2_marin_policy_0924_65k16k.yaml \
  TASKS=$SCRATCH/tasks/terminal_bench_2_1 NTASKS=89 EXCLUDE_TASKS=train-fasttext DRIVER_TIME=04:00:00 \
  bash run_tb2.sh $S tb21_<model>_$D
# SWE-bench, one serve job per shard i in 0 1 (TASKS=$SCRATCH/tasks/swebench_verified_random100 NTASKS=100 SHARD=$i/2)
# TB-lite, one serve job per shard i in 0 1 (TASKS=$SCRATCH/tasks/openthoughts_tblite_2_0 NTASKS=100 SHARD=$i/2)
```

`M` is the model's HF dir. Each `run_tb2.sh` renders the policy to `$SCRATCH/experiments/tb2/runs/<name>.yaml` and
submits `tb2_driver.sbatch` as `tb2_drv_<name>`. That job cancels its serve job when harbor exits (`SCANCEL_SERVE=0`
keeps it).

- Held-out results: `$E/runs/<name>/` (driver.log, readout.json) and harbor trials in
  `$SCRATCH/experiments/relay/jobs/<name>_student_only/`.
- TB2.1 / SWE / TB-lite: `$SCRATCH/experiments/tb2_jobs/<name>/`, driver log
  `$SCRATCH/experiments/tb2/logs/tb2_drv_<name>_<job>.out`, harbor's own log
  `$SCRATCH/experiments/tb2/runs/logs/drv_<job>/harbor.log`.

`data/relay/horizon/gate4_paired.py` scores such runs against another run of the same tasks (pass@1 with a Wilson CI,
and a paired bootstrap CI over tasks); shards merge by passing both dirs:

```
python data/relay/horizon/gate4_paired.py --set tb21 --kind tb2 --jupiter <ref job dir> --horizon <job dir>
python data/relay/horizon/gate4_paired.py --set heldout300 --kind heldout --jupiter <ref> --horizon $E/runs/<name>
```

Held-out usable = no harness error, verifier timeout or deadline censoring; the others usable = the verifier returned
a reward. Jupiter's arm A references are on Luke's scratch, in `eval_port/jupiter_refs/`.

## Noise

One trial per task per run. Two runs of the same model differ by sampling alone. The rough SD of a paired difference
between single runs is ~0.025 on held-out 300, ~0.05 on TB2.1 and ~0.065 on SWE-bench and TB-lite, so read a single
difference against that; more reps shrink it.
