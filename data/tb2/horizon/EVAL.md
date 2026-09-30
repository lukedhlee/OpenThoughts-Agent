# Snowball evals on Horizon

**Horizon runs the four evals the relay SFT arms were judged on exactly as Jupiter ran them: the same harness revision,
policy file, task trees (task checksums match Jupiter's trials) and Daytona snapshots, at the same concurrency per server.**
The only structural change is where harbor runs. Horizon compute nodes have no internet and the login node is shared,
so harbor runs in its own one-node driver job. That job reaches Daytona through login-side ssh tunnels (`tunnel.sh`) and
a local CONNECT bridge, as the relay driver does (`data/relay/horizon/DRIVER.md`).

| eval | tasks | harness | policy | concurrency | launcher |
|---|---|---|---|---|---|
| Held-out 300 CalibForge | 300 | harbor-relay @ 89098635 | student_only, Terminus-2 strict, 65k/16k, CLOCK=wall | 100 on 1 server | `data/relay/horizon/launch_eval_heldout.sh` |
| TB2.1 | 88 (89 minus train-fasttext) | harbor-p0924 @ 761fb516 | `tb2_marin_policy_0924_65k16k.yaml` | 16 on 1 server | `run_tb2.sh` |
| SWE-bench Verified random-100 | 100, 2 shards of 50 | harbor-p0924 @ 761fb516 | same file | 16 per server, 2 servers | `run_tb2.sh` with `SHARD=i/2` |
| OpenThoughts-TBLite 2.0 | 100, 2 shards of 50 | harbor-p0924 @ 761fb516 | same file | 16 per server, 2 servers | `run_tb2.sh` with `SHARD=i/2` |

## What you need on Horizon

- **Model server.** `data/relay/horizon/serve_submit.sh 1 1 <time> <name>` with `STUDENT_MODEL=<hf snapshot dir>` serves
  one student node. Its command is Jupiter's `serve_snowball.sbatch` `POLICY=trained`: EAGLE-3 draft, TP1 × DP4 × EP,
  65,536 context, 32 sequences per replica, temperature 1.0. It writes `$E/endpoints/<job>.student` when it is up (about 7 min).
- **Harbor clones** (standalone, never installed, imported through `PYTHONPATH`):
  `~/snowball/harbor-relay` @ 89098635, `~/snowball/harbor-p0924` @ 761fb516.
- **Task trees** in `/scratch/11584/lukedhlee/tasks/`: `calibforge_heldout300`, `terminal_bench_2_1`
  (laude-institute/terminal-bench-2 @ 53ff2b8), `swebench_verified_random100`, `openthoughts_tblite_2_0` (GitHub
  open-thoughts/OpenThoughts-TBLite @ f075e463, the commit harbor's registry pins; the HF dataset has no such commit).
  `/scratch` purges files untouched for 10 days, so re-stage a tree if it is gone.
- **Snapshots.** Nothing to build. Check read-only before a run:
  `PYTHONPATH=~/snowball/harbor-p0924/src python snapshot_check.py --key-file ~/.config/otagent/daytona_eval.env --tree <tree>`.
  Every task must resolve to an ACTIVE snapshot. TB2.1 and SWE-bench use Daytona's global snapshots. TB-lite has 8 tasks,
  and TB2.1 has train-fasttext, whose snapshots are the eval org's own. Never create or delete snapshots.

## Run one model

From `~/snowball/ota` (the job checkout), with `M=<hf snapshot dir>` and a date tag `D`:

```
cd ~/snowball/ota/data/relay/horizon
# held-out 300 (driver job + tunnels; ~1.2 h, ~2.5 node-h)
S=$(STUDENT_MODEL=$M bash serve_submit.sh 1 1 03:00:00 eval_srv_heldout) && bash launch_eval_heldout.sh $S heldout_<model>_$D
# TB2.1 (~2 h). Wait until $E/endpoints/$S.student exists, then:
S=$(STUDENT_MODEL=$M bash serve_submit.sh 1 1 04:00:00 eval_srv_tb21)
cd ../../tb2/horizon
HARBOR_SRC=$HOME/snowball/harbor-p0924/src HARBOR_SHA=761fb516 POLICY_FILE=tb2_marin_policy_0924_65k16k.yaml \
  TASKS=/scratch/11584/lukedhlee/tasks/terminal_bench_2_1 NTASKS=89 EXCLUDE_TASKS=train-fasttext DRIVER_TIME=04:00:00 \
  bash run_tb2.sh $S tb21_<model>_$D
# SWE-bench, one serve job per shard i in 0 1 (TASKS=.../swebench_verified_random100 NTASKS=100 SHARD=$i/2)
# TB-lite, one serve job per shard i in 0 1 (TASKS=.../openthoughts_tblite_2_0 NTASKS=100 SHARD=$i/2)
```

`E=/scratch/11584/lukedhlee/experiments/relay/pilot`. The serve endpoints are written there by `serve_relay.sbatch`.
Each `run_tb2.sh` renders the policy to `/scratch/11584/lukedhlee/experiments/tb2/runs/<name>.yaml` and submits
`tb2_driver.sbatch` as `tb2_drv_<name>`. That job cancels its serve job when harbor exits (`SCANCEL_SERVE=0` keeps it).

## Where results land

- Held-out: `$E/runs/<name>/` (driver.log, readout.json) and harbor trials in `/scratch/11584/lukedhlee/experiments/relay/jobs/<name>_student_only/`.
- TB2.1 / SWE / TB-lite: `/scratch/11584/lukedhlee/experiments/tb2_jobs/<name>/`, driver log `.../tb2/logs/tb2_drv_<name>_<job>.out`,
  harbor's own log `.../tb2/runs/logs/drv_<job>/harbor.log`.

## Score

`data/relay/horizon/gate4_paired.py` scores a run with `paired_eval.py`'s rules. Held-out usable = no harness error,
verifier timeout or deadline censoring; the others usable = the verifier returned a reward. Each gives pass@1 with a
Wilson CI and, against another run of the same tasks, a paired bootstrap CI over tasks:

```
python data/relay/horizon/gate4_paired.py --set tb21 --kind tb2 --jupiter <ref job dir> --horizon <job dir>
python data/relay/horizon/gate4_paired.py --set heldout300 --kind heldout --jupiter <ref> --horizon $E/runs/<name>
```

Shards merge by passing both dirs. Jupiter's arm A references: `/scratch/11584/lukedhlee/eval_port/jupiter_refs/`.

## Noise

One trial per task. Two runs of the same model differ by sampling alone. The rough SD of a paired difference is ~0.025
on held-out 300, ~0.05 on TB2.1 and ~0.065 on SWE-bench and TB-lite, so read a single difference against that.
