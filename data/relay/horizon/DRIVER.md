# Relay driver on Horizon

**The relay driver runs on Horizon in its own one-node compute job, and it reaches Daytona through login-side ssh
tunnels. One tunnel carries far more than a relay run needs.** Relay-like Daytona traffic for 192 concurrent trials ran
with no errors and a p99 of 1.1 s per command. Even at 384 seats with no think gaps (300 commands/s, several times a
relay run's rate), there were no errors, and each tunnel's ssh process on the login node stayed under 8 % of a core.
The driver still opens two tunnels, so that one dropping does not stall the run. Finished runs go to Jupiter through the private HF dataset repo `laion/relay-rollouts-horizon`.

## Launch a run end to end

1. **Serve job** (the serving side, `SERVING.md`):
   `S=$(STUDENT_MODEL=<student snapshot> bash serve_submit.sh 8 4 <time> relay_srv_<name>)`.
2. **Driver job**, on the login node, from `~/snowball/ota/data/relay/horizon`:
   `NODES=8 CAP_NODE_H=<cap> TREE=<tree> TASK_LIST=<list> N_ATTEMPTS=2 bash launch_driver.sh $S <name>`.
   It submits `driver.sbatch` as `relay_drv_<name>` and starts one `tunnel.sh` per port in `TUNNEL_PORTS`
   (default 18080 and 18081). Then it returns.
3. **Watch** these files:
   - `$E/runs/<name>/driver.log`: the same driver log as on Jupiter.
   - `FLAGS` and `arm_gates.log` in the run dir: the finetuned-student gates.
   - `$E/logs/relay_drv_<name>_<job>.out`: the job's own output.
   - `$E/logs/driver_<job>/bridge.log`: the proxy, with a stats line every minute.

   `E` is `/scratch/11584/lukedhlee/experiments/relay/pilot`.
4. **Transfer.** After `RUN_DONE`, run `bash push_rollouts.sh <name>` on the Horizon login node, under setsid or tmux.
   Then run `bash pull_rollouts.sh <name>` on the Jupiter login node, from a worktree at this commit
   (`code/ota-relay-hz`).

`launch_driver.sh` sets everything explicitly. It uses the full relay runs' settings (`relay_repair`, `CLOCK=repair`,
32k context budget, 64k row limit, Qwen at 128k, `STAGGER_SEC=180`, the latency/KV gate from minute 15, the stop rule,
`CONC=128`) plus the new gates. It does not run Jupiter's `check_decide.py` / `relay_plan.py` gating, and every setting
can be overridden from the environment.

## Jobs and tunnels

- **Two Slurm jobs.** The serve job holds N nodes and the driver job holds 1 node.
  - The driver scancels the serve job when harbor is done, at the cap, or on an abort. Its own job ends when
    `run_pilot.sh` returns.
  - The node-hour cap counts both jobs (`DRIVER_JOB` / `DRIVER_NODES`). The router deadline reserves the driver's
    time before the serve job started and its 45 min verify tail. `run.meta` gets `serve_node_hours` and
    `driver_node_hours`.
- **Why a separate job.** The login node has one core, so the driver cannot run there. It cannot run inside the serve
  job either, because it cancels that job.
- **Why tunnels.** Compute nodes have no internet, and `sbatch` is refused there. `squeue`, `sacct` and `scancel` work.
  - `tunnel.sh` runs on the login node and keeps one `ssh -R PORT` per port open to the driver node.
  - `socks_connect_bridge.py` on the driver node turns those SOCKS ports into one HTTP CONNECT proxy,
    `HTTPS_PROXY=http://127.0.0.1:18946`. Each connection goes to the least-loaded tunnel, and a failed handshake is
    retried once on the other.
  - Model traffic is plain http to the serve nodes' short names, which resolve from `/etc/hosts`. It never touches the
    proxy, and `run_pilot.sh` adds the model hosts to `NO_PROXY`.
- **Three Horizon traps, handled.**
  - **Slow DNS.** Every outside DNS lookup takes 5 s, because the first nameserver never answers. This happens on
    compute and login nodes alike. The bridge caches lookups, and `RES_OPTIONS=timeout:1` cuts the rest to 1 s.
  - **The sync SDK ignores the proxy.** The sync Daytona SDK's API client ignores `HTTPS_PROXY` and hangs in
    `SYN-SENT`. So the driver cleans up sandboxes with `cleanup_sandboxes_async.py` (`CLEANUP_PY`). Harbor and the
    snapshot census already use the async client.
  - **Student tokenizer.** The router must use the tokenizer of the student the serve job actually loaded.
    `STUDENT_TOKENIZER=meta` reads it from the serve job's endpoints `.meta`.

## Gates

**Unchanged from the Jupiter full run:**
- the pre-flight: harbor SHA, tree count, held-out split disjoint, Daytona key, the three snapshots ACTIVE;
- the early gate at 25 min;
- the latency / KV gate from minute 15;
- the stop rule from 300 scored episodes: overflow > 0.30, format < 0.98, or harness errors > 0.10;
- the node-hour cap;
- the sandbox cleanup and the final readout.

**New for the finetuned student** (`ARM_GATES=1`, `arm_gates.py`, every 10 min from minute 10). Both gates were
calibrated on 09-21 only. So they **flag** by default: a line in the log and in `FLAGS`, never a cancel.
`TAKEOVER_ACTION=stop` and `ACCEPT_STOP=<x>` turn them into stops.

- **Takeover rate: flag outside [0.50, 0.95], judged once 100 relay episodes have finished.**
  - **What 09-21 did.** With the 32k context budget, 09-21 handed over in 0.89 of finished episodes on this gate's
    own measure (attempts 6a and 6b, 09-27/28, the untouched CalibForge tasks; 0.78–0.85 of all uncensored router
    episodes in the earlier full runs). About three quarters of the handovers were context_budget, most of the rest
    done_claim.
  - **What to expect from arm A.** A better student should shift handovers from context_budget to done_claim, not
    remove them, because every done claim is handed to the teacher.
  - **Below 0.5.** Most episodes end without a teacher turn. Either the student's claims no longer parse, or it loops
    or times out without claiming done. The rows would then be mostly repair-only.
  - **Above 0.95.** Nearly every episode hands over, more often than 09-21 did. Check the takeover turn (per-trigger turn p50 in the readout)
    for early false claims.
  - **In the readout.** The final readout's S3 uses the same band (`--takeover-min` / `--takeover-max`). It judges the
    rate without deadline-censored episodes, which are mostly episodes cut before any trigger could fire.
- **Draft acceptance: flag below a mean acceptance length of 1.8** (`ACCEPT_FLAG`).
  - **How it is measured.** The value is 1 + accepted / drafts, from the student servers' vLLM spec-decode counters,
    over the last window of at least 2,000 drafts.
  - **Reference.** 09-21 with its adapted EAGLE-3 draft reads about 2.5–2.6.
  - **Why only a flag.** Low acceptance costs speed, not data quality.

## Measured on 2026-09-29 (driver node c102-009, CalibForge snapshot, 8 sandboxes)

The load test is `daytona_probe.py load`. Each seat runs its own session and tmux, driven by harbor's
execute / poll / logs protocol: tmux send-keys, capture-pane, a small file check, and a 64 KB output every 8th call.

| seats | pattern | calls | errors | p50 / p99 s | 64 KB output p99 s | login ssh CPU |
|---|---|---:|---:|---|---|---|
| 64 | 35 s gap every 4th call (reconnects) | 1,280 | 0 | 0.28 / 0.88 | 1.0 | ≤ 2 % |
| 128 | same | 2,560 | 0 | 0.28 / 1.00 | 1.2 | ≤ 2 % |
| 192 | same | 3,840 | 0 | 0.24 / 1.13 | 1.5 | ≤ 5 % |
| 384 | same | 7,680 | 0 | 0.23 / 2.29 | 2.8 | ≤ 5 % |
| 192 | no gaps, 2 tunnels (151 calls/s) | 18,451 | 0 | 0.23 / 0.55 | 0.9 | ≤ 3 % each |
| 384 | no gaps, 2 tunnels (301 calls/s) | 36,720 | 0 | 0.22 / 1.22 | 2.0 | ≤ 8 % each |

Other checks:
- **Harbor end to end.** Harbor's nop agent ran through the bridge on 3 tasks, one per snapshot. All 3 were verified,
  in 36 s, with no exceptions.
- **The driver job.** `launch_driver.sh` → `driver.sbatch` passed the full pre-flight on a 4-task test tree.
- **The transfer.** A push → pull round trip worked, with sha256 checked and the run dir linked. The test files were
  removed from the repo afterwards.

## Transfer layout

**In the repo**, per run: `<run>/run.tar.zst` holds the run dir. `<run>/<job dir>/meta.tar.zst` holds each harbor job
dir's own files, and `<run>/<job dir>/trials_NNNN.tar.zst` holds 500 trial dirs each. `<run>/MANIFEST.json` records
sha256, episodes and files per unit. `<run>/DONE` is written last.

**On Jupiter:**
- Everything goes to `/e/data1/mmlaion/lee27/relay/horizon/<run>/`: `run/` and `jobs/`, with `run/jobs -> ../jobs`.
- `$E/runs/<run>` is a symlink there, so `readout.py`, `merge_runs.py` and `select_kept.py` treat it as a Jupiter run.
- Only that symlink lands on reformo fscratch.
- `MAX_NEW_FILES` (default 400k) caps the inodes one pull creates. A relay trial is about 11 files.
