# Snowball agentic RL on Horizon

**Status (2026-10-01): the launcher works end to end; port gate 5 and the two pass@8 screens are running.** Results go
in § Results as they land. The arms reuse Jupiter's successful SFT→GRPO arm `snowball_ttband_ota3d517u_b` verbatim
(SWE-bench Verified random-100 .241 → .307 in 30 steps) with Horizon's transport and Daytona as the only changes.

## How a run is built

Everything lives in `data/r2egym/horizon/rl/` (this branch) and runs on the login node with stdlib python3:

1. **Render** a run dir with `make_arm.py`. It copies a Jupiter reference (`refs/`, vendored, key redacted) and changes
   only paths, model, data, layout, seats and the Horizon transport; every anchor it touches is asserted.
   - `--recipe gate5`: 62ft5sky's hydra args (Stage-3 step 1888, KL 0.01, no draft), the port-gate-5 reference.
   - `--recipe arm`: `snowball_ttband_ota3d517u_b`'s hydra args (rloo_n, no KL, clip .2/.2, sequence_mean, TIS cap 2,
     lr 5e-7, warmup 3, staleness 2, n 8, batch 64, grouped_mm, temp 1, 65,536 / 16,384 / 49,152, summarize off,
     agent 1,800 s, verifier 2,400 s with preserve-on-timeout off, connect timeout 120 s, 16 coordinators, overflow = 0).
   - `--probe K`: eval-only screen (Jupiter's `refresh_screen.py` probe): eval-before-train with K attempts per tree
     entry, zero epochs, trials kept on /scratch, the job stops at `WANDB_MIRROR kind=eval step=0`. The harbor runner
     samples eval trials with the training sampler (temperature 1.0), not `eval_sampling_params` (greedy).
   - `--drop KEY` / `--set KEY=VALUE` for anything else.
2. **Launch** with `launch_arm.sh <run dir>` from `~/snowball/ota-rl`. Pre-flight, all read-only: the hydra args compose
   against the installed MarinSkyRL schema; every environment hash of the tree has a snapshot in the org (a missing
   one would make harbor try to build a snapshot, which is forbidden at the cap); the org's started sandboxes plus the
   run's seats stay under 1,200. Then `sbatch` and one login-side `tunnel.sh` per port (18080, 18081).
3. **Watch** with `rl_watch.sh [run ...]`: job state, log age, the `DP rank -> node` lines (each engine's 4 ranks must
   share a node), train/eval steps, probe results, tracebacks, last-step tis/reward/entropy.
4. **Read a screen** with `screen_report.py --out <dir> --repo-map <coverage.csv> --band <train.txt> <run dir>`:
   per-task attempts/solves, the band (solved ≥ 1 of 8), per-repo counts; masked/unscored trials are reported, never
   counted as failures.

Run dirs: `/scratch/11584/lukedhlee/experiments/rl/<name>/` (configs, sbatch, logs, gateway, trials for probes).

## What the job does on Horizon (deltas from the Jupiter sbatch)

- **Two checkouts.** The RL runner imports OTA from `~/snowball/ota-rl-runtime`, branch `lukedhlee/horizon-rl` =
  f3edec45 (`lukedhlee/rl_acceleration`, the commit port gate 1 verified as Jupiter's RL stack) + a `horizon` entry in
  `hpc/hpc.py`. This branch's `hpc/` is 160 files behind it and lacks helpers the template sources. The scripts
  (bridge, shm_prune) come from `~/snowball/ota-rl` (this branch). Venv `~/snowball/envs/snowball`, MarinSkyRL a03b2773,
  harbor dcf609bc, vLLM fa50698a + the EAGLE-3 overlay, as gate 1.
- **Daytona egress.** Each node runs `node_bridge.sh` → `data/relay/horizon/socks_connect_bridge.py` on 127.0.0.1:18946
  over its two `ssh -R` SOCKS ports; `HTTPS_PROXY` points there, `HTTP_PROXY` is unset (vLLM traffic is plain http
  inside the cluster). The sbatch checks the Daytona API with the key (12 tries) before starting Ray.
- **Daytona backend.** `environment_type=daytona`, `auto_snapshot=true` (the 11 restored per-repo snapshots), create
  pacing 5/s shared by the coordinators, Daytona infra errors masked (as Jupiter's `build_ota_darm.sh`).
- **No modules**, `CC=gcc CXX=g++`, compile caches node-local under `/tmp/rlcache_<job>`, no artifact store (trials on
  `/dev/shm`, pruned by `shm_prune.sh`; probes keep theirs on /scratch).

## Horizon traps found while porting (all handled in make_arm.py)

- **XALT LD_PRELOAD.** TACC preloads `libxalt_init.so` into every process; it prints an NVML stub warning on stdout,
  which lands in every `$(...)`. The Daytona key read back as 978 characters and every authenticated call failed while
  the unauthenticated health check passed (jobs 39600/39601). The sbatch unsets `LD_PRELOAD` on its first line.
- **Mixed IB/RoCE HCAs.** Nodes have 4 InfiniBand HCAs (`mlx5_0/1/4/5`, 800 Gb/s) and 2 RoCE (`mlx5_2/3`). Unpinned,
  NCCL paired an IB device with a RoCE one across nodes and the first policy→engine weight broadcast died
  ("Remote IB device is incompatible", jobs 39609/39610). `NCCL_IB_HCA==mlx5_0,mlx5_1,mlx5_4,mlx5_5` (same names on
  28 of 28 nodes checked).
- **`chat_template_content_format=string`** (62ft5sky) came from a Jupiter-only MarinSkyRL branch; the installed stack
  rejects it at engine start, so gate 5 drops it (the newstack arms never had it).
- **TACC sbatch banner.** `sbatch --parsable` prints a banner first; the job id is the last line.

## Models

The 09-21 relay SFT exports (H9 `relay_acont` step 999, H8 `relay_allkimi` step 1203) carry their own
`chat_template.jinja` (= the tokenizer's default, "Reasoning: /think"), the template the gist's evals served; SkyRL uses
the tokenizer default, so rollouts and evals render the same. No reasoning parser, `skip_special_tokens=false` per
request, frozen router bias (`grug_query_bias_update_mode=frozen`), 65,536 context via `hf_overrides`. The EAGLE-3
draft (trained on the Stage-3 line) reaches a mean acceptance length of 2.14 (H9) / 2.02 (H8) in the gist's eval serves.

## Results

(filled in as they land: gate 5 verdict, screen bands, arms, evals)
