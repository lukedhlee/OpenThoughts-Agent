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
     agent 1,800 s, verifier 2,400 s with preserve-on-timeout off, connect timeout 120 s, 16 coordinators; ContextLengthExceeded and agent timeouts are passthrough: the trial keeps its verifier reward).
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

- **Port gate 5: FAIL by the written rule, by 4 %.** Step-1 `policy/tis/log_ratio_abs_mean` 0.03679 vs 62ft5sky's 0.03531
  (steps 2-3: 0.03729, 0.03700), job 39668, Stage-3 step 1888 on the full Daytona v3 pool, 128 seats. Token alignment
  exact, importance ratio mean 1.00002, 0.04 % of tokens capped, 0 masked. Read as a token-set difference (different pool
  and backend; trajectories at the 49k cap, mean 45k tokens), not numerics: on identical tokens the 09-29 proxy put
  Horizon's trainer at 0.0304 vs Horizon vLLM and 0.0309 vs Jupiter's vLLM. The arms went ahead (Luke's call to revisit).
- **Where the trainer-vs-vLLM log-ratio comes from (`rl/tis_decomp/`, 0.76 node-h).** It is mostly MoE router flips, not
  a trainer bug.
  - **The test.** 96 fixed turns per model were scored by vLLM and by the trainer: Stage-3 step 1888 (143k completion
    tokens) and H9 (42k tokens, from screen_h9). The table gives mean |Δ log p| with a 95 % bootstrap over turns.
  - **Router replay explains most of it.** Forcing vLLM's chosen experts into the trainer cuts the gap from 0.0304 to
    0.0074 on Stage-3 (−76 %) and from 0.0164 to 0.0066 on H9 (−60 %).
  - **Routing is near-tied.** 46 % of Stage-3 routing decisions have a top-4 vs 5th margin under 0.1 logit. So any bf16
    difference flips 13–19 % of expert sets, about 4 % at layer 0 and about 30 % by layers 20–24. vLLM's batched
    run-to-run noise (0.0255) is the same flip mechanism, and with one request at a time vLLM is bitwise deterministic.
  - **The trainer is clean.** It is bitwise reproducible, and grouped_mm matches the per-expert loop (0.00005). An fp32 LM
    head moves it by 0.0005. Swapping in fp32-score attention under fixed routing does not move it closer to vLLM.
  - **The non-routing residual (0.0074) is spread over many bf16 kernels.**
  - **Entropy sets the size; context length does not.** The gap is 0.0003 at < 0.01 nats, 0.045 at 0.3–1 nats and 0.12
    above 2 nats. H9 reads lower than Stage-3 because its tokens are more confident (mean entropy 0.28 vs 0.49, which
    accounts for about 90 % of the difference) and its router margins are wider.
  - **Same mechanism on gate 5's pool.** That fits gate 5's +4 % being a token-set difference.
- **Clean pool:** 1,227 tasks (1,199 sympy + 28 orange3) = 1,104 train + 123 held-out; the 09-21 SFT data covers every
  task of the other nine repos. Builder: `data/r2egym/horizon/pool/`.
- **Screens (K=8):** band H9 600 / H8 589 train tasks solved >= 1; train pass@1 .291 / .288; held-out pass@1 .501 / .509.
  66 % of trials end in ContextLengthExceeded (49,152-token budget).
- **EAGLE-3 draft at RL load:** 1.36-1.53x node output on H8/H9 (acceptance 2.3) -> kept.
- **Export:** `export_hf.sbatch` (2 min per checkpoint) verified on rl_h9 step 6.
- **Arm H9 (rl_h9, GRPO 30 steps from H9 on its 600-task band): PASS.** All 30 steps were healthy (tis 0.015-0.018,
  entropy 0.27-0.29, q >= 0.70, 0 masked; reward 0.49-0.66 early, 0.66 at step 30). Paired against H9, with 95 %
  bootstrap ranges:
  - Held-out clean R2E-Gym (123 tasks x 8): .497 -> .568, +.071 [+.039, +.105]. 51 tasks went up and 20 went down; 7
    were newly solved and 5 lost.
  - TB2.1: .208 -> .235, +.042 [-.002, +.086].
  - TB-lite: .282 -> .330, +.054 [-.003, +.116].
  - SWE-bench Verified random-100: .453 -> .418, -.035 [-.083, +.010] (without sympy -.033 [-.082, +.016]).
  - Uploaded as `laion/snowball-67b-a2b-rl-r2egym-acont-step30`.
- **Arm H8 (rl_h8, same recipe from H8 on its 589-task band): FAIL.** Training was as healthy as H9's (q 0.77-0.91,
  0 masked, reward 0.69 at step 30). Paired against H8:
  - Held-out: .511 -> .539, +.027 [-.004, +.059].
  - TB2.1: .188 -> .182, -.004 [-.059, +.049].
  - TB-lite: .272 -> .322, +.052 [-.012, +.117].
  - SWE random-100: .466 -> .401, -.066 [-.120, -.014].
  - Not uploaded.
- **Why SWE drops (both arms, pooled -4.4, p .02, all on the easy tasks).** GRPO mostly taught the model to finish
  inside the 49k training budget: task_complete went .46 -> .74 at a flat pass rate per declared trial, with no reward
  hacking. On SWE that shows up as:
  - half the turns;
  - no test-runner setup when pytest is missing, because the R2E-Gym images ship pytest and have no pip;
  - patches that break existing tests;
  - crash-site fixes.
  Reports: `experiments/rl/swe_diag/{stats_flips,behaviour,training_drift}.md`.
