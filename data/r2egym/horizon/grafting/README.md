# Snowball: transfer the V7 SFT delta onto direct RL

Build `baserl30 + alpha * (V7 - 09-21)` for alpha 0.25, 0.5, 0.75 and 1.0. Every input starts at or after 09-21. This experiment requires no new training.

`build_grafts.py` checks complete tensor inventories, configs, tokenizers and templates before creating outputs. It copies target router biases and any `pending_qb_betas` exactly, including dtype. Weight arithmetic is FP32, with output cast to the target dtype (BF16 weights). Processing one target shard at a time bounds memory; every saved tensor is checked against the computed tensor. Existing outputs are never overwritten. `graft_manifest.json` records sources, revision, alpha, state policy and per-tensor delta norms. `graft_complete.json` and `build_complete.json` are written only after validation.

Run from an isolated committed checkout on Horizon, in a detached login-side tmux session:

```bash
export GRAFT_CODE=/scratch/11584/$USER/experiments/snowball-grafting-20261030/code
bash "$GRAFT_CODE/data/r2egym/horizon/grafting/run.sh"
```

The controller submits a one-node, one-hour compute build through the existing project booking guard. After a successful build it waits for room under the existing 900-seat eval admission threshold, then delegates to the matched Horizon `main_eval.sh` stack: TB2.1, SWE random100 and TB-lite, two reps. Four models × two reps × five nodes × 3.5 hours is at most 140 requested node-hours for initial evaluation, plus one for the build. Historical `baserl30` evaluations are the reference; no baseline rerun is submitted automatically.

Outputs, logs and controller status live under `/scratch/11584/$USER/experiments/snowball-grafting-20261030`. Eval tags are `graft25`, `graft50`, `graft75`, `graft100`; day is `graft_20261030`. Eval results live under the existing `sft_eval/tb2_jobs` tree. Inspect coverage and rerun only infrastructure losses using the handoff's tools before the fair paired readout. TB2.1 overlaps CalibForge; SWE and TB-lite are the clean agent benchmarks. Chat/math retention is unmeasured by this sweep.

No uploads, DB writes, snapshot mutations, changes to shared runtime checkouts, earlier donor checkpoints or new RL runs are part of this setup.
