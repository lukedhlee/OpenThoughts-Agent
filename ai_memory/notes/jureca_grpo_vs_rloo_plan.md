# GRPO vs RLOO A/B on JURECA (sandbox-free gsm8k, lee27) — DONE

The completed clean A/B that found GRPO and RLOO statistically tied, with the source-verified proof that
only one knob differed, the fairness table, and the offline-uv/no-internet setup recipe.
Read as the template for a fair single-knob A/B, or for the JURECA SkyRL offline setup. The estimator
question itself is settled — see [[decisions]].

Status: **DONE (2026-07-13).** Both arms COMPLETED (GRPO job 15426255, RLOO 15426256), synced to WandB
`lukeleeai/gsm8k-grpo-vs-rloo` (runs nsb1bipp=grpo, dt0pjlma=rloo).

## RESULT — held-out gsm8k test pass@1 (Qwen2.5-1.5B, 1319 problems, 5 epochs / 35 steps)
| step | GRPO | RLOO |
|------|------|------|
| 0 (baseline) | 8.26% | 7.81% |
| 5 | 71.19% | 71.57% |
| 10 | 74.30% | 74.53% |
| 15 | 75.51% | 75.44% |
| 20 | 77.18% | 76.95% |
| 25 | 77.18% | 77.48% |
| 30 | 77.48% | 77.33% |
| **35 final** | **77.71%** | **78.24%** |
**Conclusion: statistically TIED** (track within ±0.5pp every checkpoint; RLOO +0.53pp final = noise).
Both lift base ~8% → ~78% on held-out (generalization). Clean A/B: only `advantage_estimator` differed.
NOTE: per-step eval jsonl dumps at `~/exports/dumped_evals/global_step_*` are UNRELIABLE for the A/B —
both arms wrote the same paths concurrently (last-writer-wins). WandB is the source of truth.

Goal (orig): get familiar with the cluster + produce a clean, fair GRPO-vs-RLOO comparison on a held-out
test set, both tracked in one WandB project, run in parallel.

## SETUP COMPLETE (2026-07-13) — verified on JURECA
- uv 0.11.28 at `~/.local/bin/uv`.
- SkyRL: `git clone https://github.com/NovaSky-AI/SkyRL.git` → `/p/project1/synthlaion/lee27_jureca/code/SkyRL`,
  pinned `f10c3959` (== shared checkout, NO leaked token). uv project = `skyrl-train/`.
- venv (11 GB) at `/p/scratch/synthlaion/lee27/envs/skyrl-venv`: **python 3.12.13, torch 2.7.0+cu128, vllm 0.9.2,
  skyrl_train** (imports from our checkout). Built with `uv sync --extra vllm`; run offline with `uv run --no-sync`.
- Data: `/p/scratch/synthlaion/lee27/data/gsm8k/{train.parquet(7473), validation.parquet(1319=test)}`.
- Model cached offline: `Qwen/Qwen2.5-1.5B-Instruct` (2.9 GB) in `$HF_HOME=/p/scratch/synthlaion/lee27/cache/hf`.
- Paths env: `source /p/project1/synthlaion/lee27_jureca/env.sh` (JSC_*, HF_HOME, UV_*, WANDB_DIR, TMPDIR, PATH).
- Secrets: `~/.config/otagent/secrets.env` (600, private lee27 group) — WANDB/DAYTONA/SUPABASE all present.
- Run pattern (compute node, offline): `cd skyrl-train && uv run --no-sync -m skyrl_train.entrypoints.main_base ...`.

## Finalized sizing
- ~7–8 steps/epoch (7473/1024). Smoke = 2 epochs (~15 steps, devel ≤2h). A/B = 8 epochs (~58 steps, dc-gpu 24h cap).
- 4 GPU/node (A100-40GB), 1 node/experiment → A/B = 2 nodes parallel. Account `synthlaion`.
- QOS: dc-gpu = 24h/job, 64 running/user; dc-gpu-devel = 2h/job, 2 running/user, 4 nodes.

## The experiment in one sentence
Train Qwen2.5-1.5B-Instruct on gsm8k **train** with SkyRL, twice — identical everything **except**
`trainer.algorithm.advantage_estimator` (`grpo` vs `rloo`) — and compare eval accuracy on the held-out
gsm8k **test** set (logged as `data.val_data`) in WandB.

## Why this is a valid A/B (verified from source)
`SkyRL/skyrl-train/skyrl_train/utils/ppo_utils.py` — the two estimators differ by exactly one thing,
both averaging over the `n_samples_per_prompt` rollouts that share a prompt index:
- **GRPO** `compute_grpo_outcome_advantage`: `adv = (score − group_mean) / (group_std + ε)` (std-normalized).
- **RLOO** `compute_rloo_outcome_advantage`: `adv = (score − leave_one_out_mean) × n/(n−1)` (LOO baseline, no std norm).
- Native SkyRL calls it **`rloo`** (NOT `rloo_n` — that's the OTA/MarinSkyRL fork's name). Valid values here:
  `gae, grpo, rloo, reinforce_pp`.
- Both **require `n_samples_per_prompt > 1`** (RLOO logs a warning and zeroes advantage if n=1). Example uses n=5.

So a fair comparison flips ONE knob. Everything else held constant (see fairness table).

## Runtime decision: native SkyRL + uv (recommended) vs Marianna's conda (fallback)
- **Recommended (A):** our own clone of **NovaSky-AI/SkyRL** (public; pin to the same commit as the shared
  checkout at `/p/project1/laionize/dcagent-shared/SkyRL`, currently `f10c395`). Turnkey example scripts
  (`skyrl-train/examples/gsm8k/run_gsm8k.sh`), and we've read the exact estimator code. Self-contained/reproducible.
  - Clone from the **public** repo, NOT the shared checkout (its git remote embeds a leaked EtashGuha PAT — flag for rotation).
- **Fallback (B):** activate `/p/project1/ccstdl/envs/marianna/py3.12/` (verified: torch 2.10+cu128, vllm 0.18,
  skyrl_train importable) and run `python -m skyrl_train.entrypoints.main_base ...` directly. Uses her fork; only if uv path stalls.

### JSC no-internet gotcha (critical)
The example's `uv run --isolated --extra vllm` **resolves deps live → needs internet**. Compute nodes have NO internet.
So:
1. On the **login node** (has internet): `uv sync --extra vllm` once to build the venv + prefetch wheels.
2. In the **job** (compute node): run **offline** — `uv run --offline --no-sync -m skyrl_train.entrypoints.main_base ...`,
   or activate the pre-built venv and use plain `python -m ...`.
3. Point uv at scratch so the venv is NOT an inode bomb on project1:
   `export UV_CACHE_DIR=/p/scratch/synthlaion/lee27/cache/uv`
   `export UV_PROJECT_ENVIRONMENT=/p/scratch/synthlaion/lee27/envs/skyrl-venv`

## Paths (one project = synthlaion; code on project1, everything heavy on scratch)
```
/p/project1/synthlaion/lee27_jureca/code/SkyRL/     # git clone (small inode footprint, ~278k headroom OK)
/p/scratch/synthlaion/lee27/
  envs/skyrl-venv/        # uv venv  (2.3M inode headroom — SAFE)
  cache/{uv,hf,pip,torch} # HF_HOME etc.
  data/gsm8k/             # train.parquet + validation.parquet
  experiments/<run>/      # SLURM .out logs
  wandb/                  # offline wandb runs → sync from login node
```
Env vars: `HF_HOME`, `HF_HUB_CACHE`, `XDG_CACHE_HOME`, `UV_CACHE_DIR`, `UV_PROJECT_ENVIRONMENT`, `WANDB_DIR`,
`TMPDIR` all under `/p/scratch/synthlaion/lee27/...` (see jureca_what_goes_where.md).

## Held-out eval — already built into the example
`gsm8k_dataset.py` maps gsm8k **train** split → `train.parquet` (7473) and gsm8k **test** split →
`validation.parquet` (1319). SkyRL evaluates on `data.val_data`, so the held-out test IS gsm8k test.
`trainer.eval_before_train=true` logs the **step-0 baseline** (base model, no RL); `trainer.eval_interval=5`
logs during training. → exactly the "baseline vs RL-ed on a held-out set" the PI wants, per algorithm, in WandB.

## WandB (no-internet → offline + sync)
- Same **project** for both runs so curves overlay: `trainer.project_name=gsm8k-grpo-vs-rloo`,
  `trainer.run_name=grpo_1p5b` / `rloo_1p5b`.
- Compute node offline: `export WANDB_MODE=offline`, `WANDB_DIR=/p/scratch/synthlaion/lee27/wandb`.
- After (or during, from login node): `wandb sync /p/scratch/synthlaion/lee27/wandb/offline-run-*`.
- Set `WANDB_API_KEY` (your own account) + `WANDB_ENTITY` (your entity, or `dogml` if you're a member).
- Live alternative (later): proxychains SOCKS tunnel → `WANDB_MODE=online`. Offline+sync first.

## Fairness table — hold ALL of these identical across the two runs
| Knob | Value | Note |
|------|-------|------|
| base model | Qwen/Qwen2.5-1.5B-Instruct | same weights |
| train data | data/gsm8k/train.parquet | identical |
| held-out eval | data/gsm8k/validation.parquet (=gsm8k test) | identical |
| n_samples_per_prompt | 5 | REQUIRED >1 for both |
| train_batch_size / policy_mini_batch_size | 1024 / 256 | identical |
| lr | 1.0e-6 | identical |
| epochs | e.g. 8 (fits walltime; a knob) | identical |
| eval_before_train / eval_interval | true / 5 | baseline @ step0 + curve |
| **use_kl_loss** | **same for both** (recommend true, matching 1.5B example) | see caveat |
| seed | set `trainer.seed` same in both | reduce variance |
| **advantage_estimator** | **grpo vs rloo ← THE ONLY DIFFERENCE** | |

**KL caveat:** canonical RLOO/LOOP often runs *without* KL. But dropping KL for RLOO-only confounds the A/B
(two variables change). For a clean comparison, hold `use_kl_loss` identical. If you later want "canonical RLOO,"
run it as a separate third arm.

## The two commands (differ only in the marked lines)
Base on `examples/gsm8k/run_gsm8k.sh`, override for offline + our paths + one project:
```bash
# common overrides (both):
  data.train_data="['$DATA/train.parquet']" data.val_data="['$DATA/validation.parquet']" \
  trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct" \
  trainer.placement.colocate_all=true trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=4 trainer.placement.ref_num_gpus_per_node=4 \
  generator.num_inference_engines=4 generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=8 trainer.train_batch_size=1024 trainer.policy_mini_batch_size=256 \
  trainer.micro_forward_batch_size_per_gpu=64 trainer.micro_train_batch_size_per_gpu=64 \
  trainer.max_prompt_length=512 generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 trainer.algorithm.use_kl_loss=true \
  generator.n_samples_per_prompt=5 generator.gpu_memory_utilization=0.8 \
  trainer.eval_before_train=true trainer.eval_interval=5 trainer.seed=42 \
  trainer.logger=wandb trainer.project_name=gsm8k-grpo-vs-rloo \
# run GRPO:
  trainer.algorithm.advantage_estimator=grpo  trainer.run_name=grpo_1p5b \
  trainer.ckpt_path=$SCRATCH/experiments/grpo_1p5b/ckpts
# run RLOO:
  trainer.algorithm.advantage_estimator=rloo  trainer.run_name=rloo_1p5b \
  trainer.ckpt_path=$SCRATCH/experiments/rloo_1p5b/ckpts
```

## Parallelism
Two independent SLURM jobs, **1 node × 4 GPU each** → they run concurrently (12 devel / 180 prod nodes; budget trivial).
`--account=synthlaion`. Submit both with `sbatch`; they schedule independently.

## Staged rollout (nothing runs until you say go)
- **Setup (no GPU, no budget):** create dirs → clone SkyRL (public) → set env vars → **on login node**:
  `uv sync --extra vllm`, prefetch model+dataset into HF cache, run `gsm8k_dataset.py` → parquet on scratch.
- **Smoke (dc-gpu-devel, ≤2h):** one job, `epochs=2`, confirm reward moves + eval logs + `wandb sync` works.
- **A/B (dc-gpu, parallel):** two jobs (grpo, rloo), `epochs=8`, submitted together. Compare eval-accuracy
  curves on the held-out gsm8k test in the shared WandB project; step-0 = shared baseline.

## Open items to confirm before running
1. Account = `synthlaion` (assumed). 2. `dc-gpu` per-job walltime cap (typically 24h). 3. Your WandB entity/key.
4. A100-40GB fits n=5 × batch at gpu_mem_util 0.8 for 1.5B (very likely; smoke confirms).
5. Flag leaked EtashGuha PAT in the shared SkyRL remote → rotate (relay to Etash/PI).
