# Snowball SFT on Horizon (Marin JAX / Levanter Grug chain)

**Verdict (2026-09-30): Horizon trains Snowball SFT like Jupiter under the current recipe** (Grug Datakit 09-21 base,
router bias frozen at its non-zero value, 16 x 65,536 on 4 nodes). The 30-step port pair passed its pre-registered rule
(Horizon − Jupiter train loss: 0.0013 mean |Δ|, 0.0007 at step 0; marin `HORIZON_JUPITER_PAIR.md`), and Horizon's retrain
of Jupiter's relay SFT arm A scores the same held-out NLL as Jupiter's arm A (0.5814 vs 0.5834; 09-21 is 0.6428). The
older recipe below (Stage-3 import, per-batch router bias) trains the same model, but its loss curve sits ~0.010 off.

Horizon runs the same Levanter Grug SFT chain as Jupiter, on the same 64 one-GPU ranks (16 nodes x 4 GB200). The port
changes no Python: the marin branch `lukedhlee/horizon-snowball-sft` (lukedhlee/marin fork) is Jupiter's
`lukedhlee/vista-snowball-sft` at a3840f826 plus `horizon_*` copies of the Jupiter sbatch files, and the chain runner
here (`sft/horizon_sft_chain.sh`) is the Jupiter one with Horizon paths. Port-gate verdict and numbers: § Verdict.

## What differs from Jupiter

- **Env build runs on a compute node.** Horizon's login node is shared and compute nodes have no internet, so
  `horizon_snowball_env.sbatch` waits for `tunnel.sh <jobid>` (login side) and runs `uv sync --frozen` from the branch
  lock through the SOCKS proxy, then the NCCL 2.30.7 override, a CPU import smoke and a 4-GPU JAX smoke. ~7 min.
- **Compiler and CUDA.** TACC's login env loads `nvidia/26.9` (`CC=nvc`); every script sets `CC=gcc CXX=g++` (system
  GCC 14) and `CUDA_HOME=/home1/apps/nvidia/Linux_aarch64/26.9/cuda/13.3`. JAX itself uses the pip CUDA 13 wheels.
- **Geometry.** `-p debug -A CCR24067`, 144 CPUs per node (36 per rank at 4 ranks), `--gres=gpu:4`, same `srun
  --gpu-bind=none` one-device-per-rank split.
- **NCCL bootstrap** on `ib*` (compute nodes name their IPoIB ports `ibs2`, `ibP2p1s0`, `ibP16s4`, `ibP18p1s0`; the
  login node's `ibp1s0` is not what compute nodes have).
- **Paths.** Code and env in `/home1` (`~/snowball/marin-sft`, `~/snowball/envs/marin-grug-sft`, 11 GB: the home quota
  went 45 % -> 69 %), data, caches, checkpoints and logs in `/scratch/11584/$USER/snowball-sft` (purged after 10 days
  untouched; native checkpoints are ~625 GiB each, delete them once the export is verified).

## How to run (the Kimi SWE-smith reference, or your own stage)

1. **Code:** `git clone -b lukedhlee/horizon-snowball-sft https://github.com/lukedhlee/marin.git ~/snowball/marin-sft`
   (use `--filter=blob:none`); OT-Agent job checkout `~/snowball/ota-train` (this branch).
2. **Env (once):** from `/scratch/11584/$USER/snowball-sft/logs`:
   `J=$(sbatch --parsable --export=ALL,MARIN_ROOT=$HOME/snowball/marin-sft,SNOWBALL_ENV=$HOME/snowball/envs/marin-grug-sft,SNOWBALL_UV_CACHE=/scratch/11584/$USER/cache/uv-grug $HOME/snowball/marin-sft/experiments/june_tpu_67b_a2b/moe/horizon_snowball_env.sbatch | tail -1)`
   then `setsid nohup bash ~/snowball/ota-train/data/r2egym/horizon/tunnel.sh $J > tunnel.$J.log 2>&1 < /dev/null &`.
   Done at `ENV_BUILD_OK`. (Horizon's sbatch prints a banner; take the last line of `--parsable`.)
3. **Data:** a parquet with a `conversations` column in serve format (think spans as `<|start_think|>` /
   `<|end_think|>` special tokens) plus `parquet.list`. The Kimi set rebuilds from HF in ~4 min on the login node:
   `hf download --repo-type dataset open-athena/Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k --revision 1b87a9cf78b897a2355105d47afd44d818cad25d --local-dir src_kimi`,
   then `data/swesmith/kimi_sft_convert.py` with the id lists in `ai_memory/active/snowball-sft/data/`
   (`taskindex_to_instance.json`, `kimi_train_ids.txt`, `kimi_heldout_ids.txt`) and the Stage-3 snapshot as tokenizer.
   Expect 4,398 train rows / 66,269,429 tokens / 628 held-out.
4. **Chain (login node, tmux):** `tmux new -d -s sft_kimi_hz "bash ~/snowball/ota-train/data/r2egym/horizon/sft/kimi_ref_lr5e-5.sh"`.
   It submits prep (1 node) -> gate (2 nodes) -> import (4 nodes, CPU) -> run (16 nodes) -> export (4 nodes, CPU), each
   gated on its OK marker; log `/scratch/11584/$USER/snowball-sft/logs/chain.log`. For another dataset copy the
   wrapper and change the stage, parquet list, dataset pin, output and run id (the stage must exist in marin STAGES).
5. **Read the loss:** the run log's tqdm lines (`Progress on:train N.0it/96.0it ... loss=`) in
   `logs/snowball-<stage>.<job>.log`.

## Verdict (2026-09-29)

**It runs and trains the same model, but its loss curve is not Jupiter's.** Horizon ran the Jupiter Kimi SWE-smith
arm (job 1867383: lr 5e-5, 96 steps, same data, init and commit). The resulting model scores 0.3698 held-out NLL
against Jupiter's 0.374, and the base scores 0.5857 against 0.586. The training loss, though, sits ~0.010 below
Jupiter's at every reference step and 0.007 above it at init. That offset is about 10x Horizon's own run-to-run noise
(an identical rerun differs by 0.001 per step), so it is systematic. It fails the pass rule written before the run
(step 2 within ±0.005; mean |Δ| ≤ 0.007) and passes its per-point clause (max 0.0132 ≤ 0.0135).

| | Jupiter | Horizon |
|---|---|---|
| loss at step 0 / 9 / 29 / 59 / 95 | .628 / .447 / .395 / .379 / .378 | .635 / .436 / .382 / .370 / .366 |
| held-out NLL, base | 0.586 | 0.5857 |
| held-out NLL, step-96 export (think / rest) | 0.374 (0.588 / 0.279) | 0.3698 (0.583 / 0.276) |
| step time, 16 nodes | 6.4 s | 3.6 s |

Why the curves can differ while the models match: the trainer's step-0 loss is not a plain forward pass. vLLM scores the
same first batch at 0.582, and both trainers log ~0.63. With the plain Stage-3 import (`pending_qb_betas` zeroed), the
Grug router recomputes its balancing bias from each batch's routing quantiles. That state depends on numerics and can
move with the GPU. This is likely but not proven. So compare Horizon and Jupiter SFT runs by held-out NLL of the
export, not by train loss at the 0.01 level.

Tools: `sft/heldout_nll.sbatch` (held-out NLL through vLLM; `MODEL=<export> NAME=<tag>`) and `sft/batch0_nll.py`
(rebuild a step's training batch and score it with vLLM). The per-step loss is in the offline W&B run under
`~/snowball/marin-sft/wandb/`. tqdm's `N.0it ... loss=` shows the loss of step N−2.

## Relay SFT arms and their evals (current recipe, 2026-09-30)

Horizon retrained Jupiter's 09-28 relay SFT arms A (relay, 246 steps) and B (Qwen alone, 147 steps) on Jupiter's rows.
Everything runs from the login node in tmux; each step is a Slurm job.

1. **Train one arm:** `ARM=relay|qwen bash data/r2egym/horizon/sft/relay_sft_arm.sh` (marin branch
   `lukedhlee/horizon-snowball-sft-0921`, init `init-dk0921-step0` from `pair_kimi0921.sh`'s import). It converts the
   rows and counts packs on a compute node, stops unless the steps per pass equal Jupiter's (A 82, B 49), builds the cache,
   trains (~7 s/step), exports every pass, and syncs W&B (`lukedhlee-marin/horizon-relay-sft`). About 30 min of training
   per arm. The trainer's tqdm lines stop updating early; `GUARDED_RUN_EXIT rc=0` is the end signal.
2. **Evaluate:** `MODELS="hzA=<export>" GROUPED=1 bash data/r2egym/horizon/sft/eval_sft.sh` runs TB2.1, SWE-bench
   Verified random-100 and TB-lite 3 times each, with Jupiter's policy (harbor-p0924 @ 761fb516, 65k/16k, 16 concurrent,
   one serve node per run). The per-user cap is 20 running jobs, so the harbor driver runs inside its serve job, and
   `GROUPED=1` puts a rep's 5 runs in one 5-node job. A rep takes ~2–2.5 h (TB2.1 is the long one).
   Readout: `eval_readout.py --tags hzA hzB --day <YYYYMMDD> [--ref jupA=tb21:<Jupiter job dir> ...]`.
3. **Held-out NLL of an export:** `relay_heldout_docs.py` builds unseen relay episodes in the arms' ids + loss format.
   Score them with `SCRIPT=.../batch0_nll.py PARQUET=<docs> HELDOUT_MAX_MODEL_LEN=66560 HELDOUT_MAX_POS=131072
   HELDOUT_MAX_LEN=65536 MODEL=<export> NAME=<tag> sbatch heldout_nll.sbatch` (1 node, ~10 min).
