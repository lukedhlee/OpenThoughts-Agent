# Snowball SFT on Horizon (Marin JAX / Levanter Grug chain)

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

## Verdict

(filled in when the gate run finishes; pass rule in `~/briefs/train-port.STATUS.md`, written before the run)
