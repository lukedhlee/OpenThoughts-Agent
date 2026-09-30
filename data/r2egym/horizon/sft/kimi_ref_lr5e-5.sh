#!/bin/bash
# kimi_ref_lr5e-5.sh — the Horizon twin of Jupiter's Kimi SWE-smith full arm (job 1867383, 2026-09-17): same data,
# init, marin commit, lr 5e-5, 3 epochs = 96 steps on 64 ranks. The port gate compares its train loss with Jupiter's
# (SFT.md). Data: build it first with data/swesmith/kimi_sft_convert.py (SFT.md step 1).
#   tmux new -d -s sft_kimi "bash data/r2egym/horizon/sft/kimi_ref_lr5e-5.sh"
S=${SNOWBALL_SCRATCH:-/scratch/11584/$USER/snowball-sft}
export SNOWBALL_STAGE=kimi_swesmith
export SNOWBALL_PARQUET_LIST=$S/data/kimi_swesmith_v1/parquet.list
export SNOWBALL_DATASET_ID=open-athena/Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k
export SNOWBALL_DATASET_REVISION=1b87a9cf78b897a2355105d47afd44d818cad25d
export SNOWBALL_CACHE=$S/experiments/snowball-kimi_swesmith-sft/cache-v1
export SNOWBALL_OUTPUT=${SNOWBALL_OUTPUT:-$S/experiments/snowball-kimi_swesmith-sft/full-lr5e-5-ep3}
export SNOWBALL_RUN_ID=${SNOWBALL_RUN_ID:-snowball-kimi-full-lr5e-5-ep3-horizon}
export SNOWBALL_LR=5e-5 EPOCHS=3 SNOWBALL_WALL=${SNOWBALL_WALL:-01:00:00}
export CHAIN_STEPS=${CHAIN_STEPS:-prep gate import run export}
exec bash "$(dirname "$0")/horizon_sft_chain.sh"
