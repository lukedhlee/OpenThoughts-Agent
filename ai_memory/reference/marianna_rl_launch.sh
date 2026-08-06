# ============================================================================
# MARIANNA'S r2egym RL LAUNCH SCRIPT (verbatim, PARTIAL — first ~90 of ~522
# lines; operator pasted it into the 2026-08-07 session and said "store it").
# Her cluster checkout (/p/project1/jureap59/...) is NOT readable by lee27, so
# this paste is our only durable copy. If the remaining 432 lines are ever
# needed, ask the operator or Marianna directly.
#
# THE FACTS THIS SETTLES (see ai_memory/marianna_parity.md for the distilled
# sheet):
#   - TIMEOUT_SEC=1800 (30 min/trial; DeepSWE reference = 5400; her own comment
#     says raise to 3600 for long-context runs)
#   - AGENT=terminus-structured, MAX_EPISODES=50, TP=4, 8 engines
# ============================================================================
#!/bin/bash

set -e

export PATH=$PATH:~/.local/bin
export SSH_KEY=~/.ssh/docker

# ============================================================================
# CONFIGURATION
# ============================================================================

HF=/e/data1/datasets/playground/ot/hf_hub
G1_SNAP=$HF/models--DCAgent--g1_diverse_tezos_100k_8b/snapshots/a5c4fa2b2c3580d2060773719144d113d115f7b8

export MODEL="${MODEL:-$G1_SNAP}"
# FULL r2egym pool (4578, swebench-fmt) — /e mirror of the /p transfernetx pool
# the user is g1-pass-rate filtering. Bigger set = less repetition (size was the
# bottleneck). Raw pool includes pass@8∈{0,1}-for-g1 tasks (0 RLOO gradient) until
# the g1 filter narrows it. ~71 steps/epoch at bs=64 -> EPOCHS=1 is plenty.
export DATASET="${DATASET:-$HF/r2egym_swebenchfmt_pool}"
export EVAL_DATASET2="${EVAL_DATASET2:-}"    # ID heldout is a subset of the pool -> would leak; drop it
export EVAL_DATASET3="${EVAL_DATASET3:-}"    # drop r2egymfmt swebench eval
export EPOCHS="${EPOCHS:-1}"                  # 1 epoch = 71 steps; capped below
# NOTE (empirical): the trainer stops ONE step early — MAX_STEPS=60 produced a last
# completed step of 59 (no step-60 ckpt). So pass target+1: MAX_STEPS=61 -> 60 updates.
# Keep raw and filtered runs on the SAME value so #updates is matched.
export MAX_STEPS="${MAX_STEPS:-61}"           # = 60 actual updates
export LR="${LR:-8e-6}"
# DECOUPLED eval: training saves an HF snapshot every 5 steps to exports/<run>/hf_ckpts/
# and does NOT block on eval; watch_and_eval_jupiter.sh fires a struct eval per ckpt.
export HF_SAVE_INTERVAL="${HF_SAVE_INTERVAL:-5}"
# max_ckpts_to_keep stays at the base default (3): only resumable ckpts are pruned;
# HF export ckpts (what the decoupled eval uses) are ALL kept regardless.
export EVAL_INTERVAL="${EVAL_INTERVAL:-100000}"   # disable in-loop eval
export EVAL_BEFORE="${EVAL_BEFORE:-false}"
# DeepSWE evaluates on SWE-Bench-Verified. We use the DCAgent2 random-100 split
# of the same benchmark (faster eval, same task source). This is CROSS-benchmark
# transfer.
EVAL_DATASET="/e/scratch/jureap59/ot/ttt/hf_hub/datasets--DCAgent2--swebench-verified-random-100-folders/snapshots/a2e51e9e0e8029156ed340719eb8cc7ceee3ed1a"
# IN-DISTRIBUTION transfer: a held-out slice of r2egym's learnable band (built by
# merge_split_learnable.py --out-test ...). Disjoint from DATASET. Set to "" to
# skip. Reported as a SECOND val dataset alongside swebench every eval_interval.
EVAL_DATASET2=${EVAL_DATASET2:-/e/data1/datasets/playground/ot/hf_hub/r2egym_learnable_heldout}
# FORMAT-CONTROL: swebench-100 with r2egym's exact instruction framing (same
# repos/tests/verifier, only instruction.md reformatted). Isolates the format
# confound: compare resolved@1 here vs EVAL_DATASET (raw swebench format) for
# BOTH base (step 0) and every RL checkpoint -> does RL transfer once format
# matches? Built by reformat_swebench_to_r2egym.py. Set "" to skip.
EVAL_DATASET3=${EVAL_DATASET3-/e/data1/datasets/playground/ot/hf_hub/swebench_verified_100_r2egymfmt}  # EVAL_DATASET3="" drops it

NUM_NODES=${NUM_NODES:-12}
NUM_GPUS_PER_NODE=4

PARTITION="booster"
ACCOUNT="reformo"
TIME_LIMIT=${TIME_LIMIT:-12:00:00}

# -- Algorithm / batch sizing aligned with DeepSWE --
# DeepSWE: data.train_batch_size=8 with policy_dp=8 (64 GPUs / 8-way Ulysses SP).
# SkyRL: FSDP-only, no sequence parallel → policy_dp = POLICY_NUM_NODES × 4 GPUs = 16.
# Bumped to 16 so SkyRL's lcm(policy_dp, batch) check passes while keeping the
# "1 sample per DP rank" ratio DeepSWE had.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
EVAL_BATCH_SIZE=512
# 358 learnable tasks @ bs=64 = ~5.6 unique steps/epoch. EPOCHS=4 -> ~22 steps,
# enough to read stability + the swebench-100 resolved@1 trajectory.
EPOCHS=${EPOCHS:-4}
# Reference DeepSWE-seqmean LR is 1e-6 (tuned for this exact loss_reduction +
# dual_clip + grad-clip stack). The pymethods *token_mean* sweep found 3e-6
# stable; that does NOT transfer 1:1 to this config. Defaulting to the PoC's
SCHEDULER=${SCHEDULER:-constant_with_warmup}
NUM_WARMUP_STEPS=${NUM_WARMUP_STEPS:-0}
AGENT=${AGENT:-terminus-structured}
# DeepSWE: agent.max_steps=50 (we mirror via harbor.max_episodes)
MAX_EPISODES=50
# DeepSWE: agent.trajectory_timeout=5400 (90 min per trial)
TIMEOUT_SEC=${TIMEOUT_SEC:-1800}   # raise for long-context (t2/no-summ) runs, e.g. 3600
NUM_STALENESS_STEPS=${NUM_STALENESS_STEPS:-1}
# DeepSWE: trainer.save_freq=10, test_freq=10
CKPT_INTERVAL=${CKPT_INTERVAL:-5}
HF_SAVE_INTERVAL=${HF_SAVE_INTERVAL:--1}   # >0 drops an HF snapshot every N steps to hf_ckpts/ for out-of-band eval
# Platform: DeepSWE uses TP=8 on 8-GPU nodes → 1 engine/node. JSC has 4 GPUs/node,
# so TP=4 matches the "1 engine per node" spirit with a single-node engine.
TENSOR_PARALLEL_SIZE=4
POLICY_NUM_NODES=${POLICY_NUM_NODES:-4}
NUM_INFERENCE_ENGINES=${NUM_INFERENCE_ENGINES:-8}
# DYNAMIC SAMPLING (DAPO-style) requires the SYNC trainer (RayPPOTrainer),
# which is selected by colocate_all=true. Async (colocate_all=false, the
# default) does NOT support dynamic_sampling -- it has no batch boundary at
# which to resample-until-full. Set COLOCATE_ALL=true + DYNAMIC_SAMPLING_TYPE=
# filter to run it. Sync is slower (no gen/train overlap). filter = drop
# all-same-reward prompts and resample up to max_sample_batches(30).
COLOCATE_ALL=${COLOCATE_ALL:-false}
DYNAMIC_SAMPLING_TYPE=${DYNAMIC_SAMPLING_TYPE:-null}
# KL LOSS. NOTE: the DeepSWE reference recipe this reproduces uses
# use_kl_loss=FALSE, but this script overrode it to true. KL loss anchors the
# policy to base -- an alignment tool, not a capability one. Capability RL
# (DAPO, R1-Zero) drops it. Set USE_KL_LOSS=false to remove (restores the
# reference recipe). It sharpens the learnable band harder but CANNOT reach
# the context-censored hard tasks (they give no gradient regardless). Watch

# [TRUNCATED — operator's paste ended here with "(432 lines left)"]
