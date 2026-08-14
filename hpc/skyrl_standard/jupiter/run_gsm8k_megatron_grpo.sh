#!/bin/bash
# Jupiter STANDARD Megatron GRPO smoke/launch wrapper.
# Defaults to a cheap 2-node Qwen3-0.6B smoke; override MODEL_PATH/RL_CONFIG/
# NUM_NODES/JOB_NAME for the 30B MoE target.
set -euo pipefail

: "${JSC_SCRATCH:=/e/scratch/reformo/lee27}"
: "${SCRATCH:=$JSC_SCRATCH}"
: "${DCFT:=$SCRATCH/OpenThoughts-Agent}"
: "${RL_VENV:=$DCFT/envs/rl-megatron}"

export RL_VENV
export RL_ENV_DIR="$RL_VENV"
export DCFT_RL_ENV="$RL_VENV"

: "${MODEL_PATH:=Qwen/Qwen3-0.6B}"
: "${RL_CONFIG:=./hpc/skyrl_yaml/jupiter/2node_qwen3_0p6b_gsm8k_grpo_megatron_smoke.yaml}"
: "${NUM_NODES:=2}"
: "${JOB_NAME:=jupiter_qwen3_0p6b_gsm8k_megatron_smoke}"
: "${TIME_LIMIT:=00:45:00}"
: "${MAX_STEPS:=1}"
: "${EVAL_BEFORE_TRAIN:=false}"
: "${EVAL_INTERVAL:=999999}"
: "${CKPT_INTERVAL:=-1}"
: "${HF_SAVE_INTERVAL:=-1}"
: "${SAVE_OPTIMIZER_STATES:=false}"
: "${LOAD_OPTIMIZER_STATES:=false}"
: "${WANDB_PROJECT:=jupiter-megatron-gsm8k-smoke}"

export MODEL_PATH RL_CONFIG NUM_NODES JOB_NAME TIME_LIMIT
export MAX_STEPS EVAL_BEFORE_TRAIN EVAL_INTERVAL CKPT_INTERVAL HF_SAVE_INTERVAL
export SAVE_OPTIMIZER_STATES LOAD_OPTIMIZER_STATES WANDB_PROJECT

exec bash "$(dirname "$0")/run_gsm8k_moe30b_grpo.sh"
