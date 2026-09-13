#!/bin/bash
# build_distill_arm.sh <dst_name> [submit=0] [extra hydra args...] — the P2O context-distillation test arm (Stage 3).
#
# What it is: clone_newstack_arm.sh (NODES=12 = 4 policy + 8 engine nodes, 528 seats, SPEC=1 draft-served) with
#   * train tree = r2egym-tt-v2-train-basecurr-x16-pA: the wave-1 curriculum tree with block A appended to every
#     instruction behind the delimiter (p2o_train_tree.py; same task dir names, so uids pair with the control arm);
#   * trainer.algorithm.context_distillation.enabled=true: the trainer strips the block from the first user message
#     before tokenizing the training prompt (gradient on the bare task) and runs one extra no-grad policy forward under
#     the prompted prompt as the TIS reference (tis_reference=rollout). KL is off on this recipe, so kl_reference is moot;
#   * 12 steps at lr 5e-7, the wave-1 control's recipe (C, snowball_ttband_ns_c_lr5e7_c), so step-12 probes pair.
# Gates (each stops the script): the -pA tree exists and verified; the Jupiter MarinSkyRL checkout carries the feature
# (base config declares context_distillation:, i.e. lukedhlee/p2o-context-distillation merged into snowball-r2egym and
# pulled); validate_hydra_args.py accepts the rendered args (Hydra struct rejects an unknown key otherwise).
# Never submits unless SUBMIT=1; the sbatch is printed. Cost line, to state before submitting: 12 Jupiter nodes x ~5 h
# (~60 node-h) + a JUWELS fleet for 528 seats (17 nodes x ~5 h, ~4k core-h). The trainer keeps generating past
# max_steps (gotcha 2026-09-10): arm an export waiter on step 12 and cancel the job by id once the export lands.
# Readout: context_distillation/shift_per_token per step in the run log (falls toward 0 when the block is absorbed),
# generate/context_distillation/{edited,failed} (edited = 512, failed = 0 expected every step), then the step-12 export
# probed bare on dev120 (k=4) and val441 (k=8) against the control's step 12; the plan and stop rule live in
# ai_memory/active/snowball-r2egym/research/2026-09-13_p2o_distillation_test.md.
set -euo pipefail
DST=$1; SUBMIT=${2:-0}; shift 2 2>/dev/null || shift $#; EXTRA=("$@")
E=/e/fscratch/reformo/lee27/experiments; T=/e/fscratch/reformo/lee27/tasks; C=/e/project1/transfernetx/lee27/code/snowball
MS=/e/project1/transfernetx/lee27/code/marinskyrl-marin
TREE=r2egym-tt-v2-train-basecurr-x16; PTREE=$TREE-pA; BLOCK=${BLOCK:-A}
[ -d $T/$TREE ] || { echo "source tree missing: $T/$TREE"; exit 1; }
if [ ! -d $T/$PTREE ]; then
  python3 $E/p2o/p2o_train_tree.py --src $T/$TREE --dst $T/$PTREE --block $BLOCK --blocks $E/p2o/blocks.json || { echo "prompted tree build FAILED"; exit 1; }
fi
n_src=$(ls $T/$TREE | wc -l); n_dst=$(ls $T/$PTREE | wc -l)
[ "$n_src" = "$n_dst" ] || { echo "tree size mismatch: $TREE $n_src vs $PTREE $n_dst"; exit 1; }
t=$(ls $T/$PTREE | head -n 1); grep -q "^Working guidance:$" $T/$PTREE/$t/instruction.md || { echo "block missing in $PTREE/$t"; exit 1; }
grep -q "^    context_distillation:" $MS/skyrl-train/skyrl_train/config/ppo_base_config.yaml || {
  echo "MarinSkyRL checkout $MS lacks trainer.algorithm.context_distillation: merge lukedhlee/p2o-context-distillation into lukedhlee/snowball-r2egym and git pull"; exit 1; }
echo "prompted tree ok: $PTREE ($n_dst tasks, block $BLOCK); MarinSkyRL checkout at $(git -C $MS rev-parse --short HEAD) carries the feature"
NODES=${NODES:-12} SPEC=${SPEC:-1} TRAIN_TREE=$PTREE bash $C/clone_newstack_arm.sh "$DST" "$SUBMIT" \
  trainer.algorithm.context_distillation.enabled=true \
  trainer.algorithm.context_distillation.tis_reference=rollout \
  trainer.algorithm.context_distillation.on_failure=mask \
  trainer.max_steps=12 \
  trainer.policy.optimizer_config.lr=5e-7 \
  "${EXTRA[@]}"
