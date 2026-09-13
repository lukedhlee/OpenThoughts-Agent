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
# (find, not ls | head: under pipefail a 16k-entry ls dies of SIGPIPE and set -e kills the script silently)
# the source tree's entries are symlinks to task dirs (x16 replicas); the prompted copy holds real dirs
n_src=$(find $T/$TREE -mindepth 1 -maxdepth 1 \( -type d -o -type l \) | wc -l); n_dst=$(find $T/$PTREE -mindepth 1 -maxdepth 1 -type d | wc -l)
[ "$n_src" = "$n_dst" ] || { echo "tree size mismatch: $TREE $n_src vs $PTREE $n_dst"; exit 1; }
t=$(find $T/$PTREE -mindepth 1 -maxdepth 1 -type d -print -quit); grep -q "^Working guidance:$" $t/instruction.md || { echo "block missing in $t"; exit 1; }
grep -q "^    context_distillation:" $MS/skyrl-train/skyrl_train/config/ppo_base_config.yaml || {
  echo "MarinSkyRL checkout $MS lacks trainer.algorithm.context_distillation: merge lukedhlee/p2o-context-distillation into lukedhlee/snowball-r2egym and git pull"; exit 1; }
echo "prompted tree ok: $PTREE ($n_dst tasks, block $BLOCK); MarinSkyRL checkout at $(git -C $MS rev-parse --short HEAD) carries the feature"
NODES=${NODES:-12} SPEC=${SPEC:-1} TRAIN_TREE=$PTREE bash $C/clone_newstack_arm.sh "$DST" 0 \
  trainer.algorithm.context_distillation.enabled=true \
  trainer.algorithm.context_distillation.tis_reference=rollout \
  trainer.algorithm.context_distillation.on_failure=error \
  trainer.max_steps=12 \
  trainer.policy.optimizer_config.lr=5e-7 \
  "${EXTRA[@]}"
# Preflight on the head node, in the env the launcher hands the entrypoint process (it re-sources hpc/dotenv/jupiter.env +
# jupiter.local.env, whose SKYRL_HOME goes first on PYTHONPATH): runs a and b (1782191, 1782377) died at Hydra composition
# because that process imported code/MarinSkyRL (the pre-migration clone, RL_REPO_DIR in jupiter.local.env) and composed
# its old ppo_base_config, while the Ray driver task ran marinskyrl-marin. Fail in seconds if the composed config lacks the key.
SB=$E/$DST/sbatch/${DST}_rl.sbatch
cat > $E/$DST/sbatch/preflight_context_distillation.sh <<'PF'
# the launcher keeps the sbatch's interpreter (DCFT_RL_ENV=snowball-v2) but takes SKYRL_HOME/PYTHONPATH from the dotenv
PY_ARM="$RL_PYTHON"
cd "$WORKDIR" && set -a && source hpc/dotenv/jupiter.env >/dev/null 2>&1; source hpc/dotenv/jupiter.local.env >/dev/null 2>&1; set +a
echo "preflight: python $PY_ARM, SKYRL_HOME=$SKYRL_HOME"
"$PY_ARM" - <<'PY' || { echo "FATAL: the entrypoint process would compose a ppo_base_config without context_distillation (check RL_REPO_DIR in hpc/dotenv/jupiter.local.env)"; exit 97; }
import os, sys, skyrl_train.entrypoints.main_base as m
yaml = open(os.path.join(m.config_dir, "ppo_base_config.yaml")).read()
print("entrypoint config dir:", m.config_dir); sys.exit(0 if "context_distillation:" in yaml else 1)
PY
PF
sed -i "/^export HARBOR_TMUX_CAPTURE_MAX_WINDOW_LINES=2000$/a bash $E/$DST/sbatch/preflight_context_distillation.sh || exit 97" $SB
grep -q "preflight_context_distillation.sh" $SB || { echo "preflight insert failed"; exit 1; }
[ "$SUBMIT" = 1 ] && cd /e/project1/transfernetx/lee27/code/OpenThoughts-Agent && DCFT=$PWD sbatch $SB || echo "not submitted (SUBMIT=$SUBMIT): $SB"
