#!/bin/bash
# clone_sync.sh <dst_name> <lr> [max_steps=10] [submit=1] [extra hydra args...] — Mercor Step 3: SYNCHRONOUS 32-task overfit run. Clones snowball_overfit32_a
# (8 policy nodes @ fsdp32 + 8 engines @ 32 seats = 256 concurrent = 32 prompts x 8 samples, RLOO-N, sequence_mean, no KL) into a new
# run dir with trainer.fully_async.max_staleness_steps=0 (the async trainer at staleness 0 admits exactly one batch of complete groups
# per step: capacity = (0 + step) * mini_batch - accepted - running, fully_async_trainer.py:291; 1 step = 1 epoch over the 32 tasks),
# the given lr, and max_steps/epochs = <max_steps>. Derived from clone_ovf.sh (2026-09-04, session 2189af97).
set -euo pipefail
SRC=snowball_overfit32_a; DST=$1; LR=$2; STEPS=${3:-10}; SUBMIT=${4:-1}; shift 4 2>/dev/null || shift $#; EXTRA=("$@"); E=/e/fscratch/reformo/lee27/experiments; OTA=/e/project1/transfernetx/lee27/code/OpenThoughts-Agent
[ -d $E/$DST ] && squeue -h -u $USER -n $DST -o %i | grep -q . && { echo "$DST has a job in squeue - refusing"; exit 1; }
rm -rf $E/$DST; mkdir -p $E/$DST/configs $E/$DST/sbatch $E/$DST/logs
sed "s/$SRC/$DST/g" $E/$SRC/configs/${SRC}_rl_config.json > $E/$DST/configs/${DST}_rl_config.json
sed "s/$SRC/$DST/g" $E/$SRC/sbatch/${SRC}_rl.sbatch > $E/$DST/sbatch/${DST}_rl.sbatch
python3 $OTA/../snowball/fix_merged_keys.py $E/$DST/configs/${DST}_rl_config.json  # merged-schema key rewrite (2026-09-04)
$OTA/../envs/snowball/bin/python $OTA/../snowball/validate_hydra_args.py $E/$DST/configs/${DST}_rl_config.json
python3 - $E/$DST/configs/${DST}_rl_config.json $LR $STEPS "${EXTRA[@]}" <<'PY'
import json, sys
p, lr, steps, extra = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]; c = json.load(open(p)); a = c["skyrl_hydra_args"]
assert not any(x.startswith("trainer.run_name=snowball_overfit32_a") for x in a)
def rep(prefix, val):
    i = [k for k, x in enumerate(a) if x.startswith(prefix)]; assert len(i) == 1, (prefix, i); a[i[0]] = prefix + val
rep("trainer.fully_async.max_staleness_steps=", "0")
rep("trainer.policy.optimizer_config.lr=", lr)
rep("trainer.max_steps=", steps); rep("trainer.epochs=", steps)
rep("trainer.ckpt_interval=", steps); rep("trainer.hf_save_interval=", steps)
for x in extra:  # extra overrides: replace an existing prefix (ignoring a ++/+ marker) or append
    pre = x.split("=")[0].lstrip("+") + "="; i = [k for k, y in enumerate(a) if y.lstrip("+").startswith(pre)]
    assert len(i) <= 1, (x, i)
    if i: a[i[0]] = x
    else: a.append(x)
    print("extra:", x, "(replaced)" if i else "(appended)")
c["skyrl_hydra_args"] = a; json.dump(c, open(p, "w"), indent=2)
print("sync hydra args:", [x for x in a if "staleness" in x or "optimizer_config.lr" in x or "max_steps" in x or "trainer.epochs" in x or "run_name" in x or "num_inference_engines" in x or "max_num_seqs" in x or "train_batch_size" in x or "n_samples_per_prompt" in x])
PY
grep -q "job-name=$DST" $E/$DST/sbatch/${DST}_rl.sbatch
[ "$SUBMIT" = 1 ] && cd $OTA && DCFT=$PWD sbatch $E/$DST/sbatch/${DST}_rl.sbatch || echo "not submitted (SUBMIT=$SUBMIT): $E/$DST/sbatch/${DST}_rl.sbatch"
