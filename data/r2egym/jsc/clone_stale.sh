#!/bin/bash
# clone_stale.sh <dst_name> <staleness> [submit=0] [extra hydra args...] — clone snowball_ttband_lr5e7_best_a (the running best
# arm, job 1684555: arm-3 recipe + real KL 0.04 + top_p 1.0/top_k -1 + max_grad_norm 0.1, no cap, no mask) into a new run dir
# with ONLY trainer.fully_async.max_staleness_steps changed. This is the throughput A/B (rollout wait vs policy lag) from
# ai_memory/active/snowball-r2egym/research/rl_throughput_and_acceleration.md, 2026-09-05. Derived from clone_sync.sh.
# The dispatcher admits groups while accepted+running < (staleness + step) * mini_batch (fully_async_trainer.py:291), so
# staleness 2 lets one extra batch of groups generate while the trainer works; the paired baseline is the running best arm.
set -euo pipefail
SRC=snowball_ttband_lr5e7_best_a; DST=$1; STALE=$2; SUBMIT=${3:-0}; shift 3 2>/dev/null || shift $#; EXTRA=("$@")
E=/e/fscratch/reformo/lee27/experiments; OTA=/e/project1/transfernetx/lee27/code/OpenThoughts-Agent
case "$DST" in *"$SRC"*) echo "dst name must not contain the src name ($SRC): sed would loop"; exit 1;; esac
[ -d $E/$DST ] && squeue -h -u $USER -n $DST -o %i | grep -q . && { echo "$DST has a job in squeue - refusing"; exit 1; }
# New runs live on the no-purge mmlaion pool (Luke, 2026-09-06: fscratch is a shared 44 TB pool that filled to 97 %); a symlink at the
# fscratch path keeps every script that hard-codes E=.../fscratch/.../experiments resolving, and the artifact-store mount hash is
# computed from the fscratch path string, so sbatch and config stay consistent.
M=/e/data1/mmlaion/lee27/experiments; mkdir -p $M
[ -L $E/$DST ] && rm -f $E/$DST; rm -rf $E/$DST $M/$DST
mkdir -p $M/$DST/configs $M/$DST/sbatch $M/$DST/logs && ln -s $M/$DST $E/$DST
sed "s/$SRC/$DST/g" $E/$SRC/configs/${SRC}_rl_config.json > $E/$DST/configs/${DST}_rl_config.json
sed "s/$SRC/$DST/g" $E/$SRC/sbatch/${SRC}_rl.sbatch > $E/$DST/sbatch/${DST}_rl.sbatch
python3 $OTA/../snowball/fix_merged_keys.py $E/$DST/configs/${DST}_rl_config.json  # merged-schema key rewrite (2026-09-04)
$OTA/../envs/snowball/bin/python $OTA/../snowball/validate_hydra_args.py $E/$DST/configs/${DST}_rl_config.json
python3 - $E/$DST/configs/${DST}_rl_config.json $SRC $STALE "${EXTRA[@]}" <<'PY'
import json, sys
p, src, stale, extra = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]; c = json.load(open(p)); a = c["skyrl_hydra_args"]
assert not any(src in x for x in a), "src name survived the sed"
def rep(prefix, val):
    i = [k for k, x in enumerate(a) if x.startswith(prefix)]; assert len(i) == 1, (prefix, i); a[i[0]] = prefix + val
rep("trainer.fully_async.max_staleness_steps=", stale)
for x in extra:  # extra overrides: replace an existing prefix (ignoring a ++/+ marker) or append
    pre = x.split("=")[0].lstrip("+") + "="; i = [k for k, y in enumerate(a) if y.lstrip("+").startswith(pre)]
    assert len(i) <= 1, (x, i)
    if i: a[i[0]] = x
    else: a.append(x)
    print("extra:", x, "(replaced)" if i else "(appended)")
c["skyrl_hydra_args"] = a; json.dump(c, open(p, "w"), indent=2)
keys = ("staleness", "optimizer_config.lr", "max_steps", "run_name", "kl_loss_coef", "top_p", "top_k", "max_grad_norm", "resume_mode",
        "ckpt_path", "use_tis", "tis_imp_ratio_cap", "num_inference_engines", "train_batch_size", "n_samples_per_prompt")
print("stale hydra args:", [x for x in a if any(k in x for k in keys)])
PY
grep -q "job-name=$DST" $E/$DST/sbatch/${DST}_rl.sbatch
# The launcher (hpc.launch -> rl_launch_utils.ensure_image) creates the run's artifact-store image and bakes a node-local
# mount path derived from sha256(image path) into the sbatch/config. A clone must redo both, or the sbatch dies at
# startup with "FATAL: artifact-store image is missing" (job 1686329, 2026-09-05) and readers would look under the
# source run's mount hash.
IMG=$E/$DST/$DST/artifact_store.img; SRC_IMG=$E/$SRC/$SRC/artifact_store.img
OLDH=$(cd $OTA && python3 -c "from pathlib import Path; from hpc.artifact_store import mount_path_for_image; print(mount_path_for_image(Path('$SRC_IMG')).name.rsplit('-', 1)[-1])")
NEWH=$(cd $OTA && python3 -c "from pathlib import Path; from hpc.artifact_store import mount_path_for_image; print(mount_path_for_image(Path('$IMG')).name.rsplit('-', 1)[-1])")
sed -i "s/$OLDH/$NEWH/g" $E/$DST/sbatch/${DST}_rl.sbatch $E/$DST/configs/${DST}_rl_config.json
grep -q "ARTIFACT_STORE_MOUNT=\"/tmp/otagent-artifact-stores/$DST-$NEWH\"" $E/$DST/sbatch/${DST}_rl.sbatch
(cd $OTA && python3 -c "from pathlib import Path; from hpc.artifact_store import ensure_image; ensure_image(Path('$IMG'), size='1T', inode_count=50_000_000)") >/dev/null
[ -f $IMG ] && echo "artifact store: $IMG mount hash $OLDH -> $NEWH"
diff <(sed "s/$DST/NAME/g; s/$NEWH/HASH/g" $E/$DST/configs/${DST}_rl_config.json) <(sed "s/$SRC/NAME/g; s/$OLDH/HASH/g" $E/$SRC/configs/${SRC}_rl_config.json) || true
[ "$SUBMIT" = 1 ] && cd $OTA && DCFT=$PWD sbatch $E/$DST/sbatch/${DST}_rl.sbatch || echo "not submitted (SUBMIT=$SUBMIT): $E/$DST/sbatch/${DST}_rl.sbatch"
