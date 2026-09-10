#!/bin/bash
# clone_resume_v2.sh <dst> <step> [submit=0] [extra hydra args...] — resume the fresh v2-train arm (snowball_ttband_v2train_fulldist_gmm_seats1584_x16_a)
# from its checkpoint global_step_<step> with two harness changes for the connect-timeout stall (2026-09-07 15:00 PT):
# trajectory_runner.process_pool.num_coordinators 12 -> 24 and HARBOR_OPENAI_CONNECT_TIMEOUT_SEC 30 -> 120 in the sbatch. Recipe/data unchanged.
set -euo pipefail
SRC=${SRC_OVERRIDE:-snowball_ttband_v2train_fulldist_gmm_seats1584_x16_a}; DST=$1; STEP=$2; SUBMIT=${3:-0}; shift 3 2>/dev/null || shift $#; EXTRA=("$@")
E=/e/fscratch/reformo/lee27/experiments; OTA=/e/project1/transfernetx/lee27/code/OpenThoughts-Agent; C=/e/project1/transfernetx/lee27/code/snowball; M=/e/data1/mmlaion/lee27/experiments
case "$DST" in *"$SRC"*) echo "dst name must not contain the src name"; exit 1;; esac
case "$DST" in *snowball_ttband*) ;; *) echo "dst must contain snowball_ttband (store_reaper)"; exit 1;; esac
CK=$M/$SRC/$SRC/checkpoints/global_step_$STEP; [ -d $CK ] || { echo "no checkpoint $CK"; exit 1; }
[ -d $E/$DST ] && squeue -h -u $USER -n $DST -o %i | grep -q . && { echo "$DST has a job in squeue - refusing"; exit 1; }
[ -L $E/$DST ] && rm -f $E/$DST; rm -rf $E/$DST $M/$DST
mkdir -p $M/$DST/configs $M/$DST/sbatch $M/$DST/logs && ln -s $M/$DST $E/$DST
sed "s/$SRC/$DST/g" $E/$SRC/configs/${SRC}_rl_config.json > $E/$DST/configs/${DST}_rl_config.json
sed "s/$SRC/$DST/g" $E/$SRC/sbatch/${SRC}_rl.sbatch > $E/$DST/sbatch/${DST}_rl.sbatch
sed -i "s/^export HARBOR_OPENAI_CONNECT_TIMEOUT_SEC=30$/export HARBOR_OPENAI_CONNECT_TIMEOUT_SEC=120/" $E/$DST/sbatch/${DST}_rl.sbatch
grep -q "HARBOR_OPENAI_CONNECT_TIMEOUT_SEC=120" $E/$DST/sbatch/${DST}_rl.sbatch || { echo "connect timeout sed failed"; exit 1; }
python3 - $E/$DST/configs/${DST}_rl_config.json $SRC "$CK" "${EXTRA[@]}" <<'PY'
import json, sys
p, src, ck, extra = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]; c = json.load(open(p)); a = c["skyrl_hydra_args"]
assert not any(src in x for x in a), "src name survived the sed"
def setk(prefix, val):
    i = [k for k, x in enumerate(a) if x.startswith(prefix)]; assert len(i) == 1, (prefix, i); a[i[0]] = prefix + val
setk("trainer.resume_mode=", "from_path"); a[:] = [x for x in a if not x.startswith("trainer.resume_path=")]; a.append("trainer.resume_path=" + ck)
setk("trajectory_runner.process_pool.num_coordinators=", "24")
for x in extra:
    pre = x.split("=")[0].lstrip("+") + "="; i = [k for k, y in enumerate(a) if y.lstrip("+").startswith(pre)]; assert len(i) <= 1, (x, i)
    if i: a[i[0]] = x
    else: a.append(x)
    print("extra:", x)
c["skyrl_hydra_args"] = a; json.dump(c, open(p, "w"), indent=2)
keys = ("resume", "num_coordinators", "cpus_per_coordinator", "train_data", "staleness", "optimizer_config.lr", "use_tis", "n_concurrent_trials")
print("resume-clone hydra args:", [x for x in a if any(k in x for k in keys)])
PY
python3 $C/fix_merged_keys.py $E/$DST/configs/${DST}_rl_config.json
$OTA/../envs/snowball/bin/python $C/validate_hydra_args.py $E/$DST/configs/${DST}_rl_config.json
grep -q "job-name=$DST" $E/$DST/sbatch/${DST}_rl.sbatch
IMG=$E/$DST/$DST/artifact_store.img; SRC_IMG=$E/$SRC/$SRC/artifact_store.img
OLDH=$(cd $OTA && python3 -c "from pathlib import Path; from hpc.artifact_store import mount_path_for_image; print(mount_path_for_image(Path('$SRC_IMG')).name.rsplit('-', 1)[-1])")
NEWH=$(cd $OTA && python3 -c "from pathlib import Path; from hpc.artifact_store import mount_path_for_image; print(mount_path_for_image(Path('$IMG')).name.rsplit('-', 1)[-1])")
sed -i "s/$OLDH/$NEWH/g" $E/$DST/sbatch/${DST}_rl.sbatch $E/$DST/configs/${DST}_rl_config.json
grep -q "ARTIFACT_STORE_MOUNT=\"/tmp/otagent-artifact-stores/$DST-$NEWH\"" $E/$DST/sbatch/${DST}_rl.sbatch
(cd $OTA && python3 -c "from pathlib import Path; from hpc.artifact_store import ensure_image; ensure_image(Path('$IMG'), size='1T', inode_count=50_000_000)") >/dev/null
echo "artifact store: $IMG mount hash $OLDH -> $NEWH"
echo "--- config diff vs src (names/hashes normalised):"; diff <(sed "s/$DST/NAME/g; s/$NEWH/HASH/g" $E/$DST/configs/${DST}_rl_config.json) <(sed "s/$SRC/NAME/g; s/$OLDH/HASH/g" $E/$SRC/configs/${SRC}_rl_config.json) || true
echo "--- sbatch diff:"; diff <(sed "s/$DST/NAME/g; s/$NEWH/HASH/g" $E/$DST/sbatch/${DST}_rl.sbatch) <(sed "s/$SRC/NAME/g; s/$OLDH/HASH/g" $E/$SRC/sbatch/${SRC}_rl.sbatch) || true
[ "$SUBMIT" = 1 ] && cd $OTA && DCFT=$PWD sbatch $E/$DST/sbatch/${DST}_rl.sbatch || echo "not submitted (SUBMIT=$SUBMIT): $E/$DST/sbatch/${DST}_rl.sbatch"
