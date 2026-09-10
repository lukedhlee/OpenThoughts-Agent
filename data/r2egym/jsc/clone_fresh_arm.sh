#!/bin/bash
# clone_fresh_arm.sh <dst_name> [submit=0] [extra hydra args...] — a FRESH arm (from the Stage-3 base) on the tt-v2 split with the
# fast stack: clones snowball_ttband_lr5e7_fulldist_r36_gmm_seats1584_x16_c12h2_a (the other session's fulldist resume on
# staleness 2 + grouped matmul + 1,584 seats + 12 coordinators, 2026-09-06) and changes ONLY: resume_mode none (no resume_path),
# train data = r2egym-tt-v2-train-x16 (728 tasks x 16, fixed prompt, hardened verifier), val data = r2egym-tt-v2-val441
# (idval + oodval + heldout; in-run eval stays off, probes are external). Derived from clone_stale.sh (store image + mount hash).
# dst must contain "snowball_ttband" (store_reaper.sh finds arms by that substring) and must not contain the src name.
set -euo pipefail
SRC=snowball_ttband_lr5e7_fulldist_r36_gmm_seats1584_x16_c12h2_a; DST=$1; SUBMIT=${2:-0}; shift 2 2>/dev/null || shift $#; EXTRA=("$@")
E=/e/fscratch/reformo/lee27/experiments; T=/e/fscratch/reformo/lee27/tasks; OTA=/e/project1/transfernetx/lee27/code/OpenThoughts-Agent; C=/e/project1/transfernetx/lee27/code/snowball
case "$DST" in *"$SRC"*) echo "dst name must not contain the src name"; exit 1;; esac
case "$DST" in *snowball_ttband*) ;; *) echo "dst must contain snowball_ttband (store_reaper)"; exit 1;; esac
[ -d $T/r2egym-tt-v2-train-x16 ] && [ -d $T/r2egym-tt-v2-val441 ] || { echo "task trees missing"; exit 1; }
[ -d $E/$DST ] && squeue -h -u $USER -n $DST -o %i | grep -q . && { echo "$DST has a job in squeue - refusing"; exit 1; }
M=/e/data1/mmlaion/lee27/experiments; mkdir -p $M
[ -L $E/$DST ] && rm -f $E/$DST; rm -rf $E/$DST $M/$DST
mkdir -p $M/$DST/configs $M/$DST/sbatch $M/$DST/logs && ln -s $M/$DST $E/$DST
sed "s/$SRC/$DST/g" $E/$SRC/configs/${SRC}_rl_config.json > $E/$DST/configs/${DST}_rl_config.json
sed "s/$SRC/$DST/g" $E/$SRC/sbatch/${SRC}_rl.sbatch > $E/$DST/sbatch/${DST}_rl.sbatch
sed -i "s/^export HARBOR_OPENAI_CONNECT_TIMEOUT_SEC=30$/export HARBOR_OPENAI_CONNECT_TIMEOUT_SEC=120/" $E/$DST/sbatch/${DST}_rl.sbatch   # 2026-09-07 connect-timeout fix
grep -q "HARBOR_OPENAI_CONNECT_TIMEOUT_SEC=120" $E/$DST/sbatch/${DST}_rl.sbatch || { echo "connect timeout sed failed"; exit 1; }
python3 - $E/$DST/configs/${DST}_rl_config.json $SRC "$T" "${EXTRA[@]}" <<'PY'
import json, sys
p, src, T, extra = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]; c = json.load(open(p)); a = c["skyrl_hydra_args"]
assert not any(src in x for x in a), "src name survived the sed"
def setk(prefix, val):
    i = [k for k, x in enumerate(a) if x.startswith(prefix)]; assert len(i) == 1, (prefix, i); a[i[0]] = prefix + val
train = "%s/r2egym-tt-v2-train-x16" % T; val = "%s/r2egym-tt-v2-val441" % T
setk("trainer.resume_mode=", "none")
setk("trajectory_runner.process_pool.num_coordinators=", "24")  # 2026-09-07: 12 -> 24 (connect-timeout fix, see clone_resume_v2.sh)
def setk_or_add(prefix, val):
    i = [k for k, x in enumerate(a) if x.lstrip("+").startswith(prefix)]; assert len(i) <= 1, (prefix, i)
    if i: a[i[0]] = "++" + prefix + val
    else: a.append("++" + prefix + val)
setk_or_add("terminal_bench_config.harbor.verifier_override_timeout_sec=", "2400")   # 2026-09-07: template 150 s made slow suites false zeros
setk_or_add("terminal_bench_config.harbor.preserve_logprobs_on_timeout=", "false")   # 2026-09-07: mask verifier/agent timeouts instead of reward 0
a[:] = [x for x in a if not x.startswith("trainer.resume_path=")]
setk("data.train_data=", '["%s"]' % train); setk("data.val_data=", '["%s"]' % val)
c["train_data"] = [train]; c["train_data_sources"] = [train]; c["val_data"] = [val]; c["val_data_sources"] = [val]
for x in extra:
    pre = x.split("=")[0].lstrip("+") + "="; i = [k for k, y in enumerate(a) if y.lstrip("+").startswith(pre)]
    assert len(i) <= 1, (x, i)
    if i: a[i[0]] = x
    else: a.append(x)
    print("extra:", x, "(replaced)" if i else "(appended)")
c["skyrl_hydra_args"] = a; json.dump(c, open(p, "w"), indent=2)
keys = ("resume", "train_data", "val_data", "staleness", "optimizer_config.lr", "run_name", "kl_loss_coef", "top_p", "top_k", "max_grad_norm", "use_tis",
        "num_inference_engines", "n_concurrent_trials", "train_batch_size", "ckpt_interval", "hf_save_interval", "eval_interval", "grouped_mm", "num_coordinators")
print("model_path:", c["model_path"]); print("fresh-arm hydra args:", [x for x in a if any(k in x for k in keys)])
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
[ -f $IMG ] && echo "artifact store: $IMG mount hash $OLDH -> $NEWH"
echo "--- config diff vs src (names/hashes normalised):"
diff <(sed "s/$DST/NAME/g; s/$NEWH/HASH/g" $E/$DST/configs/${DST}_rl_config.json) <(sed "s/$SRC/NAME/g; s/$OLDH/HASH/g" $E/$SRC/configs/${SRC}_rl_config.json) || true
[ "$SUBMIT" = 1 ] && cd $OTA && DCFT=$PWD sbatch $E/$DST/sbatch/${DST}_rl.sbatch || echo "not submitted (SUBMIT=$SUBMIT): $E/$DST/sbatch/${DST}_rl.sbatch"
