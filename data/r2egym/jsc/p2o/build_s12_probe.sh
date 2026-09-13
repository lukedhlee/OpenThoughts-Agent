#!/bin/bash
# build_s12_probe.sh <probe_name> <model_dir> — clone the confirmation probe p2oAc_s0 into a BARE-prompt dev120 probe of an
# exported checkpoint: task tree = the 120 -pctl dirs only (p2o-dev120-ctl), k=4, 24 coordinators, 120 s connect timeout,
# conc 192, draft-served, use_tis=false, bridge 9924 on login02, 8 nodes, wall 3 h. Builds only; distill_probes.sh submits.
set -uo pipefail
export OMP_NUM_THREADS=1
E=/e/fscratch/reformo/lee27/experiments; M=/e/data1/mmlaion/lee27/experiments; T=/e/fscratch/reformo/lee27/tasks
SRC=p2oAc_s0; DST=$1; MODEL=$2; SRCT=p2oAc-s0; DSTT=p2o-dev120-ctl; IP=10.128.1.2
BASE=/e/fscratch/reformo/lee27/models/snowball-s3-nemotron-terminal-step1888
[ -f $MODEL/config.json ] && [ -f $MODEL/model.safetensors.index.json ] || { echo "model dir incomplete: $MODEL"; exit 1; }
# 1. ctl-only tree (bare prompts)
if [ ! -d $T/$DSTT ]; then
  mkdir -p $T/$DSTT
  for d in $T/$SRCT/*-pctl; do cp -a "$d" $T/$DSTT/ || { echo "copy failed $d"; exit 1; }; done
fi
n=$(ls $T/$DSTT | wc -l); [ "$n" = 120 ] || { echo "tree has $n dirs, expected 120"; exit 1; }
# 2. experiment dir on mmlaion + symlink
[ -e $E/$DST ] && { echo "$E/$DST exists"; exit 1; }
mkdir -p $M/$DST/sbatch $M/$DST/configs $M/$DST/logs $M/$DST/$DST; ln -sfn $M/$DST $E/$DST; touch $E/$DST/.compacted
# 3. sbatch
SB=$E/$DST/sbatch/${DST}_rl.sbatch
sed -e "s/$SRC/$DST/g" -e "s#$SRCT#$DSTT#g" -e "s/^#SBATCH --time=04:00:00/#SBATCH --time=03:00:00/" $E/$SRC/sbatch/${SRC}_rl.sbatch > $SB
grep -q "^export APPTAINER_BRIDGE_URL=http://$IP:9924$" $SB || { echo "bridge url missing"; exit 1; }
grep -q "^export HARBOR_OPENAI_CONNECT_TIMEOUT_SEC=120$" $SB || { echo "connect timeout missing"; exit 1; }
grep -q "^#SBATCH --time=03:00:00" $SB || { echo "walltime not set"; exit 1; }
[ "$(grep -c "$SRC" $SB)" = 0 ] || { echo "old name left in sbatch"; exit 1; }
# 4. config json: names, tree, model (policy + ref + model_path), one eval batch of 120 tasks
python3 - <<PY || exit 1
import json
src="$E/$SRC/configs/${SRC}_rl_config.json"; dst="$E/$DST/configs/${DST}_rl_config.json"
s=open(src).read().replace("$SRC","$DST").replace("$SRCT","$DSTT").replace("$BASE","$MODEL")
d=json.loads(s); args=d["skyrl_hydra_args"]
def setk(prefix,val):
    hit=[i for i,x in enumerate(args) if x.startswith(prefix)]
    if hit:
        for i in hit: args[i]=prefix+val
    else: args.append(prefix+val)
setk("trainer.eval_batch_size=","120")
assert d["model_path"]=="$MODEL", d["model_path"]
assert sum(x.endswith("=$MODEL") for x in args)==2, [x for x in args if "model.path" in x]
assert any(x=="data.val_data=[\"$T/$DSTT\"]" for x in args), [x for x in args if x.startswith("data.val_data")]
assert any(x=="generator.eval_n_samples_per_prompt=4" for x in args)
assert any(x=="trainer.algorithm.use_tis=false" for x in args)
assert any(x=="trajectory_runner.process_pool.num_coordinators=24" for x in args)
assert any(x.startswith("++generator.engine_init_kwargs.speculative_config=") for x in args), "not draftified"
assert not any("context_distillation" in x for x in args)
d["skyrl_hydra_args"]=args; json.dump(d,open(dst,"w"),indent=2)
print("config ok:", d["model_path"], [x for x in args if x.startswith(("data.val_data","trainer.eval_batch_size","trainer.run_name"))])
print("mentions of the base model left in config:", open(dst).read().count("$BASE"))
PY
echo BUILD_OK $SB
