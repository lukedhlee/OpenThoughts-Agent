#!/bin/bash
# build_confirm.sh — clone the wave-0 probe p2o6all_s0 into the A-vs-control confirmation probe p2oAc_s0 (no submission).
# Changes vs wave 0: tasks = only *-pA and *-pctl (240 dirs), 24 coordinators, HARBOR_OPENAI_CONNECT_TIMEOUT_SEC=120,
# n_concurrent_trials 192, eval_batch_size 240 (one batch), bridge on login02 (10.128.1.2:9924), wall 4 h.
set -uo pipefail
export OMP_NUM_THREADS=1
E=/e/fscratch/reformo/lee27/experiments; M=/e/data1/mmlaion/lee27/experiments; T=/e/fscratch/reformo/lee27/tasks
SRC=p2o6all_s0; DST=p2oAc_s0; SRCT=p2o6all-s0; DSTT=p2oAc-s0; IP=10.128.1.2
# 1. task tree: A + ctl only
if [ ! -d $T/$DSTT ]; then
  mkdir -p $T/$DSTT
  for d in $T/$SRCT/*-pA $T/$SRCT/*-pctl; do cp -a "$d" $T/$DSTT/ || { echo "copy failed $d"; exit 1; }; done
fi
n=$(ls $T/$DSTT | wc -l); [ "$n" = 240 ] || { echo "tree has $n dirs, expected 240"; exit 1; }
for t in $(ls $T/$DSTT | grep -- "-pA$" | sed "s/-pA$//"); do
  n=$(wc -c < $T/$DSTT/$t-pctl/instruction.md)
  cmp -s -n $n $T/$DSTT/$t-pctl/instruction.md $T/$DSTT/$t-pA/instruction.md || { echo "byte-identity FAILED above delimiter: $t"; exit 1; }
  grep -q "^Working guidance:$" $T/$DSTT/$t-pA/instruction.md || { echo "block missing in $t-pA"; exit 1; }
done
echo "tree ok: 240 dirs, byte-identity above the delimiter holds for 120 tasks"
# 2. experiment dir on mmlaion + symlink
mkdir -p $M/$DST/sbatch $M/$DST/configs $M/$DST/logs $M/$DST/$DST; ln -sfn $M/$DST $E/$DST
touch $E/$DST/.compacted
# 3. sbatch
SB=$E/$DST/sbatch/${DST}_rl.sbatch
sed -e "s/$SRC/$DST/g" -e "s#$SRCT#$DSTT#g" -e "s#^export APPTAINER_BRIDGE_URL=http://[0-9.]*:9924\$#export APPTAINER_BRIDGE_URL=http://$IP:9924#" -e "s/^#SBATCH --time=06:00:00/#SBATCH --time=04:00:00/" $E/$SRC/sbatch/${SRC}_rl.sbatch > $SB
if grep -q "^export HARBOR_OPENAI_CONNECT_TIMEOUT_SEC=" $SB; then sed -i "s/^export HARBOR_OPENAI_CONNECT_TIMEOUT_SEC=.*/export HARBOR_OPENAI_CONNECT_TIMEOUT_SEC=120/" $SB; else sed -i "/^export APPTAINER_BRIDGE_URL=/a export HARBOR_OPENAI_CONNECT_TIMEOUT_SEC=120" $SB; fi
grep -q "^export HARBOR_OPENAI_CONNECT_TIMEOUT_SEC=120$" $SB || { echo "connect timeout not set"; exit 1; }
grep -q "APPTAINER_BRIDGE_URL=http://$IP:9924" $SB || { echo "bridge url not set"; exit 1; }
grep -q "^#SBATCH --time=04:00:00" $SB || { echo "walltime not set"; exit 1; }
[ "$(grep -c "$SRC" $SB)" = 0 ] || { echo "old name left in sbatch"; exit 1; }
# 4. config json
python3 - <<PY || exit 1
import json
src="$E/$SRC/configs/${SRC}_rl_config.json"; dst="$E/$DST/configs/${DST}_rl_config.json"
s=open(src).read().replace("$SRC","$DST").replace("$SRCT","$DSTT")
d=json.loads(s)
args=d["skyrl_hydra_args"]
def setk(prefix,val):
    global args
    hit=[i for i,x in enumerate(args) if x.startswith(prefix)]
    if hit:
        for i in hit: args[i]=prefix+val
    else: args.append(prefix+val)
setk("trajectory_runner.process_pool.num_coordinators=","24")
setk("++terminal_bench_config.harbor.n_concurrent_trials=","192")
setk("trainer.eval_batch_size=","240")
assert any(x=="data.val_data=[\"$T/$DSTT\"]" for x in args), [x for x in args if x.startswith("data.val_data")]
assert not any("chat_template_content_format" in x for x in args), "ctcf present"
assert any(x=="generator.eval_n_samples_per_prompt=4" for x in args)
assert any(x=="trainer.algorithm.use_tis=false" for x in args)
assert any(x.startswith("++generator.engine_init_kwargs.speculative_config=") for x in args), "not draftified"
d["skyrl_hydra_args"]=args
json.dump(d,open(dst,"w"),indent=2)
print("config ok:", [x for x in args if x.startswith(("trajectory_runner.process_pool.num_coordinators","++terminal_bench_config.harbor.n_concurrent_trials","trainer.eval_batch_size","data.val_data","trainer.run_name"))])
print("mentions of old name in config:", open(dst).read().count("$SRC"))
PY
ls -la $E/$DST/ $E/$DST/sbatch $E/$DST/configs | grep -v "^total"
echo BUILD_OK
