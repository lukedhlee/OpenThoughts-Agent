#!/bin/bash
# cont_s24_probe.sh <probe_name> <hf_export_dir> [submit=1] — today's-recipe pass@8 probe of an HF export on r2egym-tt-v2-val441
# (idval 150 + oodval 115 + heldout 176): EAGLE-3 draft, sampler as trained (top_p 1 / top_k -1), 512 seats over 24 coordinators
# with the eval session spread (MarinSkyRL a03b2773), verifier 2,400 s, agent 1,800 s, 61,440 in / 4,096 out, 8 nodes = 4 policy
# + 4 DP4xEP4 engines, bridge 9930, wall 3:30. Clones the last s24 refresh-screen shard (refresh_s24_20260915r2_b1, 2026-09-15,
# the most recent probe that ran clean at exactly this shape: 4,428 attempts, 0.6 % lost) and changes ONLY the name, the val
# tree, k (4 -> 8), the eval batch, the seed and the model (policy + ref + served name + json model_path, all four together).
# Run on the Jupiter login node that hosts the 9930 bridge (login02). Starts probe_watch.sh in a tmux (tables pass@8 into
# <probe>/pass8_pass8_table.csv for v2val_compare.py / heldout_compare.py, then archives the trace tree). Writes <probe>/job_id.
set -uo pipefail
P=$1; M=$2; SUBMIT=${3:-1}
SRC=${SRC:-refresh_s24_20260915r2_b1}; V=${VAL:-/e/fscratch/reformo/lee27/tasks/r2egym-tt-v2-val441}
BRIDGE=${BRIDGE:-http://10.128.1.2:9930}; WALL=${WALL:-03:30:00}; SEED=${SEED:-42}
E=/e/fscratch/reformo/lee27/experiments; MM=/e/data1/mmlaion/lee27/experiments; C=/e/project1/transfernetx/lee27/code/snowball
PY=/e/project1/transfernetx/lee27/code/envs/snowball-v2/bin/python; O=/e/project1/transfernetx/lee27/code/OpenThoughts-Agent
export OMP_NUM_THREADS=1
[ -f "$M/config.json" ] && [ -f "$M/model.safetensors.index.json" ] || { echo "no HF export under $M"; exit 1; }
[ -d "$V" ] || { echo "val tree missing: $V"; exit 1; }
[ -f $E/$SRC/configs/${SRC}_rl_config.json ] && [ -f $E/$SRC/sbatch/${SRC}_rl.sbatch ] || { echo "template $SRC missing"; exit 1; }
case "$P" in *"$SRC"*) echo "probe name must not contain the template name"; exit 1;; esac
squeue -h -u $USER -n $P -o %i | grep -q . && { echo "$P is in squeue; scancel it first"; exit 1; }
if [ -L $E/$P ]; then echo "clearing previous probe dir $(readlink $E/$P)"; rm -rf "$(readlink $E/$P)"; rm -f $E/$P; fi
rm -rf $E/$P $MM/$P
mkdir -p $MM/$P/configs $MM/$P/sbatch $MM/$P/logs $MM/$P/$P && ln -s $MM/$P $E/$P && touch $MM/$P/.compacted
sed "s/$SRC/$P/g" $E/$SRC/configs/${SRC}_rl_config.json > $E/$P/configs/${P}_rl_config.json
sed "s/$SRC/$P/g" $E/$SRC/sbatch/${SRC}_rl.sbatch > $E/$P/sbatch/${P}_rl.sbatch
SB=$E/$P/sbatch/${P}_rl.sbatch
python3 - $E/$P/configs/${P}_rl_config.json "$M" "$V" "$SEED" <<'PY'
import json, os, sys
p, m, v, seed = sys.argv[1:5]; c = json.load(open(p)); a = c["skyrl_hydra_args"]
def setk(key, val):
    pre = key + "="; i = [k for k, x in enumerate(a) if x.lstrip("+").startswith(pre)]
    assert len(i) == 1, (key, i)
    a[i[0]] = a[i[0]][: a[i[0]].index(pre)] + pre + val
setk("data.val_data", json.dumps([v])); c["val_data"] = [v]; c["val_data_sources"] = [v]
setk("generator.eval_n_samples_per_prompt", "8")
setk("trainer.eval_batch_size", "32")
setk("trainer.seed", seed)
setk("trainer.policy.model.path", m); setk("trainer.ref.model.path", m)
setk("generator.engine_init_kwargs.served_model_name", os.path.basename(m.rstrip("/")))
c["model_path"] = m
for must in ("trainer.epochs=0", "generator.sampling_params.top_p=1.0", "generator.sampling_params.top_k=-1",
             "trajectory_runner.process_pool.eval_spread_coordinators=true", "trajectory_runner.process_pool.num_coordinators=24",
             "++terminal_bench_config.harbor.n_concurrent_trials=512",
             "++terminal_bench_config.harbor.verifier_override_timeout_sec=2400", "trainer.algorithm.use_tis=false"):
    assert must in a, "template lost " + must
assert any("speculative_config=" in x for x in a), "draft missing"
assert not any("refresh_s24_20260915r2_b1" in x for x in a), "template name survived"
json.dump(c, open(p, "w"), indent=2)
keys = ("val_data", "eval_n_samples", "eval_batch", "seed", "model.path", "served_model", "n_concurrent", "num_coordinators",
        "eval_spread", "verifier", "top_p", "top_k", "epochs", "speculative", "max_input", "eval_timeout")
print("probe hydra:", *[x[:160] for x in a if any(k in x for k in keys)], sep="\n   ")
print("model_path:", c["model_path"], "num_nodes:", c["num_nodes"])
PY
[ $? -eq 0 ] || { echo "PROBE CONFIG FAILED ($P)"; exit 1; }
sed -i "s/^#SBATCH --time=.*$/#SBATCH --time=$WALL/" $SB
grep -q "^#SBATCH --time=$WALL$" $SB || { echo "wall sed failed"; exit 1; }
grep -q "^#SBATCH --nodes=8$" $SB || { echo "template sbatch is not 8 nodes"; exit 1; }
grep -q "APPTAINER_BRIDGE_URL=$BRIDGE" $SB || { echo "bridge $BRIDGE not pinned in the template sbatch"; exit 1; }
grep -q "marin_vllm_eagle3" $SB || { echo "eagle3 vllm overlay missing from the template sbatch"; exit 1; }
bash -n $SB || { echo "sbatch syntax"; exit 1; }
$PY $C/validate_hydra_args.py $E/$P/configs/${P}_rl_config.json 2>&1 | tail -1
[ "$SUBMIT" = 1 ] || { echo "not submitted (SUBMIT=$SUBMIT): $SB"; exit 0; }
cd $O && J=$(DCFT=$PWD sbatch --parsable $SB) || { echo "sbatch failed"; exit 1; }
echo $J > $E/$P/job_id
tmux new -d -s probe_watch_$P "bash $C/probe_watch.sh $P"
echo "$P: job $J model=$M val=$V bridge=$BRIDGE wall=$WALL watch=$(tmux ls 2>/dev/null | grep -c probe_watch_$P)"
