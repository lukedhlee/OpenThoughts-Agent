#!/bin/bash
# clone_ovf.sh <dst_name> <variant> — clone snowball_overfit32_a (8 policy @ fsdp32 + 8 engines @ 32 conc; the deterministic
# DP4xEP4 first-step hang reproducer, 2026-09-03) into a new run dir with one variant applied, and submit it.
# variants: ctrl | tp4 (TP4xDP1xEP4) | eager (generator.enforce_eager=true) | nomix (NCCL_GRAPH_MIXING_SUPPORT=0) | noasync (engine async_scheduling=false)
set -euo pipefail
SRC=snowball_overfit32_a; DST=$1; VAR=$2; E=/e/fscratch/reformo/lee27/experiments; OTA=/e/project1/transfernetx/lee27/code/OpenThoughts-Agent
rm -rf $E/$DST; mkdir -p $E/$DST/configs $E/$DST/sbatch $E/$DST/logs
sed "s/$SRC/$DST/g" $E/$SRC/configs/${SRC}_rl_config.json > $E/$DST/configs/${DST}_rl_config.json
sed "s/$SRC/$DST/g" $E/$SRC/sbatch/${SRC}_rl.sbatch > $E/$DST/sbatch/${DST}_rl.sbatch
python3 $OTA/../snowball/fix_merged_keys.py $E/$DST/configs/${DST}_rl_config.json  # merged-schema key rewrite (2026-09-04)
$OTA/../envs/snowball/bin/python $OTA/../snowball/validate_hydra_args.py $E/$DST/configs/${DST}_rl_config.json
python3 - $E/$DST/configs/${DST}_rl_config.json $VAR <<'PY'
import json, sys
p, var = sys.argv[1], sys.argv[2]; c = json.load(open(p)); a = c["skyrl_hydra_args"]
assert not any(x.startswith("trainer.run_name=snowball_overfit32_a") for x in a)
if var == "eager":
    i = [k for k, x in enumerate(a) if x.startswith("generator.enforce_eager=")]; assert len(i) == 1; a[i[0]] = "generator.enforce_eager=true"
elif var == "noasync":
    a.append("++generator.engine_init_kwargs.async_scheduling=false")
elif var == "tp4":
    # TP4 x DP1 x EP4 per node: no DP lockstep / coordinator / dummy batches; SkyRL asserts dp*tp == ep (utils.py:933)
    def rep(prefix, val):
        i = [k for k, x in enumerate(a) if x.startswith(prefix)]; assert len(i) == 1, prefix; a[i[0]] = prefix + val
    rep("generator.inference_engine_tensor_parallel_size=", "4"); rep("++generator.inference_engine_data_parallel_size=", "1")
    rep("generator.inference_engine_expert_parallel_size=", "4"); c["tensor_parallel_size"] = 4
elif var == "piecewise":
    a.append("++generator.engine_init_kwargs.compilation_config={cudagraph_mode:PIECEWISE}")
elif var == "tp4b":
    # TP4 x DP1 x EP4 with the SKYRL_VLLM_PORT_BASE override disabled (the base collided with the TP init port in probe tp4)
    def rep(prefix, val):
        i = [k for k, x in enumerate(a) if x.startswith(prefix)]; assert len(i) == 1, prefix; a[i[0]] = prefix + val
    rep("generator.inference_engine_tensor_parallel_size=", "4"); rep("++generator.inference_engine_data_parallel_size=", "1")
    rep("generator.inference_engine_expert_parallel_size=", "4"); c["tensor_parallel_size"] = 4
elif var == "eng4":
    # b's geometry: 4 engine nodes @ 64 trials/node (256 conc), FULL_AND_PIECEWISE graphs; 8 policy + 4 engines = 12 nodes
    def rep(prefix, val):
        i = [k for k, x in enumerate(a) if x.startswith(prefix)]; assert len(i) == 1, prefix; a[i[0]] = prefix + val
    rep("generator.num_inference_engines=", "4"); rep("generator.max_num_seqs=", "64"); c["num_nodes"] = 12
elif var == "nosync":
    pass  # env SKYRL_SKIP_STARTUP_WEIGHT_SYNC=1 added to the sbatch below
elif var == "nonuma":
    pass  # sbatch: SKYRL_ENABLE_NUMA_AFFINITY=0 (upstream #509 "Await worker NUMA affinity setup" hints at a startup race)
elif var == "iterlog":
    # per-iteration engine logging on every rank (num ctx/gen reqs+tokens, dummy steps) + cudagraph metrics -> read the last
    # iteration each rank logged on a hung node to see where the 4 ranks diverge
    a.append("++generator.engine_init_kwargs.enable_logging_iteration_details=true")
    a.append("++generator.engine_init_kwargs.cudagraph_metrics=true")
elif var in ("ctrl", "nomix"):
    pass
else:
    raise SystemExit(f"unknown variant {var}")
c["skyrl_hydra_args"] = a; json.dump(c, open(p, "w"), indent=2)
print(var, "hydra args:", [x for x in a if "enforce_eager" in x or "async_scheduling" in x or "run_name" in x or "parallel_size" in x or "num_inference_engines" in x or "compilation_config" in x or "max_num_seqs" in x], "tp_top=", c.get("tensor_parallel_size"))
PY
if [ "$VAR" = tp4b ]; then
  sed -i "s|^export NCCL_PXN_DISABLE=1$|export NCCL_PXN_DISABLE=1\nexport SKYRL_VLLM_PORT_BASE=0|" $E/$DST/sbatch/${DST}_rl.sbatch
  grep -c "SKYRL_VLLM_PORT_BASE=0" $E/$DST/sbatch/${DST}_rl.sbatch | grep -q 1
fi
if [ "$VAR" = eng4 ]; then
  sed -i "s|^#SBATCH --nodes=16$|#SBATCH --nodes=12|; s|^export NUM_INFERENCE_ENGINES=64$|export NUM_INFERENCE_ENGINES=48|; s|^export POLICY_NUM_NODES=16$|export POLICY_NUM_NODES=12|" $E/$DST/sbatch/${DST}_rl.sbatch
  grep -q -- "--nodes=12" $E/$DST/sbatch/${DST}_rl.sbatch
fi
if [ "$VAR" = nosync ]; then
  sed -i "s|^export NCCL_PXN_DISABLE=1$|export NCCL_PXN_DISABLE=1\nexport SKYRL_SKIP_STARTUP_WEIGHT_SYNC=1|" $E/$DST/sbatch/${DST}_rl.sbatch
  grep -c "SKYRL_SKIP_STARTUP_WEIGHT_SYNC=1" $E/$DST/sbatch/${DST}_rl.sbatch | grep -q 1
fi
if [ "$VAR" = nonuma ]; then
  sed -i "s|^export SKYRL_ENABLE_NUMA_AFFINITY=1$|export SKYRL_ENABLE_NUMA_AFFINITY=0|" $E/$DST/sbatch/${DST}_rl.sbatch
  grep -c "SKYRL_ENABLE_NUMA_AFFINITY=0" $E/$DST/sbatch/${DST}_rl.sbatch | grep -q 1
fi
if [ "$VAR" = nomix ]; then
  sed -i "s|^export NCCL_PXN_DISABLE=1$|export NCCL_PXN_DISABLE=1\nexport NCCL_GRAPH_MIXING_SUPPORT=0|" $E/$DST/sbatch/${DST}_rl.sbatch
  grep -c "NCCL_GRAPH_MIXING_SUPPORT=0" $E/$DST/sbatch/${DST}_rl.sbatch | grep -q 1
fi
grep -q "job-name=$DST" $E/$DST/sbatch/${DST}_rl.sbatch
cd $OTA && DCFT=$PWD sbatch $E/$DST/sbatch/${DST}_rl.sbatch
