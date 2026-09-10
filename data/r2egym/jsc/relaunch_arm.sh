#!/bin/bash
# relaunch_arm.sh <run> <variant> — clean a dead RL run's per-step artefacts and resubmit it with one engine-hang variant applied.
# variants: none | eager (generator.enforce_eager=true) | nomix (export NCCL_GRAPH_MIXING_SUPPORT=0 in the sbatch) |
#           noasync (++generator.engine_init_kwargs.async_scheduling=false) | piecewise (compilation_config={cudagraph_mode:PIECEWISE}) |
#           tp4b (TP4 x DP1 x EP4 engines + SKYRL_VLLM_PORT_BASE=0 in the sbatch). Idempotent on the config (flags replaced, not duplicated).
set -euo pipefail
RUN=$1; VAR=$2; E=/e/fscratch/reformo/lee27/experiments; OTA=/e/project1/transfernetx/lee27/code/OpenThoughts-Agent
CFG=$E/$RUN/configs/${RUN}_rl_config.json; SB=$E/$RUN/sbatch/${RUN}_rl.sbatch; [ -f $CFG ] && [ -f $SB ]
squeue -h -u $USER -n $RUN -o %i | grep -q . && { echo "$RUN still has a job in squeue"; exit 1; }
for d in trace_jobs exports/dumped_data checkpoints eval_tables eval_archive; do rm -rf $E/$RUN/$RUN/$d; done
python3 - $CFG $VAR <<'PY'
import json, sys
p, var = sys.argv[1], sys.argv[2]; c = json.load(open(p)); a = c["skyrl_hydra_args"]
a = [x for x in a if not x.startswith(("++generator.engine_init_kwargs.async_scheduling=", "++generator.engine_init_kwargs.compilation_config="))]
i = [k for k, x in enumerate(a) if x.startswith("generator.enforce_eager=")]; assert len(i) == 1
a[i[0]] = "generator.enforce_eager=" + ("true" if var == "eager" else "false")
if var == "noasync": a.append("++generator.engine_init_kwargs.async_scheduling=false")
if var == "piecewise": a.append("++generator.engine_init_kwargs.compilation_config={cudagraph_mode:PIECEWISE}")
def rep(prefix, val):
    i = [k for k, x in enumerate(a) if x.startswith(prefix)]; assert len(i) == 1, prefix; a[i[0]] = prefix + val
if var == "tp4b":
    rep("generator.inference_engine_tensor_parallel_size=", "4"); rep("++generator.inference_engine_data_parallel_size=", "1")
    rep("generator.inference_engine_expert_parallel_size=", "4"); c["tensor_parallel_size"] = 4
else:
    rep("generator.inference_engine_tensor_parallel_size=", "1"); rep("++generator.inference_engine_data_parallel_size=", "4"); c["tensor_parallel_size"] = 1
assert var in ("none", "eager", "nomix", "noasync", "piecewise", "tp4b"), var
c["skyrl_hydra_args"] = a; json.dump(c, open(p, "w"), indent=2)
print("cfg:", [x for x in a if "enforce_eager" in x or "async_scheduling" in x or "compilation_config" in x or "parallel_size" in x or "epochs=" in x or "advantage_estimator" in x or "max_staleness" in x])
PY
sed -i "/^export NCCL_GRAPH_MIXING_SUPPORT=0$/d; /^export SKYRL_VLLM_PORT_BASE=0$/d" $SB
[ "$VAR" = tp4b ] && sed -i "s|^export NCCL_PXN_DISABLE=1$|export NCCL_PXN_DISABLE=1\nexport SKYRL_VLLM_PORT_BASE=0|" $SB
[ "$VAR" = nomix ] && sed -i "s|^export NCCL_PXN_DISABLE=1$|export NCCL_PXN_DISABLE=1\nexport NCCL_GRAPH_MIXING_SUPPORT=0|" $SB
echo "sbatch env: nomix=$(grep -c NCCL_GRAPH_MIXING_SUPPORT=0 $SB) portbase0=$(grep -c SKYRL_VLLM_PORT_BASE=0 $SB); nodes=$(grep -m1 -- '--nodes=' $SB)"
cd $OTA && DCFT=$PWD sbatch $SB
