#!/bin/bash
# resume_arm.sh <run> [min_step] — resume a Snowball band arm from its latest checkpoint after a crash (2026-09-05, session 20ae8169:
# the in-run eval at trainer.eval_interval dies on the TIS eval-path trap "rollout_logprobs are required for every generated group").
# Refuses while the run still has a job in squeue. Patches the run's rl_config.json in place: trainer.resume_mode=latest,
# trainer.eval_interval=9999 (in-run eval OFF; use the standalone probes), keeps everything else (artifact store image reused),
# validates, resubmits the same sbatch. The new log is <run>_<newjob>.out; W&B gets a new offline run id.
set -euo pipefail
RUN=$1; MIN=${2:-6}
E=/e/fscratch/reformo/lee27/experiments; C=/e/project1/transfernetx/lee27/code/snowball; O=/e/project1/transfernetx/lee27/code/OpenThoughts-Agent
PY=/e/project1/transfernetx/lee27/code/envs/snowball/bin/python
squeue -h -u $USER -n $RUN -o %i | grep -q . && { echo "$RUN still has a job in squeue - refusing"; exit 1; }
CK=$E/$RUN/$RUN/checkpoints; STEP=$(cat $CK/latest_ckpt_global_step.txt 2>/dev/null || echo 0)
[ "$STEP" -ge "$MIN" ] || { echo "$RUN latest checkpoint step $STEP < $MIN - refusing"; exit 1; }
[ -f $CK/global_step_$STEP/trainer_state.pt ] && [ -d $CK/global_step_$STEP/policy ] || { echo "checkpoint global_step_$STEP incomplete"; exit 1; }
[ -f $E/$RUN/$RUN/artifact_store.img.lock ] && fuser $E/$RUN/$RUN/artifact_store.img 2>/dev/null && { echo "artifact store image still busy"; exit 1; }
python3 - $E/$RUN/configs/${RUN}_rl_config.json <<'PY'
import json, sys
p = sys.argv[1]; c = json.load(open(p)); a = c["skyrl_hydra_args"]
def setk(prefix, val):
    i = [k for k, x in enumerate(a) if x.lstrip("+").startswith(prefix)]; assert len(i) <= 1, (prefix, i)
    if i: a[i[0]] = prefix + val
    else: a.append(prefix + val)
setk("trainer.resume_mode=", "latest"); setk("trainer.eval_interval=", "9999"); setk("trainer.eval_before_train=", "false")
c["skyrl_hydra_args"] = a; json.dump(c, open(p, "w"), indent=2)
print("patched:", [x for x in a if "resume_mode" in x or "eval_interval" in x or "eval_before" in x])
PY
$PY $C/validate_hydra_args.py $E/$RUN/configs/${RUN}_rl_config.json 2>&1 | tail -1
echo "resuming $RUN from global_step_$STEP"
cd $O && DCFT=$PWD sbatch --parsable $E/$RUN/sbatch/${RUN}_rl.sbatch
