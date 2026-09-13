#!/bin/bash
# distill_extend.sh <run> <fleet_id> <new_max_steps> — after the watcher has cancelled the arm at its checkpoint: patch the
# run's config to the new max_steps, check the generation-buffer checkpoint is not a 0-byte mid-write leftover (gotcha
# 2026-09-11: move it aside if so), resubmit the SAME sbatch (resume_mode=latest -> resumes from the latest checkpoint),
# record the id, restart the watcher with STOP_STEP=<new_max_steps>. The fleet keeps running through the restart.
set -u; export OMP_NUM_THREADS=1
E=/e/fscratch/reformo/lee27/experiments; OTA=/e/project1/transfernetx/lee27/code/OpenThoughts-Agent
RUN=$1; F=$2; NEW=$3; CK=$E/$RUN/$RUN/checkpoints; SB=$E/$RUN/sbatch/${RUN}_rl.sbatch; CFG=$E/$RUN/configs/${RUN}_rl_config.json
IDS=$E/p2o/distill_ids.txt; LOG=$E/p2o/distill_watch.log
log(){ echo "$(date '+%m-%d %H:%M') $*" | tee -a $LOG; }
while squeue -h -n $RUN -o %i | grep -q .; do sleep 30; done
L=$(cat $CK/latest_ckpt_global_step.txt 2>/dev/null); log "extend $RUN: arm gone, latest checkpoint $L -> max_steps $NEW"
[ -f $CK/global_step_$L/trainer_state.pt ] || { log "no trainer_state at checkpoint $L; not resubmitting"; exit 1; }
B=$CK/global_step_$L/generation_buffer_state.pt
if [ -f "$B" ] && [ ! -s "$B" ]; then mv "$B" "$B.empty_$(date +%s)"; log "moved aside a 0-byte generation buffer checkpoint"; fi
python3 - "$CFG" "$NEW" <<'PY'
import json, sys
p, new = sys.argv[1], sys.argv[2]; c = json.load(open(p)); a = c["skyrl_hydra_args"]
i = [k for k, x in enumerate(a) if x.startswith("trainer.max_steps=")]; assert len(i) == 1, i
a[i[0]] = "trainer.max_steps=" + new
assert any(x == "trainer.resume_mode=latest" for x in a), "resume_mode is not latest"
json.dump(c, open(p, "w"), indent=2); print("config:", a[i[0]])
PY
J=$(cd $OTA && DCFT=$PWD sbatch --parsable $SB); [ -n "$J" ] || { log "resubmit failed"; exit 1; }
echo "arm $J $(date -Is) resume->$NEW" >> $IDS; log "resubmitted as $J (resume from $L, max_steps $NEW)"
tmux kill-session -t p2od_watch 2>/dev/null; tmux new -d -s p2od_watch "STOP_STEP=$NEW bash $E/p2o/distill_watch.sh $J $F $RUN"
log "watcher restarted for $J, stop at checkpoint $NEW"
# the step-<NEW> follow-up: export + bare dev120 probes of distill and control at that step (MECHANISM_GATE=off runs the
# probes even if the shift metric has not fallen; the outcome measure is the question, a rising gap is not a dead run)
tmux kill-session -t p2od_probes 2>/dev/null; tmux new -d -s p2od_probes "RUN=$RUN STEP=$NEW MECHANISM_GATE=${MECHANISM_GATE:-off} bash $E/p2o/distill_probes.sh"
log "probes orchestrator armed for step $NEW (mechanism gate ${MECHANISM_GATE:-off})"
