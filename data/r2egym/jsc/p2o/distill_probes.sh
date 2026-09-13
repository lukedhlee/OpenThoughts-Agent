#!/bin/bash
# distill_probes.sh — run in a tmux on login02 right after the arm launch. Waits for the arm to end; applies the mechanism gate
# (shift_per_token over steps 11-12 must be <= 0.7 x steps 1-2, else no probe spend); exports checkpoint 12 (1 node); builds
# the two BARE dev120 probes (distill-s12, control-s12); brings the bridge/fleet back (12 nodes x 32 = 384 seats, 3.5 h, no
# chain); submits both probes (8 nodes each); cancels each probe once its eval block is logged; releases the fleet by id when
# both are gone; stops the bridge; prints the paired readout. Ids -> $E/p2o/distill_ids.txt, log -> $E/p2o/distill_probes.log.
set -u; export OMP_NUM_THREADS=1
E=/e/fscratch/reformo/lee27/experiments; C=/e/project1/transfernetx/lee27/code/snowball; OTA=/e/project1/transfernetx/lee27/code/OpenThoughts-Agent
RUN=${RUN:-snowball_ttband_p2o_distill_a}; STEP=${STEP:-12}; CK=$E/$RUN/$RUN/checkpoints; CTL=snowball_ttband_ns_c_lr5e7_c
SOCK=$(eval echo ~/.ssh/cm_juwels/bridge); W="ssh -o BatchMode=yes -S $SOCK juwels"; IP=10.128.1.2
IDS=$E/p2o/distill_ids.txt; LOG=$E/p2o/distill_probes.log; GATES=$E/p2o/distill_gates.log
log(){ echo "$(date '+%m-%d %H:%M') $*" | tee -a $LOG; }
# 0. wait for the arm to leave the queue
while squeue -h -n $RUN -o %i | grep -q .; do sleep 120; done
log "arm gone; latest checkpoint: $(cat $CK/latest_ckpt_global_step.txt 2>/dev/null)"
[ "$(cat $CK/latest_ckpt_global_step.txt 2>/dev/null)" = "$STEP" ] && [ -f $CK/global_step_$STEP/trainer_state.pt ] || { log "no complete checkpoint $STEP; no probes"; exit 1; }
# 1. mechanism gate from the gates log
python3 - $GATES $STEP <<'PY' || { log "MECHANISM GATE FAILED: shift_per_token did not fall by 30 % from steps 1-2 to the last two (override: MECHANISM_GATE=off)"; [ "${MECHANISM_GATE:-on}" = off ] || exit 1; }
import re, sys, statistics
v = {}
for line in open(sys.argv[1]):
    m = re.search(r"^step (\d+) .*? shift=([-0-9.e+nan]+)", line)
    if m and m.group(2) not in ("nan", "-"): v[int(m.group(1))] = float(m.group(2))
S = int(sys.argv[2]); early = [v[s] for s in (1, 2) if s in v]; late = [v[s] for s in (S - 1, S) if s in v]
print("shift early", early, "late", late)
sys.exit(0 if early and late and statistics.mean(late) <= 0.7 * statistics.mean(early) else 1)
PY
log "mechanism gate passed"
# 2. export checkpoint $STEP
cd $C && X=$(RUN=$RUN STEP=$STEP sbatch --parsable --export=ALL,RUN=$RUN,STEP=$STEP export_hf.sbatch); echo "export $X $(date -Is)" >> $IDS; log "export job $X"
while squeue -h -j $X -o %T | grep -q .; do sleep 30; done
MD=$E/exports/${RUN}_step${STEP}/model; [ -f $MD/model.safetensors.index.json ] || { log "EXPORT MISSING ($MD)"; exit 1; }
MC=$E/exports/${CTL}_step${STEP}/model; [ -f $MC/model.safetensors.index.json ] || { log "control export missing ($MC)"; exit 1; }
# fixed-token-set mechanism readout (1 node-h each): the exported checkpoint, and the base once, on the same traces
bash $E/p2o/distill_shift.sh $MD s${STEP} | tee -a $LOG
[ -f $E/p2o/shift_base/summary.tsv ] || bash $E/p2o/distill_shift.sh /e/fscratch/reformo/lee27/models/snowball-s3-nemotron-terminal-step1888 base | tee -a $LOG
# 3. probes
bash $E/p2o/build_s12_probe.sh p2od_s${STEP}_bare $MD > $E/p2o/build_p2od_s${STEP}_bare.log 2>&1 || { log "probe build failed (distill): $(tail -3 $E/p2o/build_p2od_s${STEP}_bare.log)"; exit 1; }
bash $E/p2o/build_s12_probe.sh p2oc_s${STEP}_bare $MC > $E/p2o/build_p2oc_s${STEP}_bare.log 2>&1 || { log "probe build failed (control): $(tail -3 $E/p2o/build_p2oc_s${STEP}_bare.log)"; exit 1; }
log "probes built"
# 4. bridge + forward + fleet
if ! curl -s -m 3 localhost:9924/status >/dev/null 2>&1; then
  S=/e/project1/transfernetx/lee27/code/harbor-marin/src/harbor/environments/apptainer/server.py; BL=/e/data1/mmlaion/lee27/apptainer_bridge/server_9924_$(hostname -s).log
  tmux new -d -s apptainer_bridge_9924 "BRIDGE_CLOSE_AFTER_RESPONSE=1 BRIDGE_LISTEN_BACKLOG=1024 BRIDGE_STALE_READY_SEC=2400 BRIDGE_MAX_HANDLER_THREADS=1500 python3 $S --port 9924 --host 0.0.0.0 >> $BL 2>&1"
  for i in $(seq 1 24); do sleep 5; curl -s -m 3 localhost:9924/status >/dev/null 2>&1 && break; done
fi
curl -s -m 5 http://$IP:9924/status | grep -q workers_alive || { log "bridge 9924 not answering"; exit 1; }
JW=$($W "getent hosts jwlogin03i" | awk '{print $1}'); ssh -O cancel -R "${JW}:9925:${IP}:9924" -S $SOCK juwels 2>/dev/null
ssh -O forward -R "${JW}:9925:${IP}:9924" -S $SOCK juwels || { log "forward failed"; exit 1; }
F=$($W "cd /p/project1/synthlaion/lee27/fleet && sbatch --parsable --nodes=12 --time=03:30:00 --export=ALL,HARBOR_SRC=/p/project1/synthlaion/lee27/harbor/src,WORKERS_PER_NODE=32,STAGING_BASE=/tmp/apptainer_staging,BRIDGE_LOGIN=jwlogin03i,BRIDGE_PORT=9925,MAX_CHAIN=0 -J apptainer_workers_juwels_p2odp juwels_workers.sbatch")
[ -n "$F" ] || { log "fleet sbatch failed"; exit 1; }; echo "fleet $F $(date -Is)" >> $IDS; log "probe fleet $F (12 nodes, 3.5 h)"
ok=0; for i in $(seq 1 120); do sleep 5; curl -s -m 3 http://$IP:9924/status | grep -q '"workers_alive": true' && { ok=1; break; }; done
[ $ok = 1 ] || { log "fleet did not register in 10 min; cancelling $F"; $W "scancel $F"; exit 2; }
# 5. submit both probes
cd $OTA; P1=$(DCFT=$PWD sbatch --parsable $E/p2od_s${STEP}_bare/sbatch/p2od_s${STEP}_bare_rl.sbatch); P2=$(DCFT=$PWD sbatch --parsable $E/p2oc_s${STEP}_bare/sbatch/p2oc_s${STEP}_bare_rl.sbatch)
[ -n "$P1" ] && [ -n "$P2" ] || { log "probe sbatch failed ($P1/$P2); cancelling fleet"; $W "scancel $F"; exit 3; }
echo "probe $P1 $(date -Is)" >> $IDS; echo "probe $P2 $(date -Is)" >> $IDS; log "probes $P1 (distill s$STEP) $P2 (control s$STEP)"
# 6. watch: eval block logged -> cancel that probe; both gone -> release fleet, stop bridge, readout
while true; do
  for p in $P1 $P2; do
    nm=$(squeue -h -j $p -o %j 2>/dev/null); [ -n "$nm" ] || continue
    l=$(ls -t $E/$nm/logs/*.out 2>/dev/null | head -1)
    [ -n "$l" ] && grep -q "WANDB_MIRROR kind=eval" "$l" && { log "$nm eval block logged -> scancel $p"; scancel $p; }
  done
  squeue -h -j $P1,$P2 -o %i 2>/dev/null | grep -q . || break
  sleep 120
done
log "both probes gone -> scancel fleet $F"; $W "scancel $F" >> $LOG 2>&1
sleep 600; tmux kill-session -t apptainer_bridge_9924 2>/dev/null; pkill -u $USER -f "server.py --port 9924" 2>/dev/null; log "bridge 9924 stopped"
python3 $E/p2o/s12_compare.py $E/p2od_s${STEP}_bare/p2od_s${STEP}_bare/trace_jobs/eval_sessions/p2od_s${STEP}_bare_eval_step0 $E/p2oc_s${STEP}_bare/p2oc_s${STEP}_bare/trace_jobs/eval_sessions/p2oc_s${STEP}_bare_eval_step0 2>&1 | tee -a $LOG
log "DONE"
