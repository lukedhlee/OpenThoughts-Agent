#!/bin/bash
# launch_arm.sh <run dir made by make_arm.py> — pre-flight, submit, and start the login-side tunnels for a Horizon RL arm.
# Run on the Horizon login node. The job's WORKDIR (what the RL runner imports) is the runtime checkout named in
# configs/checkouts.json (~/snowball/ota-rl-runtime, branch lukedhlee/horizon-rl); its scripts come from ~/snowball/ota-rl. Pre-flight (read-only; any
# failure refuses):
#   - the hydra args compose against the installed MarinSkyRL schema (validate_hydra_args.py)
#   - every environment hash of the train tree has a snapshot in the org (no task may make harbor build one: the org is
#     at its snapshot cap, and creating snapshots is forbidden)
#   - the org's started sandboxes + this arm's seats stay under MAX_ORG (default 1200)
# Then sbatch, and one tunnel.sh per TUNNEL_PORTS port (setsid; each exits when the job ends).
set -euo pipefail
RUN=${1:?run dir}; NAME=$(basename "$RUN")
OTA=${OTA:-$HOME/snowball/ota-rl}; HZ=$OTA/data/r2egym/horizon; PY=$HOME/snowball/envs/snowball/bin/python
RUNTIME=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['runtime'])" "$1/configs/checkouts.json")   # WORKDIR of the job
[ -f "$RUNTIME/hpc/shell_utils/nccl_flight_recorder.sh" ] && grep -q '^horizon = HPC' "$RUNTIME/hpc/hpc.py" || { echo "$RUNTIME is not the horizon-rl runtime"; exit 1; }
[ -z "$(git -C "$RUNTIME" status --short --untracked-files=no)" ] || { echo "$RUNTIME has local changes"; exit 1; }
TUNNEL_PORTS=${TUNNEL_PORTS:-18080,18081}; MAX_ORG=${MAX_ORG:-1200}   # Luke 10-01: org total under 1,200 (other sessions share it)
CFG=$RUN/configs/${NAME}_rl_config.json; SBF=$RUN/sbatch/${NAME}_rl.sbatch
[ -f "$CFG" ] && [ -f "$SBF" ] || { echo "missing $CFG or $SBF"; exit 1; }
grep -q "$OTA/data/r2egym/horizon/rl/node_bridge.sh" "$SBF" || { echo "sbatch does not run from $OTA"; exit 1; }
if squeue -h -u "$USER" -n "$NAME" -t PENDING,CONFIGURING,RUNNING -o %i | grep -q .; then echo "$NAME already in squeue"; exit 1; fi
export OMP_NUM_THREADS=1
echo "== hydra schema"; (cd "$HOME" && $PY $HZ/rl/validate_hydra_args.py "$CFG")
TREE=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['train_data'][0])" "$CFG")
SEATS=$($PY -c "import json,sys; a=json.load(open(sys.argv[1]))['skyrl_hydra_args']; print([x for x in a if 'n_concurrent_trials=' in x][-1].split('=')[1])" "$CFG")
echo "== snapshots for $TREE"
CEN=$(cd $OTA/data/r2egym/daytona_artifacts && $PY snapshot_census.py --key-file ~/.config/otagent/daytona_eval.env --tree "$TREE" 2>&1 | tail -4)
echo "$CEN"
if echo "$CEN" | grep -q "tree hashes not built yet"; then echo "REFUSED: some tree hashes have no snapshot (harbor would try to build one)"; exit 1; fi
echo "$CEN" | grep -Eq "tree hashes present in the org: ([0-9]+) of \1 distinct" || { echo "REFUSED: could not confirm every tree hash is present"; exit 1; }
if [ "${SKIP_SANDBOX_CHECK:-0}" != 1 ]; then
  N=$($PY $HZ/rl/sandbox_count.py | sed -n "s/.*'started': \([0-9]*\).*/\1/p")
  echo "== org sandboxes started: ${N:-?}; this arm: $SEATS seats; limit $MAX_ORG"
  [ -n "$N" ] && [ $((N + SEATS)) -le "$MAX_ORG" ] || { echo "REFUSED: org would exceed $MAX_ORG started sandboxes"; exit 1; }
fi
JOB=$(cd "$RUNTIME" && DCFT=$RUNTIME sbatch --parsable "$SBF" | tail -n 1 | grep -oE '^[0-9]+')   # TACC prints a banner first
[ -n "$JOB" ] || { echo "sbatch returned no job id"; exit 1; }
echo "submitted $NAME as job $JOB"; echo "$JOB" >> "$RUN/jobs.txt"
for p in ${TUNNEL_PORTS//,/ }; do
  setsid nohup bash $HZ/tunnel.sh "$JOB" "$p" > "$RUN/logs/tunnel_${JOB}_$p.log" 2>&1 < /dev/null &
done
echo "tunnels started on $TUNNEL_PORTS (logs $RUN/logs/tunnel_${JOB}_*.log); job log $RUN/logs/${NAME}_${JOB}.out"
