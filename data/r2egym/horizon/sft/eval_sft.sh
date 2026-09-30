#!/bin/bash
# eval_sft.sh — agentic evals of Horizon-trained SFT checkpoints, REPS independent runs of each set: TB2.1 (88 tasks,
# train-fasttext excluded), SWE-bench Verified random-100 (2 shards of 50) and TB-lite (openthoughts tblite 2.0, 2 shards
# of 50). Every run is Jupiter's (the tb21/swe/tblite_6516_A_* runs): harbor-p0924 @ 761fb516,
# tb2_marin_policy_0924_65k16k.yaml (65,536 in / 16,384 out, 1,800 s agent budget, Daytona), 16 concurrent trials, one
# trial per task, against its own 1-node student serve (serve_relay.sbatch N_STUDENT=1 = serve_snowball POLICY=trained:
# EAGLE-3 draft, TP1 x DP4 x EP). The harbor side is data/tb2/horizon/tb2_driver.sbatch, the policy rendering is
# data/tb2/horizon/run_tb2.sh's.
#
# One Slurm job per run: Horizon's debug QOS allows 20 running and 40 submitted jobs per user, so the harbor driver runs
# as an overlapping step inside the serve job (srun --jobid --overlap) instead of as a second job, and this script is a
# queue that keeps at most MAXJOBS of the user's jobs submitted (all sessions count). It adopts serve jobs already named
# esrv_<run> and leaves alone a run whose separate tb2_drv_<run> job is RUNNING (the first launcher's two-job runs); a
# PENDING tb2_drv_<run> job of this script's runs is cancelled and replaced by an in-serve driver.
#
#   MODELS="hzA=<export dir> hzB=<export dir>" [REPS="1 2 3"] [SETS="tb21 swe_s0 swe_s1 tblite_s0 tblite_s1"] \
#     [MAXJOBS=34] [GROUPED=1] bash eval_sft.sh        (login node, inside tmux; it only submits and polls)
# GROUPED=1 serves all SETS of one (model, rep) from one multi-node job (one server and one run per node): 1 job slot per
# 5 runs. Two queues (e.g. one per model) may run side by side; each counts every job of the user against MAXJOBS.
#
# Runs are named <set>_<tag>_r<rep>_<DAY>; results under $S/experiments/sft_eval/tb2_jobs/<run>; per-run state in
# $S/experiments/sft_eval/state/<run>/ (serve job id, tries, driver log, done). Readout: eval_readout.py.
set -uo pipefail
MODELS=${MODELS:?"tag=<export dir> ..."}
REPS=${REPS:-1 2 3}; SETS=${SETS:-tb21 swe_s0 swe_s1 tblite_s0 tblite_s1}
MAXJOBS=${MAXJOBS:-34}
HERE=$(cd "$(dirname "$0")" && pwd); OTA=$(cd "$HERE/../../../.." && pwd)
if [ -z "${STUDENT_DRAFT:-}" ]; then   # Horizon serve knobs (draft, venv, caches) from serve_env.sh, read against the real scratch
  _sd=${SCRATCH_DIR-}; unset SCRATCH_DIR; source "$OTA/data/relay/horizon/serve_env.sh"; [ -n "$_sd" ] && export SCRATCH_DIR=$_sd
fi
S=${SCRATCH_DIR:-/scratch/11584/$USER}; T=$S/tasks
EV=$S/experiments/sft_eval; ST=$EV/state
export RELAY_EXP_DIR=$EV/serve RELAY_PILOT_DIR=$OTA/data/relay/pilot JOBS=$EV/tb2_jobs
export HARBOR_SRC=${HARBOR_SRC:-$HOME/snowball/harbor-p0924/src} HARBOR_SHA=${HARBOR_SHA:-761fb516}
POLICY_FILE=${POLICY_FILE:-tb2_marin_policy_0924_65k16k.yaml}
PY=${PY:-$HOME/snowball/envs/snowball/bin/python}
KEYF=${KEYF:-$HOME/.config/otagent/daytona_eval.env}
SERVE_TIME=${SERVE_TIME:-08:00:00}
DAY=${DAY:-$(date +%Y%m%d)}
TUNNEL_PORTS=18080,18081
mkdir -p "$RELAY_EXP_DIR/logs" "$RELAY_EXP_DIR/endpoints" "$EV/runs" "$EV/logs" "$JOBS" "$ST"; LOG=$EV/eval.log
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
declare -A MODEL
for m in $MODELS; do MODEL[${m%%=*}]=${m#*=}; [ -f "${m#*=}/config.json" ] || { say "no export at ${m#*=}"; exit 1; }; done
[ "$(git -C "${HARBOR_SRC%/src}" rev-parse --short=8 HEAD)" = "${HARBOR_SHA:0:8}" ] || { say "$HARBOR_SRC is not at $HARBOR_SHA"; exit 1; }
grep -q "type: daytona" "$OTA/data/tb2/jupiter/$POLICY_FILE" || { say "$POLICY_FILE is not a Daytona policy"; exit 1; }

set_env() {  # the task tree, count, exclusions and shard of one set, as Jupiter ran it
  EXCLUDE_TASKS=; SHARD=
  case $1 in
    tb21)      TASKS=$T/terminal_bench_2_1 NTASKS=89 EXCLUDE_TASKS=train-fasttext;;
    swe_s0)    TASKS=$T/swebench_verified_random100 NTASKS=100 SHARD=0/2;;
    swe_s1)    TASKS=$T/swebench_verified_random100 NTASKS=100 SHARD=1/2;;
    tblite_s0) TASKS=$T/openthoughts_tblite_2_0 NTASKS=100 SHARD=0/2;;
    tblite_s1) TASKS=$T/openthoughts_tblite_2_0 NTASKS=100 SHARD=1/2;;
    *) return 1;;
  esac
  [ "$(find "$TASKS" -mindepth 2 -maxdepth 2 -name task.toml | wc -l)" = "$NTASKS" ] || { say "task tree $TASKS does not have $NTASKS tasks"; return 1; }
}

render() {  # $1 run, $2 set, $3 model URL -> $EV/runs/$1.yaml (run_tb2.sh's rendering: jobs dir swapped, the rest verbatim)
  local cfg=$EV/runs/$1.yaml
  set_env "$2" || return 1
  sed "s#__JOB_NAME__#$1#; s#__API_BASE__#$3#; s#^jobs_dir: .*#jobs_dir: $JOBS#" "$OTA/data/tb2/jupiter/$POLICY_FILE" > "$cfg"
  EXCLUDE_TASKS=$EXCLUDE_TASKS SHARD=$SHARD MODE=full OMP_NUM_THREADS=1 "$PY" - "$cfg" "$TASKS" <<'PY' >> "$ST/$1/render.log" 2>&1 || return 1
import os, re, sys, yaml
p, tasks = sys.argv[1:3]; c = yaml.safe_load(open(p))
def agent_timeout(t):
    s = open(f"{tasks}/{t}/task.toml").read(); m = re.search(r"\[agent\][^\[]*?timeout_sec\s*=\s*([0-9.]+)", s)
    return float(m.group(1)) if m else 0.0
names = sorted((d for d in os.listdir(tasks) if os.path.isfile(f"{tasks}/{d}/task.toml")), key=lambda t: (-agent_timeout(t), t))
skip = [t for t in os.environ.get("EXCLUDE_TASKS", "").split(",") if t]
if skip:
    names = [t for t in names if t not in skip]; print(f"excluded: {skip}")
shard = os.environ.get("SHARD")
if shard:
    i, n = map(int, shard.split("/")); names = names[i::n]
c["n_attempts"] = 1
c.pop("datasets", None); c["tasks"] = [{"path": f"{tasks}/{t}"} for t in names]
yaml.safe_dump(c, open(p, "w"), sort_keys=False)
print(f"{len(names)} tasks at {c['n_concurrent_trials']} concurrent")
PY
  PYTHONPATH=$HARBOR_SRC "$PY" -c "import yaml; from harbor_config.models.job.config import JobConfig; JobConfig.model_validate(yaml.safe_load(open('$cfg')))" >> "$ST/$1/render.log" 2>&1
}

start_driver() {  # $1 run, $2 set, $3 serve job, $4 model URL, $5 node, $6 1 = release the serve job when harbor ends,
                  # $7 1 = start the login-side tunnels for this job: render, then the driver as an overlapping step on $5
  local run=$1 j=$3 url=$4 node=$5 rel=$6 left
  render "$run" "$2" "$url" || { say "$run render/validate FAILED ($ST/$run/render.log)"; return 1; }
  left=$(squeue -h -j "$j" -o %L)
  if [ "$7" = 1 ]; then
    for p in ${TUNNEL_PORTS//,/ }; do
      setsid nohup bash "$OTA/data/r2egym/horizon/tunnel.sh" "$j" "$p" > "$EV/logs/tunnel_${j}_$p.log" 2>&1 < /dev/null &
    done
  fi
  CFG=$EV/runs/$run.yaml SERVE_JOB=$j RUN_NAME=$run OTA=$OTA TUNNEL_PORTS=$TUNNEL_PORTS SCANCEL_SERVE=0 LOGD=$ST/$run/drv \
  setsid nohup bash -c "srun -p debug -A CCR24067 -t $left --jobid=$j --overlap -N1 -n1 -w $node -c 32 --export=ALL bash $OTA/data/tb2/horizon/tb2_driver.sbatch; echo DRIVER_STEP_EXIT \$?$([ "$rel" = 1 ] && echo "; scancel $j")" \
    > "$ST/$run/driver.out" 2>&1 < /dev/null &
  echo "$j" > "$ST/$run/driver_started"
  say "$run driver started in serve $j on $node ($url, ${left} left)"
}

# GROUPED=1: one multi-node serve job per (model, rep), one student server per node (serve_relay N_STUDENT = nodes), and
# every set of that rep on its own node, each against its own node's server (the same 1-node server and concurrency as
# a single run). Five runs then take one job slot instead of five. A server that dies takes the group's job down.
if [ "${GROUPED:-0}" = 1 ]; then
  read -ra SETA <<<"$SETS"; N=${#SETA[@]}
  GRPS=(); for rep in $REPS; do for tag in "${!MODEL[@]}"; do GRPS+=("$tag:$rep"); done; done
  say "EVAL_QUEUE_START grouped models='$MODELS' groups=${#GRPS[@]} x $N sets maxjobs=$MAXJOBS ota=$(git -C "$OTA" rev-parse --short HEAD) harbor=$HARBOR_SHA policy=$POLICY_FILE"
  while :; do
    Q=$(squeue -u "$USER" -h -o '%i %j %T'); ntot=$(echo "$Q" | grep -c .); left=0
    for g in "${GRPS[@]}"; do
      IFS=: read -r tag rep <<<"$g"; gname=${tag}_r${rep}_$DAY; gd=$ST/g_$gname; mkdir -p "$gd"
      [ -f "$gd/done" ] && continue
      left=$((left + 1))
      gj=$(echo "$Q" | awk -v n="esrv_g_$gname" '$2==n {print $1" "$3}')
      if [ -f "$gd/started" ]; then
        ndone=0
        for set in "${SETA[@]}"; do grep -q DRIVER_STEP_EXIT "$ST/${set}_${gname}/driver.out" 2>/dev/null && ndone=$((ndone + 1)); done
        if [ "$ndone" = "$N" ] || [ -z "$gj" ]; then
          for set in "${SETA[@]}"; do run=${set}_${gname}; touch "$ST/$run/done"
            say "$run $(grep -q DRIVER_STEP_EXIT "$ST/$run/driver.out" 2>/dev/null && echo "DONE ($(grep -h TB2_DONE "$ST/$run/driver.out" | tail -1 | cut -c1-120))" || echo "ENDED without the driver's exit line; see $ST/$run/driver.out")"; done
          [ -n "$gj" ] && scancel "${gj%% *}"; touch "$gd/done"; say "group $gname done"
        fi
        continue
      fi
      if [ -n "$gj" ]; then
        j=${gj%% *}; echo "$j" > "$gd/serve"
        if [ "${gj#* }" = RUNNING ] && [ -f "$RELAY_EXP_DIR/endpoints/$j.student" ]; then
          IFS=, read -ra U < "$RELAY_EXP_DIR/endpoints/$j.student"
          [ "${#U[@]}" = "$N" ] || { say "group $gname serve $j has ${#U[@]} student URLs, not $N"; scancel "$j"; continue; }
          for k in "${!SETA[@]}"; do
            run=${SETA[$k]}_${gname}; mkdir -p "$ST/$run"
            start_driver "$run" "${SETA[$k]}" "$j" "${U[$k]}" "$(echo "${U[$k]}" | sed -E 's#^[a-z]+://([^:/]+).*#\1#')" 0 "$([ "$k" = 0 ] && echo 1 || echo 0)"
          done
          touch "$gd/started"
        fi
        [ -f "$RELAY_EXP_DIR/endpoints/$j.DEAD" ] && { say "group $gname serve $j DEAD before its drivers"; scancel "$j"; }
        continue
      fi
      tries=$(cat "$gd/tries" 2>/dev/null || echo 0)
      if [ "$tries" -ge 2 ]; then touch "$gd/done"; say "group $gname FAILED: no serve after 2 tries"; continue; fi
      [ "$ntot" -lt "$MAXJOBS" ] || continue
      j=$(STUDENT_MODEL=${MODEL[$tag]} bash "$OTA/data/relay/horizon/serve_submit.sh" "$N" "$N" "$SERVE_TIME" "esrv_g_$gname" 2>&1 | tail -1)
      if [ "$j" -eq "$j" ] 2>/dev/null; then echo $((tries + 1)) > "$gd/tries"; ntot=$((ntot + 1)); say "group $gname serve job $j ($N nodes, try $((tries + 1)))"
      else say "group $gname serve submit failed: $j"; fi
    done
    [ "$left" -eq 0 ] && break
    sleep 60
  done
  say "EVAL_QUEUE_DONE; readout: $PY $OTA/data/r2egym/horizon/sft/eval_readout.py --tags ${!MODEL[*]} --day $DAY"
  exit 0
fi

RUNS=()
for rep in $REPS; do for tag in "${!MODEL[@]}"; do for set in $SETS; do RUNS+=("$set:$tag:$rep"); done; done; done
say "EVAL_QUEUE_START models='$MODELS' runs=${#RUNS[@]} maxjobs=$MAXJOBS ota=$(git -C "$OTA" rev-parse --short HEAD) harbor=$HARBOR_SHA policy=$POLICY_FILE"
while :; do
  Q=$(squeue -u "$USER" -h -o '%i %j %T')
  ntot=$(echo "$Q" | grep -c .)
  left=0
  for r in "${RUNS[@]}"; do
    IFS=: read -r set tag rep <<<"$r"; run=${set}_${tag}_r${rep}_$DAY; d=$ST/$run; mkdir -p "$d"
    [ -f "$d/done" ] && continue
    left=$((left + 1))
    ext=$(echo "$Q" | awk -v n="tb2_drv_$run" '$2==n {print $1" "$3}')
    if [ -n "$ext" ]; then
      if [ "${ext#* }" = PENDING ]; then scancel "${ext%% *}"; say "$run pending separate driver ${ext%% *} cancelled (in-serve driver instead)"
      else [ -f "$d/external" ] || { echo "${ext%% *}" > "$d/external"; say "$run runs with its own driver job ${ext%% *}"; }; continue; fi
    fi
    if [ -f "$d/external" ]; then   # its driver job has ended
      touch "$d/done"; say "$run DONE (driver job $(cat "$d/external"))"; continue
    fi
    sj=$(echo "$Q" | awk -v n="esrv_$run" '$2==n {print $1" "$3}')
    if [ -f "$d/driver_started" ]; then
      if grep -q DRIVER_STEP_EXIT "$d/driver.out" 2>/dev/null; then
        touch "$d/done"; say "$run DONE ($(grep -h TB2_DONE "$d/driver.out" 2>/dev/null | tail -1 | cut -c1-120))"
      elif [ -z "$sj" ]; then
        touch "$d/done"; say "$run ENDED without the driver's exit line (serve $(cat "$d/driver_started") gone); see $d/driver.out"
      fi
      continue
    fi
    if [ -n "$sj" ]; then
      j=${sj%% *}; echo "$j" > "$d/serve"
      if [ "${sj#* }" = RUNNING ] && [ -f "$RELAY_EXP_DIR/endpoints/$j.student" ]; then
        u=$(cut -d, -f1 "$RELAY_EXP_DIR/endpoints/$j.student"); start_driver "$run" "$set" "$j" "$u" "$(echo "$u" | sed -E 's#^[a-z]+://([^:/]+).*#\1#')" 1 1
      fi
      [ -f "$RELAY_EXP_DIR/endpoints/$j.DEAD" ] && { say "$run serve $j DEAD before its driver"; scancel "$j"; }
      continue
    fi
    tries=$(cat "$d/tries" 2>/dev/null || echo 0)
    if [ "$tries" -ge 2 ]; then touch "$d/done"; say "$run FAILED: no serve after 2 tries"; continue; fi
    [ "$ntot" -lt "$MAXJOBS" ] || continue
    j=$(STUDENT_MODEL=${MODEL[$tag]} bash "$OTA/data/relay/horizon/serve_submit.sh" 1 1 "$SERVE_TIME" "esrv_$run" 2>&1 | tail -1)
    if [ "$j" -eq "$j" ] 2>/dev/null; then echo $((tries + 1)) > "$d/tries"; echo "$j" > "$d/serve"; ntot=$((ntot + 1)); say "$run serve job $j (try $((tries + 1)))"
    else say "$run serve submit failed: $j"; fi
  done
  [ "$left" -eq 0 ] && break
  sleep 60
done
say "EVAL_QUEUE_DONE; readout: $PY $OTA/data/r2egym/horizon/sft/eval_readout.py --tags ${!MODEL[*]} --day $DAY"
