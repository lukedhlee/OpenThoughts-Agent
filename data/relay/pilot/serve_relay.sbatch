#!/bin/bash
# serve_relay.sbatch — the relay pilot's two model servers in one 2-node Jupiter allocation: the student (09-21 Datakit
# SFT, as its thinking-on TB2 evals served it) on the first node, the teacher (Qwen3.8-27B) on the second. The login-node
# driver run_pilot.sh does everything else (routers, harbor on Daytona, gates, readout) and scancels this job when the
# run ends; --time is the hard node-hour ceiling (2 nodes x 2 h = 4 node-h; Luke's cap for the 3-arm pilot).
#
#   sbatch serve_relay.sbatch            # then, on the login node: bash run_pilot.sh <jobid> <run-name>
#
# After both servers answer /health, each gets one real completion from here (served name, the student's think markers
# kept with skip_special_tokens=false, the teacher's reasoning split off by the parser) and only then are the endpoint
# files written: $E/endpoints/<jobid>.student and .teacher (URL), <jobid>.meta. Death check: if either server exits,
# the job writes <jobid>.DEAD and cancels itself; if they are not both up within UP_TIMEOUT s, the same.
#
#SBATCH --job-name=relay_pilot
#SBATCH --time=02:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH -p booster
#SBATCH --account transfernetx
#SBATCH --output=/e/fscratch/reformo/lee27/experiments/relay/pilot/logs/%x_%j.out
#SBATCH --exclude=jpbo-031-[01-48],jpbo-011-[01-48],jpbo-038-38,jpbo-004-46,jpbo-065-17,jpbo-074-22,jpbo-074-40,jpbo-048-41,jpbo-091-05,jpbo-044-0[1-5]
set -uo pipefail
C=/e/project1/transfernetx/lee27/code
HERE=${RELAY_PILOT_DIR:-$C/relay/pilot}
PY=$C/envs/snowball-v2/bin/python
E=/e/fscratch/reformo/lee27/experiments/relay/pilot; EP=$E/endpoints; LOGS=$E/logs; mkdir -p $EP $LOGS
UP_TIMEOUT=${UP_TIMEOUT:-1800}
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
SN=${NODES[0]}; TN=${NODES[1]}
echo "relay pilot serve job=$SLURM_JOB_ID student=$SN teacher=$TN ($(date -Is))"
srun --nodes=1 --ntasks=1 --overlap --cpu-bind=none --gres=gpu:4 --cpus-per-task=72 -w "$SN" bash $HERE/serve_node.sh student > $LOGS/serve_student_$SLURM_JOB_ID.log 2>&1 &
SPID=$!
srun --nodes=1 --ntasks=1 --overlap --cpu-bind=none --gres=gpu:4 --cpus-per-task=72 -w "$TN" bash $HERE/serve_node.sh teacher > $LOGS/serve_teacher_$SLURM_JOB_ID.log 2>&1 &
TPID=$!
die() { echo "DEAD: $* ($(date -Is))"; echo "$*" > $EP/$SLURM_JOB_ID.DEAD; rm -f $EP/$SLURM_JOB_ID.student $EP/$SLURM_JOB_ID.teacher
        kill $SPID $TPID 2>/dev/null; scancel $SLURM_JOB_ID; exit 1; }
SURL="http://$SN.jupiter.internal:8000/v1"; TURL="http://$TN.jupiter.internal:8000/v1"
t=0; up_s=""; up_t=""
while [ -z "$up_s" ] || [ -z "$up_t" ]; do
  kill -0 $SPID 2>/dev/null || die "student server exited before it came up (tail: $(tail -3 $LOGS/serve_student_$SLURM_JOB_ID.log | tr '\n' ' '))"
  kill -0 $TPID 2>/dev/null || die "teacher server exited before it came up (tail: $(tail -3 $LOGS/serve_teacher_$SLURM_JOB_ID.log | tr '\n' ' '))"
  [ -z "$up_s" ] && curl -sf --max-time 5 ${SURL%/v1}/health >/dev/null && { up_s=$t; echo "student UP after ${t}s"; }
  [ -z "$up_t" ] && curl -sf --max-time 5 ${TURL%/v1}/health >/dev/null && { up_t=$t; echo "teacher UP after ${t}s"; }
  [ $t -ge $UP_TIMEOUT ] && die "servers not up after ${UP_TIMEOUT}s (student=${up_s:-no} teacher=${up_t:-no})"
  sleep 10; t=$((t+10))
done
# one real completion each, from this node: served name, markers, reasoning split
SM=$(curl -s --max-time 600 $SURL/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"snowball","messages":[{"role":"user","content":"Print hello in bash."}],"max_tokens":400,"skip_special_tokens":false}')
echo "$SM" | $PY -c 'import json,sys; r=json.load(sys.stdin); c=r["choices"][0]["message"].get("content") or ""; print("STUDENT_SMOKE finish=%s usage=%s think_markers=%s content=%r" % (r["choices"][0]["finish_reason"], r.get("usage"), "<|start_think|>" in c or "<|end_think|>" in c, c[:200]))' \
  || die "student smoke completion failed: ${SM:0:300}"
TM=$(curl -s --max-time 600 $TURL/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen38","messages":[{"role":"user","content":"Print hello in bash. Be brief."}],"max_tokens":2000}')
echo "$TM" | $PY -c 'import json,sys; r=json.load(sys.stdin); m=r["choices"][0]["message"]; rs=m.get("reasoning") or m.get("reasoning_content") or ""; print("TEACHER_SMOKE finish=%s usage=%s reasoning_chars=%d content=%r" % (r["choices"][0]["finish_reason"], r.get("usage"), len(rs), (m.get("content") or "")[:200])); sys.exit(0 if rs else 3)' \
  || die "teacher smoke completion failed or has no reasoning: ${TM:0:300}"
echo "$SURL" > $EP/$SLURM_JOB_ID.student
echo "$TURL" > $EP/$SLURM_JOB_ID.teacher
printf 'student_node=%s\nteacher_node=%s\nstudent_up_s=%s\nteacher_up_s=%s\nstudent_model=%s\nteacher_model=%s\n' \
  "$SN" "$TN" "$up_s" "$up_t" "${STUDENT_MODEL:-/e/data1/mmlaion/lee27/models/grug-datakit-sft-20260921}" \
  "${TEACHER_MODEL:-/e/data1/mmlaion/lee27/models/Qwen3.8-27B}" > $EP/$SLURM_JOB_ID.meta
echo "ENDPOINTS student=$SURL teacher=$TURL ($(date -Is))"
trap 'echo "wall/term: stopping servers"; kill $SPID $TPID 2>/dev/null; rm -f $EP/$SLURM_JOB_ID.student $EP/$SLURM_JOB_ID.teacher' TERM EXIT
# hold; if either server dies mid-run, say so and release both nodes
while kill -0 $SPID 2>/dev/null && kill -0 $TPID 2>/dev/null; do sleep 30; done
die "a server exited mid-run (student alive: $(kill -0 $SPID 2>/dev/null && echo yes || echo no), teacher alive: $(kill -0 $TPID 2>/dev/null && echo yes || echo no))"
