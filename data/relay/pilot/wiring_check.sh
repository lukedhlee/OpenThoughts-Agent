#!/bin/bash
# wiring_check.sh <task-tree> <n-tasks> <out-dir> — CPU-only end-to-end check of the pilot's wiring on real Daytona
# sandboxes, with scripted fake models in place of the GPUs: fake student + fake teacher (router/tests/fake_openai.py),
# both routers exactly as run_pilot.sh starts them (health checks included), relay_pilot.yaml rendered per arm, harbor
# (lukedhlee/terminus2-relay) running Terminus-2 on CalibForge sandboxes, then readout.py --gate final. Checks the
# parts the unit tests fake: the harbor config, the setup-files hook, trajectory raw_content + session_id joining the
# router log, and the readout. Arms (ARMS): control, relay_repair; the fake student follows SCENARIO (default
# repair: one unparseable reply, repaired, then a done claim). Sandboxes: n-tasks x arms, deleted by harbor
# at the end of each trial (cleanup_sandboxes.py lists any left behind).
#
#   PYTHONPATH=<harbor lukedhlee/terminus2-relay>/src bash wiring_check.sh <tree> 2 <out-dir>
#   env: PY (python with harbor deps + aiohttp), HARBOR (harbor entry point), KEYF (Daytona key file)
set -uo pipefail
TREE=${1:?task tree}; N=${2:-2}; OUT=${3:?out dir}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-/Users/lukedhlee/harbor/.venv/bin/python}; HARBOR=${HARBOR:-$(dirname $PY)/harbor}
KEYF=${KEYF:-$HOME/.config/otagent/daytona_eval.env}
NAME=$(basename "$OUT")
[ -e "$OUT" ] && { echo "$OUT exists"; exit 1; }
mkdir -p $OUT/jobs
P=$((22000 + RANDOM % 8000))
$PY $HERE/../router/tests/fake_openai.py --role student --model snowball --port $P --scenario ${SCENARIO:-repair} > $OUT/fake_student.log 2>&1 & F1=$!
$PY $HERE/../router/tests/fake_openai.py --role teacher --model qwen38 --port $((P+1)) > $OUT/fake_teacher.log 2>&1 & F2=$!
trap 'kill $F1 $F2 ${RPIDS:-} 2>/dev/null' EXIT
sleep 3
[ -f "$TREE/TASKS.txt" ] && head -$N "$TREE/TASKS.txt" > $OUT/TASKS.txt || ls "$TREE" | grep calibforge | head -$N > $OUT/TASKS.txt
$PY $HERE/pilot_tasks.py router-json --tree $TREE --list $OUT/TASKS.txt --out $OUT/router_tasks.json
ARMS=${ARMS:-control relay_repair}
PARSER=$(dirname $($PY -c "import harbor, os; print(os.path.dirname(harbor.__file__))"))/harbor/agents/terminus_2/terminus_json_plain_parser.py
port_of() { case $1 in control) echo $((P+2));; relay) echo $((P+3));; relay_keep) echo $((P+4));; relay_repair) echo $((P+5));; esac; }   # bash 3.2-safe
RPIDS=""
for arm in $ARMS; do
  case $arm in control) M=(--mode teacher);; relay) M=(--mode relay --student-think strip);; relay_keep) M=(--mode relay --student-think keep);;
    relay_repair) M=(--mode relay --student-think strip --repair-on-parse-error --terminus-parser $PARSER);; esac
  port=$(port_of $arm)
  $PY $HERE/../router/relay_router.py "${M[@]}" --arm $arm --port $port --log-dir $OUT/router_$arm \
    --tasks $OUT/router_tasks.json --budget-mode on --student-url http://127.0.0.1:$P/v1 --student-model snowball \
    --teacher-url http://127.0.0.1:$((P+1))/v1 --teacher-model qwen38 --status-every 30 > $OUT/router_$arm.log 2>&1 &
  RPIDS="$RPIDS $!"
done
for arm in $ARMS; do
  for i in $(seq 1 30); do grep -q RELAY_ROUTER_READY $OUT/router_$arm.log && break; sleep 1; done
  grep -q RELAY_ROUTER_READY $OUT/router_$arm.log || { echo "router $arm not ready"; cat $OUT/router_$arm.log; exit 1; }
done
set -a; source $KEYF; set +a
HP=""
for arm in $ARMS; do
  port=$(port_of $arm)
  CFG=$OUT/${NAME}_$arm.yaml
  sed "s#__JOB_NAME__#${NAME}_$arm#; s#__JOBS_DIR__#$OUT/jobs#; s#__API_BASE__#http://127.0.0.1:$port/v1#; s#__CONC__#$N#" $HERE/relay_pilot.yaml > $CFG
  $PY - $CFG $TREE $OUT/TASKS.txt <<'PY' || exit 1
import sys, yaml
p, tree, lst = sys.argv[1:4]
c = yaml.safe_load(open(p)); c['tasks'] = [{'path': f'{tree}/{l.strip()}'} for l in open(lst) if l.strip()]
yaml.safe_dump(c, open(p, 'w'), sort_keys=False)
from harbor_config.models.job.config import JobConfig
JobConfig.model_validate(c); print(p, 'validates')
PY
  $HARBOR jobs start --config $CFG > $OUT/harbor_$arm.log 2>&1 &
  HP="$HP $!"
done
for p in $HP; do wait $p; echo "harbor exit $?"; done
kill -TERM $RPIDS 2>/dev/null; sleep 2
echo "node_hours=0" > $OUT/run.meta
$PY $HERE/readout.py --run-dir $OUT --gate final --json $OUT/readout.json > /dev/null 2> $OUT/readout.txt
cat $OUT/readout.txt
