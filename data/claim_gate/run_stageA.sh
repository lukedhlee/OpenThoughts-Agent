#!/bin/bash
# run_stageA.sh — claim gate stage A on the login node: oracle agent, 8 train tasks, bridge 9930. Needs a live fleet.
set -uo pipefail
C=/e/project1/transfernetx/lee27/code
PY=$C/envs/snowball-v2/bin/python; HARBOR=$C/envs/snowball-v2/bin/harbor
export PYTHONPATH=${HARBOR_SRC:-$C/harbor-claimgate/src} OMP_NUM_THREADS=1
TREE=${TREE:-/e/fscratch/reformo/lee27/tasks/r2egym-tt-v2-train}
R=${R:-/e/data1/mmlaion/lee27/experiments/claim_gate/stageA}; mkdir -p $R/jobs
NAME=stageA_$(date +%m%d_%H%M)
$PY - "$R" "$TREE" "$NAME" <<'PY'
import sys, yaml
r, tree, name = sys.argv[1:]
c = yaml.safe_load(open('/e/home/jusers/lee27/jupiter/OpenThoughts-Agent/data/claim_gate/stageA_oracle.yaml'))
c['job_name'], c['jobs_dir'] = name, r + '/jobs'
c['tasks'] = [{'path': '%s/%s' % (tree, t.strip())} for t in open(r + '/tasks.txt') if t.strip()]
yaml.safe_dump(c, open('%s/%s.yaml' % (r, name), 'w'), sort_keys=False)
PY
echo "harbor $NAME ($(date))"
$HARBOR jobs start --config $R/$NAME.yaml > $R/$NAME.log 2>&1
echo "exit $? ($(date))"; tail -20 $R/$NAME.log
