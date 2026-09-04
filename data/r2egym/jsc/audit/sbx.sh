#!/bin/bash
# sbx.sh — Mac-side driver for ONE r2egym sandbox on JURECA (used by the Opus audit subagents).
#   sbx.sh start  <task>            start the sandbox on the sleeper (SBX_JOB) -> writes state to audit/state/<task>
#   sbx.sh exec   <task> '<cmd>'    run a shell command in /testbed inside that sandbox (as the agent would)
#   sbx.sh verify <task>            run the task's official verifier; prints reward
#   sbx.sh stop   <task>            stop the sandbox
# Each round trip is Mac -> Jupiter login -> JURECA login (ControlMaster) -> srun --overlap on the sandbox's node (~5-10 s).
set -u
S=$(cd "$(dirname "$0")/.." && pwd); ST=$S/audit/state; mkdir -p $ST
JOB=${SBX_JOB:?export SBX_JOB=<JURECA sleeper/fleet jobid>}; R=/p/scratch/synthlaion/lee27
jrc() { bash $S/jrc.sh "$1"; }
case "${1:?start|exec|verify|stop}" in
  start)
    T=${2:?task}; NODEOPT=""; [ -n "${SBX_NODE:-}" ] && NODEOPT="SWG_NODE=$SBX_NODE"
    out=$(jrc "cd $R && SWG_JOB=$JOB $NODEOPT bash $R/r2egym_shell.sh startbg $T")
    echo "$out" | tail -1 > $ST/$T; cat $ST/$T ;;
  exec)
    T=${2:?task}; CMD=${3:?command}; read -r INST NODE < <(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(d['inst'],d['node'])" $ST/$T)
    ENC=$(printf '%s' "$CMD" | base64 | tr -d '\n')
    jrc "srun --jobid=$JOB --overlap -N1 -n1 -w $NODE --time=15 bash $R/r2egym_shell.sh exec $INST \"\$(echo $ENC | base64 -d)\"" 2>&1 | grep -v '^srun: ' ;;
  verify)
    T=${2:?task}; read -r INST NODE W < <(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(d['inst'],d['node'],d['w'])" $ST/$T)
    jrc "srun --jobid=$JOB --overlap -N1 -n1 -w $NODE --time=40 bash $R/r2egym_shell.sh verify $INST $W" 2>&1 | grep -v '^srun: ' ;;
  stop)
    T=${2:?task}; read -r INST NODE W < <(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(d['inst'],d['node'],d['w'])" $ST/$T)
    jrc "srun --jobid=$JOB --overlap -N1 -n1 -w $NODE --time=5 bash $R/r2egym_shell.sh stop $INST $W" 2>&1 | grep -v '^srun: ' ;;
  *) echo "unknown: $1"; exit 2 ;;
esac
