#!/bin/bash
# swesmith_shell.sh — drive a live raw SWE-smith sandbox the way an agent would, on ONE JURECA node.
#
#   swesmith_shell.sh start  <task_dir>          -> JSON with inst, w, node (instance left running)
#   swesmith_shell.sh exec   <inst> '<command>'  -> run <command> in /testbed as the agent (conda env active)
#   swesmith_shell.sh verify <inst> <w>          -> run the task's verifier (tests/test.sh), print reward + grade
#   swesmith_shell.sh stop   <inst> <w>          -> stop the instance and delete its staging dir
#
# All calls must run on the same node (apptainer instances are per node): from a login node use
#   srun --jobid=<sleeper job> --overlap -N1 -n1 bash swesmith_shell.sh ...
set -uo pipefail
ROOT=/p/scratch/synthlaion/lee27
export APPTAINER_CACHEDIR=$ROOT/apptainer_cache APPTAINER_TMPDIR=$ROOT/apptainer_tmp
AT=${BRIDGE_AGENT_TOOLS:-$ROOT/agent_tools}
HPATH="$AT/bin:$AT/uv_env/.venv/bin:/root/.local/bin:/testbed/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
# Order matters: the worker exports its PATH first and the agent's interactive .bashrc then activates
# conda on top, so conda's bin must end up FIRST (an earlier version exported PATH after activation and
# `python` silently became the agent_tools interpreter without the project's deps).
ENVPRE="export HOME=/tmp/fakehome XDG_DATA_HOME=/tmp/fakehome/.local/share XDG_CACHE_HOME=/tmp/fakehome/.cache XDG_CONFIG_HOME=/tmp/fakehome/.config; export PATH=$HPATH; source /opt/miniconda3/bin/activate testbed 2>/dev/null; cd /testbed;"

case "${1:?startbg|start|exec|verify|stop}" in
  startbg)
    # Run from the LOGIN node. The `start` step stays alive as long as the instance lives (the instance
    # daemons are children of the srun step), so it must run detached with its output in a file;
    # exec/verify/stop are separate --overlap steps that do return.
    TD=${2:?task_dir}; JOB=${SWG_JOB:-15588622}; OUT=$ROOT/swesmith_gate_runs/start_$$_$RANDOM.log
    setsid nohup srun --jobid=$JOB --overlap -N1 -n1 bash $ROOT/swesmith_shell.sh start "$TD" > $OUT 2>&1 </dev/null &
    for i in $(seq 120); do grep -q '"inst"' $OUT 2>/dev/null && break; sleep 2; done
    grep '"inst"' $OUT | tail -1 || { echo "start failed after 240 s:"; tail -5 $OUT; } ;;
  start)
    SWG_KEEP=1 bash $ROOT/swesmith_gate.sh "${2:?task_dir}" shell ;;
  exec)
    INST=${2:?inst}; CMD=${3:?command}
    timeout -k 5 "${SWG_EXEC_TIMEOUT:-600}" apptainer exec --cleanenv --env "PATH=$HPATH" --env LANG=C.UTF-8 --env LC_ALL=C.UTF-8 \
      --pwd /testbed "instance://$INST" /usr/bin/bash -lc "$ENVPRE $CMD" 2>&1
    echo "[exit=$?]" ;;
  verify)
    INST=${2:?inst}; W=${3:?w}
    rm -f $W/logs/verifier/reward.txt
    timeout -k 10 2100 apptainer exec --cleanenv --env "PATH=$HPATH" --env LANG=C.UTF-8 --env LC_ALL=C.UTF-8 --pwd /testbed \
      "instance://$INST" /usr/bin/bash -lc "export HOME=/tmp/fakehome; export PATH=$HPATH; chmod +x /tests/test.sh; (/tests/test.sh) 2>&1" > $W/verify_out.txt 2>&1
    echo "verifier exit: $?"; grep -E '^grade:|ENV_NOT_READY|passed|failed' $W/verify_out.txt | tail -4
    echo "reward: $(cat $W/logs/verifier/reward.txt 2>/dev/null || echo none)" ;;
  stop)
    INST=${2:?inst}; W=${3:?w}
    apptainer instance stop "$INST" >/dev/null 2>&1; rm -rf "$W"; echo "stopped $INST" ;;
  *) echo "unknown: $1"; exit 2 ;;
esac
