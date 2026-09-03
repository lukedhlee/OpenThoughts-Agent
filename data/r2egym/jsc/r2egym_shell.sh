#!/bin/bash
# r2egym_shell.sh — drive a live raw R2E-Gym sandbox the way the agent does, on ONE JURECA node.
# Port of data/swesmith/jsc/swesmith_shell.sh for the r2egym image family (no conda: the project venv is
# /testbed/.venv, HOME is the image's /root, the verifier is the raw tests/test.sh offline-patched by the gate).
#
#   r2egym_shell.sh startbg <task>            -> JSON with inst, w, node (from the LOGIN node; needs SWG_JOB=<sleeper>)
#   r2egym_shell.sh start   <task>            -> same, run ON the node (blocks while the instance lives)
#   r2egym_shell.sh exec    <inst> '<command>' -> run <command> in /testbed as the agent would
#   r2egym_shell.sh verify  <inst> <w>        -> run the task's verifier (pristine leg of envgate_tt: no oracle), print reward
#   r2egym_shell.sh stop    <inst> <w>        -> stop the instance and delete its staging dir
# <task> is a dir name under $EG_STAGE_ROOT (default /p/scratch/synthlaion/lee27/envgate_tt). exec/verify/stop must run
# on the node that started the instance: `srun --jobid=<sleeper> --overlap -N1 -n1 -w <node> bash r2egym_shell.sh ...`.
set -uo pipefail
ROOT=/p/scratch/synthlaion/lee27
export APPTAINER_CACHEDIR=$ROOT/apptainer_cache APPTAINER_TMPDIR=$ROOT/apptainer_tmp
AT=${BRIDGE_AGENT_TOOLS:-$ROOT/agent_tools}
HPATH="$AT/bin:$AT/uv_env/.venv/bin:/root/.local/bin:/testbed/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ENVPRE="export HOME=/root XDG_DATA_HOME=/root/.local/share XDG_CACHE_HOME=/root/.cache XDG_CONFIG_HOME=/root/.config; export PATH=$HPATH; cd /testbed;"
GATE=${EG_GATE:-$ROOT/envgate_tt.sh}

case "${1:?startbg|start|exec|verify|stop}" in
  startbg)
    T=${2:?task}; JOB=${SWG_JOB:?set SWG_JOB=<sleeper jobid>}; NODEOPT=${SWG_NODE:+-w $SWG_NODE}
    mkdir -p $ROOT/r2egym_shell_runs; OUT=$ROOT/r2egym_shell_runs/start_${T}_$$.log
    setsid nohup srun --jobid=$JOB --overlap -N1 -n1 $NODEOPT bash $ROOT/r2egym_shell.sh start "$T" > $OUT 2>&1 </dev/null &
    for i in $(seq 120); do grep -q '"inst"' $OUT 2>/dev/null && break; sleep 2; done
    grep '"inst"' $OUT | tail -1 || { echo "start failed after 240 s:"; tail -5 $OUT; } ;;
  start)
    EG_KEEP=1 bash $GATE "${2:?task}" shell ;;
  exec)
    INST=${2:?inst}; CMD=${3:?command}
    timeout -k 5 "${SWG_EXEC_TIMEOUT:-600}" apptainer exec --cleanenv --env "PATH=$HPATH" --env LANG=C.UTF-8 --env LC_ALL=C.UTF-8 \
      --pwd /testbed "instance://$INST" /usr/bin/bash -lc "$ENVPRE $CMD" 2>&1
    echo "[exit=$?]" ;;
  verify)
    INST=${2:?inst}; W=${3:?w}
    rm -f $W/logs/verifier/reward.txt
    timeout -k 10 2100 apptainer exec --cleanenv --env "PATH=$HPATH" --env LANG=C.UTF-8 --env LC_ALL=C.UTF-8 --pwd /testbed \
      "instance://$INST" /usr/bin/bash -lc "$ENVPRE chmod +x /tests/test.sh; (/tests/test.sh) 2>&1" > $W/verify_out.txt 2>&1
    echo "verifier exit: $?"; grep -oE "[0-9]+ failed,? ?[0-9]* ?p?a?s?s?e?d?|[0-9]+ passed" $W/verify_out.txt | tail -1
    echo "reward: $(cat $W/logs/verifier/reward.txt 2>/dev/null || echo none)" ;;
  stop)
    INST=${2:?inst}; W=${3:?w}
    apptainer instance stop "$INST" >/dev/null 2>&1; rm -rf "$W"; echo "stopped $INST" ;;
  *) echo "unknown: $1"; exit 2 ;;
esac
