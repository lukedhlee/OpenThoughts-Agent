#!/bin/bash
# gepa_wave.sh <wave> [k=1] [nodes=8] — FALLBACK LAUNCHER. One self-contained SkyRL probe job per wave.
#
# ⚠ NOT the default path. The loop runs on a STANDING serve job plus a candidate queue: gepa_serve.sh up, then
# gepa_queue.sh add. The reason is measured (2026-09-20): a probe job spends ~0.5 h x nodes loading the 67B MoE
# before its first trial, about 4 node-hours on 8 nodes, and this launcher pays that again for every wave while the
# session reflects between them. The serve job pays it once and the GPUs stay busy.
#
# Use this only when the serve + queue path is unavailable: no free multi-node allocation to hold, a harbor/Daytona
# problem that the SkyRL generator path routes around, or a one-off replay of an older wave under the exact probe
# recipe the earlier numbers were produced with. It is kept working and tested for those cases.
#
# On Jupiter: turn one GEPA wave tree (tasks/gepa-<wave>, built by gepa_tree.py) into a Daytona pass@k probe job,
# print the COST LINE, and submit only with SUBMIT=1.
#
# One job per wave, one shard: every candidate of a task sits in the same shard, so every candidate meets the same
# sandbox and engine load (the p2o6all_s0 shape). The recipe is the screens' recipe, not a new one:
#   make_tt_wave.py --daytona   pool60k parity contract, harbor.environment_type=daytona + auto_snapshot=true,
#                               preset SOCKS via the jpbl-s01-02 microsocks, eval-org key file, setup-files-hook harbor
#   draftify_probe.sh           envs/snowball-v2 + the marin_vllm_eagle3 overlay + the EAGLE-3 draft
#                               (policy 2026-09-12: EVERY job, probes included, uses the draft -- numbers from a
#                                no-draft probe are not paired with anything here)
#   then, because draftify resets mask_exceptions from the arm template, the Daytona infra names go back on
#   and the 2026-09-07 transport fixes are pinned: 24 coordinators, HARBOR_OPENAI_CONNECT_TIMEOUT_SEC=120
#   and verifier 2400 s with preserve_logprobs_on_timeout=false (a 150 s budget scored 4-14 %/step as false zeros).
#
# HARD RULE (see .claude/skills/gepa-prompt-loop/SKILL.md): print the cost line, show it to Luke, get the go, and only
# then run with SUBMIT=1. This script never submits on its own.
#
#   bash gepa_wave.sh w1 1 8              # dry run: builds the config + sbatch, prints the cost line, submits nothing
#   SUBMIT=1 bash gepa_wave.sh w1 1 8     # submits
set -uo pipefail
WAVE=${1:?wave name (the one passed to gepa_tree.py)}; K=${2:-1}; NODES=${3:-8}
CONC=${CONC:-256}; WALL=${WALL:-06:00:00}; NOTIS=${NOTIS:-1}; COORD=${COORD:-24}; CONNECT=${CONNECT:-120}
VERIFIER=${VERIFIER:-2400}; ACCOUNT=${ACCOUNT:-laionize}; SUBMIT=${SUBMIT:-0}
C=/e/project1/transfernetx/lee27/code/snowball
G=$C/gepa
E=/e/fscratch/reformo/lee27/experiments
T=/e/fscratch/reformo/lee27/tasks
OTA=/e/project1/transfernetx/lee27/code/OpenThoughts-Agent
PREFIX=gepa$WAVE; NAME=${PREFIX}_s0
TREE=$T/gepa-$WAVE; ALLOW=$E/gepa/$WAVE/allow.txt
export OMP_NUM_THREADS=1

[ -d "$TREE" ] || { echo "no wave tree at $TREE -- run gepa_tree.py --wave $WAVE first"; exit 1; }
[ -f "$ALLOW" ] || { echo "no allowlist at $ALLOW -- run gepa_tree.py --wave $WAVE first"; exit 1; }
[ -x "$C/draftify_probe.sh" ] || [ -f "$C/draftify_probe.sh" ] || { echo "missing $C/draftify_probe.sh (the spec-decode step)"; exit 1; }
# the compute nodes reach Daytona only through this microsocks; without it every sandbox create fails
timeout 5 bash -c "</dev/tcp/10.128.1.2/7011" 2>/dev/null || { echo "microsocks 10.128.1.2:7011 not reachable; not building"; exit 1; }

NDIRS=$(grep -cve '^\s*$' "$ALLOW")
NCAND=$(python3 -c "import json;d=json.load(open('$E/gepa/$WAVE/cands.json'));print(len(d['arms']))")
NTASK=$((NDIRS / NCAND))
ATTEMPTS=$((NDIRS * K))
# Cost model measured from the p2o probes on this layout (2026-09-20): first-to-last trial gives 93 attempts/node-h
# (p2oAc_s0, 960 trials in 1.29 h on 8 nodes) and 111 (p2o6all_s0, 2880 in 3.24 h) -> ~100 steady state. On top of
# that sits a FIXED engine-load cost: p2oAc's job elapsed 1:48 against 1:19 of eval, so ~0.5 h x 8 nodes ~= 4 node-h
# before the first trial. Small waves are dominated by that startup, which is why a wave carries every candidate at
# once instead of one job per candidate.
COSTLINE=$(python3 -c "
ev = $ATTEMPTS / 100.0; st = 0.5 * $NODES; nh = ev + st
print('COST: wave $WAVE = %d tasks x %d arms x k=$K = %d attempts ~= %.1f GPU node-hours '
      '(%.1f eval + %.1f engine startup; ~%.1f h wall on $NODES nodes); we want a per-task reward + behaviour '
      'vector for each arm to score the Pareto front.' % ($NTASK, $NCAND, $ATTEMPTS, nh, ev, st, nh / $NODES))")

echo "== building $NAME =="
python3 $C/make_tt_wave.py --tasks "$TREE" --allow "$ALLOW" --prefix "$PREFIX" --shards 1 --daytona \
  --k "$K" --conc "$CONC" --nodes "$NODES" --engines $((NODES - 4)) --wall "$WALL" --verifier-timeout "$VERIFIER" \
  $([ "$NOTIS" = 1 ] && echo --no-tis) || exit 1

CF=$E/$NAME/configs/${NAME}_rl_config.json
SB=$E/$NAME/sbatch/${NAME}_rl.sbatch
echo "== draftifying (EAGLE-3 + snowball-v2 serving env) =="
bash $C/draftify_probe.sh "$NAME" || exit 1

echo "== re-pinning the transport settings draftify reset =="
python3 - "$CF" <<'PY' || exit 1
import json, sys
# draftify_probe.sh copies the ARM's mask list, which has no Daytona infra names: put them back (refresh_screen.py
# does exactly this after its own draftify call). Then pin the 2026-09-07 connect-timeout fix and the verifier budget.
INFRA = ['DaytonaError', 'DaytonaRateLimitError', 'DaytonaTimeoutError', 'DaytonaNotFoundError', 'DaytonaConflictError',
         'DaytonaSandboxStopError', 'SandboxBuildFailedError', 'SetupScriptError', 'EnvironmentStartTimeoutError']
cf = sys.argv[1]; c = json.load(open(cf)); a = c['skyrl_hydra_args']
def key(x): return x.lstrip('+').partition('=')[0]
def setarg(k, v):
    a[:] = [x for x in a if key(x) != k.lstrip('+')]
    a.append(k + '=' + str(v))
masks = []
for x in a:
    if key(x) == 'terminal_bench_config.harbor.mask_exceptions':
        masks = json.loads(x.split('=', 1)[1])
setarg('++terminal_bench_config.harbor.mask_exceptions',
       json.dumps(masks + [m for m in INFRA if m not in masks], separators=(',', ':')))
setarg('trajectory_runner.process_pool.num_coordinators', 24)
setarg('++terminal_bench_config.harbor.preserve_logprobs_on_timeout', 'false')
setarg('trajectory_runner.process_pool.eval_spread_coordinators', 'true')
json.dump(c, open(cf, 'w'), indent=2)
assert any('speculative_config=' in x for x in a), 'draft missing after draftify_probe.sh'
assert '++terminal_bench_config.harbor.environment_type=daytona' in a, 'daytona backend lost'
assert any('DaytonaRateLimitError' in x for x in a), 'daytona infra masks lost'
print('config: draft ok, daytona masks restored, 24 coordinators, preserve_logprobs_on_timeout=false')
PY
grep -q '^export HARBOR_OPENAI_CONNECT_TIMEOUT_SEC=' $SB \
  && sed -i "s/^export HARBOR_OPENAI_CONNECT_TIMEOUT_SEC=.*$/export HARBOR_OPENAI_CONNECT_TIMEOUT_SEC=$CONNECT/" $SB \
  || sed -i "/^export DCFT_RL_ENV=/a export HARBOR_OPENAI_CONNECT_TIMEOUT_SEC=$CONNECT" $SB
grep -q "HARBOR_OPENAI_CONNECT_TIMEOUT_SEC=$CONNECT" $SB || { echo "connect-timeout pin failed"; exit 1; }
bash -n $SB || { echo "generated sbatch does not parse"; exit 1; }

echo
echo "$COSTLINE"
echo "sbatch:  $SB"
echo "config:  $CF"
echo "tree:    $T/$PREFIX-s0  (allowlist $ALLOW)"
if [ "$SUBMIT" != 1 ]; then
  echo
  echo "DRY RUN -- nothing submitted. Show the cost line to Luke, get the go, then:"
  echo "  SUBMIT=1 bash $G/gepa_wave.sh $WAVE $K $NODES"
  exit 0
fi
JID=$(cd $OTA && DCFT=$OTA sbatch --parsable -A "$ACCOUNT" "$SB")
echo "$JID" > $E/gepa/$WAVE/job_s0
echo "SUBMITTED $NAME job $JID"
sleep 20; squeue -u "$USER" -o '%.9i %.16j %.2t %.8M %.5D %R' | grep -E "$PREFIX|JOBID"
