#!/usr/bin/env bash
# BAND PRE-SUBMIT GATE -- run on a JUPITER LOGIN NODE, before sbatch.
#
# WHY THIS FILE EXISTS
# Every check below encodes a rule that was already written down in
# ai_memory/gotchas.md and then violated anyway, costing at least one job each.
# Prose does not enforce invariants; this script does. If you are tempted to add
# a rule to gotchas.md, ask first whether it belongs here instead.
#
# Exit non-zero = DO NOT SUBMIT. Every failure names the job it already cost.
#
# Usage:  bash band_preflight.sh <rl_config.yaml> <walltime HH:MM:SS> <fleet_jobid>
set -uo pipefail

YAML="${1:?usage: band_preflight.sh <rl_config.yaml> <walltime> <fleet_jobid>}"
WALL="${2:?missing walltime HH:MM:SS}"
FLEET="${3:?missing JURECA fleet jobid}"

CM="$HOME/.ssh/cm_jureca/qwen36"
JH="jureca05.fz-juelich.de"
# Startup is engine load + weight sync + sandbox spin-up. Measured 13-20 min on
# 8 nodes; 30 min is the margin, not the estimate.
STARTUP_MIN=30

FAIL=0
ok()   { printf '  \033[32mPASS\033[0m %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; FAIL=1; }
note() { printf '       %s\n' "$1"; }

jureca() { ssh -o BatchMode=yes -S "$CM" "$JH" "$@" 2>/dev/null; }

hms_to_min() {  # accepts D-HH:MM:SS, HH:MM:SS, MM:SS
  python3 - "$1" <<'PY'
import sys
t = sys.argv[1].strip()
d = 0
if "-" in t:
    d, t = t.split("-", 1)
    d = int(d)
p = [int(x) for x in t.split(":")]
while len(p) < 3:
    p.insert(0, 0)
print(d * 1440 + p[0] * 60 + p[1] + (1 if p[2] else 0))
PY
}

echo "=== BAND PRE-SUBMIT GATE ==="
echo "config:   $YAML"
echo "walltime: $WALL"
echo "fleet:    $FLEET"
echo

# ---------------------------------------------------------------- 1. config file
echo "[1] config sanity"
if [[ ! -f "$YAML" ]]; then
  bad "config not found: $YAML"
else
  ok "config exists"

  # TP>1 makes vLLM build a distributed executor that copies the parent actor's
  # os.environ into a nested Ray runtime_env; Ray then asserts on its own
  # __RAY_WORKER_PROCESS_SETUP_HOOK_ENV_VAR and NO engine ever starts. SkyRL
  # mislabels it "port collision (EADDRINUSE)" and burns 5x120s of allocation on
  # a deterministic assertion.  Cost: 8-node job 1247578, entire allocation.
  TP=$(grep -E '^\s*inference_engine_tensor_parallel_size:' "$YAML" | tail -1 | tr -dc '0-9')
  if [[ -n "$TP" && "$TP" != "1" ]]; then
    bad "inference_engine_tensor_parallel_size=$TP -- TP>1 never brings up an engine (cost: 1247578)"
    note "vLLM's distributed executor trips Ray's __RAY_WORKER_PROCESS_SETUP_HOOK assertion."
    note "TP=1 is also faster here: 4 engines/node instead of 1, no collectives."
  else
    ok "TP=1"
  fi

  # The launcher derives these from context_budget; declaring one is a hard error
  # AFTER submit, i.e. it wastes a queue wait. Fail here instead.
  for K in max_model_len max_prompt_length max_input_tokens max_episodes; do
    if grep -qE "^\s+${K}:" "$YAML"; then
      bad "declares derived context field '$K' -- launcher computes it from context_budget"
    fi
  done

  # Keys the consumer accepts, echoes, and ignores. See ai_memory/DEAD_KEYS.md.
  # A dead key is a silent no-op, which is worse than an error because it reads
  # as a fix.
  #
  # ONLY still-inert keys belong here. store_all_messages and override_timeout_sec
  # were BOTH later fixed (harbor 179b31e9; BRIDGE_EXEC_TIMEOUT above the agent
  # budget) and listing them produced false FAILs on a correct config -- a gate
  # that cries wolf gets switched off, which is worse than no gate. When a dead
  # key is fixed, delete it from here in the same commit.
  for K in strict_json_parser compaction_reserved; do
    grep -qE "^\s+${K}:" "$YAML" && bad "declares DEAD key '$K' (inert -- see ai_memory/DEAD_KEYS.md)"
  done
  grep -qE '^\s+reserved:' "$YAML" && grep -qE '^\s*compaction:' "$YAML" \
    && bad "declares DEAD key 'compaction.reserved' -- use context_budget.client_window_tokens"

  # hf_upload_mode is not inert by itself: the whole upload callback is gated on
  # hf_hub_repo_id, a DIFFERENT key. Grep for the guard, not the key.
  if grep -qE 'hf_upload_mode:' "$YAML" && ! grep -qE 'hf_hub_repo_id:' "$YAML"; then
    bad "hf_upload_mode set without hf_hub_repo_id -- upload callback is never constructed"
  fi

  # extra_body / interleaved_thinking are implemented for terminus_2, openhands and
  # mini_swe_agent only. Under opencode they are inert AND misleading: the config
  # claims a thinking mode it cannot deliver. Server-side
  # default_chat_template_kwargs is the only lever that reaches an external agent.
  if grep -qE 'agent(_name)?:\s*opencode' "$YAML" && grep -qE '^\s+extra_body:' "$YAML"; then
    note "WARN: extra_body under opencode is inert (no OpenCode path in harbor)."
    note "      Use engine_init_kwargs.default_chat_template_kwargs instead."
  fi

  # A band is per-group: 0 < passes < n. n=1 cannot produce a band at all.
  NS=$(grep -E '^\s*n_samples_per_prompt:' "$YAML" | tail -1 | tr -dc '0-9')
  if [[ -z "$NS" || "$NS" -lt 2 ]]; then
    bad "n_samples_per_prompt=${NS:-unset} -- a band needs n>=2 (pass@4 for Marianna parity)"
  else
    ok "n_samples_per_prompt=$NS"
  fi

  # max_turns: 999999 is why trials ran unbounded; Marianna caps 50.
  MT=$(grep -E '^\s*max_turns:' "$YAML" | tail -1 | tr -dc '0-9')
  if [[ -n "$MT" && "$MT" -gt 200 ]]; then
    bad "max_turns=$MT is effectively unbounded -- a single trial can eat a slot for an hour"
  else
    ok "max_turns=${MT:-unset}"
  fi
fi
echo

# --------------------------------------------------------- 2. ControlMaster + BOTH forwards
# THE defining failure of this project: the master carries TWO reverse forwards
# and restoring only one yields a PASSING route gate with 100% rollout timeouts.
# Until 08-05 no rollout had EVER produced a multi-step trajectory because of it.
echo "[2] JURECA ControlMaster and BOTH reverse forwards"
if ! ssh -S "$CM" -O check "$JH" >/dev/null 2>&1; then
  bad "no ControlMaster at $CM (note: this socket lives on the JUPITER login node, not your laptop)"
else
  ok "ControlMaster alive"
  LISTEN=$(jureca 'ss -ltn')
  if grep -q '10\.14\.0\.46:9923' <<<"$LISTEN"; then
    ok "bridge forward 9923 present (workers -> bridge)"
  else
    bad "MISSING bridge forward 10.14.0.46:9923 -- workers cannot reach the bridge"
    note "This is the forward whose absence caused every one-step rollout in the project."
  fi
  RPORT=$(grep -E 'SKYRL_ROLLOUT_HTTP_ENDPOINT_PORT' "$YAML" | tr -dc '0-9')
  if [[ -n "$RPORT" ]]; then
    if grep -q "10\.14\.0\.46:${RPORT}\b" <<<"$LISTEN"; then
      note "rollout forward $RPORT already listening -- it MUST be re-pointed at the new"
      note "head node after allocation. A stale target gives an empty agent/ dir with a"
      note "healthy bridge and fails only at the 2100s timeout (cost: 1246853)."
    else
      note "rollout forward $RPORT not yet installed -- expected; install it after allocation."
    fi
    note "cancel needs the EXACT original connect address, not a wildcard:"
    note "  ssh -S \$CM -O cancel -R 10.14.0.46:${RPORT}:<OLD_HEAD_IP>:8000 $JH"
  fi
fi
echo

# ----------------------------------------------------------------- 3. fleet budget
# The old preflight checked workers_alive -- a liveness check doing duty as a
# sufficiency check. A fleet that expires mid-run wastes the whole allocation.
echo "[3] JURECA fleet outlives the job"
LEFT=$(jureca "squeue -j $FLEET -h -o '%L'" | tr -d ' ')
if [[ -z "$LEFT" ]]; then
  bad "fleet $FLEET not in the JURECA queue"
else
  LMIN=$(hms_to_min "$LEFT"); WMIN=$(hms_to_min "$WALL")
  NEED=$((WMIN + STARTUP_MIN))
  if (( LMIN < NEED )); then
    bad "fleet has ${LMIN}min left, job needs ${WMIN}min + ${STARTUP_MIN}min startup = ${NEED}min"
    note "dc-cpu is usually near-empty; a fresh 32-node/24h fleet starts in under a minute."
  else
    ok "fleet ${LMIN}min left >= ${NEED}min needed"
  fi
  NW=$(jureca "grep -c 'starting 16 workers' /p/scratch/synthlaion/\$USER/../lee27/dc_agent_eval/logs/apptainer_workers_${FLEET}.out" | tr -dc '0-9')
  [[ -n "$NW" && "$NW" -gt 0 ]] && ok "$NW worker dispatchers started" \
    || note "could not count worker dispatchers (non-fatal)"
fi
echo

# ------------------------------------------------------------------- 4. bridge
echo "[4] bridge is reachable AND has capacity"
BURL="${APPTAINER_BRIDGE_URL:-http://10.128.1.2:9920}"
ST=$(curl -s -m 10 "$BURL/status")
if [[ -z "$ST" ]]; then
  bad "bridge unreachable at $BURL"
else
  python3 - "$ST" <<'PY' || FAIL=1
import json, sys
try:
    d = json.loads(sys.argv[1])
except Exception as e:
    print(f"  \033[31mFAIL\033[0m bridge /status unparseable: {e}"); raise SystemExit(1)
alive = d.get("workers_alive")
ready = (d.get("envs") or {}).get("ready", 0)
GREEN, RED, OFF = "\033[32m", "\033[31m", "\033[0m"
tag = f"{GREEN}PASS{OFF}" if alive else f"{RED}FAIL{OFF}"
print(f"  {tag} workers_alive={alive}")
print(f"       envs.ready={ready} queue={d.get('queue_size')} active_jobs={d.get('active_jobs')}")
if not alive:
    print("       workers_alive:false WITH live worker procs = missing 9923 forward,")
    print("       not dead workers. Check with: pgrep -fc worker.py")
    raise SystemExit(1)
PY
fi
echo

echo "=== $( ((FAIL)) && echo 'GATE FAILED -- DO NOT SUBMIT' || echo 'GATE PASSED' ) ==="
exit $FAIL
