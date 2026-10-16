# router_argv.sh: the relay router's command line for one arm of run_pilot.sh (sourced by run_pilot.sh, and by
# router/tests/test_run_pilot_argv.py, which checks it on a Mac without a cluster).
#
#   router_argv <arm> <port>     sets RARGV: everything after `$PY $ROUTER`
#
# Plain bash 3.2 (the Mac's /bin/bash): no associative arrays, and an array that can be empty is expanded as
# ${X[@]+"${X[@]}"}, since bash < 4.4 calls "${X[@]}" of an empty array unbound under set -u.
#
# Reads run_pilot.sh's knobs and state: AGENT TEACHER_MAX_TOKENS TEACHER_GUARD TEACHER_RESAMPLES VERIFY_NOTE
# VERIFY_NOTE_TEXT TEACHER_HINT_TEXT MAX_INPUT STUDENT_TOKENIZER CTX_BUDGET ROW_MAX ROW_RESERVE STUDENT_VIEW_STRIP ASSIST
# ASSIST_CTX ASSIST_ERR ASSIST_GAP ASSIST_MAX TIME_TAKEOVER CLOCK BALANCE FAILOVER_5XX, SURL TURL DEADLINE R TREE
# HARBOR_SRC, and TC: the run's copy of TRIGGER_CONFIG (empty = none). With TC the relay arms and student_only get
# --trigger-config TC, and the intervention flags the router refuses next to it are left out (--context-budget-tokens,
# --assist*, --time-takeover-frac, and in a relay arm --verify-note*); relay_repair keeps --repair-on-parse-error (the
# router puts its parse_repair in front of the config's triggers); control is the teacher-alone baseline and runs as
# without it. Calls abort() (run_pilot.sh's) on an unknown arm or CLOCK.
router_argv() {
  local arm=$1 port=$2 PARSER PARSER_ARGS HARNESS_ARGS TARGS SARGS M CARGS
  PARSER=$HARBOR_SRC/harbor/agents/terminus_2/terminus_json_plain_parser.py
  PARSER_ARGS=(--terminus-parser $PARSER); HARNESS_ARGS=()
  [ "$AGENT" = msa ] && { PARSER_ARGS=(); HARNESS_ARGS=(--harness msa); }
  local tc=; [ -n "${TC:-}" ] && [ "$arm" != control ] && tc=1   # this arm runs the trigger config
  TARGS=(); [ -n "$TURL" ] && TARGS=(--teacher-url $TURL --teacher-model qwen38 --teacher-max-tokens $TEACHER_MAX_TOKENS)
  if [ -n "$TURL" ] && [ "$TEACHER_GUARD" = 1 ]; then TARGS+=(--teacher-format-guard --teacher-resamples $TEACHER_RESAMPLES)
    case $arm in relay_repair) ;; *) TARGS+=(${PARSER_ARGS[@]+"${PARSER_ARGS[@]}"});; esac; fi   # relay_repair passes the parser below
  local note=$VERIFY_NOTE; case $arm in relay*) [ -n "$tc" ] && note=0;; esac   # relay + config: its done_claim's teacher_note
  [ -n "$TURL" ] && [ "$note" = 1 ] && TARGS+=(--verify-note)
  [ -n "$TURL" ] && [ "$note" = 1 ] && [ -n "${VERIFY_NOTE_TEXT:-}" ] && TARGS+=(--verify-note-text "$VERIFY_NOTE_TEXT")
  [ -n "$TURL" ] && [ -n "${TEACHER_HINT_TEXT:-}" ] && TARGS+=(--teacher-hint-text "$TEACHER_HINT_TEXT")   # msa: teacher-only instruction
  [ "$MAX_INPUT" != 65536 ] && TARGS+=(--report-max-model-len $MAX_INPUT)
  SARGS=(); [ -n "$SURL" ] && SARGS=(--student-url $SURL --student-model snowball)   # empty on a teacher-only serve
  case $arm in control) M=(--mode teacher);; student_only) M=(--mode student);; relay) M=(--mode relay --student-think strip);; relay_keep) M=(--mode relay --student-think keep);;
    relay_repair) M=(--mode relay --student-think strip --repair-on-parse-error)   # with a config too (in front of it)
                  M+=(${PARSER_ARGS[@]+"${PARSER_ARGS[@]}"} --autofix --student-tokenizer $STUDENT_TOKENIZER);;
    *) abort "unknown arm $arm";; esac
  case $arm in relay*)
    [ -n "$CTX_BUDGET" ] && [ -z "$tc" ] && M+=(--context-budget-tokens $CTX_BUDGET)
    [ -n "$ROW_MAX" ] && M+=(--student-row-max-tokens $ROW_MAX --student-row-reserve $ROW_RESERVE)
    [ "${STUDENT_VIEW_STRIP:-0}" = 1 ] && M+=(--student-view-after-takeover strip)
    [ "${ASSIST:-0}" = 1 ] && [ -z "$tc" ] && M+=(--assist --assist-ctx "${ASSIST_CTX:-20000,35000}" --assist-error-streak ${ASSIST_ERR:-4} --assist-gap ${ASSIST_GAP:-3} --assist-max ${ASSIST_MAX:-6})
    [ -n "${TIME_TAKEOVER:-}" ] && [ -z "$tc" ] && M+=(--time-takeover-frac $TIME_TAKEOVER);; esac   # msa: one-turn teacher assists; time_budget hand-off
  [ -n "$tc" ] && M+=(--trigger-config $TC)
  case $CLOCK in wall) CARGS=(--budget-mode off);; repair) CARGS=(--budget-mode on);; paused) CARGS=(--budget-mode on --pause-model-calls);;
    *) abort "unknown CLOCK $CLOCK";; esac
  RARGV=("${M[@]}" ${HARNESS_ARGS[@]+"${HARNESS_ARGS[@]}"} --arm $arm --port $port --log-dir $R/router_$arm --tasks $TREE/router_tasks.json
         "${CARGS[@]}" --balance $BALANCE
         --engine-metrics --deadline-epoch $DEADLINE ${SARGS[@]+"${SARGS[@]}"} ${TARGS[@]+"${TARGS[@]}"} $([ "$FAILOVER_5XX" = 1 ] && echo --failover-5xx))
}
