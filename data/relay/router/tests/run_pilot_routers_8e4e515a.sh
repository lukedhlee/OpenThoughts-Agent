# run_pilot.sh at 8e4e515a, "# ---- 3. routers" (lines 206-235), verbatim: the routers' argv before TRIGGER_CONFIG and
# pilot/router_argv.sh. test_run_pilot_argv.py runs it next to router_argv.sh (without a config they must agree).
PARSER=$HARBOR_SRC/harbor/agents/terminus_2/terminus_json_plain_parser.py
PARSER_ARGS=(--terminus-parser $PARSER); HARNESS_ARGS=()
[ "$AGENT" = msa ] && { PARSER_ARGS=(); HARNESS_ARGS=(--harness msa); }
for arm in $ARMS; do
  TARGS=(); [ -n "$TURL" ] && TARGS=(--teacher-url $TURL --teacher-model qwen38 --teacher-max-tokens $TEACHER_MAX_TOKENS)
  if [ -n "$TURL" ] && [ "$TEACHER_GUARD" = 1 ]; then TARGS+=(--teacher-format-guard --teacher-resamples $TEACHER_RESAMPLES)
    case $arm in relay_repair) ;; *) TARGS+=("${PARSER_ARGS[@]}");; esac; fi   # relay_repair passes the parser below
  [ -n "$TURL" ] && [ "$VERIFY_NOTE" = 1 ] && TARGS+=(--verify-note)
  [ -n "$TURL" ] && [ "$VERIFY_NOTE" = 1 ] && [ -n "${VERIFY_NOTE_TEXT:-}" ] && TARGS+=(--verify-note-text "$VERIFY_NOTE_TEXT")
  [ -n "$TURL" ] && [ -n "${TEACHER_HINT_TEXT:-}" ] && TARGS+=(--teacher-hint-text "$TEACHER_HINT_TEXT")   # msa: teacher-only instruction
  [ "$MAX_INPUT" != 65536 ] && TARGS+=(--report-max-model-len $MAX_INPUT)
  SARGS=(); [ -n "$SURL" ] && SARGS=(--student-url $SURL --student-model snowball)   # empty on a teacher-only serve
  case $arm in control) M=(--mode teacher);; student_only) M=(--mode student);; relay) M=(--mode relay --student-think strip);; relay_keep) M=(--mode relay --student-think keep);;
    relay_repair) M=(--mode relay --student-think strip --repair-on-parse-error "${PARSER_ARGS[@]}"
                  --autofix --student-tokenizer $STUDENT_TOKENIZER);;
    *) abort "unknown arm $arm";; esac
  case $arm in relay*)
    [ -n "$CTX_BUDGET" ] && M+=(--context-budget-tokens $CTX_BUDGET)
    [ -n "$ROW_MAX" ] && M+=(--student-row-max-tokens $ROW_MAX --student-row-reserve $ROW_RESERVE)
    [ "${STUDENT_VIEW_STRIP:-0}" = 1 ] && M+=(--student-view-after-takeover strip)
    [ "${ASSIST:-0}" = 1 ] && M+=(--assist --assist-ctx "${ASSIST_CTX:-20000,35000}" --assist-error-streak ${ASSIST_ERR:-4} --assist-gap ${ASSIST_GAP:-3} --assist-max ${ASSIST_MAX:-6})
    [ -n "${TIME_TAKEOVER:-}" ] && M+=(--time-takeover-frac $TIME_TAKEOVER);; esac   # msa: one-turn teacher assists; time_budget hand-off
  case $CLOCK in wall) CARGS=(--budget-mode off);; repair) CARGS=(--budget-mode on);; paused) CARGS=(--budget-mode on --pause-model-calls);;
    *) abort "unknown CLOCK $CLOCK";; esac
  $PY $ROUTER "${M[@]}" "${HARNESS_ARGS[@]}" --arm $arm --port ${PORT[$arm]} --log-dir $R/router_$arm --tasks $TREE/router_tasks.json \
    "${CARGS[@]}" --balance $BALANCE \
    --engine-metrics --deadline-epoch $DEADLINE "${SARGS[@]}" "${TARGS[@]}" $([ "$FAILOVER_5XX" = 1 ] && echo --failover-5xx) \
    > $R/router_$arm.log 2>&1 &
  RPID[$arm]=$!
done
