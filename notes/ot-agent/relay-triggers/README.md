# Pluggable decision triggers for relay interventions (staged plan)

**Status (2026-10-15):** scoped, propose-only, no code yet. Waiting for Luke's review.
**Repo:** `lukedhlee/OpenThoughts-Agent`, code in `data/relay/router/` and `data/relay/triggers/`.
**Working copy:** a new git worktree off `lukedhlee/relay-coworker-ready`, branch `lukedhlee/relay-triggers` (never
the canonical checkout, never a cluster). Line anchors below are from `relay-coworker-ready` @ 5144c3e2 and drift.

## What this is

Coworkers will study student failure modes and let the teacher take over online. Today every intervention is hard-coded
in the router with its own flag and its own shape. This plan replaces that with three parts that can be swapped
separately:

- **Trigger (when):** a detector that looks at the episode and says "intervene now", with a reason and a score. Rules,
  an LLM judge, a learned score or a scripted plan all have the same interface.
- **Where the teacher steps in:** `rewrite` holds the student's reply before its action runs, throws it away, and has
  the teacher answer the same request, so the faulty action never runs. `next` lets the action run and has the teacher
  answer the next request, to recover from it.
- **Span (how long):** the teacher answers `k` turns (1, 2, 3, ...) and then hands back to the student, or `-1` keeps
  it to the end of the episode.

**The rule that shapes everything:** a trigger can only `rewrite` a turn if it can see the fault in the student's
*proposed* action. Faults that show up only in the *result* (an error, a timeout, a loop) are seen one turn later,
after the action ran, and we can't undo a sandbox action. So request-time triggers always act as `next`; reply-time
triggers can choose either. Running the same reply-time trigger as `rewrite` and as `next` is the clean test of
"prevent the mistake" vs "recover from it".

**Testing is offline first.** The router already logs every turn and every request body. A replay harness rebuilds
each turn of a recorded episode and runs any trigger over it causally, so a coworker can measure when it fires, how
often it fires on episodes that passed, and how early it fires on episodes that failed, without a sandbox. Online runs
are for the things replay can't answer: whether the teacher's takeover actually helps, and rewrite vs next.

## The interface

```python
@dataclass(frozen=True)
class Event:                      # what a trigger sees; built only from the past (causal)
    phase: Literal["request", "reply"]   # request: before the student is called, with the last action's result
    turn: int                            # reply: the student answered, its action has not run yet
    is_retry: bool
    owner: str                           # who answers this turn under the current policy
    messages: list[dict]                 # the harness's history, as the student sees it
    last_result: ActionResult | None     # request phase: the previous action and its output, parsed per harness
                                         # (command, exit code, timed_out, output text, truncated); Terminus-2 and
                                         # MSA observations normalized to one shape so a rule is harness-agnostic
    reply: Reply | None                  # reply phase: raw text, parsed action, parse_error
    scanner: EpisodeScanner              # relay_triggers.py features (loops, waits, error streaks, ...)
    clock: Clock                         # agent time used / budget
    view_tokens: Callable[[], int]       # lazy /tokenize count of the student's view
    interventions: tuple[Intervention, ...]

@dataclass(frozen=True)
class Decision:
    trigger: str; reason: str; score: float | None = None
    where: Literal["rewrite", "next"] = "next"
    span: int = -1                       # k >= 1 turns, or -1 = to the end
    teacher_note: str | None = None      # shown to the teacher only, never enters the harness history
    log_only: bool = False               # shadow mode: logged, never acted on

class Trigger(Protocol):
    name: str
    phases: frozenset[str]               # {"request"}, {"reply"} or both
    async def observe(self, ev: Event) -> Decision | None: ...
```

**Trigger families the interface has to make easy** (each is a small class or a YAML entry):

| family | phase | example | built-in |
|---|---|---|---|
| action output | request | the last command exited non-zero, timed out, or its output matches a pattern (`Traceback`, `command not found`, `No such file`), n times in a row | `action_error` (new: exit code / timeout / regex / streak params), plus today's `error_streak`, `cmd_timeout`, `same_output` |
| behavior pattern | request | loops, passive waits, edits that don't apply | `loop`, `no_progress_wait`, `edit_failed` |
| proposed action | reply | a destructive command, giving up, claiming done, a parse error | `destructive`, `gave_up`, `done_claim`, `parse_repair` |
| budget | request | context or time used past a fraction | `context_budget`, `time_budget` |
| judge | either | an LLM reads the recent turns and scores "stuck / about to fail" | `llm_judge` (stage 4) |
| score | either | student log-probs or a learned failure classifier over the turn | (coworkers' own; the router can ask the student for logprobs) |
| scripted | either | fire at turn t, or from a per-task plan (controlled experiments) | `scripted` (stage 4) |

A per-episode **Controller** owns the policy, so triggers stay pure detectors. It holds who owns the episode (student,
teacher with `k` turns left, or teacher to the end), cooldown, the per-episode cap, the earliest turn, the re-arm rule
after a hand-back, the retry memo (a repeated harness request gets the same answer), and arbitration when several
triggers fire (config order is priority; every fire is logged, the first non-shadow one acts). A `next` decision made at
reply time is stored and applied at the next request, which is how Terminus-2's done-claim already works.

Configured by one YAML file (`--trigger-config`). Built-ins are referenced by name, and a coworker's own trigger by
`module:Class`:

```yaml
policy: {cooldown: 3, max_interventions: 6, earliest_turn: 2, show_rejected_reply: false}
triggers:
  - use: loop                      # built-in, request phase
    span: -1
  - use: my_lab.triggers:DangerousEdit   # coworker's class, reply phase
    where: rewrite
    span: 2
  - use: llm_judge
    params: {endpoint: judge, prompt: prompts/stuck.txt, threshold: 0.7, timeout_s: 20}
    where: next
    span: 1
    log_only: true                 # shadow-test online before letting it act
```

## Knobs

| knob | values | default (today's behavior) |
|---|---|---|
| where | `rewrite` / `next` | per built-in, as today |
| span | `k >= 1` / `-1` | per built-in, as today |
| cooldown | student turns between interventions | assist: 2 (= --assist-gap 3 minus 1), others: n/a |
| max_interventions | per episode | assist: 6, takeover: 1 |
| earliest_turn | first turn a trigger may act | 2 |
| teacher sees the rejected reply (rewrite only) | no / yes | no |
| teacher's view of the failed turn (next only) | keep / hide the failed action and its output (the sandbox keeps its effects either way) | keep |
| student's view after a hand-back | teacher reasoning full / capped / stripped | latest full, older capped at 1,000 tokens |
| teacher_note | text in the teacher's view only | verify note on done-claim |
| log_only | shadow mode | off |
| judge timeout and clock | judge time paused or charged to the agent | paused |
| release (hand-back condition) | a trigger judged on the teacher's turns (an entry's `release:`, or a policy-level `release:` list); when it fires, the intervention ends at the next main request, sticky spans included | none: spans end after k turns or never |
| per-intervention view | `view:` on an entry or a decision overrides the three view knobs above for that intervention | the policy's |
| teacher | `teacher:` endpoint name on an entry or a decision | `teacher` (the only endpoint today) |
| combinators | `any_of`, `all_of`, `streak` (n turns in a row), `after_turn`, `before_turn`, `every`, nestable around any triggers | n/a |
| function triggers | any `f(ev, state)` (`state` per episode), in Python or as `use: module:func` / `use: path/file.py:func` | n/a |
| decision info | free-form diagnostics on a decision (features, a judge's rationale), kept in the decision log | none |

## Stage map

| stage | title | what | cost | gate |
|---|---|---|---|---|
| 0 | Interface and controller | `interventions.py`: Event, Decision, Trigger, Controller, YAML loader. No router change. | CPU | controller scenario table passes; router untouched |
| 1 | Offline replay harness | `triggers/replay.py`: rebuild events from `turns.jsonl` + bodies, run a trigger config, report fires vs outcome and lead time | CPU | reproduces the router's logged `would_fire` turn for turn on recorded student-only runs |
| 2 | Wire the router, same behavior | handle()/handle_msa() call the Controller with a legacy config built from today's flags | CPU | 81 existing tests unchanged; replayed request bodies give identical routing and legacy flags on 100 % of turns |
| 3 | New behavior | `k`-turn spans with hand-back, several interventions, cooldown/cap/re-arm, `next` for any reply trigger, uniform `intervention` record field; readout gates, select_kept and render read it | CPU | scenario tests; stage-2 identity still holds; a synthetic k=2 hand-back episode renders the right loss masks |
| 4 | LLM judge and scripted triggers | async judge (serial in reply phase, overlapped with tool time in request phase), timeout = no decision, clock pause; scripted plan trigger | ~2 node-h | judge latency p50/p95 per turn measured; replay of the judge on 200 episodes |
| 5 | Online smoke | 2 x 2 of where x span with one reply-time rule, plus one shadow arm, ~32 tasks | ~20-30 node-h | no router errors; records match controller logs; rows render; readout runs |
| 6 | Coworker docs and examples | "write a trigger" section, a 20-line rule example, a judge config | none | a fresh agent following only the docs writes a toy trigger and replays it |

**Critical path:** 0, then 1 (ship to coworkers as soon as it passes: they can study failure modes offline while the
router work continues), then 2, 3, 5. Stage 4 needs 3; stage 6 grows with 1 and 3.

## Global invariants (every stage)

- **No config, no change.** Without `--trigger-config` the router routes every request exactly as today and writes the
  same legacy record fields (`repair`, `assist`, `takeover`, `done_claim`, ...). New fields are additive.
- **The existing tests pass unchanged** (81 in `router/tests` and `triggers/tests`).
- **Causal.** A trigger only sees the past. The replay harness builds events turn by turn from the log, never from the
  finished episode.
- **Nothing is undone.** `rewrite` happens only before the harness sees the reply.
- **Teacher-only text never trains.** Notes and judge output stay out of the harness history, as the verify note does
  today. Thrown-away student replies are logged as preference pairs, never rendered as targets.
- **Minimal diff.** Built-ins keep their current logic; they are re-expressed, not rewritten.

## Borrow map (anchors drift; reconfirm at implementation)

- Router lifecycle, `data/relay/router/relay_router.py`: `h_chat` 787 (episode id, per-episode lock), `handle` 815 and
  `handle_msa` 924, request-time decisions 890-903 / 968-999, `answer_student` 1205 with reply-time decisions 1224-1273,
  `answer_student_msa` 1064 with 1083-1141, `take_over` 1183 (the only owner change), `answer` 1275 (teacher view,
  verify note, format guard, clock pause), `finish` 1483 (`ep.replies` join key, one `turns.jsonl` line per request),
  views `for_teacher` 454 / 506 and `for_student` 552 / 596, summarization routing 826-829, `/tokenize` 436.
- Triggers, `data/relay/triggers/relay_triggers.py`: `EpisodeScanner` (causal), `current_fires`, `CALIBRATED_CONFIG`
  1049, offline `scan_messages` / `fires_by_turn` / `detect`; `calibrate.py` (the 2026-09-25 calibration at <= 5 %
  false fires on passing episodes).
- Consumers that assume "one takeover, teacher to the end": `readout.py` `repair_view` 595-601 and H4 784-788,
  `would_fire` 558, `relay_takeovers`; `horizon/arm_gates.py:73`; `msa/gate.py`; `select_kept` (`stalled`,
  `teacher_worked`); `render.py` 85-125 and `render_msa.py` 109-160 (`--student-view-after-takeover`).
- Precedent: `pedagepa-horizon` branch `maybe_inject()` (a plan-driven trigger plus a per-record loss label).

## Risks

- **Judge cost in rewrite mode.** Every student reply waits for the judge before the harness sees it. Stage 4 measures
  this; a judge that is too slow should run as `next`.
- **Clean labels.** Only student-alone episodes give fire-vs-outcome labels without the teacher's influence. Which
  student-only runs with outcomes exist on Horizon is not yet counted, and the 2026-09-25 calibration transcripts are
  gone (their scratchpad was deleted).
- **Leaks.** A judge or rule must see only what the agent saw. Grader files or the verify note in a judge prompt would
  make the trigger look better than it is.
