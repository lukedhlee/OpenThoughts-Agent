#!/usr/bin/env python3
"""Relay interventions: pluggable decision triggers and the per-episode Controller that turns their fires into routing.

Plan: notes/ot-agent/relay-triggers/ (README.md, stage0_interface_scope.md). Three parts that swap separately:
  * Trigger (when): a detector that looks at an Event and returns a Decision (intervene now, why, how) or None. Rules,
    an LLM judge, a learned score or a scripted plan all have this interface.
  * where: `rewrite` holds the student's reply before its action runs, discards it and has the teacher answer the same
    request (the faulty action never runs); `next` lets the action run and has the teacher answer the next request.
  * span: the teacher answers k >= 1 turns and hands back to the student, or -1 keeps the episode to the end (sticky).
A request-time trigger sees a fault only in an action's result, after the action ran, so it always acts as `next` (the
teacher answers the request it fired on); a reply-time trigger may choose either.

The Controller owns the policy so triggers stay pure detectors: who owns the episode, cooldown, caps, earliest turn,
re-arm after a hand-back, the retry memo, and arbitration when several triggers fire. It is a state machine over
events in order and does no I/O itself (a trigger may: Event.view_tokens, a judge call).

Per main request the router calls
    r = await ctl.begin_request(ev)   # hand-back of a finished span, a pending `next`, the retry memo; no trigger asked
    ... the router's endings (deadline, budgets, hard end) go here: they read ctl.sticky, as they read ep.owner today
    r = await ctl.on_request(ev)      # request-time triggers (calls begin_request itself when the router did not)
    if r.who == 'student': r = await ctl.on_reply(reply_ev)   # reply-time triggers; r.who == 'teacher' -> rewrite
Every call returns a Routing (who answers, why, the Intervention in force, the teacher note, the DecisionRecords of
every fire it saw: acted / pending / suppressed with why / shadow).

Semantics (pinned in tests/test_interventions.py):
  * owner = teacher while a sticky intervention is active or one has turns left; a span's turn count advances only on
    main (not auxiliary, e.g. Terminus-2 summarization), non-retry requests the teacher answers. A reply-time rewrite
    counts as the span's turn 1 (the teacher answers that same request).
  * a reply-time `next` is pending until the next main request, where it starts (turn 1 = that request) unless the
    trigger's optional pending_applies(ev, decision) says no (dropped). A retry drops it: the reply it was decided on
    never reached the harness, and the retry's own reply is judged again.
  * a span that has had all its turns hands back at the next main request. Limits, checked when a decision is made:
    earliest_turn (no decision acts before this turn), max_interventions (interventions started), cooldown (student
    turns between the last intervention's last teacher turn and the new one's first; 0 = none). Episode values come
    from Policy and count every intervention; a trigger may override any of the three, and an override counts only
    that trigger's own interventions (today's assist gap/max count assists only, not repairs).
  * retry (the router's is_retry: the same request again): a request the teacher answered is answered by the teacher
    again under the same intervention, not counted again, same note (memo). A request the student answered is the
    student's again and request-time triggers are not asked again, except triggers with on_retry: observe (today's
    context/time budgets); a rewrite whose trigger has on_retry: observe and whose span ended on that request is
    re-decided instead: the span ends and the student is asked again (today's parse repair).
  * several fires on one event: config order is priority; the first that may act acts, every other fire is logged
    (lower_priority, or its own suppression reason). log_only fires are logged as shadow, never acted on.
  * a trigger is asked only on the owners it lists (default: the student's turns); fires on the teacher's turns are
    logged (owner) and never act. observe: armed asks a trigger only when a decision from it could act now (skips
    cooldown/cap/earliest, a decision already made on this event, the teacher's turns): today's short-circuit order,
    no judge calls wasted; always (default) asks it on every event it can see, so every fire is logged.

Config (load_config(path_or_dict) -> (Policy, [TriggerSpec]); Controller.from_config):
    policy: {cooldown: 0, max_interventions: null, earliest_turn: 2, show_rejected_reply: false,
             student_view_after_handback: capped, failed_turn_view: keep}
    triggers:                        # list order = priority
      - use: loop                    # a built-in (BUILTINS), `package.module:Class`, or `path/file.py:Class`
        name: loop_strict            # optional, unique; default the trigger's own name
        params: {loop_k: 4}          # constructor kwargs
        where: rewrite | next        # reply-time decisions only; a request-only trigger with rewrite is an error
        span: 2                      # k >= 1 or -1
        phases: [request]            # narrow (or, if the trigger allows it, change) when it is asked
        log_only: false
        teacher_note: "..."          # in the teacher's view on the intervention's first turn only
        policy: {cooldown: 2, max_interventions: 6, earliest_turn: 1, observe: armed, on_retry: memo,
                 owners: [student]}

Today's behavior -> the legacy config stage 2 builds from the router's flags (Policy defaults: cooldown 0, no cap,
earliest_turn 2; entries in config order, which is today's evaluation order):
  request time (always next)
    repair_confirm  Terminus-2: the teacher's one-turn repair claimed done, so it answers the confirmation too.
                    span 1; earliest 1, on_retry observe (today a retry of that confirmation goes to the student).
    edit_failed, loop, no_progress_wait, error_streak
                    --takeover environment triggers. span -1; observe armed; scanner order, the first fire wins.
    context_budget  --context-budget-tokens. span -1; earliest --context-budget-min-turn, on_retry observe, armed.
    time_budget     msa --time-takeover-frac. span -1; policy as context_budget.
    assist          msa --assist. span 1; cooldown --assist-gap - 1, max --assist-max, earliest 2, observe armed. One
                    trigger for its four kinds, so the gap and the cap stay shared and repairs do not count.
  reply time
    parse_repair    rewrite, span 1; earliest 1, on_retry observe. Terminus-2: on whenever its parser is loaded
                    (--repair-on-parse-error or --teacher-format-guard); msa: --repair-on-parse-error. The router's
                    autofix stays in front of it.
    done_claim      msa: rewrite, span -1; params harness msa, teacher_note VERIFY_NOTE_MSA (--verify-note), earliest
                    1. Listed before the other decision triggers.
    gave_up, destructive, ... (--takeover decision triggers)
                    rewrite, span -1; observe armed; dec_fires order.
    done_claim      Terminus-2: next, span -1; params confirm_kind confirm (any other next request drops it: today's
                    done_claim_dropped), teacher_note VERIFY_NOTE (--verify-note), earliest 1. Listed last.
  take_over         sticky, at most once: any span -1 decision (a sticky span never hands back); ctl.sticky is today's
                    ep.owner == 'teacher'.
  --mode teacher / student
                    Controller(initial_owner=...), every trigger log_only with owners [student, teacher]; the shadow
                    records are would_fire.
Not routing, so they stay in the router: endings (deadline, budgets, context hard end), autofix, the teacher format
guard, LOGGED signals (rec['logged']), and --mode teacher's verify note on the first confirmation.
"""
import dataclasses
import functools
import importlib
import importlib.util
import inspect
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Protocol, runtime_checkable

PHASES = ('request', 'reply')
WHERES = ('rewrite', 'next')
OWNERS = ('student', 'teacher')
LIMIT_KEYS = ('cooldown', 'max_interventions', 'earliest_turn')
OBSERVE = ('always', 'armed')
ON_RETRY = ('memo', 'observe')
SUPPRESSED = ('cooldown', 'cap', 'earliest', 'owner', 'lower_priority', 'dropped')


class ConfigError(ValueError):
    """A trigger config (YAML or TriggerSpec) that cannot run as written."""


# ==== what a trigger sees and says ===========================================================================
@dataclass(frozen=True)
class ActionResult:
    """The previous action and what it returned, normalized per harness so a rule is harness-agnostic. Terminus-2: the
    keystrokes and the new terminal output (no exit code: the screen does not carry one); MSA: the bash command, its
    returncode and output. timed_out: the harness's command timeout; truncated: the harness cut the output."""
    command: str = ''
    exit_code: int | None = None
    timed_out: bool = False
    output: str = ''
    truncated: bool = False


@dataclass(frozen=True)
class Reply:
    """The student's candidate reply (reply phase). action: the parsed action as the harness will run it
    (relay_triggers.parse_reply / tool_reply dict); None when the harness rejects the reply, and then parse_error holds
    the harness parser's message (set by the router after its autofix)."""
    text: str = ''
    action: Any = None
    parse_error: str | None = None


@dataclass(frozen=True)
class Clock:
    """The owner's agent clock: seconds used (model calls paused as the router pauses them) and the task's budget."""
    used_s: float = 0.0
    budget_s: float | None = None


@dataclass(frozen=True)
class Intervention:
    """One teacher intervention. turns_done counts the main, non-retry requests the teacher answered under it (a rewrite
    is turn 1); fired_turn is the turn the decision was made (a reply-time `next` fires one turn before it starts)."""
    id: int
    trigger: str
    where: str
    span: int
    start_turn: int
    turns_done: int
    reason: str = ''
    fired_turn: int | None = None
    last_turn: int | None = None          # the latest turn the teacher answered under it
    teacher_note: str | None = None
    score: float | None = None
    handed_back: bool = False             # the student answered again after it (never for span -1)

    @property
    def sticky(self):
        return self.span == -1

    @property
    def turns_left(self):
        return None if self.sticky else max(0, self.span - self.turns_done)

    def as_dict(self):
        return dataclasses.asdict(self)


async def _no_count():
    return None


@dataclass(frozen=True)
class Event:
    """What a trigger sees; built only from the past (causal). phase 'request': before the student is asked, with the
    last action's result; 'reply': the student answered and its action has not run. main=False marks an auxiliary
    request (Terminus-2 summarization): no turn, routed by the owner, no trigger asked. The Controller fills `owner`
    (who answers this request) and `interventions` (the episode's so far) before a trigger sees the event."""
    phase: Literal['request', 'reply']
    turn: int
    is_retry: bool = False
    owner: str = 'student'
    messages: Any = ()                    # the harness's history, as the student sees it
    last_result: ActionResult | None = None
    reply: Reply | None = None
    scanner: Any = None                   # relay_triggers.EpisodeScanner (loops, waits, error streaks, ...)
    clock: Clock = Clock()
    view_tokens: Callable[[], Awaitable[int | None]] = _no_count   # lazy /tokenize count of the student's view
    interventions: tuple = ()
    extra: dict = field(default_factory=dict)   # harness-specific: request_kind, task_id, harness, ...
    main: bool = True


@dataclass(frozen=True)
class Decision:
    """A trigger's verdict. teacher_note is shown to the teacher only (never enters the harness history); log_only
    decisions are logged as shadow and never acted on."""
    trigger: str
    reason: str
    score: float | None = None
    where: Literal['rewrite', 'next'] = 'next'
    span: int = -1
    teacher_note: str | None = None
    log_only: bool = False

    def __post_init__(self):
        if self.where not in WHERES:
            raise ValueError(f'{self.trigger}: where must be one of {WHERES}, not {self.where!r}')
        if not _span_ok(self.span):
            raise ValueError(f'{self.trigger}: span must be k >= 1 or -1, not {self.span!r}')


def _span_ok(span):
    return type(span) is int and (span == -1 or span >= 1)


@runtime_checkable
class Trigger(Protocol):
    """name, phases ({'request'}, {'reply'} or both) and observe(ev). Optional: reset() (fresh episode),
    pending_applies(ev, decision) -> bool (a reply-time `next`: does it still apply at this request?),
    on_outcome(record) (what became of each of its fires), allowed_phases (the phases a config may choose, when wider
    than the default `phases`)."""
    name: str
    phases: frozenset

    async def observe(self, ev: Event) -> Decision | None: ...


# ==== policy and per-trigger settings ========================================================================
@dataclass(frozen=True)
class Policy:
    """Episode policy. The defaults reproduce today's router: no cooldown (assists bring their own), no cap (a sticky
    takeover cannot recur anyway), no decision before turn 2 (done_claim and parse_repair override to 1). The last
    three are view knobs the router reads; the Controller only carries show_rejected_reply on rewrites."""
    cooldown: int = 0
    max_interventions: int | None = None
    earliest_turn: int = 2
    show_rejected_reply: bool = False     # rewrite: the teacher sees the discarded student reply
    student_view_after_handback: str = 'capped'   # teacher reasoning in the student's view: full | capped | stripped
    failed_turn_view: str = 'keep'        # next: the failed action and its output in the teacher's view: keep | hide

    def __post_init__(self):
        _check_limits(self.cooldown, self.max_interventions, self.earliest_turn, 'policy')
        if not isinstance(self.show_rejected_reply, bool):
            raise ConfigError('policy: show_rejected_reply must be true or false')
        if self.student_view_after_handback not in ('full', 'capped', 'stripped'):
            raise ConfigError('policy: student_view_after_handback must be full, capped or stripped')
        if self.failed_turn_view not in ('keep', 'hide'):
            raise ConfigError('policy: failed_turn_view must be keep or hide')


def _check_limits(cooldown, cap, earliest, where):
    if type(cooldown) is not int or cooldown < 0:
        raise ConfigError(f'{where}: cooldown must be an int >= 0, not {cooldown!r}')
    if cap is not None and (type(cap) is not int or cap < 0):
        raise ConfigError(f'{where}: max_interventions must be an int >= 0 or null, not {cap!r}')
    if type(earliest) is not int or earliest < 1:
        raise ConfigError(f'{where}: earliest_turn must be an int >= 1, not {earliest!r}')


@dataclass(frozen=True)
class TriggerSpec:
    """A trigger as configured. where/span/teacher_note/log_only override its decisions; limits holds the per-trigger
    overrides of Policy's cooldown / max_interventions / earliest_turn (counted over this trigger's interventions).
    make: builds a fresh trigger (the loader sets it), so each Controller (episode) gets its own state."""
    trigger: Any
    name: str | None = None
    phases: frozenset | None = None
    where: str | None = None
    span: int | None = None
    log_only: bool = False
    teacher_note: str | None = None
    limits: dict = field(default_factory=dict)
    observe: str = 'always'
    on_retry: str = 'memo'
    owners: frozenset = frozenset({'student'})
    make: Callable[[], Any] | None = None

    def __post_init__(self):
        t = self.trigger
        if not callable(getattr(t, 'observe', None)):
            raise ConfigError(f'{type(t).__name__}: a trigger needs an async observe(ev) method')
        name = self.name or getattr(t, 'name', None) or type(t).__name__
        object.__setattr__(self, 'name', name)
        own = frozenset(getattr(t, 'phases', None) or ())
        phases = frozenset(self.phases) if self.phases is not None else own
        allowed = frozenset(getattr(t, 'allowed_phases', None) or own)
        if not phases or not phases <= set(PHASES):
            raise ConfigError(f'{name}: phases must be a non-empty subset of {PHASES}, not {sorted(phases)}')
        if not phases <= allowed:
            raise ConfigError(f'{name}: runs at {sorted(allowed)} only, not {sorted(phases - allowed)}')
        object.__setattr__(self, 'phases', phases)
        if self.where is not None and self.where not in WHERES:
            raise ConfigError(f'{name}: where must be one of {WHERES}, not {self.where!r}')
        if self.where == 'rewrite' and 'reply' not in phases:
            raise ConfigError(f'{name}: where: rewrite needs the reply phase; a request-time trigger sees a fault only '
                              'after the action ran, so it always acts as next')
        if self.span is not None and not _span_ok(self.span):
            raise ConfigError(f'{name}: span must be k >= 1 or -1, not {self.span!r}')
        bad = set(self.limits) - set(LIMIT_KEYS)
        if bad:
            raise ConfigError(f'{name}: unknown limit(s) {sorted(bad)}; known {LIMIT_KEYS}')
        lim = self.limits
        _check_limits(lim.get('cooldown', 0), lim.get('max_interventions'), lim.get('earliest_turn', 1), name)
        if self.observe not in OBSERVE:
            raise ConfigError(f'{name}: observe must be one of {OBSERVE}, not {self.observe!r}')
        if self.on_retry not in ON_RETRY:
            raise ConfigError(f'{name}: on_retry must be one of {ON_RETRY}, not {self.on_retry!r}')
        owners = frozenset(self.owners)
        if not owners or not owners <= set(OWNERS):
            raise ConfigError(f'{name}: owners must be a non-empty subset of {OWNERS}, not {sorted(owners)}')
        object.__setattr__(self, 'owners', owners)

    def fresh(self):
        """This spec with its own trigger state: a new instance when it can be rebuilt, else the instance reset."""
        if self.make is not None:
            return dataclasses.replace(self, trigger=self.make())
        if callable(getattr(self.trigger, 'reset', None)):
            self.trigger.reset()
        return self


# ==== what the Controller says ===============================================================================
@dataclass(frozen=True)
class DecisionRecord:
    """One fire and what became of it. outcome: acted (an intervention started; its id), pending (a reply-time next
    waiting for the next request), suppressed (why: cooldown | cap | earliest | owner | lower_priority | dropped) or
    shadow (log_only)."""
    decision: Decision
    phase: str
    turn: int
    outcome: str
    why: str | None = None
    intervention: int | None = None

    def as_dict(self):
        d = self.decision
        return dict(trigger=d.trigger, phase=self.phase, turn=self.turn, outcome=self.outcome, why=self.why,
                    reason=d.reason, score=d.score, where=d.where, span=d.span, intervention=self.intervention)


@dataclass(frozen=True)
class Routing:
    """Who answers this request and why: student (the student owns the episode), owner (the teacher owns it from the
    start, --mode teacher), start (an intervention starts at this request: a request-time decision or a pending next),
    span (a later turn of an active intervention, or an auxiliary request during one), rewrite (the student's reply is
    discarded and the teacher answers the same request), retry (the memoized teacher routing of this request)."""
    who: str
    why: str
    intervention: Intervention | None = None
    teacher_note: str | None = None       # only on an intervention's first turn (and its retries)
    show_rejected_reply: bool = False     # rewrite only (Policy.show_rejected_reply)
    records: tuple = ()


# ==== the controller =========================================================================================
class Controller:
    """Per-episode policy state machine; see the module docstring for the semantics. One per episode: each gets fresh
    trigger state (TriggerSpec.fresh)."""

    def __init__(self, policy=None, triggers=(), initial_owner='student'):
        if initial_owner not in OWNERS:
            raise ConfigError(f'initial_owner must be one of {OWNERS}, not {initial_owner!r}')
        self.policy = policy or Policy()
        specs = [t if isinstance(t, TriggerSpec) else TriggerSpec(t) for t in triggers]
        names = [s.name for s in specs]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise ConfigError(f'trigger names must be unique (set name:), repeated: {dup}')
        self.specs = [s.fresh() for s in specs]
        self._spec = {s.name: s for s in self.specs}
        self.initial_owner = initial_owner
        self.observes_teacher = any('teacher' in s.owners for s in self.specs)   # the router sends teacher replies
        self._init_state()

    @classmethod
    def from_config(cls, src, **kw):
        policy, specs = load_config(src)
        return cls(policy, specs, **kw)

    def _init_state(self):
        self.interventions = []           # every intervention of the episode, oldest first (frozen; replaced on update)
        self.records = []                 # every DecisionRecord of the episode
        self.pending = None               # (spec, Decision, fired_turn): a reply-time next for the next main request
        self._active_id = None
        self._memo = None                 # the routing of the last main request (a retry gets it again)
        self._cur = Routing(self.initial_owner, 'owner' if self.initial_owner == 'teacher' else 'student')
        self._begun = None                # the Event begin_request ran for (on_request runs it when the router did not)
        self._recs = []                   # records of the request being routed

    def reset(self):
        """Start a new episode with the same config (fresh trigger state)."""
        self.specs = [s.fresh() for s in self.specs]
        self._spec = {s.name: s for s in self.specs}
        self._init_state()

    # -- state ---------------------------------------------------------------------------------------------------
    @property
    def active(self):
        return self.interventions[self._active_id - 1] if self._active_id else None

    @property
    def owner(self):
        """'teacher' while sticky or while the active span has turns left; the router routes /tokenize by this."""
        if self.initial_owner == 'teacher':
            return 'teacher'
        I = self.active
        return 'teacher' if I is not None and (I.sticky or I.turns_left > 0) else 'student'

    @property
    def sticky(self):
        """The teacher keeps the episode to the end: today's ep.owner == 'teacher' (budgets, the context hard end)."""
        I = self.active
        return self.initial_owner == 'teacher' or (I is not None and I.sticky)

    def summary(self):
        return dict(owner=self.owner, sticky=self.sticky, interventions=[I.as_dict() for I in self.interventions],
                    pending=self.pending[1].trigger if self.pending else None, fires=len(self.records))

    # -- request phase -------------------------------------------------------------------------------------------
    async def begin_request(self, ev):
        """Everything decided before any trigger is asked: an auxiliary request's owner, the retry memo, the hand-back
        of a span that has had its turns, the next turn of an active span, a pending next. Pass the same Event object
        to on_request afterwards."""
        self._begun, self._recs = ev, []
        if self.initial_owner == 'teacher':
            return self._route(Routing('teacher', 'owner'))
        I = self.active
        if not ev.main:
            return self._route(Routing('teacher', 'span', I) if self.owner == 'teacher'
                               else Routing('student', 'student'))
        if ev.is_retry and self._memo is not None:
            if self.pending is not None:
                spec, d, _ = self.pending
                self.pending = None
                self._log(spec, DecisionRecord(d, 'request', ev.turn, 'suppressed', 'dropped'))
            m = self._memo
            J = self.interventions[m.intervention.id - 1] if m.who == 'teacher' and m.intervention else None
            if J is not None:
                done = not J.sticky and J.turns_done >= J.span
                if not (done and J.last_turn == ev.turn and self._spec[J.trigger].on_retry == 'observe'):
                    return self._route(dataclasses.replace(m, why='retry', intervention=J, records=()))
                self._end(J)              # re-decided: its span ends on this request and the student is asked again
            return self._route(Routing('student', 'student'))
        if I is not None and not I.sticky and I.turns_left == 0:
            self._end(I)                  # hand-back: the span has had its turns, the student resumes
            I = None
        if I is not None:
            return self._route(Routing('teacher', 'span', self._turn(I, ev.turn)))
        if self.pending is not None:
            spec, d, fired = self.pending
            self.pending = None
            f = getattr(spec.trigger, 'pending_applies', None)
            ok = True if f is None else f(self._view(ev, 'student'), d)
            if inspect.isawaitable(ok):
                ok = await ok
            if ok:
                I = self._start(spec, d, ev.turn, fired)
                self._log(spec, DecisionRecord(d, 'request', ev.turn, 'acted', intervention=I.id))
                return self._route(Routing('teacher', 'start', I, d.teacher_note))
            self._log(spec, DecisionRecord(d, 'request', ev.turn, 'suppressed', 'dropped'))
        return self._route(Routing('student', 'student'))

    async def on_request(self, ev):
        """Request-time triggers, after begin_request. Returns the request's routing with every record it produced."""
        if self._begun is not ev:
            await self.begin_request(ev)
        self._begun = None
        r = self._cur
        if ev.main and r.why != 'retry':
            fires, win = await self._arbitrate(ev, 'request')
            I = None
            if win is not None:
                spec, d = win
                I = self._start(spec, d, ev.turn, ev.turn)
                r = self._route(Routing('teacher', 'start', I, d.teacher_note))
            self._log_fires(fires, 'request', ev.turn, I, self._recs)
            self._memo = r
        return dataclasses.replace(r, records=tuple(self._recs))

    # -- reply phase ---------------------------------------------------------------------------------------------
    async def on_reply(self, ev):
        """Reply-time triggers on the student's candidate reply. A rewrite routes the same request to the teacher
        (Routing.who == 'teacher'); a next is kept pending and the reply goes through. When the request's answerer is
        the teacher (the router sends the teacher's reply only if ctl.observes_teacher), fires are logged, never acted
        on."""
        r = self._cur
        if not ev.main:
            return dataclasses.replace(r, records=())
        fires, win = await self._arbitrate(ev, 'reply')
        recs, I, pending = [], None, False
        if win is not None:
            spec, d = win
            if d.where == 'rewrite':
                I = self._start(spec, d, ev.turn, ev.turn)
                r = self._route(Routing('teacher', 'rewrite', I, d.teacher_note, self.policy.show_rejected_reply))
                self._memo = r
            else:
                self.pending = (spec, d, ev.turn)   # applied (or dropped) by the next main request's begin_request
                pending = True
        self._log_fires(fires, 'reply', ev.turn, I, recs, pending)
        return dataclasses.replace(r, records=tuple(recs))

    # -- internals -----------------------------------------------------------------------------------------------
    async def _arbitrate(self, ev, phase):
        """Ask the triggers in config order. Returns ([(spec, decision, why)], winner): why is None for the winner, else
        the suppression reason or 'shadow'."""
        who = self._cur.who
        tev = self._view(ev, who)
        fires, win = [], None
        for spec in self.specs:
            if phase not in spec.phases or who not in spec.owners:
                continue
            if phase == 'request' and ev.is_retry and spec.on_retry != 'observe':
                continue
            if spec.observe == 'armed' and (who != 'student' or win is not None
                                            or self._blocked(spec, phase, spec.where or 'next', ev.turn)):
                continue                  # it could not act now: not asked (no judge call, no state consumed)
            d = await spec.trigger.observe(tev)
            if d is None:
                continue
            d = self._normalize(spec, d, phase)
            why = ('shadow' if d.log_only else 'owner' if who != 'student' else 'lower_priority' if win is not None
                   else self._blocked(spec, phase, d.where, ev.turn))
            if why is None:
                win = (spec, d)
            fires.append((spec, d, why))
        return fires, win

    def _blocked(self, spec, phase, where, turn):
        """Why a decision of this trigger at this turn may not act (earliest | cap | cooldown), or None. A per-trigger
        override counts the trigger's own interventions, the episode policy all of them."""
        lim, pol = spec.limits, self.policy
        if turn < lim.get('earliest_turn', pol.earliest_turn):
            return 'earliest'
        own = [I for I in self.interventions if I.trigger == spec.name]
        cap = lim.get('max_interventions', pol.max_interventions)
        if cap is not None and len(own if 'max_interventions' in lim else self.interventions) >= cap:
            return 'cap'
        cd = lim.get('cooldown', pol.cooldown)
        pool = own if 'cooldown' in lim else self.interventions
        first = turn + (phase == 'reply' and where == 'next')   # the teacher's first turn under the new intervention
        if cd and pool and first - max(I.last_turn for I in pool) - 1 < cd:
            return 'cooldown'
        return None

    def _normalize(self, spec, d, phase):
        if not isinstance(d, Decision):
            raise TypeError(f'{spec.name}: observe() must return a Decision or None, not {type(d).__name__}')
        kw = dict(trigger=spec.name)
        if spec.span is not None:
            kw['span'] = spec.span
        if spec.teacher_note is not None:
            kw['teacher_note'] = spec.teacher_note
        if spec.log_only:
            kw['log_only'] = True
        if phase == 'reply' and spec.where is not None:
            kw['where'] = spec.where
        d = dataclasses.replace(d, **kw)
        if phase == 'request' and d.where == 'rewrite':
            raise ValueError(f'{spec.name}: a request-time decision cannot rewrite (the action already ran); it acts '
                             'as next')
        return d

    def _log_fires(self, fires, phase, turn, I, out, pending=False):
        for spec, d, why in fires:
            if why is None:
                rec = DecisionRecord(d, phase, turn, 'pending' if pending else 'acted',
                                     intervention=None if pending else I.id)
            elif why == 'shadow':
                rec = DecisionRecord(d, phase, turn, 'shadow')
            else:
                rec = DecisionRecord(d, phase, turn, 'suppressed', why)
            self._log(spec, rec, out)

    def _log(self, spec, rec, out=None):
        self.records.append(rec)
        (self._recs if out is None else out).append(rec)
        f = getattr(spec.trigger, 'on_outcome', None)
        if f is not None:
            f(rec)

    def _view(self, ev, who):
        return dataclasses.replace(ev, owner=who, interventions=tuple(self.interventions))

    def _route(self, r):
        self._cur = r
        return r

    def _start(self, spec, d, turn, fired):
        I = Intervention(id=len(self.interventions) + 1, trigger=spec.name, where=d.where, span=d.span,
                         start_turn=turn, turns_done=1, reason=d.reason, fired_turn=fired, last_turn=turn,
                         teacher_note=d.teacher_note, score=d.score)
        self.interventions.append(I)
        self._active_id = I.id
        return I

    def _turn(self, I, turn):
        return self._put(dataclasses.replace(I, turns_done=I.turns_done + 1, last_turn=turn))

    def _end(self, I):
        self._put(dataclasses.replace(I, handed_back=True))
        self._active_id = None

    def _put(self, I):
        self.interventions[I.id - 1] = I
        return I


# ==== config =================================================================================================
BUILTINS = {}
TOP_KEYS = ('policy', 'triggers')
ENTRY_KEYS = ('use', 'name', 'params', 'where', 'span', 'phases', 'log_only', 'teacher_note', 'policy')
TRIGGER_POLICY_KEYS = LIMIT_KEYS + ('observe', 'on_retry', 'owners')


def builtin(cls):
    BUILTINS[cls.name] = cls
    return cls


def load_config(src):
    """A trigger config (a YAML path or an already-parsed dict) -> (Policy, [TriggerSpec]). Raises ConfigError naming
    the entry and the key at fault."""
    base = None
    if isinstance(src, (str, os.PathLike)):
        import yaml
        base = os.path.dirname(os.path.abspath(src))
        with open(src) as f:
            src = yaml.safe_load(f) or {}
    if not isinstance(src, dict):
        raise ConfigError(f'a trigger config is a mapping with keys {TOP_KEYS}, not {type(src).__name__}')
    bad = set(src) - set(TOP_KEYS)
    if bad:
        raise ConfigError(f'unknown top-level key(s) {sorted(bad)}; known {TOP_KEYS}')
    pol = src.get('policy') or {}
    if not isinstance(pol, dict):
        raise ConfigError('policy must be a mapping')
    bad = set(pol) - {f.name for f in dataclasses.fields(Policy)}
    if bad:
        raise ConfigError(f'policy: unknown key(s) {sorted(bad)}; known {[f.name for f in dataclasses.fields(Policy)]}')
    policy = Policy(**pol)
    entries = src.get('triggers') or []
    if not isinstance(entries, list):
        raise ConfigError('triggers must be a list')
    specs = [_entry(e, i, base) for i, e in enumerate(entries)]
    names = [s.name for s in specs]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ConfigError(f'trigger names must be unique (set name:), repeated: {dup}')
    return policy, specs


def _entry(e, i, base):
    at = f'triggers[{i}]'
    if not isinstance(e, dict):
        raise ConfigError(f'{at}: an entry is a mapping with at least `use:`')
    bad = set(e) - set(ENTRY_KEYS)
    if bad:
        raise ConfigError(f'{at}: unknown key(s) {sorted(bad)}; known {ENTRY_KEYS}')
    use = e.get('use')
    if not isinstance(use, str) or not use:
        raise ConfigError(f'{at}: `use:` names a built-in ({sorted(BUILTINS)}) or module:Class')
    at = f'{at} ({use})'
    params = e.get('params') or {}
    pol = e.get('policy') or {}
    if not isinstance(params, dict) or not isinstance(pol, dict):
        raise ConfigError(f'{at}: params and policy are mappings')
    bad = set(pol) - set(TRIGGER_POLICY_KEYS)
    if bad:
        raise ConfigError(f'{at}: unknown policy key(s) {sorted(bad)}; known {TRIGGER_POLICY_KEYS}')
    factory = _resolve(use, base, at)
    try:
        make = functools.partial(factory, **params)
        trig = make()
    except ConfigError as err:
        raise ConfigError(f'{at}: {err}') from None
    except TypeError as err:
        raise ConfigError(f'{at}: bad params: {err}') from None
    phases = e.get('phases')
    if phases is not None and not isinstance(phases, (list, tuple)):
        raise ConfigError(f'{at}: phases is a list')
    try:
        return TriggerSpec(trig, name=e.get('name'), phases=None if phases is None else frozenset(phases),
                           where=e.get('where'), span=e.get('span'), log_only=bool(e.get('log_only', False)),
                           teacher_note=e.get('teacher_note'), limits={k: pol[k] for k in LIMIT_KEYS if k in pol},
                           observe=pol.get('observe', 'always'), on_retry=pol.get('on_retry', 'memo'),
                           owners=frozenset(pol.get('owners', ('student',))), make=make)
    except ConfigError as err:
        raise ConfigError(f'{at}: {err}') from None


def _resolve(use, base, at):
    """A built-in name, `package.module:Class` (importlib, on sys.path) or `path/to/file.py:Class` (relative to the
    config file) -> the class (or factory)."""
    if ':' not in use:
        if use not in BUILTINS:
            raise ConfigError(f'{at}: no built-in trigger {use!r}; built-ins {sorted(BUILTINS)}, or use module:Class')
        return BUILTINS[use]
    mod_name, _, attr = use.rpartition(':')
    try:
        if mod_name.endswith('.py'):
            path = mod_name if os.path.isabs(mod_name) or base is None else os.path.join(base, mod_name)
            name = 'relay_trigger_plugin_' + re.sub(r'\W', '_', os.path.splitext(os.path.basename(path))[0])
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod       # dataclasses in the plugin look their module up there
            spec.loader.exec_module(mod)
        else:
            mod = importlib.import_module(mod_name)
    except Exception as err:  # noqa: BLE001 - any import failure is a config error the user must see
        raise ConfigError(f'{at}: cannot import {mod_name!r}: {err!r}') from None
    if not hasattr(mod, attr):
        raise ConfigError(f'{at}: {mod_name!r} has no {attr!r}')
    return getattr(mod, attr)


# ==== built-in triggers ======================================================================================
def _rt():
    """relay_triggers (data/relay/triggers), imported on first use as relay_router.py imports it."""
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'triggers')
    if d not in sys.path:
        sys.path.insert(0, d)
    import relay_triggers
    return relay_triggers


def _judged(ev):
    """The reply as the scanner judges it: the parsed action, else the raw text; None when the harness rejects it
    (parse_repair's case: today the router never judges such a reply)."""
    r = ev.reply
    if r is None:
        return None
    if r.action is not None:
        return r.action
    return None if r.parse_error else r.text


@builtin
class ActionError:
    """The previous action failed `streak` times in a row (request phase, on Event.last_result). A result counts as
    failed when the command timed out (timeouts), its exit code is in exit_codes ('nonzero': any but 0; a list: those
    codes; null: codes ignored; Terminus-2 results carry none), or its output matches one of `patterns` (Python regexes,
    e.g. 'Traceback', 'command not found', 'No such file'). A command matching one of ignore_commands never counts
    (e.g. '^\\s*(grep|diff|test)\\b', whose exit 1 means no match). A request without a result (a parse-error re-prompt)
    leaves the streak as it is; turns the trigger did not see (the teacher's) restart it."""
    name = 'action_error'
    phases = frozenset({'request'})

    def __init__(self, exit_codes='nonzero', timeouts=True, patterns=(), streak=1, ignore_commands=()):
        if not (exit_codes is None or exit_codes == 'nonzero' or
                (isinstance(exit_codes, (list, tuple)) and all(type(c) is int for c in exit_codes))):
            raise ConfigError(f"exit_codes must be 'nonzero', a list of ints or null, not {exit_codes!r}")
        if type(streak) is not int or streak < 1:
            raise ConfigError(f'streak must be an int >= 1, not {streak!r}')
        try:
            self.patterns = [re.compile(p) for p in patterns]
            self.ignore = [re.compile(p) for p in ignore_commands]
        except (re.error, TypeError) as err:
            raise ConfigError(f'bad regex: {err}') from None
        self.exit_codes = exit_codes if exit_codes in (None, 'nonzero') else frozenset(exit_codes)
        self.timeouts, self.streak = bool(timeouts), streak
        self.reset()

    def reset(self):
        self._fails, self._last_turn = [], 0

    def failure(self, r):
        """Why one result counts as failed, or None."""
        if any(p.search(r.command or '') for p in self.ignore):
            return None
        if self.timeouts and r.timed_out:
            return 'timed out'
        c = r.exit_code
        if c is not None and self.exit_codes is not None and (c != 0 if self.exit_codes == 'nonzero'
                                                              else c in self.exit_codes):
            return f'exit code {c}'
        for p in self.patterns:
            if p.search(r.output or ''):
                return f'output matches /{p.pattern}/'
        return None

    async def observe(self, ev):
        if ev.turn > self._last_turn:     # each turn's result counts once (a retry re-reads the same one)
            if ev.turn != self._last_turn + 1:
                self._fails = []
            self._last_turn = ev.turn
            if ev.last_result is not None:
                why = self.failure(ev.last_result)
                self._fails = self._fails + [why] if why else []
        n = len(self._fails)
        if n < self.streak:
            return None
        reason = self._fails[-1] if self.streak == 1 else f'{n} failed actions in a row, last: {self._fails[-1]}'
        return Decision(self.name, reason, score=float(n))


@builtin
class Scripted:
    """Fire at fixed turns: `turns` (every episode) and/or `plan` ({task_id: [turns]}, matched on
    ev.extra['task_id']). For controlled experiments and replay. Request time by default (the teacher answers that
    turn); phases: [reply] judges the student's reply at those turns instead (with where: rewrite it is discarded)."""
    name = 'scripted'
    phases = frozenset({'request'})
    allowed_phases = frozenset(PHASES)

    def __init__(self, turns=(), plan=None, reason=None):
        self.turns = frozenset(turns)
        self.plan = {k: frozenset(v) for k, v in (plan or {}).items()}
        self.reason = reason

    async def observe(self, ev):
        if ev.turn in self.turns or ev.turn in self.plan.get(ev.extra.get('task_id'), ()):
            return Decision(self.name, self.reason or f'scripted at turn {ev.turn}')
        return None


class _ScannerTrigger:
    """Legacy adapter: one relay_triggers detector, as the router runs it today (EpisodeScanner.current_fires on
    CALIBRATED_CONFIG with only this trigger enabled; params override that config's keys, e.g. loop_k)."""
    name = kind = None

    def __init__(self, **params):
        rt = _rt()
        bad = set(params) - (set(rt.CALIBRATED_CONFIG) - {'enabled'})
        if bad:
            raise ConfigError(f'unknown param(s) {sorted(bad)}; known: the keys of relay_triggers.CALIBRATED_CONFIG')
        self.cfg = dict(rt.CALIBRATED_CONFIG, **params, enabled=[self.name])

    async def observe(self, ev):
        if ev.scanner is None:
            return None
        view = None
        if ev.phase == 'reply':
            view = _judged(ev)
            if view is None:
                return None
        for f in ev.scanner.current_fires(view, self.cfg):
            if f['trigger'] == self.name and f['kind'] == self.kind:
                return Decision(self.name, f.get('reason') or self.name)
        return None


def _scanner_builtin(name, phase):
    kind = 'environment' if phase == 'request' else 'decision'
    BUILTINS[name] = type(name, (_ScannerTrigger,), dict(
        name=name, kind=kind, phases=frozenset({phase}),
        __doc__=f'relay_triggers {kind} trigger {name!r} ({phase} phase), as the router runs it today.'))


# scanner order (env_fires, dec_fires): the legacy config lists them in this order, so the first fire wins as today
for _n in ('edit_failed', 'loop', 'no_progress_wait', 'error_streak'):
    _scanner_builtin(_n, 'request')
for _n in ('submit_check', 'gave_up', 'success_claim_unchecked', 'success_contradicted', 'destructive'):
    _scanner_builtin(_n, 'reply')


@builtin
class DoneClaim:
    """Legacy adapter: the student's reply claims the task is done (Terminus-2 task_complete: true; MSA the submit
    command). The scanner fires it on the episode's first claim only, so as the router does today a later claim is
    judged on the reply itself (a teacher repair turn may have claimed first). Terminus-2 runs it as next with
    confirm_kind 'confirm': the claim goes through and the teacher answers Terminus-2's "Are you sure?" request; any
    other next request (Terminus-2 rejected the claim) drops it. MSA runs it as rewrite (the submit never runs)."""
    name = 'done_claim'
    phases = frozenset({'reply'})

    def __init__(self, harness='terminus2', confirm_kind=None):
        if harness not in ('terminus2', 'msa'):
            raise ConfigError(f'harness must be terminus2 or msa, not {harness!r}')
        self.harness, self.confirm_kind = harness, confirm_kind

    async def observe(self, ev):
        view = _judged(ev)
        if view is None or (self.harness == 'msa' and not isinstance(view, dict)):
            return None
        rt = _rt()
        if ev.scanner is not None:
            cfg = dict(rt.CALIBRATED_CONFIG, enabled=['done_claim'])
            f = next((f for f in ev.scanner.current_fires(view, cfg) if f['trigger'] == 'done_claim'), None)
            if f is not None:
                return Decision(self.name, f['reason'])
        if rt.parse_reply(view, 'terminus2' if self.harness == 'terminus2' else None)['done']:
            return Decision(self.name, 'task_complete: true (student)' if self.harness == 'terminus2'
                            else 'submit command (student)')
        return None

    def pending_applies(self, ev, d):
        return self.confirm_kind is None or ev.extra.get('request_kind') == self.confirm_kind


@builtin
class ParseRepair:
    """Legacy adapter: the harness's own parser rejects the student's reply (Reply.parse_error, set after the router's
    autofix). Today: rewrite, one turn, the student keeps the episode; a retry asks the student again (on_retry:
    observe)."""
    name = 'parse_repair'
    phases = frozenset({'reply'})

    async def observe(self, ev):
        if ev.reply is not None and ev.reply.parse_error:
            return Decision(self.name, ev.reply.parse_error[:500], where='rewrite', span=1)
        return None


@builtin
class RepairConfirm:
    """Legacy adapter (Terminus-2): the teacher's one-turn repair claimed done, so Terminus-2 asks "Are you sure?"; the
    teacher answers that too, then the student resumes. Fires at a request of kind confirm_kind whose previous turn was
    a one-turn `after` intervention: Terminus-2 asks only after a reply its parser accepted with task_complete true,
    which is the router's repair_confirm condition today."""
    name = 'repair_confirm'
    phases = frozenset({'request'})

    def __init__(self, after='parse_repair', confirm_kind='confirm'):
        self.after, self.confirm_kind = after, confirm_kind

    async def observe(self, ev):
        if ev.extra.get('request_kind') != self.confirm_kind or not ev.interventions:
            return None
        last = ev.interventions[-1]
        if last.trigger == self.after and last.span == 1 and last.last_turn == ev.turn - 1:
            return Decision(self.name, f'the teacher claimed done in its {self.after} turn', span=1)
        return None


@builtin
class ContextBudget:
    """Legacy adapter (--context-budget-tokens): the student's own view of this request (Event.view_tokens, the
    router's /tokenize count) has reached `tokens`; sticky. relay_triggers.context_budget_fire."""
    name = 'context_budget'
    phases = frozenset({'request'})

    def __init__(self, tokens=32000, min_turn=2):
        self.tokens, self.min_turn = tokens, min_turn

    async def observe(self, ev):
        n = await ev.view_tokens()
        f = _rt().context_budget_fire(n, ev.turn, self.tokens, self.min_turn)
        return Decision(self.name, f['reason'], score=float(n)) if f else None


@builtin
class TimeBudget:
    """Legacy adapter (msa --time-takeover-frac): the student has used `frac` of the task's agent budget on its own
    clock (paused time excluded); sticky."""
    name = 'time_budget'
    phases = frozenset({'request'})

    def __init__(self, frac):
        self.frac = float(frac)

    async def observe(self, ev):
        used, b = ev.clock.used_s, ev.clock.budget_s
        if not b or used < self.frac * b:
            return None
        return Decision(self.name, f'student used {used:.0f} s of {b:.0f} s', score=used / b)


class _Stub:
    """A built-in name reserved for a later stage: configs may name it, loading one fails with where it lands."""
    stage = None

    def __init__(self, **params):
        raise ConfigError(f'built-in {self.name!r} has no body before stage {self.stage} of '
                          'notes/ot-agent/relay-triggers')


@builtin
class Assist(_Stub):
    """msa --assist (cmd_timeout, error_streak, same_output, ctx check-ins) moves from Router.assist_trigger in stage 2:
    request phase, span 1, observe armed, cooldown --assist-gap - 1, max --assist-max; a context milestone is used up
    only when the assist acts (on_outcome)."""
    name, phases, stage = 'assist', frozenset({'request'}), 2


@builtin
class LlmJudge(_Stub):
    """An LLM reads the recent turns and scores "stuck / about to fail" (stage 4)."""
    name, phases, stage = 'llm_judge', frozenset(PHASES), 4
