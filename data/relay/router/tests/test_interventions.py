"""CPU tests for interventions.py: the Controller's routing semantics as a scenario table, the config loader, and the
built-ins that need no router (action_error, scripted) plus the legacy adapters' wiring. Fake triggers, no network.

    python -m pytest -q data/relay/router/tests/test_interventions.py
"""
import asyncio
import collections
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import interventions as iv  # noqa: E402
from interventions import (ActionResult, Clock, ConfigError, Controller, Decision, Event, Policy, Reply,  # noqa: E402
                           TriggerSpec)


class Fake:
    """Fires in its phase at the listed turns (each listed turn once: list it twice to fire on a retry too), on every
    turn from `every`, or when `when(ev)` holds. Records every event it was asked about in `seen`."""

    def __init__(self, name='A', phase='reply', at=(), every=None, when=None, where='next', span=-1, note=None,
                 log_only=False):
        self.name, self.phases = name, frozenset({phase})
        self.left = collections.Counter(at)
        self.every, self.when = every, when
        self.kw = dict(where=where, span=span, teacher_note=note, log_only=log_only)
        self.seen = []
        self.outcomes = []

    async def observe(self, ev):
        self.seen.append((ev.phase, ev.turn, ev.is_retry, ev.owner))
        hit = self.left[ev.turn] > 0
        if hit:
            self.left[ev.turn] -= 1
        hit = hit or (self.every is not None and ev.turn >= self.every) or bool(self.when and self.when(ev))
        return Decision(self.name, f'{self.name}@{ev.turn}', **self.kw) if hit else None

    def on_outcome(self, rec):
        self.outcomes.append((rec.turn, rec.outcome, rec.why))


def R(name='A', **kw):          # reply phase, rewrite
    return Fake(name, 'reply', where='rewrite', **kw)


def N(name='A', **kw):          # reply phase, next
    return Fake(name, 'reply', where='next', **kw)


def Q(name='A', **kw):          # request phase (always next)
    return Fake(name, 'request', **kw)


def run(coro):
    return asyncio.run(coro)


async def _play(ctl, script, reply=None, extra=None):
    out = []
    turn = 0
    for tok in script.split():
        if tok == 's':                # an auxiliary request (Terminus-2 summarization): no turn
            r = await ctl.on_request(Event('request', turn, main=False))
        else:
            retry, turn = tok.endswith('r'), int(tok.rstrip('r'))
            r = await ctl.on_request(Event('request', turn, is_retry=retry, extra=dict(extra or {})))
            if r.who == 'student':
                r = await ctl.on_reply(Event('reply', turn, is_retry=retry, reply=reply or Reply('x')))
        out.append('T' if r.who == 'teacher' else 'S')
    return ''.join(out)


def play(ctl, script, **kw):
    """Drive one episode. Tokens: 'N' = the main request of turn N, 'Nr' = a retry of it, 's' = a summary request.
    Returns who answered each: S (student) or T (teacher; a rewrite counts as T)."""
    return run(_play(ctl, script, **kw))


def spec(trig, **kw):
    return TriggerSpec(trig, **kw)


EIGHT = '1 2 3 4 5 6 7 8'

# id, triggers (built fresh per case), policy kwargs, script, who answered each request
SCENARIOS = [
    ('quiet', lambda: [R()], {}, '1 2 3 4', 'SSSS'),
    # where x span
    ('rewrite_span1', lambda: [R(at=[3], span=1)], {}, '1 2 3 4 5', 'SSTSS'),
    ('rewrite_span2', lambda: [R(at=[3], span=2)], {}, '1 2 3 4 5', 'SSTTS'),
    ('rewrite_sticky', lambda: [R(at=[3])], {}, '1 2 3 4 5', 'SSTTT'),
    ('next_span1', lambda: [N(at=[3], span=1)], {}, '1 2 3 4 5 6', 'SSSTSS'),
    ('next_span2', lambda: [N(at=[3], span=2)], {}, '1 2 3 4 5 6', 'SSSTTS'),
    ('next_sticky', lambda: [N(at=[3])], {}, '1 2 3 4 5', 'SSSTT'),
    ('request_span1', lambda: [Q(at=[3], span=1)], {}, '1 2 3 4 5', 'SSTSS'),
    ('request_span2', lambda: [Q(at=[3], span=2)], {}, '1 2 3 4 5', 'SSTTS'),
    ('request_sticky', lambda: [Q(at=[3])], {}, '1 2 3 4 5', 'SSTTT'),
    # cooldown: a trigger firing every turn; the student keeps `cooldown` turns after each hand-back
    ('cooldown_rewrite_span1', lambda: [R(every=2, span=1)], dict(cooldown=2), EIGHT, 'STSSTSST'),
    ('cooldown_rewrite_span2', lambda: [R(every=2, span=2)], dict(cooldown=2), EIGHT + ' 9', 'STTSSTTSS'),
    # a next decided at reply t starts at t+1, so it fires one turn earlier for the same gap
    ('cooldown_next_span1', lambda: [N(every=2, span=1)], dict(cooldown=2), EIGHT + ' 9', 'SSTSSTSST'),
    ('cooldown_request_span1', lambda: [Q(every=2, span=1)], dict(cooldown=2), EIGHT, 'STSSTSST'),
    ('cooldown0_back_to_back', lambda: [R(every=2, span=1)], {}, '1 2 3 4', 'STTT'),
    # cap
    ('cap_rewrite_span1', lambda: [R(every=2, span=1)], dict(max_interventions=2), '1 2 3 4 5', 'STTSS'),
    ('cap_next_span1', lambda: [N(every=2, span=1)], dict(max_interventions=2), '1 2 3 4 5 6 7', 'SSTSTSS'),
    ('cap_request_span2', lambda: [Q(every=2, span=2)], dict(max_interventions=1), '1 2 3 4 5 6', 'STTSSS'),
    # earliest_turn
    ('earliest_rewrite', lambda: [R(every=1, span=1)], dict(earliest_turn=4), '1 2 3 4 5', 'SSSTT'),
    ('earliest_next_decided_too_early', lambda: [N(at=[3], span=1)], dict(earliest_turn=4), '1 2 3 4 5', 'SSSSS'),
    ('earliest_request', lambda: [Q(at=[2, 3])], dict(earliest_turn=3), '1 2 3 4', 'SSTT'),
    # retry: the same request gets the same routing
    ('retry_rewrite_span1', lambda: [R(at=[3], span=1)], {}, '1 2 3 3r 4', 'SSTTS'),
    ('retry_rewrite_sticky', lambda: [R(at=[3])], {}, '1 2 3 3r 4', 'SSTTT'),
    ('retry_request_span1', lambda: [Q(at=[3], span=1)], {}, '1 2 3 3r 4', 'SSTTS'),
    ('retry_last_turn_of_span2', lambda: [Q(at=[3], span=2)], {}, '1 2 3 4 4r 5', 'SSTTTS'),
    ('retry_student_turn_not_reasked', lambda: [Q(when=lambda ev: ev.is_retry)], {}, '1 2 3 3r 4', 'SSSSS'),
    ('retry_on_retry_observe_request', lambda: [spec(Q(when=lambda ev: ev.is_retry), on_retry='observe')], {},
     '1 2 3 3r 4', 'SSSTT'),
    # on_retry observe on a rewrite (today's parse repair): the retry asks the student again and judges its new reply
    ('retry_on_retry_observe_rewrite', lambda: [spec(R(at=[3], span=1), on_retry='observe')], {}, '1 2 3 3r 4',
     'SSTSS'),
    ('retry_on_retry_observe_rewrite_refires', lambda: [spec(R(at=[3, 3], span=1), on_retry='observe')], {},
     '1 2 3 3r 4', 'SSTTS'),
    ('retry_on_retry_observe_span2_keeps_teacher', lambda: [spec(R(at=[3], span=2), on_retry='observe')], {},
     '1 2 3 3r 4 5', 'SSTTTS'),
    # a retry drops a pending next (its reply never reached the harness); the retry's own reply is judged again
    ('retry_drops_pending', lambda: [N(at=[3], span=1)], {}, '1 2 3 3r 4 5', 'SSSSSS'),
    ('retry_pending_refired', lambda: [N(at=[3, 3], span=1)], {}, '1 2 3 3r 4 5', 'SSSSTS'),
    # summary requests: routed by the owner, never a turn of the span
    ('summary_in_span', lambda: [R(at=[3], span=2)], {}, '1 2 3 s 4 5', 'SSTTTS'),
    ('summary_after_span', lambda: [R(at=[3], span=1)], {}, '1 2 3 s 4', 'SSTSS'),
    ('summary_sticky', lambda: [Q(at=[2])], {}, '1 2 s 3', 'STTT'),
    ('summary_next_pending', lambda: [N(at=[3], span=1)], {}, '1 2 3 s 4 5', 'SSSSTS'),
    # several fires on one turn: config order wins
    ('two_rewrites_first_wins', lambda: [R('A', at=[3], span=1), R('B', at=[3])], {}, '1 2 3 4 5', 'SSTSS'),
    ('two_rewrites_order_swapped', lambda: [R('B', at=[3]), R('A', at=[3], span=1)], {}, '1 2 3 4 5', 'SSTTT'),
    ('next_before_rewrite', lambda: [N('A', at=[3]), R('B', at=[3], span=1)], {}, '1 2 3 4 5', 'SSSTT'),
    ('request_fire_preempts_reply', lambda: [Q('A', at=[3], span=1), R('B', at=[3])], {}, '1 2 3 4 5', 'SSTSS'),
    # log_only
    ('log_only', lambda: [R(at=[3], log_only=True)], {}, '1 2 3 4 5', 'SSSSS'),
    ('log_only_config', lambda: [spec(R(at=[3]), log_only=True)], {}, '1 2 3 4 5', 'SSSSS'),
    ('log_only_then_acting', lambda: [R('A', at=[3], log_only=True), R('B', at=[3], span=1)], {}, '1 2 3 4 5',
     'SSTSS'),
    # re-arm after a hand-back
    ('rearm_after_cooldown', lambda: [R(at=[3, 5, 6], span=2)], dict(cooldown=1), '1 2 3 4 5 6 7', 'SSTTSTT'),
    ('sticky_never_hands_back', lambda: [Q('A', at=[2, 4]), R('B', at=[5], span=1)], {}, '1 2 3 4 5', 'STTTT'),
    # per-trigger limits count that trigger's interventions only (today's assist gap/max vs repairs)
    ('own_cooldown_ignores_other_triggers',
     lambda: [spec(Q('A', every=2, span=1), limits=dict(cooldown=2, max_interventions=6)), R('B', at=[4], span=1)],
     {}, '1 2 3 4 5 6', 'STSTTS'),
    ('episode_cooldown_counts_all',
     lambda: [Q('A', every=2, span=1), R('B', at=[4], span=1)], dict(cooldown=2), '1 2 3 4 5 6', 'STSSTS'),
    ('own_cap', lambda: [spec(R('A', every=2, span=1), limits=dict(max_interventions=1)), R('B', at=[5], span=1)],
     {}, '1 2 3 4 5 6', 'STSSTS'),
    ('own_earliest_beats_episode', lambda: [spec(R(at=[1], span=1), limits=dict(earliest_turn=1))], {}, '1 2',
     'TS'),
]


@pytest.mark.parametrize('name,triggers,policy,script,expect', SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_scenario(name, triggers, policy, script, expect):
    ctl = Controller(Policy(**policy), triggers())
    assert play(ctl, script) == expect


# ---- records, intervention fields, notes ----------------------------------------------------------------------
def test_records_two_fires_one_turn():
    ctl = Controller(Policy(), [R('A', at=[3], span=1), R('B', at=[3]), R('C', at=[3], log_only=True)])
    seen = []

    async def go():
        for t in (1, 2, 3):
            await ctl.on_request(Event('request', t))
            r = await ctl.on_reply(Event('reply', t, reply=Reply('x')))
            seen.append(r)
    run(go())
    recs = seen[-1].records
    assert [(x.decision.trigger, x.outcome, x.why) for x in recs] == [
        ('A', 'acted', None), ('B', 'suppressed', 'lower_priority'), ('C', 'shadow', None)]
    assert recs[0].intervention == 1 and seen[-1].why == 'rewrite' and seen[-1].intervention.trigger == 'A'
    assert [x.as_dict()['outcome'] for x in ctl.records] == ['acted', 'suppressed', 'shadow']


def test_records_suppression_reasons():
    ctl = Controller(Policy(cooldown=1, max_interventions=2, earliest_turn=2), [R(every=1, span=1)])
    assert play(ctl, '1 2 3 4 5 6 7') == 'STSTSSS'
    got = [(x.turn, x.outcome, x.why) for x in ctl.records]
    assert got == [(1, 'suppressed', 'earliest'), (2, 'acted', None), (3, 'suppressed', 'cooldown'), (4, 'acted', None),
                   (5, 'suppressed', 'cap'), (6, 'suppressed', 'cap'), (7, 'suppressed', 'cap')]


def test_pending_next_lifecycle():
    trig = N(at=[3], span=2, note='verify')
    ctl = Controller(Policy(), [trig])
    rs = []

    async def go():
        for t in (1, 2, 3, 4, 5, 6):
            r = await ctl.on_request(Event('request', t))
            if r.who == 'student':
                r = await ctl.on_reply(Event('reply', t, reply=Reply('x')))
            rs.append(r)
    run(go())
    assert [r.who[0].upper() for r in rs] == list('SSSTTS')
    assert [(x.phase, x.turn, x.outcome) for x in ctl.records] == [('reply', 3, 'pending'),
                                                                   ('request', 4, 'acted')]
    assert rs[3].why == 'start' and rs[3].teacher_note == 'verify'
    assert rs[4].why == 'span' and rs[4].teacher_note is None     # the note is for the intervention's first turn only
    I = ctl.interventions[0]
    assert (I.fired_turn, I.start_turn, I.turns_done, I.last_turn, I.handed_back) == (3, 4, 2, 5, True)
    assert trig.outcomes == [(3, 'pending', None), (4, 'acted', None)]


def test_pending_dropped_when_trigger_says_so():
    class Claim(Fake):
        def pending_applies(self, ev, d):
            return ev.extra.get('request_kind') == 'confirm'
    ctl = Controller(Policy(), [Claim('done_claim', 'reply', at=[3])])
    assert play(ctl, '1 2 3 4 5') == 'SSSSS'
    assert [(x.turn, x.outcome, x.why) for x in ctl.records] == [(3, 'pending', None), (4, 'suppressed', 'dropped')]
    ctl = Controller(Policy(), [Claim('done_claim', 'reply', at=[3])])
    assert play(ctl, '1 2 3 4 5', extra=dict(request_kind='confirm')) == 'SSSTT'


def test_retry_memo_same_intervention_and_note():
    ctl = Controller(Policy(), [R(at=[3], span=1, note='n')])

    async def go():
        for t in (1, 2):
            await ctl.on_request(Event('request', t))
            await ctl.on_reply(Event('reply', t, reply=Reply('x')))
        await ctl.on_request(Event('request', 3))
        first = await ctl.on_reply(Event('reply', 3, reply=Reply('x')))
        again = await ctl.on_request(Event('request', 3, is_retry=True))
        return first, again
    first, again = run(go())
    assert first.why == 'rewrite' and again.why == 'retry'
    assert again.intervention.id == first.intervention.id and again.intervention.turns_done == 1
    assert again.teacher_note == 'n' and again.records == ()


def test_show_rejected_reply_on_rewrite_only():
    ctl = Controller(Policy(show_rejected_reply=True), [R(at=[2], span=1), Q('B', at=[4], span=1)])

    async def go():
        out = []
        for t in (1, 2, 3, 4):
            r = await ctl.on_request(Event('request', t))
            if r.who == 'student':
                r = await ctl.on_reply(Event('reply', t, reply=Reply('x')))
            out.append(r.show_rejected_reply)
        return out
    assert run(go()) == [False, True, False, False]


def test_request_phase_rewrite_decision_is_an_error():
    class Bad:
        name, phases = 'bad', frozenset({'request'})

        async def observe(self, ev):
            return Decision('bad', 'x', where='rewrite')
    with pytest.raises(ValueError, match='cannot rewrite'):
        play(Controller(Policy(), [Bad()]), '1 2')


def test_triggers_see_owner_and_interventions():
    a, b = R('A', at=[2], span=1), Q('B')
    play(Controller(Policy(earliest_turn=1), [a, b]), '1 2 3')
    assert a.seen == [('reply', 1, False, 'student'), ('reply', 2, False, 'student'), ('reply', 3, False, 'student')]
    seen_iv = []

    class Look(Fake):
        async def observe(self, ev):
            seen_iv.append(tuple(i.trigger for i in ev.interventions))
    play(Controller(Policy(earliest_turn=1), [a.__class__('A', 'reply', at=[2], where='rewrite', span=1),
                                              Look('L', 'request')]), '1 2 3')
    assert seen_iv == [(), (), ('A',)]


def test_teacher_turns_and_summaries_are_not_observed_by_default():
    a = Q('A', at=[2], span=2)
    play(Controller(Policy(), [a]), '1 2 s 3 4')
    assert [(p, t) for p, t, *_ in a.seen] == [('request', 1), ('request', 2), ('request', 4)]


def test_initial_owner_teacher_logs_would_fire():
    a = spec(Q('A', at=[2]), owners=('student', 'teacher'))
    b = spec(R('B', at=[3]), owners=('student', 'teacher'), log_only=True)
    c = R('C', at=[2])                                     # student turns only: never asked
    ctl = Controller(Policy(), [a, b, c], initial_owner='teacher')

    async def go():
        out = []
        for t in (1, 2, 3):
            r = await ctl.on_request(Event('request', t))
            r2 = await ctl.on_reply(Event('reply', t, reply=Reply('teacher text')))   # the teacher's reply, logged only
            out.append((r.who, r.why, r2.who))
        return out
    assert run(go()) == [('teacher', 'owner', 'teacher')] * 3
    assert [(x.decision.trigger, x.turn, x.outcome, x.why) for x in ctl.records] == [
        ('A', 2, 'suppressed', 'owner'), ('B', 3, 'shadow', None)]
    assert ctl.sticky and ctl.interventions == [] and c.seen == []


@pytest.mark.parametrize('mode,asked,fates', [
    ('always', [1, 2, 3, 4, 5, 6], [(2, 'lower_priority'), (3, 'cooldown'), (4, None), (5, 'cooldown'), (6, None)]),
    # armed: not asked before earliest_turn (1), after A acted (2) or in its cooldown (3, 5): only acting fires remain
    ('armed', [4, 6], [(4, None), (6, None)]),
])
def test_observe_armed_skips_what_cannot_act(mode, asked, fates):
    a, b = R('A', at=[2], span=1), R('B', every=2, span=1)
    ctl = Controller(Policy(cooldown=1), [a, spec(b, observe=mode, where='rewrite')])
    assert play(ctl, '1 2 3 4 5 6') == 'STSTST'
    assert [t for _, t, *_ in b.seen] == asked
    assert [(x.turn, x.why) for x in ctl.records if x.decision.trigger == 'B'] == fates


def test_begin_request_applies_pending_before_endings():
    ctl = Controller(Policy(), [N('done_claim', at=[3]), Q('env', every=4, span=1)])

    async def go():
        for t in (1, 2, 3):
            await ctl.on_request(Event('request', t))
            await ctl.on_reply(Event('reply', t, reply=Reply('x')))
        ev = Event('request', 4)
        r0 = await ctl.begin_request(ev)
        sticky_between = ctl.sticky            # what the router's budget check reads
        r1 = await ctl.on_request(ev)
        return r0, sticky_between, r1
    r0, sticky, r1 = run(go())
    assert (r0.who, r0.why, sticky) == ('teacher', 'start', True)
    assert r1.intervention.trigger == 'done_claim' and [x.outcome for x in r1.records] == ['acted']


def test_owner_property_tracks_span():
    ctl = Controller(Policy(), [R(at=[2], span=2)])
    owners = []

    async def go():
        for t in (1, 2, 3, 4):
            r = await ctl.on_request(Event('request', t))
            if r.who == 'student':
                await ctl.on_reply(Event('reply', t, reply=Reply('x')))
            owners.append((ctl.owner, ctl.sticky))
    run(go())
    # after the rewrite at 2 the span has a turn left (teacher); after turn 3 it has none (the student is next)
    assert owners == [('student', False), ('teacher', False), ('student', False), ('student', False)]


def test_trigger_state_is_per_controller():
    _, specs = iv.load_config({'triggers': [{'use': 'action_error', 'params': {'streak': 2}}]})
    a, b = Controller(Policy(), specs), Controller(Policy(), specs)
    assert a.specs[0].trigger is not b.specs[0].trigger
    shared = Fake('S', 'request')
    assert Controller(Policy(), [spec(shared)]).specs[0].trigger is shared   # hand-built: shared (reset if it can)


def test_duplicate_names_rejected():
    with pytest.raises(ConfigError, match='unique'):
        Controller(Policy(), [R('A'), N('A')])


# ---- config loader -------------------------------------------------------------------------------------------
def test_load_builtin_with_overrides(tmp_path):
    cfg = {'policy': {'cooldown': 3, 'max_interventions': 6, 'show_rejected_reply': True},
           'triggers': [{'use': 'scripted', 'name': 'probe', 'params': {'turns': [3]}, 'phases': ['reply'],
                         'where': 'rewrite', 'span': 2, 'teacher_note': 'hi',
                         'policy': {'cooldown': 0, 'earliest_turn': 1, 'observe': 'armed', 'on_retry': 'observe'}},
                        {'use': 'action_error', 'log_only': True}]}
    pol, specs = iv.load_config(cfg)
    assert (pol.cooldown, pol.max_interventions, pol.show_rejected_reply) == (3, 6, True)
    s = specs[0]
    assert (s.name, s.phases, s.where, s.span, s.teacher_note) == ('probe', frozenset({'reply'}), 'rewrite', 2, 'hi')
    assert (s.limits, s.observe, s.on_retry) == ({'cooldown': 0, 'earliest_turn': 1}, 'armed', 'observe')
    assert specs[1].log_only and specs[1].name == 'action_error'
    ctl = Controller(pol, specs)
    assert play(ctl, '1 2 3 4 5') == 'SSTTS'
    assert ctl.interventions[0].teacher_note == 'hi'
    p = tmp_path / 'cfg.yaml'
    p.write_text('policy: {earliest_turn: 3}\ntriggers:\n  - use: scripted\n    params: {turns: [2, 4]}\n    span: 1\n')
    ctl = Controller.from_config(str(p))
    assert play(ctl, '1 2 3 4 5') == 'SSSTS'                        # turn 2 is before earliest_turn 3


PLUGIN = '''
from dataclasses import dataclass
from interventions import Decision


@dataclass
class EveryOther:
    period: int = 2
    name: str = 'every_other'
    phases: frozenset = frozenset({'request'})

    async def observe(self, ev):
        return Decision(self.name, 'tick') if ev.turn % self.period == 0 else None
'''


def test_load_module_class(tmp_path, monkeypatch):
    (tmp_path / 'my_lab_triggers.py').write_text(PLUGIN)
    monkeypatch.syspath_prepend(str(tmp_path))
    cfg = {'triggers': [{'use': 'my_lab_triggers:EveryOther', 'params': {'period': 3}, 'span': 1}]}
    assert play(Controller.from_config(cfg), '1 2 3 4 5 6 7') == 'SSTSSTS'
    # a file path next to the config file
    (tmp_path / 'sub').mkdir()
    (tmp_path / 'sub' / 'mine.py').write_text(PLUGIN)
    (tmp_path / 'sub' / 'cfg.yaml').write_text('triggers:\n  - use: mine.py:EveryOther\n    span: 1\n')
    assert play(Controller.from_config(str(tmp_path / 'sub' / 'cfg.yaml')), '1 2 3 4') == 'STST'


BAD_CONFIGS = [
    ('unknown_top', {'triggerz': []}, 'unknown top-level'),
    ('unknown_entry_key', {'triggers': [{'use': 'scripted', 'spam': 1}]}, 'unknown key'),
    ('unknown_policy_key', {'policy': {'cooldwn': 1}}, 'policy: unknown key'),
    ('bad_policy_value', {'policy': {'student_view_after_handback': 'some'}}, 'student_view_after_handback'),
    ('negative_cooldown', {'policy': {'cooldown': -1}}, 'cooldown'),
    ('unknown_trigger_policy_key', {'triggers': [{'use': 'scripted', 'policy': {'gap': 3}}]}, 'unknown policy key'),
    ('request_rewrite', {'triggers': [{'use': 'action_error', 'where': 'rewrite'}]}, 'always acts as next'),
    ('request_rewrite_builtin', {'triggers': [{'use': 'loop', 'where': 'rewrite'}]}, 'always acts as next'),
    ('span_zero', {'triggers': [{'use': 'scripted', 'span': 0}]}, 'span must be'),
    ('span_minus_two', {'triggers': [{'use': 'scripted', 'span': -2}]}, 'span must be'),
    ('bad_where', {'triggers': [{'use': 'scripted', 'where': 'later'}]}, 'where must be'),
    ('missing_use', {'triggers': [{'span': 1}]}, '`use:`'),
    ('unknown_builtin', {'triggers': [{'use': 'loopz'}]}, 'no built-in'),
    ('bad_module', {'triggers': [{'use': 'no_such_module_xyz:T'}]}, 'cannot import'),
    ('missing_class', {'triggers': [{'use': 'interventions:NoSuchTrigger'}]}, 'has no'),
    ('duplicate_names', {'triggers': [{'use': 'scripted'}, {'use': 'scripted'}]}, 'unique'),
    ('phase_not_allowed', {'triggers': [{'use': 'action_error', 'phases': ['reply']}]}, 'runs at'),
    ('bad_observe', {'triggers': [{'use': 'scripted', 'policy': {'observe': 'sometimes'}}]}, 'observe must be'),
    ('bad_owners', {'triggers': [{'use': 'scripted', 'policy': {'owners': ['judge']}}]}, 'owners must be'),
    ('bad_params', {'triggers': [{'use': 'scripted', 'params': {'turnz': [1]}}]}, 'bad params'),
    ('bad_scanner_param', {'triggers': [{'use': 'loop', 'params': {'loop_q': 1}}]}, 'unknown param'),
    ('bad_regex', {'triggers': [{'use': 'action_error', 'params': {'patterns': ['(']}}]}, 'bad regex'),
    ('stub_assist', {'triggers': [{'use': 'assist'}]}, 'stage 2'),
    ('stub_judge', {'triggers': [{'use': 'llm_judge'}]}, 'stage 4'),
    ('not_a_mapping', [1, 2], 'mapping'),
]


@pytest.mark.parametrize('name,cfg,match', BAD_CONFIGS, ids=[b[0] for b in BAD_CONFIGS])
def test_config_errors(name, cfg, match):
    with pytest.raises(ConfigError, match=match):
        iv.load_config(cfg)


def test_registry_names():
    for n in ('action_error', 'scripted', 'loop', 'no_progress_wait', 'edit_failed', 'error_streak', 'gave_up',
              'destructive', 'submit_check', 'success_claim_unchecked', 'success_contradicted', 'done_claim',
              'parse_repair', 'repair_confirm', 'context_budget', 'time_budget', 'assist', 'llm_judge'):
        assert n in iv.BUILTINS, n


# ---- action_error and scripted -------------------------------------------------------------------------------
def _ae(results, **params):
    """action_error's verdict at each request, fed one ActionResult (or None) per turn starting at turn 2."""
    t = iv.ActionError(**params)

    async def go():
        out = []
        for i, r in enumerate([None] + list(results)):
            d = await t.observe(Event('request', i + 1, last_result=r))
            out.append(d.reason if d else None)
        return out[1:]
    return run(go())


OK = ActionResult('ls', 0, output='a b')
FAIL = ActionResult('make', 2, output='make: *** [all] Error 2')


def test_action_error_exit_code_and_timeout():
    assert _ae([OK, FAIL, ActionResult('x', None)]) == [None, 'exit code 2', None]
    assert _ae([ActionResult('sleep 99', None, timed_out=True)]) == ['timed out']
    assert _ae([ActionResult('sleep 99', None, timed_out=True)], timeouts=False) == [None]
    assert _ae([FAIL, ActionResult('x', 137)], exit_codes=[137, 124]) == [None, 'exit code 137']
    assert _ae([FAIL], exit_codes=None) == [None]


def test_action_error_patterns_and_ignore():
    tb = ActionResult('python run.py', None, output='Traceback (most recent call last):\n  ...')
    assert _ae([tb, OK], patterns=['Traceback', 'command not found']) == ['output matches /Traceback/', None]
    grep = ActionResult('grep -n foo x.py', 1, output='')
    assert _ae([grep]) == ['exit code 1']
    assert _ae([grep], ignore_commands=[r'^\s*(grep|diff|test)\b']) == [None]


def test_action_error_streak():
    assert _ae([FAIL, FAIL, FAIL], streak=2) == [None, '2 failed actions in a row, last: exit code 2',
                                                 '3 failed actions in a row, last: exit code 2']
    assert _ae([FAIL, OK, FAIL], streak=2) == [None, None, None]
    assert _ae([FAIL, None, FAIL], streak=2)[-1] is not None       # a request without a result leaves the streak


def test_action_error_streak_gap_and_retry():
    t = iv.ActionError(streak=2)

    async def go():
        out = []
        for turn, r in ((2, FAIL), (2, FAIL), (3, OK), (4, FAIL), (6, FAIL), (7, FAIL)):
            d = await t.observe(Event('request', turn, last_result=r))
            out.append(bool(d))
        return out
    # the retry at 2 does not count twice; turn 5 (not seen, e.g. the teacher's) restarts the streak at 6
    assert run(go()) == [False, False, False, False, False, True]


def test_action_error_through_controller():
    ctl = Controller.from_config({'triggers': [{'use': 'action_error', 'params': {'streak': 2}, 'span': 1}]})
    results = {3: FAIL, 4: FAIL, 5: OK, 6: OK}

    async def go():
        out = []
        for t in range(1, 7):
            r = await ctl.on_request(Event('request', t, last_result=results.get(t)))
            out.append(r.who[0].upper())
            if r.who == 'student':
                await ctl.on_reply(Event('reply', t, reply=Reply('x')))
        return ''.join(out)
    assert run(go()) == 'SSSTSS'
    assert ctl.interventions[0].reason == '2 failed actions in a row, last: exit code 2'


def test_scripted_turns_plan_and_phase():
    ctl = Controller.from_config({'triggers': [{'use': 'scripted', 'params': {'turns': [2], 'plan': {'t1': [4]}},
                                                'span': 1}]})
    assert play(ctl, '1 2 3 4 5', extra={'task_id': 't1'}) == 'STSTS'
    ctl = Controller.from_config({'triggers': [{'use': 'scripted', 'params': {'turns': [3]}, 'phases': ['reply'],
                                                'where': 'next', 'span': 1}]})
    assert play(ctl, '1 2 3 4 5') == 'SSSTS'


# ---- legacy adapters (wiring only; stage 2 proves identity against the router) --------------------------------
def T2(analysis='a', cmds=(), done=False):
    return json.dumps(dict(analysis=analysis, plan='p', task_complete=done,
                           commands=[dict(keystrokes=k, duration=1.0) for k in cmds]))


def test_terminus2_repair_confirm_and_done_claim():
    """Today's Terminus-2 relay_repair flow through the legacy adapters: a parse error at 2 is repaired (one turn);
    the teacher's repair claimed done, so it answers the confirmation at 3 and the student resumes at 4; the
    student's own claim at 5 goes through and the teacher takes the episode at the confirmation (6) with the note."""
    ctl = Controller.from_config({'triggers': [
        {'use': 'repair_confirm', 'policy': {'earliest_turn': 1, 'on_retry': 'observe'}},
        {'use': 'parse_repair', 'policy': {'earliest_turn': 1, 'on_retry': 'observe'}},
        {'use': 'done_claim', 'params': {'confirm_kind': 'confirm'}, 'teacher_note': 'VERIFY',
         'policy': {'earliest_turn': 1}}]})
    turns = {1: ('main', Reply(T2(cmds=['ls\n']))), 2: ('main', Reply('{"analysis": ', parse_error='no JSON')),
             3: ('confirm', None), 4: ('main', Reply(T2(cmds=['make\n']))), 5: ('main', Reply(T2(done=True))),
             6: ('confirm', None), 7: ('main', None)}

    async def go():
        out = []
        for t, (kind, reply) in turns.items():
            r = await ctl.on_request(Event('request', t, extra={'request_kind': kind}))
            if r.who == 'student':
                r = await ctl.on_reply(Event('reply', t, reply=reply or Reply(T2())))
            out.append((r.who[0].upper(), r.why, r.intervention.trigger if r.intervention else None, r.teacher_note))
        return out
    got = run(go())
    assert [g[0] for g in got] == list('STTSSTT')
    assert got[1][1:3] == ('rewrite', 'parse_repair') and got[2][1:3] == ('start', 'repair_confirm')
    assert got[5] == ('T', 'start', 'done_claim', 'VERIFY') and got[6][1] == 'span'
    assert ctl.sticky and ctl.interventions[-1].fired_turn == 5


def test_budget_adapters():
    async def n(v):
        return v

    async def go():
        cb = iv.ContextBudget(tokens=1000)
        hit = await cb.observe(Event('request', 3, view_tokens=lambda: n(1200)))
        early = await cb.observe(Event('request', 1, view_tokens=lambda: n(5000)))
        failed = await cb.observe(Event('request', 3))                 # the count failed: no fire
        tb = iv.TimeBudget(frac=0.5)
        late = await tb.observe(Event('request', 3, clock=Clock(600, 1000)))
        fine = await tb.observe(Event('request', 3, clock=Clock(400, 1000)))
        return hit, early, failed, late, fine
    hit, early, failed, late, fine = run(go())
    assert hit.reason == 'student view 1200 tokens >= 1000' and hit.score == 1200
    assert early is None and failed is None and fine is None
    assert late.reason == 'student used 600 s of 1000 s'


def test_scanner_adapter_loop():
    rt = iv._rt()
    msgs = [dict(role='user', content='Task Description:\nfix it\n\nCurrent terminal state:\nroot@h:/app# ')]
    for _ in range(3):
        msgs += [dict(role='assistant', content=T2(cmds=['python run.py\n'])),
                 dict(role='user', content='New Terminal Output:\nroot@h:/app# python run.py\nboom\nroot@h:/app# ')]
    loop = iv.BUILTINS['loop']()

    async def at(k):            # the request after k assistant turns
        sc = rt.scan_messages(msgs[:1 + 2 * k])
        return await loop.observe(Event('request', sc.turn + 1, scanner=sc))
    assert run(at(2)) is None
    d = run(at(3))
    assert d.trigger == 'loop' and d.reason.startswith('exact repeat x3')


# ==== stage 0b: function triggers, combinators, releases, per-intervention view / teacher, info ===============
from interventions import View, trigger  # noqa: E402

EXAMPLES = HERE.parent.parent / 'triggers' / 'examples'


async def _drive(ctl, script, results=None, replies=None, teacher_replies=None):
    """Like _play, plus ActionResults per request turn, the student's reply text per token, and (teacher_replies) the
    teacher's reply per token sent to on_reply after the teacher answers, as the router does when observes_teacher."""
    out, routings, turn = [], [], 0
    for tok in script.split():
        retry, turn = tok.endswith('r'), int(tok.rstrip('r'))
        r = await ctl.on_request(Event('request', turn, is_retry=retry, last_result=(results or {}).get(turn)))
        if r.who == 'student':
            r = await ctl.on_reply(Event('reply', turn, is_retry=retry, reply=Reply((replies or {}).get(tok, 'x'))))
        if r.who == 'teacher' and teacher_replies is not None:
            await ctl.on_reply(Event('reply', turn, is_retry=retry, reply=Reply(teacher_replies.get(tok, 'x'))))
        out.append('T' if r.who == 'teacher' else 'S')
        routings.append(r)
    return ''.join(out), routings


def drive(ctl, script, **kw):
    return run(_drive(ctl, script, **kw))


# ---- function triggers ----------------------------------------------------------------------------------------
def test_function_trigger_state_is_per_episode():
    async def third_request(ev, state):
        state['n'] = state.get('n', 0) + 1
        if state['n'] == 3:
            return Decision(reason='third request', span=1)     # no trigger name: the function's is used
    for _ in range(2):                                           # a second episode starts from an empty state
        ctl = Controller(Policy(), [third_request])
        assert play(ctl, '1 2 3 4') == 'SSTS'
        assert ctl.interventions[0].trigger == 'third_request' and ctl.records[0].decision.reason == 'third request'
    s = spec(third_request)
    a, b = Controller(Policy(), [s]), Controller(Policy(), [s])
    assert a.specs[0].trigger is not b.specs[0].trigger           # concurrent episodes never share a state dict
    play(a, '1 2')
    assert play(b, '1 2 3') == 'SST'


def test_function_trigger_forms():
    @trigger(phases='reply', name='claims')
    def claims(ev, state):
        return 'claims done' if 'done' in ev.reply.text else None

    def as_dict(ev, state):
        return {'reason': 'every third', 'span': 1} if ev.turn % 3 == 0 else None
    assert claims(Event('reply', 1, reply=Reply('done')), {}) == 'claims done'      # still a plain function
    ctl = Controller(Policy(), [spec(claims, where='rewrite', span=1)])
    assert ctl.specs[0].phases == frozenset({'reply'}) and ctl.specs[0].name == 'claims'
    assert play(ctl, '1 2 3', reply=Reply('done')) == 'STT'
    ctl = Controller(Policy(), [as_dict])
    assert play(ctl, '1 2 3 4') == 'SSTS' and ctl.interventions[0].reason == 'every third'

    def bad(ev, state):
        return 42
    with pytest.raises(TypeError, match='function trigger returns'):
        play(Controller(Policy(), [bad]), '1')


FUNC_PLUGIN = '''
from interventions import Decision, Scripted, trigger


def at_turn(ev, state, turn=3):
    if ev.turn == turn:
        return Decision(reason=f'turn {turn}', span=1)


@trigger(phases=('request', 'reply'), name='tagged')
async def tagged(ev, state):
    return None


def make_scripted(turns=(2,)):          # fewer than two required parameters: a factory, as before
    return Scripted(turns=turns)
'''


def test_load_function_triggers(tmp_path, monkeypatch):
    (tmp_path / 'my_func_plugin.py').write_text(FUNC_PLUGIN)
    monkeypatch.syspath_prepend(str(tmp_path))
    ctl = Controller.from_config({'triggers': [{'use': 'my_func_plugin:at_turn', 'params': {'turn': 4}}]})
    assert play(ctl, '1 2 3 4 5') == 'SSSTS'
    ctl = Controller.from_config({'triggers': [{'use': 'my_func_plugin:make_scripted', 'params': {'turns': [3]},
                                                'span': 1}]})
    assert play(ctl, '1 2 3 4 5') == 'SSTSS' and isinstance(ctl.specs[0].trigger, iv.Scripted)
    _, specs = iv.load_config({'triggers': [{'use': 'my_func_plugin:tagged'}]})
    assert specs[0].name == 'tagged' and specs[0].phases == frozenset(iv.PHASES)
    with pytest.raises(ConfigError, match='bad params'):
        iv.load_config({'triggers': [{'use': 'my_func_plugin:at_turn', 'params': {'nope': 1}}]})
    (tmp_path / 'sub').mkdir()
    (tmp_path / 'sub' / 'funcs.py').write_text(FUNC_PLUGIN)
    (tmp_path / 'sub' / 'cfg.yaml').write_text('triggers:\n  - use: funcs.py:at_turn\n')
    assert play(Controller.from_config(str(tmp_path / 'sub' / 'cfg.yaml')), '1 2 3 4') == 'SSTS'


# ---- combinators ----------------------------------------------------------------------------------------------
FAIL2 = ActionResult('make', 2)


def test_all_of_action_error_after_turn():
    cfg = {'triggers': [{'use': 'all_of', 'span': 1, 'params': {'of': [
        {'use': 'action_error', 'params': {'streak': 2}}, {'use': 'after_turn', 'params': {'turn': 5}}]}}]}
    ctl = Controller.from_config(cfg)
    who, _ = drive(ctl, '1 2 3 4 5 6 7', results={3: FAIL2, 4: FAIL2, 5: FAIL2, 6: FAIL2, 7: OK})
    assert who == 'SSSSSTS'                      # the streak reached 2 at turn 4 but only turn 6 is after turn 5
    d = ctl.records[0].decision
    assert d.trigger == 'all_of' and d.reason == '4 failed actions in a row, last: exit code 2; after turn 5'
    assert [c['trigger'] for c in d.info['of']] == ['action_error', 'after_turn']


def test_any_of_first_firing_child_and_phases():
    cfg = {'triggers': [{'use': 'any_of', 'where': 'rewrite', 'span': 1, 'params': {'of': [
        {'use': 'scripted', 'params': {'turns': [2]}},
        {'use': 'scripted', 'name': 'late', 'params': {'turns': [2, 4]}, 'phases': ['reply']}]}}]}
    ctl = Controller.from_config(cfg)
    assert ctl.specs[0].phases == frozenset(iv.PHASES)
    assert play(ctl, '1 2 3 4 5') == 'STSTS'     # 2: the request-time child (next); 4: the reply-time one (rewrite)
    first, second = ctl.records
    assert (first.phase, first.decision.info['of'][0]['trigger']) == ('request', 'scripted')
    assert (second.phase, second.decision.info['of'][0]['trigger'], ctl.interventions[1].where) == (
        'reply', 'late', 'rewrite')


def test_streak_counts_turns():
    def fires(st, turns):
        async def go():
            return [bool(await st.observe(Event('request', t))) for t in turns]
        return run(go())
    st = iv.Streak(child=iv.Scripted(turns=[2, 3, 4, 6, 7, 9]), n=2)
    # 3r: a retry re-judges turn 3; 5 breaks the run; 9 follows an unseen turn 8, so it restarts
    assert fires(st, [1, 2, 3, 3, 4, 5, 6, 7, 9]) == [False, False, True, True, True, False, False, True, False]
    assert fires(iv.Streak(child=iv.Scripted(turns=[2, 3, 5]), n=3), [2, 3, 5]) == [False, False, False]
    assert fires(iv.Streak(child=iv.Scripted(turns=[2, 3, 5]), n=3, contiguous=False), [2, 3, 5]) == [
        False, False, True]
    d = run(iv.Streak(child=iv.Scripted(turns=[1, 2]), n=2).observe(Event('request', 1)))
    assert d is None


def test_turn_gates():
    def at(t, turns):
        async def go():
            return [bool(await t.observe(Event('request', x))) for x in turns]
        return run(go())
    assert at(iv.AfterTurn(turn=5), [5, 6]) == [False, True]
    assert at(iv.BeforeTurn(turn=3), [2, 3]) == [True, False]
    assert at(iv.Every(n=3), [2, 3, 4, 6]) == [False, True, False, True]
    assert at(iv.Every(n=2, start=5), [3, 5, 6, 7]) == [False, True, False, True]


def test_nested_combinators_with_relative_function_child(tmp_path):
    (tmp_path / 'funcs.py').write_text(FUNC_PLUGIN)
    (tmp_path / 'cfg.yaml').write_text('''triggers:
  - use: all_of
    span: 1
    params:
      of:
        - use: any_of
          params: {of: [{use: funcs.py:at_turn, params: {turn: 4}}, {use: scripted, params: {turns: [6]}}]}
        - use: after_turn
          params: {turn: 3}
''')
    policy, specs = iv.load_config(str(tmp_path / 'cfg.yaml'))
    a, b = Controller(policy, specs), Controller(policy, specs)
    assert play(a, '1 2 3 4 5 6 7') == 'SSSTSTS'
    inner = lambda c: c.specs[0].trigger.children[0].trigger.children[0].trigger   # noqa: E731
    assert inner(a) is not inner(b)                  # nested children are rebuilt per episode


COMBINATOR_ERRORS = [
    ('child_key', {'use': 'any_of', 'params': {'of': [{'use': 'scripted', 'span': 1}]}}, r'of\[0\]: unknown key'),
    ('child_use', {'use': 'any_of', 'params': {'of': [{'use': 'nope'}]}}, r'of\[0\] \(nope\): no built-in'),
    ('empty_of', {'use': 'all_of', 'params': {'of': []}}, 'non-empty list'),
    ('no_shared_phase', {'use': 'all_of', 'params': {'of': [{'use': 'action_error'},
                                                             {'use': 'scripted', 'phases': ['reply']}]}},
     'share no phase'),
    ('streak_n', {'use': 'streak', 'params': {'child': {'use': 'scripted'}, 'n': 0}}, 'n must be an int'),
    ('gate_turn', {'use': 'after_turn', 'params': {'turn': 'five'}}, 'turn must be an int'),
    ('release_key', {'use': 'action_error', 'release': {'use': 'action_ok', 'span': 1}}, r'\.release: unknown key'),
    ('view_key', {'use': 'scripted', 'view': {'colour': 'red'}}, 'unknown view key'),
    ('view_value', {'use': 'scripted', 'view': {'failed_turn_view': 'blur'}}, 'failed_turn_view must be'),
    ('teacher_empty', {'use': 'scripted', 'teacher': ''}, 'teacher names an endpoint'),
]


@pytest.mark.parametrize('name,entry,match', COMBINATOR_ERRORS, ids=[c[0] for c in COMBINATOR_ERRORS])
def test_new_config_errors(name, entry, match):
    with pytest.raises(ConfigError, match=match):
        iv.load_config({'triggers': [entry]})


def test_policy_release_config_errors():
    with pytest.raises(ConfigError, match='release is a list'):
        iv.load_config({'policy': {'release': {'use': 'action_ok'}}})
    with pytest.raises(ConfigError, match=r'policy\.release\[0\] \(action_ok\): bad params'):
        iv.load_config({'policy': {'release': [{'use': 'action_ok', 'params': {'nope': 1}}]}})


# ---- releases -------------------------------------------------------------------------------------------------
UNTIL_FIXED = {2: OK, 3: FAIL2, 4: ActionResult('fix', 1), 5: OK, 6: OK}


def test_stay_until_fixed_example():
    ctl = Controller.from_config(str(EXAMPLES / 'stay_until_fixed.yaml'))
    assert not ctl.observes_teacher                  # a request-time release needs no teacher replies
    who, rs = drive(ctl, '1 2 3 4 5 6 7', results=UNTIL_FIXED)
    # 3: the student's command failed -> teacher; 4: the teacher's first fix failed; 5: its fix exited 0 -> released
    # on that request, so the student answers 5
    assert who == 'SSTTSSS'
    I = ctl.interventions[0]
    assert (I.released_by, I.handed_back, I.last_turn, I.sticky) == ('action_error.release', True, 4, True)
    rel = [(x.turn, x.outcome, x.decision.kind, x.intervention) for x in ctl.records if x.decision.kind == 'release']
    assert rel == [(5, 'acted', 'release', 1)]
    assert (rs[4].who, rs[4].why) == ('student', 'student') and not ctl.sticky
    plain = Controller.from_config({'triggers': [{'use': 'action_error', 'span': -1}]})
    assert drive(plain, '1 2 3 4 5 6 7', results=UNTIL_FIXED)[0] == 'SSTTTTT'


def test_policy_release_and_span_k_ends_early():
    ctl = Controller.from_config({'policy': {'release': [{'use': 'action_ok'}]},
                                  'triggers': [{'use': 'action_error', 'span': 4}]})
    assert drive(ctl, '1 2 3 4 5 6 7', results=UNTIL_FIXED)[0] == 'SSTTSSS'
    assert ctl.interventions[0].released_by == 'action_ok' and ctl.interventions[0].turns_done == 2


def test_release_not_asked_on_the_start_request():
    r = Fake('R', 'request', every=1)
    ctl = Controller(Policy(release=(r,)), [N('A', at=[2])])
    assert play(ctl, '1 2 3 4 5 6') == 'SSTSSS'     # pending next starts at 3; the release first judges turn 4
    assert [t for _, t, *_ in r.seen] == [4]


def test_request_time_release_before_endings():
    """begin_request already hands back, so the router's endings (between begin_request and on_request) read the
    student's clock; the student's own request triggers are then asked on that request as usual."""
    b = Q('B')
    ctl = Controller.from_config({'policy': {'release': [{'use': 'action_ok'}]},
                                  'triggers': [{'use': 'action_error', 'span': -1}]})
    ctl = Controller(ctl.policy, ctl.specs + [spec(b)])
    drive(ctl, '1 2 3 4', results=UNTIL_FIXED)
    assert ctl.sticky
    ev = Event('request', 5, last_result=UNTIL_FIXED[5])
    r0 = run(ctl.begin_request(ev))
    assert (r0.who, ctl.sticky, ctl.owner) == ('student', False, 'student')
    r1 = run(ctl.on_request(ev))
    assert r1.who == 'student' and [x.outcome for x in r1.records] == ['acted']
    assert ('request', 5, False, 'student') in b.seen and ('request', 4, False, 'teacher') not in b.seen


def test_release_retries_keep_the_memo():
    r = Fake('R', 'request', when=lambda ev: ev.is_retry)    # would release on any retry, were it asked
    ctl = Controller(Policy(release=(r,)), [Q('A', at=[2])])
    assert play(ctl, '1 2 3 3r 4') == 'STTTT'
    assert [(t, retry) for _, t, retry, _ in r.seen] == [(3, False), (4, False)]
    ctl = Controller.from_config(str(EXAMPLES / 'stay_until_fixed.yaml'))
    assert drive(ctl, '1 2 3 4 4r 5 5r 6', results=UNTIL_FIXED)[0] == 'SSTTTSSS'
    assert len(ctl.interventions) == 1 and ctl.interventions[0].handed_back
    assert [x.turn for x in ctl.records if x.decision.kind == 'release'] == [5]


def test_reply_time_release_and_retry():
    @trigger(phases='reply')
    def fixed(ev, state):
        return Decision(reason='says fixed', kind='release') if 'FIXED' in ev.reply.text else None
    ctl = Controller(Policy(release=(fixed,)), [Q('A', at=[2])])
    assert ctl.observes_teacher
    assert drive(ctl, '1 2 3 4 5', teacher_replies={'2': 'working', '3': 'FIXED'})[0] == 'STTSS'
    ctl = Controller(Policy(release=(fixed,)), [Q('A', at=[2])])
    who, _ = drive(ctl, '1 2 3 3r 4 5', teacher_replies={'3': 'FIXED', '3r': 'still broken'})
    assert who == 'STTTTT'                           # the retry's reply replaced the one that said FIXED
    assert [(x.turn, x.outcome, x.why) for x in ctl.records if x.decision.kind == 'release'] == [
        (3, 'pending', None), (3, 'suppressed', 'dropped')]


def test_main_list_trigger_can_release_on_teacher_turns():
    @trigger(phases='request')
    def judge(ev, state):
        if ev.owner == 'student' and ev.turn == 2:
            return Decision(reason='stuck')
        if ev.owner == 'teacher' and ev.turn == 4:
            return Decision(reason='unstuck', kind='release')
        if ev.owner == 'student' and ev.turn == 6:
            return Decision(reason='nothing to release', kind='release')
    ctl = Controller(Policy(), [spec(judge, owners=('student', 'teacher'))])
    assert play(ctl, '1 2 3 4 5 6 7') == 'STTSSSS'
    assert [(x.turn, x.outcome, x.why, x.decision.kind) for x in ctl.records] == [
        (2, 'acted', None, 'intervene'), (4, 'acted', None, 'release'), (6, 'suppressed', 'owner', 'release')]


def test_shadow_release_never_hands_back():
    r = spec(Fake('R', 'request', every=1), log_only=True)
    ctl = Controller(Policy(release=(r,)), [Q('A', at=[2])])
    assert play(ctl, '1 2 3 4') == 'STTT'
    assert [(x.decision.trigger, x.turn, x.outcome) for x in ctl.records[1:]] == [('R', 3, 'shadow'),
                                                                                ('R', 4, 'shadow')]


# ---- per-intervention view and teacher, info ------------------------------------------------------------------
def test_view_and_teacher_resolution():
    def b(ev, state):
        if ev.turn == 4:
            return Decision(reason='b', span=1, view={'show_rejected_reply': True}, teacher='alt')

    def c(ev, state):
        if ev.turn == 6:
            return Decision(reason='c', span=1, view={'failed_turn_view': 'hide'}, teacher='x')
    a = spec(R('A', at=[2], span=1), view={'failed_turn_view': 'hide', 'show_rejected_reply': True}, teacher='big')
    ctl = Controller(Policy(student_view_after_handback='stripped'),
                     [a, b, spec(c, view={'failed_turn_view': 'keep'}, teacher='y')])
    who, rs = drive(ctl, '1 2 3 4 5 6 7')
    assert who == 'STSTSTS'
    assert (rs[1].why, rs[1].teacher, rs[1].show_rejected_reply, rs[1].view) == (
        'rewrite', 'big', True, View(True, 'hide', 'stripped'))
    assert (rs[3].teacher, rs[3].show_rejected_reply, rs[3].view) == ('alt', False, View(True, 'keep', 'stripped'))
    assert (rs[5].teacher, rs[5].view.failed_turn_view) == ('y', 'keep')     # the entry beats the decision
    assert (rs[0].view, rs[0].teacher) == (None, None)
    assert [I.teacher for I in ctl.interventions] == ['big', 'alt', 'y']
    assert ctl.interventions[0].as_dict()['view'] == dict(show_rejected_reply=True, failed_turn_view='hide',
                                                          student_view_after_handback='stripped')
    hash(ctl.interventions[0])                       # interventions stay hashable
    r = run(Controller(Policy(), [], initial_owner='teacher').on_request(Event('request', 1)))
    assert (r.who, r.teacher, r.view) == ('teacher', 'teacher', None)


def test_decision_info_and_kind_in_records():
    def f(ev, state):
        if ev.turn == 2:
            return Decision(reason='p high', info={'p_fail': 0.9, 'features': ['loop']})
    ctl = Controller(Policy(), [f])
    play(ctl, '1 2')
    d = ctl.records[0].as_dict()
    assert (d['info'], d['kind'], d['trigger']) == ({'p_fail': 0.9, 'features': ['loop']}, 'intervene', 'f')
    hash(Decision('A', 'x'))                         # decisions without dict fields stay hashable
    for bad in (dict(info='x'), dict(kind='stop'), dict(view={'failed_turn_view': 'blur'}), dict(teacher='')):
        with pytest.raises(ValueError):
            Decision('A', 'x', **bad)


# ---- examples -------------------------------------------------------------------------------------------------
def test_two_errors_example():
    cfg = {'triggers': [{'use': f'{EXAMPLES / "two_errors.py"}:two_errors', 'span': 1}]}
    ctl = Controller.from_config(cfg)
    assert ctl.specs[0].name == 'two_errors' and ctl.specs[0].phases == frozenset({'request'})
    who, _ = drive(ctl, '1 2 3 4 5', results={2: FAIL2, 3: FAIL2, 4: FAIL2, 5: OK})
    assert who == 'SSSTS'                            # two failures in a row at 3, but only after turn 3 counts
    assert ctl.records[0].decision.reason == '3 failed actions in a row, last: exit code 2'


# ---- stage 1 asks: done_claim first_only (the shadow config reproduces would_fire), public ScannerTrigger / judged
def _claim_scanner(prior_done):
    """The scanner after one Terminus-2 turn that did, or did not, claim the task done."""
    msgs = [dict(role='user', content='Task Description:\nfix it\n\nCurrent terminal state:\nroot@h:/app# '),
            dict(role='assistant', content=T2(cmds=['ls\n'], done=prior_done)),
            dict(role='user', content='New Terminal Output:\nroot@h:/app# ls\nrun.py\nroot@h:/app# ')]
    return iv._rt().scan_messages(msgs)


@pytest.mark.parametrize('first_only,prior,expect', [
    (False, False, 'task_complete: true'), (True, False, 'task_complete: true'),     # the episode's first claim
    (False, True, 'task_complete: true (student)'), (True, True, None),            # a later claim
])
def test_done_claim_first_only_with_scanner(first_only, prior, expect):
    sc = _claim_scanner(prior)
    d = run(iv.DoneClaim(first_only=first_only).observe(
        Event('reply', sc.turn + 1, scanner=sc, reply=Reply(T2(done=True)))))
    assert (d.reason if d else None) == expect


def test_done_claim_first_only_without_scanner_per_episode():
    shadow = {'triggers': [{'use': 'done_claim', 'params': {'first_only': True}, 'log_only': True,
                            'policy': {'earliest_turn': 1, 'owners': ['student', 'teacher']}}]}
    for _ in range(2):                                # the first claim is per episode
        ctl = Controller.from_config(shadow)
        play(ctl, '1 2 3 4', reply=Reply(T2(done=True)))
        assert [(x.turn, x.outcome) for x in ctl.records] == [(1, 'shadow')]
    every = Controller.from_config({'triggers': [{'use': 'done_claim', 'log_only': True,
                                                  'policy': {'earliest_turn': 1}}]})
    play(every, '1 2 3', reply=Reply(T2(done=True)))
    assert [x.turn for x in every.records] == [1, 2, 3]


def test_scanner_trigger_and_judged_are_public():
    assert iv.ScannerTrigger is iv._ScannerTrigger and iv.judged is iv._judged
    assert issubclass(iv.BUILTINS['loop'], iv.ScannerTrigger) and iv.BUILTINS['gave_up'].kind == 'decision'
    act = {'cmds': [], 'done': True}
    assert iv.judged(Event('reply', 1)) is None
    assert iv.judged(Event('reply', 1, reply=Reply('raw', action=act))) is act
    assert iv.judged(Event('reply', 1, reply=Reply('raw'))) == 'raw'
    assert iv.judged(Event('reply', 1, reply=Reply('raw', parse_error='no JSON'))) is None
