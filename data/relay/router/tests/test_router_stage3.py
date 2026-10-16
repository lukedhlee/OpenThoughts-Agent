"""CPU tests for stage 3 of notes/ot-agent/relay-triggers: under --trigger-config every knob of the interface acts in
the live router. k-turn spans and their hand-back, several interventions with cooldown and cap, summarization and
/tokenize by the current owner, the views (the student's view after a hand-back, the failed turn hidden, the rejected
reply shown, a resumed student keeps its thinking), the teacher note on any intervention, named endpoints (a second
teacher, a trigger's own calls), the student's clock without the teacher's turns, and the records (`intervention` on
every teacher turn, `decisions`, intervention / handback events). Router-level, no harbor agent needed.

    python -m pytest -q data/relay/router/tests/test_router_stage3.py
"""
import asyncio
import json
import socket
import sys
import threading
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import fake_openai  # noqa: E402
import relay_router as rr  # noqa: E402
from test_relay_router import Stack as T2Stack, _post, instruction  # noqa: E402
from test_relay_router_msa import Driver, MsaServer, Stack, qwen_xml  # noqa: E402

LOOK = ['Look.\n\n' + qwen_xml('ls'), 'More.\n\n' + qwen_xml('pwd'), 'Again.\n\n' + qwen_xml('whoami')]


def config(tmp, d):
    p = Path(tmp) / 'trig.yaml'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(d))
    return ['--trigger-config', str(p)]


def episode(tmp, cfg, n=6, sc='work', outputs=None, script=LOOK * 3, args=(), setup=None):
    """One msa episode under the config: (rows, events, the stack's teacher and student request bodies)."""
    with Stack(tmp, router_args=config(tmp, cfg) + list(args)) as st:
        st.teacher.script = list(script)
        if setup:
            setup(st)
        Driver(st, sc, outputs=outputs).run(n)
        ev = [json.loads(x) for x in (st.log_dir / 'events.jsonl').read_text().splitlines()]
        return st.turns(f'sid-{sc}'), ev, [b for b in st.teacher.requests if b.get('tools')], \
            [b for b in st.student.requests if b.get('tools')]


def owners(rows):
    return ''.join('T' if r['owner'] == 'teacher' else 'S' if r['owner'] == 'student' else 'R' for r in rows)


def iv_of(rows):
    return [(r['intervention']['id'], r['intervention']['i']) if r.get('intervention') else None for r in rows]


# ---- spans and hand-back ------------------------------------------------------------------------------------
def test_request_time_span_two_hands_back(tmp_path):
    rows, ev, _, _ = episode(tmp_path, {'triggers': [{'use': 'scripted', 'params': {'turns': [3]}, 'span': 2}]})
    assert owners(rows) == 'SSTTSS'
    assert iv_of(rows) == [None, None, (1, 1), (1, 2), None, None]
    assert rows[2]['intervention'] == dict(id=1, trigger='scripted', where='next', span=2, i=1, teacher='teacher',
                                           start_turn=3, view=dict(show_rejected_reply=False, failed_turn_view='keep',
                                                                   student_view_after_handback='capped'))
    assert [(e['event'], e['turn']) for e in ev if e['event'] in ('intervention', 'handback')] == [
        ('intervention', 3), ('handback', 5)]
    assert not any(r.get('takeover') for r in rows)          # a span is not a takeover


def test_reply_time_rewrite_span_two_and_next_span_one(tmp_path):
    rows, _, _, _ = episode(tmp_path / 'rw', {'triggers': [{'use': 'scripted', 'params': {'turns': [3]},
                                                            'phases': ['reply'], 'where': 'rewrite', 'span': 2}]})
    assert owners(rows) == 'SSTTSS' and iv_of(rows)[2:4] == [(1, 1), (1, 2)]
    assert rows[2]['discarded_student_reply'] and rows[2]['intervention']['where'] == 'rewrite'
    rows, _, _, _ = episode(tmp_path / 'nx', {'triggers': [{'use': 'scripted', 'params': {'turns': [3]},
                                                            'phases': ['reply'], 'where': 'next', 'span': 1}]})
    assert owners(rows) == 'SSSTSS' and rows[3]['intervention']['start_turn'] == 4
    pend = [d for d in rows[2]['decisions'] if d['outcome'] == 'pending']
    assert pend and pend[0]['trigger'] == 'scripted'


def test_several_interventions_with_cooldown_and_cap(tmp_path):
    cfg = {'policy': {'cooldown': 1, 'max_interventions': 2},
           'triggers': [{'use': 'every', 'params': {'n': 1, 'start': 2}, 'phases': ['request'], 'span': 1}]}
    rows, ev, _, _ = episode(tmp_path, cfg, n=7)
    assert owners(rows) == 'STSTSSS'
    assert [r['intervention']['id'] for r in rows if r.get('intervention')] == [1, 2]
    why = [d['why'] for r in rows for d in r['decisions'] if d['outcome'] == 'suppressed']
    assert why == ['cooldown', 'cap', 'cap', 'cap']
    assert [e['id'] for e in ev if e['event'] == 'handback'] == [1, 2]


def test_released_takeover_keeps_the_resumed_students_thinking(tmp_path, monkeypatch):
    """--student-view-after-takeover strip strips the student's thinking only while the teacher holds the episode."""
    import test_relay_router_msa as m
    codes = {'echo step-1': 1}
    monkeypatch.setattr(m, 'observation', lambda cmd, out='', rc=0: json.dumps(
        {'returncode': codes.get(cmd, 0), 'output': out or f'{cmd} done'}, indent=2))
    rows, ev, _, sreq = episode(tmp_path, {'triggers': [{'use': 'action_error', 'release': {'use': 'action_ok'}}]},
                                args=['--student-view-after-takeover', 'strip'])
    assert owners(rows) == 'SSTSSS' and rows[2]['takeover']['trigger'] == 'action_error'
    resumed = sreq[2]['messages']                            # the student's first request after the hand-back
    own = [x for x in resumed if x['role'] == 'assistant' and 'student thinking' in (x.get('reasoning_content') or '')]
    assert len(own) == 2


# ---- the views -------------------------------------------------------------------------------------------------
def test_student_view_after_handback_stripped_and_capped(tmp_path):
    base = {'use': 'scripted', 'params': {'turns': [3]}, 'span': 1}
    _, _, _, sreq = episode(tmp_path / 'c', {'triggers': [base]})
    tt = [x for x in sreq[2]['messages'] if x['role'] == 'assistant' and 'teacher-step' in (x.get('reasoning') or '')]
    assert len(tt) == 1                                      # capped (default): the latest teacher turn keeps it all
    _, _, _, sreq = episode(tmp_path / 's', {'triggers': [dict(base, view={'student_view_after_handback': 'stripped'})]})
    t3 = [x for x in sreq[2]['messages'] if x['role'] == 'assistant' and x['tool_calls'][0]['id'] == 'call_3_0']
    assert t3 and not t3[0].get('reasoning') and not t3[0].get('reasoning_content')


def test_full_view_keeps_every_teacher_turn_whole(tmp_path):
    from test_relay_router import _word_tokenizer
    tok = _word_tokenizer(tmp_path / 'tok.json')
    args = ['--student-tokenizer', tok, '--reasoning-cap', '20']
    trig = {'use': 'scripted', 'params': {'turns': [3, 5]}, 'span': 1}

    def longer(st):
        st.teacher.long_reasoning = 30
    for view, cut in (('capped', True), ('full', False)):
        _, _, _, sreq = episode(tmp_path / view, {'triggers': [dict(trig, view={'student_view_after_handback': view})]},
                                n=7, args=args, setup=longer)
        tt = [x for x in sreq[-1]['messages'] if x['role'] == 'assistant'
              and 'teacher-step' in (x.get('reasoning_content') or '')]
        assert len(tt) == 2 and (len(tt[0]['reasoning_content'].split()) <= 20) == cut


def test_failed_turn_hidden_from_the_teacher(tmp_path):
    trig = {'use': 'scripted', 'params': {'turns': [3]}, 'span': 2}
    for view, seen in (('keep', True), ('hide', False)):
        _, _, treq, _ = episode(tmp_path / view, {'triggers': [dict(trig, view={'failed_turn_view': view})]})
        for b in treq[:2]:                                   # both teacher turns of the intervention
            cmds = json.dumps([x.get('tool_calls') for x in b['messages'] if x['role'] == 'assistant'])
            assert ('echo step-1' in cmds) == seen and 'echo step-0' in cmds
            assert b['messages'][-1]['role'] == 'tool'


def test_rejected_reply_shown_to_the_teacher(tmp_path):
    trig = {'use': 'scripted', 'params': {'turns': [3]}, 'phases': ['reply'], 'where': 'rewrite', 'span': 1,
            'view': {'show_rejected_reply': True}}
    rows, _, treq, _ = episode(tmp_path, {'triggers': [trig]})
    last = treq[0]['messages'][-1]['content']
    assert rr.REJECTED_NOTE in last and 'echo step-2' in last and 'student thinking' not in last
    assert rr.REJECTED_NOTE not in json.dumps(rows[3].get('response'))   # teacher-only, never in harbor's history


def test_teacher_note_on_the_first_teacher_turn_only(tmp_path):
    trig = {'use': 'scripted', 'params': {'turns': [3]}, 'span': 2, 'teacher_note': 'NOTE-X: check the logs first.'}
    rows, _, treq, sreq = episode(tmp_path, {'triggers': [trig]})
    assert treq[0]['messages'][-1]['content'].endswith('\n\nNOTE-X: check the logs first.')
    assert 'NOTE-X' not in json.dumps(treq[1]['messages']) and 'NOTE-X' not in json.dumps(sreq)
    assert rows[2].get('teacher_note') and not rows[3].get('teacher_note') and not rows[2].get('verify_note')


# ---- named endpoints -------------------------------------------------------------------------------------------
class Server:
    """A fake server on its own loop, up before the router starts (so its health check reaches it)."""

    def __init__(self, srv):
        self.srv, self.loop = srv, asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        s = socket.socket()
        s.bind(('127.0.0.1', 0))
        self.port = s.getsockname()[1]
        s.close()
        self.url = asyncio.run_coroutine_threadsafe(srv.start(port=self.port), self.loop).result(10)

    def close(self):
        asyncio.run_coroutine_threadsafe(self.srv.stop(), self.loop).result(10)
        self.loop.call_soon_threadsafe(self.loop.stop)


def test_second_teacher_endpoint_and_health(tmp_path):
    t2 = Server(MsaServer('teacher', 'qwen38'))
    t2.srv.script = list(LOOK)
    try:
        cfg = {'endpoints': {'teacher2': {'url': t2.url, 'model': 'qwen38'}},
               'triggers': [{'use': 'scripted', 'params': {'turns': [3]}, 'span': 2, 'teacher': 'teacher2'}]}
        with Stack(tmp_path, router_args=config(tmp_path, cfg)) as st:
            assert st.health_ok and 'teacher2' in st.health['endpoints']
            Driver(st, 'work').run(6)
            rows = st.turns('sid-work')
            default_turns = [b for b in st.teacher.requests if b.get('tools')]
        mine = [b for b in t2.srv.requests if b.get('tools')]
        assert owners(rows) == 'SSTTSS' and len(mine) == 2 and not default_turns
        assert {r['intervention']['teacher'] for r in rows if r.get('intervention')} == {'teacher2'}
    finally:
        t2.close()


JUDGE = '''
from interventions import Decision


class AskJudge:
    name, phases = 'ask_judge', frozenset({'request'})

    async def observe(self, ev):
        if ev.turn != 3:
            return None
        resp = await ev.extra['call_endpoint']('judge', {'messages': [{'role': 'user', 'content': 'Is it stuck?'}]})
        return Decision('', 'judge: ' + resp['choices'][0]['message']['content'][:20], span=1)
'''


def test_trigger_calls_a_named_endpoint(tmp_path):
    judge = Server(MsaServer('teacher', 'qwen38'))
    try:
        (tmp_path / 'judge_trigger.py').write_text(JUDGE)
        cfg = {'endpoints': {'judge': judge.url}, 'triggers': [{'use': 'judge_trigger.py:AskJudge'}]}
        rows, _, _, _ = episode(tmp_path, cfg)
        assert owners(rows) == 'SSTSSS' and rows[2]['intervention']['trigger'] == 'ask_judge'
        call = rows[2]['endpoint_calls']
        assert [(c['name'], c['status']) for c in call] == [('judge', 200)] and len(judge.srv.requests) >= 2
        assert rows[2]['decisions'][0]['reason'].startswith('judge: ')
    finally:
        judge.close()


# ---- the clock ---------------------------------------------------------------------------------------------------
def test_student_clock_leaves_out_the_teachers_turns(tmp_path):
    """The teacher's two turns take 0.3 s each in the sandbox: the time until the next request is the teacher's."""
    with Stack(tmp_path, router_args=config(tmp_path, {'triggers': [
            {'use': 'scripted', 'params': {'turns': [3]}, 'span': 2}]})) as st:
        st.teacher.script = list(LOOK)
        d = Driver(st, 'work')
        for _ in range(5):
            d.step()
            if st.turns('sid-work')[-1]['owner'] == 'teacher':
                time.sleep(0.3)                              # its command running
        rows = st.turns('sid-work')
    r = rows[4]                                              # the student's first request after the hand-back
    assert owners(rows) == 'SSTTS' and rows[2]['teacher_clock_sec'] == 0.0
    assert r['teacher_clock_sec'] >= 0.55
    assert r['student_clock_sec'] <= r['elapsed_sec'] - 0.55


# ---- Terminus-2: summarization and /tokenize go by the current owner --------------------------------------------
def test_t2_summarization_and_tokenize_route_by_the_current_owner(tmp_path):
    p = tmp_path / 'trig.yaml'
    p.write_text(yaml.safe_dump({'triggers': [{'use': 'scripted', 'params': {'turns': [2]}, 'span': 2}]}))
    with T2Stack(tmp_path, router_args=['--trigger-config', str(p)]) as st:
        h = {rr.SESSION_HEADER: 's1'}
        msgs = [{'role': 'user', 'content': instruction('work') + '\n\nCurrent terminal state:\nroot@h:/app# '}]

        def ask():
            status, resp = _post(st.url + '/chat/completions', {'model': 'relay', 'messages': msgs}, h)
            assert status == 200
            return resp['choices'][0]['message']['content']
        msgs += [{'role': 'assistant', 'content': ask()}, {'role': 'user', 'content': 'New Terminal Output:\n\n$ '}]
        msgs += [{'role': 'assistant', 'content': ask()}, {'role': 'user', 'content': 'New Terminal Output:\n\n$ '}]
        _, who = st.router.owner_for_tokenize(msgs)          # the span has a turn left: the teacher's
        summ = [{'role': 'user', 'content': fake_openai.SUMMARY_PREFIX + ' Summarize.'}]
        status, _ = _post(st.url + '/chat/completions', {'model': 'relay', 'messages': summ}, h)
        rows = st.turns('s1')
    assert status == 200 and who == 'teacher'
    assert [r['owner'] for r in rows] == ['student', 'teacher', 'teacher']
    assert rows[2]['turn'] is None and rows[2]['intervention']['id'] == 1
