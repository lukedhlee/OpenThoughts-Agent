"""CPU tests for judge.py (the LLM judge trigger) against a local stub judge server: fires and threshold, verdicts,
timeouts, malformed answers, view hygiene (no teacher-only text, reasoning or grader data reaches the judge), every /
max_calls, prefetch overlap, endpoint names, the cache, the per-ask log, and the trigger inside the Controller.

    python -m pytest -q data/relay/router/tests/test_judge.py
"""
import asyncio
import json
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import interventions as iv  # noqa: E402
import judge as jg  # noqa: E402
import relay_router as rr  # noqa: E402
import router_triggers as rtg  # noqa: E402
from interventions import ActionResult, Controller, Event, Reply  # noqa: E402
from stub_judge import StubJudge, chat, prompt_of  # noqa: E402


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def fresh_cache():
    jg.clear_cache()
    jg.set_resolver(None)
    yield
    jg.clear_cache()
    jg.set_resolver(None)


@pytest.fixture
def registered():
    """`use: llm_judge` -> LlmJudge for one test, then interventions' stub again (other suites test the stub)."""
    old = jg.register()
    yield
    iv.BUILTINS['llm_judge'] = old


def answering(content, status=200):
    return lambda body: (status, chat(content))


def T2(analysis='a', cmds=(), done=False):
    return json.dumps(dict(analysis=analysis, plan='p', task_complete=done,
                           commands=[dict(keystrokes=k, duration=1.0) for k in cmds]))


def history(n, tag='x'):
    """A Terminus-2 history of n turns: the task, then n (reply, observation) pairs."""
    msgs = [{'role': 'user', 'content': f'Task: fix run.py ({tag})'}]
    for i in range(1, n + 1):
        msgs += [{'role': 'assistant', 'content': T2(cmds=[f'cmd{i}\n'])},
                 {'role': 'user', 'content': f'New Terminal Output:\n$ cmd{i}\nout{i}'}]
    return msgs


def req(turn, tag='x', **kw):
    return Event('request', turn, messages=history(turn - 1, tag), **kw)


def judge(url, **params):
    return jg.LlmJudge(endpoint=url, model='judge', **params)


# ---- answers ---------------------------------------------------------------------------------------------------
def test_fires_with_info_and_threshold():
    with StubJudge(answering('{"score": 0.9, "rationale": "it repeats the same failing command"}')) as stub:
        j = judge(stub.url, threshold=0.7)
        d = run(j.observe(req(3)))
    assert d is not None and d.trigger == 'llm_judge' and d.score == 0.9
    assert d.reason == 'judge 0.90 >= 0.7: it repeats the same failing command'
    info = d.info or {}
    assert (info['rationale'], info['score'], info['prompt_tokens'], info['completion_tokens'], info['cached'],
            info['model']) == ('it repeats the same failing command', 0.9, 123, 7, False, 'judge')
    assert info['latency_s'] > 0 and info['waited_s'] >= info['latency_s'] * 0.5
    body = stub.bodies[0]
    assert (body['model'], body['temperature'], body['max_tokens'], body['messages'][-1]['role']) == (
        'judge', 0.0, 512, 'user')
    assert j.stats['fired'] == 1 and j.stats['calls'] == 1
    for score, fires in ((0.69, False), (0.7, True)):
        with StubJudge(answering(json.dumps({'score': score, 'rationale': 'r'}))) as stub:
            assert (run(judge(stub.url, threshold=0.7).observe(req(3))) is not None) == fires


@pytest.mark.parametrize('content,fires', [
    ('{"verdict": "Intervene", "rationale": "stuck"}', True),
    ('{"verdict": "continue"}', False),
    ('STUCK.', True),
    ('<think>long thought {"verdict": "continue"}</think>```json\n{"verdict": "yes"}\n```', True),
])
def test_verdict_output(content, fires):
    with StubJudge(answering(content)) as stub:
        d = run(judge(stub.url, output='verdict').observe(req(3)))
    assert (d is not None) == fires
    if d is not None:
        assert d.score is None and (d.info or {})['verdict'] in ('intervene', 'stuck', 'yes')


@pytest.mark.parametrize('answer,outcome', [
    (answering('the agent looks fine to me'), 'malformed'),
    (answering('{"score": "high", "rationale": "x"}'), 'malformed'),
    (lambda body: (200, {'not': 'a chat completion'}), 'malformed'),
    (answering('{"score": 0.9}', status=500), 'error'),
])
def test_malformed_and_errors_are_no_decision(tmp_path, answer, outcome):
    log = tmp_path / 'calls.jsonl'
    with StubJudge(answer) as stub:
        j = judge(stub.url, log=str(log))
        assert run(j.observe(req(3))) is None
    assert j.stats[outcome] == 1
    line = json.loads(log.read_text())
    assert line['outcome'] == outcome and line['error']


def test_parse_answer():
    assert jg.parse_answer('{"score": 0.4, "rationale": "ok"}') == (0.4, 'ok', None)
    assert jg.parse_answer('Sure. {"score": "0.8"} trailing') == (0.8, '', None)
    assert jg.parse_answer('{"p": 1, "why": "w"}', score_key='p', rationale_key='why') == (1.0, 'w', None)
    assert jg.parse_answer('<|start_think|>{"score": 0.1}<|end_think|>{"score": 0.6}')[0] == 0.6
    assert jg.parse_answer('nothing', 'score')[2] and jg.parse_answer('', 'verdict')[2]


def test_timeout_is_no_decision_and_never_cached(tmp_path):
    log = tmp_path / 'calls.jsonl'
    with StubJudge(answering('{"score": 0.9}'), delay=1.0) as stub:
        j = judge(stub.url, timeout_s=0.2, log=str(log))
        t0 = time.monotonic()
        assert run(j.observe(req(3))) is None
        assert time.monotonic() - t0 < 0.9
        assert run(judge(stub.url, timeout_s=0.2).observe(req(3))) is None   # a new episode asks again
        n = len(stub.bodies)
    assert j.stats['timeout'] == 1 and n == 2
    line = json.loads(log.read_text())
    assert line['outcome'] == 'timeout' and 0.15 < line['latency_s'] < 0.9

    async def slow_call(name, body):            # the router's retries have far longer timeouts than a judge's
        await asyncio.sleep(1.0)
        return chat('{"score": 0.9}')
    j = judge('judge', timeout_s=0.2)
    t0 = time.monotonic()
    assert run(j.observe(req(3, extra={'call_endpoint': slow_call}))) is None
    assert time.monotonic() - t0 < 0.9 and j.stats['timeout'] == 1


# ---- view hygiene ----------------------------------------------------------------------------------------------
GRADER = 'def test_hidden(): assert open("/app/out.txt").read() == "SECRET-ANSWER"'
HINT = 'Teacher hint: work economically and verify before you submit.'


def leaky_event(phase='request'):
    """An event whose history carries everything a judge must not see: the student's thinking (stripped by
    default), the teacher's reasoning fields, the verify note and a teacher hint appended as if the router had passed
    a teacher-bound body, and grader text / task metadata in Event.extra."""
    msgs = [{'role': 'system', 'content': 'You are a terminal agent.\n\n' + HINT},
            {'role': 'user', 'content': 'Task: make /app/out.txt contain ok\n\n' + HINT},
            {'role': 'assistant', 'content': '<|start_think|>STUDENT-THINKING-42<|end_think|>' + T2(cmds=['ls\n'])},
            {'role': 'user', 'content': 'New Terminal Output:\n$ ls\nrun.py'},
            {'role': 'assistant', 'content': T2('teacher visible analysis', ['python run.py\n']),
             'reasoning': 'TEACHER-REASONING-7', 'reasoning_content': 'TEACHER-REASONING-7'},
            {'role': 'user', 'content': 'New Terminal Output:\n$ python run.py\nboom\n\n' + rr.VERIFY_NOTE}]
    reply = Reply('<think>STUDENT-THINKING-43</think>' + T2('propose rm', ['rm -rf /app\n']),
                  action={'cmds': [('rm -rf /app\n', 1.0)], 'done': False}) if phase == 'reply' else None
    return Event(phase, 3, messages=msgs, reply=reply,
                 last_result=ActionResult('python run.py', 1, output='boom\n\n' + rr.VERIFY_NOTE_OWN),
                 extra={'teacher_only': [HINT], 'tests': GRADER, 'task_id': 'secret-task-id', 'solution': 'echo ok'})


@pytest.mark.parametrize('phase', ['request', 'reply'])
def test_view_hygiene(phase):
    with StubJudge(answering('{"score": 0.1}')) as stub:
        run(judge(stub.url).observe(leaky_event(phase)))
        sent = json.dumps(stub.bodies[0])
    for secret in ('STUDENT-THINKING', 'TEACHER-REASONING', rr.VERIFY_NOTE[:40], rr.VERIFY_NOTE_OWN[:40],
                   'Teacher hint', 'SECRET-ANSWER', 'test_hidden', 'secret-task-id', 'echo ok', 'reasoning'):
        assert secret not in sent, secret
    for seen in ('make /app/out.txt contain ok', 'teacher visible analysis', 'boom', '$ python run.py, exit code 1'):
        assert seen in prompt_of(stub.bodies[0]), seen
    assert ('rm -rf /app' in sent) == (phase == 'reply')


def test_think_keep_shows_the_students_thinking_never_the_teachers_reasoning():
    p = judge('http://unused', think='keep').render(leaky_event())
    assert 'STUDENT-THINKING-42' in p and 'TEACHER-REASONING' not in p and rr.VERIFY_NOTE[:40] not in p


def test_window_and_template_fields():
    j = judge('http://unused', window=2, prompt='T=$task|H=$history|R=$last_result|P=$proposal|$turn $phase|{"json"}')
    p = j.render(Event('request', 6, messages=history(5), last_result=ActionResult('cmd5', 0, output='out5')))
    assert p.startswith('T=Task: fix run.py (x)|H=[assistant]') and p.endswith('|6 request|{"json"}')
    assert 'cmd4' in p and 'cmd5' in p and 'cmd3' not in p          # the last two turns only
    assert 'R=$ cmd5, exit code 0\nout5|P=|' in p


# ---- cost bounds -----------------------------------------------------------------------------------------------
def test_every_max_calls_and_retries(tmp_path):
    with StubJudge(answering('{"score": 0.1}')) as stub:
        j = judge(stub.url, every=2, max_calls=2)

        async def go():
            for t in range(1, 9):
                await j.observe(req(t))
            await j.observe(req(4))           # a retry of turn 4: the same request, the same answer
        run(go())
    asked = sorted(int(prompt_of(b).split('cmd')[-1].split()[0]) + 1 for b in stub.bodies)
    assert asked == [2, 4]
    assert (j.stats['skipped_every'], j.stats['skipped_max_calls'], j.stats['repeats'], j.asks) == (4, 2, 1, 2)


def test_same_prompt_on_another_turn_is_a_new_ask(tmp_path):
    log = tmp_path / 'calls.jsonl'
    with StubJudge(answering('{"score": 0.1}')) as stub:
        j = judge(stub.url, prompt='LAST: $last_result', log=str(log))    # the same text on every turn

        async def go():
            for t in (2, 3, 3):                 # turn 3 twice: its retry
                await j.observe(Event('request', t, last_result=ActionResult('ls', 0, output='a')))
        run(go())
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert [(x['turn'], x['cached']) for x in lines] == [(2, False), (3, True)] and len(stub.bodies) == 1
    assert (j.asks, j.stats['repeats']) == (2, 1)


def test_cache_one_call_for_identical_prompts():
    with StubJudge(answering('{"score": 0.9}')) as stub:
        a, b = judge(stub.url), judge(stub.url)      # e.g. replay's live and shadow controllers
        da, db = run(a.observe(req(3))), run(b.observe(req(3)))
        assert len(stub.bodies) == 1
        dc = run(judge(stub.url, cache=False).observe(req(3)))
        assert len(stub.bodies) == 2
    assert da and db and dc and ((da.info or {})['cached'], (db.info or {})['cached']) == (False, True)


def test_prefetch_overlaps_tool_time():
    with StubJudge(answering('{"score": 0.9}'), delay=0.3) as stub:
        j = judge(stub.url)

        async def go():
            j.prefetch(req(3))                # e.g. at finish() of turn 2
            await asyncio.sleep(0.25)         # the tool runs meanwhile
            return await j.observe(req(3))
        d = run(go())
    info = (d.info or {}) if d else {}
    assert d and info['prefetched'] and info['latency_s'] >= 0.3 and info['waited_s'] < 0.2 and len(stub.bodies) == 1


# ---- endpoint names --------------------------------------------------------------------------------------------
def test_named_endpoint_through_call_endpoint(tmp_path):
    calls = []

    async def call_endpoint(name, body):         # the router's: it sets its endpoint's served model name
        calls.append((name, body.get('model'), body['messages'][-1]['role']))
        return dict(chat('{"score": 0.95, "rationale": "r"}'), model='served-judge')
    log = tmp_path / 'calls.jsonl'
    j = jg.LlmJudge(endpoint='judge', log=str(log))                 # no model: the router's
    d = run(j.observe(req(3, extra={'call_endpoint': call_endpoint})))
    assert d and d.score == 0.95 and calls == [('judge', None, 'user')]
    assert (d.info or {})['model'] == 'served-judge' and json.loads(log.read_text())['model'] == 'served-judge'
    with StubJudge(answering('{"score": 0.9}')) as stub:      # a URL is posted to directly, even online
        assert run(judge(stub.url).observe(req(3, tag='url', extra={'call_endpoint': call_endpoint})))
        assert len(stub.bodies) == 1 and len(calls) == 1


@pytest.mark.parametrize('exc,error', [
    (rtg.EndpointError('judge', 503, 'overloaded'), 'HTTP 503: overloaded'),
    (rtg.EndpointError('judge', None, 'no endpoint'), 'HTTP unreachable: no endpoint'),
    (ValueError('bad body'), "ValueError('bad body')"),
])
def test_endpoint_errors_are_no_decision(tmp_path, exc, error):
    async def failing(name, body):
        raise exc
    log = tmp_path / 'calls.jsonl'
    j = jg.LlmJudge(endpoint='judge', log=str(log))
    assert run(j.observe(req(3, extra={'call_endpoint': failing}))) is None and j.stats['error'] == 1
    assert json.loads(log.read_text())['error'] == error


def test_endpoint_names_offline():
    with StubJudge(answering('{"score": 0.9}')) as stub:
        jg.set_resolver(lambda name: stub.url if name == 'judge' else None)
        assert run(judge('judge').observe(req(3, tag='resolver')))
        jg.set_resolver(None)
        assert len(stub.bodies) == 1
    j = jg.LlmJudge(endpoint='nowhere')                        # replay has no call_endpoint
    assert run(j.observe(req(3, tag='none'))) is None and j.stats['unresolved'] == 1 and j.stats['calls'] == 0


def test_endpoint_env_expansion(monkeypatch):
    monkeypatch.setenv('JUDGE_URL', 'http://example.invalid:9/v1')
    assert judge('${JUDGE_URL}').endpoint == 'http://example.invalid:9/v1'


# ---- inside the Controller, from YAML --------------------------------------------------------------------------
def test_reply_time_rewrite_from_config(tmp_path, registered):
    with StubJudge(lambda body: (200, chat(json.dumps({'score': 0.9 if 'rm -rf' in prompt_of(body) else 0.1,
                                                       'rationale': 'destructive'})))) as stub:
        cfg = {'triggers': [{'use': 'llm_judge', 'phases': ['reply'], 'where': 'rewrite', 'span': 1,
                             'params': {'endpoint': stub.url, 'model': 'judge', 'threshold': 0.5,
                                        'log': str(tmp_path / 'calls.jsonl')}}]}
        ctl = Controller.from_config(cfg)

        async def go():
            who = ''
            for t in (1, 2, 3, 4):
                r = await ctl.on_request(req(t))
                if r.who == 'student':
                    cmd = 'rm -rf /app\n' if t == 3 else f'cmd{t}\n'
                    r = await ctl.on_reply(Event('reply', t, messages=history(t - 1),
                                                 reply=Reply(T2(cmds=[cmd]), action={'cmds': [(cmd, 1.0)]})))
                who += r.who[0].upper()
            return who
        assert run(go()) == 'SSTS'
    I = ctl.interventions[0]
    assert (I.trigger, I.where, I.start_turn) == ('llm_judge', 'rewrite', 3) and 'destructive' in I.reason
    lines = [json.loads(x) for x in (tmp_path / 'calls.jsonl').read_text().splitlines()]
    assert [x['outcome'] for x in lines] == ['quiet', 'quiet', 'fired', 'quiet']   # turn 1 too: earliest is policy's
    assert all(x['latency_s'] is not None and x['prompt_tokens'] == 123 for x in lines)


def test_config_errors(registered):
    for params, match in (({'endpoint': 'http://x'}, 'model'), ({'endpoint': '', 'model': 'm'}, 'endpoint'),
                          ({'endpoint': 'http://x', 'model': 'm', 'output': 'prob'}, 'output'),
                          ({'endpoint': 'http://x', 'model': 'm', 'prompt': 'a', 'prompt_file': 'b'}, 'not both'),
                          ({'endpoint': 'http://x', 'model': 'm', 'every': 0}, 'every'),
                          ({'endpoint': 'http://x', 'model': 'm', 'prompt_file': '/no/such/file'}, 'prompt_file')):
        with pytest.raises(iv.ConfigError, match=match):
            iv.load_config({'triggers': [{'use': 'judge:LlmJudge', 'params': params}]})
    _, specs = iv.load_config({'triggers': [{'use': 'llm_judge', 'phases': ['request', 'reply'],
                                             'params': {'endpoint': 'judge'}}]})      # a name: no model needed
    assert isinstance(specs[0].trigger, jg.LlmJudge) and specs[0].phases == frozenset(iv.PHASES)
    assert 'model' not in specs[0].trigger.body('p')


def test_registration_is_explicit():
    with pytest.raises(iv.ConfigError, match='stage 4'):     # importing judge.py changed nothing
        iv.load_config({'triggers': [{'use': 'llm_judge', 'params': {'endpoint': 'judge', 'model': 'm'}}]})
    old = jg.register()
    try:
        assert iv.BUILTINS['llm_judge'] is jg.LlmJudge and old is not jg.LlmJudge
    finally:
        iv.BUILTINS['llm_judge'] = old


def test_example_config_loads(monkeypatch):
    monkeypatch.setenv('JUDGE_URL', 'http://localhost:8000/v1')
    examples = HERE.parent.parent / 'triggers' / 'examples'
    policy, specs = iv.load_config(str(examples / 'judge.yaml'))
    t = specs[0].trigger
    assert isinstance(t, jg.LlmJudge) and t.endpoint == 'http://localhost:8000/v1' and specs[0].log_only
    assert '$history' in t.prompt.template and t.render(req(3))
