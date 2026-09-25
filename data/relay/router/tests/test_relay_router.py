"""CPU tests for the relay router: real Terminus-2 (harbor lukedhlee/terminus2-relay) against scripted fake servers.

The student and teacher are fake_openai.FakeServer instances; the router is relay_router.Router; the agent is harbor's
own Terminus2 (real prompt template, JSON parser, confirmation turn, Chat, LiteLLM, summarization, the session
header) with a fake tmux session standing in for the sandbox. Servers and router run on their own event loop in a
background thread, so the agent's blocking /v1/models and /tokenize probes reach them as they would on the cluster.

    PYTHONPATH=<harbor lukedhlee/terminus2-relay>/src <harbor venv>/bin/python -m pytest data/relay/router/tests -q

Without harbor on the path only the router-level tests run.
"""
import asyncio
import gzip
import json
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import jinja2
import pytest
from aiohttp import web

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import fake_openai  # noqa: E402
import relay_router as rr  # noqa: E402

try:
    import inspect

    from harbor.agents.terminus_2.terminus_2 import Terminus2
    from harbor.models.agent.context import AgentContext
    HAVE_HARBOR = 'llm_session_header' in inspect.signature(Terminus2.__init__).parameters
except Exception:  # noqa: BLE001
    HAVE_HARBOR = False
needs_harbor = pytest.mark.skipif(not HAVE_HARBOR, reason='harbor lukedhlee/terminus2-relay (llm_session_header) not on PYTHONPATH')

MODEL_INFO = {'max_input_tokens': 32768, 'max_output_tokens': 8192, 'input_cost_per_token': 0.0,
              'output_cost_per_token': 0.0}


def instruction(scenario, tag=''):
    return f'SCENARIO={scenario}{tag}. Fix the program in /app so that /app/out.txt contains the word ok.'


class Stack:
    """Fake student + fake teacher + router on a background event loop."""

    def __init__(self, tmp, mode='relay', router_args=(), teacher_key='reasoning', student_delay=0.0,
                 budgets=None, teacher_model='qwen38', strict=True, skip_health=False, teacher_delay=0.0,
                 two_teachers=False):
        self.tmp = Path(tmp)
        self.mode, self.router_args = mode, list(router_args)
        self.student = fake_openai.FakeServer('student', 'snowball', delay=student_delay)
        self.teacher = fake_openai.FakeServer('teacher', 'qwen38', reasoning_key=teacher_key, delay=teacher_delay)
        self.teacher2 = fake_openai.FakeServer('teacher', 'qwen38', delay=teacher_delay) if two_teachers else None
        self.teacher_model, self.strict, self.skip_health = teacher_model, strict, skip_health
        self.budgets = budgets or {}
        self.log_dir = self.tmp / 'router'
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)

    def run(self, coro, timeout=60):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def __enter__(self):
        self.thread.start()
        self.run(self._start())
        return self

    def __exit__(self, *exc):
        self.run(self._stop())
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(5)

    async def _start(self):
        s_url = await self.student.start()
        t_url = await self.teacher.start()
        if self.teacher2:
            t_url += ',' + await self.teacher2.start()
        tasks = [dict(task_id=f'cf-{sc}', instruction=instruction(sc, tag), agent_timeout_sec=self.budgets.get(sc, 1000))
                 for sc in ('selfdone', 'done', 'loop', 'wait', 'budget', 'summ', 'gaveup', 'repair', 'repairs')
                 for tag in [''] + [f'-{i}' for i in range(8)]]
        tf = self.tmp / 'tasks.json'
        tf.write_text(json.dumps(tasks))
        args = ['--mode', self.mode, '--port', '0', '--log-dir', str(self.log_dir), '--tasks', str(tf),
                '--student-url', s_url, '--student-model', 'snowball', '--teacher-url', t_url,
                '--teacher-model', self.teacher_model, '--health-retries', '1', '--health-wait', '0',
                '--connect-retries', '0'] + self.router_args
        if not self.strict:
            args.append('--no-strict-smoke')
        self.a = rr.parse_args(args)
        self.router = rr.Router(self.a)
        await self.router.start_http()
        self.health_ok, self.health = (True, None) if self.skip_health else await self.router.health_check()
        self.runner = web.AppRunner(rr.build_app(self.router), access_log=None)
        await self.runner.setup()
        site = web.TCPSite(self.runner, '127.0.0.1', 0)
        await site.start()
        self.url = f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/v1'

    async def _stop(self):
        await self.runner.cleanup()
        await self.router.http.close()
        await self.student.stop()
        await self.teacher.stop()
        if self.teacher2:
            await self.teacher2.stop()

    def turns(self, sid=None):
        p = self.log_dir / 'turns.jsonl'
        rows = []
        for l in (p.read_text().splitlines() if p.exists() else []):
            try:
                rows.append(json.loads(l))
            except json.JSONDecodeError:   # a concurrent episode's line being written
                pass
        return [r for r in rows if sid is None or r['sid'] == sid]

    def teacher_bodies(self, first_contains):
        return [b for b in self.teacher.requests
                if first_contains in fake_openai.text_of(b['messages'][0].get('content'))]


class FakeTerm:
    """Stands in for Terminus-2's tmux session: commands 'run' instantly and print scripted output."""

    OUT = {'python run.py': 'Traceback (most recent call last):\n  File "/app/run.py", line 1, in <module>\n'
                            "NameError: name 'x' is not defined",
           'make build': 'building target all ...', 'ls -la': 'total 8\n-rw-r--r-- 1 root root 12 run.py'}

    def __init__(self):
        self.files = {}
        self.sent = []

    async def is_session_alive(self):
        return True

    async def get_incremental_output(self):
        return 'Current Terminal Screen:\nroot@box:/app# '

    async def capture_pane(self, capture_entire=False):
        return 'root@box:/app# '

    def _run(self, k):
        if k.startswith('echo ') and '>' in k:
            text, _, path = k[5:].partition('>')
            self.files[path.strip()] = text.strip()
            return ''
        if k.startswith('echo '):
            return k[5:]
        if k.startswith('cat '):
            return self.files.get(k[4:].strip(), f'cat: {k[4:].strip()}: No such file or directory')
        return self.OUT.get(k, '')

    async def send_keys_and_capture(self, batches):
        lines = []
        for b in batches:
            k = b.keystrokes.strip()
            self.sent.append(k)
            if not k:
                continue
            lines.append(f'root@box:/app# {k}')
            out = self._run(k)
            if out:
                lines.append(out)
        if not lines:
            return 'New Terminal Output:\n', True
        return 'New Terminal Output:\n\n' + '\n'.join(lines + ['root@box:/app# ']), True


def run_agent(stack, scenario, tmp, interleaved=True, tag='', max_turns=40, **agent_kwargs):
    logs = Path(tmp) / f'agent_{scenario}{tag}'
    logs.mkdir(parents=True, exist_ok=True)
    agent = Terminus2(logs_dir=logs, model_name='hosted_vllm/relay', api_base=stack.url, model_info=MODEL_INFO,
                      extra_body={'skip_special_tokens': False}, interleaved_thinking=interleaved,
                      llm_session_header='X-Harbor-Session-Id', trajectory_config={'raw_content': True},
                      max_turns=max_turns, enable_episode_logging=False, **agent_kwargs)
    term = FakeTerm()
    agent._session = term
    ctx = AgentContext()

    async def go():
        try:
            await agent.run(instruction(scenario, tag), environment=SimpleNamespace(), context=ctx)
        except Exception as e:  # noqa: BLE001 - the tests inspect the stop reason / exception
            return e
        return None

    exc = asyncio.run(go())
    tp = logs / 'trajectory.json'
    import time
    for _ in range(100):          # harbor writes the final trajectory from a background writer (slow on GPFS)
        if tp.exists():
            break
        time.sleep(0.1)
    assert tp.exists(), f'no trajectory for {scenario}{tag}: agent raised {exc!r}; files {sorted(p.name for p in logs.iterdir())}'
    traj = json.loads(tp.read_text())
    return SimpleNamespace(agent=agent, traj=traj, sid=traj['session_id'], term=term, exc=exc,
                           stop=agent._stop_reason, ctx=ctx)


def agent_steps(traj):
    return [s for s in traj['steps'] if s.get('source') == 'agent' and not s.get('is_copied_context')]


def owners(stack, sid):
    return [r['owner'] for r in stack.turns(sid) if r.get('turn') is not None]


def render_qwen(messages):
    """Qwen3.8's chat template on what vLLM hands it: reasoning only from the `reasoning` key (chat_utils.py)."""
    env = jinja2.Environment(extensions=['jinja2.ext.loopcontrols'])

    def raise_exception(msg):
        raise jinja2.TemplateError(msg)
    env.globals['raise_exception'] = raise_exception
    tpl = env.from_string((HERE / 'qwen38_chat_template.jinja').read_text())
    conv = []
    for m in messages:
        c = {'role': m['role'], 'content': m.get('content')}
        if m['role'] == 'assistant' and m.get('reasoning') is not None:
            c['reasoning_content'] = m['reasoning']
        conv.append(c)
    return tpl.render(messages=conv, add_generation_prompt=True)


# ---- router-level tests (no harbor) -----------------------------------------------------------------------------

def test_strip_think_and_request_kinds():
    assert rr.strip_think('<|start_think|>a\nb<|end_think|>{"x": 1}') == '{"x": 1}'
    assert rr.strip_think('<|start_think|>never closed {"x": 1}') == ''
    assert rr.strip_think('{"x": 1}') == '{"x": 1}'
    u = lambda c: {'role': 'user', 'content': c}  # noqa: E731
    a = {'role': 'assistant', 'content': '{}'}
    assert rr.request_kind([u('task')]) == 'initial'
    assert rr.request_kind([u('task'), a, u('New Terminal Output:\n')]) == 'main'
    assert rr.request_kind([u('task'), a, u('Current terminal state:\nx\n\n' + rr.CONFIRM_MARK + ' If so')]) == 'confirm'
    assert rr.request_kind([u('task'), a, u(rr.SUMMARY_PREFIX + ' ...')]) == 'summary'
    assert rr.request_kind([u(rr.QUESTIONS_PREFIX + '\n**Original Task:** x')]) == 'questions'
    assert rr.request_kind([u('task'), a, u(rr.ANSWERS_PREFIX + ', please')]) == 'answers'
    assert rr.request_kind([u('task'), u('q'), a, u(rr.HANDOFF_PREFIX + '\n\nx')]) == 'handoff'


def test_health_check_rejects_a_wrong_served_name(tmp_path):
    with Stack(tmp_path, teacher_model='qwen38-typo') as st:
        assert not st.health_ok
        assert 'served model name mismatch' in st.health['endpoints']['teacher']['fail']
        assert st.health['endpoints']['student'].get('fail') is None
        assert json.loads((st.log_dir / 'health.json').read_text())['ok'] is False


def test_health_check_passes_and_reports_the_smallest_context(tmp_path):
    with Stack(tmp_path) as st:
        assert st.health_ok, st.health
        assert st.health['max_model_len'] == 65536
        assert st.health['endpoints']['teacher']['smoke']['reasoning']


def _post(url, body, headers=None):
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json',
                                                                               **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_request_without_session_header_is_fatal(tmp_path):
    with Stack(tmp_path) as st:
        status, _ = _post(st.url + '/chat/completions',
                          {'model': 'relay', 'messages': [{'role': 'user', 'content': instruction('done')}]})
        assert status == 503
        assert 'header' in (st.log_dir / 'FATAL').read_text()


def test_upstream_404_mid_run_is_fatal(tmp_path):
    with Stack(tmp_path, mode='teacher') as st:
        h = {rr.SESSION_HEADER: 's1'}
        body = {'model': 'relay', 'messages': [{'role': 'user', 'content': instruction('done')}]}
        assert _post(st.url + '/chat/completions', body, h)[0] == 200
        st.teacher.fail_model = True
        assert _post(st.url + '/chat/completions', body, {rr.SESSION_HEADER: 's2'})[0] == 404
        assert (st.log_dir / 'FATAL').exists()
        assert _post(st.url + '/chat/completions', body, {rr.SESSION_HEADER: 's3'})[0] == 503
        # a wrong model name from harbor is refused like vLLM does, without a fatal of its own
        assert _post(st.url + '/chat/completions', dict(body, model='snowball'), h)[0] == 503


# ---- Terminus-2 episodes -----------------------------------------------------------------------------------------

@needs_harbor
def test_a1_no_trigger_student_completes(tmp_path):
    """done_claim switched off: nothing fires, the student works, claims and confirms on its own."""
    with Stack(tmp_path, router_args=['--takeover', 'loop,no_progress_wait']) as st:
        r = run_agent(st, 'selfdone', tmp_path)
        assert r.exc is None and getattr(r.stop, 'value', r.stop) == 'task_complete'
        assert owners(st, r.sid) == ['student'] * 4
        assert not st.teacher_bodies('SCENARIO=selfdone')
        assert not any(x.get('takeover') for x in st.turns(r.sid))
        assert all(s['model_name'] == 'snowball' for s in agent_steps(r.traj))


@needs_harbor
def test_a2_student_budget_ends_the_episode_without_a_teacher(tmp_path):
    """No trigger and never done: the router ends the episode at 1x the task budget, as a timeout would."""
    with Stack(tmp_path, router_args=['--budget-mode', 'on'], budgets={'budget': 0.6}, student_delay=0.05) as st:
        r = run_agent(st, 'budget', tmp_path)
        own = owners(st, r.sid)
        assert own[-2:] == ['router', 'router'] and set(own[:-2]) == {'student'}
        assert getattr(r.stop, 'value', r.stop) == 'task_complete'
        assert [x['ending'] for x in st.turns(r.sid) if x['owner'] == 'router'] == ['student_budget'] * 2
        assert not st.teacher_bodies('SCENARIO=budget')


@needs_harbor
def test_b_done_claim_teacher_answers_the_confirmation(tmp_path):
    with Stack(tmp_path) as st:
        r = run_agent(st, 'done', tmp_path)
        rows = [x for x in st.turns(r.sid) if x.get('turn') is not None]
        assert [x['owner'] for x in rows] == ['student'] * 3 + ['teacher'] * 4
        claim, take = rows[2], rows[3]
        assert claim.get('done_claim') and not claim.get('takeover')
        assert take['request_kind'] == 'confirm'
        assert take['takeover']['trigger'] == 'done_claim' and take['takeover']['turn'] == 4
        assert take['takeover']['fired_turn'] == 3
        assert not any('discarded_student_reply' in x for x in rows)       # nothing discarded
        # the student's done reply ran: Terminus-2 asked "Are you sure?" and the teacher checked before confirming
        first_teacher = st.teacher_bodies('SCENARIO=done')[0]['messages']
        assert rr.CONFIRM_MARK in first_teacher[-1]['content']
        assert 'cat out.txt' in r.term.sent
        # the trajectory carries the served model per turn: an owner label independent of the router log
        assert [s['model_name'] for s in agent_steps(r.traj)] == ['snowball'] * 3 + ['qwen38'] * 4
        assert getattr(r.stop, 'value', r.stop) == 'task_complete'


@needs_harbor
def test_c_exact_repeat_loop_takes_over_mid_episode(tmp_path):
    with Stack(tmp_path) as st:
        r = run_agent(st, 'loop', tmp_path)
        rows = [x for x in st.turns(r.sid) if x.get('turn') is not None]
        take = next(x for x in rows if x.get('takeover'))
        assert take['takeover']['trigger'] == 'loop' and take['takeover']['kind'] == 'environment'
        assert take['turn'] == 4 and 'exact repeat x3' in take['takeover']['reason']
        assert [x['owner'] for x in rows[:3]] == ['student'] * 3
        assert {x['owner'] for x in rows[3:]} == {'teacher'}
        # the environment trigger acted before the student was asked for turn 4
        student_calls = [b for b in st.student.requests if 'SCENARIO=loop' in fake_openai.text_of(b['messages'][0]['content'])]
        assert len(student_calls) == 3


@needs_harbor
def test_d_no_progress_wait_takes_over(tmp_path):
    with Stack(tmp_path) as st:
        r = run_agent(st, 'wait', tmp_path)
        rows = [x for x in st.turns(r.sid) if x.get('turn') is not None]
        take = next(x for x in rows if x.get('takeover'))
        assert take['takeover']['trigger'] == 'no_progress_wait'
        assert take['turn'] == 5 and '3 waits in a row' in take['takeover']['reason']
        assert [x['owner'] for x in rows[:4]] == ['student'] * 4


@needs_harbor
@pytest.mark.parametrize('interleaved,teacher_key,source', [
    (True, 'reasoning', 'reasoning_key_from_harbor'),           # the pilot's setting: harbor re-sends it
    (True, 'reasoning_content', 'reasoning_key_from_harbor'),   # older vLLM response key, normalised by the router
    (False, 'reasoning', 'reasoning_restored'),                 # harbor without interleaved_thinking: router restores
])
def test_e_teacher_reasoning_is_refed_on_its_next_turn(tmp_path, interleaved, teacher_key, source):
    with Stack(tmp_path, teacher_key=teacher_key) as st:
        r = run_agent(st, 'done', tmp_path, interleaved=interleaved)
        bodies = st.teacher_bodies('SCENARIO=done')
        assert len(bodies) == 4
        second = bodies[1]['messages']
        prior = [m for m in second if m['role'] == 'assistant' and 'teacher-step 0' in m['content']]
        assert len(prior) == 1 and prior[0]['reasoning'] == 'teacher reasoning 0'
        assert prior[0]['reasoning_content'] == 'teacher reasoning 0'
        # the teacher never sees the student's thinking; the student's JSON actions stay
        for b in bodies:
            assert not any('<|start_think|>' in fake_openai.text_of(m.get('content')) for m in b['messages'])
            assert 'skip_special_tokens' not in b
        student_turns = [m for m in second if m['role'] == 'assistant' and '"analysis"' in m['content']
                         and 'teacher-step' not in m['content']]
        assert len(student_turns) == 3 and all(m['content'].startswith('{') for m in student_turns)
        # the rendered history: Qwen3.8's template puts the teacher's reasoning back into its turn
        rendered = render_qwen(bodies[3]['messages'])
        for k in range(3):
            assert f'<think>\nteacher reasoning {k}\n</think>' in rendered
        assert 'student thinking' not in rendered
        assert rendered.count('<think>\n\n</think>') == 3        # student turns render with an empty think block
        stats = [x['refeed'] for x in st.turns(r.sid) if x['owner'] == 'teacher']
        assert [s['prior_teacher_turns'] for s in stats] == [0, 1, 2, 3]
        assert [s[source] for s in stats] == [0, 1, 2, 3]
        assert all(s['think_stripped'] == 3 for s in stats)
        # harbor recorded the teacher's reasoning in the trajectory (the SFT converter's source)
        assert [s.get('reasoning_content') for s in agent_steps(r.traj)[3:]] == [f'teacher reasoning {k}' for k in range(4)]


@needs_harbor
def test_f_concurrent_episodes_do_not_cross(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    with Stack(tmp_path, student_delay=0.02) as st:
        scen = ['loop', 'done', 'wait', 'loop', 'done', 'wait']
        with ThreadPoolExecutor(len(scen)) as ex:
            res = list(ex.map(lambda i: run_agent(st, scen[i], tmp_path, tag=f'-{i}'), range(len(scen))))
        assert len({r.sid for r in res}) == len(scen)
        want = {'loop': ('loop', 4), 'done': ('done_claim', 4), 'wait': ('no_progress_wait', 5)}
        for i, r in enumerate(res):
            rows = [x for x in st.turns(r.sid) if x.get('turn') is not None]
            takes = [x for x in rows if x.get('takeover')]
            assert len(takes) == 1, (scen[i], takes)
            assert (takes[0]['takeover']['trigger'], takes[0]['turn']) == want[scen[i]]
            assert {x['task_id'] for x in rows} == {f'cf-{scen[i]}'}
            # every body sent upstream for this episode belongs to this episode's conversation
            for x in rows:
                path, seq = x['sent_body'].split('#') if x.get('sent_body') else (None, None)
                if path:
                    with gzip.open(st.log_dir / path, 'rt') as f:
                        b = [json.loads(l) for l in f if json.loads(l)['seq'] == int(seq)][0]
                    assert f'SCENARIO={scen[i]}-{i}.' in b['body']['messages'][0]['content']
        eps = json.loads((st.log_dir / 'episodes.json').read_text())['episodes'] if (st.log_dir / 'episodes.json').exists() else None
        assert eps is None or len(eps) == len(scen)


@needs_harbor
def test_g_summarization_mid_episode_stays_with_the_owner(tmp_path):
    """Terminus-2's context summarization (on in the v0.1 policy) runs through the router: its three sub-requests go to
    whoever owns the episode, and the main chat continues on the same episode after the history is replaced."""
    with Stack(tmp_path, router_args=['--budget-mode', 'on'], budgets={'summ': 0.8}, student_delay=0.02) as st:
        r = run_agent(st, 'summ', tmp_path)
        rows = st.turns(r.sid)
        kinds = [x['request_kind'] for x in rows]
        assert {'summary', 'questions', 'answers', 'handoff'} <= set(kinds)
        for x in rows:
            if x['request_kind'] in ('summary', 'questions', 'answers'):
                assert x['owner'] == 'student' and x['turn'] is None
        main = [x for x in rows if x.get('turn') is not None]
        turns = [x['turn'] for x in main]
        assert turns == list(range(1, len(turns) + 1))            # one agent turn per main request, across the reset
        assert main[-1]['owner'] == 'router'


@needs_harbor
def test_h_decision_trigger_discards_the_student_reply(tmp_path):
    """A decision trigger other than done_claim (gave_up, when enabled): the reply never runs, it is saved, and the
    teacher answers the same request."""
    with Stack(tmp_path, router_args=['--takeover', 'done_claim,loop,no_progress_wait,gave_up']) as st:
        r = run_agent(st, 'gaveup', tmp_path)
        rows = [x for x in st.turns(r.sid) if x.get('turn') is not None]
        take = next(x for x in rows if x.get('takeover'))
        assert take['takeover']['trigger'] == 'gave_up' and take['turn'] == 3
        assert 'impossible' in take['discarded_student_reply']['choices'][0]['message']['content']
        assert take['owner'] == 'teacher'
        assert 'echo stub > out.txt' not in r.term.sent                   # the student's bad action never ran
        tb = st.teacher_bodies('SCENARIO=gaveup')[0]['messages']
        sb = [b for b in st.student.requests if 'SCENARIO=gaveup' in b['messages'][0]['content']][-1]['messages']
        assert len(tb) == len(sb) and tb[-1] == sb[-1]                    # the same request


@needs_harbor
def test_control_arm_teacher_answers_everything(tmp_path):
    with Stack(tmp_path, mode='teacher') as st:
        r = run_agent(st, 'done', tmp_path)
        own = owners(st, r.sid)
        assert set(own) == {'teacher'} and not st.student.requests
        rows = [x for x in st.turns(r.sid) if x.get('turn') is not None]
        assert all('would_fire' in x for x in rows)
        assert rows[-1]['refeed']['reasoning_key_from_harbor'] == rows[-1]['refeed']['prior_teacher_turns'] > 0


@needs_harbor
@pytest.mark.parametrize('mode', ['strip', 'keep'])
def test_student_think_modes_in_the_rendered_teacher_prompt(tmp_path, mode):
    """--student-think: what Qwen3.8's template renders for the student's turns at the hand-off.
    strip: the student's turns render with an empty think block, their JSON (analysis/plan) intact.
    keep: the student's thinking renders inside that turn's <think>. RELAY_WRITE_RENDERED=<dir> saves both prompts."""
    with Stack(tmp_path, router_args=['--student-think', mode]) as st:
        r = run_agent(st, 'done', tmp_path)
        bodies = st.teacher_bodies('SCENARIO=done')
        first = bodies[0]['messages']                      # the hand-off request (the "Are you sure?" turn)
        rendered = render_qwen(first)
        student = [m for m in first if m['role'] == 'assistant']
        assert len(student) == 3 and all(m['content'].startswith('{"analysis"') for m in student)
        assert '<|start_think|>' not in rendered and '"analysis": "I am done"' in rendered
        if mode == 'strip':
            assert 'student thinking' not in rendered
            assert rendered.count('<think>\n\n</think>') == 3
            assert all('reasoning' not in m for m in student)
        else:
            for k in range(3):
                assert f'<think>\nstudent thinking {k}\n</think>\n\n{{"analysis"' in rendered
            assert [m['reasoning'] for m in student] == [f'student thinking {k}' for k in range(3)]
        rows = st.turns(r.sid)
        assert {x['student_think'] for x in rows} == {mode}
        assert all(x['refeed']['student_think_as_reasoning'] == (3 if mode == 'keep' else 0)
                   for x in rows if x['owner'] == 'teacher')
        ev = [json.loads(l) for l in (st.log_dir / 'events.jsonl').read_text().splitlines()]
        assert [e['student_think'] for e in ev if e['event'] == 'episode_start'] == [mode]
        out = os.environ.get('RELAY_WRITE_RENDERED')
        if out:
            Path(out).mkdir(parents=True, exist_ok=True)
            (Path(out) / f'teacher_prompt_at_handoff_{mode}.txt').write_text(rendered)


def test_deadline_ends_episodes(tmp_path):
    import time
    with Stack(tmp_path, mode='teacher', router_args=['--deadline-epoch', str(time.time() - 1)]) as st:
        body = {'model': 'relay', 'messages': [{'role': 'user', 'content': instruction('done')}]}
        status, resp = _post(st.url + '/chat/completions', body, {rr.SESSION_HEADER: 's1'})
        assert status == 200 and json.loads(resp['choices'][0]['message']['content'])['task_complete'] is True
        assert st.turns('s1')[0]['ending'] == 'deadline' and st.turns('s1')[0]['owner'] == 'router'
        assert not st.teacher.requests[1:]      # only the health smoke reached the teacher


def test_readout_false_done_guard():
    """g1: the teacher's first task_complete within 2 of its turns, no verification command before it, task failed."""
    sys.path.insert(0, str(HERE.parent.parent / 'pilot'))
    import readout

    def rec(turn, cmds, done):
        c = json.dumps(dict(analysis='a', plan='p', task_complete=done, commands=[dict(keystrokes=k) for k in cmds]))
        return dict(owner='teacher', turn=turn, upstream_status=200, usage={'completion_tokens': 10, 'prompt_tokens': 100},
                    response={'choices': [{'message': {'content': c}}]})
    fail = dict(harness_error=False, verifier_timeout=False, censored=False, reward=0.0, exc=None)
    ok = dict(fail, reward=1.0)
    ep = lambda recs: dict(takeover=dict(turn=5), main=recs, aux=[])  # noqa: E731
    g = readout.takeover_guards(ep([rec(5, [], True), rec(6, [], True)]), fail)          # rubber stamp, fails
    assert g['g1_false_done'] and not g['g2_context_exceeded'] and g['teacher_turns'] == 2
    assert not readout.takeover_guards(ep([rec(5, ['python -m pytest tests/\n'], False), rec(6, [], True)]), fail)['g1_false_done']
    assert not readout.takeover_guards(ep([rec(5, [], True)]), ok)['g1_false_done']        # confirmed and passed
    assert not readout.takeover_guards(ep([rec(5, ['ls\n'], False), rec(6, ['cat x\n'], False), rec(7, [], True)]),
                                       fail)['g1_false_done']                              # claim outside 2 turns
    assert readout.takeover_guards(ep([rec(5, ['cat out.txt\n'], False), rec(6, [], True)]), fail)['g1_false_done']


REPAIR = ['--repair-on-parse-error']


def _student_bodies(st, tag):
    return [b for b in st.student.requests if tag in fake_openai.text_of(b['messages'][0].get('content'))]


def _no_bad_format_in_trace(r, st, tag):
    """The discarded reply never reached harbor: not in the trajectory, no parse-error re-prompt anywhere."""
    raw = json.dumps(r.traj)
    assert 'BADFORMAT' not in raw
    assert 'Previous response had parsing errors' not in raw
    for b in _student_bodies(st, tag) + st.teacher_bodies(tag):
        assert not any('BADFORMAT' in fake_openai.text_of(m.get('content')) for m in b['messages'])
        assert not any('Previous response had parsing errors' in fake_openai.text_of(m.get('content')) for m in b['messages'])


@needs_harbor
def test_repair_then_the_student_resumes_then_done_claim_takes_over(tmp_path):
    with Stack(tmp_path, router_args=REPAIR) as st:
        r = run_agent(st, 'repair', tmp_path)
        rows = [x for x in st.turns(r.sid) if x.get('turn') is not None]
        assert [x['owner'] for x in rows] == ['student', 'teacher', 'student', 'student', 'teacher']
        rep = rows[1]
        assert rep['repair'] and rep['repair_kind'] == 'parse_error' and 'Missing required fields' in rep['repair_reason']
        assert 'BADFORMAT' in rep['discarded_student_reply']['choices'][0]['message']['content']
        assert not rep.get('takeover') and rep['repair_reply_parse_error'] is None
        assert rows[4]['takeover']['trigger'] == 'done_claim' and rows[4]['request_kind'] == 'confirm'
        _no_bad_format_in_trace(r, st, 'SCENARIO=repair.')
        # every executed turn parsed: the trajectory holds 5 agent steps, one per router record, owners by model
        assert [s['model_name'] for s in agent_steps(r.traj)] == ['snowball', 'qwen38', 'snowball', 'snowball', 'qwen38']
        # the student sees the repair turn as 09-21's SFT renders a teacher turn: reasoning inside its think markers
        after = _student_bodies(st, 'SCENARIO=repair.')[2]['messages']
        tm = [m for m in after if m['role'] == 'assistant' and 'teacher-step 0' in m['content']]
        assert len(tm) == 1 and tm[0]['content'].startswith('<|start_think|>teacher reasoning 0<|end_think|>{"analysis": "teacher-step 0')
        assert 'reasoning' not in tm[0] and 'reasoning_content' not in tm[0]
        assert rows[2]['student_view'] == {'teacher_turns_inline': 1, 'teacher_turns_without_reasoning': 0}
        # the teacher's repair request is the same request the student failed
        tb = st.teacher_bodies('SCENARIO=repair.')[0]['messages']
        sb = _student_bodies(st, 'SCENARIO=repair.')[1]['messages']
        assert len(tb) == len(sb) and tb[-1]['content'] == sb[-1]['content']
        assert getattr(r.stop, 'value', r.stop) == 'task_complete'


@needs_harbor
def test_repeated_repairs_and_the_teacher_sees_its_repair_reasoning(tmp_path):
    with Stack(tmp_path, router_args=REPAIR) as st:
        r = run_agent(st, 'repairs', tmp_path)
        rows = [x for x in st.turns(r.sid) if x.get('turn') is not None]
        assert [x['owner'] for x in rows] == ['student', 'teacher', 'teacher', 'student', 'student', 'teacher']
        assert [bool(x.get('repair')) for x in rows] == [False, True, True, False, False, False]
        assert rows[5]['takeover']['trigger'] == 'done_claim'
        # the second repair request re-feeds the first repair's reasoning (harbor, under both keys)
        second = st.teacher_bodies('SCENARIO=repairs.')[1]['messages']
        prior = [m for m in second if m['role'] == 'assistant' and 'teacher-step 0' in m['content']]
        assert prior[0]['reasoning'] == 'teacher reasoning 0' and rows[2]['refeed']['reasoning_key_from_harbor'] == 1
        _no_bad_format_in_trace(r, st, 'SCENARIO=repairs.')
        eps = st.router.episodes[r.sid]
        assert eps.n_repairs == 2 and st.router.counts['repairs'] == 2


@needs_harbor
def test_two_teacher_servers_split_episodes_and_pin_each(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    with Stack(tmp_path, mode='teacher', two_teachers=True) as st:
        assert st.health_ok and set(st.health['endpoints']) == {'teacher0', 'teacher1'}
        with ThreadPoolExecutor(4) as ex:
            res = list(ex.map(lambda i: run_agent(st, 'done', tmp_path, tag=f'-{i}'), range(4)))
        for r in res:
            tag = r.traj['steps'][0]['message'].split('SCENARIO=')[1].split('.')[0]
            on = [srv for srv in (st.teacher, st.teacher2)
                  if any(f'SCENARIO={tag}.' in fake_openai.text_of(b['messages'][0]['content']) for b in srv.requests)]
            assert len(on) == 1                               # every request of an episode went to one server
        n1 = len([b for b in st.teacher.requests if 'SCENARIO=' in fake_openai.text_of(b['messages'][0]['content'])])
        n2 = len([b for b in st.teacher2.requests if 'SCENARIO=' in fake_openai.text_of(b['messages'][0]['content'])])
        assert n1 > 0 and n2 > 0


@needs_harbor
def test_student_clock_pauses_while_a_repair_is_with_the_teacher(tmp_path):
    """Two repairs at 0.6 s each would use the student's whole 1.0 s budget; paused, the student goes on to its
    done claim."""
    with Stack(tmp_path, router_args=REPAIR + ['--budget-mode', 'on'], budgets={'repairs': 1.0}, teacher_delay=0.6) as st:
        r = run_agent(st, 'repairs', tmp_path)
        rows = [x for x in st.turns(r.sid) if x.get('turn') is not None]
        assert not any(x.get('ending') for x in rows)
        assert rows[-1]['takeover']['trigger'] == 'done_claim'
        reps = [x for x in rows if x.get('repair')]
        assert len(reps) == 2 and all(x['paused_this_turn_sec'] >= 0.6 for x in reps)
        assert rows[3]['paused_sec'] >= 1.2 and rows[3]['student_clock_sec'] < rows[3]['elapsed_sec'] - 1.2 + 0.01
        assert st.router.episodes[r.sid].paused_sec >= 1.2


@needs_harbor
def test_summarization_off_context_overflow_ends_the_episode(tmp_path):
    """Run 3 policy: enable_summarize=false. A nearly full context ends the episode with ContextLengthExceededError
    (harbor still verifies it); no summarization request ever reaches the router."""
    with Stack(tmp_path, router_args=REPAIR) as st:
        r = run_agent(st, 'summ', tmp_path, enable_summarize=False)
        assert type(r.exc).__name__ == 'ContextLengthExceededError'
        kinds = {x['request_kind'] for x in st.turns(r.sid)}
        assert not kinds & {'summary', 'questions', 'answers', 'handoff'}


@needs_harbor
def test_teacher_failover_and_drain(tmp_path):
    """A teacher server that dies mid-run: its episodes re-pin to a live one and finish. A server dropped from
    --teacher-url-file gets no new requests (drain before releasing its node)."""
    from concurrent.futures import ThreadPoolExecutor
    f = tmp_path / 'teachers.txt'
    with Stack(tmp_path, mode='teacher', two_teachers=True, router_args=['--connect-retries', '2']) as st:
        st.run(st.teacher2.stop())                                   # server 2 dies after the health check
        with ThreadPoolExecutor(4) as ex:
            res = list(ex.map(lambda i: run_agent(st, 'done', tmp_path, tag=f'-{i}'), range(4)))
        assert all(getattr(r.stop, 'value', r.stop) == 'task_complete' for r in res)
        assert st.router.down_until and st.router.counts['upstream_errors'] == 0   # marked down, nothing lost
        # drain: only server 1 listed
        f.write_text(st.router.urls['teacher'][0] + '\n')
        st.a.teacher_url_file = str(f)
        st.router.reload_teacher_urls()
        assert st.router.urls['teacher'] == [st.router.urls['teacher'][0]]
