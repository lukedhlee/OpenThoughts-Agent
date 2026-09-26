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
                 two_teachers=False, teacher_long=0):
        self.tmp = Path(tmp)
        self.mode, self.router_args = mode, list(router_args)
        self.student = fake_openai.FakeServer('student', 'snowball', delay=student_delay)
        self.teacher = fake_openai.FakeServer('teacher', 'qwen38', reasoning_key=teacher_key, delay=teacher_delay,
                                              long_reasoning=teacher_long)
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
                 for sc in ('selfdone', 'done', 'loop', 'wait', 'budget', 'summ', 'gaveup', 'repair', 'repairs', 'autofix')
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
        assert rows[2]['student_view'] == {'teacher_turns_inline': 1, 'teacher_turns_without_reasoning': 0, 'cuts': []}
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



# ---- run 4 changes: format autofix, the cap on older teacher reasoning, the SFT render ---------------------------

sys.path.insert(0, str(HERE.parent.parent / 'sft'))


def _word_tokenizer(path):
    """A word-level stand-in for 09-21's tokenizer with its special tokens (atomic, as in the real one)."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    t = Tokenizer(models.WordLevel({'[UNK]': 0}, unk_token='[UNK]'))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    t.add_special_tokens(['<|begin_of_text|>', '<|start_header_id|>', '<|end_header_id|>', '<|eot_id|>',
                          '<|start_think|>', '<|end_think|>'])
    t.save(str(path))
    return str(path)


@needs_harbor
def test_autofix_runs_the_students_own_action(tmp_path):
    with Stack(tmp_path, router_args=REPAIR + ['--autofix']) as st:
        r = run_agent(st, 'autofix', tmp_path)
        rows = [x for x in st.turns(r.sid) if x.get('turn') is not None]
        assert [x['owner'] for x in rows] == ['student', 'student', 'teacher', 'student', 'teacher']
        fx = rows[1]
        assert fx['autofix'] and fx['autofix_kind'] == 'tool_call_command' and not fx.get('repair')
        assert '<tool_call>' in fx['original_student_reply']['choices'][0]['message']['content']
        assert 'echo hi > out.txt' in r.term.sent                       # the student's own action ran
        assert rows[2]['repair'] and rows[2]['autofix_unfixable'] == 'no_action'
        assert rows[4]['takeover']['trigger'] == 'done_claim'
        step = agent_steps(r.traj)[1]['message']
        assert step.startswith('<|start_think|>AUTOFIX me: list the files.<|end_think|>{"analysis": "I will list the files."')
        assert 'Previous response had parsing errors' not in json.dumps(r.traj)
        assert st.router.counts['autofixes'] == 1


@needs_harbor
def test_older_teacher_reasoning_is_cut_in_the_student_view_and_the_render_matches(tmp_path):
    import render
    tokp = _word_tokenizer(tmp_path / 'tok.json')
    with Stack(tmp_path, router_args=REPAIR + ['--student-tokenizer', tokp, '--reasoning-cap', '20'], teacher_long=60) as st:
        r = run_agent(st, 'repairs', tmp_path)
        rows = [x for x in st.turns(r.sid) if x.get('turn') is not None]
        assert [x['owner'] for x in rows] == ['student', 'teacher', 'teacher', 'student', 'student', 'teacher']
        body = _student_bodies(st, 'SCENARIO=repairs.')[3]['messages']          # after both repairs
        t0 = next(m['content'] for m in body if m['role'] == 'assistant' and 'teacher-step 0' in m['content'])
        t1 = next(m['content'] for m in body if m['role'] == 'assistant' and 'teacher-step 1' in m['content'])
        assert 'Sentence 59 of step 1 is here.<|end_think|>' in t1               # the latest teacher turn: whole
        assert 'Sentence 59 of step 0' not in t0 and '.<|end_think|>{"analysis": "teacher-step 0' in t0   # older: cut
        cuts = rows[3]['student_view']['cuts']
        assert len(cuts) == 1 and 0 < cuts[0]['cut_at'] < cuts[0]['reasoning_chars']
        # the SFT row: same capped history, masks as specified
        tok = render.rcap.load_tokenizer(tokp)
        tpl = render.load_template(HERE / 'grug0921_chat_template.jinja')
        row = render.render_episode(r.traj, st.turns(r.sid), tok, tpl, '<|begin_of_text|>', cap=20)
        c = render.check_row(row, tok)
        assert c['teacher_turns'] == 3 and c['cut_turns'] == 2                  # two repairs cut, the final turn whole
        assert t0 in row['text']                                               # byte-identical to the student's view
        assert row['text'].startswith('<|begin_of_text|><|start_header_id|>system<|end_header_id|>Reasoning: /think<|eot_id|>')


@needs_harbor
def test_render_trains_an_autofixed_action_not_its_reasoning(tmp_path):
    import render
    tokp = _word_tokenizer(tmp_path / 'tok.json')
    with Stack(tmp_path, router_args=REPAIR + ['--autofix', '--student-tokenizer', tokp]) as st:
        r = run_agent(st, 'autofix', tmp_path)
        tok = render.rcap.load_tokenizer(tokp)
        tpl = render.load_template(HERE / 'grug0921_chat_template.jinja')
        row = render.render_episode(r.traj, st.turns(r.sid), tok, tpl, '<|begin_of_text|>')
        c = render.check_row(row, tok)
        assert c['autofix_turns'] == 1 and c['cut_turns'] == 0
        fx = next(m for m in row['turns'] if m.get('autofix'))
        trained = ''.join(row['text'][a:b] for (a, b), l in zip(row['offsets'], row['loss'])
                          if l and a >= fx['span'][0] and b <= fx['span'][1])
        assert 'echo hi > out.txt' in row['text'] and 'AUTOFIX' not in trained
        off = render.render_episode(r.traj, st.turns(r.sid), tok, tpl, '<|begin_of_text|>', autofix_loss='none')
        assert sum(off['loss']) < sum(row['loss'])



@needs_harbor
def test_teacher_reply_cap_cut_case_and_context_fallback(tmp_path):
    """--teacher-max-tokens: every teacher request carries it, the student's never; a reply cut there is logged
    (teacher_cut_at_cap) and handed to harbor as served; a cap that does not fit the context is lowered to what is left."""
    with Stack(tmp_path, mode='relay', router_args=REPAIR + ['--teacher-max-tokens', '16384']) as st:
        st.teacher.cut_first = True
        r = run_agent(st, 'done', tmp_path)
        tb = st.teacher_bodies('SCENARIO=done.')
        assert tb and all(b['max_tokens'] == 16384 for b in tb)
        assert not any('max_tokens' in b for b in _student_bodies(st, 'SCENARIO=done.'))
        rows = [x for x in st.turns(r.sid) if x.get('owner') == 'teacher']
        assert rows[0].get('teacher_cut_at_cap') and st.router.counts['teacher_cut_at_cap'] == 1
        assert getattr(r.stop, 'value', r.stop) == 'task_complete'     # Terminus-2 asked again and the episode ended
    (tmp_path / 'b').mkdir()
    with Stack(tmp_path / 'b', mode='teacher', router_args=['--teacher-max-tokens', '70000']) as st:
        r = run_agent(st, 'done', tmp_path, tag='-0')
        rows = [x for x in st.turns(r.sid) if x.get('turn') is not None]
        assert all(x['upstream_status'] == 200 for x in rows)
        assert all(x['teacher_max_tokens_dropped'] for x in rows)


@needs_harbor
def test_every_model_call_pauses_the_budget_clock(tmp_path):
    """--pause-model-calls: budgets count sandbox/tool time, not serving latency, for the student and the teacher."""
    with Stack(tmp_path, router_args=REPAIR + ['--budget-mode', 'on', '--pause-model-calls'],
               budgets={'budget': 1.0, 'done': 1.0}, student_delay=0.2, teacher_delay=0.4) as st:
        r = run_agent(st, 'budget', tmp_path, max_turns=8)                 # 8 x 0.2 s of student calls > 1.0 s budget
        rows = [x for x in st.turns(r.sid) if x.get('turn') is not None]
        assert not any(x.get('ending') for x in rows) and type(r.exc).__name__ == 'TurnCapExhaustedError'
        assert st.router.episodes[r.sid].paused_sec >= 8 * 0.2
        r2 = run_agent(st, 'done', tmp_path, tag='-0')                     # 4 teacher turns x 0.4 s > 1.0 s after takeover
        rows2 = [x for x in st.turns(r2.sid) if x.get('turn') is not None]
        assert rows2[3]['takeover']['trigger'] == 'done_claim' and not any(x.get('ending') for x in rows2)
        assert getattr(r2.stop, 'value', r2.stop) == 'task_complete'
        assert all(x['paused_this_turn_sec'] >= 0.2 for x in rows2)


def test_reported_max_model_len(tmp_path):
    import urllib.request
    with Stack(tmp_path, mode='teacher', router_args=['--report-max-model-len', '131072']) as st:
        d = json.load(urllib.request.urlopen(st.url + '/models'))
        assert d['data'][0]['max_model_len'] == 131072


@needs_harbor
def test_render_masks_uncut_reasoning_over_the_think_limit(tmp_path):
    """Same rule in both arms: a Qwen turn's thinking is trained only if never cut AND within the limit (8,192 tokens
    of 09-21's tokenizer in production; 5 here). Control (teacher-only) episodes render with the same function."""
    import render
    tokp = _word_tokenizer(tmp_path / 'tok.json')
    with Stack(tmp_path, mode='teacher', router_args=['--student-tokenizer', tokp, '--reasoning-cap', '20'],
               teacher_long=60) as st:
        r = run_agent(st, 'done', tmp_path)
        tok = render.rcap.load_tokenizer(tokp)
        tpl = render.load_template(HERE / 'grug0921_chat_template.jinja')
        row = render.render_episode(r.traj, st.turns(r.sid), tok, tpl, '<|begin_of_text|>', cap=20, think_limit=5)
        c = render.check_row(row, tok)
        t = [m for m in row['turns'] if m['owner'] == 'teacher']
        assert c['cut_turns'] == len(t) - 1 and c['long_think_masked'] == 1 and not any(m['think_trained'] for m in t)
        wide = render.render_episode(r.traj, st.turns(r.sid), tok, tpl, '<|begin_of_text|>', cap=20, think_limit=10 ** 6)
        assert render.check_row(wide, tok)['long_think_masked'] == 0 and sum(wide['loss']) > sum(row['loss'])



def test_active_balance_and_engine_metrics(tmp_path):
    """--balance active: a new episode goes to the server with the fewest ACTIVE episodes (finished ones drop out);
    --engine-metrics logs each server's KV / running / waiting."""
    import time as _t
    with Stack(tmp_path, mode='teacher', two_teachers=True,
               router_args=['--balance', 'active', '--active-window', '100', '--engine-metrics']) as st:
        rt_ = st.router
        u0, u1 = rt_.urls['teacher']

        def ep(sid, url, age):
            e = rr.Episode(sid, 0, 'x', None, 'teacher')
            e.pinned['teacher'] = url
            e.last_t = _t.time() - age
            rt_.episodes[sid] = e
            return e
        for i in range(3):
            ep(f'old{i}', u0, 1000)          # finished long ago: not active
        ep('a', u1, 5)                       # active on server 1
        new = rr.Episode('new', 0, 'x', None, 'teacher')
        assert rt_.pick('teacher', new) == u0      # server 0 has 0 active (3 pinned), server 1 has 1
        st.run(rt_.log_engines())
        rows = [json.loads(l) for l in (st.log_dir / 'engines.jsonl').read_text().splitlines()]
        assert {r['url'] for r in rows} == {u0, u1}
        assert all(r['kv'] == 0.42 and r['running'] == 3.0 and r['waiting'] == 1.0 for r in rows)


# ---- context_budget and the trainability hard end (2026-09-26) ---------------------------------------------------
# The fake student's /tokenize reports 1,000 tokens per assistant turn in the request, so agent turn t (1-based) is
# (t - 1) * 1,000 tokens: turn 4 reaches a 3,000 threshold. Harbor's own /tokenize probes see the same counts, far
# under its 32,768 input limit.

def _per_turn(k=1000):
    return lambda msgs: k * sum(1 for m in msgs if m.get('role') == 'assistant')


CTX = REPAIR + ['--context-budget-tokens', '3000']


def _main_rows(st, sid):
    return [x for x in st.turns(sid) if x.get('turn') is not None]


def test_context_budget_fire_rule():
    import relay_triggers as rt_
    assert rt_.context_budget_fire(31999, 5, 32000) is None
    f = rt_.context_budget_fire(32000, 5, 32000)
    assert f['trigger'] == 'context_budget' and f['kind'] == 'decision' and f['prompt_tokens'] == 32000 and f['turn'] == 5
    assert rt_.context_budget_fire(60000, 1, 32000) is None          # never before turn 2
    assert rt_.context_budget_fire(60000, 2, 32000) is not None
    assert rt_.context_budget_fire(None, 5, 32000) is None and rt_.context_budget_fire(60000, 5, None) is None
    a = rr.parse_args(['--mode', 'teacher', '--port', '0', '--log-dir', 'x', '--teacher-url', 'u', '--teacher-model', 'm'])
    assert a.context_budget_tokens is None and a.student_row_max_tokens is None and a.student_row_reserve == 8192   # off by default


@needs_harbor
def test_context_budget_fires_at_the_threshold_and_is_sticky(tmp_path):
    with Stack(tmp_path, router_args=CTX) as st:
        st.student.count_fn = _per_turn()
        r = run_agent(st, 'budget', tmp_path, max_turns=9)
        rows = _main_rows(st, r.sid)
        assert [x['owner'] for x in rows[:4]] == ['student', 'student', 'student', 'teacher']
        assert [x['student_view_tokens'] for x in rows[:4]] == [0, 1000, 2000, 3000]
        tk = rows[3]['takeover']
        assert tk['trigger'] == 'context_budget' and tk['kind'] == 'decision' and tk['turn'] == 4 and tk['prompt_tokens'] == 3000
        assert 'discarded_student_reply' not in rows[3]                 # the student was never asked at turn 4
        assert len(_student_bodies(st, 'SCENARIO=budget.')) == 3
        assert all(x['owner'] == 'teacher' for x in rows[3:])            # sticky
        assert sum(1 for x in rows if x.get('takeover')) >= 1 and {x['takeover']['trigger'] for x in rows if x.get('takeover')} == {'context_budget'}
        ev = [json.loads(l) for l in (st.log_dir / 'events.jsonl').read_text().splitlines()]
        tev = [e for e in ev if e.get('event') == 'takeover' and e.get('sid') == r.sid]
        assert len(tev) == 1 and tev[0]['trigger'] == 'context_budget' and tev[0]['prompt_tokens'] == 3000 and tev[0]['turn'] == 4
        assert st.router.counts['takeovers'] == 1 and st.router.episodes[r.sid].takeover['trigger'] == 'context_budget'
        assert getattr(r.stop, 'value', r.stop) == 'task_complete'
        assert not any(x.get('ending') for x in rows)                    # no hard end configured


@needs_harbor
def test_context_budget_never_fires_at_turn_1(tmp_path):
    with Stack(tmp_path, router_args=CTX) as st:
        st.student.count_fn = lambda msgs: 5000 if not any(m.get('role') == 'assistant' for m in msgs) else 100
        r = run_agent(st, 'selfdone', tmp_path)
        rows = _main_rows(st, r.sid)
        assert rows[0]['student_view_tokens'] == 5000 and rows[0]['owner'] == 'student' and not rows[0].get('takeover')
        assert not any((x.get('takeover') or {}).get('trigger') == 'context_budget' for x in rows)


@needs_harbor
def test_context_budget_off_counts_nothing(tmp_path):
    with Stack(tmp_path, router_args=REPAIR) as st:
        r = run_agent(st, 'done', tmp_path)
        assert not any('student_view_tokens' in x for x in st.turns(r.sid))
        assert _main_rows(st, r.sid)[3]['takeover']['trigger'] == 'done_claim'


@needs_harbor
def test_hard_end_after_takeover_ends_as_context_overflow(tmp_path):
    """Row limit 7,000 - reserve 2,000 = 5,000: 5,000 still runs (turn 6), 6,000 ends (turn 7, the teacher's
    confirmation request). Harbor sees vLLM's context 400 and raises ContextLengthExceededError (scored overflow)."""
    sys.path.insert(0, str(HERE.parent.parent / 'pilot'))
    import readout
    with Stack(tmp_path, router_args=CTX + ['--student-row-max-tokens', '7000', '--student-row-reserve', '2000']) as st:
        st.student.count_fn = _per_turn()
        r = run_agent(st, 'budget', tmp_path, max_turns=12, enable_summarize=False)
        assert type(r.exc).__name__ == 'ContextLengthExceededError'
        rows = _main_rows(st, r.sid)
        assert [x['owner'] for x in rows] == ['student'] * 3 + ['teacher'] * 3 + ['router']
        assert [x['student_view_tokens'] for x in rows] == [0, 1000, 2000, 3000, 4000, 5000, 6000]
        end = rows[-1]
        assert end['ending'] == 'context_hard_end' and end['upstream_status'] == 400 and end['turn'] == 7
        assert 'maximum context length' in end['upstream_error']
        assert end['hard_end'] == {'student_view_tokens': 6000, 'limit': 5000, 'turn': 7, 'takeover_trigger': 'context_budget'}
        assert len(st.teacher_bodies('SCENARIO=budget.')) == 3           # the teacher never saw the ended request
        assert st.router.counts['context_hard_ends'] == 1 and st.router.episodes[r.sid].ending == 'context_hard_end'
        # labels: an overflow scored as a model failure, never a harness error; the router record is no upstream error
        rv = readout.router_view(str(st.log_dir))
        h = readout.harness_router(rv)
        assert not h['upstream_errors'] and h['context_length_400'] == 1
        row = dict(trial='t', task='cf-budget', reward=0.0, exc='ContextLengthExceededError', sid=r.sid, steps=[],
                   verifier_timeout=False, censored=False,
                   harness_error=False if 'ContextLengthExceededError' in readout.AGENT_ENDS else True)
        assert not row['harness_error'] and readout.usable(row)
        readout.mark_censored(rv, [row])
        assert readout.failure_cause(row, rv['eps'][r.sid]['ending']) == 'context_overflow'
        cb = readout.context_budget_view(rv, [row])
        assert cb['fires'] == 1 and cb['fire_turn_p50'] == 4 and cb['fire_prompt_tokens_p50'] == 3000
        assert cb['hard_ends'] == 1 and cb['overflows_from_hard_end'] == 1 and cb['hard_ends_not_overflow'] == 0
        assert cb['hard_ends_by_takeover'] == {'context_budget': 1} and cb['view_count_errors'] == 0
        assert cb['recovery_after_context_budget']['scored'] == 1 and cb['recovery_after_context_budget']['recovery'] == 0.0
        rp = readout.repair_view(rv, [row])                                # the hard end is not an executed turn
        assert rp['executed_turns'] == {'student': 3, 'teacher_sticky': 3}


@needs_harbor
def test_hard_end_applies_to_other_takeovers_too(tmp_path):
    """done_claim takeover at turn 4 (confirmation); with the budget off the hard end still guards the row."""
    with Stack(tmp_path, router_args=REPAIR + ['--student-row-max-tokens', '5000', '--student-row-reserve', '1500']) as st:
        st.student.count_fn = _per_turn()
        r = run_agent(st, 'done', tmp_path, enable_summarize=False)
        rows = _main_rows(st, r.sid)
        assert rows[3]['takeover']['trigger'] == 'done_claim' and rows[3]['owner'] == 'teacher'
        assert rows[3]['student_view_tokens'] == 3000 and rows[3]['upstream_status'] == 200
        assert rows[4]['ending'] == 'context_hard_end' and rows[4]['hard_end']['takeover_trigger'] == 'done_claim'
        assert type(r.exc).__name__ == 'ContextLengthExceededError'


@needs_harbor
def test_context_budget_after_parse_error_repairs(tmp_path):
    """Repairs are counted in the student's view (teacher turns inline); the budget fires at turn 4 before the
    student is asked, so no third student call and no further repair."""
    with Stack(tmp_path, router_args=CTX) as st:
        st.student.count_fn = _per_turn()
        r = run_agent(st, 'repairs', tmp_path)
        rows = _main_rows(st, r.sid)
        assert [x['owner'] for x in rows[:4]] == ['student', 'teacher', 'teacher', 'teacher']
        assert [bool(x.get('repair')) for x in rows[:4]] == [False, True, True, False]
        assert rows[3]['takeover']['trigger'] == 'context_budget' and rows[3]['takeover']['prompt_tokens'] == 3000
        assert rows[2]['student_view_tokens'] == 2000 and not rows[2].get('takeover')   # repair turns stay non-sticky below it
        assert len(_student_bodies(st, 'SCENARIO=repairs.')) == 3
        assert not any(x.get('repair') for x in rows[4:]) and all(x['owner'] == 'teacher' for x in rows[3:])
        _no_bad_format_in_trace(r, st, 'SCENARIO=repairs.')
        assert getattr(r.stop, 'value', r.stop) == 'task_complete'


@needs_harbor
def test_context_budget_and_done_claim(tmp_path):
    """A done claim below the threshold keeps done_claim (its confirmation is not a student request); at a lower
    threshold the budget fires first and the student never claims."""
    with Stack(tmp_path, router_args=CTX) as st:
        st.student.count_fn = _per_turn()
        r = run_agent(st, 'done', tmp_path)
        rows = _main_rows(st, r.sid)
        assert rows[2]['done_claim'] and rows[2]['student_view_tokens'] == 2000
        assert rows[3]['takeover']['trigger'] == 'done_claim' and rows[3]['request_kind'] == 'confirm'
        assert 'student_view_tokens' not in rows[3]                      # no count once the teacher owns (no hard end set)
        assert st.router.counts['takeovers'] == 1
    (tmp_path / 'b').mkdir()
    with Stack(tmp_path / 'b', router_args=REPAIR + ['--context-budget-tokens', '2000']) as st:
        st.student.count_fn = _per_turn()
        r = run_agent(st, 'done', tmp_path, tag='-0')
        rows = _main_rows(st, r.sid)
        assert rows[2]['takeover']['trigger'] == 'context_budget' and rows[2]['turn'] == 3
        assert not any(x.get('done_claim') for x in rows)
        assert len(_student_bodies(st, 'SCENARIO=done-0.')) == 2
        assert getattr(r.stop, 'value', r.stop) == 'task_complete'


def test_count_failure_leaves_the_student_in_charge(tmp_path):
    """A failed count (student /tokenize down) logs the error and neither rule acts on that request."""
    with Stack(tmp_path, router_args=CTX + ['--student-row-max-tokens', '7000']) as st:
        def boom(msgs):
            raise RuntimeError('tokenize down')
        st.student.count_fn = boom
        ep = rr.Episode('s1', 1, 'x', None, 'student')
        rec = {}
        n = st.run(st.router.count_student_view(ep, {'model': 'relay'}, [{'role': 'user', 'content': 'hi'}], rec))
        assert n is None and rec['student_view_tokens'] is None and rec['student_view_tokens_error'].startswith('HTTP 500')
        assert st.router.counts['view_count_errors'] == 1
