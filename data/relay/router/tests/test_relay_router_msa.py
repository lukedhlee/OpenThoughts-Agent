"""CPU tests for the relay router's --harness msa (mini-swe-agent tool mode).

Router-level tests post the requests harbor's mini-swe-agent-host sends (system + task, then assistant turns with
`tool_calls` and `tool` results named "bash", the bash tool schema, tool_choice "none") to a router in front of
scripted student / teacher servers. The end-to-end test runs harbor's own MiniSweAgentHost (lukedhlee/mini-swe-relay:
upstream mini-swe-agent 2.4.6's loop, tool mode, the session header) against the router on a local shell.

    PYTHONPATH=<harbor lukedhlee/mini-swe-relay>/src:<harbor>:<msa 2.4.6 dir> <venv>/bin/python -m pytest \
        data/relay/router/tests/test_relay_router_msa.py -q
"""
import asyncio
import json
import re
import sys
import threading
import urllib.request
from pathlib import Path

import pytest
from aiohttp import web

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import fake_openai  # noqa: E402
import msa_tool as mt  # noqa: E402
import relay_router as rr  # noqa: E402

SYSTEM = 'You are a helpful assistant that can interact with a computer.'
NOTE = rr.VERIFY_NOTE_MSA


def task_text(sc):
    return f'Please solve this issue: SCENARIO={sc}. Make /app/out.txt contain ok.\n\nYou can execute bash commands.'


def tc_json(cmd):
    return '<tool_call>\n' + json.dumps({'name': 'bash', 'arguments': {'command': cmd}}) + '\n</tool_call>'


def qwen_xml(cmd):
    return f'<tool_call>\n<function=bash>\n<parameter=command>\n{cmd}\n</parameter>\n</function>\n</tool_call>'


SUBMIT = 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'


def student_reply(sc, n):
    """The student's n-th reply (0-based), as 09-21/H9 writes it in tool mode (thinking inline, a JSON tool call)."""
    think = f'<|start_think|>student thinking {n}<|end_think|>'
    if sc == 'done':
        return think + ['Look.' + tc_json('ls -la'), 'Write.' + tc_json('echo ok > out.txt'), 'Done.' + tc_json(SUBMIT)][min(n, 2)]
    if sc == 'loop':
        return think + 'Retry.' + tc_json('python run.py')
    if sc == 'badjson':      # a tool call never closed (autofix recovers it), then prose with no action (teacher repair)
        return [think + 'Look.' + tc_json('ls -la'),
                think + 'Grep.<tool_call>\n{"name": "bash", "arguments": {"command": "grep x f"}}\n',
                think + 'The task looks complete to me.',
                think + 'Done.' + tc_json(SUBMIT)][min(n, 3)]
    return think + f'Step {n}.' + tc_json(f'echo step-{n}')


class MsaServer(fake_openai.FakeServer):
    """Student: student_reply by scenario and assistant turns so far. Teacher: Qwen3.8 with a reasoning parser and no
    tool parser (the XML call left in content); `script` overrides its next replies."""

    async def chat(self, request):
        body = await request.json()
        self.requests.append(body)
        msgs = body['messages']
        last = fake_openai.text_of(msgs[-1].get('content'))
        if 'Print hello' in last:
            msg = ({'role': 'assistant', 'content': '<|start_think|>hi<|end_think|>echo hello'} if self.role == 'student'
                   else {'role': 'assistant', 'content': 'echo hello', 'reasoning': 'The user wants hello.'})
        elif self.role == 'student':
            sc = fake_openai.scenario_of(msgs)
            n = sum(1 for m in msgs if m.get('role') == 'assistant')
            msg = {'role': 'assistant', 'content': student_reply(sc, n)}
        else:
            k = sum(1 for m in msgs if m.get('role') == 'assistant' and 'teacher-step' in (m.get('reasoning') or ''))
            content = ('Checking.\n\n' + qwen_xml('cat out.txt')) if k == 0 else ('Finished.\n\n' + qwen_xml(SUBMIT))
            if self.script:
                item = self.script.pop(0)
                content = content if item is None else item
            msg = {'role': 'assistant', 'content': content,
                   'reasoning': f'teacher-step {k} reasoning' + ''.join(f'. Sentence {j} is here' for j in range(self.long_reasoning))}
        return web.json_response({'id': f'fake-{len(self.requests)}', 'object': 'chat.completion', 'model': self.model,
                                  'choices': [{'index': 0, 'finish_reason': 'stop', 'message': msg}],
                                  'usage': {'prompt_tokens': 100, 'completion_tokens': 10, 'total_tokens': 110}})


class Stack:
    def __init__(self, tmp, router_args=(), long_reasoning=0, budgets=None):
        self.tmp = Path(tmp)
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.router_args = list(router_args)
        self.student = MsaServer('student', 'snowball')
        self.teacher = MsaServer('teacher', 'qwen38', long_reasoning=long_reasoning)
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
        s_url, t_url = await self.student.start(), await self.teacher.start()
        tasks = [dict(task_id=f'cf-{sc}', instruction=f'SCENARIO={sc}. Make /app/out.txt contain ok.',
                      agent_timeout_sec=self.budgets.get(sc, 1000)) for sc in ('done', 'loop', 'badjson', 'work')]
        (self.tmp / 'tasks.json').write_text(json.dumps(tasks))
        self.a = rr.parse_args(['--mode', 'relay', '--harness', 'msa', '--port', '0', '--log-dir', str(self.log_dir),
                                '--tasks', str(self.tmp / 'tasks.json'), '--student-url', s_url, '--student-model', 'snowball',
                                '--teacher-url', t_url, '--teacher-model', 'qwen38', '--health-retries', '1',
                                '--health-wait', '0', '--connect-retries', '0'] + self.router_args)
        self.router = rr.Router(self.a)
        await self.router.start_http()
        self.health_ok, self.health = await self.router.health_check()
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

    def turns(self, sid=None):
        p = self.log_dir / 'turns.jsonl'
        rows = [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
        return [r for r in rows if sid is None or r['sid'] == sid]


def post(st, messages, sid):
    body = {'model': 'relay', 'messages': messages, 'tools': [{'type': 'function', 'function': {'name': 'bash'}}],
            'tool_choice': 'none', 'skip_special_tokens': False}
    req = urllib.request.Request(st.url + '/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json', 'X-Harbor-Session-Id': sid})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def observation(cmd, out='', rc=0):
    return json.dumps({'returncode': rc, 'output': out}, indent=2)


class Driver:
    """Plays mini-swe-agent tool mode against the router: appends each returned assistant message (reasoning under
    both keys, as harbor's tool_mode_message_view sends it) and one tool result per call."""

    def __init__(self, st, sc, sid=None, outputs=None):
        self.st, self.sid = st, sid or f'sid-{sc}'
        self.msgs = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': task_text(sc)}]
        self.outputs = outputs or {}
        self.replies = []
        self.submitted = False

    def step(self):
        status, resp = post(self.st, self.msgs, self.sid)
        if status != 200:
            return status, resp
        m = resp['choices'][0]['message']
        self.replies.append(m)
        if not m.get('tool_calls'):
            self.msgs.append({'role': 'user', 'content': 'Tool call error:\n\n<error>\nNo tool calls found\n</error>'})
            return status, resp
        am = {'role': 'assistant', 'content': m.get('content') or '', 'tool_calls': m['tool_calls']}
        if m.get('reasoning_content'):
            am['reasoning_content'] = am['reasoning'] = m['reasoning_content']
        self.msgs.append(am)
        for tc in m['tool_calls']:
            cmd = json.loads(tc['function']['arguments'])['command']
            if mt.SUBMIT_SENTINEL in cmd:
                self.submitted = True
            self.msgs.append({'role': 'tool', 'tool_call_id': tc['id'], 'name': 'bash',
                              'content': observation(cmd, self.outputs.get(cmd, ''))})
        return status, resp

    def run(self, n=12):
        for _ in range(n):
            st, _ = self.step()
            if self.submitted or st != 200:
                break
        return self


def test_msa_parsers():
    ids = lambda j: f'c{j}'  # noqa: E731
    p = mt.parse('<|start_think|>hmm<|end_think|>Look.' + tc_json('ls -la'), None, ids)
    assert (p.reasoning, p.content, p.error) == ('hmm', 'Look.', '')
    assert p.tool_calls == [{'id': 'c0', 'type': 'function', 'function': {'name': 'bash', 'arguments': '{"command": "ls -la"}'}}]
    assert mt.toolcall_error(p.tool_calls) == '' and not mt.is_submit(p.tool_calls)
    assert mt.is_submit(mt.parse(tc_json(SUBMIT), None, ids).tool_calls)
    assert 'No tool calls' in mt.toolcall_error(mt.parse('<|start_think|>x<|end_think|>prose', None, ids).tool_calls)
    assert mt.parse('<|start_think|>x<|end_think|><tool_call>{bad}</tool_call>', None, ids).error.startswith('Tool call 1 is not valid JSON')
    q = mt.teacher_reply({'content': 'Run it.\n\n' + qwen_xml('cat a\nb'), 'reasoning': 'R'}, ids)
    assert (q.reasoning, q.content, q.shape) == ('R', 'Run it.', 'xml')
    assert json.loads(q.tool_calls[0]['function']['arguments']) == {'command': 'cat a\nb'}
    j = mt.teacher_reply({'content': tc_json('pwd'), 'reasoning': 'R'}, ids)
    assert json.loads(j.tool_calls[0]['function']['arguments']) == {'command': 'pwd'} and j.shape == 'json'
    srv = mt.teacher_reply({'content': '', 'reasoning': 'R', 'tool_calls': [
        {'id': 'x', 'type': 'function', 'function': {'name': 'bash', 'arguments': '{"command": "ls"}'}}]}, ids)
    assert srv.tool_calls[0]['id'] == 'c0' and srv.shape == 'server'
    assert 'Unknown tool' in mt.toolcall_error(mt.teacher_reply({'content': '<tool_call><function=python><parameter=code>1'
                                                                  '</parameter></function></tool_call>'}, ids).tool_calls)
    assert rr.request_kind([{'role': 'system'}, {'role': 'user'}], msa=True) == 'initial'
    assert rr.request_kind([{'role': 'system'}, {'role': 'user'}, {'role': 'assistant'}, {'role': 'tool'}], msa=True) == 'main'


def test_student_reply_is_returned_parsed_with_router_ids(tmp_path):
    with Stack(tmp_path) as st:
        assert st.health_ok, st.health
        d = Driver(st, 'work')
        d.step()
        m = d.replies[0]
        assert m['content'] == 'Step 0.' and m['reasoning_content'] == 'student thinking 0'
        assert m['tool_calls'][0]['id'] == 'call_1_0' and json.loads(m['tool_calls'][0]['function']['arguments']) == {'command': 'echo step-0'}
        d.step()
        rows = st.turns('sid-work')
        assert [r['owner'] for r in rows] == ['student', 'student'] and [r['turn'] for r in rows] == [1, 2]
        assert rows[0]['task_id'] == 'cf-work'
        # the student saw its own history as harbor sent it
        assert st.student.requests[-1]['messages'][2]['reasoning_content'] == 'student thinking 0'


def test_done_claim_teacher_answers_the_same_request_with_the_note(tmp_path):
    with Stack(tmp_path, router_args=['--verify-note']) as st:
        d = Driver(st, 'done').run()
        rows = st.turns('sid-done')
        assert [r['owner'] for r in rows] == ['student', 'student', 'teacher', 'teacher']
        to = rows[2]
        assert to['takeover']['trigger'] == 'done_claim' and to['done_claim'] and to['verify_note']
        assert mt.SUBMIT_SENTINEL in json.dumps(to['discarded_student_reply'])
        # the student's submit never reached the harness; the teacher's check and its own submit did
        cmds = [json.loads(m['tool_calls'][0]['function']['arguments'])['command'] for m in d.replies]
        assert cmds == ['ls -la', 'echo ok > out.txt', 'cat out.txt', SUBMIT] and d.submitted
        tb = st.teacher.requests[-2]                      # the takeover request, as the teacher got it
        assert tb['messages'][-1]['role'] == 'tool' and tb['messages'][-1]['content'].endswith('\n\n' + NOTE)
        assert NOTE not in json.dumps(d.msgs)                # harbor's history never holds the note
        assert sum(NOTE in json.dumps(b['messages']) for b in st.teacher.requests[1:]) == 1
        # the teacher sees the student's turns without their thinking (strip), with their tool calls
        sturns = [m for m in tb['messages'] if m['role'] == 'assistant']
        assert all('reasoning' not in m and 'reasoning_content' not in m and m['tool_calls'] for m in sturns)
        # the teacher's own earlier turn comes back with its reasoning
        last = st.teacher.requests[-1]['messages']
        assert [m.get('reasoning') for m in last if m['role'] == 'assistant'][-1] == 'teacher-step 0 reasoning'
        assert rows[3]['teacher_guard']['outcome'] == 'ok' and rows[3]['teacher_guard']['shapes'] == ['xml']


def test_loop_takes_over_before_the_student_is_asked(tmp_path):
    with Stack(tmp_path) as st:
        out = {'python run.py': "Traceback (most recent call last):\nNameError: name 'x' is not defined"}
        Driver(st, 'loop', outputs=out).run(6)
        rows = st.turns('sid-loop')
        first_t = next(r for r in rows if r['owner'] == 'teacher')
        assert first_t['takeover']['trigger'] == 'loop' and first_t['takeover']['kind'] == 'environment'
        assert 'discarded_student_reply' not in first_t
        assert all(r['owner'] == 'teacher' for r in rows[rows.index(first_t):])


def test_autofix_then_repair_then_done_claim(tmp_path):
    with Stack(tmp_path, router_args=['--repair-on-parse-error', '--autofix']) as st:
        d = Driver(st, 'badjson').run()
        rows = st.turns('sid-badjson')
        assert [r['owner'] for r in rows][:4] == ['student', 'student', 'teacher', 'teacher']
        assert rows[1]['autofix'] and rows[1]['autofix_kind'].startswith('tool_call')
        assert rows[1]['student_parse_error'] == 'A tool call was opened but never closed.'
        assert json.loads(d.replies[1]['tool_calls'][0]['function']['arguments'])['command'] == 'grep x f'
        assert d.replies[1]['content'] == 'Grep.' and d.replies[1]['reasoning_content'] == 'student thinking 1'
        assert rows[2]['repair'] and rows[2]['repair_kind'] == 'parse_error' and 'No tool calls' in rows[2]['repair_reason']
        assert rows[3]['takeover']['trigger'] == 'done_claim' and not rows[2].get('takeover')
        # the student's next view holds the teacher's repair turn with the teacher's reasoning
        sv = st.student.requests[-1]['messages']
        rep = [m for m in sv if m['role'] == 'assistant'][2]
        assert rep['reasoning_content'] == 'teacher-step 0 reasoning' and rep['tool_calls']


def test_teacher_guard_resamples_and_passes_through(tmp_path):
    with Stack(tmp_path, router_args=['--teacher-format-guard', '--teacher-resamples', '1']) as st:
        st.teacher.script = ['I will now check the file.', None]          # no tool call, then the default XML call
        d = Driver(st, 'done').run()
        rows = st.turns('sid-done')
        g = rows[2]['teacher_guard']
        assert g['outcome'] == 'resampled_ok' and g['attempts'] == 2 and len(rows[2]['teacher_resampled_replies']) == 1
        st.teacher.script = ['no call', 'still no call']
        d2 = Driver(st, 'done', sid='sid-done2').run(3)
        g2 = st.turns('sid-done2')[2]['teacher_guard']
        assert g2['outcome'] == 'unparseable_passed' and not d2.replies[2].get('tool_calls')


def test_older_teacher_reasoning_is_cut_in_the_student_view(tmp_path):
    from test_relay_router import _word_tokenizer
    tok = _word_tokenizer(tmp_path / 'tok.json')
    with Stack(tmp_path, router_args=['--repair-on-parse-error', '--student-tokenizer', tok, '--reasoning-cap', '20'],
               long_reasoning=30) as st:
        st.teacher.script = ['Check.\n\n' + qwen_xml('ls'), 'Check again.\n\n' + qwen_xml('pwd')]
        Driver(st, 'badjson').run(4)
        # two repairs give two teacher turns in the student's history: the older is cut, the latest whole
        sv = st.student.requests[-1]['messages']
        tt = [m for m in sv if m['role'] == 'assistant' and 'teacher-step' in (m.get('reasoning_content') or '')]
        assert len(tt) == 2
        assert len(tt[0]['reasoning_content'].split()) <= 20 < len(tt[1]['reasoning_content'].split())
        assert st.turns('sid-badjson')[3]['student_view']['cuts']


def test_hard_end_after_takeover_and_synthetic_submit_at_budget(tmp_path):
    with Stack(tmp_path, router_args=['--student-row-max-tokens', '400', '--student-row-reserve', '100']) as st:
        st.student.count_fn = lambda msgs: 100 * len(msgs)
        d = Driver(st, 'done').run()
        rows = st.turns('sid-done')
        assert rows[-1]['owner'] == 'router' and rows[-1]['ending'] == 'context_hard_end'
        assert 'maximum context length' in rows[-1]['upstream_error']
    with Stack(tmp_path / 'b', router_args=['--budget-mode', 'on'], budgets={'work': 0.001}) as st:
        d = Driver(st, 'work').run(3)
        assert d.submitted and st.turns('sid-work')[-1]['ending'] == 'student_budget'


def test_student_view_after_takeover_drops_the_students_thinking(tmp_path):
    with Stack(tmp_path, router_args=['--student-row-max-tokens', '100000', '--student-view-after-takeover', 'strip']) as st:
        Driver(st, 'done').run()
        assert st.turns('sid-done')[3]['student_view_tokens'] is not None     # counted after the takeover
        tok_bodies = [b for b in st.student.tokenize_requests if any(m.get('role') == 'assistant' for m in b['messages'])]
        last = tok_bodies[-1]['messages']
        assert [m['tool_calls'][0]['id'] for m in last if m['role'] == 'assistant'][:2] == ['call_1_0', 'call_2_0']
        assert not any(m.get('reasoning_content') for m in last if m['role'] == 'assistant'
                       and m['tool_calls'][0]['id'] in ('call_1_0', 'call_2_0'))


# ---- end to end: harbor's MiniSweAgentHost against the router ----------------------------------------------------
try:
    import inspect

    from harbor.agents.mini_swe_agent_host.agent import MiniSweAgentHost
    from harbor.models.agent.context import AgentContext
    import importlib.util
    import harbor
    _hr = Path(harbor.__file__).resolve().parents[2] / 'tests' / 'unit' / 'agents' / 'mini_swe_agent_host' / 'host_replay.py'
    sys.path.insert(0, str(_hr.parents[4]))      # host_replay imports tests.unit.agents... from the harbor checkout
    _spec = importlib.util.spec_from_file_location('harbor_host_replay', _hr)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    LocalShellEnvironment = _mod.LocalShellEnvironment
    HAVE_MSA = 'llm_session_header' in inspect.signature(MiniSweAgentHost.__init__).parameters
except Exception:  # noqa: BLE001
    HAVE_MSA = False


@pytest.mark.skipif(not HAVE_MSA, reason='harbor lukedhlee/mini-swe-relay + mini-swe-agent 2.4.6 not on PYTHONPATH')
def test_end_to_end_mini_swe_agent_host_through_the_router(tmp_path):
    work = tmp_path / 'app'
    work.mkdir()
    with Stack(tmp_path, router_args=['--verify-note', '--repair-on-parse-error', '--autofix']) as st:
        agent = MiniSweAgentHost(logs_dir=tmp_path / 'logs', model_name='hosted_vllm/relay', api_base=st.url,
                                 config_file='mini.yaml', model_class='litellm',
                                 config_overrides={'environment': {'cwd': str(work)}},
                                 model_info={'max_input_tokens': 65536, 'max_output_tokens': 8192,
                                             'input_cost_per_token': 0.0, 'output_cost_per_token': 0.0},
                                 llm_call_kwargs={'tool_choice': 'none'}, extra_body={'skip_special_tokens': False},
                                 interleaved_thinking=True, llm_session_header='X-Harbor-Session-Id')
        env = LocalShellEnvironment(work)
        ctx = AgentContext()

        async def go():
            await agent.setup(env)
            await agent.run('SCENARIO=done. Make /app/out.txt contain ok.', env, ctx)
        asyncio.run(go())
        rows = st.turns()
        assert [r['owner'] for r in rows] == ['student', 'student', 'teacher', 'teacher']
        assert rows[2]['takeover']['trigger'] == 'done_claim' and rows[2]['verify_note']
        assert rows[0]['sid'] == ctx.metadata['session_id']
        assert ctx.metadata['exit_status'] == 'Submitted' and ctx.metadata['n_format_errors'] == 0
        assert (work / 'out.txt').read_text().strip() == 'ok'
        msgs = ctx.metadata['all_messages']
        assert [m['role'] for m in msgs].count('tool') == 3      # the submit raises before its observation
        # harbor's history kept the router's ids, and the tool results are named after their calls
        assert [m['tool_calls'][0]['id'] for m in msgs if m['role'] == 'assistant'] == [
            'call_1_0', 'call_2_0', 'call_3_0', 'call_4_0']
        # the teacher's reasoning reached harbor (re-sent in the next request under both keys)
        last_req = st.teacher.requests[-1]['messages']
        assert [m.get('reasoning') for m in last_req if m['role'] == 'assistant'][-1] == 'teacher-step 0 reasoning'


def test_teacher_hint_reaches_only_the_teachers_system_message(tmp_path):
    hint = 'Keep command output short.'
    with Stack(tmp_path, router_args=['--teacher-hint-text', hint]) as st:
        d = Driver(st, 'done').run()
        assert d.submitted
        treqs = [b for b in st.teacher.requests if b.get('tools')]          # episode requests (not the health probe)
        assert treqs and all(b['messages'][0]['role'] == 'system' and b['messages'][0]['content'].endswith('\n\n' + hint)
                             for b in treqs)
        assert hint not in json.dumps(d.msgs)                                  # harbor's history never holds it
        assert all(hint not in json.dumps(b.get('messages')) for b in st.student.requests)
