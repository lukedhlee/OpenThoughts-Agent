#!/usr/bin/env python3
"""Scripted OpenAI-compatible student and teacher servers for the relay router's CPU tests.

Pattern of data/mini_swe_host/scripted_openai.py, but stateless per request so concurrent episodes cannot bleed into
each other: the student's reply is chosen by the SCENARIO=<name> tag in the conversation's first message and by how
many assistant turns the request already holds; the teacher's reply by how many of its own turns ("teacher-step")
the request holds. Every request body is kept in memory (``server.requests``) for the tests to inspect.

Student replies look like the served Snowball: `<|start_think|>...<|end_think|>{Terminus-2 JSON}` in `content`, no
reasoning field. Teacher replies look like vLLM with a reasoning parser: JSON in `content`, the thinking under
`reasoning` (or `reasoning_content` with reasoning_key='reasoning_content').

    python fake_openai.py --role student --port 18001 --model snowball
"""
import argparse
import asyncio
import json
import re

from aiohttp import web

SUMMARY_PREFIX = 'You are about to hand off your work to another AI agent.'
QUESTIONS_PREFIX = 'You are picking up work from a previous AI agent on this task:'
ANSWERS_PREFIX = 'The next agent has a few questions for you'
CONFIRM_MARK = 'Are you sure you want to mark the task as complete?'


def t2(analysis, cmds=(), done=False, plan='next'):
    return json.dumps(dict(analysis=analysis, plan=plan, task_complete=done,
                           commands=[dict(keystrokes=k, duration=0.1) for k in cmds]))


def text_of(c):
    if isinstance(c, list):
        return '\n'.join(p.get('text', '') for p in c if isinstance(p, dict))
    return c or ''


DEFAULT_SCENARIO = ['selfdone']


def scenario_of(messages):
    for m in messages:
        mm = re.search(r'SCENARIO=(\w+)', text_of(m.get('content')))
        if mm:
            return mm.group(1)
    return DEFAULT_SCENARIO[0]


def student_json(scenario, n):
    """The student's n-th reply (0-based) in a scenario."""
    if scenario == 'selfdone':      # works, claims done, confirms
        return [t2('look', ['ls -la\n']), t2('write', ['echo ok > out.txt\n']), t2('done', [], True),
                t2('sure', [], True)][min(n, 3)]
    if scenario == 'done':          # claims done at its 3rd reply
        return [t2('look', ['ls -la\n']), t2('write', ['echo hi > out.txt\n']), t2('I am done', [], True),
                t2('confirm', [], True)][min(n, 3)]
    if scenario == 'loop':          # the same failing command forever
        return t2(f'retry {n}', ['python run.py\n'])
    if scenario == 'wait':          # starts a build, then waits on a silent screen
        return t2('build', ['make build\n']) if n == 0 else t2(f'wait {n}', [])
    if scenario in ('budget', 'summ'):   # varied work, never done
        return t2(f'step {n}', [f'echo step-{n}\n'])
    bad = (f'<|start_think|>BADFORMAT {n}: I will call the tool.<|end_think|>'
           '<tool_call>{"keystrokes": "ls -la\\n", "duration": 0.1}</tool_call>')
    if scenario == 'repair':        # one unparseable reply, then valid work and a done claim
        return [t2('look', ['ls -la\n']), bad, t2('write', ['echo hi > out.txt\n']), t2('I am done', [], True),
                t2('confirm', [], True)][min(n, 4)]
    if scenario == 'repairs':       # two unparseable replies in a row
        return [t2('look', ['ls -la\n']), bad, bad, t2('write', ['echo hi > out.txt\n']), t2('I am done', [], True),
                t2('confirm', [], True)][min(n, 5)]
    if scenario == 'autofix':       # a fixable tool_call, then prose with no action (teacher repair), then done
        return [t2('look', ['ls -la\n']),
                '<|start_think|>AUTOFIX me: list the files.<|end_think|>I will list the files.<tool_call>'
                '{"name": "bash", "arguments": {"command": "echo hi > out.txt"}}</tool_call>',
                '<|start_think|>NOACTION prose only<|end_think|>The task looks complete to me.',
                t2('I am done', [], True), t2('confirm', [], True)][min(n, 4)]
    if scenario == 'gaveup':        # gives up in words at its 3rd reply (gave_up is a decision trigger when enabled)
        if n == 2:
            return t2('The requirement is impossible without internet, so we cannot solve the task as specified.',
                      ['echo stub > out.txt\n'])
        return t2(f'step {n}', [f'echo step-{n}\n'])
    return t2('noop', ['true\n'])


def teacher_json(k, last_user):
    """The teacher's k-th own reply (0-based): check once, work, claim, confirm."""
    if CONFIRM_MARK in last_user and k >= 1:
        return t2('teacher-step confirm', [], True)
    if CONFIRM_MARK in last_user:
        return t2(f'teacher-step {k}: verify before confirming', ['cat out.txt\n'])
    if k < 2:
        return t2(f'teacher-step {k}', [f'echo teacher-{k}\n'])
    return t2(f'teacher-step {k}: finished', [], True)


class FakeServer:
    def __init__(self, role, model, reasoning_key='reasoning', max_model_len=65536, delay=0.0, long_reasoning=0):
        self.role, self.model, self.reasoning_key = role, model, reasoning_key
        self.max_model_len, self.delay = max_model_len, delay
        self.long_reasoning = long_reasoning     # sentences appended to the teacher's reasoning (the cap tests)
        self.cut_first = False                   # the teacher's first real reply ends at max_tokens (finish length)
        self.requests = []          # chat bodies
        self.tokenize_requests = []
        self.summarized = set()
        self.fail_model = None      # set to make chat answer 404 (a served-name change mid-run)
        self.runner = None
        self.url = None

    def app(self):
        app = web.Application(client_max_size=256 * 1024 * 1024)
        app.router.add_get('/v1/models', self.models)
        app.router.add_post('/v1/chat/completions', self.chat)
        app.router.add_post('/tokenize', self.tokenize)
        return app

    async def models(self, request):
        return web.json_response({'object': 'list', 'data': [
            {'id': self.model, 'object': 'model', 'max_model_len': self.max_model_len}]})

    async def tokenize(self, request):
        body = await request.json()
        self.tokenize_requests.append(body)
        msgs = body.get('messages') or []
        count = sum(len(json.dumps(m)) for m in msgs) // 4
        first = text_of(msgs[0].get('content')) if msgs else ''
        last = text_of(msgs[-1].get('content')) if msgs else ''
        # 'summ' scenario: report a nearly full context once, at the 4th agent turn, to force Terminus-2's summarization
        if (scenario_of(msgs) == 'summ' and len(msgs) >= 7 and first not in self.summarized
                and not last.startswith((SUMMARY_PREFIX, QUESTIONS_PREFIX, ANSWERS_PREFIX))):
            count = 32000
        return web.json_response({'count': count, 'max_model_len': self.max_model_len, 'tokens': []})

    async def chat(self, request):
        body = await request.json()
        self.requests.append(body)
        if self.fail_model or body.get('model') != self.model:
            return web.json_response({'error': {'message': f"The model `{body.get('model')}` does not exist.",
                                                'type': 'NotFoundError', 'code': 404}}, status=404)
        if self.delay:
            await asyncio.sleep(self.delay)
        msgs = body['messages']
        prompt = sum(len(json.dumps(m)) for m in msgs) // 4
        if body.get('max_tokens') and prompt + body['max_tokens'] > self.max_model_len:   # vLLM's check and wording
            return web.json_response({'error': {'message': (
                f"This model's maximum context length is {self.max_model_len} tokens. However, you requested "
                f"{body['max_tokens']} output tokens and your prompt contains at least {prompt + 1} input tokens, for a "
                f"total of at least {prompt + 1 + body['max_tokens']} tokens."), 'type': 'BadRequestError', 'code': 400}},
                status=400)
        last = text_of(msgs[-1].get('content'))
        first = text_of(msgs[0].get('content'))
        reasoning = None
        if last.startswith(SUMMARY_PREFIX):
            self.summarized.add(first)
            content = f'{self.role} summary: worked on the task.'
        elif last.startswith(QUESTIONS_PREFIX):
            content = f'{self.role} questions: 1. What is in out.txt?'
        elif last.startswith(ANSWERS_PREFIX):
            content = f'{self.role} answers: out.txt holds ok.'
        elif self.role == 'student':
            n = sum(1 for m in msgs if m.get('role') == 'assistant' and '"analysis"' in text_of(m.get('content')))
            j = student_json(scenario_of(msgs), n)
            content = j if j.startswith('<|start_think|>') else f'<|start_think|>student thinking {n}<|end_think|>' + j
        else:
            k = sum(1 for m in msgs if m.get('role') == 'assistant' and 'teacher-step' in text_of(m.get('content')))
            content = teacher_json(k, last)
            reasoning = f'teacher reasoning {k}' + ''.join(f'. Sentence {j} of step {k} is here'
                                                            for j in range(self.long_reasoning)) + ('.' if self.long_reasoning else '')
        msg = {'role': 'assistant', 'content': content}
        finish = 'stop'
        if reasoning is not None:
            msg[self.reasoning_key] = reasoning
            if self.cut_first and 'Print hello' not in last:
                self.cut_first = False
                msg, finish = {'role': 'assistant', 'content': '', self.reasoning_key: reasoning + ' and then I would'}, 'length'
        if 'Print hello' in last:   # the router's health smoke
            msg = {'role': 'assistant', 'content': ('<|start_think|>hi<|end_think|>echo hello' if self.role == 'student'
                                                    else 'echo hello')}
            if self.role == 'teacher':
                msg[self.reasoning_key] = 'The user wants hello.'
        return web.json_response({
            'id': f'fake-{len(self.requests)}', 'object': 'chat.completion', 'model': self.model,
            'choices': [{'index': 0, 'finish_reason': finish, 'message': msg}],
            'usage': {'prompt_tokens': 100, 'completion_tokens': 10, 'total_tokens': 110}})

    async def start(self, port=0, host='127.0.0.1'):
        self.runner = web.AppRunner(self.app(), access_log=None)
        await self.runner.setup()
        site = web.TCPSite(self.runner, host, port)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.url = f'http://{host}:{port}/v1'
        return self.url

    async def stop(self):
        if self.runner:
            await self.runner.cleanup()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--role', choices=['student', 'teacher'], required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--reasoning-key', default='reasoning')
    p.add_argument('--scenario', default='selfdone', help='student script for conversations without a SCENARIO= tag')
    a = p.parse_args()
    DEFAULT_SCENARIO[0] = a.scenario
    srv = FakeServer(a.role, a.model, a.reasoning_key)
    web.run_app(srv.app(), host='127.0.0.1', port=a.port, access_log=None)


if __name__ == '__main__':
    main()
