"""A local OpenAI-compatible judge for tests: aiohttp on 127.0.0.1 in its own thread and event loop (so the code under
test can run its own loops, as replay does per episode). answer(body) -> (status, json) decides each reply; every body
received is kept in `bodies`. It lists `model` on /v1/models, so the relay router's health check accepts it as a named
endpoint."""
import asyncio
import threading

from aiohttp import web


def chat(content, prompt_tokens=123, completion_tokens=7):
    """A chat completion whose message content is `content`."""
    return {'id': 'judge-1', 'object': 'chat.completion', 'model': 'judge',
            'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': content}}],
            'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens,
                      'total_tokens': prompt_tokens + completion_tokens}}


def prompt_of(body):
    return body['messages'][-1]['content']


class StubJudge:
    def __init__(self, answer, delay=0.0, model='judge'):
        self.answer, self.delay, self.model = answer, delay, model
        self.bodies = []
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.url = ''

    def __enter__(self):
        self.thread.start()
        asyncio.run_coroutine_threadsafe(self._start(), self.loop).result(10)
        return self

    def __exit__(self, *exc):
        asyncio.run_coroutine_threadsafe(self.runner.cleanup(), self.loop).result(10)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(5)

    async def _start(self):
        app = web.Application()
        app.router.add_post('/v1/chat/completions', self._chat)
        app.router.add_get('/v1/models', self._models)
        self.runner = web.AppRunner(app, access_log=None)
        await self.runner.setup()
        await web.TCPSite(self.runner, '127.0.0.1', 0).start()
        self.url = 'http://127.0.0.1:%d/v1' % self.runner.addresses[0][1]

    async def _models(self, request):
        return web.json_response({'object': 'list', 'data': [{'id': self.model, 'object': 'model',
                                                               'max_model_len': 32768}]})

    async def _chat(self, request):
        body = await request.json()
        self.bodies.append(body)
        if self.delay:
            await asyncio.sleep(self.delay)
        status, data = self.answer(body)
        return web.json_response(data, status=status)
