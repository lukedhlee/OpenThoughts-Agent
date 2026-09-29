#!/usr/bin/env python3
"""OpenAI-compatible pass-through proxy that shows the model its own history the way the relay SFT rows showed it.

    python think_cut_proxy.py --upstream http://<node>:8000/v1 --port 18123 --tokenizer <09-21>/tokenizer.json --log <jsonl>

Every assistant message except the latest one has the thinking inside its <|start_think|>...<|end_think|> span cut to
its first --cap (1,000) tokens at a sentence/line boundary, with router/reasoning_cap.cut_reasoning (the same function
the relay router and render.py used, so the text matches the training rows' history). The latest assistant message,
the user/tool messages and every other field pass unchanged; any other path (GET /models, ...) is forwarded as is.
A diagnostic for the relay SFT eval (2026-09-28): in training every older turn's reasoning was cut to 1,000 tokens,
at eval it stays whole and fills the 65k context. One JSON line per chat request: messages, cuts, chars removed.
Login-node safe: one process, one tokenizer thread.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault('RAYON_NUM_THREADS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'router'))
import reasoning_cap as rcap  # noqa: E402
from aiohttp import ClientSession, ClientTimeout, TCPConnector, web  # noqa: E402

START, END = '<|start_think|>', '<|end_think|>'


def cut_history(messages, tok, cap):
    last = max((i for i, m in enumerate(messages) if m.get('role') == 'assistant'), default=None)
    cuts, removed = 0, 0
    for i, m in enumerate(messages):
        c = m.get('content')
        if m.get('role') != 'assistant' or i == last or not isinstance(c, str) or START not in c or END not in c:
            continue
        a = c.index(START) + len(START)
        b = c.index(END, a)
        new, at = rcap.cut_reasoning(c[a:b], tok, cap)
        if at is not None:
            m['content'] = c[:a] + new + c[b:]
            cuts += 1
            removed += (b - a) - len(new)
    return cuts, removed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--upstream', required=True, help='http://host:port/v1')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--cap', type=int, default=rcap.CAP_TOKENS)
    ap.add_argument('--log', required=True)
    a = ap.parse_args()
    tok = rcap.load_tokenizer(a.tokenizer)
    up = a.upstream.rstrip('/')
    base = up[:-3] if up.endswith('/v1') else up
    logf = open(a.log, 'a', buffering=1)

    async def on_start(app):
        app['s'] = ClientSession(connector=TCPConnector(limit=1024), timeout=ClientTimeout(total=None, sock_read=3600))

    async def on_stop(app):
        await app['s'].close()

    async def handle(req):
        body = await req.read()
        rec = None
        if req.method == 'POST' and req.path.endswith('/chat/completions'):
            j = json.loads(body)
            n, removed = cut_history(j.get('messages') or [], tok, a.cap)
            rec = dict(t=round(time.time(), 1), messages=len(j.get('messages') or []), cuts=n, chars_removed=removed)
            body = json.dumps(j).encode()
        headers = {k: v for k, v in req.headers.items() if k.lower() not in ('host', 'content-length', 'transfer-encoding')}
        async with req.app['s'].request(req.method, base + req.path_qs, data=body, headers=headers) as r:
            out = web.StreamResponse(status=r.status, headers={k: v for k, v in r.headers.items()
                                                               if k.lower() not in ('content-length', 'transfer-encoding', 'content-encoding')})
            await out.prepare(req)
            async for chunk in r.content.iter_any():
                await out.write(chunk)
            await out.write_eof()
        if rec is not None:
            rec['status'] = r.status
            logf.write(json.dumps(rec) + '\n')
        return out

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.on_startup.append(on_start)
    app.on_cleanup.append(on_stop)
    app.router.add_route('*', '/{tail:.*}', handle)
    web.run_app(app, host=a.host, port=a.port, access_log=None, print=lambda *x: None)


if __name__ == '__main__':
    main()
