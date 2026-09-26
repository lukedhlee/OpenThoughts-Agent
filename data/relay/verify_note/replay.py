#!/usr/bin/env python3
"""replay.py — send each done_claim takeover request to Qwen3.8 twice (A unchanged, B + verification note), N samples
each, and save every reply. Runs on the serve node (snowball-v2 python, aiohttp) against the per-GPU servers.

Body = the logged sent_body (model qwen38, the teacher's view of the history, no sampler keys: Qwen3.8's
generation_config, as the router sent it). max_tokens = the router's 32,768 cap for every run (run 3b sent none, the
check run 16,384); on a context-length 400 the request is sent again without max_tokens, exactly as
relay_router.py answer() does (vLLM then generates into what the context has left).

Per takeover: one max_tokens=1 warm request primes the prefix cache, then the 2 x N samples go to the same server.
Output lines are appended as they finish; a rerun skips (key, variant, sample) already present.

    python replay.py --requests requests.jsonl.gz --out replies.jsonl --urls http://127.0.0.1:8000/v1,...
"""
import argparse
import asyncio
import copy
import gzip
import json
import os
import time

import aiohttp

NOTE = ("Note: another agent did the previous work, and its claim that the task is complete may be wrong. Before "
        "confirming, run commands that check the task's key requirements (outputs, files, tests).")
CAP = 32768


def variant(body, v):
    b = copy.deepcopy(body)
    b['max_tokens'] = CAP
    if v == 'B':
        last = b['messages'][-1]
        assert last['role'] == 'user' and 'Are you sure you want to mark the task as complete?' in last['content']
        last['content'] = last['content'] + '\n\n' + NOTE
    return b


async def post(http, url, body):
    for attempt in range(6):
        try:
            async with http.post(url + '/chat/completions', json=body) as r:
                return r.status, await r.read()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last = e
            await asyncio.sleep(5 * (attempt + 1))
    return 599, repr(last).encode()


async def complete(http, url, body):
    status, data = await post(http, url, body)
    dropped = False
    if status == 400 and b'maximum context length' in data and 'max_tokens' in body:
        body = {k: v for k, v in body.items() if k != 'max_tokens'}
        dropped = True
        status, data = await post(http, url, body)
    return status, data, dropped


async def ntok(http, url, text):
    if not text:
        return 0
    try:
        async with http.post(url.rsplit('/v1', 1)[0] + '/tokenize',
                             json={'model': 'qwen38', 'prompt': text, 'add_special_tokens': False}) as r:
            return (await r.json()).get('count')
    except Exception:  # noqa: BLE001
        return None


async def run_one(http, url, req, n, done, fo, sem, stats):
    todo = [(v, s) for v in 'AB' for s in range(n) if (req['key'], v, s) not in done]
    if not todo:
        return
    async with sem:
        w = variant(req['body'], 'A')
        w['max_tokens'] = 1
        await complete(http, url, w)

        async def one(v, s):
            t0 = time.time()
            status, data, dropped = await complete(http, url, variant(req['body'], v))
            rec = dict(key=req['key'], variant=v, sample=s, status=status, max_tokens_dropped=dropped,
                       latency_sec=round(time.time() - t0, 2), url=url)
            if status == 200:
                d = json.loads(data)
                m = d['choices'][0]['message']
                rec.update(content=m.get('content'), reasoning=m.get('reasoning_content') or m.get('reasoning'),
                           finish_reason=d['choices'][0].get('finish_reason'), usage=d.get('usage'))
                rec['reasoning_tokens'] = await ntok(http, url, rec['reasoning'])
            else:
                rec['error'] = data[:1000].decode('utf-8', 'replace')
                stats['errors'] += 1
            fo.write(json.dumps(rec) + '\n')
            fo.flush()
            stats['done'] += 1
        await asyncio.gather(*(one(v, s) for v, s in todo))


async def main():
    p = argparse.ArgumentParser()
    p.add_argument('--requests', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--urls', required=True)
    p.add_argument('--n', type=int, default=4)
    p.add_argument('--per-server', type=int, default=12, help='takeovers in flight per server (x 2N requests each)')
    a = p.parse_args()
    reqs = [json.loads(l) for l in gzip.open(a.requests, 'rt')]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            d = json.loads(l)
            if d.get('status') == 200:
                done.add((d['key'], d['variant'], d['sample']))
    urls = [u.strip().rstrip('/') for u in a.urls.split(',') if u.strip()]
    # longest prompts first, round-robin over servers so each gets a similar mix
    reqs.sort(key=lambda r: -sum(len(m.get('content') or '') for m in r['body']['messages']))
    sems = {u: asyncio.Semaphore(a.per_server) for u in urls}
    stats = dict(done=0, errors=0)
    total = sum(1 for r in reqs for v in 'AB' for s in range(a.n) if (r['key'], v, s) not in done)
    print(f'{len(reqs)} takeovers, {total} replies to collect, {len(urls)} servers', flush=True)
    t0 = time.time()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=3600),
                                     connector=aiohttp.TCPConnector(limit=0)) as http:
        with open(a.out, 'a') as fo:
            tasks = [asyncio.create_task(run_one(http, urls[i % len(urls)], r, a.n, done, fo, sems[urls[i % len(urls)]], stats))
                     for i, r in enumerate(reqs)]

            async def progress():
                while True:
                    await asyncio.sleep(60)
                    print(f'[{time.time() - t0:.0f}s] {stats["done"]}/{total} replies, {stats["errors"]} errors', flush=True)
            pr = asyncio.create_task(progress())
            await asyncio.gather(*tasks)
            pr.cancel()
    print(f'DONE {stats["done"]}/{total} replies, {stats["errors"]} errors in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
