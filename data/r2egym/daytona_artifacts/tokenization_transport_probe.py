#!/usr/bin/env python3
"""Isolate client-loop starvation from CPU rendering and external HTTP latency."""
import argparse
import asyncio
import collections
import json
import os
from pathlib import Path
import time

import aiohttp
from aiohttp import web
import httpx


def quantile(xs, p):
    xs = sorted(xs)
    return round(xs[int((len(xs)-1)*p)], 4) if xs else None


async def serve(args):
    print('loading tokenizer',flush=True)
    from transformers import AutoTokenizer
    print('transformers imported',flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    async def tokenize(request):
        payload = await request.json()
        start = time.monotonic()
        tokens = await asyncio.to_thread(tokenizer.apply_chat_template, payload['messages'],
                                        tokenize=True, add_generation_prompt=True, return_dict=False)
        return web.json_response({'tokens': tokens}, headers={'X-Render-Seconds': str(time.monotonic()-start)})
    app = web.Application()
    app.router.add_post('/tokenize', tokenize)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, '127.0.0.1', args.port).start()
    Path(args.ready).touch()
    await asyncio.Event().wait()


async def probe(args):
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    events = []; lag = []; stop = asyncio.Event()
    async def monitor():
        while not stop.is_set():
            start=time.monotonic(); await asyncio.sleep(.05)
            lag.append(max(0,time.monotonic()-start-.05))
    async def token(client, seat, turn):
        start=time.monotonic(); spans=[]
        async def trace(name, info):
            spans.append([name, round(time.monotonic()-start,6)])
        row=dict(kind='tokenize',seat=seat,turn=turn,ok=False)
        try:
            result=await client.post(f'http://127.0.0.1:{args.port}/tokenize',
                json={'messages':[{'role':'user','content':('Inspect the repository, run tests, and explain the failure.\n' * 100)}]},
                extensions={'trace':trace})
            result.raise_for_status()
            row.update(ok=True, tokens=len(result.json()['tokens']),render_seconds=float(result.headers['X-Render-Seconds']))
        except Exception as exc:
            row['error_type']=type(exc).__name__
        row.update(seconds=time.monotonic()-start,trace=spans)
        events.append(row)
    async def external(session, seat, turn):
        start=time.monotonic(); row=dict(kind='external',seat=seat,turn=turn,ok=False)
        try:
            async with session.get('https://app.daytona.io/api/health') as result:
                await result.read()
                # HTTP status is recorded; this probe measures transport, not API correctness.
                row.update(ok=True,status=result.status)
        except Exception as exc:
            row['error_type']=type(exc).__name__
        row['seconds']=time.monotonic()-start;events.append(row)
    if args.mode=='native':
        from aiohttp_socks import ProxyConnector
        connector=ProxyConnector(host='10.128.1.2',port=7011,username=os.environ['SOCKS_USER'],
                                 password=os.environ['SOCKS_PASS'],rdns=False,limit=2048,
                                 force_close=args.cold)
    else:
        connector=aiohttp.TCPConnector(limit=2048,force_close=args.cold)
    watcher=asyncio.create_task(monitor())
    async with httpx.AsyncClient(timeout=10.0) as client, aiohttp.ClientSession(
        connector=connector,timeout=aiohttp.ClientTimeout(total=40)) as session:
        for turn in range(args.rounds):
            tasks=[asyncio.create_task(token(client,i,turn)) for i in range(args.seats)]
            # Tokenization's local TCP requests are already in flight when external connections open.
            await asyncio.sleep(.001)
            if args.mode!='baseline':
                tasks.extend(asyncio.create_task(external(session,i,turn)) for i in range(args.seats))
            await asyncio.gather(*tasks)
            await asyncio.sleep(1)
    stop.set();await watcher
    summary=dict(mode=args.mode,cold=args.cold,seats=args.seats,rounds=args.rounds,
                 loop_lag_max=max(lag,default=0),loop_lag_p99=quantile(lag,.99))
    for kind in ['tokenize','external']:
        rows=[r for r in events if r['kind']==kind]; times=[r['seconds'] for r in rows if r['ok']]
        summary[kind]=dict(count=len(rows),errors=dict(collections.Counter(r['error_type'] for r in rows if not r['ok'])),
            p50=quantile(times,.5),p90=quantile(times,.9),p99=quantile(times,.99),
            render_p99=quantile([r['render_seconds'] for r in rows if 'render_seconds' in r],.99))
    out.write_text(json.dumps(dict(summary=summary,events=events),indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--serve',action='store_true')
    p.add_argument('--model',default='/e/fscratch/reformo/lee27/models/snowball-s3-nemotron-terminal-step1888')
    p.add_argument('--port',type=int,default=18945);p.add_argument('--ready',default='tokenizer-ready')
    p.add_argument('--mode',choices=['baseline','preload','native'],default='baseline')
    p.add_argument('--cold',action='store_true');p.add_argument('--seats',type=int,default=64)
    p.add_argument('--rounds',type=int,default=4);p.add_argument('--out',default='transport.json')
    args=p.parse_args();asyncio.run(serve(args) if args.serve else probe(args))
