#!/usr/bin/env python3
"""Paired cold-connection test using real Daytona session commands and CPU tokenization."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import httpx
from daytona import AsyncDaytona, DaytonaConfig
from command_ramp import Ramp, load_key, KEY_FILE, quantile


def ramp_args(args):
    return SimpleNamespace(out=args.out,run=args.run,snapshot='harbor__8c0fc945338b__snapshot',command_timeout=30,pool=2048)


async def worker(args):
    async with AsyncDaytona(DaytonaConfig(api_key=load_key(KEY_FILE),connection_pool_maxsize=2048)) as daytona:
        ramp=Ramp(ramp_args(args),daytona)
        seats=[dict(i=row['i'],sb=await daytona.get(row['id'])) for row in json.loads(Path(args.ids).read_text())]
        monitor=asyncio.create_task(ramp.monitor())
        tokens=[];commands=[]
        async with httpx.AsyncClient(timeout=10) as client:
            async def token(seat,turn):
                start=time.monotonic();row=dict(seat=seat['i'],turn=turn,ok=False)
                try:
                    response=await client.post('http://127.0.0.1:18945/tokenize',json={'messages':[{'role':'user','content':'Inspect the repository and run tests.\n'*100}]})
                    response.raise_for_status();row.update(ok=True,tokens=len(response.json()['tokens']),render_seconds=float(response.headers['X-Render-Seconds']))
                except Exception as exc:
                    row['error_type']=type(exc).__name__
                row['seconds']=time.monotonic()-start;tokens.append(row);ramp.emit('tokenize',**row)
            try:
                for turn in range(args.rounds):
                    calls=[asyncio.create_task(token(seat,turn)) for seat in seats]
                    await asyncio.sleep(.001)
                    commands.extend(await asyncio.gather(*(ramp.call(seat,'printf ramp-ok','mixed','noop',turn) for seat in seats)))
                    await asyncio.gather(*calls)
                    if turn<args.rounds-1:await asyncio.sleep(35)
            finally:
                monitor.cancel()
                ramp.f.close()
        summary=dict(mode=args.mode,seats=len(seats),loop_lag_max=max(ramp.lag,default=0))
        for name,rows in [('tokenize',tokens),('commands',commands)]:
            ts=[r['seconds'] for r in rows if r['ok']]
            summary[name]=dict(count=len(rows),errors=sum(not r['ok'] for r in rows),p50=quantile(ts,.5),p90=quantile(ts,.9),p99=quantile(ts,.99),error_types=sorted({r.get('error_type','nonzero_exit') for r in rows if not r['ok']}))
        Path(args.out,'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


async def parent(args):
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
    async with AsyncDaytona(DaytonaConfig(api_key=load_key(KEY_FILE),connection_pool_maxsize=2048)) as daytona:
        ramp=Ramp(ramp_args(args),daytona)
        try:
            await asyncio.gather(*(ramp.add_one(i) for i in range(args.seats)))
            assert len(ramp.seats)==args.seats, 'incomplete sandbox setup'
            ids=root/'ids.json';ids.write_text(json.dumps([dict(i=s['i'],id=s['sb'].id) for s in ramp.seats]))
            for mode in ['gateway','preload','gateway_repeat']:
                env=os.environ.copy()
                command=[sys.executable,str(Path(__file__).resolve()),'--worker','--mode',mode,'--ids',str(ids),'--out',str(root/mode),'--run',args.run,'--rounds',str(args.rounds)]
                if mode=='preload':
                    for key in ['HTTPS_PROXY','HTTP_PROXY','https_proxy','http_proxy','ALL_PROXY','all_proxy']:env.pop(key,None)
                    command=[args.proxychains,'-q','-f',args.proxyconf,*command]
                process=await asyncio.create_subprocess_exec(*command,env=env)
                status=await process.wait()
                if status:raise RuntimeError(f'{mode} worker failed: {status}')
        finally:
            await ramp.cleanup();ramp.f.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--run',required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--ids');p.add_argument('--mode',default='gateway')
    p.add_argument('--seats',type=int,default=64);p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--proxychains');p.add_argument('--proxyconf');a=p.parse_args()
    asyncio.run(worker(a) if a.worker else parent(a))
