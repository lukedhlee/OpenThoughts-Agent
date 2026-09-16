#!/usr/bin/env python3
"""Bounded command-serving ramp using the established creation harness.

No model inference. One persistent process session and tmux terminal per seat.
Uses Harbor's execute/poll/log protocol, without hiding failed calls in retries.
Only this run's unique harbor.instance label is ever deleted.
"""
import argparse
import asyncio
import collections
import json
import importlib.metadata
import hashlib
import os
from pathlib import Path
import random
import resource
import shlex
import signal
import sys
import time
from types import SimpleNamespace

from ramp_stress import Stress, load_key, KEY_FILE


def quantile(xs, p):
    xs = sorted(xs)
    return round(xs[min(len(xs)-1, int((len(xs)-1)*p))], 4) if xs else None


class PacedAPI:
    def __init__(self, api):
        self.api, self.next = api, 0.0

    def __getattr__(self, name):
        return getattr(self.api, name)

    async def create_sandbox_with_http_info(self, *args, **kwargs):
        now = time.monotonic()
        slot = max(now, self.next)
        self.next = slot + .2
        await asyncio.sleep(max(0, slot-now))
        return await self.api.create_sandbox_with_http_info(*args, **kwargs)


class Ramp:
    def __init__(self, args, daytona):
        self.a, self.d = args, daytona
        self.path = Path(args.out)
        self.path.mkdir(parents=True, exist_ok=True)
        self.f = (self.path / 'events.jsonl').open('a', buffering=1)
        self.seats, self.summaries = [], []
        self.inflight, self.peak, self.lag = 0, 0, []
        self.stop = asyncio.Event()
        sa = SimpleNamespace(run=args.run, snapshot=args.snapshot, auto_stop=60, ttl=60,
            create_http_timeout=90, retry_mode='header', retry_jitter=1,
            start_timeout=180, max_wait_limit_errors=50)
        self.stress = Stress(sa, daytona, PacedAPI(daytona._sandbox_api))

    def emit(self, kind, **kw):
        row = dict(kind=kind, utc_epoch=time.time(), **kw)
        self.f.write(json.dumps(row)+'\n')
        return row

    async def monitor(self):
        while True:
            t = time.monotonic()
            await asyncio.sleep(.25)
            self.lag.append(max(0, time.monotonic()-t-.25))

    async def call(self, seat, command, stage, cmdkind, ordinal):
        from daytona import SessionExecuteRequest
        t = time.monotonic()
        self.inflight += 1
        self.peak = max(self.peak, self.inflight)
        row = dict(stage=stage, seat=seat['i'], command_kind=cmdkind, ordinal=ordinal,
                   start_epoch=time.time(), api_calls=0, ok=False)
        async def operation():
            p = seat['sb'].process
            row['api_calls'] += 1
            response = await p.execute_session_command('seat-ramp',
                SessionExecuteRequest(command='timeout 20 bash -c '+shlex.quote(command), run_async=True), timeout=25)
            if response.cmd_id is None:
                raise RuntimeError('missing command id')
            while True:
                row['api_calls'] += 1
                status = await p.get_session_command('seat-ramp', response.cmd_id, request_timeout=25)
                if status.exit_code is not None:
                    break
                await asyncio.sleep(.001+random.random()*.001)
            row['api_calls'] += 1
            logs = await p.get_session_command_logs('seat-ramp', response.cmd_id, request_timeout=25)
            row['exit_code'] = status.exit_code
            row['output_bytes'] = len((logs.stdout or '').encode())+len((logs.stderr or '').encode())
            row['ok'] = status.exit_code == 0
            if not row['ok']:
                row['stderr'] = (logs.stderr or '')[-200:]
        try:
            await asyncio.wait_for(operation(), self.a.command_timeout)
        except Exception as exc:
            # Never serialize an SDK exception body: it can include credentials/URLs.
            row['error_type'] = type(exc).__name__
            row['status'] = getattr(exc, 'status', None) or getattr(exc, 'status_code', None)
        finally:
            self.inflight -= 1
            row['seconds'] = time.monotonic()-t
            self.emit('command', **row)
        return row

    async def add_one(self, i):
        await self.stress.open_one(i, 'open', time.time())
        row = next(r for r in self.stress.rows if r['idx'] == i)
        self.emit('create', seat=i, started=bool(row.get('t_started')),
                  seconds=(row.get('t_started', time.time())-row['t_sched']),
                  retries_429=row.get('n429', 0), sandbox_id=row.get('id'))
        if not row.get('t_started'):
            return
        try:
            sb = await asyncio.wait_for(self.d.get(row['id']), 40)
            await asyncio.wait_for(sb.process.create_session('seat-ramp'), 40)
            seat = dict(i=i, sb=sb)
            init = await self.call(seat, "tmux new-session -d -s ramp; tmux set-option -t ramp history-limit 2000",
                                   'setup', 'tmux_init', 0)
            if init['ok']:
                self.seats.append(seat)
        except Exception as exc:
            self.emit('setup_error', seat=i, error_type=type(exc).__name__)

    async def plateau(self, n, tag, seconds):
        start = time.monotonic()
        def cpu_seconds():
            return sum(r.ru_utime+r.ru_stime for r in
                       (resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN)))
        cpu_start=cpu_seconds()
        self.peak, self.lag = 0, []
        rows = []
        commands = [
            ('noop', "printf ramp-ok"),
            ('terminal_send', "tmux send-keys -t ramp 'printf command-ok; pwd; ls /tmp | head -10' Enter"),
            ('terminal_capture', 'tmux capture-pane -p -t ramp -S -2000'),
            ('file_check', "set -e; printf 'ramp-check\n' > /tmp/ramp-check; test $(wc -c < /tmp/ramp-check) -eq 11; sleep .2; head -c 2048 /dev/zero | tr '\\0' x"),
        ]
        async def worker(seat):
            await asyncio.sleep(random.random()*self.a.think_seconds)
            k = 0
            while time.monotonic()-start < seconds and not self.stop.is_set():
                kind, command = commands[k % len(commands)]
                rows.append(await self.call(seat, command, tag, kind, k))
                if len(rows) >= 20 and sum(not row['ok'] for row in rows) / len(rows) >= .05:
                    self.stop.set()
                k += 1
                if self.a.cold_every and k % self.a.cold_every == 0:
                    await asyncio.sleep(self.a.cold_gap)
                else:
                    await asyncio.sleep(self.a.think_seconds)
        self.emit('plateau_start', stage=tag, seats=n, duration_seconds=seconds)
        if not self.a.worker_ids:
            self.emit('org_census', stage=tag, census=await self.stress.census())
        if self.a.worker_ids:
            ready = self.path/'ready'
            ready.touch()
            barrier = self.path.parent/'start'
            while not barrier.exists():
                await asyncio.sleep(.1)
            start = time.monotonic()
            await asyncio.gather(*(worker(s) for s in self.seats[:n]))
        else:
            # Match production's 64 seats per coordinator, each with its own SDK
            # client and event loop, rather than benchmark a single 1,056-seat loop.
            workroot=self.path/('workers_'+tag)
            workroot.mkdir(exist_ok=True)
            barrier=workroot/'start'
            barrier.unlink(missing_ok=True)
            processes=[]
            files=[]
            for offset in range(0,n,64):
                path=workroot/str(offset//64)
                path.mkdir(exist_ok=True)
                ids=path/'ids.json'
                ids.write_text(json.dumps([dict(i=s['i'],id=s['sb'].id) for s in self.seats[offset:min(offset+64,n)]]))
                log=(path/'process.log').open('w')
                files.append(log)
                proc=await asyncio.create_subprocess_exec(sys.executable,__file__,
                    '--run',self.a.run,'--out',str(path),'--worker-ids',str(ids),
                    '--worker-stage',tag,'--plateau-seconds',str(seconds),
                    '--think-seconds',str(self.a.think_seconds),'--pool',str(self.a.pool),
                    '--cold-every',str(self.a.cold_every),'--cold-gap',str(self.a.cold_gap),
                    stdout=log,stderr=log)
                processes.append((proc,path))
            try:
                deadline=time.monotonic()+90
                while not all((path/'ready').exists() for _,path in processes):
                    if time.monotonic()>deadline or any(proc.returncode is not None for proc,_ in processes):
                        raise RuntimeError('command coordinator did not become ready')
                    await asyncio.sleep(.25)
                start=time.monotonic()
                barrier.touch()
                codes=await asyncio.gather(*(proc.wait() for proc,_ in processes))
                if any(codes):
                    raise RuntimeError('command coordinator failed; inspect process.log')
                worker_peaks=[]
                for _,path in processes:
                    for line in (path/'events.jsonl').read_text().splitlines():
                        row=json.loads(line)
                        if row['kind']=='command':
                            rows.append(row)
                            self.f.write(json.dumps(row)+'\n')
                        elif row['kind']=='plateau_summary':
                            worker_peaks.append(row['peak_inflight'])
                            self.lag.append(row['loop_lag_max'])
                # Exact overlap from command intervals, not sum of worker peaks.
                points=[]
                for r in rows:
                    points.extend([(r['start_epoch'],1),(r['start_epoch']+r['seconds'],-1)])
                live=0
                for _,delta in sorted(points):
                    live+=delta
                    self.peak=max(self.peak,live)
            finally:
                for proc,_ in processes:
                    if proc.returncode is None:
                        proc.terminate()
                await asyncio.gather(*(proc.wait() for proc,_ in processes))
                for log in files:
                    log.close()
        wall = time.monotonic()-start
        good = [r['seconds'] for r in rows if r['ok']]
        summary = dict(stage=tag, seats=n, attempts=len(rows), errors=sum(not r['ok'] for r in rows),
            error_types=dict(collections.Counter(r.get('error_type', 'nonzero_exit') for r in rows if not r['ok'])),
            p50=quantile(good,.5), p90=quantile(good,.9), p99=quantile(good,.99),
            completed_per_second=len(good)/wall, peak_inflight=self.peak,
            loop_lag_p99=(quantile(self.lag,.99) if self.a.worker_ids else None),
            loop_lag_max=max(self.lag, default=0), seconds=wall,
            coordinators=(1 if self.a.worker_ids else (n+63)//64),
            driver_cpu_seconds=cpu_seconds()-cpu_start,
            per_kind={kind:dict(attempts=sum(r['command_kind']==kind for r in rows),
                errors=sum(r['command_kind']==kind and not r['ok'] for r in rows),
                p50=quantile([r['seconds'] for r in rows if r['command_kind']==kind and r['ok']],.5),
                p99=quantile([r['seconds'] for r in rows if r['command_kind']==kind and r['ok']],.99))
                for kind,_ in commands})
        self.summaries.append(summary)
        self.emit('plateau_summary', **summary)
        print(json.dumps(summary), flush=True)
        (self.path/'summary.json').write_text(json.dumps(self.summaries, indent=2)+'\n')
        if (n <= 4 and summary['errors']) or (summary['attempts'] and summary['errors']/summary['attempts'] >= .05):
            raise RuntimeError('canary had command errors; refusing the full ramp')

    async def cleanup(self):
        # Discover by exact label, including any create accepted before a client timeout.
        for attempt in range(4):
            ids, cursor, states = [], None, collections.Counter()
            while True:
                page = await self.d._sandbox_api.list_sandboxes(cursor=cursor, limit=200,
                    labels=json.dumps({'harbor.instance':self.stress.label}), _request_timeout=30)
                ids.extend(s.id for s in page.items if (s.labels or {}).get('harbor.instance')==self.stress.label
                           and str(getattr(s.state,'value',s.state)) not in ('destroyed','destroying'))
                states.update(str(getattr(s.state,'value',s.state)) for s in page.items
                              if (s.labels or {}).get('harbor.instance')==self.stress.label)
                cursor = page.next_cursor
                if not cursor:
                    break
            self.emit('cleanup_census', attempt=attempt, remaining=len(ids), states=dict(states))
            if not ids and not states.get('destroying'):
                return
            sem = asyncio.Semaphore(32)
            async def delete(sid):
                async with sem:
                    try:
                        await self.d._sandbox_api.delete_sandbox(sid, _request_timeout=30)
                    except Exception as exc:
                        self.emit('cleanup_error', sandbox_id=sid, error_type=type(exc).__name__)
            await asyncio.gather(*(delete(sid) for sid in ids))
            await asyncio.sleep(5)
        raise RuntimeError('cleanup not verified')

    async def run(self):
        mon = asyncio.create_task(self.monitor())
        self.emit('meta', run=self.a.run, label=self.stress.label, args=vars(self.a),
                  pid=os.getpid(), host=os.uname().nodename,
                  sdk_version=importlib.metadata.version('daytona'),
                  driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  nofile=resource.getrlimit(resource.RLIMIT_NOFILE))
        print('ramp_start '+self.stress.label, flush=True)
        try:
            for n in self.a.levels:
                if self.stop.is_set():
                    break
                old = len(self.stress.rows)
                await asyncio.gather(*(self.add_one(i) for i in range(old,n)))
                self.seats.sort(key=lambda s:s['i'])
                if self.stop.is_set():
                    break
                if len(self.seats) != n:
                    raise RuntimeError(f'only {len(self.seats)} ready of {n}; refusing mislabeled plateau')
                await self.plateau(n, str(n), self.a.final_seconds if n == max(self.a.levels) and self.a.final_seconds else self.a.plateau_seconds)
            if self.a.cold_every and max(self.a.levels) == 1056 and not self.stop.is_set():
                saved = self.a.cold_every
                self.a.cold_every = 0
                try:
                    await self.plateau(len(self.seats), 'warm1056', 120)
                finally:
                    self.a.cold_every = saved
            if self.a.replace_seats and not self.stop.is_set():
                victims = self.seats[:self.a.replace_seats]
                for seat in victims:
                    await self.d._sandbox_api.delete_sandbox(seat['sb'].id, _request_timeout=30)
                self.seats = self.seats[self.a.replace_seats:]
                first = len(self.stress.rows)
                await asyncio.gather(*(self.add_one(i) for i in range(first, first+len(victims))))
                if len(self.seats) != max(self.a.levels):
                    raise RuntimeError('replacement seat count mismatch')
                self.emit('replacement_complete', seats=len(self.seats), replaced=len(victims))
                await self.plateau(len(self.seats), 'replacement', self.a.plateau_seconds)
            if not self.stop.is_set() and self.seats:
                await self.plateau(min(128,len(self.seats)), 'recovery128', 45)
        finally:
            await self.cleanup()
            mon.cancel()
            self.f.close()


async def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--levels', type=int, nargs='+', default=[128,256,512,1056])
    p.add_argument('--snapshot', default='harbor__8c0fc945338b__snapshot')
    p.add_argument('--plateau-seconds', type=float, default=120)
    p.add_argument('--think-seconds', type=float, default=1)
    p.add_argument('--command-timeout', type=float, default=30)
    p.add_argument('--pool', type=int, default=2048)
    p.add_argument('--cold-every', type=int, default=0)
    p.add_argument('--cold-gap', type=float, default=35)
    p.add_argument('--final-seconds', type=float, default=0)
    p.add_argument('--replace-seats', type=int, default=0)
    p.add_argument('--worker-ids')
    p.add_argument('--worker-stage')
    a=p.parse_args()
    assert a.levels == sorted(set(a.levels)) and 0 < max(a.levels) <= 1056
    from daytona import AsyncDaytona, DaytonaConfig
    async with AsyncDaytona(DaytonaConfig(api_key=load_key(KEY_FILE),connection_pool_maxsize=a.pool)) as d:
        ramp=Ramp(a,d)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
            asyncio.get_running_loop().add_signal_handler(sig,ramp.stop.set)
        if a.worker_ids:
            for item in json.loads(Path(a.worker_ids).read_text()):
                ramp.seats.append(dict(i=item['i'],sb=await d.get(item['id'])))
            mon=asyncio.create_task(ramp.monitor())
            try:
                await ramp.plateau(len(ramp.seats),a.worker_stage,a.plateau_seconds)
            finally:
                mon.cancel()
                ramp.f.close()
        else:
            await ramp.run()


if __name__=='__main__':
    asyncio.run(main())
