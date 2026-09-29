#!/usr/bin/env python3
"""Daytona from a Horizon compute node, through the login-side `ssh -R` SOCKS tunnel(s) and socks_connect_bridge.py.

    python daytona_probe.py check --key-file K [--snapshot S ...]
    python daytona_probe.py load  --key-file K --snapshot S --levels 64 128 192 --seconds 180 --out DIR

check  the CalibForge snapshots ACTIVE (default: the three the relay tree uses), 2 sandboxes created from the first,
       one command run in each, both deleted and gone from the org listing.
load   the tunnel under relay-like Daytona traffic without relay-like sandbox counts: SEATS concurrent "trials" share at
       most --sandboxes (8) sandboxes, each seat with its own session and tmux session, driven with harbor's
       execute / poll / logs protocol over a rotation of tmux send-keys, capture-pane, a small file check and, every
       8th call, a 64 KB output. 1 s think time; a 35 s gap after every 4th call, so the SDK's 30 s keep-alive lapses
       and the next call opens a new connection through the tunnel ("cold"). 96 seats per worker process (at 192 the
       relay runs two staggered harbor jobs of 96). One JSON line per level: calls, errors, calls/s, p50/p90/p99 by
       kind and cold vs warm; rows in DIR/level_<n>.jsonl.
Every sandbox is labelled harbor.instance=relay-drv-<run>, ephemeral with a 15 min auto-stop (leak guard), and deleted
by label at the end, on error and on SIGTERM. Snapshots are only read. The key is never printed.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import random
import re
import shlex
import signal
import sys
import time

CALIBFORGE = ['harbor__313e69c036ad__snapshot', 'harbor__13bf95818376__snapshot', 'harbor__239bee0dfdf6__snapshot']


def load_key(path):
    text = Path(path).read_text().strip()
    m = re.search(r"DAYTONA_API_KEY\s*=\s*['\"]?([^'\"\s]+)", text)
    return m.group(1) if m else text.splitlines()[0].strip()


def q(xs, f):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int((len(xs) - 1) * f))], 3) if xs else None


def state(x):
    s = getattr(x, 'state', '')
    return str(getattr(s, 'value', s))


async def create(d, snapshot, label, n):
    from daytona import CreateSandboxFromSnapshotParams
    out = []
    for i in range(n):   # paced, well under the org's ~5 creates/s
        t = time.monotonic()
        sb = await d.create(CreateSandboxFromSnapshotParams(snapshot=snapshot, labels={'harbor.instance': label},
                                                            auto_stop_interval=15, ephemeral=True), timeout=300)
        out.append((sb, round(time.monotonic() - t, 2)))
        await asyncio.sleep(.5)
    return out


async def listed(d, label):
    from daytona import ListSandboxesQuery
    return [sb async for sb in d.list(ListSandboxesQuery(labels={'harbor.instance': label}))]


async def delete_label(d, label):
    gone = 0
    for _ in range(3):
        items = await listed(d, label)
        if not items:
            break
        for sb in items:
            try:
                await d.delete(sb, timeout=120)
                gone += 1
            except Exception as e:  # noqa: BLE001
                print(json.dumps(dict(event='delete_error', id=sb.id, type=type(e).__name__)), flush=True)
        await asyncio.sleep(5)
    left = len([sb for sb in await listed(d, label) if state(sb).lower() not in ('destroyed', 'destroying')])
    return gone, left


async def check(a, d):
    page = await d.snapshot.list()
    snaps = {s.name: state(s) for s in getattr(page, 'items', page)}
    want = a.snapshot or CALIBFORGE
    res = dict(snapshots={s: snaps.get(s, 'MISSING') for s in want}, custom_snapshots=len(snaps))
    print(json.dumps(res), flush=True)
    if any(v.lower() != 'active' for v in res['snapshots'].values()):
        sys.exit('snapshot not ACTIVE')
    label = f'relay-drv-check-{int(time.time())}'
    try:
        made = await create(d, want[0], label, 2)
        for sb, secs in made:
            t = time.monotonic()
            r = await sb.process.exec('echo relay-drv-ok; uname -m; which tmux python3; df -h / | tail -1', timeout=60)
            print(json.dumps(dict(event='exec', id=sb.id, create_s=secs, exec_s=round(time.monotonic() - t, 2),
                                  exit_code=r.exit_code, out=(r.result or '')[:300])), flush=True)
    finally:
        gone, left = await delete_label(d, label)
        print(json.dumps(dict(event='cleanup', label=label, deleted=gone, left=left)), flush=True)


COMMANDS = [
    ('send_keys', "tmux send-keys -t {s} 'printf command-ok; pwd; ls /tmp | head -10' Enter"),
    ('capture', 'tmux capture-pane -p -t {s} -S -2000'),
    ('file_check', "set -e; printf 'ramp-check\\n' > /tmp/{s}.chk; test $(wc -c < /tmp/{s}.chk) -eq 11; head -c 2048 /dev/zero | tr '\\0' x"),
    ('capture', 'tmux capture-pane -p -t {s} -S -2000'),
    ('send_keys', "tmux send-keys -t {s} 'printf command-ok; pwd; ls /tmp | head -10' Enter"),
    ('capture', 'tmux capture-pane -p -t {s} -S -2000'),
    ('file_check', "set -e; printf 'ramp-check\\n' > /tmp/{s}.chk; test $(wc -c < /tmp/{s}.chk) -eq 11; head -c 2048 /dev/zero | tr '\\0' x"),
    ('big_output', 'head -c 49152 /dev/urandom | base64 -w 120'),
]


async def worker(a):
    """One process = one SDK client and event loop, like one harbor job; seats a.seat0 .. a.seat0 + a.nseats - 1."""
    from daytona import AsyncDaytona, DaytonaConfig, SessionExecuteRequest
    ids = json.loads(a.ids)
    rows = open(a.out, 'a', buffering=1)
    async with AsyncDaytona(DaytonaConfig(api_key=load_key(a.key_file))) as d:
        sbs = {i: await d.get(i) for i in ids}

        async def call(seat, sess, kind, cmd, ordinal, cold):
            p = seat['sb'].process
            t = time.monotonic()
            row = dict(seat=seat['i'], kind=kind, ordinal=ordinal, cold=cold, ok=False, start=time.time())
            try:
                async def op():
                    r = await p.execute_session_command(sess, SessionExecuteRequest(
                        command='timeout 20 bash -c ' + shlex.quote(cmd), run_async=True), timeout=25)
                    while True:
                        st = await p.get_session_command(sess, r.cmd_id)
                        if st.exit_code is not None:
                            break
                        await asyncio.sleep(.05)
                    logs = await p.get_session_command_logs(sess, r.cmd_id)
                    row['bytes'] = len((logs.stdout or '').encode()) + len((logs.stderr or '').encode())
                    row['ok'] = st.exit_code == 0
                await asyncio.wait_for(op(), 60)
            except Exception as e:  # noqa: BLE001  (never serialize SDK bodies: they can carry URLs / keys)
                row['error'] = type(e).__name__
            row['seconds'] = round(time.monotonic() - t, 4)
            rows.write(json.dumps(row) + '\n')

        async def run_seat(i):
            seat = dict(i=i, sb=sbs[ids[i % len(ids)]])
            sess = f'l{a.level}s{i}'
            try:
                await seat['sb'].process.create_session(sess)
                await call(seat, sess, 'setup', f'tmux new-session -d -s {sess} -x 160 -y 40', -1, True)
            except Exception as e:  # noqa: BLE001
                rows.write(json.dumps(dict(seat=i, kind='setup', ok=False, error=type(e).__name__, seconds=0)) + '\n')
                return
            while not Path(a.barrier).exists():
                await asyncio.sleep(.2)
            t0 = time.monotonic()
            await asyncio.sleep(random.random() * a.think)
            k = 0
            cold = True
            while time.monotonic() - t0 < a.seconds:
                kind, cmd = COMMANDS[k % len(COMMANDS)]
                await call(seat, sess, kind, cmd.format(s=sess), k, cold)
                k += 1
                cold = k % a.cold_every == 0
                await asyncio.sleep(a.cold_gap if cold else a.think)
            try:
                await seat['sb'].process.exec(f'tmux kill-session -t {sess}; rm -f /tmp/{sess}.chk', timeout=30)
                await seat['sb'].process.delete_session(sess)
            except Exception:  # noqa: BLE001
                pass

        await asyncio.gather(*(run_seat(i) for i in range(a.seat0, a.seat0 + a.nseats)))


def summarize(level, path, wall):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    main = [r for r in rows if r['kind'] != 'setup']
    ok = [r['seconds'] for r in main if r['ok']]
    out = dict(level=level, calls=len(main), errors=sum(1 for r in main if not r['ok']),
               setup_errors=sum(1 for r in rows if r['kind'] == 'setup' and not r['ok']),
               error_types=sorted({r.get('error', 'exit') for r in main if not r['ok']}),
               calls_per_s=round(len(main) / wall, 1), p50=q(ok, .5), p90=q(ok, .9), p99=q(ok, .99), max=q(ok, 1),
               cold=[q([r['seconds'] for r in main if r['ok'] and r['cold']], f) for f in (.5, .99)],
               warm=[q([r['seconds'] for r in main if r['ok'] and not r['cold']], f) for f in (.5, .99)],
               by_kind={k: [len([r for r in main if r['kind'] == k]), q([r['seconds'] for r in main if r['ok'] and r['kind'] == k], .5),
                            q([r['seconds'] for r in main if r['ok'] and r['kind'] == k], .99)]
                        for k in sorted({r['kind'] for r in main})},
               mb_moved=round(sum(r.get('bytes', 0) for r in main) / 1e6, 1))
    print(json.dumps(out), flush=True)
    return out


async def load(a, d):
    label = f'relay-drv-load-{int(time.time())}'
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    summ = []
    try:
        made = await create(d, a.snapshot[0], label, a.sandboxes)
        ids = [sb.id for sb, _ in made]
        print(json.dumps(dict(event='sandboxes', n=len(ids), create_s=[s for _, s in made])), flush=True)
        for level in a.levels:
            if stop.is_set():
                break
            path = out / f'level_{level}.jsonl'
            barrier = out / f'start_{level}'
            barrier.unlink(missing_ok=True)
            procs = []
            for s0 in range(0, level, a.per_worker):
                procs.append(await asyncio.create_subprocess_exec(
                    sys.executable, __file__, 'worker', '--key-file', a.key_file, '--ids', json.dumps(ids),
                    '--seat0', str(s0), '--nseats', str(min(a.per_worker, level - s0)), '--level', str(level),
                    '--seconds', str(a.seconds), '--think', str(a.think), '--cold-every', str(a.cold_every),
                    '--cold-gap', str(a.cold_gap), '--barrier', str(barrier), '--out', str(path)))
            await asyncio.sleep(min(90, 10 + level / 4))   # sessions + tmux set up in every worker
            barrier.touch()
            t0 = time.monotonic()
            waiter = asyncio.gather(*(p.wait() for p in procs))
            done, _ = await asyncio.wait([asyncio.ensure_future(waiter), asyncio.ensure_future(stop.wait())],
                                         return_when=asyncio.FIRST_COMPLETED, timeout=a.seconds + 300)
            if not waiter.done():
                for p in procs:
                    p.kill()
            summ.append(summarize(level, path, time.monotonic() - t0))
    finally:
        gone, left = await delete_label(d, label)
        print(json.dumps(dict(event='cleanup', label=label, deleted=gone, left=left)), flush=True)
        (out / 'summary.json').write_text(json.dumps(dict(label=label, levels=summ, cleanup=dict(deleted=gone, left=left)), indent=1))


async def amain(a):
    from daytona import AsyncDaytona, DaytonaConfig
    async with AsyncDaytona(DaytonaConfig(api_key=load_key(a.key_file))) as d:
        await (check(a, d) if a.cmd == 'check' else load(a, d))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['check', 'load', 'worker'])
    ap.add_argument('--key-file', default=os.path.expanduser('~/.config/otagent/daytona_eval.env'))
    ap.add_argument('--snapshot', action='append')
    ap.add_argument('--levels', type=int, nargs='+', default=[64, 128, 192])
    ap.add_argument('--seconds', type=float, default=180)
    ap.add_argument('--sandboxes', type=int, default=8)
    ap.add_argument('--per-worker', type=int, default=96)
    ap.add_argument('--think', type=float, default=1.0)
    ap.add_argument('--cold-every', type=int, default=4)
    ap.add_argument('--cold-gap', type=float, default=35.0)
    ap.add_argument('--out', default='daytona_probe_out')
    # worker-only
    ap.add_argument('--ids'); ap.add_argument('--seat0', type=int); ap.add_argument('--nseats', type=int)
    ap.add_argument('--level', type=int); ap.add_argument('--barrier')
    a = ap.parse_args()
    if a.sandboxes > 8:
        sys.exit('at most 8 sandboxes')
    if a.cmd == 'worker':
        asyncio.run(worker(a))
    else:
        if a.cmd == 'load' and not a.snapshot:
            a.snapshot = CALIBFORGE[:1]
        asyncio.run(amain(a))
