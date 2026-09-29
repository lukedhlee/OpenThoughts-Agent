#!/usr/bin/env python3
"""Local HTTP CONNECT gateway to one or more upstream SOCKS5 proxies, without LD_PRELOAD (Horizon relay driver).

By diff from data/r2egym/daytona_artifacts/async_socks_connect_proxy.py (the Jupiter gateway, 264629ba). Deltas:
  - several upstream ports (--socks-port 18080,18081,...): each CONNECT goes to the upstream with the fewest open
    tunnels, and a failed SOCKS handshake is retried once on another upstream. On Horizon each port is one login-side
    `ssh -R` (data/r2egym/horizon/tunnel.sh), so this spreads the driver's Daytona traffic over several ssh processes
    and rides over one of them dropping (tunnel.sh reopens it within 20 s).
  - SOCKS credentials are optional (an `ssh -R` SOCKS5 has none); --rdns lets the upstream resolve names (the compute
    node then needs no DNS for the outside world).
  - --stats-every N prints one JSON line every N s: open tunnels per upstream, connects, errors, handshake p50/p99.
Bind to loopback. Set HTTPS_PROXY to this listener and NO_PROXY for internal services. Credentials are never logged.
"""
import argparse
import asyncio
import contextlib
import itertools
import json
import os
from pathlib import Path
import time

from python_socks import ProxyType
from python_socks.async_.asyncio import Proxy


async def relay(reader, writer):
    while data := await reader.read(65536):
        writer.write(data)
        await writer.drain()
    if writer.can_write_eof():
        writer.write_eof()


class Upstreams:
    def __init__(self, host, ports, rdns):
        user, pw = os.environ.get('SOCKS_USER') or None, os.environ.get('SOCKS_PASS') or None
        self.ports = ports
        self.proxies = {p: Proxy(ProxyType.SOCKS5, host, p, username=user, password=pw, rdns=rdns) for p in ports}
        self.open = {p: 0 for p in ports}
        self.rr = itertools.count()
        self.stats = dict(connects=0, errors=0, retried=0, handshake=[])

    def order(self):
        k = next(self.rr)
        return sorted(self.ports, key=lambda p: (self.open[p], (self.ports.index(p) - k) % len(self.ports)))

    async def connect(self, dest_host, dest_port, timeout):
        """(port, sock): the least-loaded upstream first, one retry on the next."""
        err = None
        t = time.monotonic()
        for i, p in enumerate(self.order()[:2]):
            try:
                sock = await self.proxies[p].connect(dest_host=dest_host, dest_port=dest_port, timeout=timeout)
                self.stats['handshake'].append(time.monotonic() - t)
                self.stats['retried'] += i > 0
                return p, sock
            except Exception as exc:  # noqa: BLE001
                err = exc
        raise err


async def handle(reader, writer, up):
    upstream = None
    established = False
    port = None
    counted = False
    start = time.monotonic()
    try:
        async with asyncio.timeout(30):
            header = await reader.readuntil(b'\r\n\r\n')
            method, authority, version = header.split(b'\r\n', 1)[0].decode('ascii').split()
            if method != 'CONNECT':
                writer.write(b'HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\n\r\n')
                await writer.drain()
                return
            host, dport = authority.rsplit(':', 1)
            port, sock = await up.connect(host.strip('[]'), int(dport), timeout=20)
            up.open[port] += 1
            counted = True
            try:
                upstream_reader, upstream = await asyncio.open_connection(sock=sock)
            except BaseException:
                sock.close()
                raise
            writer.write(b'HTTP/1.1 200 Connection Established\r\n\r\n')
            await writer.drain()
            established = True
            up.stats['connects'] += 1
        async with asyncio.timeout(1800):
            await asyncio.gather(relay(reader, upstream), relay(upstream_reader, writer))
    except (Exception,) as exc:
        if not established:
            up.stats['errors'] += 1
        print(json.dumps({'event': 'connection_error', 'type': type(exc).__name__, 'upstream': port,
                          'established': established, 'seconds': round(time.monotonic() - start, 4)}), flush=True)
        if not established:
            with contextlib.suppress(ConnectionError):
                writer.write(b'HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n')
                await writer.drain()
    finally:
        if counted:
            up.open[port] -= 1
        for stream in (writer, upstream):
            if stream is not None:
                stream.close()
                with contextlib.suppress(ConnectionError):
                    await stream.wait_closed()


async def report(up, every):
    while True:
        await asyncio.sleep(every)
        h = sorted(up.stats['handshake'])
        up.stats['handshake'] = []
        q = (lambda f: round(h[int(f * (len(h) - 1))], 4) if h else None)
        print(json.dumps(dict(event='stats', ts=round(time.time(), 1), open=dict(up.open), connects=up.stats['connects'],
                              errors=up.stats['errors'], retried=up.stats['retried'], handshakes=len(h),
                              handshake_p50=q(.5), handshake_p99=q(.99))), flush=True)


async def main(args):
    ports = [int(p) for p in str(args.socks_port).split(',') if p]
    up = Upstreams(args.socks_host, ports, args.rdns)
    server = await asyncio.start_server(lambda r, w: handle(r, w, up), '127.0.0.1', args.port, limit=16384)
    print(json.dumps(dict(event='listening', port=args.port, upstreams=ports, rdns=args.rdns)), flush=True)
    if args.stats_every:
        asyncio.get_running_loop().create_task(report(up, args.stats_every))
    if args.ready:
        Path(args.ready).touch()
    async with server:
        await server.serve_forever()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=18946)
    p.add_argument('--socks-host', default='127.0.0.1')
    p.add_argument('--socks-port', default='18080', help='one port or a comma-separated list')
    p.add_argument('--rdns', action='store_true', help='let the SOCKS upstream resolve host names')
    p.add_argument('--stats-every', type=float, default=0)
    p.add_argument('--ready')
    asyncio.run(main(p.parse_args()))
