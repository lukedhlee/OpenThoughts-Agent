#!/usr/bin/env python3
"""Local HTTP CONNECT gateway to an upstream SOCKS5 proxy, without LD_PRELOAD.

Bind to loopback. Set HTTPS_PROXY to this listener and NO_PROXY for internal
services. SOCKS credentials are read from the environment and never logged.
"""
import argparse
import asyncio
import contextlib
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


async def handle(reader, writer, proxy):
    upstream = None
    established = False
    start = time.monotonic()
    try:
        async with asyncio.timeout(30):
            header = await reader.readuntil(b'\r\n\r\n')
            method, authority, version = header.split(b'\r\n', 1)[0].decode('ascii').split()
            if method != 'CONNECT':
                writer.write(b'HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\n\r\n')
                await writer.drain()
                return
            host, port = authority.rsplit(':', 1)
            sock = await proxy.connect(dest_host=host.strip('[]'), dest_port=int(port), timeout=20)
            try:
                upstream_reader, upstream = await asyncio.open_connection(sock=sock)
            except BaseException:
                sock.close()
                raise
            writer.write(b'HTTP/1.1 200 Connection Established\r\n\r\n')
            await writer.drain()
            established = True
        async with asyncio.timeout(1800):
            await asyncio.gather(relay(reader, upstream), relay(upstream_reader, writer))
    except (Exception,) as exc:
        print(json.dumps({'event':'connection_error','type':type(exc).__name__,
                          'established':established,'seconds':round(time.monotonic()-start,4)}), flush=True)
        if not established:
            with contextlib.suppress(ConnectionError):
                writer.write(b'HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n')
                await writer.drain()
    finally:
        for stream in (writer, upstream):
            if stream is not None:
                stream.close()
                with contextlib.suppress(ConnectionError):
                    await stream.wait_closed()


async def main(args):
    proxy = Proxy(ProxyType.SOCKS5, args.socks_host, args.socks_port,
                  username=os.environ['SOCKS_USER'], password=os.environ['SOCKS_PASS'], rdns=False)
    server = await asyncio.start_server(lambda r,w: handle(r,w,proxy),'127.0.0.1',args.port,limit=16384)
    if args.ready:
        Path(args.ready).touch()
    async with server:
        await server.serve_forever()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=18946)
    p.add_argument('--socks-host',default='10.128.1.2');p.add_argument('--socks-port',type=int,default=7011)
    p.add_argument('--ready');asyncio.run(main(p.parse_args()))
