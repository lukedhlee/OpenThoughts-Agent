"""Exercise concurrent bidirectional tunnels and half-close against real TCP sockets."""
import asyncio
import hashlib
import socket
import time

from async_socks_connect_proxy import handle


class DirectConnector:
    async def connect(self, dest_host, dest_port, timeout):
        await asyncio.sleep(.05)
        sock=socket.socket();sock.setblocking(False)
        await asyncio.get_running_loop().sock_connect(sock,(dest_host,dest_port))
        return sock


async def main():
    async def digest(reader,writer):
        data=await reader.read()
        writer.write(hashlib.sha256(data).digest());await writer.drain()
        writer.close();await writer.wait_closed()
    target=await asyncio.start_server(digest,'127.0.0.1',0)
    target_port=target.sockets[0].getsockname()[1]
    proxy=await asyncio.start_server(lambda r,w:handle(r,w,DirectConnector()),'127.0.0.1',0)
    port=proxy.sockets[0].getsockname()[1]
    async def client(i):
        reader,writer=await asyncio.open_connection('127.0.0.1',port)
        writer.write(f'CONNECT 127.0.0.1:{target_port} HTTP/1.1\r\n\r\n'.encode());await writer.drain()
        assert (await reader.readuntil(b'\r\n\r\n')).startswith(b'HTTP/1.1 200')
        data=bytes(range(256))*512+str(i).encode()
        writer.write(data);await writer.drain();writer.write_eof()
        assert await reader.read()==hashlib.sha256(data).digest()
        writer.close();await writer.wait_closed()
    started=time.monotonic()
    async with target,proxy:
        await asyncio.wait_for(asyncio.gather(*(client(i) for i in range(64))),10)
        reader,writer=await asyncio.open_connection('127.0.0.1',port)
        writer.write(b'GET / HTTP/1.1\r\n\r\n');await writer.drain()
        assert (await reader.read()).startswith(b'HTTP/1.1 405')
        writer.close();await writer.wait_closed()
    print(f'64 concurrent tunnels, binary payloads, half-close, and unsupported-method rejection passed ({time.monotonic()-started:.3f}s)')


if __name__=='__main__':asyncio.run(main())
