#!/usr/bin/env python3
"""Minimal HTTP forward proxy (CONNECT + plain GET/POST) for compute nodes that have no internet.

Runs on a login node (which has egress) bound to the cluster-internal address, like the microsocks SOCKS gateway
on 10.128.1.2:7011, but speaks HTTP so apt, pip, npm, curl, git and apptainer's OCI pull all use it without
SOCKS support:

    python http_proxy.py --bind 10.128.1.2 --port 7012

Client side: http_proxy=https_proxy=http://10.128.1.2:7012. Stdlib only, one thread per connection.
"""
import argparse, select, socket, socketserver, urllib.parse


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        c = self.request; c.settimeout(600)
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = c.recv(65536)
            if not chunk: return
            head += chunk
            if len(head) > 1 << 20: return
        head, rest = head.split(b"\r\n\r\n", 1)
        lines = head.decode("latin1").split("\r\n"); req = lines[0].split()
        if len(req) < 3: return
        method, target = req[0], req[1]
        try:
            if method == "CONNECT":
                host, _, port = target.partition(":")
                up = socket.create_connection((host, int(port or 443)), timeout=30)
                c.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
                if rest: up.sendall(rest)
            else:
                u = urllib.parse.urlsplit(target)
                if not u.netloc:  # transparent-style request without an absolute URI
                    return
                host, port = u.hostname, u.port or 80
                path = urllib.parse.urlunsplit(("", "", u.path or "/", u.query, ""))
                out = [f"{method} {path} {req[2]}"] + [l for l in lines[1:] if not l.lower().startswith(("proxy-connection:", "connection:"))] + ["Connection: close"]
                up = socket.create_connection((host, port), timeout=30)
                up.sendall(("\r\n".join(out) + "\r\n\r\n").encode("latin1") + rest)
        except Exception as e:
            try: c.sendall(f"HTTP/1.1 502 Bad Gateway\r\n\r\n{e}\r\n".encode())
            except Exception: pass
            return
        up.settimeout(600)
        socks = [c, up]
        try:
            while True:
                r, _, x = select.select(socks, [], socks, 600)
                if x or not r: break
                for s in r:
                    data = s.recv(65536)
                    if not data: return
                    (up if s is c else c).sendall(data)
        except Exception:
            pass
        finally:
            up.close()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True; daemon_threads = True; request_queue_size = 512


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--bind", default="0.0.0.0"); ap.add_argument("--port", type=int, default=7012)
    a = ap.parse_args()
    with Server((a.bind, a.port), Handler) as s:
        print(f"http proxy on {a.bind}:{a.port}", flush=True); s.serve_forever()
