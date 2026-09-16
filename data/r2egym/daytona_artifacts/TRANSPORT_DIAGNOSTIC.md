# Daytona transport diagnosis

On Jupiter, proxychains makes external connection setup block the Python event loop. A local model-tokenization request can therefore wait behind an unrelated Daytona connection. Warm command loops conceal much of this cost.

The measured comparison and its limits live in `ai_memory/active/snowball-r2egym/research/2026-09-15_daytona_proxy_tokenization.md`.

## Candidate deployment change

Run `async_socks_connect_proxy.py` as a separate process on the same node as the Harbor coordinator. It binds only to `127.0.0.1`, accepts HTTP CONNECT, and negotiates the existing SOCKS connection asynchronously. TLS stays between the client and Daytona.

- Supply `SOCKS_USER` and `SOCKS_PASS` through the existing protected environment, never command-line arguments.
- Install `python-socks==3.1.1` in an isolated dependency directory. The diagnostic used pure-Python wheels; the shared environment was unchanged.
- Set `HTTPS_PROXY=http://127.0.0.1:18946` and `HTTP_PROXY` to the same address for the child workload.
- Set `NO_PROXY` to `localhost`, `127.0.0.1`, and every exact internal HTTP service address used by that workload. Do not assume CIDR exclusions work identically in httpx, aiohttp, and urllib.
- Do not wrap that workload in proxychains or inherit its `LD_PRELOAD` library.
- Start one gateway on **every coordinator node** for a multi-node job, using the same local port. A gateway running only on the head node does not provide a service on other nodes' loopback interfaces.
- Stop the gateway with its workload. Do not run it as an unowned login-node service.

The installed Daytona SDK uses a shared aiohttp session with `trust_env=True`, so this uses its existing environment-proxy support. The single-node SDK comparison passed. Multi-node Ray deployment and full rollout capacity are separate validation gates.

## Tests

`tokenization_transport_probe.py` separates tokenization-server work from transport-induced client delay. `tokenization_daytona_probe.py` uses real SDK session commands with 35-second gaps, longer than the SDK's 30-second keepalive. It compares gateway, original proxychains, and a gateway repeat on the same sandboxes, then deletes only its exact test label.

`test_async_socks_connect_proxy.py` checks 64 simultaneous TCP tunnels, binary payload integrity, half-close response delivery, and rejection of unsupported HTTP methods.

`paired_rollout_probe.py` runs a small real-agent cohort on frozen, matched task instructions and separately records generation, tokenization, and terminal-call timing. Its source does not modify the shared Harbor checkout. The sbatch file contains deployment-specific paths and is an experiment, not a portable launcher.
