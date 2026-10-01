#!/usr/bin/env python3
"""bench_client.py — closed-loop agent-turn load against one vLLM chat server, for the EAGLE-3 draft on/off bench.

CONC workers each replay real terminus-2 sessions (build_sessions.py output) turn by turn: turn k sends the recorded
history messages[:2k+1] as a non-streaming chat completion (as harbor does: skip_special_tokens=false, sampler from
the server's --override-generation-config), waits for the answer, then sends turn k+1. Turn k+1's prompt extends turn
k's, so prefix caching sees what it sees in a rollout. Worker i owns the fixed trajectory list perm[i::SLOTS] and starts
its first one at a seeded random turn, so the first 32 workers of a 128 run are exactly a 32 run, and a draft-on and a
draft-off server get the same request plans.

Numbers (window = after WARMUP seconds, for WINDOW seconds; deltas of the server's own Prometheus counters):
  node_out_tok_s       generation tokens / s over the whole node (all 4 DP engines)
  req_decode_tok_s     per-request decode speed, token-weighted: sum(gen tokens - 1) / sum(decode time) of the requests
                       that finished in the window (decode time = first to last token, so prefill and queueing excluded)
  req_tpot_inv         1 / mean per-request time-per-output-token (unweighted mean over requests)
  accept_len           1 + accepted / drafts (spec decode only)
  client_req_tok_s     per request completion_tokens / end-to-end latency (includes queue + prefill), mean over requests
At the end of the window the workers are cancelled (the server aborts their requests) and the client waits for the
server to drain. Writes one JSON summary line to --out and the per-request records to --out.reqs.jsonl.
"""
import argparse, asyncio, json, random, re, time
import aiohttp

COUNTERS = [
    "vllm:generation_tokens_total", "vllm:prompt_tokens_total", "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total", "vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total", "vllm:num_preemptions_total", "vllm:request_success_total",
    "vllm:request_decode_time_seconds_sum", "vllm:request_decode_time_seconds_count",
    "vllm:request_generation_tokens_sum", "vllm:request_generation_tokens_count",
    "vllm:request_time_per_output_token_seconds_sum", "vllm:request_time_per_output_token_seconds_count",
    "vllm:inter_token_latency_seconds_sum", "vllm:inter_token_latency_seconds_count",
    "vllm:time_to_first_token_seconds_sum", "vllm:time_to_first_token_seconds_count",
    "vllm:request_prefill_time_seconds_sum", "vllm:request_queue_time_seconds_sum",
]
GAUGES = ["vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc"]
LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)")


async def scrape(sess, url):
    async with sess.get(url + "/metrics") as r:
        text = await r.text()
    out = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = LINE.match(line)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2) or "", m.group(3)
        try:
            v = float(val)
        except ValueError:
            continue
        out[name] = out.get(name, 0.0) + v
        out["n:" + name] = out.get("n:" + name, 0) + 1
        if name == "vllm:spec_decode_num_accepted_tokens_per_pos_total":
            pos = re.search(r'position="(\d+)"', labels)
            if pos:
                k = f"pos{pos.group(1)}"
                out[k] = out.get(k, 0.0) + v
    return out


def plan(sessions, slots, seed):
    rng = random.Random(seed)
    perm = list(range(len(sessions)))
    rng.shuffle(perm)
    plans = []
    for i in range(slots):
        trajs = perm[i::slots] or perm
        start = rng.randrange(len(sessions[trajs[0]]["prompt_tokens"]))
        plans.append((trajs, start))
    return plans


async def worker(i, sess, a, sessions, trajs, start, recs, stop_at):
    ti, k = 0, start
    while time.time() < stop_at:
        s = sessions[trajs[ti % len(trajs)]]
        body = {"model": a.model, "messages": s["messages"][: 2 * k + 1], "max_tokens": a.max_tokens,
                "skip_special_tokens": False}
        t0 = time.time()
        rec = {"w": i, "traj": s["id"], "turn": k, "t0": t0}
        try:
            async with sess.post(a.url + "/v1/chat/completions", json=body) as r:
                j = await r.json(content_type=None)
            rec["t1"] = time.time()
            if "usage" in j:
                rec.update(p=j["usage"]["prompt_tokens"], c=j["usage"]["completion_tokens"],
                           fin=j["choices"][0]["finish_reason"])
            else:
                rec["err"] = str(j)[:300]
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            rec["t1"] = time.time()
            rec["err"] = repr(e)[:300]
        recs.append(rec)
        k += 1
        if k >= len(s["prompt_tokens"]):
            ti, k = ti + 1, 0


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="snowball")
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--conc", type=int, required=True)
    ap.add_argument("--slots", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=float, default=120)
    ap.add_argument("--window", type=float, default=240)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sessions = [json.loads(l) for l in open(a.sessions)]
    plans = plan(sessions, a.slots, a.seed)[: a.conc]
    recs = []
    conn = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=None)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as sess:
        t_begin = time.time()
        far = t_begin + 10 * (a.warmup + a.window)
        tasks = [asyncio.create_task(worker(i, sess, a, sessions, tr, st, recs, far)) for i, (tr, st) in enumerate(plans)]
        await asyncio.sleep(a.warmup)
        m0, w0 = await scrape(sess, a.url), time.time()
        gauges = []
        while time.time() < w0 + a.window:
            await asyncio.sleep(min(10, w0 + a.window - time.time()))
            g = await scrape(sess, a.url)
            gauges.append({k: g.get(k, 0.0) / (max(1, g.get("n:" + k, 1)) if k.endswith("_perc") else 1) for k in GAUGES})
        m1, w1 = await scrape(sess, a.url), time.time()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        drained = None
        for _ in range(90):
            await asyncio.sleep(2)
            g = await scrape(sess, a.url)
            if g.get("vllm:num_requests_running", 1) == 0 and g.get("vllm:num_requests_waiting", 1) == 0:
                drained = time.time() - w1
                break
    d = {k: m1.get(k, 0.0) - m0.get(k, 0.0) for k in set(m0) | set(m1) if k in COUNTERS or k.startswith("pos")}
    dt = w1 - w0
    win = [r for r in recs if "c" in r and w0 <= r["t1"] <= w1]
    errs = [r for r in recs if "err" in r and w0 <= r["t1"] <= w1]
    per_req = [r["c"] / (r["t1"] - r["t0"]) for r in win if r["c"] >= 16]
    drafts = d.get("vllm:spec_decode_num_drafts_total", 0.0)
    q = lambda x, p: sorted(x)[int(p * (len(x) - 1))] if x else None
    out = {
        "label": a.label, "conc": a.conc, "window_s": round(dt, 1), "warmup_s": a.warmup,
        "node_out_tok_s": round(d.get("vllm:generation_tokens_total", 0) / dt, 1),
        "node_prompt_tok_s": round(d.get("vllm:prompt_tokens_total", 0) / dt, 1),
        "prefix_hit_rate": round(d.get("vllm:prefix_cache_hits_total", 0) / max(1, d.get("vllm:prefix_cache_queries_total", 0)), 3),
        "req_decode_tok_s": round((d.get("vllm:request_generation_tokens_sum", 0) - d.get("vllm:request_generation_tokens_count", 0))
                                  / max(1e-9, d.get("vllm:request_decode_time_seconds_sum", 0)), 2),
        "req_tpot_inv": round(d.get("vllm:request_time_per_output_token_seconds_count", 0)
                              / max(1e-9, d.get("vllm:request_time_per_output_token_seconds_sum", 0)), 2),
        "itl_mean_ms": round(1000 * d.get("vllm:inter_token_latency_seconds_sum", 0) / max(1, d.get("vllm:inter_token_latency_seconds_count", 0)), 2),
        "ttft_mean_s": round(d.get("vllm:time_to_first_token_seconds_sum", 0) / max(1, d.get("vllm:time_to_first_token_seconds_count", 0)), 3),
        "prefill_mean_s": round(d.get("vllm:request_prefill_time_seconds_sum", 0) / max(1, d.get("vllm:request_decode_time_seconds_count", 0)), 3),
        "queue_mean_s": round(d.get("vllm:request_queue_time_seconds_sum", 0) / max(1, d.get("vllm:request_decode_time_seconds_count", 0)), 3),
        "server_reqs_done": int(d.get("vllm:request_success_total", 0)),
        "accept_len": round(1 + d.get("vllm:spec_decode_num_accepted_tokens_total", 0) / drafts, 3) if drafts else None,
        "accept_per_pos": [round(d[f"pos{i}"] / drafts, 3) for i in range(8) if f"pos{i}" in d] if drafts else None,
        "preemptions": int(d.get("vllm:num_preemptions_total", 0)),
        "client_reqs": len(win), "client_errs": len(errs), "err_sample": errs[0]["err"] if errs else None,
        "client_out_tok_s": round(sum(r["c"] for r in win) / dt, 1),
        "client_req_tok_s_mean": round(sum(per_req) / len(per_req), 2) if per_req else None,
        "client_req_tok_s_p50": round(q(per_req, .5), 2) if per_req else None,
        "client_prompt_p50": q([r["p"] for r in win], .5), "client_compl_p50": q([r["c"] for r in win], .5),
        "client_compl_mean": round(sum(r["c"] for r in win) / len(win), 1) if win else None,
        "finish_length": sum(r.get("fin") == "length" for r in win),
        "running_mean": round(sum(g["vllm:num_requests_running"] for g in gauges) / max(1, len(gauges)), 1),
        "waiting_mean": round(sum(g["vllm:num_requests_waiting"] for g in gauges) / max(1, len(gauges)), 1),
        "kv_usage_mean": round(sum(g["vllm:kv_cache_usage_perc"] for g in gauges) / max(1, len(gauges)), 3),
        "drain_s": round(drained, 1) if drained is not None else None,
        "w0": w0, "w1": w1,
    }
    print("RESULT " + json.dumps(out), flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(out) + "\n")
    with open(a.out + f".{a.label}.reqs.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
