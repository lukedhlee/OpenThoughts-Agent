#!/usr/bin/env python3
"""Snowball serving benchmark client (asyncio + streaming, OpenAI chat API).

Modes
  sweep   decode curve: one shared (prefix-cached) context of --prefix-tokens, unique tails, forced
          --out-tokens completions, at each concurrency in --conc  -> per-stream decode tok/s vs batch
  prefill unique-prefix prefill throughput: --conc requests x --prefix-tokens unique tokens, 1 output token
  replay  multi-turn replay of real R2E-Gym trajectories (prep_replay.py JSONL): --conc trajectories in
          flight, each walking its turns with a --gap sleep (simulated sandbox time) between turns.
          --forced reproduces the production completion length of every turn exactly
          (min_tokens=max_tokens=L, ignore_eos); otherwise the model stops naturally (cap --max-tokens).
Sampling mirrors production: temperature 1.0, top_p 0.95, top_k 20, logprobs requested.
Writes a JSON summary to --out and prints RESULT lines.
"""
import argparse, asyncio, json, random, statistics, sys, time

import aiohttp

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["sweep", "prefill", "replay"])
ap.add_argument("--url", default="http://localhost:8000")
ap.add_argument("--model", default="snowball")
ap.add_argument("--label", default="run")
ap.add_argument("--out", default=None)
ap.add_argument("--conc", default="16")
ap.add_argument("--prefix-tokens", type=int, default=12000)
ap.add_argument("--out-tokens", type=int, default=256)
ap.add_argument("--replay", default=None)
ap.add_argument("--n-traj", type=int, default=192)
ap.add_argument("--gap", type=float, default=2.2)
ap.add_argument("--forced", action="store_true")
ap.add_argument("--max-tokens", type=int, default=4096)
ap.add_argument("--time-budget", type=float, default=480)
ap.add_argument("--steady-after", type=float, default=90)
ap.add_argument("--no-logprobs", action="store_true")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-model-len", type=int, default=32768)
a = ap.parse_args()
random.seed(a.seed)


def pctl(x, f):
    x = sorted(v for v in x if v is not None)
    return x[int(f * (len(x) - 1))] if x else None


def make_blob(n_tokens, nonce=""):
    # code-like filler, ~3 chars/token for this tokenizer (empirically ~2.9-3.2)
    lines = []
    i = 0
    while sum(len(s) for s in lines) < n_tokens * 3:
        lines.append(f"def fn_{nonce}{i}(x, y):\n    return (x * {i} + y) % 97\n\n")
        i += 1
    return "".join(lines)


async def chat(session, messages, max_tokens, forced=False):
    body = {"model": a.model, "messages": messages, "max_tokens": max_tokens,
            "temperature": 1.0, "top_p": 0.95, "top_k": 20,
            "stream": True, "stream_options": {"include_usage": True}}
    if forced:
        body["min_tokens"] = max_tokens
        body["ignore_eos"] = True
    if not a.no_logprobs:
        body["logprobs"] = True
    t0 = time.perf_counter(); ttft = None; usage = None; finish = None; nchunk = 0
    try:
        async with session.post(f"{a.url}/v1/chat/completions", json=body) as r:
            if r.status != 200:
                txt = await r.text()
                return {"error": f"{r.status} {txt[:200]}", "latency": time.perf_counter() - t0}
            async for raw in r.content:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                d = json.loads(data)
                if d.get("usage"):
                    usage = d["usage"]
                ch = d.get("choices") or []
                if ch:
                    delta = ch[0].get("delta") or {}
                    if delta.get("content"):
                        nchunk += 1
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                    if ch[0].get("finish_reason"):
                        finish = ch[0]["finish_reason"]
    except Exception as e:
        return {"error": f"EXC {type(e).__name__}: {str(e)[:160]}", "latency": time.perf_counter() - t0}
    lat = time.perf_counter() - t0
    ct = (usage or {}).get("completion_tokens"); pt = (usage or {}).get("prompt_tokens")
    dec = (ct - 1) / (lat - ttft) if (ct and ttft and lat > ttft and ct > 1) else None
    return {"latency": lat, "ttft": ttft, "completion_tokens": ct, "prompt_tokens": pt,
            "finish": finish, "decode_tps": dec, "chunks": nchunk}


async def metrics(session):
    try:
        async with session.get(f"{a.url}/metrics") as r:
            txt = await r.text()
    except Exception:
        return {}
    out = {}
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        for key in ("vllm:generation_tokens_total", "vllm:prompt_tokens_total", "vllm:request_success_total"):
            if line.startswith(key):
                try:
                    out[key] = out.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
                except Exception:
                    pass
    return out


def summarize(recs):
    ok = [r for r in recs if "error" not in r]
    return {"n": len(recs), "errors": len(recs) - len(ok),
            "latency_p50": pctl([r["latency"] for r in ok], .5), "latency_p90": pctl([r["latency"] for r in ok], .9),
            "ttft_p50": pctl([r["ttft"] for r in ok], .5), "ttft_p90": pctl([r["ttft"] for r in ok], .9),
            "decode_tps_p10": pctl([r["decode_tps"] for r in ok], .1),
            "decode_tps_p50": pctl([r["decode_tps"] for r in ok], .5),
            "decode_tps_p90": pctl([r["decode_tps"] for r in ok], .9),
            "completion_tokens_p50": pctl([r["completion_tokens"] for r in ok], .5),
            "completion_tokens_mean": statistics.mean([r["completion_tokens"] or 0 for r in ok]) if ok else None,
            "prompt_tokens_p50": pctl([r["prompt_tokens"] for r in ok], .5),
            "finish": {f: sum(1 for r in ok if r["finish"] == f) for f in set(r["finish"] for r in ok)}}


async def run_sweep(session):
    res = {"mode": "sweep", "prefix_tokens_req": a.prefix_tokens, "out_tokens": a.out_tokens, "points": {}}
    blob = make_blob(a.prefix_tokens)
    mk = lambda tail: [{"role": "user", "content": "Here is a module for context.\n\n" + blob + "\n\n" + tail}]
    # warm the prefix into every DP rank's cache
    warm = await asyncio.gather(*[chat(session, mk(f"warmup {i}. Reply OK."), 4) for i in range(8)])
    res["prefix_prompt_tokens"] = next((w.get("prompt_tokens") for w in warm if "error" not in w), None)
    print(f"[sweep] prefix prompt_tokens={res['prefix_prompt_tokens']}", flush=True)
    for conc in [int(c) for c in a.conc.split(",")]:
        tasks = [chat(session, mk(f"Request {i} nonce {random.random():.8f}. Write detailed documentation for fn_{i}."),
                      a.out_tokens, forced=True) for i in range(conc)]
        m0 = await metrics(session); t0 = time.perf_counter()
        rs = await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0; m1 = await metrics(session)
        s = summarize(rs); ok = [r for r in rs if "error" not in r]
        toks = sum(r["completion_tokens"] or 0 for r in ok)
        s.update({"wall": wall, "agg_tps": toks / wall, "per_stream_tps": toks / wall / conc,
                  "server_gen_tokens": (m1.get("vllm:generation_tokens_total", 0) - m0.get("vllm:generation_tokens_total", 0))})
        res["points"][conc] = s
        print(f"RESULT {a.label} sweep ctx={res['prefix_prompt_tokens']} conc={conc} agg_tps={s['agg_tps']:.1f} "
              f"per_stream_tps={s['per_stream_tps']:.2f} decode_tps_p50={s['decode_tps_p50']:.2f} ttft_p50={s['ttft_p50']:.2f} "
              f"ttft_p90={s['ttft_p90']:.2f} wall={wall:.1f} errors={s['errors']}", flush=True)
        if s["errors"]:
            print("  errors:", [r["error"] for r in rs if "error" in r][:3], flush=True)
    return res


async def run_prefill(session):
    res = {"mode": "prefill", "prefix_tokens_req": a.prefix_tokens, "points": {}}
    for conc in [int(c) for c in a.conc.split(",")]:
        msgs = [[{"role": "user", "content": f"request-id {random.random():.10f}\n" + make_blob(a.prefix_tokens, nonce=f"{i}_") + "\n\nSummarize in one word."}] for i in range(conc)]
        t0 = time.perf_counter()
        rs = await asyncio.gather(*[chat(session, m, 1) for m in msgs])
        wall = time.perf_counter() - t0
        ok = [r for r in rs if "error" not in r]
        ptoks = sum(r["prompt_tokens"] or 0 for r in ok)
        s = summarize(rs); s.update({"wall": wall, "prefill_tps": ptoks / wall, "prompt_tokens_total": ptoks})
        res["points"][conc] = s
        print(f"RESULT {a.label} prefill conc={conc} prompt_tokens={s['prompt_tokens_p50']} prefill_tps={s['prefill_tps']:.0f} "
              f"ttft_p50={s['ttft_p50']:.2f} ttft_p90={s['ttft_p90']:.2f} latency_p90={s['latency_p90']:.2f} wall={wall:.1f} errors={s['errors']}", flush=True)
    return res


async def run_replay(session):
    trajs = [json.loads(l) for l in open(a.replay)]
    random.shuffle(trajs); trajs = trajs[:a.n_traj]
    conc = int(a.conc.split(",")[0])
    recs = []; state = {"running": 0}
    t_start = time.perf_counter(); deadline = t_start + a.time_budget
    sem = asyncio.Semaphore(conc)

    async def run_traj(tr):
        async with sem:
            if time.perf_counter() > deadline:
                return
            state["running"] += 1
            try:
                for i in range(tr["n_turns"]):
                    if time.perf_counter() > deadline:
                        break
                    msgs = tr["messages"][:2 * i + 1]
                    room = a.max_model_len - 64 - tr["prompt_len"][i]
                    if room < 32:
                        break
                    L = min(tr["completion_len"][i], room)
                    r = await chat(session, msgs, L if a.forced else min(a.max_tokens, room), forced=a.forced)
                    r.update({"turn": i, "trial": tr["trial"], "prod_api_s": tr["api_ms"][i] / 1000, "prod_completion": L,
                              "prod_prompt": tr["prompt_len"][i], "t": time.perf_counter() - t_start, "running": state["running"]})
                    recs.append(r)
                    if "error" in r:
                        print(f"  turn error: {r['error']}", flush=True); break
                    await asyncio.sleep(a.gap)
            finally:
                state["running"] -= 1

    m0 = await metrics(session)
    await asyncio.gather(*[run_traj(tr) for tr in trajs])
    wall = time.perf_counter() - t_start; m1 = await metrics(session)
    steady = [r for r in recs if r["t"] >= a.steady_after and r["t"] <= a.time_budget]
    s_all = summarize(recs); s_st = summarize(steady)
    ok_st = [r for r in steady if "error" not in r]
    win = (max(r["t"] for r in ok_st) - a.steady_after) if ok_st else 1
    toks = sum(r["completion_tokens"] or 0 for r in ok_st)
    s_st.update({"window_s": win, "agg_tps": toks / win, "turns_per_min": 60 * len(ok_st) / win,
                 "mean_running": statistics.mean([r["running"] for r in ok_st]) if ok_st else None,
                 "prod_api_p50": pctl([r["prod_api_s"] for r in ok_st], .5), "prod_api_p90": pctl([r["prod_api_s"] for r in ok_st], .9),
                 "speedup_vs_prod_p50": (pctl([r["prod_api_s"] / r["latency"] for r in ok_st if r["latency"] > 0], .5)),
                 "prefill_tokens_per_turn_p50": pctl([(r["prompt_tokens"] or 0) - (r["prod_prompt"] or 0) for r in ok_st], .5)})
    res = {"mode": "replay", "forced": a.forced, "conc": conc, "gap": a.gap, "n_traj": len(trajs), "wall": wall,
           "all": s_all, "steady": s_st,
           "server_gen_tokens": m1.get("vllm:generation_tokens_total", 0) - m0.get("vllm:generation_tokens_total", 0),
           "server_prompt_tokens": m1.get("vllm:prompt_tokens_total", 0) - m0.get("vllm:prompt_tokens_total", 0),
           "server_gen_tps_overall": (m1.get("vllm:generation_tokens_total", 0) - m0.get("vllm:generation_tokens_total", 0)) / wall,
           "records": recs}
    print(f"RESULT {a.label} replay{'_forced' if a.forced else '_natural'} conc={conc} turns={s_st['n']} errors={s_st['errors']} "
          f"latency_p50={s_st['latency_p50']:.1f} latency_p90={s_st['latency_p90']:.1f} ttft_p50={s_st['ttft_p50']:.2f} ttft_p90={s_st['ttft_p90']:.2f} "
          f"decode_tps_p50={s_st['decode_tps_p50']:.1f} completion_p50={s_st['completion_tokens_p50']} agg_tps={s_st['agg_tps']:.0f} "
          f"turns_per_min={s_st['turns_per_min']:.1f} mean_running={s_st['mean_running']:.1f} prod_api_p50={s_st['prod_api_p50']:.1f} "
          f"speedup_vs_prod_p50={s_st['speedup_vs_prod_p50']:.2f} server_gen_tps={res['server_gen_tps_overall']:.0f}", flush=True)
    return res


async def main():
    timeout = aiohttp.ClientTimeout(total=3600, sock_read=600)
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn, read_bufsize=2 ** 20) as session:
        # warm-up request
        w = await chat(session, [{"role": "user", "content": "Say hello in one word."}], 8)
        if "error" in w:
            print("warmup error:", w["error"]); sys.exit(2)
        if a.mode == "sweep":
            res = await run_sweep(session)
        elif a.mode == "prefill":
            res = await run_prefill(session)
        else:
            res = await run_replay(session)
    res["label"] = a.label; res["args"] = vars(a)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        print("wrote", a.out, flush=True)

asyncio.run(main())
