"""Serve smoke for Snowball on GH200: coherence + single-stream + concurrent throughput via the OpenAI API."""
import argparse, json, statistics, sys, threading, time, itertools
_NONCE = itertools.count()
import requests
ap = argparse.ArgumentParser(); ap.add_argument("--label", default="run"); ap.add_argument("--url", default="http://localhost:8000")
ap.add_argument("--max-tokens", type=int, default=512); ap.add_argument("--conc", default="16,64"); ap.add_argument("--pad-tokens", type=int, default=0)
a = ap.parse_args()
if a.pad_tokens:
    _filler = ("def f%d(x):\n    return x * %d + 1\n\n" % (0, 0))
    _blob = "".join("def f%d(x):\n    return x * %d + 1\n\n" % (i, i) for i in range(1, a.pad_tokens))
    _blob = _blob[: a.pad_tokens * 3]  # ~3 chars/token for this code-like filler
    PAD = "Here is a large source file for context; ignore it unless asked.\n\n" + _blob + "\n\nNow the actual task:\n"
else:
    PAD = ""
PROMPTS = [
  "Find the smallest distance between the origin and a point on the graph of y = x^2/2 - 9. Give a^2 where a is that distance. Reason step by step and put the final answer in \\boxed{}.",
  "You are in a Linux shell. A Python project at /app has a failing test in tests/test_parse.py. Describe, as a numbered plan, the first five shell commands you would run to diagnose it and why.",
  "Write a Python function that returns the k most frequent elements of a list, with a short docstring, then give one example call and its output.",
  "Explain in three sentences why expert-parallel inference for a sparse MoE shards experts rather than attention.",
]
def chat(prompt, max_tokens):
    t0 = time.time()
    r = requests.post(f"{a.url}/v1/chat/completions", json={"model": "snowball", "messages": [{"role": "user", "content": (("request-id %d\n" % next(_NONCE)) if PAD else "") + PAD + prompt}],
                      "max_tokens": max_tokens, "temperature": 0.6, "top_p": 0.95}, timeout=1800)
    r.raise_for_status(); d = r.json(); dt = time.time() - t0
    return d["choices"][0]["message"]["content"], d["usage"]["completion_tokens"], d["usage"]["prompt_tokens"], dt, d["choices"][0].get("finish_reason")
# warm-up
chat("Say hello in one word.", 8)
# coherence + single stream
print(f"### {a.label} single-stream")
single = []
for p in PROMPTS[:2]:
    txt, ct, pt, dt, fr = chat(p, a.max_tokens)
    single.append(ct / dt)
    print(f"--- prompt: {p[:60]}...\n    prompt_tokens={pt} completion_tokens={ct} finish={fr} {ct/dt:.1f} tok/s\n    head: {txt[:300]!r}\n    tail: {txt[-200:]!r}")
print(f"RESULT {a.label} single_stream_tok_s={statistics.mean(single):.1f}")
# concurrency sweep
for conc in [int(x) for x in a.conc.split(",")]:
    res = []; lock = threading.Lock()
    def work(i):
        try:
            _, ct, pt, dt, fr = chat(PROMPTS[i % len(PROMPTS)], a.max_tokens)
            with lock: res.append((ct, dt, fr))
        except Exception as e:
            with lock: res.append((0, 0, f"ERR {e}"))
    t0 = time.time(); th = [threading.Thread(target=work, args=(i,)) for i in range(conc)]
    [t.start() for t in th]; [t.join() for t in th]; wall = time.time() - t0
    toks = sum(r[0] for r in res); errs = sum(1 for r in res if str(r[2]).startswith("ERR"))
    print(f"RESULT {a.label} conc={conc} aggregate_tok_s={toks/wall:.1f} per_stream_tok_s={toks/wall/conc:.2f} wall={wall:.1f}s errors={errs} finish={[r[2] for r in res[:4]]}")
print("SMOKE_OK")
