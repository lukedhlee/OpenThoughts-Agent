"""ifeval_gen.py — generate IFEval responses from a running vLLM chat endpoint (runs on the serve node).

    python ifeval_gen.py --url http://localhost:8000/v1 --model snowball --inp ifeval_input_data.jsonl --out <tag>.jsonl

Greedy (temperature 0), one response per prompt, raw content kept (think spans stripped at scoring time on the Mac,
score_ifeval.py). 32 concurrent requests = the server's --max-num-seqs. Resumable: prompts already in --out are skipped.
"""
import argparse, json, os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True); ap.add_argument("--model", default="snowball")
ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--max-tokens", type=int, default=8192); ap.add_argument("--conc", type=int, default=32)
ap.add_argument("--temperature", type=float, default=0.0)
a = ap.parse_args()

rows = [json.loads(l) for l in open(a.inp)]
done = set()
if os.path.exists(a.out):
    done = {json.loads(l)["key"] for l in open(a.out)}
todo = [r for r in rows if r["key"] not in done]
print(f"prompts {len(rows)} done {len(done)} todo {len(todo)}", flush=True)
client = OpenAI(base_url=a.url, api_key="x", timeout=1800, max_retries=2)
lock = threading.Lock(); t0 = time.time(); n = [0]

def one(r):
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(model=a.model, messages=[{"role": "user", "content": r["prompt"]}],
                max_tokens=a.max_tokens, temperature=a.temperature,
                extra_body={"skip_special_tokens": False})
            ch = resp.choices[0]
            rec = {"key": r["key"], "prompt": r["prompt"], "response": ch.message.content or "",
                   "reasoning": getattr(ch.message, "reasoning_content", None) or getattr(ch.message, "reasoning", None),
                   "finish_reason": ch.finish_reason, "usage": resp.usage.model_dump() if resp.usage else None,
                   "instruction_id_list": r["instruction_id_list"], "kwargs": r["kwargs"]}
            break
        except Exception as e:
            print(f"key {r['key']} attempt {attempt} error {e!r}", flush=True); time.sleep(5)
    else:
        rec = {"key": r["key"], "prompt": r["prompt"], "response": "", "finish_reason": "error", "usage": None,
               "instruction_id_list": r["instruction_id_list"], "kwargs": r["kwargs"]}
    with lock:
        with open(a.out, "a") as f: f.write(json.dumps(rec) + "\n")
        n[0] += 1
        if n[0] % 20 == 0: print(f"{n[0]}/{len(todo)} {time.time()-t0:.0f}s", flush=True)

with ThreadPoolExecutor(a.conc) as ex:
    list(ex.map(one, todo))
print(f"DONE {len(todo)} in {time.time()-t0:.0f}s -> {a.out}", flush=True)
