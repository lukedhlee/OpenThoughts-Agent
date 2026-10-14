"""Behaviour probe of one export against stored reflections: resend N stored prompts to the served model, compare length,
cap hits and repetition (zlib ratio) with the stored completions of the same prompts. Run inside a job serving MODEL
as cf-student on :29010 (probe.sbatch).

    python -m data.relay.reflection.horizon.probe_export --generations <update dir>/generations --n 32 --out probe.json
"""

import argparse
import json
import statistics
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def stats(rows):
    tokens = sorted(r["tokens"] for r in rows)
    ratios = [r["zlib"] for r in rows if r["zlib"] is not None]
    return {
        "n": len(rows),
        "tokens_p50": tokens[len(tokens) // 2],
        "tokens_p90": tokens[int(0.9 * len(tokens))],
        "cap_hits": sum(r["finish"] == "length" for r in rows),
        "zlib_p50": round(statistics.median(ratios), 3) if ratios else None,
    }


def measure(text, tokens, finish):
    data = text.encode()
    return {"tokens": tokens, "finish": finish, "zlib": len(zlib.compress(data)) / len(data) if len(data) > 200 else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--url", default="http://127.0.0.1:29010/v1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    records = [json.loads(p.read_text()) for p in sorted(Path(a.generations).glob("*.json"))]
    records = [r for r in records if r.get("raw_response")][: a.n]

    def one(record):
        payload = {
            "model": "cf-student",
            "messages": [{"role": "user", "content": record["prompt"]}],
            "temperature": 0.9,
            "top_p": 1.0,
            "max_tokens": 8192,
            "skip_special_tokens": False,
        }
        req = urllib.request.Request(a.url + "/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as f:
            response = json.load(f)
        choice = response["choices"][0]
        return measure(choice["message"].get("content") or "", response["usage"]["completion_tokens"],
                       choice.get("finish_reason")), (choice["message"].get("content") or "")[:600]

    with ThreadPoolExecutor(16) as pool:
        probed = list(pool.map(one, records))
    stored = []
    for r in records:
        m = r["raw_response"]["choices"][0]["message"]
        stored.append(measure(m.get("content") or "", r["usage"]["completion_tokens"], r.get("finish_reason")))
    report = {"stored": stats(stored), "probed": stats([p for p, _ in probed]), "samples": [s for _, s in probed[:3]]}
    Path(a.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("stored", "probed")}))


if __name__ == "__main__":
    main()
