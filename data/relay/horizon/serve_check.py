#!/usr/bin/env python3
"""serve_check.py JOBID [--conc N] — real chat completions against a relay serve job's endpoints (stdlib only, so it runs
on the Horizon login node or inside the job). Teacher: a thinking question per server; PASS needs parsed reasoning, a
non-empty answer and no think tags left in it. Student: the serve_relay.sbatch smoke (think markers kept). --conc N sends
N requests at once to every server and reports decode tokens/s per server.

    python3 serve_check.py 35812 [--conc 16] [--exp /scratch/11584/$USER/experiments/relay/pilot]
"""
import argparse, json, os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

TEACHER_Q = ("A directory tree holds Python files. Write one bash pipeline that prints the 3 .py files with the most "
             "lines, largest first, and say in one sentence why it handles file names with spaces.")


def post(url, body, timeout=900):
    req = urllib.request.Request(url + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r), time.time() - t


def teacher(url, max_tokens):
    r, dt = post(url, {"model": "qwen38", "messages": [{"role": "user", "content": TEACHER_Q}], "max_tokens": max_tokens})
    m, ch = r["choices"][0]["message"], r["choices"][0]
    rs = m.get("reasoning") or m.get("reasoning_content") or ""
    c = m.get("content") or ""
    ok = bool(rs) and bool(c.strip()) and "<think>" not in c and "</think>" not in c
    return ok, dt, r.get("usage", {}), ch["finish_reason"], rs, c


def student(url, max_tokens):
    r, dt = post(url, {"model": "snowball", "messages": [{"role": "user", "content": "Print hello in bash."}],
                       "max_tokens": max_tokens, "skip_special_tokens": False})
    c = r["choices"][0]["message"].get("content") or ""
    return bool(c.strip()), dt, r.get("usage", {}), r["choices"][0]["finish_reason"], "", c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("--exp", default=os.environ.get("RELAY_EXP_DIR", f"/scratch/11584/{os.environ.get('USER')}/experiments/relay/pilot"))
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=6000)
    a = ap.parse_args()
    ep = os.path.join(a.exp, "endpoints", a.job)
    urls = {role: [u for u in open(f"{ep}.{role}").read().strip().split(",") if u] if os.path.exists(f"{ep}.{role}") else []
            for role in ("student", "teacher")}
    print("endpoints", json.dumps(urls))
    bad = 0
    for role, fn in (("teacher", teacher), ("student", student)):
        jobs = [(u, i) for u in urls[role] for i in range(a.conc)]
        if not jobs:
            continue
        t0 = time.time()
        with ThreadPoolExecutor(len(jobs)) as ex:
            res = list(ex.map(lambda j: (j[0], j[1], *fn(j[0], a.max_tokens)), jobs))
        wall = time.time() - t0
        for u in urls[role]:
            rows = [r for r in res if r[0] == u]
            toks = sum(r[4].get("completion_tokens", 0) for r in rows)
            nok = sum(r[2] for r in rows)
            bad += len(rows) - nok
            _, _, ok, dt, us, fin, rs, c = rows[0]
            print(f"{role.upper()} {u} pass={nok}/{len(rows)} tok={toks} tok/s={toks / max(r[3] for r in rows):.0f} "
                  f"first: {dt:.1f}s finish={fin} usage={us} reasoning_chars={len(rs)}")
            if rs:
                print("  reasoning[:300] =", repr(rs[:300]))
            print("  content[:500]   =", repr(c[:500]))
        print(f"{role} wall {wall:.1f}s for {len(jobs)} requests")
    print("SERVE_CHECK", "PASS" if bad == 0 else f"FAIL ({bad} bad)")


if __name__ == "__main__":
    main()
