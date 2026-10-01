#!/usr/bin/env python3
"""summarize.py — table of draft_bench results (results_*.jsonl) with ON/OFF ratios, plus the server log's own
'SpecDecoding metrics' mean acceptance length averaged (draft-token weighted) over each measurement window.

  python summarize.py /scratch/11584/$USER/experiments/rl/draft_bench
"""
import glob, json, re, sys, time

D = sys.argv[1]
SPEC = re.compile(r"INFO (\d\d-\d\d \d\d:\d\d:\d\d) \[metrics.py:\d+\] SpecDecoding metrics: Mean acceptance length: "
                  r"([\d.]+).*?Accepted: (\d+) tokens, Drafted: (\d+) tokens")


def log_accept(label, job, w0, w1):
    model, conc, cfg = label.rsplit("_", 2)
    if cfg != "on":
        return None
    f = f"{D}/logs/server_{model}_on.{job}.log"
    year = time.localtime(w0).tm_year
    acc = drafted = 0
    try:
        for line in open(f, errors="replace"):
            m = SPEC.search(line)
            if not m:
                continue
            t = time.mktime(time.strptime(f"{year}-{m.group(1)}", "%Y-%m-%d %H:%M:%S"))
            if w0 + 10 <= t <= w1:
                acc += int(m.group(3)); drafted += int(m.group(4))
    except FileNotFoundError:
        return None
    return round(1 + 3 * acc / drafted, 3) if drafted else None  # k=3: drafts = drafted/3


rows = {}
for f in sorted(g for g in glob.glob(f"{D}/results_*.jsonl") if ".reqs." not in g):
    job = f.rsplit(".", 2)[-2]
    for line in open(f):
        r = json.loads(line)
        r["job"] = job
        r["log_accept"] = log_accept(r["label"], job, r["w0"], r["w1"])
        rows[r["label"]] = r
hdr = ("label", "job", "node_out_tok_s", "req_decode_tok_s", "req_tpot_inv", "client_req_tok_s_mean", "accept_len",
       "log_accept", "accept_per_pos", "ttft_mean_s", "prefix_hit_rate", "node_prompt_tok_s", "client_reqs",
       "client_errs", "client_compl_mean", "running_mean", "waiting_mean", "kv_usage_mean", "preemptions")
print("\t".join(hdr))
for k in sorted(rows):
    print("\t".join(str(rows[k].get(h)) for h in hdr))
print()
for k in sorted(rows):
    if k.endswith("_on") and k[:-3] + "_off" in rows:
        on, off = rows[k], rows[k[:-3] + "_off"]
        rat = lambda h: round(on[h] / off[h], 3) if on.get(h) and off.get(h) else None
        print(f"{k[:-3]}: ON/OFF node_out {rat('node_out_tok_s')}  req_decode {rat('req_decode_tok_s')}  "
              f"tpot_inv {rat('req_tpot_inv')}  client_req {rat('client_req_tok_s_mean')}  ttft {rat('ttft_mean_s')}")
