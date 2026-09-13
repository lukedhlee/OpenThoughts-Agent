#!/usr/bin/env python3
"""score_shift.py: policy-shift scoring for the P2O guidance blocks against a local vLLM server (no generation).

Two modes, one machinery. Every assistant turn's served completion C_t is re-scored under two versions of the first user
message via /v1/completions prompt_logprobs; served prompts come from metrics.prompt_token_ids and are round-trip checked
through the tokenizer; edits are made at the text level (where instruction.md ends, before "\n\nCurrent terminal state:")
and re-encoded, which is what the harness itself would have served.

  --mode prescreen  control trajectories (09-06 tthd, header prompt) -> variants ctl, A..E (block inserted).
                    log-ratio = lp(variant) - lp(ctl) on the CONTROL's own tokens (reverse direction, needs no rollouts).
  --mode wave       prompted trajectories from the wave shards (<task>-p<arm>) -> own (as served) vs bare (block removed).
                    log-ratio = lp(own) - lp(bare) on the PROMPTED tokens (forward KL direction, what TIS would see).

Per (task, trajectory, variant) record: n tokens, sum log-ratio, sum |log-ratio|, count log-ratio > ln2 and < -ln2 (the
TIS cap), per-turn-bucket sums. --merge prints per-variant/arm: mean per-token log-ratio, total nats per trajectory,
capped fraction, per-turn profile. Python 3.9; needs transformers (tokenizer) only; server via urllib.
"""
import argparse, collections, csv, glob, json, math, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

E = "/e/fscratch/reformo/lee27/experiments"
M = "/e/fscratch/reformo/lee27/models/snowball-s3-nemotron-terminal-step1888"
MARK = "\n\nCurrent terminal state:"
LN2 = math.log(2)
BUCKETS = [(1, 1), (2, 3), (4, 6), (7, 10), (11, 15), (16, 25), (26, 10 ** 6)]
ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["prescreen", "wave"], default="prescreen")
ap.add_argument("--dev", default=E + "/p2o/dev120.tsv")
ap.add_argument("--blocks", default=E + "/p2o/blocks.json")
ap.add_argument("--traces", default=E + "/tthd_s*/tthd_s*/trace_jobs/eval_sessions/*," + E + "/snowball_v2rest_base_fixed/snowball_v2rest_base_fixed/trace_jobs/eval_sessions/*",
                help="glob of eval-session dirs; wave mode: the shard trees, e.g. '$E/p2o6*_s0/p2o6*_s0/trace_jobs/eval_sessions/*'")
ap.add_argument("--per-task", type=int, default=3, help="prescreen: trajectories per task (wins first, then losses)")
ap.add_argument("--url", default="http://localhost:8000"); ap.add_argument("--served", default="snowball")
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--out", default=E + "/p2o/prescreen")
ap.add_argument("--max-len", type=int, default=65536)
ap.add_argument("--tokenizer", default=M, help="tokenizer dir (an exported checkpoint carries the base tokenizer)")
ap.add_argument("--time-budget", type=int, default=0, help="seconds; stop starting new units after this")
ap.add_argument("--merge", action="store_true"); ap.add_argument("--dry", action="store_true")
ap.add_argument("--per-turn", action="store_true", help="score every turn's served prompt separately (slow: DP routing defeats the prefix cache); default = one request per unit-variant on the final context, assistant spans located inside it")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)


def bucket(t):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= t <= hi:
            return i
    return len(BUCKETS) - 1


if a.merge:
    rows = [json.loads(l) for p in sorted(glob.glob(a.out + "/*.jsonl")) for l in open(p)]
    by = collections.defaultdict(list)
    for r in rows:
        if r["variant"] != "ctl":
            by[r["variant"]].append(r)
    lines = ["variant\tn_traj\tn_tok\tmean_logratio_per_tok\tmean_abs_per_tok\ttotal_nats_per_traj\tfrac_gt_ln2\tfrac_lt_-ln2\t" +
             "\t".join("turn%d-%d" % (lo, hi if hi < 10 ** 6 else 99) for lo, hi in BUCKETS)]
    for v in sorted(by):
        rs = by[v]; n = sum(r["n"] for r in rs)
        if not n:
            continue
        prof = []
        for i in range(len(BUCKETS)):
            nb = sum(r["bn"][i] for r in rs); prof.append(("%.4f" % (sum(r["bs"][i] for r in rs) / nb)) if nb else "-")
        lines.append("%s\t%d\t%d\t%.5f\t%.5f\t%.2f\t%.4f\t%.4f\t%s" % (
            v, len(rs), n, sum(r["s"] for r in rs) / n, sum(r["sa"] for r in rs) / n, sum(r["s"] for r in rs) / len(rs),
            sum(r["gt"] for r in rs) / n, sum(r["lt"] for r in rs) / n, "\t".join(prof)))
    open(a.out + "/summary.tsv", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
    pt = collections.defaultdict(lambda: [0.0, 0])
    for v in by:
        for r in by[v]:
            pt[(v, r["task"])][0] += r["s"]; pt[(v, r["task"])][1] += r["n"]
    with open(a.out + "/per_task.tsv", "w") as f:
        f.write("variant\ttask\tmean_logratio_per_tok\tn_tok\n")
        for (v, t), (s, n) in sorted(pt.items()):
            f.write("%s\t%s\t%.5f\t%d\n" % (v, t, s / n if n else 0, n))
    sys.exit(0)

B = json.load(open(a.blocks)); DELIM = B["delim"]
INS = {k: DELIM + v + ("" if v.endswith("\n") else "\n") for k, v in B["blocks"].items()}

from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(a.tokenizer)
enc = lambda s: tok.encode(s, add_special_tokens=False)
dec = lambda ids: tok.decode(ids, skip_special_tokens=False)


def units():
    """(task, arm, traj_idx, trajectory.json) work units."""
    out = []
    if a.mode == "prescreen":
        tasks = [r["task"] for r in csv.DictReader(open(a.dev), delimiter="\t")]
        for t in tasks:
            atts = []
            for d in sorted(x for g in a.traces.split(",") for x in glob.glob(g + "/%s__*/attempts/*" % t)):
                tj = os.path.join(d, "agent", "trajectory.json")
                if not os.path.exists(tj):
                    continue
                try:
                    rw = float(open(os.path.join(d, "verifier", "reward.txt")).read().strip())
                except Exception:
                    rw = None
                atts.append((rw, tj))
            wins = [x for x in atts if x[0] == 1.0]; rest = [x for x in atts if x[0] != 1.0]
            pick = (wins[: a.per_task // 2] + rest)[: a.per_task]
            pick += [x for x in wins if x not in pick][: a.per_task - len(pick)]
            out += [(t, "ctl", j, tj) for j, (_, tj) in enumerate(pick)]
    else:
        for td in sorted(x for g in a.traces.split(",") for x in glob.glob(g + "/*__*")):
            name = os.path.basename(td).split("__", 1)[0]
            if "-p" not in name:
                continue
            task, arm = name.rsplit("-p", 1)
            if arm == "ctl":
                continue
            for j, tj in enumerate(sorted(glob.glob(td + "/attempts/*/agent/trajectory.json"))):
                out.append((task, arm, j, tj))
    return out


def prompt_logprobs(ids):
    body = json.dumps({"model": a.served, "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1}).encode()
    req = urllib.request.Request(a.url + "/v1/completions", data=body, headers={"Content-Type": "application/json"})
    err = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                return json.load(r)["choices"][0]["prompt_logprobs"]
        except Exception as e:  # noqa: BLE001
            err = e; time.sleep(5 * (attempt + 1))
    raise RuntimeError("server failed: %s" % err)


def lp_of(ids, start):
    """per-token logprobs of ids[start:] given the prefix (list, None where missing)."""
    pl = prompt_logprobs(ids)
    out = []
    for j in range(start, len(ids)):
        d = pl[j] if j < len(pl) else None
        k = str(ids[j])
        if d is None:
            out.append(None)
        elif k in d:
            out.append(d[k]["logprob"])
        elif ids[j] in d:
            out.append(d[ids[j]]["logprob"])
        else:
            out.append(None)
    return out


def find_sub(sub, seq, start):
    n = len(sub); h = tuple(sub[:8])
    for i in range(start, len(seq) - n + 1):
        if tuple(seq[i:i + 8]) == h and seq[i:i + n] == sub:
            return i
    return -1


def run_unit_whole(u):
    """One prompt_logprobs request per variant on the FINAL served context; earlier completions are located as
    contiguous spans inside it (23-25 of 25 in the 09-06 traces; missing ones are dropped and counted)."""
    task, arm, j, tj = u
    d = json.load(open(tj))
    steps = [s for s in d["steps"] if s.get("source") == "agent" and s.get("metrics", {}).get("prompt_token_ids")]
    if not steps:
        return None, "no_steps"
    P = steps[-1]["metrics"]["prompt_token_ids"]; C = steps[-1]["metrics"].get("completion_token_ids") or []
    text = dec(P)
    if enc(text) != P:
        return None, "roundtrip_fail"
    off = text.find(MARK)
    if off < 0:
        return None, "no_mark"
    head = text[:off]
    spans = []; pos = 0; stats = collections.Counter()
    for t, s in enumerate(steps[:-1], 1):
        c = s["metrics"].get("completion_token_ids") or []
        if not c:
            continue
        i = find_sub(c, P, pos)
        if i < 0:
            stats["span_missing"] += 1; continue
        spans.append((t, i, i + len(c))); pos = i + len(c)
    if C:
        spans.append((len(steps), len(P), len(P) + len(C)))
    if not spans:
        return None, "no_spans"
    if a.mode == "prescreen":
        pairs = [("ctl", None)] + [(v, ins) for v, ins in INS.items()]
    else:
        ins = INS[arm]
        if not head.endswith(ins):
            return None, "own_prompt_lacks_block"
        pairs = [("bare", None), (arm, ins)]
    meta = dict(task=task, traj=j, arm=arm, src=tj)
    acc = {v: dict(n=0, s=0.0, sa=0.0, gt=0, lt=0) for v, _ in pairs}
    bs = {v: [0.0] * len(BUCKETS) for v, _ in pairs}; bn = {v: [0] * len(BUCKETS) for v, _ in pairs}
    base_lp = None; base_start = spans[0][1]
    for v, ins in pairs:
        if a.mode == "prescreen":
            Pv = P if ins is None else enc(head + ins + text[off:])
        else:
            Pv = P if ins is not None else enc(head[: len(head) - len(INS[arm])] + text[off:])
        delta = len(Pv) - len(P)
        ids = Pv + C
        cap = a.max_len - 1
        if len(ids) > cap:
            ids = ids[:cap]; stats["clipped"] += 1
        if a.dry:
            lps = [0.0] * (len(ids) - (base_start + delta))
        else:
            lps = lp_of(ids, base_start + delta)  # logprobs from the first span onward, indexed relative to base_start
        if base_lp is None:
            base_lp = lps; continue
        for t, s0, e0 in spans:
            b = bucket(t)
            for k in range(s0 - base_start, e0 - base_start):
                if k >= len(lps) or k >= len(base_lp):
                    break
                x, y = lps[k], base_lp[k]
                if x is None or y is None:
                    continue
                lr = x - y
                r = acc[v]; r["n"] += 1; r["s"] += lr; r["sa"] += abs(lr); r["gt"] += int(lr > LN2); r["lt"] += int(lr < -LN2)
                bs[v][b] += lr; bn[v][b] += 1
    stats["spans"] += len(spans)
    recs = [dict(meta, variant=v, bs=bs[v], bn=bn[v], **acc[v]) for v, _ in pairs[1:]]
    return recs, dict(stats)


def run_unit(u):
    if not a.per_turn:
        return run_unit_whole(u)
    task, arm, j, tj = u
    d = json.load(open(tj))
    steps = [s for s in d["steps"] if s.get("source") == "agent" and s.get("metrics", {}).get("prompt_token_ids")]
    if not steps:
        return None, "no_steps"
    P1 = steps[0]["metrics"]["prompt_token_ids"]; text1 = dec(P1)
    if enc(text1) != P1:
        return None, "roundtrip_fail"
    off = text1.find(MARK)
    if off < 0:
        return None, "no_mark"
    head = text1[:off]
    if a.mode == "prescreen":
        pairs = [("ctl", None)] + [(v, ins) for v, ins in INS.items()]  # base = ctl (served); variant = inserted
    else:
        ins = INS[arm]
        if not head.endswith(ins):
            return None, "own_prompt_lacks_block"
        pairs = [("bare", None), (arm, ins)]
    meta = dict(task=task, traj=j, arm=arm, src=tj)
    acc = {v: dict(n=0, s=0.0, sa=0.0, gt=0, lt=0) for v, _ in pairs}
    bs = {v: [0.0] * len(BUCKETS) for v, _ in pairs}; bn = {v: [0] * len(BUCKETS) for v, _ in pairs}
    stats = collections.Counter()
    for t, s in enumerate(steps, 1):
        P = s["metrics"]["prompt_token_ids"]; C = s["metrics"].get("completion_token_ids") or []
        if not C:
            continue
        text = dec(P)
        if not text.startswith(head) or enc(text) != P:
            stats["turn_roundtrip_fail"] += 1; continue
        base_lp = None
        for v, ins in pairs:
            if a.mode == "prescreen":
                Pv = P if ins is None else enc(head + ins + text[off:])
            else:
                Pv = P if ins is not None else enc(head[: len(head) - len(INS[arm])] + text[off:])
            ids = Pv + C
            if len(ids) > a.max_len - 1:
                ids = ids[: a.max_len - 1]; stats["clipped"] += 1
            if a.dry:
                lps = [0.0] * (len(ids) - len(Pv))
            else:
                lps = lp_of(ids, len(Pv))
            if base_lp is None:
                base_lp = lps; continue
            r = acc[v]; b = bucket(t)
            for x, y in zip(lps, base_lp):
                if x is None or y is None:
                    continue
                lr = x - y
                r["n"] += 1; r["s"] += lr; r["sa"] += abs(lr); r["gt"] += int(lr > LN2); r["lt"] += int(lr < -LN2)
                bs[v][b] += lr; bn[v][b] += 1
        stats["turns"] += 1
    recs = [dict(meta, variant=v, bs=bs[v], bn=bn[v], **acc[v]) for v, _ in pairs[1:]]
    return recs, dict(stats)


if __name__ == "__main__":
    U = units()
    print("%s: %d units, %d workers" % (a.mode, len(U), a.workers), flush=True)
    outp = open(a.out + "/%s.jsonl" % a.mode, "a")
    t0 = time.time(); done = 0; tot = collections.Counter()
    def _go(u):
        if a.time_budget and time.time() - t0 > a.time_budget:
            return u, None, "budget"
        try:
            recs, st = run_unit(u); return u, recs, st
        except Exception as e:  # noqa: BLE001
            return u, None, "error: %s" % e
    with ThreadPoolExecutor(a.workers) as ex:
        for u, recs, st in ex.map(_go, U):
            done += 1
            if recs:
                for r in recs:
                    outp.write(json.dumps(r) + "\n")
                outp.flush(); tot["ok"] += 1; tot["tok"] += int(recs[0]["n"])
            else:
                tot[str(st)[:40]] += 1
            if done % 5 == 0 or done == len(U):
                el = time.time() - t0
                print("%d/%d units, %s, %.0fs elapsed, %.0f assistant tok/s per unit-variant" % (done, len(U), dict(tot), el, tot["tok"] / max(1, el)), flush=True)
    print("DONE", dict(tot), flush=True)
