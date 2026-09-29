#!/usr/bin/env python3
"""Horizon port gate 2 (ai_memory/active/horizon-port/objective.md): do the Blackwell kernels compute the same model?

Fixed prompts with fixed completions (exact token ids the step-1888 policy sampled in an R2E-Gym probe) are scored with
vLLM prompt logprobs on each serving config. Per prompt the gap is the mean over completion tokens of |p_A - p_B|
(probabilities). PASS = gap(Horizon, Jupiter-DP4) is not significantly larger than gap(Jupiter-DP4, Jupiter-TP4)
(paired 95 % bootstrap over prompts). TP4 does not exist for GrugMoE (tensor_parallel_size=1 only),
so the second Jupiter layout is DP2-EP2 (--j2).

  build    Jupiter login node, stdlib only: one turn per task from a probe's result.json files -> set JSONL
  score    inside a serve job: POST /v1/completions with prompt_logprobs=0, write the completion-token logprobs
  compare  anywhere, stdlib only: gaps, paired bootstrap, outliers, verdict
"""

import argparse
import concurrent.futures as cf
import glob
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
import urllib.request

START_THINK = 128002  # <|start_think|>


# ---------------------------------------------------------------- build
def build(a):
    paths = sorted(glob.glob(os.path.join(a.trials, "*", "attempts", "*", "result.json")))
    by_task = {}
    for p in paths:
        trial = p.split(os.sep)[-4]
        by_task.setdefault(trial.split("__")[0], []).append(p)
    rng = random.Random(a.seed)
    tasks = sorted(by_task)
    rng.shuffle(tasks)
    edges = [int(x) for x in a.edges.split(",")]
    nbins = len(edges) - 1
    if a.n % nbins:
        sys.exit("--n must divide evenly over the bins")
    quota = a.n // nbins
    filled = [0] * nbins
    rows, read = [], 0
    for task in tasks:
        if min(filled) >= quota:
            break
        cands = sorted(by_task[task])
        rng.shuffle(cands)
        for p in cands:
            read += 1
            try:
                d = json.load(open(p))
            except Exception:
                continue
            rd = ((d.get("agent_result") or {}).get("rollout_details") or [None])[0] or {}
            P, C, L = rd.get("prompt_token_ids") or [], rd.get("completion_token_ids") or [], rd.get("logprobs") or []
            if not P or len(P) != len(C):
                continue
            model = ((d.get("config") or {}).get("agent") or {}).get("model_name", "")
            per_bin = [[] for _ in range(nbins)]
            for t in range(len(P)):
                total, c = len(P[t]) + len(C[t]), len(C[t])
                if c < a.min_completion or (L and len(L) > t and L[t] is not None and len(L[t]) != c):
                    continue
                for b in range(nbins):
                    if edges[b] <= total < edges[b + 1]:
                        per_bin[b].append(t)
            open_bins = [b for b in range(nbins) if filled[b] < quota and per_bin[b]]
            if not open_bins:
                break  # this task has nothing the open bins need; one trial per task
            b = min(open_bins, key=lambda x: (filled[x], x))
            t = rng.choice(per_bin[b])
            rows.append({
                "id": f"{d.get('trial_name')}_t{t}", "task": task, "trial": d.get("trial_name"), "turn": t,
                "bin": f"{edges[b]}-{edges[b + 1]}", "model": model, "source": p,
                "prompt_ids": P[t], "completion_ids": C[t],
                "orig_logprobs": L[t] if L and len(L) > t else None,
            })
            filled[b] += 1
            break
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    sha = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    comp = [len(r["completion_ids"]) for r in rows]
    tot = [len(r["prompt_ids"]) + len(r["completion_ids"]) for r in rows]
    print(json.dumps({"out": a.out, "sha256": sha, "n": len(rows), "files_read": read, "per_bin": dict(zip(
        [f"{edges[b]}-{edges[b + 1]}" for b in range(nbins)], filled)), "completion_tokens": sum(comp),
        "completion_p50": statistics.median(comp), "total_tokens": sum(tot), "total_max": max(tot),
        "models": sorted({r["model"] for r in rows})}, indent=1))
    if len(rows) < a.n:
        sys.exit(f"only {len(rows)} of {a.n} prompts found")


# ---------------------------------------------------------------- score
def post(url, body, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def lp_of(entry, tok):
    if entry is None:
        return None
    e = entry.get(str(tok))
    return None if e is None else e["logprob"]


def score_one(a, row):
    P, C = row["prompt_ids"], row["completion_ids"]
    t0 = time.time()
    r = post(a.url + "/v1/completions", {"model": a.model, "prompt": P + C, "max_tokens": 1, "temperature": 0.0,
                                         "prompt_logprobs": 0})
    pl = r["choices"][0]["prompt_logprobs"]
    if len(pl) != len(P) + len(C):
        raise ValueError(f"{row['id']}: {len(pl)} prompt logprobs for {len(P) + len(C)} tokens")
    lps = [lp_of(pl[len(P) + j], tok) for j, tok in enumerate(C)]
    if any(x is None for x in lps):
        raise ValueError(f"{row['id']}: completion token missing from its prompt-logprob entry")
    return {"id": row["id"], "lp": lps, "secs": round(time.time() - t0, 2)}


NEWLINE = 198
DOUBLE = [START_THINK, NEWLINE, START_THINK]  # the Horizon smoke 35811 opening: '<|start_think|>\n<|start_think|>\n'


def think_probe(a):
    """The Horizon smoke sample (35811, draft on, temperature 1.0) that opened '<|start_think|>\\n<|start_think|>\\n':
    the target's P of each opening token scored directly (prompt logprobs), and the rate of that opening over seeded
    samples at the RL sampler (temperature 1.0, top_p 1, top_k -1), the same seeds on every cluster."""
    msgs = [{"role": "user", "content": "Print hello in bash."}]
    ids = post(a.url + "/tokenize", {"model": a.model, "messages": msgs, "add_generation_prompt": True})["tokens"]
    top = lambda e: sorted(((int(k), round(math.exp(v["logprob"]), 6)) for k, v in e.items()), key=lambda x: -x[1])[:5]
    out = {"prompt_len": len(ids)}
    try:
        r = post(a.url + "/v1/completions", {"model": a.model, "prompt": ids + DOUBLE, "max_tokens": 1,
                                             "temperature": 0.0, "prompt_logprobs": 5})
        pl = r["choices"][0]["prompt_logprobs"]
        out["p_opening_tokens"] = [math.exp(lp_of(pl[len(ids) + j], t)) for j, t in enumerate(DOUBLE)]
        out["top5_before_second_start_think"] = top(pl[len(ids) + 2])
    except Exception as e:  # prompt logprobs with the draft loaded: record, keep sampling
        out["prompt_logprobs_error"] = str(e)[:200]
    firsts = []
    for seed in range(a.think_samples):
        g = post(a.url + "/v1/completions", {"model": a.model, "prompt": ids, "max_tokens": 6, "temperature": 1.0,
                                             "top_p": 1.0, "top_k": -1, "seed": seed, "return_token_ids": True,
                                             "skip_special_tokens": False})
        firsts.append(g["choices"][0].get("token_ids") or [])
    out["samples"] = len(firsts)
    out["double_start_think"] = sum(1 for t in firsts if t[:3] == DOUBLE)
    out["single_start_think"] = sum(1 for t in firsts if t[:1] == [START_THINK])
    out["first_tokens_by_seed"] = [t[:3] for t in firsts]
    return out


def think(a):
    th = think_probe(a)
    th["label"] = a.label
    if a.out:
        open(a.out, "w").write(json.dumps(th) + "\n")
    print("THINK " + json.dumps({k: v for k, v in th.items() if k != "first_tokens_by_seed"}), flush=True)


def score(a):
    rows = [json.loads(l) for l in open(a.set)]
    meta = {"label": a.label, "set": a.set, "set_sha256": hashlib.sha256(open(a.set, "rb").read()).hexdigest(),
            "n": len(rows), "conc": a.conc, "reps": a.reps, "host": os.uname().nodename,
            "job": os.environ.get("SLURM_JOB_ID")}
    print(json.dumps(meta), flush=True)
    with open(a.out, "w") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for rep in range(a.reps):
            t0 = time.time()
            with cf.ThreadPoolExecutor(a.conc) as ex:
                futs = [ex.submit(score_one, a, r) for r in rows]
                for i, fu in enumerate(futs):
                    res = fu.result()
                    res["rep"] = rep
                    f.write(json.dumps(res) + "\n")
                    if (i + 1) % 16 == 0:
                        print(f"rep {rep}: {i + 1}/{len(rows)} scored, {time.time() - t0:.0f}s", flush=True)
            f.flush()
            print(f"SCORE_REP_DONE {a.label} rep={rep} n={len(rows)} secs={time.time() - t0:.0f}", flush=True)
        if a.think_samples:
            th = think_probe(a)
            f.write(json.dumps({"think": th}) + "\n")
            print("THINK " + json.dumps({k: v for k, v in th.items() if k != "first_tokens_by_seed"}), flush=True)
    print(f"SCORE_DONE {a.label} -> {a.out}", flush=True)


# ---------------------------------------------------------------- compare
def load_scores(path, rep):
    lp, think, meta = {}, None, None
    for l in open(path):
        d = json.loads(l)
        if "meta" in d:
            meta = d["meta"]
        elif "think" in d:
            think = d["think"]
        elif d["rep"] == rep:
            lp[d["id"]] = d["lp"]
    return lp, think, meta


def per_prompt(ids, x, y):
    gp, gl = [], []
    for i in ids:
        a, b = x[i], y[i]
        gp.append(statistics.fmean(abs(math.exp(u) - math.exp(v)) for u, v in zip(a, b)))
        gl.append(statistics.fmean(abs(u - v) for u, v in zip(a, b)))
    return gp, gl


def boot_ci(d, n, seed):
    rng = random.Random(seed)
    k = len(d)
    ms = sorted(statistics.fmean(d[rng.randrange(k)] for _ in range(k)) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n) - 1]


def outliers(rows, x, y, top):
    worst = []
    for r in rows:
        for j, (u, v) in enumerate(zip(x[r["id"]], y[r["id"]])):
            worst.append((abs(math.exp(u) - math.exp(v)), r["id"], j, r["completion_ids"][j], math.exp(u), math.exp(v)))
    worst.sort(reverse=True)
    return [{"dp": round(w[0], 4), "id": w[1], "pos": w[2], "token": w[3], "p_a": round(w[4], 4),
             "p_b": round(w[5], 4)} for w in worst[:top]]


def compare(a):
    rows = [json.loads(l) for l in open(a.set)]
    ids = [r["id"] for r in rows]
    H, thH, mH = load_scores(a.horizon, a.rep)
    D, thD, mD = load_scores(a.dp4, a.rep)
    T, thT, mT = load_scores(a.j2, a.rep)
    for name, s in (("horizon", H), ("dp4", D), ("j2", T)):
        missing = [i for i in ids if i not in s]
        if missing:
            sys.exit(f"{name}: {len(missing)} prompts missing (e.g. {missing[:3]})")
    hd_p, hd_l = per_prompt(ids, H, D)
    dt_p, dt_l = per_prompt(ids, D, T)
    diff = [u - v for u, v in zip(hd_p, dt_p)]
    lo, hi = boot_ci(diff, a.boot, a.seed)
    verdict = "PASS" if lo <= 0 else "FAIL"
    ntok = sum(len(r["completion_ids"]) for r in rows)
    res = {
        "n_prompts": len(ids), "completion_tokens": ntok, "rep": a.rep,
        "gap_horizon_vs_jdp4": statistics.fmean(hd_p), "gap_jdp4_vs_j2": statistics.fmean(dt_p),
        "ratio": statistics.fmean(hd_p) / statistics.fmean(dt_p),
        "mean_diff": statistics.fmean(diff), "ci95": [lo, hi],
        "prompts_where_horizon_gap_larger": sum(1 for x in diff if x > 0),
        "dlogprob_horizon_vs_jdp4": statistics.fmean(hd_l), "dlogprob_jdp4_vs_j2": statistics.fmean(dt_l),
        "verdict": verdict,
        "rule": "PASS iff the 95% paired-bootstrap CI of mean(gap_HJ - gap_JJ) has lower bound <= 0",
    }
    # the same numbers per prompt-length bin, for context
    bins = {}
    for r, x, y in zip(rows, hd_p, dt_p):
        bins.setdefault(r["bin"], []).append((x, y))
    res["by_bin"] = {b: {"n": len(v), "gap_hj": statistics.fmean(x for x, _ in v),
                         "gap_jj": statistics.fmean(y for _, y in v)} for b, v in sorted(bins.items(),
                                                                                       key=lambda kv: int(kv[0].split("-")[0]))}
    # token-weighted means (every completion token equal weight) for context
    tw = lambda x, y: statistics.fmean(abs(math.exp(u) - math.exp(v)) for i in ids for u, v in zip(x[i], y[i]))
    res["token_weighted"] = {"horizon_vs_jdp4": tw(H, D), "jdp4_vs_j2": tw(D, T), "horizon_vs_j2": tw(H, T)}
    res["outliers_horizon_vs_jdp4"] = outliers(rows, H, D, a.top)
    res["outliers_jdp4_vs_j2"] = outliers(rows, D, T, a.top)
    # run-to-run noise inside each config (rep 1 vs rep 0), when present
    noise = {}
    for name, path, base in (("horizon", a.horizon, H), ("jdp4", a.dp4, D), ("j2", a.j2, T)):
        other, _, _ = load_scores(path, 1 - a.rep)
        if all(i in other for i in ids):
            noise[name] = statistics.fmean(per_prompt(ids, base, other)[0])
    res["same_config_rerun_gap"] = noise
    # sanity: the probe's own sampled logprobs (decode path, Jupiter, September) vs today's Jupiter DP4 prefill scores
    if all(r.get("orig_logprobs") for r in rows):
        O = {r["id"]: r["orig_logprobs"] for r in rows}
        res["sanity_orig_sampled_vs_jdp4"] = statistics.fmean(per_prompt(ids, O, D)[0])
    res["think"] = {"horizon": thH, "jdp4": thD, "j2": thT}
    res["meta"] = {"horizon": mH, "jdp4": mD, "j2": mT}
    s = json.dumps(res, indent=1)
    if a.out:
        open(a.out, "w").write(s)
    print(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--trials", required=True, help="eval_sessions/<run> dir holding <trial>/attempts/*/result.json")
    b.add_argument("--out", required=True)
    b.add_argument("--n", type=int, default=96)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--edges", default="4096,16384,32768,49152,62000", help="prompt+completion token bins")
    b.add_argument("--min-completion", type=int, default=16)
    s = sub.add_parser("score")
    s.add_argument("--url", default="http://localhost:8000")
    s.add_argument("--model", default="snowball")
    s.add_argument("--set", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--label", required=True)
    s.add_argument("--conc", type=int, default=4)
    s.add_argument("--reps", type=int, default=2)
    s.add_argument("--think-samples", type=int, default=32, help="0 skips the <|start_think|> probe")
    k = sub.add_parser("think", help="only the <|start_think|> opening probe, against any running server")
    k.add_argument("--url", default="http://localhost:8000")
    k.add_argument("--model", default="snowball")
    k.add_argument("--label", required=True)
    k.add_argument("--out")
    k.add_argument("--think-samples", type=int, default=64)
    c = sub.add_parser("compare")
    c.add_argument("--set", required=True)
    c.add_argument("--horizon", required=True)
    c.add_argument("--dp4", required=True)
    c.add_argument("--j2", required=True, help="the second Jupiter layout (DP2-EP2)")
    c.add_argument("--rep", type=int, default=0)
    c.add_argument("--boot", type=int, default=10000)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--top", type=int, default=8)
    c.add_argument("--out")
    a = ap.parse_args()
    {"build": build, "score": score, "think": think, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    main()
