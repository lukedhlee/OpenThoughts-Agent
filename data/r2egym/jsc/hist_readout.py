#!/usr/bin/env python3
"""hist_readout.py — read-out of the history-think probes (2026-09-06).

Usage: hist_readout.py --probes keep=snowball_hist_keep_base drop=snowball_hist_drop_base [last2=...]
                       [--splits idval=tt_v2_idval.txt,oodval=tt_v2_oodval.txt,heldout=tt_v2_heldout.txt]
                       [--exclude file] [--out prefix] [--min-scored 8]

One pass over every attempt's result.json (reward, exception, per-turn prompt/completion token lengths from
rollout_details, and task_complete from the last assistant message in agent_result.metadata.all_messages), so no
trajectory.json is opened. Per probe x split: mean per-task pass on fully-sampled tasks, tasks solved at least once,
trial pass, context-death share, median turns, declared-done share, P(win | done), prompt tokens per turn, prompt
tokens at turns 10/20/30. Paired per-task deltas against the first probe with a bootstrap 95 % interval on the
tasks fully sampled by both. Also writes <out>_attempts.jsonl (one record per attempt) for further analysis.
Python 3.9 / stdlib; run on the Jupiter login node with OMP_NUM_THREADS=1 (one process, streaming).
"""
import argparse, collections, glob, json, os, random, statistics, sys, tarfile, time

E = "/e/fscratch/reformo/lee27/experiments"
ap = argparse.ArgumentParser()
ap.add_argument("--probes", nargs="+", required=True, help="label=probe_name ...; the first label is the reference")
ap.add_argument("--splits", default=None, help="name=file,name=file (task-id lists); default = one split 'all'")
ap.add_argument("--exclude", default=None, help="task ids to drop everywhere (one per line, '#' comments)")
ap.add_argument("--out", default=None); ap.add_argument("--min-scored", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

TC_TRUE = '"task_complete": true'
THINK_ID = 128002  # Snowball's think-start token; a completion that starts with it is a thinking turn


def read_attempt(p, data=None):
    try:
        d = json.loads(data) if data is not None else json.load(open(p))
    except Exception:
        return None
    task = d.get("task_name") or p.split("/eval_sessions/")[1].split("/")[1].split("__")[0]
    v = d.get("verifier_result"); r = (v.get("rewards") or {}).get("reward") if isinstance(v, dict) else None
    e = (d.get("exception_info") or {}).get("exception_type")
    ar = d.get("agent_result") or {}
    rd = ar.get("rollout_details") or []
    x = rd[0] if rd else {}
    pl = [len(t) for t in (x.get("prompt_token_ids") or [])]
    ct = x.get("completion_token_ids") or []
    cl = [len(t) for t in ct]
    th = [bool(t) and t[0] == THINK_ID for t in ct]
    md = ar.get("metadata") or {}
    am = md.get("all_messages") or []
    last_asst = next((m for m in reversed(am) if isinstance(m, dict) and m.get("role") == "assistant"), None)
    tc = bool(last_asst and TC_TRUE in (last_asst.get("content") or "").replace(" ", "").replace('"task_complete":true', TC_TRUE))
    n_asst = sum(1 for m in am if isinstance(m, dict) and m.get("role") == "assistant")
    return dict(task=task, reward=r, exc=e, turns=len(pl) or n_asst, prompt_lens=pl, comp_lens=cl, think=th,
                in_tok=ar.get("n_input_tokens"), out_tok=ar.get("n_output_tokens"), tc=tc)


def load_probe(name):
    tj = f"{E}/{name}/{name}/trace_jobs"
    tar = f"{E}/{name}/{name}/trace_archive.tar"
    recs = []
    t0 = time.time()
    if os.path.isdir(f"{tj}/eval_sessions"):
        files = glob.glob(f"{tj}/eval_sessions/*/*/attempts/*/result.json")
        for i, p in enumerate(sorted(files)):
            rec = read_attempt(p)
            if rec: recs.append(rec)
            if (i + 1) % 500 == 0: sys.stderr.write(f"  {name}: {i + 1}/{len(files)} ({time.time() - t0:.0f}s)\n")
        sys.stderr.write(f"{name}: {len(recs)} attempts read from {len(files)} result files\n")
    elif os.path.isfile(tar):
        # probe_watch archived the tree (tar of trace_jobs, one inode): stream the result.json members
        n = 0
        with tarfile.open(tar, "r|") as tf:
            for m in tf:
                if m.isfile() and m.name.endswith("/result.json") and "/eval_sessions/" in m.name and "/attempts/" in m.name:
                    n += 1
                    rec = read_attempt(m.name, data=tf.extractfile(m).read())
                    if rec: recs.append(rec)
                    if n % 500 == 0: sys.stderr.write(f"  {name}: {n} from tar ({time.time() - t0:.0f}s)\n")
        sys.stderr.write(f"{name}: {len(recs)} attempts read from {n} result files in {tar}\n")
    else:
        sys.stderr.write(f"{name}: no trace_jobs tree and no trace_archive.tar\n")
    return recs


def per_task(recs, min_scored):
    by = collections.defaultdict(list)
    for r in recs: by[r["task"]].append(r)
    out = {}
    for t, at in by.items():
        scored = [r for r in at if r["reward"] is not None]
        if len(scored) < min_scored: continue
        out[t] = sum(1 for r in scored if r["reward"] >= 1.0) / len(scored)
    return out


def summarize(recs, tasks):
    at = [r for r in recs if r["task"] in tasks]
    scored = [r for r in at if r["reward"] is not None]
    wins = [r for r in scored if r["reward"] >= 1.0]
    done = [r for r in scored if r["tc"]]
    ctx = sum(1 for r in at if r["exc"] == "ContextLengthExceededError")
    turns = [r["turns"] for r in at if r["turns"]]
    ptt = [sum(r["prompt_lens"]) / len(r["prompt_lens"]) for r in at if r["prompt_lens"]]
    def p_at(k):
        v = [r["prompt_lens"][k - 1] for r in at if len(r["prompt_lens"]) >= k]
        return statistics.median(v) if v else None
    death_turns = [r["turns"] for r in at if r["exc"] == "ContextLengthExceededError" and r["turns"]]
    th_all = [f for r in at for f in r["think"]]
    def th_at(k):
        v = [r["think"][k - 1] for r in at if len(r["think"]) >= k]
        return (sum(v) / len(v)) if v else None
    comp_all = [c for r in at for c in r["comp_lens"]]
    return dict(attempts=len(at), scored=len(scored), nulls=len(at) - len(scored),
                trial_pass=len(wins) / max(1, len(scored)),
                ctx_death=ctx / max(1, len(at)), turns_med=statistics.median(turns) if turns else None,
                turns_at_death_med=statistics.median(death_turns) if death_turns else None,
                done=len(done) / max(1, len(scored)),
                p_win_given_done=(sum(1 for r in done if r["reward"] >= 1.0) / len(done)) if done else None,
                wins_declared=(sum(1 for r in wins if r["tc"]) / len(wins)) if wins else None,
                prompt_tok_per_turn=statistics.mean(ptt) if ptt else None,
                prompt_t10=p_at(10), prompt_t20=p_at(20), prompt_t30=p_at(30),
                think=(sum(th_all) / len(th_all)) if th_all else None, think_t10=th_at(10), think_t20=th_at(20),
                comp_med=statistics.median(comp_all) if comp_all else None,
                out_tok_med=statistics.median([r["out_tok"] for r in at if r["out_tok"]]) if any(r["out_tok"] for r in at) else None)


def boot(pairs, n=4000, seed=0):
    rng = random.Random(seed); ds = [b - x for x, b in pairs]
    if not ds: return None, None, None
    bs = sorted(sum(ds[rng.randrange(len(ds))] for _ in ds) / len(ds) for _ in range(n))
    return sum(ds) / len(ds), bs[int(.025 * n)], bs[int(.975 * n)]


def fmt(v, d=3):
    if v is None: return "-"
    if isinstance(v, float): return f"{v:.{d}f}"
    return str(v)


probes = []
for arg in a.probes:
    label, name = arg.split("=", 1); probes.append((label, name, load_probe(name)))
exclude = set()
if a.exclude:
    exclude = set(l.split("#")[0].strip() for l in open(a.exclude) if l.split("#")[0].strip())
splits = {}
if a.splits:
    for kv in a.splits.split(","):
        k, f = kv.split("=", 1); splits[k] = set(l.strip() for l in open(f) if l.strip()) - exclude
all_tasks = set(r["task"] for _, _, recs in probes for r in recs) - exclude
splits["all"] = all_tasks

if a.out:
    with open(a.out + "_attempts.jsonl", "w") as fh:
        for label, name, recs in probes:
            for r in recs:
                rr = dict(r); rr["probe"] = label; rr.pop("prompt_lens", None); rr.pop("comp_lens", None); th = rr.pop("think", [])
                rr["think_share"] = (sum(th) / len(th)) if th else None
                rr["prompt_tok_per_turn"] = (sum(r["prompt_lens"]) / len(r["prompt_lens"])) if r["prompt_lens"] else None
                fh.write(json.dumps(rr) + "\n")

def curves(recs, tasks, max_t=40):
    """Per turn index: attempts reaching it, think share, median prompt and completion tokens (for the report's charts)."""
    out = []
    at = [r for r in recs if r["task"] in tasks]
    for t in range(1, max_t + 1):
        pl = [r["prompt_lens"][t - 1] for r in at if len(r["prompt_lens"]) >= t]
        cl = [r["comp_lens"][t - 1] for r in at if len(r["comp_lens"]) >= t]
        th = [r["think"][t - 1] for r in at if len(r["think"]) >= t]
        if len(pl) < 5: break
        out.append(dict(t=t, n=len(pl), think=(sum(th) / len(th)) if th else None, prompt_med=statistics.median(pl),
                        comp_med=statistics.median(cl) if cl else None))
    return out


curve_out = {}
lines = []
for sname, stasks in splits.items():
    curve_out[sname] = {label: curves(recs, stasks) for label, _, recs in probes}
    lines.append(f"\n## split {sname} ({len(stasks)} tasks)")
    lines.append("| probe | full tasks | mean per-task pass | solved>=1 | trial pass | ctx death | turns med | turns@death | done | P(win|done) | wins declared | prompt tok/turn | prompt@t10 | @t20 | @t30 | think turns | think@t10 | think@t20 | comp tok med | nulls |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    pt = {}
    for label, name, recs in probes:
        pt[label] = {t: v for t, v in per_task(recs, a.min_scored).items() if t in stasks}
        s = summarize(recs, stasks)
        full = pt[label]
        mean_pass = (sum(full.values()) / len(full)) if full else None
        solved = sum(1 for v in full.values() if v > 0)
        lines.append(f"| {label} | {len(full)} | {fmt(mean_pass)} | {solved} | {fmt(s['trial_pass'])} | {fmt(s['ctx_death'])} | {fmt(s['turns_med'], 0)} | {fmt(s['turns_at_death_med'], 0)} | {fmt(s['done'])} | {fmt(s['p_win_given_done'])} | {fmt(s['wins_declared'])} | {fmt(s['prompt_tok_per_turn'], 0)} | {fmt(s['prompt_t10'], 0)} | {fmt(s['prompt_t20'], 0)} | {fmt(s['prompt_t30'], 0)} | {fmt(s['think'], 2)} | {fmt(s['think_t10'], 2)} | {fmt(s['think_t20'], 2)} | {fmt(s['comp_med'], 0)} | {s['nulls']} |")
    ref = probes[0][0]
    for label, _, _ in probes[1:]:
        both = sorted(set(pt[ref]) & set(pt[label]))
        m, lo, hi = boot([(pt[ref][t], pt[label][t]) for t in both], seed=a.seed)
        new = sum(1 for t in both if pt[ref][t] == 0 and pt[label][t] > 0); lost = sum(1 for t in both if pt[ref][t] > 0 and pt[label][t] == 0)
        lines.append(f"paired {label} - {ref}: {fmt(m)} [{fmt(lo)}, {fmt(hi)}] on n={len(both)}; newly solved {new}, newly lost {lost}")
text = "\n".join(lines)
print(text)
if a.out:
    json.dump(curve_out, open(a.out + "_curves.json", "w"))
    open(a.out + "_readout.md", "w").write(text + "\n")
    sys.stderr.write(f"wrote {a.out}_readout.md and {a.out}_attempts.jsonl\n")
