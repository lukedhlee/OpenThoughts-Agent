"""Token-budget decomposition of terminus-2 trajectories for one eval session.

usage: budget.py <eval_session_dir> <context_limit>
Per-attempt records are cached in <session>_budget.json next to the script.
"""
import json, glob, os, sys, re, statistics, collections, time

M = "/e/fscratch/reformo/lee27/models/snowball-s3-nemotron-terminal-step1888"
SESS = sys.argv[1].rstrip("/"); LIMIT = int(sys.argv[2]); OUT = int(sys.argv[3]); RESERVE = 2048
CACHE = os.path.basename(SESS) + "_budget.json"
ST, EN = 128002, 128003
CAT = [("test", r"^(python3? -m pytest|pytest|python3? -m unittest|tox|nosetests|make test|python3? -c .*(assert|import))"),
       ("edit", r"(sed -i|cat > |cat >> |tee |apply_patch|patch -p|python3? - ?<<|cat <<|printf .* > |echo .* > |>> ?[A-Za-z0-9_./-]+\.py|mv |cp )"),
       ("run_python", r"^python3? (-c|[A-Za-z0-9_./-]+\.py)"),
       ("explore", r"^(ls|cat |head|tail|grep|rg|find|sed -n|wc|tree|pwd|cd |which|git (log|status|diff|show|grep)|less|more|awk|nl |stat )"),
       ("install", r"^(pip|uv |apt|conda|npm)"), ("git", r"^git ")]
WARN_RE = re.compile(r"^(Previous response had (?:warnings|parsing errors):.*?)(?:New Terminal Output:|Current Terminal Screen:|$)", re.S)


def classify(k):
    k = k.strip()
    if not k:
        return "empty"
    for c, pat in CAT:
        if re.search(pat, k):
            return c
    return "other"


def obs_content(o):
    if isinstance(o, dict):
        try:
            return o["results"][0]["content"]
        except Exception:
            return json.dumps(o)
    if isinstance(o, str):
        try:
            return json.loads(o)["results"][0]["content"]
        except Exception:
            return o
    return json.dumps(o)


def q(a, f):
    a = sorted(a)
    return a[min(len(a) - 1, int(f * len(a)))] if a else float("nan")


def pq(a):
    return f"p50={q(a, .5):.0f} p90={q(a, .9):.0f} (n={len(a)})"


def pqp(a):
    return f"p50={q(a, .5):.0%} p90={q(a, .9):.0%} (n={len(a)})"


def build():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(M)
    tj = json.load(open(M + "/tokenizer.json"))
    special = {a["id"] for a in tj["added_tokens"] if a.get("special")}
    print("skip_special decode check:", repr(tok.decode([128002, 198, 128002, 198, 9906, 128003, 198, 90], skip_special_tokens=True)))
    A, T = [], []
    t0 = time.time(); n_bad = 0
    for p in glob.glob(f"{SESS}/*/attempts/*/agent/trajectory.json"):
        att = os.path.dirname(os.path.dirname(p))
        task = os.path.basename(att.split("/attempts/")[0]).split("__")[0]
        try:
            st = json.load(open(p))["steps"]
        except Exception:
            n_bad += 1; continue
        ag = [s for s in st if s.get("source") == "agent" and (s.get("metrics") or {}).get("prompt_tokens")]
        if not ag:
            n_bad += 1; continue
        rw = None; rp = att + "/verifier/reward.txt"
        if os.path.exists(rp):
            try:
                rw = float(open(rp).read().strip())
            except Exception:
                rw = None
        exc = None; exc_tokens = None; ep = att + "/exception.txt"
        if os.path.exists(ep):
            txt = open(ep).read()
            lines = [l for l in txt.splitlines() if "Error" in l or "Exception" in l]
            if lines:
                mm = re.findall(r"([A-Za-z]+(?:Error|Exception))", lines[-1]); exc = mm[-1] if mm else lines[-1][:40]
            mm = re.search(r"(\d+) tokens used", txt); exc_tokens = int(mm.group(1)) if mm else None
        pt = [s["metrics"]["prompt_tokens"] for s in ag]
        fixed = pt[0]
        prev_cmd = None; seen = set(); first_edit = first_test = None; test_after_edit = False; declared = False
        sums = collections.Counter(); overhead = []; last_obs = 0; last_c = 0
        for i, s in enumerate(ag):
            m = s["metrics"]; ids = m.get("completion_token_ids") or []
            ct = m.get("completion_tokens") or len(ids)
            if ST in ids:
                a = ids.index(ST); b = ids.index(EN) if EN in ids else len(ids) - 1
                r_ids = ids[a:b + 1]; j_ids = ids[:a] + ids[b + 1:]
            else:
                r_ids = []; j_ids = ids
            r_ref = sum(1 for x in r_ids if x not in special); j_ref = sum(1 for x in j_ids if x not in special)
            c_ref = r_ref + j_ref
            content = obs_content(s.get("observation"))
            rejected = content.startswith("Previous response had parsing errors")
            wm = WARN_RE.match(content)
            wtok = len(tok.encode(wm.group(1), add_special_tokens=False)) if wm else 0
            otok = len(tok.encode(content, add_special_tokens=False))
            if i + 1 < len(ag) and pt[i + 1] - pt[i] - c_ref >= 0 and len(sys.argv) < 5:
                obs_fed = pt[i + 1] - pt[i] - c_ref; overhead.append(obs_fed - otok)
            elif i + 1 < len(ag):
                obs_fed = otok + 10  # summarised history: prompt delta is meaningless, use re-tokenized content
            else:
                obs_fed = otok + 6; last_obs = obs_fed; last_c = c_ref
            msg = s.get("message", ""); msg = msg if isinstance(msg, str) else ""
            cats = set(); cmd_key = None
            try:
                a_, b_ = msg.index("{"), msg.rindex("}"); j = json.loads(msg[a_:b_ + 1])
                if j.get("task_complete") is True:
                    declared = True
                cmds = [c.get("keystrokes", "") for c in (j.get("commands") or []) if isinstance(c, dict)]
                cmd_key = "\x00".join(k.strip() for k in cmds) if cmds else None
                for k in cmds:
                    cats.add(classify(k))
            except Exception:
                pass
            if cmd_key:
                if cmd_key == prev_cmd:
                    sums["rep_prev"] += 1
                if cmd_key in seen:
                    sums["rep_any"] += 1
                seen.add(cmd_key); prev_cmd = cmd_key
            if "edit" in cats and first_edit is None:
                first_edit = i + 1
            if "test" in cats:
                if first_test is None:
                    first_test = i + 1
                if first_edit is not None:
                    test_after_edit = True
            sums["reas"] += r_ref; sums["js"] += j_ref; sums["obs"] += obs_fed; sums["comp"] += ct; sums["warn"] += wtok
            if obs_fed > 2000:
                sums["obs_big"] += obs_fed
            if rejected:
                sums["n_rej"] += 1; sums["rej_comp"] += ct
            first_cmd = (cmd_key or "").split("\x00")[0][:40]
            T.append(dict(r=r_ref, j=j_ref, o=obs_fed, ct=ct, rej=rejected, w=wtok, turn=i + 1, exc=exc, cat="+".join(sorted(cats)) or "none",
                          first_cmd=first_cmd, dbl=bool(ids[:1] == [ST] and ST in ids[1:4]), noend=bool(ST in ids and EN not in ids), cap=ct >= OUT,
                          nst=ids.count(ST), nen=ids.count(EN)))
        window = pt[-1] + last_c + last_obs  # size of the prompt the harness would have sent next
        A.append(dict(task=task, fixed=fixed, turns=len(ag), reward=rw, exc=exc, exc_tokens=exc_tokens, window=window,
                      last_prompt=pt[-1], first_edit=first_edit, first_test=first_test, test_after_edit=test_after_edit,
                      declared=declared, overhead=statistics.median(overhead) if overhead else None, **sums))
    print(f"built in {time.time() - t0:.0f}s, skipped={n_bad}")
    json.dump(dict(A=A, T=T), open(CACHE, "w"))
    return A, T


if os.path.exists(CACHE):
    d = json.load(open(CACHE)); A, T = d["A"], d["T"]
else:
    A, T = build()

n = len(A)
for a in A:
    for k in ("reas", "js", "obs", "comp", "warn", "obs_big", "n_rej", "rej_comp", "rep_prev", "rep_any"):
        a.setdefault(k, 0)
print(f"session={os.path.basename(SESS)} limit={LIMIT} usable={LIMIT - RESERVE} attempts={n} turns={len(T)}")
exc_c = collections.Counter(a["exc"] for a in A)
print("outcomes:", {k: f"{v} ({v / n:.1%})" for k, v in exc_c.most_common()})
print("reward dist:", dict(collections.Counter(a["reward"] for a in A)))
ctx = [a for a in A if a["exc"] == "ContextLengthExceededError"]
clean = [a for a in A if a["exc"] is None]
succ = [a for a in clean if a["reward"] == 1.0]
print(f"context-death {len(ctx)} ({len(ctx) / n:.1%}); clean {len(clean)} ({len(clean) / n:.1%}); clean&reward1 {len(succ)} ({len(succ) / n:.1%}); reward1 total {sum(1 for a in A if a['reward'] == 1.0)}")
ov = [a["overhead"] for a in A if a["overhead"] is not None]
print(f"template overhead per turn (prompt delta - retokenized obs - refed completion): {pq(ov)}")
if ctx:
    err = [a["window"] - a["exc_tokens"] for a in ctx if a["exc_tokens"]]
    print(f"death-window estimate minus harness count: {pq(err)} | harness count at death: {pq([a['exc_tokens'] for a in ctx if a['exc_tokens']])}")

print("\n=== 1. FIXED COST (first prompt, tokens) ===")
bt = collections.defaultdict(list)
for a in A:
    bt[a["task"]].append(a["fixed"])
ft = [statistics.median(v) for v in bt.values()]
print(f"per task: {pq(ft)} min={min(ft):.0f} max={max(ft):.0f}; share of {LIMIT}: {pqp([f / LIMIT for f in ft])}; share of usable {LIMIT - RESERVE}: {pqp([f / (LIMIT - RESERVE) for f in ft])}")

print("\n=== 2. PER-TURN COST (tokens) ===")
print(f"(a) reasoning (re-fed, markers stripped): {pq([t['r'] for t in T])}")
print(f"(b) JSON / non-reasoning completion:       {pq([t['j'] for t in T])}")
print(f"(c) observation fed back (incl. harness warning prefix + template): {pq([t['o'] for t in T])}")
print(f"whole completion as generated: {pq([t['ct'] for t in T])}; reasoning share of completion per turn: {pqp([t['r'] / t['ct'] for t in T if t['ct']])}; pooled {sum(t['r'] for t in T) / sum(t['ct'] for t in T):.0%}")
print(f"per-turn total added to window (a+b+c): {pq([t['r'] + t['j'] + t['o'] for t in T])}")
by_turn = collections.defaultdict(list)
for t in T:
    by_turn[min(t["turn"], 13)].append(t)
print("by turn index (p50 r / j / o):", " | ".join(f"t{k}{'+' if k == 13 else ''}: {q([t['r'] for t in v], .5):.0f}/{q([t['j'] for t in v], .5):.0f}/{q([t['o'] for t in v], .5):.0f}" for k, v in sorted(by_turn.items())))
print(f"turns per attempt: all {pq([a['turns'] for a in A])}; context-death {pq([a['turns'] for a in ctx])}")
if ctx:
    print("--- at context death: window composition (median of per-attempt shares | pooled) ---")
    def sh(k):
        return statistics.median(a[k] / a["window"] for a in ctx), sum(a[k] for a in ctx) / sum(a["window"] for a in ctx)
    for k, lab in [("fixed", "fixed prompt"), ("reas", "re-fed reasoning"), ("js", "re-fed JSON"), ("obs", "observations")]:
        m_, p_ = sh(k); print(f"  {lab:18s} {m_:5.0%} | {p_:5.0%}   abs {pq([a[k] for a in ctx])}")
    resid = [(a["window"] - a["fixed"] - a["reas"] - a["js"] - a["obs"]) for a in ctx]
    print(f"  residual (template): {pq(resid)}; window at death: {pq([a['window'] for a in ctx])}")
    print(f"  completion tokens generated by death: {pq([a['comp'] for a in ctx])}; model-generated share of window (reas+json): {pqp([(a['reas'] + a['js']) / a['window'] for a in ctx])}")

print("\n=== 3. TURNS NEEDED vs AVAILABLE ===")
for lab, grp in [("clean & reward=1.0", succ), ("clean & reward<1", [a for a in clean if a["reward"] != 1.0]), ("reward=1.0 any outcome", [a for a in A if a["reward"] == 1.0])]:
    if grp:
        print(f"{lab}: n={len(grp)} turns {pq([a['turns'] for a in grp])}; final window tokens {pq([a['window'] for a in grp])}; completion tokens {pq([a['comp'] for a in grp])}; first edit turn {pq([a['first_edit'] for a in grp if a['first_edit']])}; declared task_complete {sum(a['declared'] for a in grp) / len(grp):.0%}; exc {dict(collections.Counter(a['exc'] for a in grp))}")
if ctx:
    e = sum(1 for a in ctx if a["first_edit"]); t_ = sum(1 for a in ctx if a["first_test"]); both = sum(1 for a in ctx if a["test_after_edit"])
    print(f"context-death: n={len(ctx)} edited before death {e / len(ctx):.0%}; ran a test before death {t_ / len(ctx):.0%}; test after an edit {both / len(ctx):.0%}; neither {sum(1 for a in ctx if not a['first_edit'] and not a['first_test']) / len(ctx):.0%}; first edit turn {pq([a['first_edit'] for a in ctx if a['first_edit']])}; declared task_complete {sum(a['declared'] for a in ctx) / len(ctx):.0%}")
for lab, grp in [("clean & reward=1.0", succ)]:
    print(f"{lab}: total tokens processed if never summarized (fixed+reas+json+obs): {pq([a['fixed'] + a['reas'] + a['js'] + a['obs'] for a in grp])}; reasoning {pq([a['reas'] for a in grp])}; JSON {pq([a['js'] for a in grp])}; obs {pq([a['obs'] for a in grp])}")
print(f"all attempts: never edited {sum(1 for a in A if not a['first_edit']) / n:.0%}; first edit turn {pq([a['first_edit'] for a in A if a['first_edit']])}")

print("\n=== 4. WASTE ITEMS ===")
tot_o = sum(t["o"] for t in T); tot_c = sum(t["ct"] for t in T)
print(f"observation tokens in dumps >2k tokens: {sum(t['o'] for t in T if t['o'] > 2000) / tot_o:.0%} of observation tokens ({sum(1 for t in T if t['o'] > 2000) / len(T):.1%} of turns); >4k: {sum(t['o'] for t in T if t['o'] > 4000) / tot_o:.0%} ({sum(1 for t in T if t['o'] > 4000) / len(T):.1%} of turns); per-attempt share {pqp([a['obs_big'] / a['obs'] for a in A if a['obs']])}")
print(f"harness warning/error boilerplate: {sum(t['w'] for t in T) / tot_o:.1%} of observation tokens; turns with a warning block {sum(1 for t in T if t['w']) / len(T):.0%}; warning block size {pq([t['w'] for t in T if t['w']])}")
print(f"rejected (unparseable) turns: {sum(1 for t in T if t['rej']) / len(T):.1%} of turns; their completion tokens {sum(t['ct'] for t in T if t['rej']) / tot_c:.1%} of completion tokens; attempts with >=1 rejected turn {sum(1 for a in A if a['n_rej']) / n:.0%}")
print(f"turns repeating the exact previous command batch: {sum(a['rep_prev'] for a in A) / len(T):.1%}; repeating any earlier batch in the attempt: {sum(a['rep_any'] for a in A) / len(T):.1%}")
big = [t for t in T if t["o"] > 2000]
cc = collections.Counter(); ct_ = collections.Counter()
for t in big:
    cc[t["cat"]] += t["o"]; ct_[t["first_cmd"]] += t["o"]
print("dumps >2k by command category (share of dump tokens):", ", ".join(f"{k}={v / sum(cc.values()):.0%}" for k, v in cc.most_common(7)))
print("dumps >2k top first-commands:", [(k, f"{v / sum(ct_.values()):.1%}") for k, v in ct_.most_common(12)])
print(f"completions at the output cap ({OUT}): {sum(1 for t in T if t['cap']) / len(T):.2%} of turns; completions with <|start_think|> but no <|end_think|>: {sum(1 for t in T if t['noend']) / len(T):.2%}; completions starting with a doubled <|start_think|>: {sum(1 for t in T if t['dbl']) / len(T):.1%}; n(<|start_think|>) per completion {dict(collections.Counter(t['nst'] for t in T).most_common(4))}; n(<|end_think|>) {dict(collections.Counter(t['nen'] for t in T).most_common(4))}")
print(f"rejected turns by turn index: " + ", ".join(f"t{k}={sum(1 for t in v if t['rej']) / len(v):.0%}" for k, v in sorted(by_turn.items())))
if ctx:
    print(f"window share at death (median): reasoning {statistics.median(a['reas'] / a['window'] for a in ctx):.0%}, obs dumps>2k {statistics.median(a['obs_big'] / a['window'] for a in ctx):.0%} (pooled {sum(a['obs_big'] for a in ctx) / sum(a['window'] for a in ctx):.0%}), warnings {statistics.median(a['warn'] / a['window'] for a in ctx):.1%}, rejected-turn completions {sum(a['rej_comp'] for a in ctx) / sum(a['window'] for a in ctx):.1%} (pooled)")
