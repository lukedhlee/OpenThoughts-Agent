"""arm_verdict.py: compare two arms of a PedaGEPA teacher run on the judged composite, pass rate, leaks and length.

  python arm_verdict.py --judge-dir DIR --a control --b guided --items P0,P5a,P5b,P9 --run-dir <relay run dir>
Composite per run = mean of the numeric scores of --items (NA/LD dropped); paired per task (b - a) with a 10,000-draw
task bootstrap. Pass: paired b - a over tasks scored in both. Leaks and tokens from the router logs of arm b and a.
"""
import argparse, glob, json, random, re, statistics as st, collections
ap = argparse.ArgumentParser(); ap.add_argument('--judge-dir', required=True); ap.add_argument('--a', required=True)
ap.add_argument('--b', required=True); ap.add_argument('--items', required=True); ap.add_argument('--run-dir', required=True)
ap.add_argument('--judge-sub', default='opus1')
x = ap.parse_args(); items = x.items.split(',')
mp = json.load(open(x.judge_dir + '/id_map_PRIVATE.json'))
J = {}
for f in glob.glob(f'{x.judge_dir}/{x.judge_sub}/batch*.jsonl'):
    for l in open(f):
        if l.strip():
            o = json.loads(l); J[o['id']] = o
comp = collections.defaultdict(dict); reward = collections.defaultdict(dict); per_item = collections.defaultdict(lambda: collections.defaultdict(list))
for jid, m in mp.items():
    o = J.get(jid)
    if not o:
        continue
    v = []
    for it in items:
        s = (o.get('scores') or {}).get(it)
        sc = str(s.get('score') if isinstance(s, dict) else s)
        if re.fullmatch(r'[0-2](\.0)?', sc):
            v.append(float(sc)); per_item[m['arm']][it].append(float(sc))
    if v:
        comp[m['task']][m['arm']] = st.mean(v)
try:
    import json as _j
    for jid, m in mp.items():
        fct = _j.load(open(f"{x.judge_dir}/in/{jid}.facts.json")); reward[m['task']][m['arm']] = fct.get('reward')
except Exception:
    pass
rnd = random.Random(0)
def boot(d):
    if len(d) < 3: return (None, None)
    s = sorted(st.mean(rnd.choices(d, k=len(d))) for _ in range(10000)); return (round(s[250], 3), round(s[9750], 3))
d = [v[x.b] - v[x.a] for v in comp.values() if x.a in v and x.b in v]
pd = [(1 if (v[x.b] or 0) > 0 else 0) - (1 if (v[x.a] or 0) > 0 else 0) for v in reward.values() if v.get(x.a) is not None and v.get(x.b) is not None]
out = dict(items=items, n_paired=len(d), composite={x.a: round(st.mean(v[x.a] for v in comp.values() if x.a in v), 3), x.b: round(st.mean(v[x.b] for v in comp.values() if x.b in v), 3)},
           composite_diff=round(st.mean(d), 3) if d else None, composite_ci=boot(d),
           per_item={arm: {it: (round(st.mean(v), 2), len(v)) for it, v in per_item[arm].items()} for arm in (x.a, x.b)},
           pass_n=len(pd), pass_diff=round(st.mean(pd), 3) if pd else None, pass_ci=boot([float(p) for p in pd]))
for arm in (x.a, x.b):
    n = leak = passed = 0; comp_tok = []
    try:
        for l in open(f'{x.run_dir}/router_{arm}/turns.jsonl'):
            r = json.loads(l); n += 1; leak += bool(r.get('leak_hits')); passed += bool(r.get('leak_passed'))
            comp_tok.append((r.get('usage') or {}).get('completion_tokens') or 0)
    except FileNotFoundError:
        pass
    out[f'router_{arm}'] = dict(requests=n, leak_resampled=leak, leak_passed=passed, leak_passed_share=round(passed / n, 4) if n else None,
                                completion_tokens_median=st.median(comp_tok) if comp_tok else None)
ta, tb = out[f'router_{x.a}']['completion_tokens_median'], out[f'router_{x.b}']['completion_tokens_median']
out['token_ratio'] = round(tb / ta, 3) if ta and tb else None
print(json.dumps(out, indent=1))
