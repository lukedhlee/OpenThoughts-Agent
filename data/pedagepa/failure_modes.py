"""failure_modes.py: the student's failure-mode distribution per benchmark, from the Opus judgments.

For each checklist item: occurrence = share of judged student runs where it scored 0; decisive = share whose
failure_cause names it; timing = median position of the first cited reply as a fraction of the run's replies.
"""
import json, glob, re, sqlite3, statistics as st, collections
X = '/e/data1/mmlaion/lee27/experiments/pedagepa'
con = sqlite3.connect(X + '/evaldb/pedagepa.sqlite'); con.row_factory = sqlite3.Row
n_rep = {r['trial_dir']: (r['n_replies'], r['benchmark'], r['model_id'], r['reward']) for r in con.execute('SELECT trial_dir, n_replies, benchmark, model_id, reward FROM trials')}
recs = []   # (benchmark, model, reward, n_replies, judge object)
for bat, outs in ((X + '/inspect/cf/judge/batch*.json', X + '/inspect/cf/judge/opusB/batch*.jsonl'), (X + '/inspect/r2/batch*.json', X + '/inspect/r2/opus1/batch*.jsonl')):
    where = {it['id']: it['trial'] for b in glob.glob(bat) for it in json.load(open(b))}
    for f in glob.glob(outs):
        for l in open(f):
            o = json.loads(l); tr = where.get(o['id'])
            if tr in n_rep:
                n, bench, model, rew = n_rep[tr]; recs.append((bench, model, rew, n, o))
ITEMS = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P5a', 'P5b', 'P6', 'P7', 'P8', 'P9', 'P10', 'T1', 'S1', 'S2', 'S3', 'S4']
out = {}
for bench in sorted({r[0] for r in recs}):
    S = [r for r in recs if r[0] == bench and r[1] == 'snowball-0921']
    tab = {}
    for it in ITEMS:
        zero, cited, pos = 0, 0, []
        scored = 0
        for _, _, rew, n, o in S:
            s = (o.get('scores') or {}).get(it)
            if not s:
                continue
            sc = str(s.get('score') if isinstance(s, dict) else s)
            if re.fullmatch(r'[0-2](\.0)?', sc):
                scored += 1
                if sc.startswith('0'):
                    zero += 1
                    rp = [x for x in (s.get('replies') or []) if isinstance(x, int)]
                    if rp and n:
                        pos.append(min(rp) / n)
            fc = (o.get('failure_cause') or '')
            if re.search(r'\b' + re.escape(it) + r'\b', fc):
                cited += 1
        if scored:
            tab[it] = dict(runs=len(S), scored=scored, zero_share=round(zero / scored, 2), decisive_share=round(cited / len(S), 2),
                           timing_median=round(st.median(pos), 2) if pos else None)
    out[bench] = tab
json.dump(out, open(X + '/m1/failure_modes.json', 'w'), indent=1)
for b, t in out.items():
    print(f'\n== {b} (student runs judged: {next(iter(t.values()))["runs"]})')
    for it, x in sorted(t.items(), key=lambda kv: -kv[1]['zero_share']):
        print(f"  {it:4} zero {x['zero_share']:.2f} (of {x['scored']})  named in failure cause {x['decisive_share']:.2f}  first zero at {x['timing_median']}")
