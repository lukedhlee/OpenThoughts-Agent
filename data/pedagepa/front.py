"""front.py: Pareto front of teacher prompt candidates from blinded Opus judgments (judge-in-the-loop prompt search).

  python front.py --control control=<judge dir> --cand guided_v2=<judge dir> [--cand ...] \
      [--items P0,P5a,P5b,P9] [--tasks <task list>] [--judge-sub opus1] [--out front.json]
Each <judge dir> is a make_judge_batches.py output (id_map_PRIVATE.json, in/*.facts.json, <judge-sub>/batch*.jsonl);
the label before '=' is the arm label inside that dir, optionally `arm@name` to show another name. Every candidate is paired with the control per task (tasks
judged in both; --tasks restricts to a list). Per candidate: each item's paired mean difference (NA/LD dropped) with a
task-bootstrap 95 % CI, the composite (mean of the numeric --items per run) difference, and the pass difference.
Front: a candidate is dropped when another is >= on every item and on pass, and > on one (point estimates).
"""
import argparse, collections, glob, json, random, re, statistics as st

ap = argparse.ArgumentParser()
ap.add_argument('--control', required=True); ap.add_argument('--cand', action='append', required=True)
ap.add_argument('--items', default='P0,P5a,P5b,P9'); ap.add_argument('--tasks'); ap.add_argument('--judge-sub', default='opus1')
ap.add_argument('--all-items', default='P0,P1,P2,P3,P4,P5a,P5b,P6,P7,P8,P9,P10,T1'); ap.add_argument('--out')
x = ap.parse_args(); items = x.items.split(','); all_items = x.all_items.split(',')
keep = set(open(x.tasks).read().split()) if x.tasks else None


def load(spec):
    label, d = spec.split('=', 1)
    label, name = label.split('@', 1) if '@' in label else (label, label)
    mp = json.load(open(d + '/id_map_PRIVATE.json')); J = {}
    for f in glob.glob(f'{d}/{x.judge_sub}/batch*.jsonl'):
        for l in open(f):
            if l.strip():
                o = json.loads(l); J[o['id']] = o
    runs = {}
    for jid, m in mp.items():
        if m['arm'] != label or jid not in J or (keep and m['task'] not in keep):
            continue
        sc = {}
        for it in all_items:
            s = (J[jid].get('scores') or {}).get(it)
            v = str(s.get('score') if isinstance(s, dict) else s)
            if re.fullmatch(r'[0-2](\.0)?', v):
                sc[it] = float(v)
        fct = json.load(open(f'{d}/in/{jid}.facts.json'))
        runs[m['task']] = dict(scores=sc, passed=1.0 if (fct.get('reward') or 0) >= 1 else 0.0)
    return name, runs


rnd = random.Random(0)
def ci(d):
    if len(d) < 3:
        return None
    s = sorted(st.mean(rnd.choices(d, k=len(d))) for _ in range(5000)); return [round(s[125], 3), round(s[4875], 3)]


def comp(r):
    v = [r['scores'][i] for i in items if i in r['scores']]; return st.mean(v) if v else None


_, ctl = load(x.control)
rows = []
for spec in x.cand:
    label, runs = load(spec)
    common = sorted(set(ctl) & set(runs)); row = dict(cand=label, n_tasks=len(common), items={})
    for it in all_items:
        d = [runs[t]['scores'][it] - ctl[t]['scores'][it] for t in common if it in runs[t]['scores'] and it in ctl[t]['scores']]
        row['items'][it] = dict(n=len(d), diff=round(st.mean(d), 3) if d else None, ci=ci(d))
    d = [comp(runs[t]) - comp(ctl[t]) for t in common if comp(runs[t]) is not None and comp(ctl[t]) is not None]
    row['composite'] = dict(n=len(d), diff=round(st.mean(d), 3) if d else None, ci=ci(d))
    d = [runs[t]['passed'] - ctl[t]['passed'] for t in common]
    row['pass'] = dict(n=len(d), diff=round(st.mean(d), 3) if d else None, ci=ci(d))
    rows.append(row)


def vec(r):
    return [r['items'][i]['diff'] if r['items'][i]['diff'] is not None else float('-inf') for i in items] + [r['pass']['diff'] or 0.0]


for r in rows:
    v = vec(r)
    r['on_front'] = not any(all(a >= b for a, b in zip(vec(o), v)) and any(a > b for a, b in zip(vec(o), v)) for o in rows if o is not r)
print(f"{'candidate':<16}{'tasks':>6}{'composite':>22}{'pass':>22}  " + '  '.join(f'{i:>6}' for i in all_items) + '  front')
for r in rows:
    c, p = r['composite'], r['pass']
    print(f"{r['cand']:<16}{r['n_tasks']:>6}{str(c['diff']) + ' ' + str(c['ci']):>22}{str(p['diff']) + ' ' + str(p['ci']):>22}  "
          + '  '.join(f"{r['items'][i]['diff'] if r['items'][i]['diff'] is not None else '-':>6}" for i in all_items)
          + ('  *' if r['on_front'] else ''))
if x.out:
    json.dump(rows, open(x.out, 'w'), indent=1)
