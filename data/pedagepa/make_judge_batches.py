"""make_judge_batches.py: blinded Opus-judge inputs for a set of harbor trials.

  python make_judge_batches.py --out DIR --prefix M --per-batch 8 --benchmark "<note>" --tree <task tree> \
      --arm <label>:<jobs dir>[:refeed] [--arm ...]
Writes DIR/in/<id>.view.txt, DIR/in/<id>.facts.json, DIR/batch<k>.json (items shuffled across arms) and
DIR/id_map_PRIVATE.json (id -> arm, task, trial). Trials without a reward (unscored) are skipped.
"""
import argparse, glob, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from condense import condense
from facts import facts

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True); ap.add_argument('--prefix', default='M'); ap.add_argument('--per-batch', type=int, default=8)
ap.add_argument('--benchmark', required=True); ap.add_argument('--tree', required=True); ap.add_argument('--arm', action='append', required=True)
ap.add_argument('--limit', type=int, default=65536); ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out + '/in', exist_ok=True)
items = []
for spec in a.arm:
    parts = spec.split(':'); label, jd = parts[0], parts[1]; refeed = len(parts) > 2 and parts[2] == 'refeed'
    for t in sorted(glob.glob(jd + '/*/')):
        t = t.rstrip('/')
        try:
            f = facts(t, refeed=refeed, limit=a.limit)
        except Exception:
            continue
        if f.get('reward') is None:
            continue
        items.append(dict(arm=label, trial=t, task=f.get('task'), refeed=refeed, facts=f))
random.Random(a.seed).shuffle(items)
mp, out = {}, []
for n, it in enumerate(items, 1):
    jid = f'{a.prefix}{n:03d}'
    c = condense(it['trial'], 60000); f = dict(it['facts']); f.pop('trial', None)
    open(f"{a.out}/in/{jid}.view.txt", 'w').write(c['view']); json.dump(f, open(f"{a.out}/in/{jid}.facts.json", 'w'), indent=1)
    mp[jid] = dict(arm=it['arm'], task=it['task'], trial=it['trial'])
    out.append(dict(id=jid, task=it['task'], task_dir=os.path.join(a.tree, it['task'] or ''), view=f"{a.out}/in/{jid}.view.txt",
                    facts=f"{a.out}/in/{jid}.facts.json", trial=it['trial'], benchmark=a.benchmark))
nb = 0
for k in range(0, len(out), a.per_batch):
    json.dump(out[k:k + a.per_batch], open(f'{a.out}/batch{nb}.json', 'w'), indent=1); nb += 1
json.dump(mp, open(a.out + '/id_map_PRIVATE.json', 'w'), indent=1)
print(len(out), 'items in', nb, 'batches')
