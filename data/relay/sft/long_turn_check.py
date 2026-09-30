"""Held-out eval turns with >= 8k completion tokens: is the thinking real reasoning or a repetition loop?
loop score = share of the think text's 50-char windows (stride 25) that occur more than 3 times in it."""
import json, glob, collections, zlib, random
R = '/e/data1/mmlaion/lee27/experiments/relay_pilot_jobs/heldout6516_%s_20260928_student_only'
def q(xs, p): xs = sorted(xs); return xs[int(p * (len(xs) - 1))] if xs else None
ex = {}
for m in ['0921', 'A', 'B', 'C']:
    loops, ratios, ends, n = [], [], collections.Counter(), 0
    for f in glob.glob(R % m + '/*/result.json'):
        r = json.load(open(f)); a = r.get('trial_uri', '').replace('file://', '')
        try: t = json.load(open(a + '/agent/trajectory.json'))
        except Exception: continue
        for s in t['steps']:
            c = (s.get('metrics') or {}).get('completion_tokens') or 0
            if c < 8000: continue
            msg = s['message']; th = msg.split('<|end_think|>')[0] if '<|end_think|>' in msg else msg
            n += 1
            w = [th[i:i + 50] for i in range(0, max(0, len(th) - 50), 25)]
            cnt = collections.Counter(w)
            loops.append(sum(v for v in cnt.values() if v > 3) / max(1, len(w)))
            ratios.append(len(zlib.compress(th.encode())) / max(1, len(th.encode())))
            ends['closed' if '<|end_think|>' in msg else 'no end_think'] += 1
            ex.setdefault(m, []).append((loops[-1], c, th))
    hi = sum(x > 0.5 for x in loops)
    print(f'{m}: {n} turns >= 8k | loop score median {q(loops,.5):.2f}, >0.5 in {hi} ({hi/max(1,n):.0%}) | zlib ratio median {q(ratios,.5):.2f} | {dict(ends)}')
random.seed(0)
for m in ['A', 'B']:
    L = sorted(ex.get(m, []), key=lambda x: -x[0])
    for lab, x in (('most loopy', L[0]), ('median', L[len(L) // 2])):
        print(f'\n--- {m} {lab}: loop {x[0]:.2f}, {x[1]} tokens; think tail:\n{x[2][-700:]}')
