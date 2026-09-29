"""How much of each arm's trained signal is long per-turn thinking, and what a lower think cap / a row filter would remove.
Reads the rendered rows' turn metadata only (no tokenizer): a teacher turn's reasoning span is trained iff think_trained
(uncut, <= the 16k cap, not autofixed); its size is reasoning_tokens."""
import json, collections, sys
R = '/e/fscratch/reformo/lee27/experiments/relay/pilot/runs'
ARMS = {'A relay': (f'{R}/relay_full_relaym6_20260928/final_v2A_rendered_think16k_clean_afnone.jsonl', f'{R}/relay_full_relaym6_20260928/final_v2_manifest.jsonl'),
        'B Qwen':  (f'{R}/relay_full_baseline6m_20260926/final_v2_rendered_think16k_clean.jsonl', f'{R}/relay_full_baseline6m_20260926/final_v2_manifest.jsonl')}
def q(xs, p): xs = sorted(xs); return xs[int(p * (len(xs) - 1))] if xs else None
for name, (rows_f, man_f) in ARMS.items():
    passed = {json.loads(l)['sid']: json.loads(l)['passed'] for l in open(man_f) if l.strip()}
    trained_total = 0; tt = []; rows = []
    for line in open(rows_f):
        r = json.loads(line)
        trained_total += sum(r['loss'])
        th = [t['reasoning_tokens'] for t in r['turns'] if t['owner'] == 'teacher' and t.get('think_trained')]
        allq = [t.get('reasoning_tokens') or 0 for t in r['turns'] if t['owner'] == 'teacher']
        tt += th
        rows.append(dict(sid=r['sid'], passed=passed.get(r['sid']), th=th, maxq=max(allq or [0]), nq=len(allq)))
    think_trained = sum(tt)
    print(f'== {name}: {len(rows)} rows, trained tokens {trained_total:,}; trained thinking {think_trained:,} = {think_trained/trained_total:.0%} of trained tokens')
    print(f'   trained think spans: {len(tt)} turns, median {q(tt,.5)}, p90 {q(tt,.9)}, p99 {q(tt,.99)} tokens')
    for cap in (2048, 4096, 8192):
        over = [x for x in tt if x > cap]
        rw = [r for r in rows if any(x > cap for x in r['th'])]
        print(f'   cap {cap:>5}: masks {len(over)} spans ({len(over)/len(tt):.1%}), {sum(over):,} trained tokens ({sum(over)/trained_total:.1%} of all trained), touches {len(rw)} rows'
              f' (pass {sum(bool(r["passed"]) for r in rw)}, fail {sum(r["passed"] is False for r in rw)})')
    for lim in (8192, 16384):
        drop = [r for r in rows if r['maxq'] > lim]
        print(f'   row filter max-turn thinking > {lim}: drops {len(drop)} rows (pass {sum(bool(r["passed"]) for r in drop)}, fail {sum(r["passed"] is False for r in drop)})')
    for lab in (True, False):
        s = [x for r in rows if r['passed'] is lab for x in r['th']]
        print(f'   {"passes" if lab else "failures"}: trained think span median {q(s,.5)}, p90 {q(s,.9)}, share > 4k {sum(x>4096 for x in s)/max(1,len(s)):.1%}')
