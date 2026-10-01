#!/usr/bin/env python3
"""PedaGEPA stage 5: blinded Opus-eval batches for the three arms' run-1 eval trials (rule fixed 2026-09-30, before results).

Task sample: 32 TB2.1 + 32 SWE random-100 tasks, the same for every arm, seeded (20260930), drawn from tasks where every
arm has a scored run-1 trial and at least one arm's trial has >= 5 executed replies. Items are shuffled across arms;
ids are neutral (S001...); id_map_PRIVATE.json maps them back. Judge prompt = judge_prompt_v1_3.md (facts v1.3.2).

    python s5_eval_batches.py --jobs <tb2_jobs dir> --day 20260930 --arms pgc0,pgc1,pgp --out <dir> [--per-batch 8]
"""
import argparse
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from condense import condense  # noqa: E402
from facts import facts  # noqa: E402

SETS = {'tb21': (['tb21'], '/scratch/11584/lukedhlee/tasks/terminal_bench_2_1', 'Terminal-Bench 2.1, Terminus-2'),
        'swe': (['swe_s0', 'swe_s1'], '/scratch/11584/lukedhlee/tasks/swebench_verified_random100',
                'SWE-bench Verified random-100, Terminus-2')}


def trials(jobs, run):
    out = {}
    for t in glob.glob(os.path.join(jobs, run, '*', '')):
        t = t.rstrip('/')
        try:
            f = facts(t)
        except Exception:  # noqa: BLE001
            continue
        if f.get('reward') is None:
            continue
        out[f.get('task')] = (t, f)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--jobs', required=True)
    ap.add_argument('--day', required=True)
    ap.add_argument('--arms', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--per-batch', type=int, default=8)
    ap.add_argument('--n', type=int, default=32)
    a = ap.parse_args()
    arms = a.arms.split(',')
    os.makedirs(os.path.join(a.out, 'in'), exist_ok=True)
    items, rep = [], {}
    for sname, (runs, tree, bench) in SETS.items():
        per_arm = {}
        for arm in arms:
            d = {}
            for r in runs:
                d.update(trials(a.jobs, f'{r}_{arm}_r1_{a.day}'))
            per_arm[arm] = d
        common = sorted(set.intersection(*(set(d) for d in per_arm.values())))
        ok = [t for t in common if max(per_arm[arm][t][1].get('n_executed') or 0 for arm in arms) >= 5]
        rng = random.Random(20260930)
        pick = sorted(rng.sample(ok, min(a.n, len(ok))))
        rep[sname] = dict(common=len(common), eligible=len(ok), picked=len(pick))
        for t in pick:
            for arm in arms:
                trial, f = per_arm[arm][t]
                items.append(dict(arm=arm, task=t, trial=trial, facts=f, tree=tree, bench=bench, set=sname))
    random.Random(20260930).shuffle(items)
    mp, out = {}, []
    for n, it in enumerate(items, 1):
        jid = f'S{n:03d}'
        c = condense(it['trial'], 60000)
        f = dict(it['facts'])
        f.pop('trial', None)
        open(os.path.join(a.out, 'in', f'{jid}.view.txt'), 'w').write(c['view'])
        json.dump(f, open(os.path.join(a.out, 'in', f'{jid}.facts.json'), 'w'), indent=1)
        mp[jid] = dict(arm=it['arm'], task=it['task'], trial=it['trial'], set=it['set'])
        out.append(dict(id=jid, task=it['task'], task_dir=os.path.join(it['tree'], it['task'] or ''),
                        view=os.path.join(a.out, 'in', f'{jid}.view.txt'), facts=os.path.join(a.out, 'in', f'{jid}.facts.json'),
                        trial=it['trial'], benchmark=it['bench']))
    for b in range(0, len(out), a.per_batch):
        json.dump(out[b:b + a.per_batch], open(os.path.join(a.out, f'batch{b // a.per_batch}.json'), 'w'), indent=1)
    json.dump(mp, open(os.path.join(a.out, 'id_map_PRIVATE.json'), 'w'), indent=1)
    rep['items'] = len(out)
    rep['batches'] = (len(out) + a.per_batch - 1) // a.per_batch
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
