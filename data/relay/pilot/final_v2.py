#!/usr/bin/env python3
"""Final SFT arms v2 (coordinator 2026-09-27, after relay attempt 6): the relay and Qwen-alone baseline arms built with
ONE rule, applied identically, so they differ only in who played the early turns.

    python final_v2.py --relay <relay merged run dir, attempts 1-6> --baseline <baseline6m run dir> \
        --quarantine quarantine_tasks.txt [--seed 20260927]

Per arm (candidates = select_kept.arm_rows: scored episodes; the relay's teacher-wrote-a-turn and S5 filters):
  1. strict labels: quarantined tasks dropped, weak timeouts dropped (match_kept.weak_timeout), rows over 65,536
     rendered tokens dropped (rendered.jsonl of the run dir, the d0ddd237 render);
  2. at most 2 rows per task: a task with more eligible rows keeps 2 drawn uniformly (seeded; the relay's old pool has
     up to 7 tries per task, attempt 6 and the baseline have 2);
Both arms:
  3. the same task set: tasks with at least one eligible row in BOTH arms;
  4. N = the largest N both arms can meet = min(passes, failures) over the two arms on that set; each arm keeps N
     passes and N failures drawn uniformly (seeded).
Writes per run dir final_v2_manifest.jsonl (select_kept rows) and final_v2.json (the same stats in both dirs); the old
final_* files are untouched. Render afterwards with sft/render_think_limit.py --strip-copied-markers --think-limit
16384 --render-dir <d0ddd237>/data/relay/sft --out final_v2_rendered_think16k_clean.jsonl.
Read-only on everything else; login-node safe (stdlib + the pilot modules).
"""
import argparse
import collections
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import match_kept  # noqa: E402
import select_kept  # noqa: E402

MAX_TOKENS = 65536


def row_tokens(path, sids):
    out = {}
    for line in open(path):
        i = line.find('"ids"')
        h = json.loads(line[:i].rstrip(', ') + '}') if i > 0 else json.loads(line)
        if h['sid'] in sids:
            out[h['sid']] = h['n_tokens']
    return out


def eligible(run, arm, quarantine, rng, per_task=2):
    name = os.path.basename(os.path.normpath(run))
    rows = select_kept.arm_rows(run, name, arm)
    n_tok = row_tokens(os.path.join(run, 'rendered.jsonl'), {r['sid'] for r in rows})
    drop = collections.Counter()
    ok = []
    for r in rows:
        if r['task'] in quarantine:
            drop['quarantine'] += 1
        elif match_kept.weak_timeout(r):
            drop['weak_timeout'] += 1
        elif r['sid'] not in n_tok:
            drop['not_rendered'] += 1
        elif n_tok[r['sid']] > MAX_TOKENS:
            drop['over_64k'] += 1
        else:
            ok.append(dict(r, n_tokens=n_tok[r['sid']]))
    by_task = collections.defaultdict(list)
    for r in sorted(ok, key=lambda r: r['trial']):
        by_task[r['task']].append(r)
    capped = []
    for t in sorted(by_task):
        v = by_task[t]
        if len(v) > per_task:
            drop['over_2_per_task'] += len(v) - per_task
            v = rng.sample(v, per_task)
        capped += v
    return rows, capped, drop


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--relay', required=True)
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--quarantine', required=True)
    ap.add_argument('--seed', type=int, default=20260927)
    ap.add_argument('--dry-run', action='store_true', help='print the stats, write nothing')
    a = ap.parse_args()
    quarantine = {t.strip() for t in open(a.quarantine) if t.strip()}
    rng = random.Random(a.seed)
    arms = {'relay': (a.relay, 'relay_repair'), 'baseline': (a.baseline, 'control')}
    pools, out = {}, dict(seed=a.seed, max_tokens=MAX_TOKENS, per_task=2)
    for k, (run, arm) in arms.items():
        rows, capped, drop = eligible(run, arm, quarantine, rng)
        pools[k] = capped
        out[k + '_candidates'] = dict(candidates=len(rows), dropped=dict(drop), eligible=len(capped),
                                      tasks=len({r['task'] for r in capped}))
    common = {r['task'] for r in pools['relay']} & {r['task'] for r in pools['baseline']}
    for k in pools:
        pools[k] = [r for r in pools[k] if r['task'] in common]
    n = min(min(sum(1 for r in v if r['passed']), sum(1 for r in v if not r['passed'])) for v in pools.values())
    out.update(common_tasks=len(common), N=n)
    for k, (run, arm) in arms.items():
        v = pools[k]
        p = sorted((r for r in v if r['passed']), key=lambda r: r['trial'])
        f = sorted((r for r in v if not r['passed']), key=lambda r: r['trial'])
        final = rng.sample(p, n) + rng.sample(f, n)
        pools[k] = final
        if not a.dry_run:
            with open(os.path.join(run, 'final_v2_manifest.jsonl'), 'w') as fh:
                for r in final:
                    fh.write(json.dumps(r) + '\n')
        per_task = collections.Counter(r['task'] for r in final)
        out[k] = dict(run_dir=run, rows=len(final), passes=n, failures=n, pool_passes=len(p), pool_failures=len(f),
                      unique_tasks=len(per_task), rows_per_task=dict(sorted(collections.Counter(per_task.values()).items())),
                      failure_mix=dict(collections.Counter(r['cause'] for r in final if not r['passed'])),
                      row_tokens_max=max(r['n_tokens'] for r in final))
    tr, tb = ({r['task'] for r in pools[k]} for k in ('relay', 'baseline'))
    out['task_overlap'] = dict(both=len(tr & tb), relay_only=len(tr - tb), baseline_only=len(tb - tr),
                               jaccard=round(len(tr & tb) / len(tr | tb), 4))
    for run, _ in arms.values():
        if not a.dry_run:
            json.dump(out, open(os.path.join(run, 'final_v2.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
