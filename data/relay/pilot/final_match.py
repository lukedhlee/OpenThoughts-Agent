#!/usr/bin/env python3
"""Final matched SFT arms (coordinator 2026-09-27): relay and Qwen-alone baseline trimmed to exactly N passes + N real
failures each, every row within 09-21's 65,536 tokens.

    python final_match.py --relay <relaym run dir> --baseline <baseline6m run dir> --quarantine quarantine_tasks.txt

  1. Relay: the kept set (kept_manifest.jsonl, 979 + 979) loses its rows over 65,536 rendered tokens; lost passes are
     replaced from the relay's unkept passes (kept_manifest_all.jsonl minus the kept ones, within 65,536, seeded
     sample). Failures cannot be replaced. The relay's failures are also held to the baseline's strict definition
     (match_kept: quarantined tasks and weak timeouts dropped), so N = the relay failures that remain.
  2. Both arms are trimmed to N + N with match_kept.py (seed 20260927, --weak-timeouts drop): the relay from step 1's
     pool, the baseline from kept_manifest_strict.jsonl.
  3. Per run dir: final_manifest.jsonl, final_rendered.jsonl (the rows of rendered.jsonl for the final sessions; a
     row is rendered per episode, independent of the manifest, so the subset is exact), final_match.json (stats).
"""
import argparse
import collections
import json
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import match_kept  # noqa: E402

MAX_TOKENS = 65536
SEED = 20260927


def read(p):
    return [json.loads(line) for line in open(p) if line.strip()]


def write(p, rows):
    with open(p, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


def rendered_stats(path, sids):
    """{sid: row stats} for the rendered rows of these sessions (one streaming pass)."""
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r['sid'] in sids:
            out[r['sid']] = dict(n=r['n_tokens'], trained=sum(r['loss']),
                                 qwen_turns=sum(1 for t in r['turns'] if t['owner'] == 'teacher'))
    return out


def subset_rendered(src, dst, sids):
    n = 0
    with open(dst, 'w') as f:
        for line in open(src):
            if json.loads(line)['sid'] in sids:
                f.write(line if line.endswith('\n') else line + '\n')
                n += 1
    return n


def summary(rows, st):
    n = sorted(st[r['sid']]['n'] for r in rows)
    q = lambda f: n[int(f * (len(n) - 1))]  # noqa: E731
    fails = [r for r in rows if not r['passed']]
    return dict(rows=len(rows), passes=len(rows) - len(fails), failures=len(fails),
                failure_mix=dict(collections.Counter(r['cause'] for r in fails)),
                trained_tokens=sum(st[r['sid']]['trained'] for r in rows),
                trained_per_row=round(sum(st[r['sid']]['trained'] for r in rows) / len(rows)),
                row_tokens_p50=q(.5), row_tokens_p90=q(.9), row_tokens_max=n[-1],
                rows_over_64k=sum(1 for x in n if x > MAX_TOKENS),
                qwen_turns_per_row=round(sum(st[r['sid']]['qwen_turns'] for r in rows) / len(rows), 2))


def match(manifest, n, quarantine, out):
    res = subprocess.run([sys.executable, os.path.join(HERE, 'match_kept.py'), '--manifest', manifest, '--quarantine',
                          quarantine, '--passes', str(n), '--failures', str(n), '--weak-timeouts', 'drop', '--seed',
                          str(SEED), '--out', out], capture_output=True, text=True, check=True)
    return json.loads(res.stdout)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--relay', required=True)
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--quarantine', required=True)
    a = ap.parse_args()
    quarantine = {t.strip() for t in open(a.quarantine) if t.strip()}
    rng = random.Random(SEED)

    # 1. relay pool
    kept = read(os.path.join(a.relay, 'kept_manifest.jsonl'))
    cand = read(os.path.join(a.relay, 'kept_manifest_all.jsonl'))
    kept_sids = {r['sid'] for r in kept}
    spare = [r for r in cand if r['passed'] and r['sid'] not in kept_sids and r['task'] not in quarantine]
    st = rendered_stats(os.path.join(a.relay, 'rendered.jsonl'), kept_sids | {r['sid'] for r in spare})
    fits = lambda r: r['sid'] in st and st[r['sid']]['n'] <= MAX_TOKENS  # noqa: E731
    over = [r for r in kept if not fits(r)]
    kp = [r for r in kept if r['passed'] and fits(r)]
    kf = [r for r in kept if not r['passed'] and fits(r)]
    lost_p = sum(1 for r in over if r['passed'])
    spare_ok = sorted((r for r in spare if fits(r)), key=lambda r: r['trial'])
    repl = rng.sample(spare_ok, min(lost_p, len(spare_ok)))
    pool = kp + [dict(r, replaced_over_64k=True) for r in repl] + kf
    pool_p = os.path.join(a.relay, 'final_pool.jsonl')
    write(pool_p, pool)
    strict_f = [r for r in kf if r['task'] not in quarantine and not match_kept.weak_timeout(r)]
    n = len(strict_f)
    relay_step1 = dict(kept=len(kept), over_64k=len(over), over_64k_passes=lost_p, over_64k_failures=len(over) - lost_p,
                       replacement_passes=len(repl), spare_passes_within_64k=len(spare_ok), failures_within_64k=len(kf),
                       failures_quarantined=sum(1 for r in kf if r['task'] in quarantine),
                       failures_weak_timeout=sum(1 for r in kf if r['task'] not in quarantine and match_kept.weak_timeout(r)),
                       N=n, N_if_weak_timeouts_kept=sum(1 for r in kf if r['task'] not in quarantine))

    # 2. trim both arms to N + N
    out = dict(seed=SEED, max_tokens=MAX_TOKENS, N=n, relay_step1=relay_step1)
    for arm, run, manifest in (('relay', a.relay, pool_p),
                               ('baseline', a.baseline, os.path.join(a.baseline, 'kept_manifest_strict.jsonl'))):
        fm = os.path.join(run, 'final_manifest.jsonl')
        mk = match(manifest, n, a.quarantine, fm)
        rows = read(fm)
        sids = {r['sid'] for r in rows}
        rs = rendered_stats(os.path.join(run, 'rendered.jsonl'), sids)
        missing = sids - set(rs)
        if missing:
            raise SystemExit(f'{arm}: {len(missing)} final sessions have no rendered row')
        written = subset_rendered(os.path.join(run, 'rendered.jsonl'), os.path.join(run, 'final_rendered.jsonl'), sids)
        out[arm] = dict(summary(rows, rs), match_kept=dict((k, mk[k]) for k in ('pass_pool', 'failure_pool',
                                                                               'weak_timeout_pool', 'dropped_quarantine')),
                        rendered_rows_written=written, run_dir=run)
    for arm, run in (('relay', a.relay), ('baseline', a.baseline)):
        json.dump(out, open(os.path.join(run, 'final_match.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
