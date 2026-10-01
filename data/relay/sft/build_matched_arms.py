#!/usr/bin/env python3
"""Rows for the 2026-10-01 2x2 (Luke: "how much does relay actually help?"): Horizon relay rows (student = arm A) vs
Qwen3.8-alone rows (the relay pipeline's control arm, same CalibForge tasks, same harness and Qwen settings), matched
per task by count and pass/fail, each with and without the Kimi SWE-smith traces.

Eligibility, identical for both pools (build_hz_arms.py's Horizon rule): census row (hz_pool_census.py, --arm
relay_repair / control) not a weak timeout, the verifier's pytest ran, no leak / hunt / canary tag; a rendered row
(render_think_limit.py, 09-28 settings) that fits 65,536 tokens and has a trained token.

Matching, per task present in both pools (seeded):
  1. the relay part draws at most 2 eligible rows at random, as H4's cap did; its (passes p, failures f) is the target;
  2. the Qwen part must supply the same p and f; where it has fewer of one outcome, both parts drop to what both can
     supply, and a freed slot is refilled with the other outcome when both pools still have one;
  3. both parts then draw their rows of each outcome uniformly.
So mrel and mqwen hold the same (task, outcome) multiset; only who played the early turns differs. Trained tokens
cannot be matched (a relay row trains only Qwen's turns after the handover, a Qwen-alone row trains every turn);
they are reported. mrelk / mqwenk add the Kimi rows of H8 (4,392: the 4,398 minus 6 bottle traces).

    python build_matched_arms.py --relay-census <census.jsonl> --relay-rendered <dir> \
        --qwen-census <census.jsonl> --qwen-rendered <dir> --kimi-from <h8_rows.jsonl> --out-dir <dir>
"""
import argparse
import collections
import glob
import json
import os
import random

SEED = 20261001


def head(line):
    i = line.find('"ids"')
    return json.loads(line[:i].rstrip(', ') + '}')


def trained(line):
    j = line.find('"loss": [')
    return line[j + 9:line.find(']', j)].count('1')


def pool(census, rendered, st):
    cen = {}
    for l in open(census):
        c = json.loads(l)
        cen[c['sid']] = c
    out = {}
    for f in sorted(glob.glob(os.path.join(rendered, '*.rendered.jsonl'))):
        for line in open(f):
            h = head(line)
            c = cen.get(h['sid'])
            st['rendered'] += 1
            if c is None:
                st['no_census'] += 1
                continue
            if c['weak_timeout'] or not c['verifier_ran']:
                st['weak_timeout_or_no_verifier'] += 1
                continue
            if not h['fits']:
                st['over_64k'] += 1
                continue
            if not trained(line):
                st['no_trained_token'] += 1
                continue
            tags = dict(leak=c['leak_turns'] > 0, hunt=c['hunt_turns'] > 0, canary=c['canary'])
            st.update(f'tag_{k}' for k, v in tags.items() if v)
            if any(tags.values()):
                st['dropped_tagged'] += 1
                continue
            out[h['sid']] = dict(sid=h['sid'], task=c['task'], passed=bool(c['passed']), file=f)
    st['eligible'] = len(out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--relay-census', required=True)
    ap.add_argument('--relay-rendered', required=True)
    ap.add_argument('--qwen-census', required=True)
    ap.add_argument('--qwen-rendered', required=True)
    ap.add_argument('--kimi-from', required=True, help='H8 rows: its kimi:* rows are the Kimi part')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--per-task', type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rep = dict(relay=collections.Counter(), qwen=collections.Counter())
    R = pool(a.relay_census, a.relay_rendered, rep['relay'])
    Q = pool(a.qwen_census, a.qwen_rendered, rep['qwen'])
    by = {k: collections.defaultdict(lambda: {True: [], False: []}) for k in ('r', 'q')}
    for k, P in (('r', R), ('q', Q)):
        for r in P.values():
            by[k][r['task']][r['passed']].append(r['sid'])
    rng = random.Random(SEED)
    keep = {'r': [], 'q': []}
    m = collections.Counter()
    for t in sorted(set(by['r']) | set(by['q'])):
        if t not in by['r'] or t not in by['q']:
            m['tasks_in_one_pool_only'] += 1
            continue
        r, q = by['r'][t], by['q'][t]
        for d in (r, q):
            for o in (True, False):
                d[o].sort()
                rng.shuffle(d[o])
        draw = [o for o in (True, False) for _ in r[o]]
        rng.shuffle(draw)
        want = collections.Counter(draw[:a.per_task])          # the relay part's random cap (H4's rule)
        got = {o: min(want[o], len(q[o])) for o in (True, False)}
        for o in (True, False):                                 # refill a slot Qwen could not supply with the other outcome
            free = sum(want.values()) - sum(got.values())
            if free > 0:
                got[o] += max(0, min(free, len(r[o]) - got[o], len(q[o]) - got[o]))
        m['slots_wanted'] += sum(want.values())
        m['slots_kept'] += sum(got.values())
        m['pass_target'] += want[True]
        m['fail_target'] += want[False]
        if not sum(got.values()):
            m['tasks_dropped'] += 1
            continue
        m['tasks'] += 1
        for o in (True, False):
            keep['r'] += r[o][:got[o]]
            keep['q'] += q[o][:got[o]]
            m['pass' if o else 'fail'] += got[o]
    rep['match'] = m
    kimi = []
    for line in open(a.kimi_from):
        if head(line)['sid'].startswith('kimi:'):
            kimi.append(line if line.endswith('\n') else line + '\n')
    rep['kimi_rows'] = len(kimi)
    want = {'mrel': set(keep['r']), 'mqwen': set(keep['q'])}
    src = {'mrel': R, 'mqwen': Q}
    tok = {}
    for arm, sids in want.items():
        lines = []
        for f in sorted({src[arm][s]['file'] for s in sids}):
            for line in open(f):
                if head(line)['sid'] in sids:
                    lines.append(line if line.endswith('\n') else line + '\n')
        assert len(lines) == len(sids), (arm, len(lines), len(sids))
        tok[arm] = dict(rows=len(lines), tokens=sum(head(l)['n_tokens'] for l in lines), trained=sum(trained(l) for l in lines))
        with open(os.path.join(a.out_dir, f'{arm}_rows.jsonl'), 'w') as fo:
            fo.writelines(lines)
        with open(os.path.join(a.out_dir, f'{arm}k_rows.jsonl'), 'w') as fo:
            fo.writelines(lines + kimi)
    rep['tokens'] = tok
    json.dump(rep, open(os.path.join(a.out_dir, 'match.json'), 'w'), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
