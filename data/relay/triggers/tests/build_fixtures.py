#!/usr/bin/env python3
"""Build tests/fixtures.jsonl.gz from real transcripts (reproducible; needs the calibration data + scan cache).

Positives: real fires of each calibrated trigger (preferring fires the Opus judges rated real_mistake).
Negatives: requests where the 2026-09-25 proposal's broader definition fires but the calibrated one does not, taken
from PASSING episodes. Each fixture is a trimmed message window (task + the last few turns) that is checked to give
the same verdict as the full episode before it is kept. TB2 transcripts are excluded (benchmark text stays out of
the repo): sources are SWE-bench Verified / R2E-Gym student runs and public or our own teacher traces.
Also writes a few whole episodes for the causality test.
"""
import gzip
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import calibrate as C       # noqa: E402
import relay_triggers as rt  # noqa: E402

WINDOW = 9     # turns of history kept before the fire


def window(msgs, t, keep_reply):
    """First user message + the assistant/observation messages of turns max(1, t-WINDOW) .. t-1 (+ reply t)."""
    first_user = next(i for i, m in enumerate(msgs) if m.get('role') == 'user')
    turn, out, reply = 0, [msgs[first_user]], None
    lo = max(1, t - WINDOW)
    for i, m in enumerate(msgs):
        if i == first_user:
            continue
        if m.get('role') == 'assistant':
            turn += 1
            if turn == t:
                reply = m
                break
        if lo <= turn < t and m.get('role') != 'system':
            out.append({k: v for k, v in m.items() if k in ('role', 'content', 'tool_calls', 'orig_role')})
    return out, (reply if keep_reply else None)


def verdict(msgs, reply, trig, cfg):
    fires = rt.detect(msgs, reply=reply.get('content') if reply and not reply.get('tool_calls') else
                      (rt.tool_reply(reply) if reply else None), config=dict(cfg, enabled=[trig]))
    return [f for f in fires if f['trigger'] == trig]


def main():
    rng = random.Random(0)
    eps = [e for e in C.load_eps() if e['group'] in ('S-SWE', 'S-R2E', 'T-TLEGO', 'T-TB2LIKE', 'T-SWE-MINI')]
    rng.shuffle(eps)
    cal = rt.CALIBRATED_CONFIG
    judged = {}
    jdir = os.path.join(C.OUT, 'judge')
    for fn in os.listdir(jdir) if os.path.isdir(jdir) else []:
        if fn.endswith('.verdicts.jsonl'):
            name = fn.split('.')[0]
            meta = {json.loads(l)['jid']: json.loads(l) for l in open(os.path.join(jdir, name + '.jsonl'))}
            for l in open(os.path.join(jdir, fn)):
                v = json.loads(l)
                m = meta.get(v['jid'])
                if m:
                    judged[(m['id'], name, m['turn'])] = v['verdict']
    fixtures = []
    trigs = ['gave_up', 'no_progress_wait', 'loop', 'edit_failed', 'error_streak', 'submit_check', 'done_claim']
    for trig in trigs:
        env = trig in C.ENV_TRIGS
        pos, neg = [], []
        for e in eps:
            if len(pos) >= 3 and len(neg) >= 3:
                break
            ft = C.fire_turns(e, cal, trig) if trig != 'done_claim' else ([e['t_done']] if e['t_done'] else [])
            want_pos = ft and len(pos) < 3 and (e['outcome'] == 'fail' or trig == 'done_claim')
            if want_pos and judged.get((e['id'], trig, ft[0])) == 'not_mistake':
                want_pos = False
            fp = [] if trig == 'done_claim' else C.fire_turns(e, dict(C.BASE, loop_modes=['exact', 'same_output', 'varied'],
                                                                   wait_k=1, streak_n=2, edit_k=1,
                                                                   giveup_families=list(rt.GIVEUP_FAMILIES),
                                                                   submit_flags=['no_edit', 'stub_marker', 'stub_pass',
                                                                                 'tests_rm', 'tests_skip']), trig)
            want_neg = (e['outcome'] == 'pass' and fp and not ft and len(neg) < 3)
            if not (want_pos or want_neg):
                continue
            msgs = C.load_messages(e)
            t = ft[0] if want_pos else fp[0]
            hist, reply = window(msgs, t, keep_reply=not env or trig == 'submit_check')
            if trig == 'submit_check':
                hist, reply = window(msgs, t, keep_reply=True)
                hist = [m for m in msgs[:msgs.index(reply)] if m.get('role') != 'system']   # needs the whole history
            got = verdict(hist, reply, trig, cal)
            if bool(got) != bool(want_pos):
                continue   # the trimmed window changes the verdict: skip this example
            fx = dict(name='%s_%s_%d' % (trig, 'pos' if want_pos else 'neg', len(pos) if want_pos else len(neg)),
                      trigger=trig, expect_fire=bool(want_pos), source=e['source'], group=e['group'],
                      outcome=e['outcome'], turn=t, messages=hist, reply=reply,
                      note=(got[0]['reason'] if got else 'broad definition fired here; calibrated one does not'))
            (pos if want_pos else neg).append(fx)
        fixtures += pos + neg
        print(trig, 'pos', len(pos), 'neg', len(neg))
    with gzip.open(os.path.join(HERE, 'fixtures.jsonl.gz'), 'wt') as f:
        for fx in fixtures:
            f.write(json.dumps(fx) + '\n')
    # whole episodes for the causality test: short-ish ones with several different fires
    whole = []
    for e in eps:
        if len(whole) >= 4:
            break
        if e['group'] not in ('S-R2E', 'S-SWE', 'T-SWE-MINI') or not (8 <= e['turns'] <= 45):
            continue
        cfg = dict(cal, enabled=trigs + ['destructive', 'success_contradicted'])
        fs = rt.fires_by_turn(e['sc'], cfg)
        if len({f['trigger'] for f in fs}) >= 3 or (e['group'] == 'T-SWE-MINI' and len(whole) == 3):
            msgs = [{k: v for k, v in m.items() if k in ('role', 'content', 'tool_calls', 'orig_role')}
                    for m in C.load_messages(e)]
            whole.append(dict(id=e['id'], group=e['group'], messages=msgs))
    with gzip.open(os.path.join(HERE, 'episodes.jsonl.gz'), 'wt') as f:
        for w in whole:
            f.write(json.dumps(w) + '\n')
    print('fixtures', len(fixtures), 'episodes', len(whole))


if __name__ == '__main__':
    main()
