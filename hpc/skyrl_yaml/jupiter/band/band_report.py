#!/usr/bin/env python3
"""Gate ladder + learnable-band measurement for one RL/probe run's trace_jobs.

Walks the ladder IN ORDER, because every rung above has produced a false
positive at some point in this project:
  1. non-null reward       -- `scored=N` counts FILES, not rewards
  2. trajectory >1 step    -- a one-turn no-op scores f(task) with zero variance
  3. trial duration in min -- a 62-min trial is a timeout, not work
  4. only then, a band number

BAND = fraction of fully-sampled groups with 0 < passes < n_samples. That, not
the raw pass rate, is what GRPO can learn from: a group where every sample
agrees contributes exactly zero advantage.

usage: band.py <trace_jobs_dir> [window_minutes]   (window 0 = all time)
"""
import json, glob, os, sys, time, collections, statistics as st

BASE = sys.argv[1].rstrip('/')
WINDOW = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
cut = (time.time() - WINDOW * 60) if WINDOW else 0.0

rew = collections.Counter()
exc = collections.Counter()
steps, durs = [], []
groups = collections.defaultdict(list)   # task_name -> [(reward, nsteps)]
n = 0

for f in glob.glob(BASE + '/*/result.json'):
    if os.path.getmtime(f) < cut:
        continue
    n += 1
    try:
        d = json.load(open(f))
    except Exception:
        continue

    vr = d.get('verifier_result') or {}
    r = (vr.get('rewards') or {}).get('reward') if isinstance(vr, dict) else None
    rew[r] += 1

    ei = d.get('exception_info') or {}
    exc[(ei.get('exception_type') if isinstance(ei, dict) else None) or 'none'] += 1

    ns = None
    tj = os.path.join(os.path.dirname(f), 'agent', 'trajectory.json')
    if os.path.isfile(tj):
        try:
            ns = len(json.load(open(tj)).get('steps') or [])
            steps.append(ns)
        except Exception:
            pass

    s, e = d.get('started_at'), d.get('finished_at')
    if s and e:
        try:
            from datetime import datetime
            fmt = lambda x: datetime.fromisoformat(x.replace('Z', '+00:00'))
            durs.append((fmt(e) - fmt(s)).total_seconds() / 60.0)
        except Exception:
            pass

    groups[d.get('task_name') or '?'].append((r, ns))

print('=== trace_jobs: %s' % BASE)
print('=== window: %s' % ('all time' if not WINDOW else '%g min' % WINDOW))
print('results             : %d   (files, NOT rewards)' % n)
print('rewards             : %s' % dict(rew))
print('exceptions          : %s' % dict(exc))

# --- rung 1 -------------------------------------------------------------
scored = sum(v for k, v in rew.items() if k is not None)
print('\n[1] non-null reward : %d/%d -> %s' % (
    scored, n, 'PASS' if scored else 'FAIL (all rewards null)'))

# --- rung 2 -------------------------------------------------------------
if steps:
    med = st.median(steps)
    print('[2] trajectory steps: n=%d median=%s max=%d -> %s' % (
        len(steps), med, max(steps),
        'PASS (multi-turn)' if med > 1 else 'FAIL (one-turn no-op)'))
else:
    print('[2] trajectory steps: none yet -> PENDING')

# --- rung 3 -------------------------------------------------------------
if durs:
    durs.sort()
    p90 = durs[min(len(durs) - 1, int(0.9 * len(durs)))]
    print('[3] trial duration  : n=%d median=%.1f min p90=%.1f max=%.1f -> %s' % (
        len(durs), st.median(durs), p90, max(durs),
        'PASS (minutes, real work)' if st.median(durs) < 30 else 'SUSPECT (timeout-scale)'))
else:
    print('[3] trial duration  : none yet -> PENDING')

# --- rung 4: the band ---------------------------------------------------
NS = 4
full = {t: v for t, v in groups.items() if len(v) >= NS}
print('\n[4] groups: %d seen, %d fully sampled (>=%d trials)' % (len(groups), len(full), NS))
if full:
    band = solved = unsolved = freepass = 0
    for t, v in full.items():
        ks = [x for x, _ in v if x is not None]
        if len(ks) < NS:
            continue
        k = sum(1 for x in ks if x and x > 0)
        if k == 0:
            unsolved += 1
        elif k == len(ks):
            solved += 1
            sv = [s for _, s in v if s is not None]
            if sv and st.median(sv) <= 2:
                freepass += 1
        else:
            band += 1
    tot = band + solved + unsolved
    if tot:
        print('    always-solved   : %2d (%4.1f%%)   of which %d suspected FREE-PASS (median steps<=2)'
              % (solved, 100.0 * solved / tot, freepass))
        print('    never-solved    : %2d (%4.1f%%)' % (unsolved, 100.0 * unsolved / tot))
        print('    IN BAND (0<k<%d): %2d (%4.1f%%)  <-- the only groups that yield gradient'
              % (NS, band, 100.0 * band / tot))
        print('    BAND GATE       : %s' % ('PASS (>=1 group varies)' if band else 'FAIL (zero variance)'))
        eff = tot - freepass
        if freepass and eff:
            print('    band excl. free-pass: %4.1f%% of %d effective groups'
                  % (100.0 * band / eff, eff))
    print('    trial pass rate : %.1f%% (%d/%d scored trials)' % (
        100.0 * sum(1 for k, c in rew.items() if k and k > 0 for _ in range(c)) / max(scored, 1),
        sum(c for k, c in rew.items() if k and k > 0), scored))

# --- concurrency --------------------------------------------------------
ev = []
for f in glob.glob(BASE + '/*/result.json'):
    if os.path.getmtime(f) < cut:
        continue
    try:
        d = json.load(open(f))
        from datetime import datetime
        s, e = d.get('started_at'), d.get('finished_at')
        if s and e:
            fmt = lambda x: datetime.fromisoformat(x.replace('Z', '+00:00'))
            ev.append((fmt(s), 1)); ev.append((fmt(e), -1))
    except Exception:
        pass
if ev:
    ev.sort()
    cur = peak = 0
    for _, d_ in ev:
        cur += d_
        peak = max(peak, cur)
    span = (ev[-1][0] - ev[0][0]).total_seconds() / 60.0
    print('\nPEAK MEASURED CONCURRENCY = %d   (completed trials only)' % peak)
    if span > 0:
        rate = (len(ev) // 2) / span
        print('completed %.2f trials/min over %.0f min span' % (rate, span))
        print('full band = 13312 trials; 10h => 22.2 trials/min needed')
        if rate > 0:
            print('=> at THIS rate the full band takes %.1f h' % (13312 / rate / 60))
