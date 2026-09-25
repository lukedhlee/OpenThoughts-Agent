#!/usr/bin/env python3
"""Calibrate the relay-rollout triggers on labelled transcripts and reproduce the result tables.

    python calibrate.py scan      # parse + scan every transcript once (multiprocess), cache compact scanners
    python calibrate.py split     # fixed 60/40 task split (written once, before any tuning)
    python calibrate.py tune      # grid search per trigger on the TRAIN split -> tuned_config.json
    python calibrate.py report    # held-out tables with task-bootstrap 95% CIs -> results/*.md|json
    python calibrate.py judge-sample / judge-score   # firing-point precision via LLM judges

Paths: DATA (transcripts, scratch) and OUT (results) default to the 2026-09-25 study; override with env
RELAY_DATA / RELAY_OUT. Everything is CPU only.

Definitions (fixed before tuning; see the README in OUT):
  t_done   first turn whose reply sets task_complete (mini-swe: the submit command); t_stop = t_done, or turns+1
  fire     a trigger firing at turn t, 2 <= t < t_stop (the teacher would write turn t)
  FF       share of PASSING episodes with a fire
  hit      share of FAILED non-infra episodes with a fire while still recoverable:
           reader label if one exists (t <= point_of_no_return and teacher_could_recover != no), else the proxy
           (t <= 0.8 * turns, and elapsed(t) <= 0.8 * agent budget when timestamps exist)
  submit_check is judged AT t_done (it is a label on the done-claim, not an earlier takeover).
"""
import hashlib
import json
import os
import pickle
import random
import re
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import adapters as ad            # noqa: E402
import relay_triggers as rt      # noqa: E402

DATA = os.environ.get('RELAY_DATA', '/private/tmp/claude-503/-Users-lukedhlee-OpenThoughts-Agent/'
                      '5368fd8c-b9cb-4b09-8f5d-7153da0124c4/scratchpad/data')
OUT = os.environ.get('RELAY_OUT', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))),
                     'ai_memory/active/snowball-sft/research/2026-09-25_trigger_calibration'))
FORENSICS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))),
                         'ai_memory/active/snowball-sft/research/2026-09-25_relay_trigger_forensics')
CACHE = os.path.join(DATA, 'cache')
SPLIT_SALT = 'relay-trigger-split-2026-09-25'
FF_CAP = 0.05
TRIGGERS = ['edit_failed', 'loop', 'gave_up', 'no_progress_wait', 'error_streak', 'success_claim_unchecked',
            'success_contradicted', 'destructive', 'submit_check']
GROUPS = [('S-TB2', 'student', 'tb2'), ('S-SWE', 'student', 'swe'), ('S-R2E', 'student', 'r2egym'),
          ('T-TLEGO', 'teacher', 'tlego'), ('T-TB2LIKE', 'teacher', 'tb2like'), ('T-SWE-MINI', 'teacher', 'swe')]
STUDENT_GROUPS = ['S-TB2', 'S-SWE', 'S-R2E']


def group_of(e):
    for g, pop, bench in GROUPS:
        if e['population'] == pop and e['bench'] == bench:
            return g
    return None


# ==== 1. load + scan ========================================================================================
def descriptors():
    """Yield lightweight descriptors; workers load and scan them."""
    # forensics 756 (step-30 student + base), with Opus reader labels
    fdir = os.path.join(DATA, 'forensics756')
    for row in json.load(open(os.path.join(fdir, 'index.json'))):
        yield dict(kind='atif', path=os.path.join(fdir, row['trial'], 'attempts/000/agent/trajectory.json'),
                   meta=dict(id='forensics:' + row['trial'], source='forensics756', population='student',
                             bench=row['bench'], model=row['model'], run=row['run'], task=row.get('task') or os.path.basename(row['trial']).rsplit('__', 1)[0],
                             reward=row['reward'], exc=row['exc'] or '', trial=row['trial']))
    seen_runs = {'basex2_v01x2_20260921', 'basex2swe_v01x2_20260921'} | {
        'rld517ut%d%s_v01_20260922' % (i, s) for i in (1, 2, 3) for s in ('', 'swe')}
    for sub in ('snowball_evals', 'r2egym'):
        man = os.path.join(DATA, sub, 'manifest.jsonl')
        if not os.path.exists(man):
            continue
        for line in open(man):
            d = json.loads(line)
            if d.get('run') in seen_runs or d.get('format', 'atif') != 'atif':
                continue
            meta = {k: d.get(k) for k in ('source', 'population', 'bench', 'model', 'run', 'task', 'reward', 'exc')}
            meta['id'] = '%s:%s/%s' % (sub, d.get('run'), d.get('trial_id'))
            meta['exc'] = meta['exc'] or ''
            yield dict(kind='atif', path=d['path'], meta=meta)
    tdir = os.path.join(DATA, 'teacher')
    for fn in sorted(os.listdir(tdir)) if os.path.isdir(tdir) else []:
        if not fn.endswith('.jsonl'):
            continue
        off = 0
        with open(os.path.join(tdir, fn), 'rb') as f:
            for line in f:
                yield dict(kind='jsonl', path=os.path.join(tdir, fn), offset=off)
                off += len(line)


def _scan_one(desc):
    try:
        if desc['kind'] == 'atif':
            rec = ad.load_atif(desc['path'], desc['meta'])
            if rec is None:
                return None
        else:
            with open(desc['path'], 'rb') as f:
                f.seek(desc['offset'])
                d = json.loads(f.readline())
            if d.get('reward') is None and d.get('exc') == 'AgentTimeoutError':
                d['reward'] = 0.0      # teacher timeouts without a verifier run: count as failures
            d.pop('meta', None)
            rec = ad.load_messages_record(d)
        harness = 'terminus2' if rec.get('harness') == 'terminus2' else None
        sc = rt.scan_messages(rec['messages'], harness=harness, timestamps=rec.get('timestamps'))
        rec.pop('messages', None)
        rec.pop('timestamps', None)
        rec['desc'] = {k: v for k, v in desc.items() if k != 'meta'}
        rec['sc'] = sc.compact()
        return rec
    except Exception as ex:      # keep going; report the count
        return dict(error=repr(ex)[:200], desc=str(desc)[:200])


def cmd_scan():
    os.makedirs(CACHE, exist_ok=True)
    descs = list(descriptors())
    print('descriptors', len(descs))
    eps, errs = [], Counter()
    with Pool(12) as pool:
        for i, r in enumerate(pool.imap_unordered(_scan_one, descs, chunksize=16)):
            if r is None:
                errs['unreadable'] += 1
            elif 'error' in r:
                errs[r['error'][:80]] += 1
            else:
                eps.append(r)
            if i % 2000 == 0:
                print(i, flush=True)
    print('scanned', len(eps), 'errors', dict(errs.most_common(8)))
    pickle.dump(eps, open(os.path.join(CACHE, 'episodes.pkl'), 'wb'))


def load_eps():
    eps = pickle.load(open(os.path.join(CACHE, 'episodes.pkl'), 'rb'))
    readers = {}
    for fn in os.listdir(os.path.join(FORENSICS, 'readers')):
        if fn.endswith('.jsonl'):
            for line in open(os.path.join(FORENSICS, 'readers', fn)):
                r = json.loads(line)
                readers['forensics:' + r['trial']] = r
    out = []
    for e in eps:
        e['group'] = group_of(e)
        if e['group'] is None or e['outcome'] == 'infra':
            continue
        sc = e['sc']
        e['turns'] = sc.turn
        e['t_done'] = sc.done_turn
        e['t_stop'] = sc.done_turn if sc.done_turn else sc.turn + 1
        e['reader'] = readers.get(e['id'])
        e['split'] = split_of(task_key(e))
        out.append(e)
    return out


# ==== 2. split ==============================================================================================
def task_key(e):
    fam = {'tlego': 'tb2like', 'tb2like': 'tb2like'}.get(e['bench'], e['bench'])
    task = str(e.get('task'))
    if task.endswith('-pctl'):          # R2E-Gym dev120 bare-prompt runs: same task, other prompt
        task = task[:-5]
    return '%s|%s' % (fam, task)


def split_of(k):
    h = int(hashlib.sha1((SPLIT_SALT + k).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return 'train' if h < 0.6 else 'heldout'


def cmd_split():
    eps = pickle.load(open(os.path.join(CACHE, 'episodes.pkl'), 'rb'))
    tasks = sorted({task_key(dict(e)) for e in eps})
    assign = {}
    for k in tasks:
        assign[k] = split_of(k)
    os.makedirs(OUT, exist_ok=True)
    c = Counter(assign.values())
    json.dump(dict(salt=SPLIT_SALT, rule='sha1(salt+bench_family|task) < 0.6 -> train', counts=c, tasks=assign),
              open(os.path.join(OUT, 'split.json'), 'w'))
    print('split', dict(c))


# ==== 3. evaluation primitives ==============================================================================
ENV_TRIGS = ('edit_failed', 'loop', 'no_progress_wait', 'error_streak')


def fires_at(e, t, c, trig):
    """c must already have enabled=[trig]."""
    sc = e['sc']
    if trig in ENV_TRIGS:
        budget = e.get('budget_sec') if trig == 'no_progress_wait' else None
        el = sc.turn_ts.get(t) if budget else None
        return bool(rt.env_fires(sc, t, c, elapsed_sec=el, budget_sec=budget))
    return bool(rt.dec_fires(sc.dec[t], t, c))


def fire_turns(e, cfg, trig, first_only=True):
    """Turns where trig fires with 2 <= t < t_stop (submit_check: at t_done only)."""
    sc = e['sc']
    c = dict(cfg, enabled=[trig])
    if trig == 'submit_check':
        t = e['t_done']
        return [t] if t and fires_at(e, t, c, trig) else []
    out = []
    for t in range(cfg.get('min_turn', 2), min(e['t_stop'], sc.turn + 1)):
        if fires_at(e, t, c, trig):
            out.append(t)
            if first_only:
                break
    return out


def union_first(e, cfg, trigs):
    best = None
    for tr in trigs:
        f = fire_turns(e, cfg, tr)
        if f and (best is None or f[0] < best[0]):
            best = (f[0], tr)
    return best


def recoverable(e, t, at_done=False):
    """(recoverable?, basis) for a fire at turn t on a failed episode. at_done: a label on the done-claim itself,
    where the proxy is only 'time left' (the turn rule would reject every done-claim)."""
    r = e.get('reader')
    if r is not None:
        if str(r.get('teacher_could_recover', '')).startswith('no'):
            return False, 'reader'
        p = r.get('point_of_no_return_turn')
        if p == 'none' or p is None:
            return True, 'reader'
        try:
            return t <= int(p), 'reader'
        except (TypeError, ValueError):
            return True, 'reader'
    ok = at_done or t <= 0.8 * max(1, e['turns'])
    b = e.get('budget_sec')
    el = e['sc'].turn_ts.get(t)
    if ok and b and el is not None:
        ok = el <= 0.8 * b
    return ok, 'proxy'


def per_episode(e, cfg, trig):
    ft = fire_turns(e, cfg, trig) if trig != 'UNION' else None
    if trig == 'UNION':
        u = union_first(e, cfg, cfg['union'])
        ft = [u[0]] if u else []
    fired = bool(ft)
    t = ft[0] if ft else None
    res = dict(fired=fired, t=t, pos=(t / max(1, e['t_stop'] - 1)) if t else None)
    if e['outcome'] == 'fail':
        rec, basis = recoverable(e, t, at_done=(trig == 'submit_check')) if fired else (False, 'reader' if e.get('reader') else 'proxy')
        res.update(hit=fired and rec, basis=basis, fired_any=fired)
    return res


# ---- bootstrap over tasks --------------------------------------------------------------------------------
def boot_ci(items, key, n=1000, seed=0):
    """items: list of (task, value 0/1). Returns (mean, lo, hi) with a task-cluster bootstrap."""
    if not items:
        return (None, None, None)
    by = defaultdict(list)
    for task, v in items:
        by[task].append(v)
    tasks = list(by)
    sums = [(sum(by[k]), len(by[k])) for k in tasks]
    tot = sum(s for s, _ in sums) / sum(m for _, m in sums)
    rng = random.Random(seed)
    stats = []
    for _ in range(n):
        s = m = 0
        for _ in range(len(tasks)):
            a, b = sums[rng.randrange(len(tasks))]
            s += a; m += b
        stats.append(s / m if m else 0)
    stats.sort()
    return (tot, stats[int(0.025 * n)], stats[int(0.975 * n) - 1])


def rates(eps, cfg, trig, ci=False):
    """Per-group FF and hit. Returns {group: dict}."""
    out = {}
    by = defaultdict(list)
    for e in eps:
        by[e['group']].append(e)
    for g, es in by.items():
        P = [e for e in es if e['outcome'] == 'pass']
        F = [e for e in es if e['outcome'] == 'fail']
        rp = [(e['task'], per_episode(e, cfg, trig)) for e in P]
        rf = [(e['task'], per_episode(e, cfg, trig)) for e in F]
        d = dict(n_pass=len(P), n_fail=len(F), n_pass_tasks=len({t for t, _ in rp}))
        ffi = [(t, int(r['fired'])) for t, r in rp]
        hti = [(t, int(r['hit'])) for t, r in rf]
        d['ff'] = boot_ci(ffi, None) if ci else (sum(v for _, v in ffi) / len(ffi) if ffi else None,)
        d['hit'] = boot_ci(hti, None) if ci else (sum(v for _, v in hti) / len(hti) if hti else None,)
        d['fired_fail'] = sum(r['fired'] for _, r in rf) / len(rf) if rf else None
        rd = [r for _, r in rf if r['basis'] == 'reader']
        rp_ = [r for _, r in rf if r['basis'] == 'proxy']
        d['hit_reader'] = (sum(r['hit'] for r in rd) / len(rd), len(rd)) if rd else (None, 0)
        d['hit_proxy'] = (sum(r['hit'] for r in rp_) / len(rp_), len(rp_)) if rp_ else (None, 0)
        pos = sorted(r['pos'] for _, r in rf if r['fired'] and r['pos'] is not None)
        d['median_pos_fail'] = pos[len(pos) // 2] if pos else None
        posp = sorted(r['pos'] for _, r in rp if r['fired'] and r['pos'] is not None)
        d['median_pos_pass'] = posp[len(posp) // 2] if posp else None
        out[g] = d
    return out


# ==== 4. tuning ==============================================================================================
BASE = dict(rt.PROPOSAL_CONFIG)


def objective(rs):
    """Mean hit over student groups, if FF <= cap in every student group with >= 20 train passes."""
    ok, hits = True, []
    for g in STUDENT_GROUPS:
        d = rs.get(g)
        if not d:
            continue
        if d['n_pass'] >= 20 and d['ff'][0] is not None and d['ff'][0] > FF_CAP:
            ok = False
        if d['hit'][0] is not None:
            hits.append(d['hit'][0])
    return ok, (sum(hits) / len(hits) if hits else 0.0)


def greedy_families(eps, trig, key, families, log):
    """Add phrase/pattern families one at a time while every student group's FF stays <= cap."""
    chosen = []
    # rank families by their own objective
    single = []
    for f in families:
        cfg = dict(BASE, **{key: [f]})
        rs = rates(eps, cfg, trig)
        ok, h = objective(rs)
        single.append((f, ok, h, {g: round(rs[g]['ff'][0], 3) for g in rs if g in STUDENT_GROUPS}))
        log.append(dict(trigger=trig, family=f, ok=ok, hit=h, ff=single[-1][3]))
    for f, ok, h, ff in sorted(single, key=lambda x: -x[2]):
        if not ok or h <= 0:
            continue
        cfg = dict(BASE, **{key: chosen + [f]})
        ok2, h2 = objective(rates(eps, cfg, trig))
        cur = objective(rates(eps, dict(BASE, **{key: chosen}), trig))[1] if chosen else 0
        if ok2 and h2 > cur:
            chosen.append(f)
    return chosen, single


def grid(eps, trig, space, log):
    import itertools
    keys = list(space)
    best = None
    for vals in itertools.product(*[space[k] for k in keys]):
        cfg = dict(BASE, **dict(zip(keys, vals)))
        rs = rates(eps, cfg, trig)
        ok, h = objective(rs)
        log.append(dict(trigger=trig, params=dict(zip(keys, vals)), ok=ok, hit=round(h, 4),
                        ff={g: round(rs[g]['ff'][0], 3) for g in rs if g in STUDENT_GROUPS and rs[g]['ff'][0] is not None}))
        if ok and (best is None or h > best[1]):
            best = (dict(zip(keys, vals)), h)
    return best


def cmd_tune():
    eps = [e for e in load_eps() if e['split'] == 'train' and e['group'] in STUDENT_GROUPS]
    print('train student episodes', sorted(Counter((e['group'], e['outcome']) for e in eps).items()))
    log, choice = [], {}

    def pick(trig, params):
        rs = rates(eps, dict(BASE, **params), trig)
        ok, h = objective(rs)
        choice[trig] = dict(params=params, feasible=ok, train_hit=round(h, 4),
                            train_ff={g: round(rs[g]['ff'][0], 4) for g in rs if rs[g]['ff'][0] is not None})
        print(trig, choice[trig], flush=True)

    # edit_failed: families greedily, then a repeated-failure narrowing
    fams, _ = greedy_families(eps, 'edit_failed', 'edit_families', list(rt.EDITFAIL_FAMILIES) +
                              ['nofile_edit', 'pyerr_target', 'git_diff_empty'], log)
    b = grid(eps, 'edit_failed', dict(edit_families=[BASE['edit_families']] + ([fams] if fams else []),
                                       edit_k=[1, 2, 3, 4], edit_window=[5, 10, 20]), log)
    g_hit = objective(rates(eps, dict(BASE, edit_families=fams), 'edit_failed'))[1] if fams else -1
    if b and b[1] > g_hit:
        pick('edit_failed', b[0])
    else:
        pick('edit_failed', dict(edit_families=fams, edit_k=1))
    # gave_up: phrase families greedily
    fams, _ = greedy_families(eps, 'gave_up', 'giveup_families', list(rt.GIVEUP_FAMILIES), log)
    pick('gave_up', dict(giveup_families=fams))
    # loop: modes x window x k x similarity x error signature
    best = None
    for modes in (['varied'], ['exact'], ['same_output'], ['varied', 'same_output'], ['varied', 'exact']):
        space = dict(loop_modes=[modes], loop_window=[3, 4, 5, 6], loop_k=[2, 3, 4])
        if 'varied' in modes:
            space.update(loop_sim=[0.5, 0.6, 0.7, 0.8, 0.9], loop_err=['strict', 'broad'])
        if 'same_output' in modes:
            space.update(loop_same_output_distinct=[False, True], loop_min_out=[30, 80])
        b = grid(eps, 'loop', space, log)
        if b and (best is None or b[1] > best[1]):
            best = b
    pick('loop', best[0] if best else dict(loop_modes=[]))
    # no-progress wait: count rule; the budget rule is reported as a variant (needs timestamps + budget)
    b = grid(eps, 'no_progress_wait', dict(wait_k=[2, 3, 4, 5, 6, 8], wait_budget_frac=[None]), log)
    pick('no_progress_wait', b[0] if b else dict(wait_k=None, wait_budget_frac=None))
    b2 = grid(eps, 'no_progress_wait', dict(wait_k=[None], wait_budget_frac=[0.15, 0.25, 0.35, 0.5],
                                            wait_budget_min=[1, 2, 3]), log)
    choice['no_progress_wait']['budget_variant'] = b2
    # error_streak: narrowed variants
    b = grid(eps, 'error_streak', dict(streak_n=[2, 3, 4, 5, 6, 8], streak_same_sig=[False, True],
                                       streak_no_edit_between=[False, True]), log)
    pick('error_streak', b[0] if b else dict(streak_n=99))
    # submit-time check flags
    fams, _ = greedy_families(eps, 'submit_check', 'submit_flags',
                              ['no_edit', 'stub_marker', 'stub_pass', 'tests_rm', 'tests_skip'], log)
    pick('submit_check', dict(submit_flags=fams))
    for tr in ('success_claim_unchecked', 'success_contradicted', 'destructive'):
        pick(tr, {})
    os.makedirs(OUT, exist_ok=True)
    json.dump(choice, open(os.path.join(OUT, 'tuned_config.json'), 'w'), indent=1, default=str)
    with open(os.path.join(OUT, 'tuning_log.jsonl'), 'w') as f:
        for r in log:
            f.write(json.dumps(r, default=str) + '\n')


# Decisions taken AFTER the held-out check and the judges (documented in the README): the takeover set is
# done_claim + loop + no_progress_wait, and wait_k goes from the train optimum 2 to 3 (k=2 exceeded the cap on
# held-out). tuned_config() is the pure train result; final_config() applies these two decisions.
POST_HOC = dict(wait_k=3)
FINAL_UNION = ['loop', 'no_progress_wait']


def tuned_config():
    """BASE with every trigger's train-tuned parameters (parameter names do not collide across triggers)."""
    t = json.load(open(os.path.join(OUT, 'tuned_config.json')))
    cfg = dict(BASE)
    for trig, c in t.items():
        cfg.update(c['params'])
    return cfg


def final_config():
    return dict(tuned_config(), **POST_HOC)


# ==== 5. report ==============================================================================================
def prefix_matched(eps, cfg, trig, reps=20, seed=0):
    """Truncate each failure to the length (turns before done) of a randomly paired pass of the same group."""
    rng = random.Random(seed)
    by = defaultdict(lambda: ([], []))
    for e in eps:
        by[e['group']][0 if e['outcome'] == 'pass' else 1].append(e)
    out = {}
    for g, (P, F) in by.items():
        if not P or not F:
            continue
        firsts_f = {e['id']: (per_episode(e, cfg, trig)['t']) for e in F}
        firsts_p = [per_episode(e, cfg, trig)['t'] for e in P]
        ff = sum(1 for x in firsts_p if x) / len(P)
        vals = []
        for _ in range(reps):
            for e in F:
                p = P[rng.randrange(len(P))]
                L = p['t_stop'] - 1
                t = firsts_f[e['id']]
                vals.append(1 if (t is not None and t <= L) else 0)
        out[g] = dict(ff=ff, fail_fire_truncated=sum(vals) / len(vals))
    return out


def per100(eps, cfg, trig):
    by = defaultdict(lambda: [0, 0, 0, 0])
    for e in eps:
        ft = fire_turns(e, cfg, trig, first_only=False) if trig != 'submit_check' else []
        n = max(0, min(e['t_stop'], e['turns'] + 1) - 2)
        k = 0 if e['outcome'] == 'pass' else 2
        by[e['group']][k] += len(ft)
        by[e['group']][k + 1] += n
    return {g: dict(pass_per100=100 * v[0] / v[1] if v[1] else None, fail_per100=100 * v[2] / v[3] if v[3] else None)
            for g, v in by.items()}


def fmt(ci):
    if ci is None or ci[0] is None:
        return '-'
    if len(ci) == 1:
        return '%.1f' % (100 * ci[0])
    return '%.1f [%.1f, %.1f]' % (100 * ci[0], 100 * ci[1], 100 * ci[2])


def cmd_report():
    eps = load_eps()
    cfg = dict(final_config(), union=FINAL_UNION)
    ho = [e for e in eps if e['split'] == 'heldout']
    tr = [e for e in eps if e['split'] == 'train']
    res = dict(config=cfg, counts={}, heldout={}, train={}, prefix={}, per100={}, proposal_heldout={})
    for name, es in (('heldout', ho), ('train', tr)):
        res['counts'][name] = {g: dict(Counter(e['outcome'] for e in es if e['group'] == g)) for g, _, _ in GROUPS}
    for trig in TRIGGERS + ['UNION']:
        res['heldout'][trig] = rates(ho, cfg, trig, ci=True)
        res['train'][trig] = rates(tr, cfg, trig)
        res['proposal_heldout'][trig] = rates(ho, dict(BASE, union=['edit_failed', 'loop', 'gave_up']), trig, ci=True)
        res['prefix'][trig] = prefix_matched(ho, cfg, trig)
        res['per100'][trig] = per100(ho, cfg, trig)
        print(trig, {g: (fmt(d['hit']), fmt(d['ff'])) for g, d in res['heldout'][trig].items()}, flush=True)
    # where the union fires (held-out failures and passes)
    where = defaultdict(Counter)
    for e in ho:
        u = union_first(e, cfg, cfg['union'])
        if u:
            b = min(4, int(5 * (u[0] - 1) / max(1, e['t_stop'] - 1)))
            where[(e['group'], e['outcome'])]['q%d' % b] += 1
            where[(e['group'], e['outcome'], 'by')][u[1]] += 1
    res['union_where'] = {'|'.join(k): dict(v) for k, v in where.items()}
    json.dump(res, open(os.path.join(OUT, 'results.json'), 'w'), indent=1, default=str)
    # one row per episode: first fire turn of every trigger, calibrated and 2026-09-25-proposal definitions
    import csv
    trigs = [t for t in TRIGGERS]
    with open(os.path.join(OUT, 'per_episode.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'group', 'source', 'model', 'task', 'split', 'outcome', 'turns', 't_done', 'reader_ponr'] +
                   ['cal_' + t for t in trigs] + ['prop_' + t for t in trigs])
        for e in eps:
            cal = [(fire_turns(e, cfg, t) or [''])[0] for t in trigs]
            prop = [(fire_turns(e, BASE, t) or [''])[0] for t in trigs]
            w.writerow([e['id'], e['group'], e['source'], e.get('model'), e.get('task'), e['split'], e['outcome'],
                        e['turns'], e['t_done'] or '', (e['reader'] or {}).get('point_of_no_return_turn', '')] + cal + prop)
    print('wrote', os.path.join(OUT, 'results.json'), 'and per_episode.csv')


def cmd_extras():
    """Held-out sensitivity rows (POST-HOC, labelled as such), alternative unions, and the proposal's definitions."""
    eps = [e for e in load_eps() if e['split'] == 'heldout']
    cfg = tuned_config()
    rows = {}

    def row(name, c, trig):
        rs = rates(eps, c, trig, ci=True)
        pm = prefix_matched(eps, c, trig)
        rows[name] = {g: dict(hit=d['hit'], ff=d['ff'], n_pass=d['n_pass'], n_fail=d['n_fail'],
                              prefix_fail=pm.get(g, {}).get('fail_fire_truncated'), median_pos=d['median_pos_fail'])
                      for g, d in rs.items()}
        print(name, {g: (fmt(d['hit']), fmt(d['ff'])) for g, d in rs.items() if g.startswith('S-')}, flush=True)

    for k in (3, 4):
        row('posthoc_wait_k%d' % k, dict(cfg, wait_k=k), 'no_progress_wait')
    row('wait_budget_variant', dict(cfg, wait_k=None, wait_budget_frac=0.15, wait_budget_min=1), 'no_progress_wait')
    row('posthoc_loop_k4', dict(cfg, loop_k=4), 'loop')
    row('loop_proposal_design_varied', dict(cfg, loop_modes=['varied', 'same_output'], loop_window=4, loop_k=3,
                                            loop_sim=0.6, loop_err='strict', loop_same_output_distinct=True), 'loop')
    for u in (['gave_up'], ['gave_up', 'no_progress_wait'], ['gave_up', 'no_progress_wait', 'loop'],
              ['gave_up', 'no_progress_wait', 'loop', 'error_streak'],
              ['gave_up', 'no_progress_wait', 'loop', 'error_streak', 'edit_failed']):
        row('union_' + '+'.join(u), dict(cfg, union=u), 'UNION')
    row('union_proposal_defs', dict(BASE, union=['edit_failed', 'loop', 'gave_up']), 'UNION')
    # the final set (after the judges): loop + no-progress wait at k=3; and with gave_up added back
    fin = dict(cfg, wait_k=3)
    row('union_final_loop+wait3', dict(fin, union=['loop', 'no_progress_wait']), 'UNION')
    row('union_final+gave_up', dict(fin, union=['loop', 'no_progress_wait', 'gave_up']), 'UNION')
    # coverage of failures that never reach a done-claim (timeouts): the done-claim takeover gets nothing there
    for u in (['loop', 'no_progress_wait'], ['loop', 'no_progress_wait', 'gave_up']):
        c = dict(fin, union=u)
        out = {}
        for g in STUDENT_GROUPS:
            F = [e for e in eps if e['group'] == g and e['outcome'] == 'fail']
            for ending, sub in (('no_done_claim', [e for e in F if not e['t_done']]),
                                ('done_claimed', [e for e in F if e['t_done']])):
                items = [(e['task'], int(per_episode(e, c, 'UNION')['hit'])) for e in sub]
                out['%s|%s' % (g, ending)] = dict(n=len(sub), hit=boot_ci(items, None))
        rows['by_ending_' + '+'.join(u)] = out
        print('by_ending', u, {k: (v['n'], fmt(v['hit'])) for k, v in out.items()}, flush=True)
    json.dump(rows, open(os.path.join(OUT, 'extras.json'), 'w'), indent=1, default=str)


# ==== 6. firing-point precision (LLM judges) ================================================================
# (judge set name, trigger, config override); the broad edit_failed is the 2026-09-25 proposal's definition
# Judged 2026-09-25 with the train-tuned config before the bare-prompt parsing fix; edit_failed was then
# "4 failed edits within 5 turns" (edit_failed_4in5), a variant with the same ~1-2 % hit as the final narrowed one.
JUDGE_SETS = [('edit_failed_4in5', 'edit_failed', dict(edit_families=BASE['edit_families'], edit_k=4, edit_window=5)),
              ('loop', 'loop', {}), ('gave_up', 'gave_up', {}),
              ('no_progress_wait', 'no_progress_wait', {}), ('error_streak', 'error_streak', {}),
              ('submit_check', 'submit_check', {}),
              ('edit_failed_proposal', 'edit_failed', dict(edit_families=BASE['edit_families'], edit_k=1))]


def load_messages(e):
    d = e['desc']
    if d['kind'] == 'atif':
        rec = ad.load_atif(d['path'], {})
        return rec['messages']
    with open(d['path'], 'rb') as f:
        f.seek(d['offset'])
        return json.loads(f.readline())['messages']


def _turn_views(messages):
    """[(reply_text, observation_text)] per assistant turn, in order (tool results joined)."""
    out, cur = [], None
    for m in messages:
        role = m.get('role')
        if role == 'assistant':
            if cur is not None:
                out.append(cur)
            txt = m.get('content') or ''
            if m.get('tool_calls'):
                r = rt.tool_reply(m)
                txt = (txt + '\n' + '\n'.join('$ ' + c.strip() for c, _ in r['cmds'])).strip()
            cur = [txt, '']
        elif cur is not None and (role in ('user', 'tool')):
            cur[1] += ('\n' if cur[1] else '') + (m.get('content') or '')
    if cur is not None:
        out.append(cur)
    return out


def _clip(s, n):
    s = re.sub(r'\n\s*\n(\s*\n)+', '\n\n', s or '')
    return s if len(s) <= n else s[:n // 2] + '\n[... %d chars cut ...]\n' % (len(s) - n) + s[-n // 2:]


def excerpt(e, t, reason, trig):
    msgs = load_messages(e)
    task = next((m['content'] for m in msgs if m.get('role') == 'user'), '') or ''
    for anchor in ('Task Description:', '<pr_description>'):
        if anchor in task:
            task = task[task.index(anchor):]
            break
    task = task.split('Current terminal state:')[0]
    tv = _turn_views(msgs)
    parts = ['TASK (start of the first user message):\n' + _clip(task, 900)]
    env = trig in ENV_TRIGS
    lo = max(1, t - 2 if env else t - 1)
    for u in range(lo, t + (0 if env else 1)):
        if u - 1 >= len(tv):
            break
        rep, obs = tv[u - 1]
        rep = rt.THINK_RE.sub('', rep)
        parts.append('--- TURN %d: agent reply ---\n%s' % (u, _clip(rep, 1300)))
        if u < t:
            parts.append('--- TURN %d: terminal output returned ---\n%s' % (u, _clip(obs, 1300)))
    where = ('The router would take over now: the teacher writes turn %d.' % t if env else
             'The router judges the reply of turn %d (shown last); if it fires, that reply is discarded and the '
             'teacher answers instead.' % t)
    parts.append('--- TRIGGER ---\n%s fired at turn %d of %d. %s\nReason: %s' % (trig, t, e['turns'], where, reason))
    return '\n\n'.join(parts)


def cmd_judge_sample(per=40, seed=0):
    eps = [e for e in load_eps() if e['group'] in STUDENT_GROUPS]
    rng = random.Random(seed)
    jdir = os.path.join(OUT, 'judge')
    os.makedirs(jdir, exist_ok=True)
    for name, trig, over in JUDGE_SETS:
        if os.path.exists(os.path.join(jdir, name + '.verdicts.jsonl')) and '--force' not in sys.argv:
            print(name, 'already judged; not resampling (pass --force to overwrite)')
            continue
        cfg = dict(tuned_config(), **over)
        c = dict(cfg, enabled=[trig])
        pools = {'pass': [], 'fail': []}
        for e in eps:
            ft = fire_turns(e, cfg, trig)
            if ft:
                pools[e['outcome']].append((e, ft[0]))
        for o in ('fail', 'pass'):
            rng.shuffle(pools[o])
        want_pass = min(len(pools['pass']), per // 2)
        want_fail = min(len(pools['fail']), per - want_pass)
        chosen = pools['pass'][:want_pass] + pools['fail'][:want_fail]
        rng.shuffle(chosen)
        xdir = os.path.join(DATA, 'judge')     # excerpts quote benchmark text: kept out of the repo
        os.makedirs(xdir, exist_ok=True)
        with open(os.path.join(jdir, name + '.jsonl'), 'w') as f, open(os.path.join(xdir, name + '.txt'), 'w') as g:
            for k, (e, t) in enumerate(chosen):
                sc = e['sc']
                fs = (rt.env_fires(sc, t, c) if trig in ENV_TRIGS else rt.dec_fires(sc.dec[t], t, c))
                reason = fs[0]['reason'] if fs else '(see transcript)'
                x = excerpt(e, t, reason, trig)
                jid = '%s-%02d' % (name, k)
                f.write(json.dumps(dict(jid=jid, id=e['id'], group=e['group'], outcome=e['outcome'], turn=t,
                                        reason=reason)) + '\n')
                g.write('=' * 100 + '\nITEM %s\n' % jid + x + '\n')
        print(name, 'pass fires', len(pools['pass']), 'fail fires', len(pools['fail']), '-> sampled', len(chosen))


def cmd_judge_score():
    jdir = os.path.join(OUT, 'judge')
    out = {}
    for trig, _, _ in JUDGE_SETS:
        meta = {json.loads(l)['jid']: json.loads(l) for l in open(os.path.join(jdir, trig + '.jsonl'))}
        vf = os.path.join(jdir, trig + '.verdicts.jsonl')
        if not os.path.exists(vf):
            continue
        V = [json.loads(l) for l in open(vf) if l.strip()]
        rows = []
        for v in V:
            m = meta.get(v['jid'])
            if m:
                rows.append((m['outcome'], v['verdict']))
        d = {}
        for o in ('all', 'pass', 'fail'):
            rr = [x for x in rows if o == 'all' or x[0] == o]
            if rr:
                d[o] = dict(n=len(rr), real=sum(1 for _, v in rr if v == 'real_mistake'),
                            precision=sum(1 for _, v in rr if v == 'real_mistake') / len(rr))
        out[trig] = d
        print(trig, d)
    json.dump(out, open(os.path.join(OUT, 'judge_precision.json'), 'w'), indent=1)


if __name__ == '__main__':
    {'scan': cmd_scan, 'split': cmd_split, 'tune': cmd_tune, 'report': cmd_report,
     'extras': cmd_extras, 'judge-sample': cmd_judge_sample, 'judge-score': cmd_judge_score}[sys.argv[1]]()
