"""Turn-level follow-ups to opd_ref_probe.py: does a teacher score rank candidate TURNS at the same state? That is what
PivotRL x CompassOPD needs (8 candidate turns per pivot state, GRPO on a per-turn teacher score), and what the
probe's episode-level AUC does not show. CPU only, on the probe's episodes.pkl and scores/.

    episode   per-token mean over the whole episode, within task (the probe's headline, for reference)
    turn1     all rollouts of a task start from the same state: within-task AUC of the first turn's score
    branch    natural branch points: rollouts of a task that ran identical commands for k turns sit in (nearly) the
              same sandbox state before turn k+1; within-node AUC of the turn-(k+1) score, bootstrapped over tasks
    early     episode score from the first k turns only: how early the signal shows (loops and retries come late)
    confound  within-task AUC of turn count / length / repeated commands, and of each signal with those regressed
              out inside each task
    claim     the first done-claim: its "true" token and its turn, thinking in context
    mix       teacher - beta * reference (beta 0 .. 1.25) and a mean-of-references reference

    python opd_ref_turns.py --out <probe run dir> --refs 0.8B,2B,4B,9B [--drop-first-chunk] [--tag T]
"""
import argparse
import json
import os
import pickle
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opd_ref_probe import load_scores, pooled_auc  # noqa: E402

KEYS_RE = re.compile(r'"keystrokes"\s*:\s*"((?:[^"\\]|\\.)*)"')
DONE_RE = re.compile(r'"task_complete"\s*:\s*true')
VIEWS = ('all', 'think', 'action')
EARLY_K = (1, 2, 3, 5, 10)
BETAS = (0.25, 0.5, 0.75, 1.25)


def signature(action_text):
    """The turn's decision: its keystrokes (whitespace-normalised) and whether it claims done."""
    keys = tuple(' '.join(k.split()) for k in KEYS_RE.findall(action_text))
    return keys, bool(DONE_RE.search(action_text))


def grouped_auc(vals, labels, groups, clusters, n_boot=2000, seed=0):
    """P(pass scores above fail) inside each group (ties 1/2), groups weighted equally; 95 % CI from a bootstrap over
    clusters (tasks), so nested or correlated groups of one task move together."""
    by = {}
    for v, l, g, c in zip(vals, labels, groups, clusters):
        if np.isfinite(v):
            by.setdefault(g, [c, [], []])[1 if l else 2].append(v)
    per_cluster = {}
    for g, (c, p, f) in by.items():
        if p and f:
            p, f = np.asarray(p)[:, None], np.asarray(f)[None, :]
            per_cluster.setdefault(c, []).append(float(np.mean((p > f) + 0.5 * (p == f))))
    if not per_cluster:
        return dict(auc=None, n_groups=0, n_tasks=0)
    cl = list(per_cluster.values())
    allv = np.concatenate([np.asarray(x) for x in cl])
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(cl), len(cl))
        boots.append(np.concatenate([np.asarray(cl[k]) for k in pick]).mean())
    return dict(auc=float(allv.mean()), lo=float(np.percentile(boots, 2.5)), hi=float(np.percentile(boots, 97.5)),
                n_groups=int(len(allv)), n_tasks=int(len(cl)))


def within_task_residual(y, X, tasks):
    """y minus its within-task OLS fit on X (task fixed effects): what the signal says beyond the features."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    tasks = np.asarray(tasks)
    yd, Xd = y.copy(), X.copy()
    for t in np.unique(tasks[ok]):
        m = ok & (tasks == t)
        yd[m] -= yd[m].mean()
        Xd[m] -= Xd[m].mean(axis=0)
    beta, *_ = np.linalg.lstsq(Xd[ok], yd[ok], rcond=None)
    r = np.full_like(y, np.nan)
    r[ok] = yd[ok] - Xd[ok] @ beta
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--refs', default='0.8B,2B,4B,9B')
    p.add_argument('--mix-refs', default='2B,9B', help='references for the teacher - beta * ref sweep')
    p.add_argument('--drop-first-chunk', action='store_true')
    p.add_argument('--no-student', action='store_true', help='runs scored without the student (no opd/student)')
    p.add_argument('--same-state-only', action='store_true',
                   help='episodes are prefixes (opd_ref_branch.py prep): report only the turn1 / branch tests')
    p.add_argument('--tag', default='')
    a = p.parse_args()

    eps = pickle.load(open(os.path.join(a.out, 'episodes.pkl'), 'rb'))
    refs = a.refs.split(',') if a.refs else []
    models = ['teacher'] + ([] if a.no_student else ['student']) + refs
    sc = {m: load_scores(a.out, m)[0] for m in models}
    keep = [i for i in range(len(eps)) if all(i in sc[m] for m in models)]
    print('episodes with every score: %d of %d' % (len(keep), len(eps)), flush=True)

    signals = ['teacher'] + ([] if a.no_student else ['opd', 'student']) + ['compass_' + r for r in refs]
    if len(refs) > 1:
        signals.append('compass_refmean')
    mix_refs = [r for r in a.mix_refs.split(',') if r in refs]
    for r in mix_refs:
        signals += ['mix%.2f_%s' % (b, r) for b in BETAS]

    E = []   # per episode: task, passed, per-turn sums per signal/view, per-turn token counts, signatures, claim
    for i in keep:
        e = eps[i]
        ch = e['chunks']
        if a.drop_first_chunk:
            ch = ch[ch[:, 6] > 0]
        if len(ch) == 0:
            continue
        t_idx, kind, sl, sh, ql, qh, _, _, is_done = [ch[:, k] for k in range(9)]
        lq = {}
        for mdl in ['teacher'] + refs:
            cq = np.cumsum(np.concatenate([[0.0], np.nan_to_num(sc[mdl][i])]))
            lq[mdl] = cq[qh] - cq[ql]
        sig = dict(teacher=lq['teacher'])
        if not a.no_student:
            cs = np.cumsum(np.concatenate([[0.0], np.nan_to_num(sc['student'][i])]))
            lS = cs[sh] - cs[sl]
            sig.update(opd=lq['teacher'] - lS, student=lS)
        for r in refs:
            sig['compass_' + r] = lq['teacher'] - lq[r]
        if len(refs) > 1:
            sig['compass_refmean'] = lq['teacher'] - np.mean([lq[r] for r in refs], axis=0)
        for r in mix_refs:
            for b in BETAS:
                sig['mix%.2f_%s' % (b, r)] = lq['teacher'] - b * lq[r]
        T = int(e['n_turns'])
        ntok = (sh - sl).astype(float)
        masks = dict(all=np.ones(len(ch), bool), think=kind == 0, action=kind == 1)
        tok = {v: np.bincount(t_idx[masks[v]], weights=ntok[masks[v]], minlength=T) for v in VIEWS}
        tsum = {s: {v: np.bincount(t_idx[masks[v]], weights=x[masks[v]], minlength=T) for v in VIEWS}
                for s, x in sig.items()}
        # each turn's action text -> its decision signature
        sigs = []
        offs, text = e['s_offs'], e['s_text']
        for t in range(T):
            am = (t_idx == t) & (kind == 1)
            if not am.any():
                sigs.append(((), False))
                continue
            a0, a1 = int(offs[sl[am].min()][0]), int(offs[sh[am].max() - 1][1])
            sigs.append(signature(text[a0:a1]))
        first_done = min(e['done_turns']) if e['done_turns'] else None
        claim = {}
        if first_done is not None and first_done < T:
            dm = (is_done == 1) & (t_idx == first_done)
            for s, x in sig.items():
                claim[s] = dict(token=float(x[dm].sum()) if dm.any() else np.nan,
                                turn_action=float(tsum[s]['action'][first_done] / tok['action'][first_done])
                                if tok['action'][first_done] > 0 else np.nan,
                                turn_all=float(tsum[s]['all'][first_done] / tok['all'][first_done])
                                if tok['all'][first_done] > 0 else np.nan)
        seen, rep = set(), 0
        for s_ in sigs:
            rep += s_ in seen and bool(s_[0])
            seen.add(s_)
        E.append(dict(task=e['task'], passed=e['reward'] > 0, T=T, tok=tok, tsum=tsum, sigs=sigs, claim=claim,
                      feats=dict(n_turns=T, log_tokens=float(np.log1p(tok['all'].sum())),
                                 think_per_turn=float(tok['think'].sum() / T),
                                 action_per_turn=float(tok['action'].sum() / T),
                                 repeat_rate=rep / T)))
    labels = [x['passed'] for x in E]
    tasks = [x['task'] for x in E]
    print('episodes %d, pass %d, tasks with both outcomes %d' % (
        len(E), sum(labels), len({t for t in tasks if {x['passed'] for x in E if x['task'] == t} == {True, False}})),
        flush=True)

    def ep_score(x, s, v, k=None):
        n = x['tok'][v][:k].sum()
        return float(x['tsum'][s][v][:k].sum() / n) if n > 0 else np.nan

    def turn_score(x, s, v, t, agg='mean'):
        if t >= x['T']:
            return np.nan
        n = x['tok'][v][t]
        if n <= 0:
            return np.nan
        if s == 'shorter':           # length-only baseline: the shorter candidate wins
            return -float(n)
        return float(x['tsum'][s][v][t] / (n if agg == 'mean' else 1.0))

    M: dict = dict(n_episodes=len(E), n_pass=int(sum(labels)), signals={})
    for s in signals:
        m: dict = {}
        for v in VIEWS:
            m['episode_' + v] = grouped_auc([ep_score(x, s, v) for x in E], labels, tasks, tasks)
            m['turn1_' + v] = grouped_auc([turn_score(x, s, v, 0) for x in E], labels, tasks, tasks)
        m['turn1_action_sum'] = grouped_auc([turn_score(x, s, 'action', 0, 'sum') for x in E], labels, tasks, tasks)
        for k in EARLY_K:
            m['early%d_action' % k] = grouped_auc([ep_score(x, s, 'action', k) for x in E], labels, tasks, tasks)
            m['early%d_all' % k] = grouped_auc([ep_score(x, s, 'all', k) for x in E], labels, tasks, tasks)
        cl = [x for x in E if s in x['claim']]
        for part in ('token', 'turn_action', 'turn_all'):
            vals = [x['claim'][s][part] for x in cl]
            m['claim_%s_within' % part] = grouped_auc(vals, [x['passed'] for x in cl], [x['task'] for x in cl],
                                                      [x['task'] for x in cl])
            m['claim_%s_pooled' % part] = pooled_auc(vals, [x['passed'] for x in cl])
        M['signals'][s] = m
    M['signals']['shorter'] = {'turn1_' + v: grouped_auc([turn_score(x, 'shorter', v, 0) for x in E], labels, tasks,
                                                         tasks) for v in VIEWS}

    # ---- natural branch points: identical decisions for k turns -> the same state before turn k+1 ----------------
    nodes = []    # (node id, task, depth, [episode index], diverges)
    by_task = {}
    for j, x in enumerate(E):
        by_task.setdefault(x['task'], []).append(j)
    for t, js in by_task.items():
        maxd = max(E[j]['T'] for j in js)
        for d in range(1, maxd):
            groups = {}
            for j in js:
                if E[j]['T'] > d:
                    groups.setdefault(tuple(E[j]['sigs'][:d]), []).append(j)
            for key, members in groups.items():
                if len(members) < 2:
                    continue
                outs = {E[j]['passed'] for j in members}
                if outs != {True, False}:
                    continue
                diverges = len({E[j]['sigs'][d] for j in members}) > 1
                nodes.append(('%s|%d|%d' % (t, d, hash(key)), t, d, members, diverges))
    M['branch_nodes'] = dict(n=len(nodes), n_diverging=sum(n[4] for n in nodes),
                             n_tasks=len({n[1] for n in nodes}),
                             depth_hist={int(d): int(c) for d, c in zip(*np.unique([n[2] for n in nodes],
                                                                                    return_counts=True))}
                             if nodes else {})
    for s in signals + ['shorter']:
        for v in ('all', 'action'):
            for sub, pick in (('', lambda n: True), ('_diverging', lambda n: n[4])):
                vals, labs, grp, clu = [], [], [], []
                for nid, t, d, members, dv in nodes:
                    if not pick((nid, t, d, members, dv)):
                        continue
                    for j in members:
                        vals.append(turn_score(E[j], s, v, d))
                        labs.append(E[j]['passed'])
                        grp.append(nid)
                        clu.append(t)
                M['signals'][s]['branch_%s%s' % (v, sub)] = grouped_auc(vals, labs, grp, clu)

    # ---- confounds ---------------------------------------------------------------------------------------------------
    fnames = list(E[0]['feats'])
    M['features'] = {f: grouped_auc([x['feats'][f] for x in E], labels, tasks, tasks) for f in fnames}
    X = [[x['feats'][f] for f in fnames] for x in E]
    for s in signals:
        for v in ('all', 'action'):
            r = within_task_residual([ep_score(x, s, v) for x in E], X, tasks)
            M['signals'][s]['episode_%s_resid' % v] = grouped_auc(list(r), labels, tasks, tasks)
    # does the per-turn score favour short turns? (Spearman over all turns, per-token mean vs action tokens)
    for s in signals:
        sv, nv = [], []
        for x in E:
            for t in range(x['T']):
                if x['tok']['action'][t] > 0:
                    sv.append(x['tsum'][s]['action'][t] / x['tok']['action'][t])
                    nv.append(x['tok']['action'][t])
        rs = np.argsort(np.argsort(sv)).astype(float)
        rn = np.argsort(np.argsort(nv)).astype(float)
        M['signals'][s]['spearman_turn_score_vs_action_len'] = float(np.corrcoef(rs, rn)[0, 1])

    json.dump(M, open(os.path.join(a.out, 'turns%s.json' % a.tag), 'w'), indent=1)

    # ---- report ------------------------------------------------------------------------------------------------------
    def f(d):
        if not d or d.get('auc') is None:
            return '–'
        return '%.2f [%.2f, %.2f]' % (d['auc'], d['lo'], d['hi']) if 'lo' in d else '%.2f' % d['auc']
    main_sig = [s for s in signals if not s.startswith('mix')]
    same_sig = main_sig + ['shorter'] + [s for s in signals if s.startswith('mix0.75')]
    L = ['# opd_ref turn-level tests%s' % (' (%s)' % a.tag if a.tag else ''), '',
         '%d episodes (%d pass). Within-task AUC = P(a passing rollout scores above a failing one of the same task); '
         'turn1/branch compare candidate turns taken from the same state.' % (len(E), sum(labels)), '',
         'Branch nodes: %d (%d where the next decision differs) in %d tasks; depth histogram %s' % (
             M['branch_nodes']['n'], M['branch_nodes']['n_diverging'], M['branch_nodes']['n_tasks'],
             M['branch_nodes']['depth_hist']), '',
         '## same-state tests (what PivotRL needs)', '',
         '| signal | turn1 all | turn1 action | turn1 think | branch all | branch action | branch action, diverging |',
         '|---|---|---|---|---|---|---|']
    for s in same_sig:
        m = M['signals'][s]
        L.append('| %s | %s | %s | %s | %s | %s | %s |' % (s, f(m['turn1_all']), f(m['turn1_action']),
                                                           f(m['turn1_think']), f(m['branch_all']),
                                                           f(m['branch_action']), f(m['branch_action_diverging'])))
    L += ['', 'shorter = length-only baseline (fewer tokens in that view wins).']
    if a.same_state_only:
        open(os.path.join(a.out, 'turns%s.md' % a.tag), 'w').write('\n'.join(L) + '\n')
        print('\n'.join(L))
        return
    L += ['', '## episode level, and how early it shows', '',
          '| signal | episode all | episode action | think | action, first 1 | 2 | 3 | 5 | 10 turns | action, resid. |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for s in main_sig:
        m = M['signals'][s]
        L.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            s, f(m['episode_all']), f(m['episode_action']), f(m['episode_think']),
            *[f(m['early%d_action' % k]) for k in EARLY_K], f(m['episode_action_resid'])))
    L += ['', 'resid. = within-task residual after regressing on %s.' % ', '.join(fnames), '',
          '## confounds alone (within-task AUC; < 0.5 means lower value goes with passing)', '']
    for fn, d in M['features'].items():
        L.append('- %s: %s' % (fn, f(d)))
    L += ['', '## done-claim, thinking in context', '',
          '| signal | token within | token pooled | turn (action) within | turn (all) within | turn score vs action length (Spearman) |',
          '|---|---|---|---|---|---|']
    for s in main_sig:
        m = M['signals'][s]
        L.append('| %s | %s | %s | %s | %s | %.2f |' % (s, f(m['claim_token_within']), f(m['claim_token_pooled']),
                                                       f(m['claim_turn_action_within']),
                                                       f(m['claim_turn_all_within']),
                                                       m['spearman_turn_score_vs_action_len']))
    if mix_refs:
        L += ['', '## teacher - beta * reference', '',
              '| signal | episode action | turn1 action | branch action | claim token within | claim turn (action) within |',
              '|---|---|---|---|---|---|']
        for r in mix_refs:
            for s in ['teacher'] + ['mix%.2f_%s' % (b, r) for b in BETAS[:3]] + ['compass_' + r] + \
                     ['mix%.2f_%s' % (BETAS[3], r)]:
                m = M['signals'][s]
                L.append('| %s | %s | %s | %s | %s | %s |' % (s, f(m['episode_action']), f(m['turn1_action']),
                                                              f(m['branch_action']), f(m['claim_token_within']),
                                                              f(m['claim_turn_action_within'])))
    open(os.path.join(a.out, 'turns%s.md' % a.tag), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    sys.exit(main())
