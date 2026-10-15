#!/usr/bin/env python3
"""Benchmark of measures for "where is the teacher off-policy for the student / where will the student struggle".

Every measure is computed from the BASE student's view of a teacher token (or turn), with no outcome information:

  token measures (from onpolicy.py's top-20 statistics, matched between teacher and the student's own samples)
    nll        -log p(y)                                   baseline: raw surprise
    gap        log p_top1 - log p(y)                       disagreement beyond the student's own uncertainty
    excess     -log p(y) - H                               surprisal minus entropy (locally typical; Meister+ 2022)
    z          (-log p(y) - H) / sd[log p]                 standardized log-prob (Fast-DetectGPT analytic curvature)
    pval       -log(1 - above)                             surprise p-value under the student's own distribution
    rank       min(rank, K+1)                              clipped rank (DetectLLM / RSR)
    conflict   nll * (1 - H / ln K)                        confident conflict: unlikely AND the student is sure (EAFT)
    entropy    H                                           student uncertainty itself (forking tokens), a contrast
  extra token measures (from the score sets, prompt path)
    dnr        nll(no-refeed) - nll(full)                  dependence on the teacher's own earlier reasoning
    xstudent   mean nll over the two students               a second student's view
    position   -log(1 + offset in turn)                    trivial baseline: early tokens are hard
  turn measures: the mean of each token measure over content tokens, plus
    D          sum(nll - H) / sqrt(sum Var)                turn-level curvature
    rsr        sum(rank) / sum(nll)                        rank-surprisal ratio (lower = more learnable, so negated)
    sub1       share of tokens with p < 0.01               RSD's off-policy token rate
    length     content tokens in the turn                  trivial baseline

Yardsticks:
  B1 off-policy detection   per turn, AUROC of the teacher's turn against the student's N own samples at the same
                            prompt; token level, pooled AUROC of teacher vs own-sample content tokens per class
  B2 learnability (SFT)     the vanilla clean-pair SFT trained on these very episodes (passes 1-3); per token NLL at
                            base, 146, 292, 438, restricted to tokens SFT actually trained (earlier-turn reasoning over
                            1,000 tokens was cut and masked; kept as a control). Rho-1 categories (H = NLL >= tau).
                            Main task: among trained tokens that start hard, which stay hard after three passes (H->H
                            "fails to learn") vs get learned (H->L)
  B3 stability              Spearman of each token measure between 09-21 and RLVR1 (and full vs no-refeed)
  (B4, judged informativeness, is built from windows.jsonl written here and labelled separately)

    python benchmark.py --root <student_difficulty dir> --out <dir>
"""
import argparse
import collections
import glob
import gzip
import json
import math
import os
import random

import numpy as np

CLASSES = ('reasoning', 'prose', 'command', 'think_tag', 'call_fmt', 'eot')
CONTENT = (0, 1, 2)
TAU = 2.3
K = 20
TOKEN_MEASURES = ('nll', 'gap', 'excess', 'z', 'pval', 'rank', 'conflict', 'entropy')
TURN_EXTRA = ('D', 'rsr', 'sub1', 'length')


def token_measures(s):
    """dict of numpy arrays (one per measure) from onpolicy stats arrays."""
    nll = np.asarray(s['nll'], float)
    h = np.asarray(s['h_lb'], float)
    m2 = np.asarray(s['m2'], float)
    var = np.maximum(m2 - h * h, 1e-4)
    above = np.asarray(s['above'], float)
    rank = np.minimum(np.asarray(s['rank'], float), K + 1)
    return dict(nll=nll, gap=np.asarray(s['gap'], float), excess=nll - h, z=(nll - h) / np.sqrt(var),
                pval=-np.log(np.maximum(1.0 - above, 1e-6)), rank=rank,
                conflict=nll * (1.0 - np.minimum(1.0, h / math.log(K))), entropy=h, _var=var)


def turn_measures(tm, cls):
    m = np.isin(np.frombuffer(cls.encode(), np.uint8) - 48, CONTENT)
    if m.sum() == 0:
        return None
    out = {k: float(tm[k][m].mean()) for k in TOKEN_MEASURES}
    out['D'] = float((tm['nll'][m] - tm['entropy'][m]).sum() / math.sqrt(tm['_var'][m].sum()))
    out['rsr'] = -float(tm['rank'][m].sum() / max(tm['nll'][m].sum(), 1e-6))
    out['sub1'] = float((tm['nll'][m] > math.log(100)).mean())
    out['length'] = float(m.sum())
    return out


def auroc(pos, neg):
    """P(score(pos) > score(neg)) + 0.5 P(tie), via average ranks."""
    from scipy.stats import rankdata
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def boot_ci(per_traj, fn, n=1000, seed=0):
    """95 % CI by resampling trajectories; per_traj = {traj: data}, fn(list of data) -> value."""
    rng = random.Random(seed)
    keys = sorted(per_traj)
    vals = []
    for _ in range(n):
        smp = [per_traj[rng.choice(keys)] for _ in keys]
        v = fn(smp)
        if not math.isnan(v):
            vals.append(v)
    vals.sort()
    return [round(vals[int(0.025 * len(vals))], 4), round(vals[int(0.975 * len(vals)) - 1], 4)] if vals else [None, None]


def load_scores(root, sset, renders):
    """{traj: [per-turn nll arrays]} from a score set (prompt path), aligned to render turns."""
    out = {}
    for tid, rec in renders.items():
        f = os.path.join(root, 'scores', sset, tid + '.json.gz')
        if not os.path.exists(f):
            return None
        s = json.load(gzip.open(f, 'rt'))
        lp = dict(zip(s['positions'], s['lp']))
        out[tid] = [np.array([-lp[j] for j in range(t['tok_start'], t['tok_end'])]) for t in rec['turns']]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--students', nargs='+', default=['dk0921', 'rlvr1s12'])
    ap.add_argument('--sft', nargs='+', default=['sftv146', 'sftv292', 'sftv438'])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    renders = {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.root, 'renders', 't*.json')))}
    res = dict(config=dict(tau=TAU, k=K, token_measures=TOKEN_MEASURES, turn_extra=TURN_EXTRA, students=a.students,
                           sft=a.sft), b1={}, b1_tokens={}, b2={}, b3={}, summary={})
    teacher_tok = {}        # (student, cond) -> {traj: [per-turn token-measure dicts]}
    # ---------------- B1: teacher vs own samples ----------------
    for st in a.students:
        for cond in ('full_history', 'norefeed'):
            d = os.path.join(a.root, 'onpolicy', st, cond)
            if not os.path.isdir(d):
                continue
            per_traj, tok_pool, tt = {}, collections.defaultdict(lambda: ([], [])), {}
            lens = collections.defaultdict(list)
            for f in sorted(glob.glob(os.path.join(d, 't*.json.gz'))):
                o = json.load(gzip.open(f, 'rt'))
                rows, tlist = [], []
                for t in o['turns']:
                    ttm = token_measures(t['teacher'])
                    tlist.append(ttm)
                    tv = turn_measures(ttm, t['teacher']['cls'])
                    sv = [turn_measures(token_measures(s), s['cls']) for s in t['samples'] if s['ids']]
                    sv = [x for x in sv if x]
                    if tv and sv:
                        rows.append((tv, sv))
                    tc = np.frombuffer(t['teacher']['cls'].encode(), np.uint8) - 48
                    for c in CONTENT:
                        for k in TOKEN_MEASURES:
                            tok_pool[(c, k)][0].append(ttm[k][tc == c])
                    for s in t['samples']:
                        if not s['ids']:
                            continue
                        sm = token_measures(s)
                        scl = np.frombuffer(s['cls'].encode(), np.uint8) - 48
                        for c in CONTENT:
                            for k in TOKEN_MEASURES:
                                tok_pool[(c, k)][1].append(sm[k][scl == c])
                        lens['sample_tokens'].append(len(s['ids']))
                        lens['sample_capped'].append(s['finish'] == 'length')
                    lens['teacher_tokens'].append(len(t['teacher']['cls']))
                per_traj[o['traj_id']] = rows
                tt[o['traj_id']] = tlist
            teacher_tok[(st, cond)] = tt
            key = f'{st}/{cond}'
            measures = list(TOKEN_MEASURES) + list(TURN_EXTRA)

            def mean_auc(trajs, m):
                v = [auroc([tv[m]], [s[m] for s in sv]) for rows in trajs for tv, sv in rows]
                v = [x for x in v if not math.isnan(x)]
                return float(np.mean(v)) if v else float('nan')
            res['b1'][key] = {m: dict(auroc=round(mean_auc(list(per_traj.values()), m), 4),
                                      ci=boot_ci(per_traj, lambda trajs, m=m: mean_auc(trajs, m), n=300))
                              for m in measures}
            res['b1'][key]['_n_turns'] = sum(len(r) for r in per_traj.values())
            res['b1'][key]['_lengths'] = dict(teacher_median=float(np.median(lens['teacher_tokens'])),
                                              sample_median=float(np.median(lens['sample_tokens'])),
                                              sample_capped=float(np.mean(lens['sample_capped'])))
            tokr = {}
            for c in CONTENT:
                for k in TOKEN_MEASURES:
                    tp, sp = tok_pool[(c, k)]
                    tpv, spv = np.concatenate(tp), np.concatenate(sp)
                    if len(spv) > 400000:
                        spv = np.random.default_rng(0).choice(spv, 400000, replace=False)
                    tokr[f'{CLASSES[c]}/{k}'] = dict(auroc=round(auroc(tpv, spv), 4), teacher_mean=round(float(tpv.mean()), 4),
                                                     sample_mean=round(float(spv.mean()), 4), n_teacher=int(len(tpv)))
            res['b1_tokens'][key] = tokr
            print('B1', key, {m: res['b1'][key][m]['auroc'] for m in measures}, flush=True)
    # ---------------- B2: learnability under SFT ----------------
    base = teacher_tok.get(('dk0921', 'full_history'))
    sft = {s: load_scores(a.root, s, renders) for s in a.sft}
    nr = load_scores(a.root, 'dk0921_norefeed', renders)
    fullk5 = load_scores(a.root, 'dk0921', renders)
    xs = load_scores(a.root, 'rlvr1s12', renders)
    if base and all(sft.values()):
        rows = collections.defaultdict(list)     # flat per content token
        for tid, turns in base.items():
            rec = renders[tid]
            for k, (t, tm) in enumerate(zip(rec['turns'], turns)):
                cls = np.array([CLASSES.index(c) for c in t['cls']])
                m = np.isin(cls, CONTENT)
                n = len(cls)
                # Was the token an SFT target? render_msa.py cuts every earlier teacher turn's reasoning to
                # reasoning_cap.CAP_TOKENS (1,000) and masks reasoning that was cut; prose and tool calls are always
                # trained. So reasoning is trained only in the episode's last turn or when it is <= 1,000 tokens.
                n_reason = int((cls == 0).sum())
                reason_trained = k == len(rec['turns']) - 1 or n_reason <= 1000
                rows['trained'].append(((cls != 0) | reason_trained)[m])
                for key in TOKEN_MEASURES:
                    rows[key].append(tm[key][m])
                rows['dnr'].append((nr[tid][k] - fullk5[tid][k])[m])
                rows['xstudent'].append(((fullk5[tid][k] + xs[tid][k]) / 2)[m])
                rows['position'].append(-np.log1p(np.arange(n))[m])
                rows['cls'].append(cls[m])
                rows['traj'].append(np.array([int(tid[1:])] * int(m.sum())))
                rows['k5base'].append(fullk5[tid][k][m])
                for s in a.sft:
                    rows[s].append(sft[s][tid][k][m])
        F = {k: np.concatenate(v) for k, v in rows.items()}
        tr = F['trained'].astype(bool)
        control = dict(n=int((~tr).sum()), mean_nll={s: float(F[s][~tr].mean()) for s in ['k5base'] + a.sft},
                       share_of_content=float((~tr).mean()))
        F = {k: v[tr] for k, v in F.items()}          # learnability is only defined for tokens SFT trained on
        b0, bf = F['k5base'], F[a.sft[-1]]
        cats = dict(HH=float(((b0 >= TAU) & (bf >= TAU)).mean()), HL=float(((b0 >= TAU) & (bf < TAU)).mean()),
                    LH=float(((b0 < TAU) & (bf >= TAU)).mean()), LL=float(((b0 < TAU) & (bf < TAU)).mean()))
        mono = {s: float(np.mean(F[s])) for s in a.sft}
        fe = (F[a.sft[0]] < TAU) & (bf >= TAU) & (b0 >= TAU)   # learned at pass 1, hard again at pass 3
        hard = b0 >= TAU
        y_stay = bf[hard] >= TAU                                  # H->H among initially hard
        preds = list(TOKEN_MEASURES) + ['dnr', 'xstudent', 'position']
        b2 = dict(untrained_control=control, categories=cats, mean_nll_by_checkpoint=dict(base=float(b0.mean()), **mono),
                  forgetting_events=float(fe.mean()), n_tokens=int(len(b0)), n_hard=int(hard.sum()),
                  stay_hard_rate=float(y_stay.mean()), predict_stay_hard={}, spearman_residual={},
                  by_class={CLASSES[c]: dict(base=float(b0[F['cls'] == c].mean()),
                                             **{s: float(F[s][F['cls'] == c].mean()) for s in a.sft},
                                             stay_hard=float((bf[(F['cls'] == c) & hard] >= TAU).mean()))
                            for c in CONTENT})
        trajs = F['traj'][hard]
        for p in preds:
            x = F[p][hard]
            per_traj = {tr: (x[trajs == tr], y_stay[trajs == tr]) for tr in np.unique(trajs)}
            fn = lambda parts: auroc(np.concatenate([q[0][q[1]] for q in parts]), np.concatenate([q[0][~q[1]] for q in parts]))  # noqa: E731
            b2['predict_stay_hard'][p] = dict(auroc=round(fn(list(per_traj.values())), 4), ci=boot_ci(per_traj, fn, n=200))
            b2['spearman_residual'][p] = round(spearman(F[p], bf), 4)
            # base-matched: AUROC within deciles of base NLL (among initially hard), weighted by bin size, so a
            # measure gets no credit for merely restating how far above the threshold the token starts
            q = np.quantile(b0[hard], np.linspace(0, 1, 11))
            binid = np.clip(np.searchsorted(q, b0[hard], side='right') - 1, 0, 9)
            aucs, ws = [], []
            for bi in range(10):
                sel_ = binid == bi
                yy, xx = y_stay[sel_], x[sel_]
                if yy.sum() and (~yy).sum():
                    aucs.append(auroc(xx[yy], xx[~yy]))
                    ws.append(sel_.sum())
            b2.setdefault('predict_stay_hard_base_matched', {})[p] = round(float(np.average(aucs, weights=ws)), 4)
        res['b2'] = b2
        print('B2', json.dumps({p: v['auroc'] for p, v in b2['predict_stay_hard'].items()}), cats, flush=True)
        # windows for the judged benchmark: top tokens per measure, one per turn, plus ground-truth H->H
        win, seen = [], {}
        flat_index = []
        for tid, turns in base.items():
            rec = renders[tid]
            for k, t in enumerate(rec['turns']):
                cls = [CLASSES.index(c) for c in t['cls']]
                reason_trained = k == len(rec['turns']) - 1 or sum(c == 0 for c in cls) <= 1000
                for j, c in enumerate(cls):
                    if c in CONTENT and (c != 0 or reason_trained):
                        flat_index.append((tid, k, j))
        sel = {p: F[p] for p in preds}
        sel['sft_stays_hard'] = np.where(hard & (bf >= TAU), bf, -np.inf)
        for p, v in sel.items():
            order = np.argsort(-v)
            used_turns, picked = set(), 0
            for i in order:
                tid, k, j = flat_index[i]
                if (tid, k) in used_turns or not np.isfinite(v[i]):
                    continue
                used_turns.add((tid, k))
                rec = renders[tid]
                t = rec['turns'][k]
                pcs = rec['pieces'][t['tok_start']:t['tok_end']]
                wid = f'{tid}:{k}:{j}'
                near = next((w for w in seen if w[0] == tid and w[1] == k and abs(w[2] - j) <= 12), None)
                if near is None:
                    seen[(tid, k, j)] = dict(id=wid, traj=tid, turn=k, tok=j, before=''.join(pcs[max(0, j - 60):j])[-400:],
                                             token=pcs[j], after=''.join(pcs[j + 1:j + 25])[:200], methods=[p],
                                             cls=t['cls'][j])
                else:
                    seen[near]['methods'].append(p)
                picked += 1
                if picked >= 25:
                    break
        win = list(seen.values())
        random.Random(0).shuffle(win)
        with open(os.path.join(a.out, 'windows.jsonl'), 'w') as f:
            for w in win:
                f.write(json.dumps(w, ensure_ascii=False) + '\n')
        print('windows', len(win), flush=True)
    # ---------------- B3: stability across students and conditions ----------------
    pairs = [(('dk0921', 'full_history'), ('rlvr1s12', 'full_history')), (('dk0921', 'full_history'), ('dk0921', 'norefeed'))]
    for p1, p2 in pairs:
        A, B = teacher_tok.get(p1), teacher_tok.get(p2)
        if not A or not B:
            continue
        key = f'{p1[0]}/{p1[1]} vs {p2[0]}/{p2[1]}'
        common = sorted(set(A) & set(B))
        res['b3'][key] = {'_n_traj': len(common)}
        for m in TOKEN_MEASURES:
            xa = np.concatenate([tm[m] for tid in common for tm in A[tid]])
            xb = np.concatenate([tm[m] for tid in common for tm in B[tid]])
            res['b3'][key][m] = round(spearman(xa, xb), 4)
        print('B3', key, res['b3'][key], flush=True)
    json.dump(res, open(os.path.join(a.out, 'benchmark.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
