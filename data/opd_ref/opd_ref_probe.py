"""Which weak reference makes a CompassOPD signal point at decisions? A scoring-only test, no training.

A strong Qwen teacher T, weak same-family references R (Qwen3.5 small models) and the student S (Snowball) score the
same Snowball rollouts. Each assistant reply is cut into chunks both tokenizers can score (common byte boundaries,
"Breaking the Tokenizer Barrier"), and each chunk gets the per-chunk log-likelihood of every model. Signals at the
student's initial policy (CompassOPD's anchor term is zero there):
    teacher      lT                      the teacher alone
    opd          lT - lS                 plain OPD
    compass_<R>  lT - lR                 CompassOPD with reference R
    student      lS                      control: the student's own confidence
The test: does the signal rank a task's passing rollout above its failing one (within task), and does it score the
done-claim of a passing rollout above that of a failing one?

    prep     trajectories -> Snowball and Qwen token ids + aligned chunks (CPU)
    score    one model's log-probability of every prompt token (vLLM, prompt_logprobs=0, GPU)
    analyze  chunks x model scores -> metrics.json + report.md (CPU)
"""
import argparse
import glob
import json
import os
import pickle
import re
import sys
import time

import numpy as np

START_THINK, END_THINK = '<|start_think|>', '<|end_think|>'
Q_START_THINK, Q_END_THINK = '<think>', '</think>'
INFRA_EXC = {'ReadError', 'APIConnectionError', 'TmuxBatchProtocolError', 'TmuxCommandError', 'TmuxSessionEndedError',
             'VerifierTimeoutError', 'EnvironmentStartTimeoutError', 'DaytonaBadGatewayError', 'DaytonaSandboxStopError',
             'VerifierRuntimeError', 'AddTestsDirError'}
DONE_RE = re.compile(r'"task_complete"\s*:\s*(true)')


# ==== prep ==========================================================================================================
def load_trial(tdir):
    """One Harbor trial -> dict(task, trial, reward, exc, messages) or None (infra / unreadable)."""
    try:
        res = json.load(open(os.path.join(tdir, 'result.json')))
    except Exception:
        return None
    exc = ((res.get('exception_info') or {}) or {}).get('exception_type') or ''
    reward = ((res.get('verifier_result') or {}).get('rewards') or {}).get('reward')
    if exc in INFRA_EXC or reward is None:
        return None
    paths = sorted(glob.glob(os.path.join(tdir, 'attempts', '*', 'agent', 'trajectory.json')))
    if not paths:
        return None
    traj = json.load(open(paths[0]))
    msgs = []
    for s in traj['steps']:
        src = s.get('source')
        if src == 'agent':
            msgs.append(dict(role='assistant', content=s.get('message') or ''))
            obs = s.get('observation') or {}
            msgs.append(dict(role='user', content='\n'.join(
                r['content'] for r in obs.get('results', []) if isinstance(r.get('content'), str))))
        elif src in ('user', 'system'):
            msgs.append(dict(role=src, content=s.get('message') or ''))
    while msgs and msgs[-1]['role'] != 'assistant':
        msgs.pop()
    if not any(m['role'] == 'assistant' for m in msgs):
        return None
    name = os.path.basename(tdir.rstrip('/'))
    return dict(task=name.rsplit('__', 1)[0], trial=name, reward=float(reward), exc=exc, messages=msgs)


def split_reply(c):
    """Assistant content (trimmed) -> [(kind, text)], markers excluded. kind = think | action."""
    if c.startswith(START_THINK) and END_THINK in c:
        body, _, rest = c[len(START_THINK):].partition(END_THINK)
        return [('marker', START_THINK), ('think', body), ('marker', END_THINK), ('action', rest)]
    return [('action', c)]


def qwen_render(messages):
    """Qwen3.5-family chat format by hand (their template would drop earlier thinking); Snowball's think markers
    become <think> / </think>, which reproduces Qwen's own '<think>\\n...\\n</think>\\n\\n' layout."""
    parts, spans = [], []          # spans: per assistant message, [(kind, start, end)] in the rendered string
    pos = 0

    def add(s):
        nonlocal pos
        parts.append(s)
        pos += len(s)

    for m in messages:
        add('<|im_start|>%s\n' % m['role'])
        c = m['content'].strip()
        if m['role'] == 'assistant':
            segs = []
            for kind, text in split_reply(c):
                if kind == 'marker':
                    add(Q_START_THINK if text == START_THINK else Q_END_THINK)
                else:
                    segs.append((kind, pos, pos + len(text)))
                    add(text)
            spans.append(segs)
        else:
            add(c)
        add('<|im_end|>\n')
    return ''.join(parts), spans


def snowball_render(tok, messages):
    """Snowball's own chat template (as served), then each assistant reply located verbatim in the rendered string."""
    s = tok.apply_chat_template(messages, tokenize=False)
    spans, pos = [], 0
    for m in messages:
        if m['role'] != 'assistant':
            continue
        c = m['content'].strip()
        i = s.find(c, pos)
        if i < 0:
            raise ValueError('assistant reply not found verbatim in the rendered template')
        segs, off = [], i
        for kind, text in split_reply(c):
            if kind != 'marker':
                segs.append((kind, off, off + len(text)))
            off += len(text)
        spans.append(segs)
        pos = i + len(c)
    return s, spans


def encode(tok, s):
    enc = tok(s, add_special_tokens=False, return_offsets_mapping=True)
    return np.asarray(enc['input_ids'], dtype=np.int64), np.asarray(enc['offset_mapping'], dtype=np.int64)


def seg_tokens(offs, a, b):
    """Token indices covering [a, b) and their end offsets relative to a; None if the segment's edges are not token
    boundaries (then no model can score exactly that text)."""
    ends, starts = offs[:, 1], offs[:, 0]
    idx = np.nonzero((ends > a) & (starts < b))[0]
    if len(idx) == 0 or starts[idx[0]] != a or ends[idx[-1]] != b:
        return None
    return idx, ends[idx] - a


def align(s_idx, s_end, q_idx, q_end):
    """Common token boundaries -> chunks [(s_lo, s_hi, q_lo, q_hi, char_lo, char_hi)] (token index ranges are
    half-open, chars relative to the segment start)."""
    common = np.intersect1d(s_end, q_end)
    chunks = []
    si = qi = 0
    lo = 0
    for c in common:
        sj = si
        while sj < len(s_end) and s_end[sj] <= c:
            sj += 1
        qj = qi
        while qj < len(q_end) and q_end[qj] <= c:
            qj += 1
        if sj > si and qj > qi:
            chunks.append((s_idx[si], s_idx[sj - 1] + 1, q_idx[qi], q_idx[qj - 1] + 1, lo, int(c)))
            si, qi, lo = sj, qj, int(c)
    return chunks


def prep_episode(tr, stok, qtok, max_s_len):
    msgs = list(tr['messages'])
    while True:
        s_str, s_spans = snowball_render(stok, msgs)
        s_ids, s_offs = encode(stok, s_str)
        if len(s_ids) <= max_s_len:
            break
        # drop the last assistant turn (and the user turn before it is kept only if an assistant follows)
        k = max(i for i, m in enumerate(msgs) if m['role'] == 'assistant')
        msgs = msgs[:k]
        while msgs and msgs[-1]['role'] != 'assistant':
            msgs.pop()
        if not msgs:
            return None, 'too_long'
    q_str, q_spans = qwen_render(msgs)
    q_ids, q_offs = encode(qtok, q_str)
    chunks, stats = [], dict(segments=0, invalid=0)
    n_turns = len(s_spans)
    done_turns = []
    for t, (ss, qs) in enumerate(zip(s_spans, q_spans)):
        assert [k for k, _, _ in ss] == [k for k, _, _ in qs]
        for (kind, a, b), (_, qa, qb) in zip(ss, qs):
            if b <= a:
                continue
            text = s_str[a:b]
            assert text == q_str[qa:qb]
            stats['segments'] += 1
            st = seg_tokens(s_offs, a, b)
            qt = seg_tokens(q_offs, qa, qb)
            if st is None or qt is None:
                stats['invalid'] += 1
                continue
            done = None
            if kind == 'action':
                mm = list(DONE_RE.finditer(text))
                if mm:
                    done = mm[-1].span(1)
                    done_turns.append(t)
            for (sl, sh, ql, qh, cl, ch) in align(st[0], st[1], qt[0], qt[1]):
                is_done = done is not None and cl < done[1] and ch > done[0]
                chunks.append((t, 0 if kind == 'think' else 1, sl, sh, ql, qh, cl, ch, int(is_done)))
    ep = dict(task=tr['task'], trial=tr['trial'], reward=tr['reward'], exc=tr['exc'], n_turns=n_turns,
              n_turns_orig=sum(m['role'] == 'assistant' for m in tr['messages']), done_turns=done_turns,
              s_ids=s_ids.astype(np.int32), q_ids=q_ids.astype(np.int32),
              chunks=np.asarray(chunks, dtype=np.int32).reshape(-1, 9), stats=stats,
              s_text=s_str, s_offs=s_offs.astype(np.int32))
    return ep, None


def cmd_prep(a):
    from transformers import AutoTokenizer
    stok = AutoTokenizer.from_pretrained(a.snowball_tok)
    qtok = AutoTokenizer.from_pretrained(a.qwen_tok)
    trials = sorted(d for d in glob.glob(os.path.join(a.session, '*__*')) if os.path.isdir(d))
    if a.limit:
        trials = trials[:a.limit]
    eps, skipped = [], {}
    for i, d in enumerate(trials):
        tr = load_trial(d)
        if tr is None:
            skipped['infra_or_unreadable'] = skipped.get('infra_or_unreadable', 0) + 1
            continue
        try:
            ep, why = prep_episode(tr, stok, qtok, a.max_s_len)
        except Exception as ex:
            ep, why = None, 'error:%s' % repr(ex)[:80]
        if ep is None:
            skipped[why] = skipped.get(why, 0) + 1
            continue
        eps.append(ep)
        if i % 50 == 0:
            print('prep', i, len(trials), flush=True)
    os.makedirs(a.out, exist_ok=True)
    pickle.dump(eps, open(os.path.join(a.out, 'episodes.pkl'), 'wb'))
    ch = np.concatenate([e['chunks'] for e in eps])
    ns, nq = ch[:, 3] - ch[:, 2], ch[:, 5] - ch[:, 4]
    seg = sum(e['stats']['segments'] for e in eps)
    inv = sum(e['stats']['invalid'] for e in eps)
    summ = dict(episodes=len(eps), skipped=skipped, pass_=sum(e['reward'] > 0 for e in eps),
                truncated=sum(e['n_turns'] < e['n_turns_orig'] for e in eps),
                snowball_tokens=int(sum(len(e['s_ids']) for e in eps)), qwen_tokens=int(sum(len(e['q_ids']) for e in eps)),
                scored_student_tokens=int(ns.sum()), chunks=int(len(ch)), chunks_1to1=float(np.mean((ns == 1) & (nq == 1))),
                student_tokens_in_1to1=float(ns[(ns == 1) & (nq == 1)].sum() / ns.sum()),
                segments=seg, invalid_segments=inv, invalid_rate=inv / max(seg, 1))
    json.dump(summ, open(os.path.join(a.out, 'prep_summary.json'), 'w'), indent=1)
    print(json.dumps(summ, indent=1))


# ==== score =========================================================================================================
def cmd_score(a):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    eps = pickle.load(open(os.path.join(a.out, 'episodes.pkl'), 'rb'))
    key = 's_ids' if a.family == 'snowball' else 'q_ids'
    mine = list(range(a.shard, len(eps), a.nshards))
    out = os.path.join(a.out, 'scores', '%s.%d.pkl' % (a.name, a.shard))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    kw: dict = dict(model=a.model, tensor_parallel_size=a.tp, max_model_len=a.max_len, gpu_memory_utilization=0.85,
              enable_prefix_caching=False, max_num_batched_tokens=a.batched_tokens, max_num_seqs=a.max_seqs,
              seed=0)
    if a.family == 'snowball':
        kw.update(hf_overrides={'max_position_embeddings': a.max_len, 'max_seq_len': a.max_len},
                  enable_expert_parallel=a.tp > 1)
    else:
        kw.update(limit_mm_per_prompt={'image': 0, 'video': 0})
    t0 = time.time()
    llm = LLM(**kw)
    t_load = time.time() - t0
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0, detokenize=False)
    res, ntok, t_score = {}, 0, 0.0
    B = 32
    for b in range(0, len(mine), B):
        idx = mine[b:b + B]
        prompts = [TokensPrompt(prompt_token_ids=eps[i][key].tolist()) for i in idx]
        t1 = time.time()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        t_score += time.time() - t1
        for i, o in zip(idx, outs):
            ids = eps[i][key]
            lp = np.full(len(ids), np.nan, dtype=np.float32)
            for j, d in enumerate(o.prompt_logprobs):
                if d is None:
                    continue
                lp[j] = d[int(ids[j])].logprob
            res[i] = lp
            ntok += len(ids)
        print('%s shard %d: %d/%d episodes, %.0f tok/s' % (a.name, a.shard, len(res), len(mine), ntok / t_score),
              flush=True)
    pickle.dump(dict(scores=res, tokens=ntok, t_load=t_load, t_score=t_score), open(out, 'wb'))
    print('%s shard %d done: %d tokens, load %.0f s, score %.0f s' % (a.name, a.shard, ntok, t_load, t_score))


# ==== analyze =======================================================================================================
def load_scores(outdir, name):
    res, tok, t = {}, 0, 0.0
    for p in sorted(glob.glob(os.path.join(outdir, 'scores', '%s.*.pkl' % name))):
        d = pickle.load(open(p, 'rb'))
        res.update(d['scores'])
        tok += d['tokens']
        t = max(t, d['t_score'])
    return res, tok, t


def paired_auc(vals, labels, tasks, rng=None, n_boot=2000):
    """Within-task AUC: over tasks with both outcomes, P(score of a pass > score of a fail) (ties 1/2), averaged over
    tasks; bootstrap over tasks for a 95 % CI."""
    by = {}
    for v, l, t in zip(vals, labels, tasks):
        if np.isfinite(v):
            by.setdefault(t, ([], []))[0 if l else 1].append(v)
    per = []
    for t, (p, f) in by.items():
        if p and f:
            p, f = np.asarray(p)[:, None], np.asarray(f)[None, :]
            per.append(float(np.mean((p > f) + 0.5 * (p == f))))
    per = np.asarray(per)
    if len(per) == 0:
        return dict(auc=None, n_tasks=0)
    rng = rng or np.random.default_rng(0)
    boots = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(n_boot)]
    return dict(auc=float(per.mean()), lo=float(np.percentile(boots, 2.5)), hi=float(np.percentile(boots, 97.5)),
                n_tasks=int(len(per)))


def pooled_auc(vals, labels):
    v = np.asarray(vals, dtype=float)
    l = np.asarray(labels, dtype=bool)
    ok = np.isfinite(v)
    v, l = v[ok], l[ok]
    p, f = v[l][:, None], v[~l][None, :]
    if p.size == 0 or f.size == 0:
        return dict(auc=None, n_pass=int(p.size), n_fail=int(f.size))
    return dict(auc=float(np.mean((p > f) + 0.5 * (p == f))), n_pass=int(p.size), n_fail=int(f.size))


def cmd_analyze(a):
    eps = pickle.load(open(os.path.join(a.out, 'episodes.pkl'), 'rb'))
    refs = a.refs.split(',') if a.refs else []
    models = ['teacher', 'student'] + refs
    sc, perf = {}, {}
    for m in models:
        sc[m], tok, t = load_scores(a.out, m)
        perf[m] = dict(tokens=tok, score_seconds_max_shard=round(t, 1))
    keep = [i for i in range(len(eps)) if all(i in sc[m] for m in models)]
    print('episodes with every score:', len(keep), 'of', len(eps))

    signals = ['teacher', 'opd', 'student'] + ['compass_' + r for r in refs]
    rows = []   # per episode: dict of per-signal aggregates
    samples = {s: [] for s in signals}
    for i in keep:
        e = eps[i]
        ch = e['chunks']
        if len(ch) == 0:
            continue
        t_idx, kind, sl, sh, ql, qh, cl, chh, is_done = [ch[:, k] for k in range(9)]
        cs = np.cumsum(np.concatenate([[0.0], np.nan_to_num(sc['student'][i])]))
        lS = cs[sh] - cs[sl]
        lq = {}
        for m in ['teacher'] + refs:
            cq = np.cumsum(np.concatenate([[0.0], np.nan_to_num(sc[m][i])]))
            lq[m] = cq[qh] - cq[ql]
        sig = dict(teacher=lq['teacher'], opd=lq['teacher'] - lS, student=lS)
        for r in refs:
            sig['compass_' + r] = lq['teacher'] - lq[r]
        ntok = (sh - sl).astype(float)
        row: dict = dict(i=i, task=e['task'], passed=e['reward'] > 0, n_turns=e['n_turns'], claimed=bool(e['done_turns']))
        first_done = min(e['done_turns']) if e['done_turns'] else None
        for s, v in sig.items():
            for kname, mask in (('all', np.ones_like(kind, bool)), ('think', kind == 0), ('action', kind == 1)):
                row['%s/%s' % (s, kname)] = float(v[mask].sum() / ntok[mask].sum()) if mask.any() else np.nan
            if first_done is not None:
                dm = (is_done == 1) & (t_idx == first_done)
                row['%s/done_token' % s] = float(v[dm].sum()) if dm.any() else np.nan
                tm = (t_idx == first_done) & (kind == 1)
                row['%s/done_turn' % s] = float(v[tm].sum() / ntok[tm].sum()) if tm.any() else np.nan
            # most negative chunks, for reading what each signal punishes
            j = np.argsort(v)[:3]
            for jj in j:
                a0 = int(e['s_offs'][sl[jj]][0])
                a1 = int(e['s_offs'][sh[jj] - 1][1])
                samples[s].append((float(v[jj]), int(kind[jj]), e['s_text'][max(0, a0 - 60):a0], e['s_text'][a0:a1]))
        # where the signal mass sits
        for s, v in sig.items():
            row['%s/abs_think_share' % s] = float(np.abs(v[kind == 0]).sum() / max(np.abs(v).sum(), 1e-9))
        rows.append(row)

    labels = [r['passed'] for r in rows]
    tasks = [r['task'] for r in rows]
    metrics: dict = dict(n_episodes=len(rows), n_pass=int(sum(labels)), perf=perf, signals={})
    for s in signals:
        m: dict = {}
        for view in ('all', 'think', 'action'):
            m['within_task_' + view] = paired_auc([r['%s/%s' % (s, view)] for r in rows], labels, tasks)
        cl = [r for r in rows if r['claimed']]
        for view in ('done_token', 'done_turn'):
            m['claim_%s_pooled' % view] = pooled_auc([r.get('%s/%s' % (s, view), np.nan) for r in cl],
                                                     [r['passed'] for r in cl])
            m['claim_%s_within' % view] = paired_auc([r.get('%s/%s' % (s, view), np.nan) for r in cl],
                                                     [r['passed'] for r in cl], [r['task'] for r in cl])
        m['abs_think_share'] = float(np.mean([r['%s/abs_think_share' % s] for r in rows]))
        m['mean_per_token'] = float(np.nanmean([r['%s/all' % s] for r in rows]))
        metrics['signals'][s] = m
    # agreement between the signals, chunk by chunk (sign agreement of teacher-vs-ref and opd)
    json.dump(metrics, open(os.path.join(a.out, 'metrics.json'), 'w'), indent=1)
    pickle.dump(rows, open(os.path.join(a.out, 'rows.pkl'), 'wb'))

    def f(d):
        if not d or d.get('auc') is None:
            return '–'
        return '%.3f' % d['auc'] + (' [%.2f, %.2f]' % (d['lo'], d['hi']) if 'lo' in d else '')
    lines = ['# opd_ref probe', '', '%d episodes (%d pass), %d with a done-claim' % (
        len(rows), sum(labels), sum(r['claimed'] for r in rows)), '',
        '| signal | within-task all | think | action | claim token (pooled) | claim token (within) | claim turn (pooled) | |signal| on think |',
        '|---|---|---|---|---|---|---|---|']
    for s in signals:
        m = metrics['signals'][s]
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %.2f |' % (
            s, f(m['within_task_all']), f(m['within_task_think']), f(m['within_task_action']),
            f(m['claim_done_token_pooled']), f(m['claim_done_token_within']), f(m['claim_done_turn_pooled']),
            m['abs_think_share']))
    n_tasks = metrics['signals']['teacher']['within_task_all'].get('n_tasks')
    lines += ['', 'within-task pairs over %s tasks with both outcomes; claim = first task_complete:true' % n_tasks, '',
              '## scoring cost', '']
    for m, p in perf.items():
        lines.append('- %s: %d tokens, slowest shard %.0f s' % (m, p['tokens'], p['score_seconds_max_shard']))
    lines += ['', '## most negative chunks per signal (sample)', '']
    rng = np.random.default_rng(0)
    for s in signals:
        lines.append('### ' + s)
        smp = sorted(samples[s])[:400]
        for jj in rng.choice(len(smp), size=min(12, len(smp)), replace=False):
            v, k, ctx, txt = smp[jj]
            lines.append('- %.1f [%s] …%s⟦%s⟧' % (v, 'think' if k == 0 else 'action',
                                                    ctx.replace('\n', '⏎')[-50:], txt.replace('\n', '⏎')[:80]))
        lines.append('')
    open(os.path.join(a.out, 'report.md'), 'w').write('\n'.join(lines))
    print('\n'.join(lines[:12 + len(signals)]))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('prep')
    q.add_argument('--session', required=True)
    q.add_argument('--snowball-tok', required=True)
    q.add_argument('--qwen-tok', required=True)
    q.add_argument('--out', required=True)
    q.add_argument('--max-s-len', type=int, default=65535)
    q.add_argument('--limit', type=int, default=0)
    q = sub.add_parser('score')
    q.add_argument('--out', required=True)
    q.add_argument('--name', required=True)
    q.add_argument('--model', required=True)
    q.add_argument('--family', choices=['snowball', 'qwen'], required=True)
    q.add_argument('--tp', type=int, default=1)
    q.add_argument('--shard', type=int, default=0)
    q.add_argument('--nshards', type=int, default=1)
    q.add_argument('--max-len', type=int, default=131072)
    q.add_argument('--batched-tokens', type=int, default=8192)
    q.add_argument('--max-seqs', type=int, default=8)
    q = sub.add_parser('analyze')
    q.add_argument('--out', required=True)
    q.add_argument('--refs', default='')
    a = p.parse_args()
    dict(prep=cmd_prep, score=cmd_score, analyze=cmd_analyze)[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
