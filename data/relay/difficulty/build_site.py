#!/usr/bin/env python3
"""Cached scores -> the site's data files, plus flat per-turn / per-span tables for analysis outside the site.

Inputs: render_full.py's renders (checkpoint independent) and score.py's output under <scores>/<tag>/ for each
checkpoint tag. Every score file is joined to its render by token position and checked against the render's text hash,
so adding a checkpoint (e.g. an SFT step) is one more score directory with the same positions.

Outputs under --out:
  index.html, app.js, style.css           the static site (copied from site/)
  data/index.json                         trajectories, per-turn summaries per checkpoint, global span table, config
  data/traj/<id>.json                     one trajectory: messages, target tokens, per-checkpoint scores
  tables/turns.csv, tables/spans.csv      the same summaries flat
Spans (here and in the site): a target token is high when its NLL >= --tau; high tokens closer than --gap tokens join
one span (the tokens between them included). isolated = 1 high token, burst = 2-3, extended = 4+.

    python build_site.py --renders <dir> --scores <dir> --tags dk0921 rlvr1s12 --manifest sample.json --out <site dir>
"""
import argparse
import csv
import gzip
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = ('reasoning', 'prose', 'command', 'think_tag', 'call_fmt', 'eot')
CONTENT = {'reasoning', 'prose', 'command'}
LABELS = {'dk0921': '09-21', 'rlvr1s12': 'RLVR1'}
CONDITIONS = {'full_history': 'full history', 'norefeed': 'no-refeed'}


def label(tag, run):
    base = LABELS.get(tag.split('_')[0], tag.split('_')[0])
    return base + ('' if run['condition'] == 'full_history' else ' · ' + CONDITIONS.get(run['condition'], run['condition']))
EVAL_CONTEXT = 65536


def spans_of(nll, cls, tau, gap, content_only=True):
    """[(first, last)] index ranges (inclusive) of high-loss spans in one turn."""
    out, cur, last_hi = [], None, None
    for i, (x, c) in enumerate(zip(nll, cls)):
        if content_only and c not in CONTENT:
            continue
        if x >= tau:
            if cur is not None and i - last_hi <= gap:
                cur[1] = i
            else:
                if cur:
                    out.append(tuple(cur))
                cur = [i, i]
            last_hi = i
    if cur:
        out.append(tuple(cur))
    return out


def span_type(n_high):
    return 'isolated' if n_high == 1 else 'burst' if n_high <= 3 else 'extended'


def turn_summary(nll, cls, tau):
    def st(xs):
        if not xs:
            return dict(n=0, mean=None, high=None)
        return dict(n=len(xs), mean=round(sum(xs) / len(xs), 4), high=round(sum(x >= tau for x in xs) / len(xs), 4))
    content = [x for x, c in zip(nll, cls) if c in CONTENT]
    out = dict(all=st(nll), content=st(content),
               max=round(max(nll), 3), content_sum=round(sum(content), 2),
               by_class={c: st([x for x, k in zip(nll, cls) if k == c]) for c in CLASSES})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--renders', required=True)
    ap.add_argument('--scores', required=True)
    ap.add_argument('--tags', nargs='+', required=True, help='first = primary')
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tau', type=float, default=2.3, help='high-loss NLL (nats); 2.3 = p < 0.10')
    ap.add_argument('--gap', type=int, default=3)
    ap.add_argument('--max-spans', type=int, default=1500, help='per score set, largest total NLL first, in index.json')
    a = ap.parse_args()
    sys.path.insert(0, HERE)
    from render_full import load_student
    man = json.load(open(a.manifest))
    runs = {t: json.load(open(os.path.join(a.scores, t, 'run.json'))) for t in a.tags}
    tok, _, _ = load_student(json.load(open(os.path.join(a.renders, 'render_report.json')))['tokenizer_dir'])
    os.makedirs(os.path.join(a.out, 'data', 'traj'), exist_ok=True)
    os.makedirs(os.path.join(a.out, 'tables'), exist_ok=True)
    for f in ('index.html', 'app.js', 'style.css'):
        shutil.copy(os.path.join(HERE, 'site', f), os.path.join(a.out, f))
    index, turn_rows, span_rows = [], [], []
    for ep in man['episodes']:
        tid = ep['traj_id']
        rec = json.load(open(os.path.join(a.renders, tid + '.json')))
        scores = {}
        for t in a.tags:
            with gzip.open(os.path.join(a.scores, t, tid + '.json.gz'), 'rt') as f:
                s = json.load(f)
            assert runs[t]['episodes'][tid]['text_sha256'] == rec['text_sha256'], (t, tid, 'render changed')
            scores[t] = {p: (lp, rk, ti, tl) for p, lp, rk, ti, tl in
                         zip(s['positions'], s['lp'], s['rank'], s['top_ids'], s['top_lp'])}
        vocab_ids = set()
        turns, tsum = [], []
        for t in rec['turns']:
            pos = list(range(t['tok_start'], t['tok_end']))
            ids = rec['ids'][t['tok_start']:t['tok_end']]
            vocab_ids.update(ids)
            per = {}
            summ = {}
            for tag in a.tags:
                sc = [scores[tag][p] for p in pos]
                lp = [x[0] for x in sc]
                for x in sc:
                    vocab_ids.update(x[2])
                per[tag] = dict(lp=[round(x, 3) for x in lp], rk=[x[1] for x in sc], top=[x[2] for x in sc],
                                tlp=[[round(v, 2) for v in x[3]] for x in sc])
                nll = [-x for x in lp]
                summ[tag] = turn_summary(nll, t['cls'], a.tau)
                spans = spans_of(nll, t['cls'], a.tau, a.gap)
                summ[tag]['n_spans'] = len(spans)
                for s0, s1 in spans:
                    seg = list(range(s0, s1 + 1))
                    n_high = sum(nll[i] >= a.tau and t['cls'][i] in CONTENT for i in seg)
                    ccount = {}
                    for i in seg:
                        ccount[t['cls'][i]] = ccount.get(t['cls'][i], 0) + 1
                    pieces = rec['pieces'][t['tok_start']:t['tok_end']]
                    span_rows.append(dict(
                        traj_id=tid, tag=tag, turn=t['k'], first=s0, last=s1, abs_pos=t['tok_start'] + s0,
                        n_tokens=len(seg), n_high=n_high, type=span_type(n_high),
                        sum_nll=round(sum(nll[i] for i in seg), 3), max_nll=round(max(nll[i] for i in seg), 3),
                        cls=max(ccount, key=ccount.get), text=''.join(pieces[s0:s1 + 1]),
                        before=''.join(pieces[max(0, s0 - 40):s0])[-200:], after=''.join(pieces[s1 + 1:s1 + 20])[:100],
                        beyond_eval_context=t['tok_start'] + s0 >= EVAL_CONTEXT))
                turn_rows.append(dict(traj_id=tid, tag=tag, turn=t['k'], n_targets=t['n_targets'],
                                      history_tokens=t['history_tokens'], n_calls=t['n_calls'],
                                      mean_nll=summ[tag]['all']['mean'], content_mean_nll=summ[tag]['content']['mean'],
                                      content_high_frac=summ[tag]['content']['high'], max_nll=summ[tag]['max'],
                                      content_sum_nll=summ[tag]['content_sum'], n_spans=len(spans),
                                      **{f'{c}_mean': summ[tag]['by_class'][c]['mean'] for c in CLASSES},
                                      **{f'{c}_n': summ[tag]['by_class'][c]['n'] for c in CLASSES}))
            turns.append(dict(k=t['k'], msg=t['msg'], tok_start=t['tok_start'], n_calls=t['n_calls'],
                              history_tokens=t['history_tokens'], has_reasoning=t['has_reasoning'],
                              ids=ids, pieces=rec['pieces'][t['tok_start']:t['tok_end']],
                              cls=''.join(str(CLASSES.index(c)) for c in t['cls']),
                              mixed=[i for i, m in enumerate(t['mixed']) if m], s=per))
            tsum.append(summ)
        msgs = []
        for m in rec['messages']:
            d = dict(i=m['i'], role=m['role'], tok_start=m['tok_start'], tok_end=m['tok_end'])
            if m['role'] != 'assistant':
                d['text'] = rec['text'][m['start']:m['end']]
            else:
                d['turn'] = next(t['k'] for t in rec['turns'] if t['msg'] == m['i'])
            msgs.append(d)
        vocab = {i: [tok.decode([i], skip_special_tokens=False), tok.id_to_token(i)] for i in sorted(vocab_ids)}
        meta = {k: ep.get(k) for k in ('traj_id', 'task', 'domain', 'generator', 'sid', 'run', 'trial', 'reward',
                                       'passed', 'teacher_turns', 'msa_trajectory', 'harbor_trajectory', 'trial_dir',
                                       'render_tokens', 'sft_render_tokens', 'msa_trajectory_sha256')}
        meta['task_meta'] = ep.get('task_meta') or {}
        traj = dict(meta=meta, n_tokens=rec['n_tokens'], n_targets=rec['n_targets'], messages=msgs, turns=turns,
                    vocab=vocab)
        with open(os.path.join(a.out, 'data', 'traj', tid + '.json'), 'w') as f:
            json.dump(traj, f, separators=(',', ':'), ensure_ascii=False)
        ep_sum = {}
        for tag in a.tags:
            allx = [-x for t in turns for x in t['s'][tag]['lp']]
            cls_all = [CLASSES[int(c)] for t in turns for c in t['cls']]
            ep_sum[tag] = turn_summary(allx, cls_all, a.tau)
            ep_sum[tag]['n_spans'] = sum(s[tag]['n_spans'] for s in tsum)
        index.append(dict(traj_id=tid, task=ep['task'], domain=ep['domain'], generator=ep.get('generator'),
                          difficulty=(ep.get('task_meta') or {}).get('difficulty'),
                          tags=(ep.get('task_meta') or {}).get('tags'), reward=ep.get('reward'),
                          n_tokens=rec['n_tokens'], n_targets=rec['n_targets'], n_turns=len(turns),
                          summary=ep_sum, turns=[dict(k=t['k'], n=len(t['ids']), history_tokens=t['history_tokens'],
                                                      s={tag: tsum[t['k']][tag] for tag in a.tags}) for t in turns]))
    span_rows.sort(key=lambda r: -r['sum_nll'])
    config = dict(tags=[dict(tag=t, label=label(t, runs[t]), condition=runs[t]['condition'], model_path=runs[t]['model_path'],
                             model_id=runs[t].get('model_id'), revision=runs[t].get('revision'),
                             vllm_version=runs[t].get('vllm_version'), serve_args=runs[t].get('serve_args'),
                             prompt_logprobs_k=runs[t]['prompt_logprobs_k'], checks=runs[t].get('checks'),
                             slurm_job=runs[t].get('slurm_job')) for t in a.tags],
                  classes=CLASSES, content=sorted(CONTENT), tau=a.tau, gap=a.gap, eval_context=EVAL_CONTEXT,
                  conditions=dict(full_history='one pass over the episode: every earlier turn with its reasoning, '
                                               'as the teacher saw it (collection ran interleaved_thinking true)',
                                  norefeed='one request per turn: earlier assistant turns without their reasoning, '
                                           'the student\'s MSA2 eval view (interleaved_thinking false)'),
                  sample=dict(pool=man['pool'], pool_rows=man['pool_rows'], pool_tasks=man['pool_tasks'],
                              seed=man['seed'], method=man['method'], exclusions=man['exclusions'],
                              teacher=man['teacher'], harness=man['harness']))
    json.dump(dict(config=config, trajectories=index,
                   spans=[r for t in a.tags for r in [x for x in span_rows if x['tag'] == t][:a.max_spans]]),
              open(os.path.join(a.out, 'data', 'index.json'), 'w'), separators=(',', ':'), ensure_ascii=False)
    for name, rows in (('turns', turn_rows), ('spans', span_rows)):
        with open(os.path.join(a.out, 'tables', name + '.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(json.dumps(dict(trajectories=len(index), turns=len(turn_rows), spans=len(span_rows),
                          mean_content_nll={t: round(sum(i['summary'][t]['content']['mean'] * i['summary'][t]['content']['n']
                                                         for i in index) / sum(i['summary'][t]['content']['n'] for i in index), 4)
                                            for t in a.tags})))


if __name__ == '__main__':
    main()
