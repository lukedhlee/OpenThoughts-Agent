#!/usr/bin/env python3
"""On-policy baseline: the student's own next turn at each teacher turn's exact history, next to the teacher's turn,
with the same per-token statistics for both.

For every assistant turn of every rendered episode (render_full.py), under one history condition:
  teacher  its body re-scored with prompt_logprobs K (full_history: one pass per episode; norefeed: one per turn)
  samples  N student replies to the same prompt (temperature 1, top_p 1, no draft = the student's policy), with the
           generation path's logprobs K for every sampled token
Per token both carry the same derived statistics, all from the top-K (default 20):
  nll        -log p(token)
  gap        log p_top1 - log p(token)                         (0 when the token is the student's argmax)
  above      probability mass on tokens strictly more likely than this one (top-K only; rank > K: the top-K mass)
             -> surprise p-value = 1 - above: on the student's own samples it is ~uniform (probability integral
             transform), so "teacher tokens with p < 0.01" has a known 1 % false-positive rate under the student
  h_lb       entropy of the top-K plus the tail mass as one token: a lower bound on the student's entropy here
  m2         E[(log p)^2] over the same support: Var[log p] = m2 - h_lb^2, for the standardized log-prob
  rank       1..K, K+1 = outside the top-K
Sampled-token logprobs come from the decode path and teacher logprobs from the prefill path; `--path-check` re-scores
some samples through the prefill path to measure the offset between the two (compared in analysis, not hidden).

    python onpolicy.py --renders <dir> --out <dir> --condition full_history|norefeed [--n 8] [--k 20] [--only t01 ...]
writes <out>/<traj_id>.json.gz (per turn: teacher stats, sample stats + classes + finish reasons) and <out>/run.json.
"""
import argparse
import concurrent.futures as cf
import gzip
import hashlib
import json
import math
import os
import re
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_full import CLASSES, load_student, norefeed_prompts, render  # noqa: E402
from score import get, post  # noqa: E402

MAX_REPLY = 16384           # the MSA2 eval's explicit reply cap
CALL_RE = re.compile(r'<tool_call>\n(.*?)\n</tool_call>', re.S)


def byte_decoder():
    """GPT-2 byte-level BPE: token-string character -> byte."""
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


class Pieces:
    """Exact per-token text of a token-id sequence (byte-split characters go to the token that completes them)."""

    def __init__(self, tokdir):
        tj = json.load(open(os.path.join(tokdir, 'tokenizer.json')))
        self.special = {t['id']: t['content'] for t in tj.get('added_tokens', [])}
        self.vocab = {i: s for s, i in tj['model']['vocab'].items()}
        self.bd = byte_decoder()

    def token_bytes(self, i):
        if i in self.special:
            return self.special[i].encode()
        return bytes(self.bd[c] for c in self.vocab[i])

    def split(self, ids):
        out, buf = [], b''
        for i in ids:
            buf += self.token_bytes(i)
            try:
                out.append(buf.decode('utf-8'))
                buf = b''
            except UnicodeDecodeError:
                out.append('')         # incomplete character: emitted with the token that completes it
        if buf:
            out[-1] += buf.decode('utf-8', 'replace')
        return out


def classify(pieces):
    """Classes of a reply's tokens, as render_full.py classes the teacher's: think span -> reasoning (all of it if
    <|end_think|> never comes), tool calls -> call_fmt with the command string -> command, the rest prose."""
    text = ''.join(pieces)
    cls = ['prose'] * len(text)
    pos = 0
    if text.startswith('<|start_think|>'):
        e = text.find('<|end_think|>')
        end = len(text) if e < 0 else e
        cls[:len('<|start_think|>')] = ['think_tag'] * len('<|start_think|>')
        cls[len('<|start_think|>'):end] = ['reasoning'] * (end - len('<|start_think|>'))
        if e >= 0:
            cls[e:e + len('<|end_think|>')] = ['think_tag'] * len('<|end_think|>')
            pos = e + len('<|end_think|>')
        else:
            pos = len(text)
    for m in CALL_RE.finditer(text, pos):
        cls[m.start():m.end()] = ['call_fmt'] * (m.end() - m.start())
        body, b0 = m.group(1), m.start(1)
        k = body.find('"command": "')
        if k >= 0:
            s = k + len('"command": "')
            j, esc = s, False
            while j < len(body):
                if esc:
                    esc = False
                elif body[j] == '\\':
                    esc = True
                elif body[j] == '"':
                    break
                j += 1
            cls[b0 + s:b0 + j] = ['command'] * (j - s)
    for tag in ('<|eot_id|>', '<|end_of_text|>'):
        for m in re.finditer(re.escape(tag), text):
            cls[m.start():m.end()] = ['eot'] * len(tag)
    out, c = [], 0
    for p in pieces:
        span = cls[c:c + len(p)] if p else cls[min(c, len(cls) - 1):min(c, len(cls) - 1) + 1]
        out.append(max(set(span), key=span.count) if span else 'prose')
        c += len(p)
    return out


def stats(lp, top):
    """Derived per-token statistics from the token's logprob and the top-K [(id, logprob)] sorted by logprob."""
    ps = [math.exp(x) for _, x in top]
    mass = sum(ps)
    tail = max(0.0, 1.0 - mass)
    above = sum(p for (_, x), p in zip(top, ps) if x > lp + 1e-9)
    in_top = any(abs(x - lp) <= 1e-9 for _, x in top) or lp >= top[-1][1]
    rank = 1 + sum(1 for _, x in top if x > lp + 1e-9) if in_top else len(top) + 1
    if not in_top:
        above = mass
    h_lb = -sum(p * math.log(p) for p in ps if p > 0) - (tail * math.log(tail) if tail > 0 else 0.0)
    # second moment of log p under the student (top-K + tail as one token): with h_lb gives Var[log p], for the
    # standardized log-prob z = (log p(token) + H) / sd (Fast-DetectGPT's analytic curvature, arXiv 2310.05130)
    m2 = sum(p * math.log(p) ** 2 for p in ps if p > 0) + (tail * math.log(tail) ** 2 if tail > 0 else 0.0)
    return dict(nll=-lp, gap=top[0][1] - lp, above=above, h_lb=h_lb, m2=m2, rank=rank)


def pack(rows):
    keys = ('nll', 'gap', 'above', 'h_lb', 'm2', 'rank')
    return {k: [round(r[k], 4) if k != 'rank' else r[k] for r in rows] for k in keys}


def teacher_turns(a, rec, r, tok):
    """Teacher stats per turn from prompt_logprobs K: one request (full_history) or one per turn (norefeed)."""
    out = {}
    if a.condition == 'full_history':
        res = post(a.url + '/completions', dict(model=a.served_name, prompt=rec['ids'], max_tokens=1, temperature=1.0,
                                                prompt_logprobs=a.k))
        plp = res['choices'][0]['prompt_logprobs']
        spans = [(t, rec['ids'], t['tok_start'], plp) for t in rec['turns']]
    else:
        spans = []
        for t, (ids, n) in zip(rec['turns'], norefeed_prompts(r, tok)):
            res = post(a.url + '/completions', dict(model=a.served_name, prompt=ids, max_tokens=1, temperature=1.0,
                                                    prompt_logprobs=a.k))
            spans.append((t, ids, len(ids) - n, res['choices'][0]['prompt_logprobs']))
    for t, ids, s0, plp in spans:
        rows = []
        for j in range(t['tok_end'] - t['tok_start']):
            e = plp[s0 + j]
            top = sorted(((int(k), v['logprob']) for k, v in e.items()), key=lambda x: -x[1])
            lp = e[str(ids[s0 + j])]['logprob']
            rows.append(stats(lp, top[:a.k]))
        out[t['k']] = pack(rows)
    return out


def sample_turn(a, prompt, pieces_of):
    res = post(a.url + '/completions', dict(model=a.served_name, prompt=prompt, n=a.n, temperature=1.0, top_p=1.0,
                                            max_tokens=min(MAX_REPLY, a.max_len - len(prompt) - 8), logprobs=a.k,
                                            return_tokens_as_token_ids=True, skip_special_tokens=False,
                                            stop_token_ids=[a.eot], seed=a.seed))
    out = []
    for ch in res['choices']:
        lg = ch['logprobs']
        ids = [int(s.split(':', 1)[1]) for s in lg['tokens']]
        rows = []
        for lp, top in zip(lg['token_logprobs'], lg['top_logprobs']):
            tt = sorted(((int(k.split(':', 1)[1]), v) for k, v in top.items()), key=lambda x: -x[1])
            rows.append(stats(lp, tt[:a.k]))
        pcs = pieces_of(ids)
        out.append(dict(ids=ids, finish=ch.get('finish_reason'), cls=''.join(str(CLASSES.index(c)) for c in classify(pcs)),
                        **pack(rows)))
    return out


def path_check(a, prompt, sample):
    """Re-score one sample through the prefill path: per-token |decode-path nll - prefill-path nll|."""
    ids = prompt + sample['ids']
    res = post(a.url + '/completions', dict(model=a.served_name, prompt=ids, max_tokens=1, temperature=1.0,
                                            prompt_logprobs=0))
    plp = res['choices'][0]['prompt_logprobs']
    d = [(-plp[len(prompt) + j][str(i)]['logprob']) - x for j, (i, x) in enumerate(zip(sample['ids'], sample['nll']))]
    return dict(n=len(d), mean=sum(d) / len(d), mean_abs=sum(abs(x) for x in d) / len(d))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--renders', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--model-path', required=True)
    ap.add_argument('--condition', choices=('full_history', 'norefeed'), required=True)
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-len', type=int, default=131072)
    ap.add_argument('--url', default='http://localhost:8000/v1')
    ap.add_argument('--served-name', default='snowball')
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--workers', type=int, default=24, help='turns in flight')
    ap.add_argument('--path-check', type=int, default=24, help='samples re-scored through the prefill path')
    a = ap.parse_args()
    tokdir = json.load(open(os.path.join(a.renders, 'render_report.json')))['tokenizer_dir']
    tok, tpl, bos = load_student(tokdir)
    a.eot = tok.token_to_id('<|eot_id|>')
    pc = Pieces(tokdir)
    ids = sorted(f[:-5] for f in os.listdir(a.renders) if f.startswith('t') and f.endswith('.json'))
    if a.only:
        ids = [i for i in ids if i in a.only]
    os.makedirs(a.out, exist_ok=True)
    run = dict(tag=a.tag, condition=a.condition, model_path=a.model_path, n=a.n, k=a.k, seed=a.seed,
               sampling=dict(temperature=1.0, top_p=1.0, max_tokens=MAX_REPLY, stop='<|eot_id|>', draft=None),
               vllm_version=get(a.url.rsplit('/v1', 1)[0] + '/version').get('version'), host=socket.gethostname(),
               slurm_job=os.environ.get('SLURM_JOB_ID'), started=time.strftime('%Y-%m-%dT%H:%M:%S%z'), episodes={},
               path_check=[])
    checked = 0
    for tid in ids:
        rec = json.load(open(os.path.join(a.renders, tid + '.json')))
        r = render(json.load(open(rec['msa_trajectory']))['messages'], tok, tpl, bos)
        assert hashlib.sha256(r['text'].encode()).hexdigest() == rec['text_sha256'], 'trajectory changed since rendering'
        t0 = time.time()
        prompts = ([rec['ids'][:t['tok_start']] for t in rec['turns']] if a.condition == 'full_history'
                   else [ids_[:len(ids_) - n] for ids_, n in norefeed_prompts(r, tok)])
        with cf.ThreadPoolExecutor(a.workers) as pool:
            fut_t = pool.submit(teacher_turns, a, rec, r, tok)
            futs = [pool.submit(sample_turn, a, p, pc.split) for p in prompts]
            samples = [f.result() for f in futs]
            teacher = fut_t.result()
        for t, p, ss in zip(rec['turns'], prompts, samples):
            if checked < a.path_check and t['k'] % 3 == 0:
                run['path_check'].append(dict(traj_id=tid, turn=t['k'], **path_check(a, p, ss[0])))
                checked += 1
        turns = [dict(k=t['k'], prompt_tokens=len(p), teacher=dict(cls=''.join(str(CLASSES.index(c)) for c in t['cls']),
                                                                   **teacher[t['k']]), samples=ss)
                 for t, p, ss in zip(rec['turns'], prompts, samples)]
        with gzip.open(os.path.join(a.out, tid + '.json.gz'), 'wt') as f:
            json.dump(dict(traj_id=tid, condition=a.condition, tag=a.tag, turns=turns), f)
        gen = sum(len(s['ids']) for ss in samples for s in ss)
        run['episodes'][tid] = dict(turns=len(turns), samples=sum(len(ss) for ss in samples), generated_tokens=gen,
                                    length_capped=sum(s['finish'] == 'length' for ss in samples for s in ss),
                                    seconds=round(time.time() - t0, 1), text_sha256=rec['text_sha256'])
        print(tid, json.dumps(run['episodes'][tid]), flush=True)
        json.dump(run, open(os.path.join(a.out, 'run.json'), 'w'), indent=1)
    run['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    json.dump(run, open(os.path.join(a.out, 'run.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
