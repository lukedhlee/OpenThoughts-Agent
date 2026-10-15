#!/usr/bin/env python3
"""Teacher-forced student log-probs of every target token of rendered episodes, from a vLLM server's prompt_logprobs.

One /v1/completions request per episode: prompt = the episode's token ids (they already start with the template's
<|begin_of_text|>; a token prompt is never re-tokenized), max_tokens 1, prompt_logprobs K. For every position i >= 1
vLLM returns log p(ids[i] | ids[:i]) from the raw logits (temperature 1, no sampler processing), the token's rank in
the full vocabulary, and the top-K alternatives. Only target positions (assistant bodies, render_full.py) are kept.

Checks (--checks, on the first episode; any failure stops the run before the other episodes):
  aligned     usage.prompt_tokens == len(ids), and every target position's entry carries the actual token id
              (vLLM always includes it, so an off-by-one would show a neighbour's id instead)
  generation  at sampled target positions j, a separate request with prompt ids[:j], max_tokens 1, logprobs K returns
              the same top-K for its first generated token: prompt_logprobs[j] is the next-token distribution given
              exactly ids[:j], i.e. the causal label shift is right
  no_future   re-scoring the prefix that ends at turn k's <|eot_id|> gives turn k the same log-probs as the full
              episode: nothing after a turn (later turns, tool results) reaches its scores
Numerical tolerance: bf16 batch composition moves log-probs by ~1e-3..1e-2; the checks bound the mean and max.

    python score.py --renders <dir> --out <dir> --tag dk0921 --model-path <dir> [--url http://localhost:8000/v1]
"""
import argparse
import concurrent.futures as cf
import gzip
import json
import os
import random
import socket
import sys
import time
import urllib.request

TOL_MEAN, TOL_MAX = 0.02, 0.25


def post(url, payload, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as f:
        return json.load(f)


def get(url):
    with urllib.request.urlopen(url, timeout=60) as f:
        return json.load(f)


def prompt_logprobs(a, ids):
    t0 = time.time()
    r = post(a.url + '/completions', dict(model=a.served_name, prompt=ids, max_tokens=1, temperature=1.0,
                                           prompt_logprobs=a.k))
    ch = r['choices'][0]
    return ch['prompt_logprobs'], r.get('usage') or {}, time.time() - t0


def entry(plp, i, tok_id):
    """(logprob, rank, [(id, logprob)] top-K) at position i. vLLM returns the top-K plus the actual token when it is
    outside them, so the K highest entries are the top-K."""
    e = plp[i]
    if e is None or str(tok_id) not in e:
        raise AssertionError(f'position {i}: entry lacks the actual token {tok_id}')
    v = e[str(tok_id)]
    return v['logprob'], v.get('rank'), sorted(((int(k), x['logprob']) for k, x in e.items()), key=lambda t: -t[1])


def score_episode(a, rec):
    ids = rec['ids']
    plp, usage, dt = prompt_logprobs(a, ids)
    assert usage.get('prompt_tokens') == len(ids), (usage, len(ids))
    assert len(plp) == len(ids), (len(plp), len(ids))
    pos, lp, rank, top_ids, top_lp = [], [], [], [], []
    for t in rec['turns']:
        for j in range(t['tok_start'], t['tok_end']):
            v, rk, top = entry(plp, j, ids[j])
            top = top[:a.k]
            pos.append(j)
            lp.append(v)
            rank.append(rk)
            top_ids.append([x[0] for x in top])
            top_lp.append([round(x[1], 5) for x in top])
    return dict(traj_id=rec['traj_id'], positions=pos, lp=lp, rank=rank, top_ids=top_ids, top_lp=top_lp,
                usage=usage, seconds=round(dt, 1)), plp


def check_generation(a, rec, plp, n=12, seed=0):
    ids, rng = rec['ids'], random.Random(seed)
    firsts = [t['tok_start'] for t in rec['turns']]
    js = sorted(set(rng.sample(firsts, min(4, len(firsts)))
                    + rng.sample([j for t in rec['turns'] for j in range(t['tok_start'], t['tok_end'])], n)))
    diffs, top1_same = [], 0
    for j in js:
        r = post(a.url + '/completions', dict(model=a.served_name, prompt=ids[:j], max_tokens=1, temperature=1.0,
                                               logprobs=a.k, return_tokens_as_token_ids=True))
        gen = r['choices'][0]['logprobs']['top_logprobs'][0]
        g = {int(k.split(':', 1)[1]): v for k, v in gen.items()}
        p = {int(k): v['logprob'] for k, v in plp[j].items()}
        common = set(g) & set(p)
        assert len(common) >= min(len(g), a.k) - 1, f'position {j}: top-K differ {sorted(g)} vs {sorted(p)}'
        diffs += [abs(g[c] - p[c]) for c in common]
        top1_same += max(g, key=g.get) == max(p, key=p.get)
    out = dict(positions=js, compared=len(diffs), mean_abs=sum(diffs) / len(diffs), max_abs=max(diffs),
               top1_agree=f'{top1_same}/{len(js)}')
    assert out['mean_abs'] < TOL_MEAN and out['max_abs'] < TOL_MAX, out
    return out


def check_no_future(a, rec, full):
    out = []
    ks = sorted({0, len(rec['turns']) // 2, len(rec['turns']) - 2} - {-1})
    by_pos = dict(zip(full['positions'], full['lp']))
    for k in ks:
        t = rec['turns'][k]
        plp, _, _ = prompt_logprobs(a, rec['ids'][:t['tok_end']])
        d = [abs(plp[j][str(rec['ids'][j])]['logprob'] - by_pos[j]) for j in range(t['tok_start'], t['tok_end'])]
        out.append(dict(turn=k, prefix_tokens=t['tok_end'], full_tokens=len(rec['ids']), compared=len(d),
                        mean_abs=sum(d) / len(d), max_abs=max(d)))
        assert out[-1]['mean_abs'] < TOL_MEAN and out[-1]['max_abs'] < TOL_MAX, out[-1]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--renders', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', required=True, help='checkpoint tag, e.g. dk0921 / rlvr1s12')
    ap.add_argument('--model-path', required=True)
    ap.add_argument('--model-id', default='')
    ap.add_argument('--revision', default='')
    ap.add_argument('--serve-args', default='')
    ap.add_argument('--url', default='http://localhost:8000/v1')
    ap.add_argument('--served-name', default='snowball')
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--only', nargs='*', help='traj ids (default: every render)')
    ap.add_argument('--checks', action='store_true')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    ids = sorted(f[:-5] for f in os.listdir(a.renders) if f.startswith('t') and f.endswith('.json'))
    if a.only:
        ids = [i for i in ids if i in a.only]
    os.makedirs(a.out, exist_ok=True)
    run = dict(tag=a.tag, condition='full_history', model_path=a.model_path, model_id=a.model_id, revision=a.revision,
               served_name=a.served_name, serve_args=a.serve_args, url=a.url, prompt_logprobs_k=a.k,
               vllm_version=get(a.url.rsplit('/v1', 1)[0] + '/version').get('version'),
               models=get(a.url + '/models'), host=socket.gethostname(), started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
               slurm_job=os.environ.get('SLURM_JOB_ID'), renders=os.path.abspath(a.renders), episodes={})
    rp = os.path.join(a.out, 'run.json')
    if os.path.exists(rp):
        old = json.load(open(rp))
        run['episodes'] = old.get('episodes', {})
        run['checks'] = old.get('checks')
    load = lambda i: json.load(open(os.path.join(a.renders, i + '.json')))  # noqa: E731
    if a.checks:
        rec = load(ids[0])
        res, plp = score_episode(a, rec)
        gen = check_generation(a, rec, plp)
        print('generation check', json.dumps(gen), flush=True)
        nf = check_no_future(a, rec, res)
        print('no_future check', json.dumps(nf), flush=True)
        run['checks'] = dict(traj_id=rec['traj_id'], aligned='every target entry carries its token; usage matches',
                             generation=gen, no_future=nf, tolerance=dict(mean=TOL_MEAN, max=TOL_MAX))

    def one(i):
        rec = load(i)
        res, _ = score_episode(a, rec)
        with gzip.open(os.path.join(a.out, i + '.json.gz'), 'wt') as f:
            json.dump(res, f)
        nll = [-x for x in res['lp']]
        return i, dict(n_targets=len(nll), mean_nll=sum(nll) / len(nll), seconds=res['seconds'],
                       prompt_tokens=res['usage'].get('prompt_tokens'), text_sha256=rec['text_sha256'])

    with cf.ThreadPoolExecutor(a.workers) as pool:
        for i, info in pool.map(one, ids):
            run['episodes'][i] = info
            print(i, json.dumps(info), flush=True)
            json.dump(run, open(rp, 'w'), indent=1)
    run['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    json.dump(run, open(rp, 'w'), indent=1)


if __name__ == '__main__':
    sys.exit(main())
