"""nll_client.py: 09-21's per-token NLL on the trained tokens of rendered SFT rows (render.py output: ids + loss).

Posts each row's token ids to a vLLM /v1/completions server with prompt_logprobs=0 and max_tokens=1, and averages
-logprob over the positions with loss == 1 (the tokens SFT would train on). Writes one line per row.

  python nll_client.py --url http://host:8000/v1 --model snowball --rows a.jsonl:label_a b.jsonl:label_b --n 120 --out nll.jsonl
"""
import argparse, json, random, urllib.request, concurrent.futures as cf, math


def score(url, model, row):
    body = dict(model=model, prompt=row['ids'], max_tokens=1, temperature=0.0, prompt_logprobs=0)
    req = urllib.request.Request(url + '/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    pl = r['choices'][0]['prompt_logprobs']
    tot = n = 0.0
    for i, (lp, l) in enumerate(zip(pl, row['loss'])):
        if not l or lp is None:
            continue
        tid = str(row['ids'][i]); d = lp.get(tid) or next(iter(lp.values()))
        v = d['logprob'] if isinstance(d, dict) else d
        tot += -v; n += 1
    return tot, n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--url', required=True); ap.add_argument('--model', default='snowball')
    ap.add_argument('--rows', nargs='+', required=True); ap.add_argument('--n', type=int, default=120)
    ap.add_argument('--out', required=True); ap.add_argument('--workers', type=int, default=8); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    out = open(a.out, 'a')
    for spec in a.rows:
        path, label = spec.rsplit(':', 1)
        rows = [json.loads(l) for l in open(path)]
        rows = [r for r in rows if sum(r['loss']) > 0 and len(r['ids']) <= 65000]
        random.Random(a.seed).shuffle(rows); rows = rows[:a.n]
        with cf.ThreadPoolExecutor(a.workers) as ex:
            futs = {ex.submit(score, a.url, a.model, r): r for r in rows}
            T = N = 0.0
            for f in cf.as_completed(futs):
                r = futs[f]
                try:
                    tot, n = f.result()
                except Exception as e:
                    out.write(json.dumps(dict(label=label, sid=r.get('sid'), error=str(e)[:300])) + '\n'); continue
                T += tot; N += n
                out.write(json.dumps(dict(label=label, sid=r.get('sid'), nll_sum=tot, n=n, nll=tot / n if n else None,
                                          n_tokens=len(r['ids']))) + '\n'); out.flush()
        print(f'{label}: rows {len(rows)} token-weighted NLL {T / N if N else float("nan"):.4f} over {int(N)} trained tokens', flush=True)


if __name__ == '__main__':
    main()
