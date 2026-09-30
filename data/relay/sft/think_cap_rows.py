#!/usr/bin/env python3
"""Cap the trained reasoning of rendered relay rows (ids + loss) without re-rendering, and measure it.

render.py trains a teacher turn's reasoning span (<|start_think|> .. <|end_think|>) only when it was never cut and is at
most --think-limit tokens; otherwise the whole span, end marker included, is masked (a truncated prefix followed by
<|end_think|> would teach abrupt stops). This applies the same rule with a lower limit to rows already rendered:
every TRAINED span whose reasoning is longer than --cap tokens gets loss 0 over the span, end marker included. The ids
and every other token's loss are unchanged, so row lengths and packing are unchanged.

    python think_cap_rows.py --rows <rendered.jsonl> --tokenizer-dir <09-21 dir> [--cap 2048 --out capped.jsonl]

Prints: rows, trained tokens, trained reasoning spans (count, length percentiles), trained tokens inside reasoning,
and for caps 1k/2k/4k/8k the share of trained spans and trained tokens each would remove.
"""
import argparse
import json


def token_id(tok_dir, text):
    tj = json.load(open(f'{tok_dir}/tokenizer.json'))
    for t in tj.get('added_tokens', []):
        if t['content'] == text:
            return t['id']
    raise SystemExit(f'{text} is not an added token of {tok_dir}')


def spans(ids, loss, start, end):
    """(s, e, trained) per reasoning span: ids[s] == start, ids[e] == end."""
    out, s = [], None
    for i, t in enumerate(ids):
        if t == start:
            s = i
        elif t == end and s is not None:
            out.append((s, i, sum(loss[s + 1:i + 1]) > 0))
            s = None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rows', required=True)
    ap.add_argument('--tokenizer-dir', required=True)
    ap.add_argument('--cap', type=int)
    ap.add_argument('--out')
    a = ap.parse_args()
    start, end = token_id(a.tokenizer_dir, '<|start_think|>'), token_id(a.tokenizer_dir, '<|end_think|>')
    caps = (1024, 2048, 4096, 8192)
    n_rows = trained = think_trained = 0
    lens = []
    removed = {c: [0, 0] for c in caps}
    fo = open(a.out, 'w') if a.out else None
    for line in open(a.rows):
        r = json.loads(line)
        ids, loss = r['ids'], r['loss']
        n_rows += 1
        trained += sum(loss)
        for s, e, tr in spans(ids, loss, start, end):
            if not tr:
                continue
            n = e - s - 1
            lens.append(n)
            think_trained += sum(loss[s:e + 1])
            for c in caps:
                if n > c:
                    removed[c][0] += 1
                    removed[c][1] += sum(loss[s:e + 1])
            if a.cap is not None and n > a.cap:
                loss[s:e + 1] = [0] * (e + 1 - s)
        if fo:
            r['loss'] = loss
            fo.write(json.dumps(r) + '\n')
    lens.sort()
    pct = lambda q: lens[min(len(lens) - 1, int(q * len(lens)))] if lens else None  # noqa: E731
    print(json.dumps(dict(
        rows=n_rows, trained_tokens=trained, trained_reasoning_spans=len(lens), trained_reasoning_tokens=think_trained,
        reasoning_share_of_trained=round(think_trained / max(1, trained), 3),
        span_tokens=dict(p10=pct(.1), p50=pct(.5), p90=pct(.9), p99=pct(.99), max=lens[-1] if lens else None),
        cap_would_remove={c: dict(spans=v[0], span_share=round(v[0] / max(1, len(lens)), 3), trained_tokens=v[1],
                                  trained_share=round(v[1] / max(1, trained), 3)) for c, v in removed.items()},
        cap=a.cap, out=a.out), indent=1))


if __name__ == '__main__':
    main()
