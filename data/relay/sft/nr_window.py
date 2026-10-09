#!/usr/bin/env python3
"""Rendered MSA SFT rows -> rows whose context matches the no-refeed eval (tb2_msa_toolmode_0924_norefeed.yaml:
interleaved_thinking false, so every earlier assistant turn reaches the model WITHOUT its reasoning).

Our renderers (render_msa.py, render_swe_traces.py, render_si2ca.py, render_orchard.py) keep each earlier turn's
reasoning in context (cut to 1k tokens). Under the no-refeed eval the model sees think-less history, and the trained
models then often open <|start_think|>, write their prose and the tool call inside it and never close it: the harness
finds no call (27-43 % of their format errors on 10-03, vs 8 % for 09-21).

A row's trained assistant turns are split into consecutive windows of --k turns; each window becomes one row: the
prefix up to the window's last turn, with the <|start_think|>..<|end_think|> span removed from every assistant turn
before the window (exactly the template's rendering of a turn without reasoning_content: checked token for token
against re-tokenizing the decoded text), the window's turns as rendered, loss only on the window's turns. So each
window's first turn sees exactly the eval's context and its later turns see at most k-1 recent reasoning spans.
In every trained turn the <|start_think|> token is trained, and so is <|end_think|> where it ends the turn's whole
reasoning: the reasoning is trained, or it is masked for being over the think limit (longer than CAP_TOKENS, so never a
cut). A turn whose reasoning was cut to reasoning_cap.CAP_TOKENS keeps <|end_think|> masked: training it there taught
the msasubnr / msafinnr / msafinnr2 / msafinsd models to stop thinking at ~1,000 tokens (eval p99 1,000-1,123 reasoning
tokens vs 4,400 for the same rows rendered without windows; 18 % of _relayfin_nr's trained <|end_think|> followed a cut).
--mode one keeps a single window per row, at a random position (for plentiful SWE replay: about the source's token
count, k trained turns per episode); --mode all keeps every window.

    python nr_window.py --rows <rows.jsonl> --out <rows_nr.jsonl> [--k 8] [--mode all|one] [--seed 0]
Rows keep their metadata; sid becomes '<sid>#w<j>', n_tokens / trained_tokens are recomputed, plus nr_window = j,
nr_k = k, nr_turns = (first, last) trained-turn index of the window.
"""
import argparse
import json
import os
import random

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
TOKDIR = '/scratch/11584/lukedhlee/models/grug-datakit-sft-20260921'
CAP_TOKENS = 1000    # reasoning_cap.CAP_TOKENS: the renderers cut a turn's reasoning to at most this many tokens


def load_ids():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(TOKDIR, 'tokenizer.json'))
    hdr = tok.encode('<|start_header_id|>assistant<|end_header_id|>\n', add_special_tokens=False).ids
    return (tok.token_to_id('<|start_think|>'), tok.token_to_id('<|end_think|>'), tok.token_to_id('<|eot_id|>'),
            hdr)


ST, ET, EOT, HDR = load_ids()


def assistant_turns(ids):
    """[(header start, eot index)] of every assistant turn."""
    out, i, n = [], 0, len(HDR)
    while i <= len(ids) - n:
        if ids[i] == HDR[0] and ids[i:i + n] == HDR:
            j = i + n
            while j < len(ids) and ids[j] != EOT:
                j += 1
            out.append((i, j))
            i = j
        i += 1
    return out


def close_think(seg_ids, seg_loss):
    """Train a trained turn's <|start_think|>, and its <|end_think|> unless the reasoning between them was cut."""
    st = seg_ids.index(ST) if ST in seg_ids else None
    et = seg_ids.index(ET, st) if st is not None and ET in seg_ids[st:] else None
    if st is None:
        return
    seg_loss[st] = 1
    if et is not None and (any(seg_loss[st + 1:et]) or et - st - 1 > CAP_TOKENS):
        seg_loss[et] = 1


def windows(r, k, mode, rng):
    ids, loss = r['ids'], r['loss']
    turns = assistant_turns(ids)
    trained = [t for t in turns if any(loss[t[0]:t[1] + 1])]
    if not trained:
        return []
    starts = list(range(0, len(trained), k))
    if mode == 'one':
        starts = [rng.choice(starts)]
    out = []
    for w, s0 in enumerate(starts):
        win = trained[s0:s0 + k]
        first, last = win[0][0], win[-1][1]
        win_set = set(win)
        new_ids, new_loss, prev = [], [], 0
        for (a, b) in turns:
            if a > last:
                break
            new_ids += ids[prev:a]
            new_loss += [0] * (a - prev)
            if a < first:            # history: drop the think span, no loss
                inside = False
                for t in ids[a:b + 1]:
                    if t == ST:
                        inside = True
                    elif t == ET and inside:
                        inside = False
                    elif not inside:
                        new_ids.append(t)
                        new_loss.append(0)
            else:                    # inside the window (every turn from `first` to `last`)
                seg_ids, seg_loss = ids[a:b + 1], list(loss[a:b + 1])
                if (a, b) in win_set:
                    close_think(seg_ids, seg_loss)
                else:
                    seg_loss = [0] * len(seg_loss)
                new_ids += seg_ids
                new_loss += seg_loss
            prev = b + 1
        j = starts.index(s0) if mode == 'all' else s0 // k
        row = {key: v for key, v in r.items() if key not in ('ids', 'loss')}
        row.update(sid=f"{r['sid']}#w{j}", ids=new_ids, loss=new_loss, n_tokens=len(new_ids),
                   trained_tokens=sum(new_loss), nr_window=j, nr_k=k,
                   nr_turns=[trained.index(win[0]), trained.index(win[-1])])
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rows', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--mode', choices=('all', 'one'), default='all')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    n_in = n_out = tok_in = tok_out = tr_in = tr_out = 0
    with open(a.out, 'w') as f:
        for line in open(a.rows):
            r = json.loads(line)
            n_in += 1
            tok_in += len(r['ids'])
            tr_in += sum(r['loss'])
            for w in windows(r, a.k, a.mode, rng):
                f.write(json.dumps(w) + '\n')
                n_out += 1
                tok_out += w['n_tokens']
                tr_out += w['trained_tokens']
    rep = dict(rows_in=n_in, rows_out=n_out, k=a.k, mode=a.mode, tokens_in=tok_in, tokens_out=tok_out,
               token_multiplier=round(tok_out / max(tok_in, 1), 3), trained_in=tr_in, trained_out=tr_out)
    json.dump(rep, open(a.out + '.json', 'w'), indent=1)
    print(json.dumps(rep))


if __name__ == '__main__':
    main()
