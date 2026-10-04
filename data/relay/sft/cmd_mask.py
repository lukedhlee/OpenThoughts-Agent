#!/usr/bin/env python3
"""Rendered MSA SFT rows -> the same rows with no loss on assistant turns whose tool call carries a long command.

Under mini-swe-agent tool mode a command is a JSON string inside <tool_call>; the longer it is the more often the
student breaks its escaping (10-03 evals, every model: 0.3-0.7 % of calls under 300 chars fail to parse, 12-17 % over
3,000), and three bad calls in a row end the attempt. The Qwen teacher's relay turns are full of long heredoc scripts
(17 % of its commands > 1,500 chars vs 6 % in the SWE sources). Masking those turns keeps them in context (the episode
stays intact) but stops training the student to write them.

    python cmd_mask.py --rows <rows.jsonl> --out <rows_masked.jsonl> [--max-chars 2000]
Rows whose every trained turn gets masked are dropped. Adds masked_turns per row; recomputes trained_tokens.
"""
import argparse
import json
import os
import re

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
TOKDIR = '/scratch/11584/lukedhlee/models/grug-datakit-sft-20260921'
TC = re.compile(r'<tool_call>\n(.*?)\n</tool_call>', re.S)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rows', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-chars', type=int, default=2000)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(TOKDIR, 'tokenizer.json'))
    eot = tok.token_to_id('<|eot_id|>')
    hdr = tok.encode('<|start_header_id|>assistant<|end_header_id|>\n', add_special_tokens=False).ids
    n_in = n_out = turns_masked = turns_trained = tr_in = tr_out = 0
    with open(a.out, 'w') as f:
        for line in open(a.rows):
            r = json.loads(line)
            ids, loss = r['ids'], list(r['loss'])
            n_in += 1
            tr_in += sum(loss)
            i, masked = 0, 0
            while i <= len(ids) - len(hdr):
                if ids[i] == hdr[0] and ids[i:i + len(hdr)] == hdr:
                    j = i + len(hdr)
                    while j < len(ids) and ids[j] != eot:
                        j += 1
                    if any(loss[i:j + 1]):
                        turns_trained += 1
                        longest = 0
                        for body in TC.findall(tok.decode(ids[i:j + 1], skip_special_tokens=False)):
                            try:
                                longest = max(longest, len(json.loads(body)['arguments']['command']))
                            except (ValueError, KeyError, TypeError):
                                pass
                        if longest > a.max_chars:
                            loss[i:j + 1] = [0] * (j + 1 - i)
                            masked += 1
                    i = j
                i += 1
            turns_masked += masked
            if not any(loss):
                continue
            r.update(loss=loss, trained_tokens=sum(loss), masked_turns=masked)
            f.write(json.dumps(r) + '\n')
            n_out += 1
            tr_out += sum(loss)
    rep = dict(rows_in=n_in, rows_out=n_out, max_chars=a.max_chars, trained_turns=turns_trained,
               masked_turns=turns_masked, trained_in=tr_in, trained_out=tr_out)
    json.dump(rep, open(a.out + '.json', 'w'), indent=1)
    print(json.dumps(rep))


if __name__ == '__main__':
    main()
