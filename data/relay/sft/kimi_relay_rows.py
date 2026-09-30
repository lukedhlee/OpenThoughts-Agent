#!/usr/bin/env python3
"""Kimi SWE-smith traces (Terminus-2, Kimi-2.5 teacher) rendered as relay SFT rows (ids + loss), for mixing into a relay
arm (marin stage relay_akimi).

Every assistant turn is rendered as a relay TEACHER turn by render.py's own render_episode (imported from --render-dir,
the checkout the relay rows were rendered with): 09-21's chat_template.jinja, reasoning inline as
<|start_think|>..<|end_think|> before the JSON, every older turn's reasoning cut to its first 1,000 tokens in the history,
the final turn's kept whole; loss on each turn's JSON + <|eot_id|> and on its reasoning only when uncut and within
--think-limit. User turns (the Terminus-2 prompt and terminal observations) are masked. Only the turn list is supplied
here (render.episode_turns is replaced); rendering and masking are unchanged, so Kimi rows and relay rows follow one rule.

Input: the kimi_swesmith_v1 parquet (kimi_sft_convert.py serve format: assistant turns are
"<|start_think|>\\n<reasoning>\\n<|end_think|>\\n\\n<json>"). Output: one JSON line per row (sid, n_tokens, fits, ids, loss,
turns), rows over 65,536 tokens dropped and counted.

    python kimi_relay_rows.py --parquet <train parquet> --render-dir <checkout>/data/relay/sft --tokenizer-dir <09-21> --out kimi_rows.jsonl
"""
import argparse
import json
import os
import re
import sys

TURN_RE = re.compile(r'^<\|start_think\|>\n(.*?)\n<\|end_think\|>\n\n(.*)$', re.S)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--parquet', required=True)
    ap.add_argument('--render-dir', required=True)
    ap.add_argument('--tokenizer-dir', required=True)
    ap.add_argument('--think-limit', type=int, default=16384)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import pyarrow.parquet as pq
    sys.path.insert(0, os.path.abspath(a.render_dir))
    import render
    tok = render.rcap.load_tokenizer(os.path.join(a.tokenizer_dir, 'tokenizer.json'))
    tpl = render.load_template(os.path.join(a.tokenizer_dir, 'chat_template.jinja'))
    bos = json.load(open(os.path.join(a.tokenizer_dir, 'tokenizer_config.json'))).get('bos_token', '<|begin_of_text|>')
    bos = bos.get('content') if isinstance(bos, dict) else bos
    render.episode_turns = lambda traj, records: traj['turns']
    st = dict(rows=0, over_64k=0, bad_turn=0, trained_tokens=0, tokens=0, turns=0)
    with open(a.out, 'w') as out:
        for r in pq.read_table(a.parquet).to_pylist():
            conv, turns, ok = r['conversations'], [], True
            for k, m in enumerate(conv):
                if m['role'] == 'assistant':
                    mt = TURN_RE.match(m['content'])
                    if not mt:
                        ok = False
                        break
                    turns.append(('assistant', mt.group(2), dict(owner='teacher', reasoning=mt.group(1), autofix=False,
                                                                 repair=False, turn=k)))
                else:
                    turns.append(('user', m['content'], {}))
            if not ok or not turns or turns[0][0] != 'user' or turns[-1][0] != 'assistant':
                st['bad_turn'] += 1
                continue
            row = render.render_episode(dict(turns=turns), [], tok, tpl, bos, autofix_loss='none', think_limit=a.think_limit)
            render.check_row(row, tok)
            if not row['fits']:
                st['over_64k'] += 1
                continue
            st['rows'] += 1
            st['tokens'] += row['n_tokens']
            st['trained_tokens'] += sum(row['loss'])
            st['turns'] += len(row['turns'])
            out.write(json.dumps(dict(sid=f"kimi:{r['instance_id']}:{r['trial_name']}", n_tokens=row['n_tokens'],
                                      fits=True, ids=row['ids'], loss=row['loss'], turns=row['turns'])) + '\n')
    print(json.dumps(st, indent=1))


if __name__ == '__main__':
    main()
