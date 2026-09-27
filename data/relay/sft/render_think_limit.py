#!/usr/bin/env python3
"""Re-render a final manifest with a different thinking-loss limit, using an unchanged render.py.

    python render_think_limit.py --render-dir <worktree@d0ddd237>/data/relay/sft --run-dir <run> --arm relay_repair \
        --manifest final_manifest.jsonl --tokenizer-dir <09-21 model dir> --think-limit 16384 \
        --out final_rendered_think16k.jsonl

render.py is imported from --render-dir, so the rendering code (template, capped history, mask rules, 65,536 row
limit) is exactly that checkout's; only render_episode's think_limit argument changes (default 8,192, 09-21's old
per-turn output limit; 16,384 under the 65k/16k TB policy, coordinator 2026-09-27). The limit only decides whether an
uncut Qwen reasoning span is trained; it never changes the text, so row lengths are unchanged.

Prints per-run stats: rows, rows over 65,536, trained tokens, Qwen turns whose uncut reasoning is over 8,192 and over
16,384 tokens (masked at the old / new limit), and the max reasoning tokens in one Qwen turn.
"""
import argparse
import collections
import glob
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--render-dir', required=True)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--arm', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--tokenizer-dir', required=True)
    ap.add_argument('--think-limit', type=int, required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.render_dir))
    import render
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(a.render_dir)), 'pilot'))
    import readout
    name = os.path.basename(os.path.normpath(a.run_dir))
    want = {json.loads(l)['sid'] for l in open(a.manifest) if l.strip()}
    tok = render.rcap.load_tokenizer(os.path.join(a.tokenizer_dir, 'tokenizer.json'))
    tpl = render.load_template(os.path.join(a.tokenizer_dir, 'chat_template.jinja'))
    bos = json.load(open(os.path.join(a.tokenizer_dir, 'tokenizer_config.json'))).get('bos_token', '<|begin_of_text|>')
    bos = bos.get('content') if isinstance(bos, dict) else bos
    rv = readout.router_view(os.path.join(a.run_dir, f'router_{a.arm}'))
    recs = collections.defaultdict(list)
    for r in rv['recs']:
        if r['sid'] in want:
            recs[r['sid']].append(r)
    st = collections.Counter()
    max_think, done, over = 0, set(), []
    with open(a.out, 'w') as out:
        for d in readout.arm_job_dirs(a.run_dir, name, a.arm):
            for tj in sorted(glob.glob(os.path.join(d, '*', '**', 'agent', 'trajectory.json'), recursive=True)):
                traj = json.load(open(tj))
                sid = traj.get('session_id')
                if sid not in want or sid in done:
                    continue
                row = render.render_episode(traj, recs[sid], tok, tpl, bos, think_limit=a.think_limit)
                render.check_row(row, tok)
                done.add(sid)
                st['rows'] += 1
                st['over_64k'] += not row['fits']
                st['trained_tokens'] += sum(row['loss'])
                for m in row['turns']:
                    if m['owner'] != 'teacher':
                        continue
                    st['qwen_turns'] += 1
                    n = m.get('reasoning_tokens') or 0
                    max_think = max(max_think, n)
                    uncut = m['cut_at'] is None and m.get('has_reasoning') and not m.get('autofix')
                    st['uncut_over_8192'] += bool(uncut and n > 8192)
                    st['uncut_over_16384'] += bool(uncut and n > 16384)
                    st['think_trained_turns'] += bool(m.get('think_trained'))
                if not row['fits']:
                    over.append(dict(sid=sid, n_tokens=row['n_tokens']))
                out.write(json.dumps(dict(sid=sid, n_tokens=row['n_tokens'], fits=row['fits'], ids=row['ids'],
                                          loss=row['loss'], turns=row['turns'])) + '\n')
    res = dict(st, over_64k_rows=over, max_reasoning_tokens_per_turn=max_think, missing=len(want - done),
               think_limit=a.think_limit,
               render_dir=os.path.abspath(a.render_dir))
    print(json.dumps(res, indent=1, default=str))


if __name__ == '__main__':
    main()
