#!/usr/bin/env python3
"""OpenThoughts-Agent-SFT-100K (@ 45fb28fc) as relay SFT rows (ids + loss) for 09-21, without the contaminated slice and
without any trace that overlaps our eval sets (marin stage relay_ota).

Selection:
  * families: every task family except `issue` (IssueTasks is built from SWE-bench test issues: 255 of Verified's 500
    rewritten; ai_memory/notes/ota_swebench_contamination.md) and `summarization-*` trace fragments (Harbor compaction
    pieces, not whole episodes). Kept: superuser, swesmith, tezos, tezos-issue.
  * eval overlap: a trace is dropped when its task prompt (first user message) shares >= --min-shared word 14-grams
    with the instruction.md of any task in --eval-dirs (TB2.1, SWE-bench Verified random-100, TB-lite), and the matched
    eval tasks are reported per set.
Rendering: every assistant turn becomes a relay teacher turn through render.py's own render_episode (--render-dir),
as kimi_relay_rows.py does: <think>..</think> (a lone closing tag gets its opening restored, as ota_sft_convert.py
does for tezos) becomes the turn's reasoning, the rest its Terminus-2 action; rows over 65,536 tokens are dropped.

    python ota_relay_rows.py --raw <dir with data/*.parquet> --render-dir <checkout>/data/relay/sft \
        --tokenizer-dir <09-21> --eval-dirs <tb21> <swe100> <tblite> --out ota_rows.jsonl --report ota_rows.json
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

THINK = re.compile(r'^\s*<think>(.*?)</think>\s*(.*)$', re.S)
WORD = re.compile(r'\w+')
N = 14


def family(task):
    return re.sub(r'[-_]?\d.*$', '', task) or task


def shingles(text):
    w = WORD.findall(text.lower())
    return {' '.join(w[i:i + N]) for i in range(max(0, len(w) - N + 1))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw', required=True)
    ap.add_argument('--files', nargs='*', help='parquet shards to read (default: every <raw>/data/*.parquet)')
    ap.add_argument('--render-dir', required=True)
    ap.add_argument('--tokenizer-dir', required=True)
    ap.add_argument('--eval-dirs', nargs='+', required=True)
    ap.add_argument('--min-shared', type=int, default=3)
    ap.add_argument('--think-limit', type=int, default=16384)
    ap.add_argument('--out', required=True)
    ap.add_argument('--report', required=True)
    a = ap.parse_args()
    import pyarrow.parquet as pq
    sys.path.insert(0, os.path.abspath(a.render_dir))
    import render
    tok = render.rcap.load_tokenizer(os.path.join(a.tokenizer_dir, 'tokenizer.json'))
    tpl = render.load_template(os.path.join(a.tokenizer_dir, 'chat_template.jinja'))
    bos = json.load(open(os.path.join(a.tokenizer_dir, 'tokenizer_config.json'))).get('bos_token', '<|begin_of_text|>')
    bos = bos.get('content') if isinstance(bos, dict) else bos
    render.episode_turns = lambda traj, records: traj['turns']

    index = collections.defaultdict(set)   # 14-gram -> {(set, task)}
    for d in a.eval_dirs:
        for ins in glob.glob(os.path.join(d, '*', 'instruction.md')):
            key = (os.path.basename(d), os.path.basename(os.path.dirname(ins)))
            for s in shingles(open(ins).read()):
                index[s].add(key)

    st = collections.Counter()
    hits_eval = collections.defaultdict(set)
    fam = collections.Counter()
    with open(a.out, 'w') as out:
        for f in (a.files or sorted(glob.glob(os.path.join(a.raw, 'data', '*.parquet')))):
            for r in pq.read_table(f).to_pylist():
                st['raw'] += 1
                src = r.get('trace_source') or 'main'
                fm = family(r['task'])
                if fm == 'issue':
                    st['drop_issue_family'] += 1
                    continue
                if src.startswith('summarization'):
                    st['drop_summarization_fragment'] += 1
                    continue
                conv = r['conversations']
                if not conv or conv[0]['role'] != 'user' or conv[-1]['role'] != 'assistant':
                    st['drop_shape'] += 1
                    continue
                c = collections.Counter()
                for s in shingles(conv[0]['content']):
                    for key in index.get(s, ()):
                        c[key] += 1
                bad = [k for k, n in c.items() if n >= a.min_shared]
                if bad:
                    st['drop_eval_overlap'] += 1
                    for k in bad:
                        hits_eval[k[0]].add(k[1])
                    continue
                turns, ok = [], True
                for k, m in enumerate(conv):
                    if m['role'] == 'assistant':
                        txt = m['content']
                        if '<think>' not in txt and txt.count('</think>') == 1:
                            txt = '<think>' + txt
                        mt = THINK.match(txt)
                        reasoning, action = (mt.group(1).strip(), mt.group(2)) if mt else (None, txt)
                        turns.append(('assistant', action, dict(owner='teacher', reasoning=reasoning or None,
                                                                autofix=False, repair=False, turn=k)))
                    elif m['role'] == 'user':
                        turns.append(('user', m['content'], {}))
                    else:
                        ok = False
                        break
                if not ok:
                    st['drop_role'] += 1
                    continue
                row = render.render_episode(dict(turns=turns), [], tok, tpl, bos, autofix_loss='none', think_limit=a.think_limit)
                if not row['fits']:
                    st['drop_over_64k'] += 1
                    continue
                if not any(row['loss']):
                    st['drop_no_trained_token'] += 1
                    continue
                st['rows'] += 1
                st['tokens'] += row['n_tokens']
                st['trained_tokens'] += sum(row['loss'])
                fam[fm] += 1
                out.write(json.dumps(dict(sid=f"ota:{r['trial_name']}:{r.get('episode')}:{src}", n_tokens=row['n_tokens'],
                                          fits=True, ids=row['ids'], loss=row['loss'], turns=row['turns'])) + '\n')
    rep = dict(st, families=dict(fam), eval_tasks_matched={k: sorted(v) for k, v in hits_eval.items()},
               min_shared_14grams=a.min_shared)
    json.dump(rep, open(a.report, 'w'), indent=1)
    print(json.dumps(dict(st, families=dict(fam), eval_tasks_matched={k: len(v) for k, v in hits_eval.items()}), indent=1))


if __name__ == '__main__':
    main()
