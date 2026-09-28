"""Same-state candidate sets from SkyRL train-rollout dumps, for opd_ref_turns.py's turn1 / branch tests at scale.

A GRPO group's 8 rollouts start from the same state, so their first turns are 8 candidates at one state. Deeper, two
rollouts of a group that ran identical commands (same keystrokes, same done flag) for k turns sit in (nearly) the same
sandbox state before turn k+1: a natural branch point. Only nodes whose members have both outcomes can rank anything.
Each rollout is cut after the deepest turn any of its nodes needs, so scoring pays for prefixes, not whole episodes.
Episodes are keyed task = '<uid>@<step>' so groups of different steps never mix.

    scan  count groups / nodes / kept prefix characters, no tokenizer (login node is fine)
    prep  the same selection -> episodes.pkl in opd_ref_probe.py's format (tokenizers; run in the job)
"""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opd_ref_probe import load_dump_row, prep_episode, split_reply  # noqa: E402
from opd_ref_turns import signature  # noqa: E402


def action_of(content):
    return ''.join(t for k, t in split_reply(content.strip()) if k == 'action')


def select(paths, max_depth, keep_messages=True):
    """-> (kept trials with messages cut to the turns they need, summary). keep_messages=False (scan) holds only
    signatures and per-turn character counts, so a 60-step scan fits a login node."""
    groups = {}
    n_rows = 0
    for p in paths:
        for line in open(p):
            d = json.loads(line)
            n_rows += 1
            tr = load_dump_row(d)
            if tr is None:
                continue
            tr['task'] = '%s@%s' % (d['uid'], d.get('step'))
            tr['sigs'] = [signature(action_of(m['content'])) for m in tr['messages'] if m['role'] == 'assistant']
            cum, n = [], 0                         # characters up to and including each assistant turn
            for m in tr['messages']:
                n += len(m['content'])
                if m['role'] == 'assistant':
                    cum.append(n)
            tr['cum_chars'] = cum
            if not keep_messages:
                del tr['messages']
            groups.setdefault(tr['task'], []).append(tr)
    nodes_by_depth, n_div, n_mixed = {}, 0, 0
    kept = []
    for task, trs in groups.items():
        if len({tr['reward'] > 0 for tr in trs}) < 2:
            continue
        n_mixed += 1
        need = [0] * len(trs)
        for d in range(0, max_depth + 1):
            nodes = {}
            for j, tr in enumerate(trs):
                if len(tr['sigs']) > d:
                    nodes.setdefault(tuple(tr['sigs'][:d]), []).append(j)
            for key, members in nodes.items():
                if len(members) < 2 or len({trs[j]['reward'] > 0 for j in members}) < 2:
                    continue
                nodes_by_depth[d] = nodes_by_depth.get(d, 0) + 1
                n_div += d > 0 and len({trs[j]['sigs'][d] for j in members}) > 1
                for j in members:
                    need[j] = max(need[j], d + 1)
        for j, tr in enumerate(trs):
            if need[j] == 0:
                continue
            k = dict(tr, need=need[j], chars=tr['cum_chars'][need[j] - 1])
            if keep_messages:
                msgs, na = [], 0
                for m in tr['messages']:
                    if m['role'] == 'assistant':
                        if na == need[j]:
                            break
                        na += 1
                    msgs.append(m)
                k['messages'] = msgs
            kept.append(k)
    summ = dict(rows=n_rows, groups=len(groups), groups_mixed=n_mixed,
                nodes_by_depth={int(k): v for k, v in sorted(nodes_by_depth.items())},
                deep_nodes_diverging=int(n_div), kept_trials=len(kept),
                kept_chars=int(sum(tr['chars'] for tr in kept)),
                full_chars_of_kept=int(sum(tr['cum_chars'][-1] for tr in kept)),
                need_hist={int(k): int(v) for k, v in zip(*np.unique([tr['need'] for tr in kept], return_counts=True))}
                if kept else {})
    return kept, summ


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['scan', 'prep'])
    p.add_argument('--dump', required=True, help='glob of *_train_rollouts.jsonl')
    p.add_argument('--max-depth', type=int, default=12, help='deepest branch point kept (0 = first turns only)')
    p.add_argument('--snowball-tok', default='')
    p.add_argument('--qwen-tok', default='')
    p.add_argument('--out', default='')
    p.add_argument('--max-s-len', type=int, default=65535)
    a = p.parse_args()
    kept, summ = select(sorted(glob.glob(a.dump)), a.max_depth, keep_messages=a.cmd == 'prep')
    print(json.dumps(summ, indent=1), flush=True)
    if a.cmd == 'scan':
        return
    from transformers import AutoTokenizer
    stok = AutoTokenizer.from_pretrained(a.snowball_tok)
    qtok = AutoTokenizer.from_pretrained(a.qwen_tok)
    eps, skipped = [], {}
    for i, tr in enumerate(kept):
        try:
            ep, why = prep_episode(tr, stok, qtok, a.max_s_len)
        except Exception as ex:
            ep, why = None, 'error:%s' % repr(ex)[:80]
        if ep is None:
            skipped[why] = skipped.get(why, 0) + 1
            continue
        eps.append(ep)
        if i % 500 == 0:
            print('prep', i, len(kept), flush=True)
    os.makedirs(a.out, exist_ok=True)
    pickle.dump(eps, open(os.path.join(a.out, 'episodes.pkl'), 'wb'))
    ch = np.concatenate([e['chunks'] for e in eps])
    ns = ch[:, 3] - ch[:, 2]
    summ.update(episodes=len(eps), skipped=skipped, qwen_tokens=int(sum(len(e['q_ids']) for e in eps)),
                snowball_tokens=int(sum(len(e['s_ids']) for e in eps)), scored_student_tokens=int(ns.sum()),
                invalid_segments=int(sum(e['stats']['invalid'] for e in eps)))
    json.dump(summ, open(os.path.join(a.out, 'prep_summary.json'), 'w'), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    sys.exit(main())
