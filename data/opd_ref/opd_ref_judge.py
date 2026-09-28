"""The teacher as a judge instead of a log-likelihood: a tokenizer-free alternative reward for PivotRL x OPD.

Qwen3.8-27B (thinking off) answers one Yes/No question per item; the score is log P(Yes) - log P(No) at its first
answer token. The same same-state and done-claim tests as opd_ref_turns.py, so the numbers line up with CompassOPD's.

    turn1  each rollout's first turn (its reasoning and commands) with the task's issue: "is this a good first step?"
           Items = the rollouts of GRPO groups with both outcomes (opd_ref_branch.py's --max-depth 0 selection).
    claim  the session up to the rollout's first task_complete:true (actions and terminal output, thinking dropped):
           "is the issue actually fixed?"

    python opd_ref_judge.py score --mode turn1 --dump '<glob>[,<glob>]' --model <qwen dir> --out <dir> --shard g --nshards 4
    python opd_ref_judge.py analyze --out <dir>
"""
import argparse
import glob
import json
import os
import pickle
import re
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opd_ref_probe import load_dump_row, split_reply, pooled_auc  # noqa: E402
from opd_ref_turns import grouped_auc  # noqa: E402
from opd_ref_branch import select, ISSUE_RE, paths_of  # noqa: E402

DONE_RE = re.compile(r'"task_complete"\s*:\s*true')
SYSTEM = 'You are an expert software engineer who reviews the work of autonomous coding agents.'
MAX_CHARS = 180_000


def issue_of(prompt):
    m = ISSUE_RE.search(prompt)
    if m:
        return m.group(0)
    a, b = prompt.find('Task Description:'), prompt.find('Current terminal state:')
    return prompt[a:b].strip() if a >= 0 and b > a else prompt[:6000]


def cut(s, head=1500, tail=2500):
    return s if len(s) <= head + tail + 50 else s[:head] + '\n[... %d characters omitted ...]\n' % (
        len(s) - head - tail) + s[-tail:]


def render(user):
    return ('<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n'
            '<|im_start|>assistant\n<think>\n\n</think>\n\n' % (SYSTEM, user))


def turn1_item(tr):
    reply = tr['messages'][1]['content']
    parts = split_reply(reply.strip())
    think = ''.join(t for k, t in parts if k == 'think').strip()
    act = ''.join(t for k, t in parts if k == 'action').strip()
    step = ('Reasoning:\n%s\n\nAction:\n%s' % (think, act)) if think else act
    return render('A coding agent is working on the task below in a Linux terminal, in a fresh checkout of the '
                  'repository at /testbed. You will see its first step: its reasoning and the commands it chose.\n\n'
                  '<task>\n%s\n</task>\n\n<first_step>\n%s\n</first_step>\n\nJudge the step as an expert engineer '
                  'would: is it a good first step, one that makes real progress toward a correct fix of this specific '
                  'issue? Answer with one word, Yes or No.' % (issue_of(tr['messages'][0]['content']), step))


def claim_item(tr):
    msgs = tr['messages']
    steps, k = [], None
    for j in range(1, len(msgs), 2):
        act = ''.join(t for kd, t in split_reply(msgs[j]['content'].strip()) if kd == 'action').strip()
        claimed = bool(DONE_RE.search(act))
        s = '[step %d%s]\n%s' % (len(steps) + 1, ' - the agent declares the task complete' if claimed else '', act)
        if not claimed and j + 1 < len(msgs):
            s += '\n[terminal output]\n' + cut(msgs[j + 1]['content'])
        steps.append(s)
        if claimed:
            k = j
            break
    if k is None:
        return None
    body = '\n\n'.join(steps)
    if len(body) > MAX_CHARS:            # keep the first step and the latest ones
        keep, n = [steps[0]], len(steps[0])
        tail = []
        for s in reversed(steps[1:]):
            if n + len(s) > MAX_CHARS:
                break
            tail.append(s)
            n += len(s)
        body = '\n\n'.join(keep + ['[... %d earlier steps omitted ...]' % (len(steps) - 1 - len(tail))] + tail[::-1])
    return render('A coding agent worked on the task below in a Linux terminal, in a checkout of the repository at '
                  '/testbed, and has now declared the task complete. Hidden tests will check whether the issue is '
                  'actually fixed.\n\n<task>\n%s\n</task>\n\n<session>\n%s\n</session>\n\nBased on the session, is '
                  'the issue actually fixed, so that the hidden tests will pass? Answer with one word, Yes or No.'
                  % (issue_of(msgs[0]['content']), body))


def items(mode, dump):
    if mode == 'turn1':
        kept, _ = select(paths_of(dump), 0)
        return [dict(trial=tr['trial'], task=tr['task'], reward=tr['reward'], prompt=turn1_item(tr)) for tr in kept]
    out = []
    for p in paths_of(dump):
        for line in open(p):
            d = json.loads(line)
            tr = load_dump_row(d)
            if tr is None:
                continue
            pr = claim_item(tr)
            if pr is not None:
                out.append(dict(trial=tr['trial'], task='%s@%s' % (d['uid'], d.get('step')), reward=tr['reward'],
                                prompt=pr))
    return out


def cmd_score(a):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    its = items(a.mode, a.dump)
    mine = [it for it in its if zlib.crc32(it['task'].encode()) % a.nshards == a.shard]   # a task's items share
    # a shard, so its common prompt prefix is served from the prefix cache
    out = os.path.join(a.out, 'judge', '%s.%d.pkl' % (a.mode, a.shard))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    llm = LLM(model=a.model, max_model_len=a.max_len, gpu_memory_utilization=0.85, enable_prefix_caching=True,
              max_num_batched_tokens=16384, max_num_seqs=32, limit_mm_per_prompt={'image': 0, 'video': 0}, seed=0)
    tok = llm.get_tokenizer()
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    prompts, over = [], 0
    for it in mine:
        ids = tok(it['prompt'], add_special_tokens=False)['input_ids']
        if len(ids) > a.max_len - 8:
            over += 1
            ids = ids[:2000] + ids[-(a.max_len - 2016):]
        prompts.append(TokensPrompt(prompt_token_ids=ids))
        it['n_tokens'] = len(ids)
    t0 = time.time()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    res = []
    for it, o in zip(mine, outs):
        yes = no = -np.inf
        for lp in (o.outputs[0].logprobs or [{}])[0].values():
            w = (lp.decoded_token or '').strip().lower()
            if w == 'yes':
                yes = np.logaddexp(yes, lp.logprob)
            elif w == 'no':
                no = np.logaddexp(no, lp.logprob)
        res.append(dict(trial=it['trial'], task=it['task'], reward=it['reward'], n_tokens=it['n_tokens'],
                        yes=float(yes), no=float(no), logit=float(np.clip(yes, -40, 0) - np.clip(no, -40, 0))))
    pickle.dump(res, open(out, 'wb'))
    print('%s shard %d: %d items, %d over length, %.0f s, %d tokens' % (a.mode, a.shard, len(res), over,
                                                                       time.time() - t0,
                                                                       sum(r['n_tokens'] for r in res)))


def cmd_analyze(a):
    M, L = {}, ['# teacher-as-judge (Qwen3.8-27B, thinking off, log P(Yes) - log P(No))', '']
    for mode in ('turn1', 'claim'):
        rs = [r for p in sorted(glob.glob(os.path.join(a.out, 'judge', '%s.*.pkl' % mode)))
              for r in pickle.load(open(p, 'rb'))]
        if not rs:
            continue
        v = [r['logit'] for r in rs]
        lab = [r['reward'] > 0 for r in rs]
        tk = [r['task'] for r in rs]
        M[mode] = dict(n=len(rs), n_pass=int(sum(lab)), within=grouped_auc(v, lab, tk, tk), pooled=pooled_auc(v, lab),
                       frac_yes=float(np.mean([r['yes'] > r['no'] for r in rs])),
                       missing=int(sum(not np.isfinite(r['yes']) and not np.isfinite(r['no']) for r in rs)))
        w = M[mode]['within']
        L.append('- %s: %d items (%d pass), within-task AUC %s, pooled %.2f, says Yes on %.0f %%, no Yes/No in top-20 %d'
                 % (mode, len(rs), sum(lab), '%.2f [%.2f, %.2f]' % (w['auc'], w['lo'], w['hi']) if w.get('auc')
                    is not None else '–', M[mode]['pooled']['auc'] or float('nan'), 100 * M[mode]['frac_yes'],
                    M[mode]['missing']))
    json.dump(M, open(os.path.join(a.out, 'judge.json'), 'w'), indent=1)
    open(os.path.join(a.out, 'judge.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('score')
    q.add_argument('--mode', choices=['turn1', 'claim'], required=True)
    q.add_argument('--dump', required=True)
    q.add_argument('--model', required=True)
    q.add_argument('--out', required=True)
    q.add_argument('--shard', type=int, default=0)
    q.add_argument('--nshards', type=int, default=1)
    q.add_argument('--max-len', type=int, default=65536)
    q = sub.add_parser('dry')
    q.add_argument('--mode', choices=['turn1', 'claim'], required=True)
    q.add_argument('--dump', required=True)
    q = sub.add_parser('analyze')
    q.add_argument('--out', required=True)
    a = p.parse_args()
    if a.cmd == 'dry':
        its = items(a.mode, a.dump)
        print(len(its), 'items; chars mean %.0f max %d' % (np.mean([len(i['prompt']) for i in its]),
                                                            max(len(i['prompt']) for i in its)))
        print(its[0]['prompt'][:3000])
        print('.....')
        print(its[0]['prompt'][-1500:])
        return
    dict(score=cmd_score, analyze=cmd_analyze)[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
