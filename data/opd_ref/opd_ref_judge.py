"""The teacher as a judge instead of a log-likelihood: a tokenizer-free alternative reward for PivotRL x OPD.

Qwen3.8-27B (thinking off) answers one Yes/No question per item; the score is log P(Yes) - log P(No) at its first
answer token. The same same-state and done-claim tests as opd_ref_turns.py, so the numbers line up with CompassOPD's.

    turn1  each rollout's first turn (its reasoning and commands) with the task's issue: "is this a good first step?"
           Items = the rollouts of GRPO groups with both outcomes (opd_ref_branch.py's --max-depth 0 selection).
    claim  the session up to the rollout's first task_complete:true (actions and terminal output, thinking dropped):
           "is the issue actually fixed?"
    value  the session after its first k turns (k in --ks, rollouts still working, no claim yet): "is the agent on
           track to a correct fix?" Compared within task at the same k, so it asks whether the teacher can see
           mid-episode what its log-likelihood could not (CompassOPD's first-k-turn AUCs were ~0.50).
    dense  the same question after EVERY turn before the first done-claim, for a per-turn reward
           r_t = V_t - V_{t-1} (V = P(Yes)). dense-analyze asks whether those changes are clean enough to credit
           single turns: how much V moves on turns that change nothing (read-only commands), and whether, among
           rollouts of a task with the same V before turn t, the one whose turn t raised V passes more often.
    export transcripts of chosen rollouts with numbered turns, for an independent turn-by-turn audit.

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


def session_steps(msgs, upto=None):
    """-> (step texts up to the first claim or the first `upto` turns, claimed?)"""
    steps = []
    for j in range(1, len(msgs), 2):
        act = ''.join(t for kd, t in split_reply(msgs[j]['content'].strip()) if kd == 'action').strip()
        claimed = bool(DONE_RE.search(act))
        s = '[step %d%s]\n%s' % (len(steps) + 1, ' - the agent declares the task complete' if claimed else '', act)
        if not claimed and j + 1 < len(msgs):
            s += '\n[terminal output]\n' + cut(msgs[j + 1]['content'])
        steps.append(s)
        if claimed or (upto is not None and len(steps) == upto):
            return steps, claimed
    return steps, False


def fit(steps):
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
    return body


def claim_item(tr):
    msgs = tr['messages']
    steps, claimed = session_steps(msgs)
    if not claimed:
        return None
    body = fit(steps)
    return render('A coding agent worked on the task below in a Linux terminal, in a checkout of the repository at '
                  '/testbed, and has now declared the task complete. Hidden tests will check whether the issue is '
                  'actually fixed.\n\n<task>\n%s\n</task>\n\n<session>\n%s\n</session>\n\nBased on the session, is '
                  'the issue actually fixed, so that the hidden tests will pass? Answer with one word, Yes or No.'
                  % (issue_of(msgs[0]['content']), body))


def value_item(tr, k):
    msgs = tr['messages']
    if sum(m['role'] == 'assistant' for m in msgs) <= k:
        return None                      # finished by turn k: nothing left to predict
    steps, claimed = session_steps(msgs, upto=k)
    if claimed or len(steps) < k:
        return None
    return render('A coding agent is working on the task below in a Linux terminal, in a checkout of the repository '
                  'at /testbed. It is not finished yet. Here is its session so far.\n\n<task>\n%s\n</task>\n\n'
                  '<session>\n%s\n</session>\n\nJudge the work so far as an expert engineer would: is the agent on '
                  'track to fix the issue correctly, so that the hidden tests will pass when it is done? Answer with '
                  'one word, Yes or No.' % (issue_of(msgs[0]['content']), fit(steps)))


EDIT_RE = re.compile(r'(sed\s+-i|perl\s+-[a-z]*i|\btee\b|\bpatch\b|git\s+(apply|checkout|stash|reset)|\brm\s|\bmv\s|'
                     r'\bcp\s|>\s*[^&/\s]|>>|<<\s*[\'"]?EOF|open\([^)]*[\'"][wa]\+?[\'"])')
TEST_RE = re.compile(r'(\bpytest\b|\bpython3?\s+(-m\s+pytest|-m\s+unittest|[\w./-]+\.py)|\btox\b|\bmake\s+test)')
KEYS_RE = re.compile(r'"keystrokes"\s*:\s*"((?:[^"\\]|\\.)*)"')


def classify(action):
    """edit > test > read: what the turn's commands do to the sandbox (keystrokes only; a heuristic)."""
    keys = ' ; '.join(KEYS_RE.findall(action)).replace('2>&1', '').replace('>/dev/null', '').replace('> /dev/null', '')
    if not keys.strip():
        return 'none'
    if EDIT_RE.search(keys):
        return 'edit'
    if TEST_RE.search(keys):
        return 'test'
    return 'read'


VALUE_Q = ('Judge the work so far as an expert engineer would: is the agent on track to fix the issue correctly, so '
           'that the hidden tests will pass when it is done? Answer with one word, Yes or No.')
VALUE_Q2 = ('Suppose the agent keeps working from this point. As an experienced reviewer, do you expect its final patch to '
            'resolve the issue and pass the hidden tests? Reply Yes or No, one word.')


def info_lines(obs):
    """Lines of terminal output that are not the echoed prompt/commands or blank: 0 = the turn showed nothing new."""
    return sum(1 for ln in obs.splitlines() if ln.strip() and not ln.startswith('$')
               and not ln.startswith('New Terminal Output') and not ln.startswith('Current Terminal Screen'))


def dense_items(tr, para_every=8):
    """(k, prompt, class of turn k, info lines of its output, paraphrased?) after every turn before the first
    done-claim; every para_every-th turn is asked twice (a reworded question) to measure the judge's own wobble."""
    msgs = tr['messages']
    steps, claimed = session_steps(msgs)
    n = len(steps) - (1 if claimed else 0)
    out = []
    for k in range(1, n + 1):
        act = ''.join(t for kd, t in split_reply(msgs[2 * k - 1]['content'].strip()) if kd == 'action')
        info = info_lines(msgs[2 * k]['content']) if 2 * k < len(msgs) else 0
        head = ('A coding agent is working on the task below in a Linux terminal, in a checkout of the repository at '
                '/testbed. It is not finished yet. Here is its session so far.\n\n<task>\n%s\n</task>\n\n<session>\n'
                '%s\n</session>\n\n' % (issue_of(msgs[0]['content']), fit(steps[:k])))
        out.append((k, render(head + VALUE_Q), classify(act), info, False))
        if k % para_every == 0:
            out.append((k, render(head + VALUE_Q2), classify(act), info, True))
    return out, n


def items(mode, dump, ks=(3, 6, 10)):
    if mode == 'turn1':
        kept, _ = select(paths_of(dump), 0)
        return [dict(trial=tr['trial'], task=tr['task'], reward=tr['reward'], k=0, prompt=turn1_item(tr))
                for tr in kept]
    out = []
    for p in paths_of(dump):
        for line in open(p):
            d = json.loads(line)
            tr = load_dump_row(d)
            if tr is None:
                continue
            task = '%s@%s' % (d['uid'], d.get('step'))
            if mode == 'dense':
                its, n = dense_items(tr)
                out += [dict(trial=tr['trial'], task=task, reward=tr['reward'], k=k, n_turns=n, cls=c, info=inf,
                             para=pa, prompt=pr) for k, pr, c, inf, pa in its]
                continue
            if mode == 'claim':
                pr = claim_item(tr)
                if pr is not None:
                    out.append(dict(trial=tr['trial'], task=task, reward=tr['reward'], k=-1, prompt=pr))
                continue
            for k in ks:
                pr = value_item(tr, k)
                if pr is not None:
                    out.append(dict(trial=tr['trial'], task=task, reward=tr['reward'], k=k, prompt=pr))
    return out


def cmd_score(a):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    its = items(a.mode, a.dump, tuple(int(k) for k in a.ks.split(',')))
    mine = [it for it in its if zlib.crc32(it['task'].encode()) % a.nshards == a.shard]   # a task's items share
    # a shard, so its common prompt prefix is served from the prefix cache
    out = os.path.join(a.out, 'judge', '%s.%d.pkl' % (a.mode, a.shard))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # dense: each rollout's queries share a growing prefix. Prefix caching on this hybrid model works per 784-token
    # block (mamba 'align' mode) and only for blocks already computed, so rollouts go in groups of G with at most G
    # requests in flight, each group ordered by turn: a rollout's next query starts after its previous one is cached.
    G = 8
    llm = LLM(model=a.model, max_model_len=a.max_len, gpu_memory_utilization=0.85, enable_prefix_caching=True,
              max_num_batched_tokens=16384, max_num_seqs=G if a.mode == 'dense' else 32,
              limit_mm_per_prompt={'image': 0, 'video': 0}, seed=0)
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
    if a.mode == 'dense':
        trials = list(dict.fromkeys(it['trial'] for it in mine))
        rank = {t: j for j, t in enumerate(trials)}
        outs = [None] * len(mine)
        ntok = 0
        for g0 in range(0, len(trials), G):
            idx = sorted((j for j, it in enumerate(mine) if g0 <= rank[it['trial']] < g0 + G),
                         key=lambda j: (mine[j]['k'], mine[j].get('para', False), rank[mine[j]['trial']]))
            for j, o in zip(idx, llm.generate([prompts[j] for j in idx], sp, use_tqdm=False)):
                outs[j] = o
                ntok += mine[j]['n_tokens']
            if (g0 // G) % 4 == 0:
                print('dense shard %d: %d/%d rollouts, %.0f prompt tok/s' % (
                    a.shard, min(g0 + G, len(trials)), len(trials), ntok / (time.time() - t0)), flush=True)
    else:
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
        res.append(dict(trial=it['trial'], task=it['task'], reward=it['reward'], k=it['k'], n_tokens=it['n_tokens'],
                        cls=it.get('cls'), n_turns=it.get('n_turns'), info=it.get('info'), para=it.get('para', False),
                        yes=float(yes), no=float(no), logit=float(np.clip(yes, -40, 0) - np.clip(no, -40, 0))))
    pickle.dump(res, open(out, 'wb'))
    print('%s shard %d: %d items, %d over length, %.0f s, %d tokens' % (a.mode, a.shard, len(res), over,
                                                                       time.time() - t0,
                                                                       sum(r['n_tokens'] for r in res)))


def matched_auc(rs, ratio=1.25, n_boot=2000):
    """Within-task AUC over pass/fail pairs whose prompts differ in length by less than `ratio`."""
    by = {}
    for r in rs:
        by.setdefault(r['task'], []).append(r)
    per = []
    for x in by.values():
        pr = [(p, f) for p in x if p['reward'] > 0 for f in x if f['reward'] <= 0
              if abs(np.log(p['n_tokens'] / f['n_tokens'])) < np.log(ratio)]
        if pr:
            per.append(np.mean([(p['logit'] > f['logit']) + 0.5 * (p['logit'] == f['logit']) for p, f in pr]))
    if not per:
        return None
    per = np.asarray(per)
    rng = np.random.default_rng(0)
    bs = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(n_boot)]
    return dict(auc=float(per.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                n_tasks=len(per))


def cmd_analyze(a):
    M, L = {}, ['# teacher-as-judge (Qwen3.8-27B, thinking off, log P(Yes) - log P(No))', '']
    groups = []
    for mode in ('turn1', 'claim', 'value'):
        rs = [r for p in sorted(glob.glob(os.path.join(a.out, 'judge', '%s.*.pkl' % mode)))
              for r in pickle.load(open(p, 'rb'))]
        if mode == 'value':
            groups += [('value_k%d' % k, [r for r in rs if r['k'] == k]) for k in sorted({r['k'] for r in rs})]
        elif rs:
            groups.append((mode, rs))
    for mode, rs in groups:
        if not rs:
            continue
        v = [r['logit'] for r in rs]
        lab = [r['reward'] > 0 for r in rs]
        tk = [r['task'] for r in rs]
        M[mode] = dict(n=len(rs), n_pass=int(sum(lab)), within=grouped_auc(v, lab, tk, tk), pooled=pooled_auc(v, lab),
                       frac_yes=float(np.mean([r['yes'] > r['no'] for r in rs])),
                       missing=int(sum(not np.isfinite(r['yes']) and not np.isfinite(r['no']) for r in rs)),
                       length_matched=matched_auc(rs))
        w = M[mode]['within']
        lm = M[mode]['length_matched']
        L.append('- %s: %d items (%d pass), within-task AUC %s, length-matched (1.25x) %s, pooled %.2f, says Yes on '
                 '%.0f %%, no Yes/No in top-20 %d'
                 % (mode, len(rs), sum(lab), '%.2f [%.2f, %.2f]' % (w['auc'], w['lo'], w['hi']) if w.get('auc')
                    is not None else '–', '%.2f [%.2f, %.2f] (%d tasks)' % (lm['auc'], lm['lo'], lm['hi'],
                                                                          lm['n_tasks']) if lm else '–',
                    M[mode]['pooled']['auc'] or float('nan'), 100 * M[mode]['frac_yes'], M[mode]['missing']))
    json.dump(M, open(os.path.join(a.out, 'judge.json'), 'w'), indent=1)
    open(os.path.join(a.out, 'judge.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


def cmd_dense_analyze(a):
    allr = [r for p in sorted(glob.glob(os.path.join(a.out, 'judge', 'dense.*.pkl'))) for r in pickle.load(open(p, 'rb'))]
    rs = [r for r in allr if not r.get('para')]
    para = {(r['trial'], r['k']): r for r in allr if r.get('para')}
    tr = {}
    for r in rs:
        r['v'] = float(1 / (1 + np.exp(-r['logit'])))
        tr.setdefault(r['trial'], []).append(r)
    for x in tr.values():
        x.sort(key=lambda r: r['k'])
        for j, r in enumerate(x):
            r['dv'] = r['v'] - x[j - 1]['v'] if j > 0 and x[j - 1]['k'] == r['k'] - 1 else np.nan
            r['v_prev'] = x[j - 1]['v'] if j > 0 else np.nan
    L = ['# dense judge: per-turn changes of V = P(on track)', '',
         '%d turns judged in %d rollouts (%d pass)' % (len(rs), len(tr), sum(x[0]['reward'] > 0 for x in tr.values())),
         '', '## 1. how much V moves, by what the turn did (|dV| median / mean dV)', '',
         '| turn kind | n | passing rollouts | failing rollouts |', '|---|---|---|---|']
    M = {'by_class': {}}
    for r in rs:
        if r['cls'] != 'none' and r.get('info') == 0:
            r['cls'] = 'no-info'          # commands ran but printed nothing new
    for c in ('read', 'test', 'edit', 'no-info', 'none'):
        cell = []
        for lab in (True, False):
            d = np.array([r['dv'] for r in rs if r['cls'] == c and (r['reward'] > 0) == lab and np.isfinite(r['dv'])])
            M['by_class']['%s_%s' % (c, 'pass' if lab else 'fail')] = dict(
                n=len(d), med_abs=float(np.median(np.abs(d))) if len(d) else None,
                mean=float(d.mean()) if len(d) else None)
            cell.append('%.3f / %+.3f (n %d)' % (np.median(np.abs(d)), d.mean(), len(d)) if len(d) else '–')
        L.append('| %s | %d | %s | %s |' % (c, sum(r['cls'] == c for r in rs), cell[0], cell[1]))
    # 1b. the judge's own wobble: the same session asked with a reworded question
    pr = [(r['v'], 1 / (1 + np.exp(-para[(r['trial'], r['k'])]['logit']))) for r in rs if (r['trial'], r['k']) in para]
    if pr:
        d = np.array([x - y for x, y in pr])
        M['retest'] = dict(n=len(d), med_abs=float(np.median(np.abs(d))),
                           corr=float(np.corrcoef([x for x, _ in pr], [y for _, y in pr])[0, 1]))
        L += ['', 'Reworded question on the same session (%d pairs): median |V - V\'| %.3f, correlation %.2f' % (
            len(d), M['retest']['med_abs'], M['retest']['corr'])]
    # 2. matched credit: same task, same turn t, same V before t (|diff| < eps): does the bigger dV_t pass more?
    by = {}
    for r in rs:
        if np.isfinite(r['dv']):
            by.setdefault((r['task'], r['k']), []).append(r)
    L += ['', '## 2. matched credit (same task, same turn, V before the turn within eps; AUC of dV_t for pass vs fail)',
          '']
    for eps in (0.03, 0.05, 0.10, 1.0):
        per = {}
        for (task, k), x in by.items():
            for p_ in x:
                if p_['reward'] <= 0:
                    continue
                for f in x:
                    if f['reward'] > 0 or abs(p_['v_prev'] - f['v_prev']) >= eps:
                        continue
                    per.setdefault(task, []).append((p_['dv'] > f['dv']) + 0.5 * (p_['dv'] == f['dv']))
        if not per:
            continue
        pt = np.array([np.mean(v) for v in per.values()])
        rng = np.random.default_rng(0)
        bs = [pt[rng.integers(0, len(pt), len(pt))].mean() for _ in range(2000)]
        M['matched_eps%.2f' % eps] = dict(auc=float(pt.mean()), lo=float(np.percentile(bs, 2.5)),
                                          hi=float(np.percentile(bs, 97.5)), n_tasks=len(pt),
                                          n_pairs=int(sum(len(v) for v in per.values())))
        L.append('- eps %.2f: %.2f [%.2f, %.2f] over %d tasks, %d pairs' % (eps, pt.mean(), np.percentile(bs, 2.5),
                                                                          np.percentile(bs, 97.5), len(pt),
                                                                          M['matched_eps%.2f' % eps]['n_pairs']))
    # 3. V by turn, pass vs fail (the trajectory the reward would follow)
    L += ['', '## 3. mean V after turn k, passing vs failing rollouts', '', '| k | pass | fail |', '|---|---|---|']
    for k in (1, 2, 3, 5, 8, 12, 16, 24, 32):
        vp = [r['v'] for r in rs if r['k'] == k and r['reward'] > 0]
        vf = [r['v'] for r in rs if r['k'] == k and r['reward'] <= 0]
        if vp and vf:
            L.append('| %d | %.2f (n %d) | %.2f (n %d) |' % (k, np.mean(vp), len(vp), np.mean(vf), len(vf)))
    pickle.dump({t: [(r['k'], r['v'], r['cls']) for r in x] for t, x in tr.items()},
                open(os.path.join(a.out, 'dense_table.pkl'), 'wb'))
    json.dump(M, open(os.path.join(a.out, 'dense.json'), 'w'), indent=1)
    open(os.path.join(a.out, 'dense.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


def cmd_export(a):
    """Numbered transcripts (issue, then per turn: the agent's JSON and the terminal output) up to and including the
    first done-claim, one file per trial, for the turn-by-turn audit. Thinking is left out, as in the judge's view."""
    want = set(a.trials.split(','))
    os.makedirs(a.out, exist_ok=True)
    n = 0
    for p in paths_of(a.dump):
        for line in open(p):
            d = json.loads(line)
            if (d.get('trajectory_id') or d['uid']) not in want:
                continue
            tr = load_dump_row(d)
            msgs = tr['messages']
            parts = ['TASK\n' + issue_of(msgs[0]['content']), '']
            for j in range(1, len(msgs), 2):
                act = ''.join(t for kd, t in split_reply(msgs[j]['content'].strip()) if kd == 'action').strip()
                parts.append('=== TURN %d ===\n%s' % ((j + 1) // 2, act))
                if DONE_RE.search(act):
                    break
                if j + 1 < len(msgs):
                    parts.append('--- terminal output after turn %d ---\n%s' % ((j + 1) // 2,
                                                                              cut(msgs[j + 1]['content'], 2000, 3000)))
            parts.append('\nFINAL RESULT: the hidden tests %s.' % ('PASSED' if tr['reward'] > 0 else 'FAILED'))
            open(os.path.join(a.out, tr['trial'] + '.txt'), 'w').write('\n'.join(parts))
            n += 1
    print('wrote', n, 'transcripts to', a.out)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('score')
    q.add_argument('--mode', choices=['turn1', 'claim', 'value', 'dense'], required=True)
    q.add_argument('--dump', required=True)
    q.add_argument('--ks', default='3,6,10')
    q.add_argument('--model', required=True)
    q.add_argument('--out', required=True)
    q.add_argument('--shard', type=int, default=0)
    q.add_argument('--nshards', type=int, default=1)
    q.add_argument('--max-len', type=int, default=65536)
    q = sub.add_parser('dry')
    q.add_argument('--mode', choices=['turn1', 'claim', 'value', 'dense'], required=True)
    q.add_argument('--dump', required=True)
    q.add_argument('--ks', default='3,6,10')
    q = sub.add_parser('analyze')
    q.add_argument('--out', required=True)
    q = sub.add_parser('dense-analyze')
    q.add_argument('--out', required=True)
    q = sub.add_parser('export')
    q.add_argument('--dump', required=True)
    q.add_argument('--trials', required=True, help='comma-separated trajectory ids')
    q.add_argument('--out', required=True)
    a = p.parse_args()
    if a.cmd == 'dry':
        its = items(a.mode, a.dump, tuple(int(k) for k in a.ks.split(',')))
        import collections
        print(collections.Counter(i['k'] for i in its))
        print(len(its), 'items; chars mean %.0f max %d' % (np.mean([len(i['prompt']) for i in its]),
                                                            max(len(i['prompt']) for i in its)))
        print(its[0]['prompt'][:3000])
        print('.....')
        print(its[0]['prompt'][-1500:])
        return
    dict(score=cmd_score, analyze=cmd_analyze, export=cmd_export, **{'dense-analyze': cmd_dense_analyze})[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
