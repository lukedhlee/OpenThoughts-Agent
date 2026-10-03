#!/usr/bin/env python3
"""nvidia/Open-SWE-Traces mini-swe-agent v2 tool-mode trajectories -> relay SFT rows in render_msa.py's schema, for
mixing SWE replay into an MSA relay arm (as kimi_relay_rows.py did for the Terminus-2 arms).

Source: minisweagent/qwen38_27b/{swe-rebench-v2, scale-swe} (Qwen3.8-27B, reasoning kept, CC-BY-4.0; SWE-rebench-V2 and
Scale-SWE tasks; the same two subsets are in 09-21's training mix, so this is replay). One record = one trajectory:
system, user, then assistant turns (reasoning_content, content, bash tool calls) and role-`tool` observations.

Kept: resolved == 1; no SWE-bench Verified instance (by instance id and by (repo, PR number)); every assistant turn has
exactly one `bash` call with {"command": str} (a trajectory with any parallel-call turn is dropped whole, as is one with
an upstream format-error user message); no benchmark canary; <= 65,536 tokens; >= 1 trained token.

--keep-parallel keeps parallel-call trajectories too (mini-swe-agent 2.4.6 tool mode runs each call of a turn as its own
action and answers each with its own role-`tool` message, in call order; harbor re-sends the turn with all its calls).
Every assistant turn then has >= 1 call, each `bash` with {"command": str}, and is followed by exactly one role-`tool`
message per call before the next assistant turn (NVIDIA's tool messages carry no tool_call_id, so the pairing is by
position; a tool_call_id, if present, must equal its call's id), except the last turn, which must be the submit: its
calls unanswered and its last call's command carrying COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT (drop reasons obs_mismatch,
no_submit). Calls are rendered in order with ids call_<turn>_<j> (harbor's scheme), each observation as its own tool
turn. Single-call trajectories render byte-identically in both modes; rows gain `parallel_turns` and `calls` only under
--keep-parallel, and qa.json gains single / parallel counts and token stats.

Rendering is render_msa.render itself (09-21's chat_template.jinja, the bash tool in the Tools block, calls as
`<tool_call>{"name": "bash", "arguments": {..}}</tool_call>`, observations as <tool_response name="bash">). Every
assistant turn is a relay TEACHER turn: reasoning as <|start_think|>..<|end_think|>, every turn but the last cut to
reasoning_cap.CAP_TOKENS at a sentence / line boundary, the last kept whole; loss on content + call + <|eot_id|> of every
turn and on its reasoning only when uncut and <= --think-limit tokens. System, user and tool turns are masked.

Prompt (--prompt):
  mini    (default) system + task rewritten to what harbor mini-swe-agent-host sends under mini-swe-agent 2.4.6 mini.yaml
          for a harbor SWE-bench task: the task is the PR description (harbor's swebench adapter writes the problem
          statement, dedented and stripped, plus a newline, as instruction.md), and <system_information> is one of the
          Daytona sandbox unames our MSA evals saw (picked per trajectory by a hash of its id).
  source  NVIDIA's own system + user text verbatim (their hand-edited swebench.yaml, the git-patch submission steps).
Observations need no change in either mode: NVIDIA rendered them with mini.yaml's own observation template (JSON
{"returncode", "output"}, or output_head / output_tail past 10,000 chars), byte-identical to what the harness sends.
The trajectories' own actions stay as they are, including the swebench.yaml submission
(`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt`, which mini-swe-agent accepts as a submit).

    python render_swe_traces.py --out <dir>/rows.jsonl [--shards-per-subset N] [--limit N] [--max-kept N] [--procs 4]
                                [--keep-parallel] [--raw-dir <dir with data/minisweagent/...>]
writes rows.jsonl (one row per kept trajectory: arm 'swe_traces', task = instance id, sid, passed, the QA tags, ids,
loss) and qa.json beside it (kept / dropped by reason, token stats). Shards are read in subset-interleaved order
(rebench 0, scale 0, rebench 1, ...) and fetched from HF into --raw-dir when reached.
"""
import argparse
import collections
import hashlib
import json
import multiprocessing as mp
import os
import re
import sys
import textwrap

os.environ.setdefault('RAYON_NUM_THREADS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render_msa as rm  # noqa: E402
from hz_pool_census import CANARY  # noqa: E402

REPO = 'nvidia/Open-SWE-Traces'
SUBSETS = ('minisweagent/qwen38_27b/swe-rebench-v2', 'minisweagent/qwen38_27b/scale-swe')
N_SHARDS = 22
ARM = 'swe_traces'
VERIFIED_REPO = 'princeton-nlp/SWE-bench_Verified'
TOKDIR = '/scratch/11584/lukedhlee/models/grug-datakit-sft-20260921'
MINI_YAML = os.path.expanduser('~/snowball/envs/msa-2.4.6/minisweagent/config/mini.yaml')
COLUMNS = ['instance_id', 'repo', 'language', 'trajectory_id', 'messages', 'resolved', 'hf_dataset_name']
PR_RE = re.compile(r'^<pr_description>\nConsider the following PR description:\n(.*?)\n</pr_description>\n', re.S)
# <system_information> of the Daytona sandboxes in our mini-swe-agent-host evals (the 10 most frequent of 5,402
# trajectories under sft_eval/tb2_jobs, 2026-10-03).
UNAMES = [
    'Linux 6.8.0-90-generic #91-Ubuntu SMP PREEMPT_DYNAMIC Tue Nov 18 14:14:30 UTC 2025 x86_64',
    'Linux 6.8.0-139-generic #139-Ubuntu SMP PREEMPT_DYNAMIC Sat Aug  1 03:52:05 UTC 2026 x86_64',
    'Linux 6.17.0-35-generic #35~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Tue May 26 19:30:42 UTC 2 x86_64',
    'Linux 6.8.0-101-generic #101-Ubuntu SMP PREEMPT_DYNAMIC Mon Feb  9 10:15:05 UTC 2026 x86_64',
    'Linux 6.8.0-124-generic #124-Ubuntu SMP PREEMPT_DYNAMIC Tue May 26 13:00:45 UTC 2026 x86_64',
    'Linux 6.8.0-142-generic #142-Ubuntu SMP PREEMPT_DYNAMIC Wed Sep  2 14:24:27 UTC 2026 x86_64',
    'Linux 6.8.0-86-generic #87-Ubuntu SMP PREEMPT_DYNAMIC Mon Sep 22 18:03:36 UTC 2025 x86_64',
    'Linux 6.8.0-117-generic #117-Ubuntu SMP PREEMPT_DYNAMIC Tue May  5 19:26:24 UTC 2026 x86_64',
    'Linux 6.8.0-136-generic #136-Ubuntu SMP PREEMPT_DYNAMIC Wed Jul  1 21:53:05 UTC 2026 x86_64',
    'Linux 6.8.0-138-generic #138-Ubuntu SMP PREEMPT_DYNAMIC Fri Jul 31 22:41:49 UTC 2026 x86_64',
]
REASONS = ('unresolved', 'resolved_unknown', 'swebench_verified', 'bad_structure', 'parallel_calls', 'no_call_turn',
           'format_error_turn', 'bad_call', 'obs_mismatch', 'no_submit', 'prompt_unparsed', 'canary', 'over_64k',
           'no_trained_token')
SENTINEL = 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'


def pr_key(instance_id, repo):
    """(owner/repo, PR number) of a SWE-bench-style id (owner__repo-123) or a Scale-SWE id (owner_repo_pr123)."""
    m = re.search(r'(?:_pr|-)(\d+)$', instance_id or '')
    return ((repo or '').lower(), int(m.group(1))) if m else None


def verified_sets(path):
    if not os.path.exists(path):
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(VERIFIED_REPO, 'data/test-00000-of-00001.parquet', repo_type='dataset')
        ids = sorted(pq.read_table(p, columns=['instance_id']).column('instance_id').to_pylist())
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        json.dump(ids, open(path, 'w'))
    ids = set(json.load(open(path)))
    keys = set()
    for i in ids:
        owner_repo, num = i.rsplit('-', 1)
        keys.add((owner_repo.replace('__', '/', 1).lower(), int(num)))
    return ids, keys


def bash_args(tc):
    """A call's arguments json string if it is `bash` with exactly {"command": str}, else None."""
    f = tc.get('function') or {}
    try:
        args = json.loads(f.get('arguments'))
    except (TypeError, ValueError):
        args = None
    if f.get('name') != 'bash' or not isinstance(args, dict) or set(args) != {'command'} \
            or not isinstance(args['command'], str):
        return None
    return f['arguments']


def screen_turns_parallel(ms):
    """--keep-parallel: (drop reason or None, per-turn (reasoning, content, [arguments json], [observation] or None))."""
    turns, i = [], 2
    while i < len(ms):
        m = ms[i]
        if m['role'] == 'user':
            return 'format_error_turn', None
        if m['role'] != 'assistant':
            return 'bad_structure', None
        tcs = m.get('tool_calls') or []
        if not tcs:
            return 'no_call_turn', None
        args = [bash_args(tc) for tc in tcs]
        if any(a is None for a in args):
            return 'bad_call', None
        j = i + 1
        if j >= len(ms):                                     # the last turn: unanswered (it submitted)
            turns.append((m.get('reasoning_content'), m.get('content'), args, None))
            break
        obs = []
        for tc in tcs:                                       # one tool message per call, in call order
            o = ms[j] if j < len(ms) else None
            if o is not None and o['role'] == 'user':
                return 'format_error_turn', None
            if o is None or o['role'] != 'tool' or o.get('tool_call_id') not in (None, tc.get('id')):
                return 'obs_mismatch', None
            obs.append(o['content'] if o['content'] is not None else '')     # what harbor's message view sends
            j += 1
        if j < len(ms) and ms[j]['role'] == 'tool':          # more answers than calls
            return 'obs_mismatch', None
        turns.append((m.get('reasoning_content'), m.get('content'), args, obs))
        i = j
    if turns[-1][3] is not None or SENTINEL not in json.loads(turns[-1][2][-1])['command']:
        return 'no_submit', None
    return None, turns


def screen(r, vids, vkeys, prompt, keep_parallel=False):
    """(drop reason or None, per-turn (reasoning, content, [arguments json], [observation] or None) list, problem
    statement)."""
    if r['resolved'] != 1:
        return ('unresolved' if r['resolved'] == 0 else 'resolved_unknown'), None, None
    if r['instance_id'] in vids or pr_key(r['instance_id'], r['repo']) in vkeys:
        return 'swebench_verified', None, None
    ms = r['messages']
    if len(ms) < 3 or ms[0]['role'] != 'system' or ms[1]['role'] != 'user':
        return 'bad_structure', None, None
    if keep_parallel:
        why, turns = screen_turns_parallel(ms)
        if why:
            return why, None, None
        return prompt_task(ms, prompt, turns)
    turns, i = [], 2
    while i < len(ms):
        m = ms[i]
        if m['role'] == 'user':
            return 'format_error_turn', None, None
        if m['role'] != 'assistant':
            return 'bad_structure', None, None
        tcs = m.get('tool_calls') or []
        if len(tcs) > 1:
            return 'parallel_calls', None, None
        if not tcs:
            return 'no_call_turn', None, None
        f = tcs[0].get('function') or {}
        try:
            args = json.loads(f.get('arguments'))
        except (TypeError, ValueError):
            args = None
        if f.get('name') != 'bash' or not isinstance(args, dict) or set(args) != {'command'} \
                or not isinstance(args['command'], str):
            return 'bad_call', None, None
        obs = ms[i + 1] if i + 1 < len(ms) else None
        if obs is not None and obs['role'] == 'user':
            return 'format_error_turn', None, None
        if obs is not None and obs['role'] != 'tool':
            return 'bad_structure', None, None
        turns.append((m.get('reasoning_content'), m.get('content'), [f['arguments']], [obs['content']] if obs else None))
        i += 2
    return prompt_task(ms, prompt, turns)


def prompt_task(ms, prompt, turns):
    ps = None
    if prompt == 'mini':
        mt = PR_RE.match(ms[1]['content'])
        if not mt:
            return 'prompt_unparsed', None, None
        ps = mt.group(1)
    return None, turns, ps


def init_worker(tokdir, mini_yaml, prompt, think_limit):
    global TOK, TPL, BOS, SYS_TPL, TASK_TPL, PROMPT, THINK
    import yaml
    from jinja2 import StrictUndefined, Template
    TOK = rm.rcap.load_tokenizer(os.path.join(tokdir, 'tokenizer.json'))
    TPL = rm.load_template(os.path.join(tokdir, 'chat_template.jinja'))
    bos = json.load(open(os.path.join(tokdir, 'tokenizer_config.json'))).get('bos_token', '<|begin_of_text|>')
    BOS = bos.get('content') if isinstance(bos, dict) else bos
    agent = yaml.safe_load(open(mini_yaml))['agent']
    SYS_TPL = Template(agent['system_template'], undefined=StrictUndefined)
    TASK_TPL = Template(agent['instance_template'], undefined=StrictUndefined)
    PROMPT, THINK = prompt, think_limit


def mini_prompt(ps, sid):
    """mini.yaml's system + instance messages as mini-swe-agent renders them (jinja2, StrictUndefined) for harbor's
    swebench instruction.md, with a sandbox uname chosen by the trajectory id."""
    uname = UNAMES[int(hashlib.sha1(sid.encode()).hexdigest(), 16) % len(UNAMES)]
    system, release, rest = uname.split(' ', 2)
    version, machine = rest.rsplit(' ', 1)
    task = textwrap.dedent(ps).strip() + '\n'
    return (SYS_TPL.render(task=task, system=system, release=release, version=version, machine=machine),
            TASK_TPL.render(task=task, system=system, release=release, version=version, machine=machine))


def check_alignment(row):
    """Every trained range starts and ends on a token boundary, and loss = 1 exactly on the tokens inside one."""
    ranges = []
    for t in row['turns']:
        a, b = t['span']
        ranges.append((a, b) if (t['think_trained'] or t['think_end'] == a) else (t['think_end'], b))
    starts = {s for s, _ in row['offsets']}
    ends = {e for _, e in row['offsets']}
    if any(a not in starts or b not in ends for a, b in ranges):
        return False
    j, ok = 0, True
    for (s, e), l in zip(row['offsets'], row['loss']):
        while j < len(ranges) and s >= ranges[j][1]:
            j += 1
        inside = j < len(ranges) and s >= ranges[j][0] and e <= ranges[j][1]
        ok &= (l == 1) == inside
    return ok


def render_one(c):
    r, turns, ps = c['r'], c['turns'], c['ps']
    ms = r['messages']
    if PROMPT == 'mini':
        system, user = mini_prompt(ps, c['sid'])
    else:
        system, user = ms[0]['content'], ms[1]['content']
    msgs = [dict(role='system', content=system), dict(role='user', content=user)]
    recs = []
    for k, (reasoning, content, arguments, obs) in enumerate(turns):
        cids = [f'call_{k}_{j}' for j in range(len(arguments))]     # harbor's call_<turn>_<j>
        m = dict(role='assistant', content=content or '',
                 tool_calls=[dict(id=cid, type='function', function=dict(name='bash', arguments=a))
                             for cid, a in zip(cids, arguments)])
        if reasoning:
            m['reasoning_content'] = reasoning
        msgs.append(m)
        for cid, o in zip(cids, obs or []):                  # one tool turn per call, in call order
            if o is not None:
                msgs.append(dict(role='tool', content=o, tool_call_id=cid))
        recs.append(dict(response=dict(choices=[dict(message=dict(tool_calls=[dict(id=cids[0])]))]),
                         upstream_status=200, owner='teacher', autofix=False, repair=False, turn=k, seq=k))
    canary = any(CANARY.search(m.get('content') or '') or CANARY.search(m.get('reasoning_content') or '') or
                 any(CANARY.search(tc['function']['arguments']) for tc in m.get('tool_calls') or []) for m in msgs)
    row = rm.render(msgs, recs, TOK, TPL, BOS, think_limit=THINK)
    aligned = check_alignment(row)
    trained = sum(row['loss'])
    reason = 'canary' if canary else 'over_64k' if not row['fits'] else 'no_trained_token' if not trained else None
    t = row['turns']
    out = dict(arm=ARM, task=r['instance_id'], trial=r['trajectory_id'], sid=c['sid'], reward=1.0, passed=True,
               cause=None, teacher_turns=len(t), student_turns=0, repairs=0, takeover=None, teacher_cut_turns=0,
               run=f"{REPO}/{c['subset']}", n_tokens=row['n_tokens'], fits=row['fits'], trained_tokens=trained,
               teacher_turns_rendered=len(t), cut_turns=sum(1 for m in t if m.get('cut_at') is not None),
               long_think_masked=sum(1 for m in t if m.get('cut_at') is None and m.get('reasoning_tokens', 0) > THINK),
               think_trained_turns=sum(1 for m in t if m['think_trained']), autofix_student_turns=0,
               view_count_logged=None, view_count_rendered=None, verifier_ran=True, weak_timeout=False, noisy=False,
               leak=False, hunt=False, canary=canary, source=REPO, subset=c['subset'], repo=r['repo'],
               language=r['language'], hf_dataset_name=r['hf_dataset_name'], prompt=PROMPT,
               final_submit=SENTINEL in (turns[-1][2][-1] if turns else ''))
    if c.get('keep_parallel'):
        out.update(parallel_turns=sum(len(x[2]) > 1 for x in turns), calls=sum(len(x[2]) for x in turns))
    out.update(ids=row['ids'], loss=row['loss'])
    return reason, aligned, out


def shard_paths(a):
    if a.parquet:
        return [(p.split('/data/', 1)[-1].rsplit('/', 1)[0] if '/data/' in p else 'local', os.path.abspath(p))
                for p in a.parquet]
    order = [(s, k) for k in range(a.shard_start, min(N_SHARDS, a.shard_start + a.shards_per_subset)) for s in a.subsets]
    return [(s, f'data/{s}/train-{k:05d}-of-{N_SHARDS:05d}.parquet') for s, k in order]     # fetched when reached


def records(a, paths):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    for subset, p in paths:
        if not os.path.isabs(p):
            p = hf_hub_download(REPO, p, repo_type='dataset', local_dir=a.raw_dir)
        for batch in pq.ParquetFile(p).iter_batches(batch_size=64, columns=COLUMNS):
            for r in batch.to_pylist():
                yield subset, p, r


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1)))] if xs else None


def tok_stats(rows):
    n = [r['n_tokens'] for r in rows]
    tt = [r['trained_tokens'] for r in rows]
    return dict(rows=len(rows), tokens=sum(n), trained_tokens=sum(tt), n_tokens_p10=pct(n, .1), n_tokens_p50=pct(n, .5),
                n_tokens_p90=pct(n, .9), n_tokens_max=max(n, default=None), trained_p50=pct(tt, .5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True, help='rows.jsonl path; qa.json is written beside it')
    ap.add_argument('--subsets', nargs='+', default=list(SUBSETS))
    ap.add_argument('--shards-per-subset', type=int, default=N_SHARDS)
    ap.add_argument('--shard-start', type=int, default=0)
    ap.add_argument('--parquet', nargs='+', help='local shards instead of the HF download')
    ap.add_argument('--raw-dir', help='HF download dir (default <out dir>/raw)')
    ap.add_argument('--tokenizer-dir', default=TOKDIR)
    ap.add_argument('--mini-yaml', default=MINI_YAML, help='mini-swe-agent 2.4.6 config/mini.yaml')
    ap.add_argument('--prompt', choices=('mini', 'source'), default='mini')
    ap.add_argument('--verified-ids', help='SWE-bench Verified instance ids json (default <out dir>/swebench_verified_ids'
                                           '.json, fetched from HF when missing)')
    ap.add_argument('--think-limit', type=int, default=rm.THINK_LIMIT)
    ap.add_argument('--limit', type=int, help='stop after reading this many records')
    ap.add_argument('--max-kept', type=int, help='stop once this many rows are kept')
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--keep-parallel', action='store_true',
                    help='keep trajectories with parallel-call turns (one tool message per call, in order)')
    a = ap.parse_args()
    out_dir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(out_dir, exist_ok=True)
    a.raw_dir = a.raw_dir or os.path.join(out_dir, 'raw')
    vids, vkeys = verified_sets(a.verified_ids or os.path.join(out_dir, 'swebench_verified_ids.json'))
    paths = shard_paths(a)

    overall = collections.Counter()
    dropped = collections.Counter()
    per_subset = collections.defaultdict(collections.Counter)
    shards_read, kept, misaligned, sids = [], [], 0, set()

    def candidates():
        for subset, p, r in records(a, paths):
            if a.limit and overall['records'] >= a.limit:
                return
            if not shards_read or shards_read[-1] != p:
                shards_read.append(p)
            overall['records'] += 1
            per_subset[subset]['records'] += 1
            overall[f"resolved={r['resolved']}"] += 1
            ms = r['messages']
            overall['has_parallel_turn'] += any(len(m.get('tool_calls') or []) > 1 for m in ms if m['role'] == 'assistant')
            overall['swebench_verified_any'] += r['instance_id'] in vids or pr_key(r['instance_id'], r['repo']) in vkeys
            overall['verified_repo_any'] += (r['repo'] or '').lower() in VREPOS
            why, turns, ps = screen(r, vids, vkeys, a.prompt, keep_parallel=a.keep_parallel)
            if why:
                dropped[why] += 1
                per_subset[subset][why] += 1
                continue
            yield dict(r=r, turns=turns, ps=ps, subset=subset, sid=f"ost:{subset.rsplit('/', 1)[-1]}:{r['trajectory_id']}",
                       keep_parallel=a.keep_parallel)

    global VREPOS
    VREPOS = {k[0] for k in vkeys}
    tmp = a.out + '.tmp'
    gen = candidates()
    with open(tmp, 'w') as f, mp.Pool(a.procs, init_worker, (a.tokenizer_dir, a.mini_yaml, a.prompt, a.think_limit)) as pool:
        while not (a.max_kept and len(kept) >= a.max_kept):
            batch = [c for _, c in zip(range(8 * a.procs), gen)]     # bounded: the reader never runs ahead of rendering
            if not batch:
                break
            for why, aligned, row in pool.imap(render_one, batch):
                misaligned += not aligned
                if why:
                    dropped[why] += 1
                    per_subset[row['subset']][why] += 1
                    continue
                if row['sid'] in sids:
                    dropped['duplicate_sid'] += 1
                    continue
                if a.max_kept and len(kept) >= a.max_kept:
                    dropped['over_max_kept'] += 1
                    continue
                sids.add(row['sid'])
                per_subset[row['subset']]['kept'] += 1
                if a.keep_parallel:
                    per_subset[row['subset']]['kept_parallel' if row['parallel_turns'] else 'kept_single'] += 1
                kept.append({k: v for k, v in row.items() if k not in ('ids', 'loss')})
                f.write(json.dumps(row) + '\n')
    os.replace(tmp, a.out)

    n = [r['n_tokens'] for r in kept]
    tt = [r['trained_tokens'] for r in kept]
    turns = [r['teacher_turns'] for r in kept]
    qa = dict(source=REPO, subsets=a.subsets, prompt=a.prompt, think_limit=a.think_limit, cap_tokens=rm.rcap.CAP_TOKENS,
              max_tokens=rm.MAX_TOKENS, shards_read=[os.path.relpath(p, a.raw_dir) if p.startswith(a.raw_dir) else p
                                                     for p in shards_read],
              swebench_verified_ids=len(vids), records=overall['records'], kept=len(kept),
              dropped_first_reason={k: dropped[k] for k in REASONS + ('duplicate_sid', 'over_max_kept') if dropped[k]},
              all_records=dict(overall), misaligned_rows=misaligned,
              per_subset={s: dict(c) for s, c in per_subset.items()},
              kept_stats=dict(tasks=len({r['task'] for r in kept}), tokens=sum(n), trained_tokens=sum(tt),
                              n_tokens_p10=pct(n, .1), n_tokens_p50=pct(n, .5), n_tokens_p90=pct(n, .9),
                              n_tokens_max=max(n, default=None), trained_p50=pct(tt, .5),
                              turns_p50=pct(turns, .5), turns_max=max(turns, default=None),
                              turns_total=sum(turns), cut_turns=sum(r['cut_turns'] for r in kept),
                              think_trained_turns=sum(r['think_trained_turns'] for r in kept),
                              long_think_masked=sum(r['long_think_masked'] for r in kept),
                              final_submit=sum(r['final_submit'] for r in kept),
                              languages=dict(collections.Counter(r['language'] for r in kept).most_common()),
                              verified_repo_rows=sum((r['repo'] or '').lower() in VREPOS for r in kept)))
    if a.keep_parallel:
        qa['keep_parallel'] = True
        ks = qa['kept_stats']
        ks.update(n_tokens_p25=pct(n, .25), n_tokens_p75=pct(n, .75), n_tokens_p99=pct(n, .99),
                  trained_p10=pct(tt, .1), trained_p90=pct(tt, .9),
                  parallel_rows=sum(1 for r in kept if r['parallel_turns']),
                  parallel_turns=sum(r['parallel_turns'] for r in kept), calls_total=sum(r['calls'] for r in kept),
                  by_kind={k: tok_stats([r for r in kept if bool(r['parallel_turns']) == (k == 'parallel')])
                           for k in ('single', 'parallel')},
                  by_subset_kind={f"{s.rsplit('/', 1)[-1]}:{k}": tok_stats(
                      [r for r in kept if r['subset'] == s and bool(r['parallel_turns']) == (k == 'parallel')])
                      for s in a.subsets for k in ('single', 'parallel')})
    qa_path = os.path.join(out_dir, os.path.splitext(os.path.basename(a.out))[0].replace('rows', 'qa') + '.json')
    json.dump(qa, open(qa_path, 'w'), indent=1)
    print(json.dumps(qa, indent=1))


TOK = TPL = BOS = SYS_TPL = TASK_TPL = PROMPT = THINK = None
VREPOS = set()

if __name__ == '__main__':
    main()
