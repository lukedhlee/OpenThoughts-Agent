#!/usr/bin/env python3
"""microsoft/Orchard `swe` config (mini-swe-agent v2 tool-mode pools) -> relay SFT rows in render_si2ca.py's schema and
rendering, for mixing SWE replay into an MSA relay arm.

Source (MIT): 107,185 trajectories in 19 shards, one per row (`tools`, `messages`, `metadata` json). Pools
(metadata.source): rebench-M2.5 (MiniMax-M2.5, SWE-rebench), rebench-Qwen3.5 (Qwen3.5-397B-A17B, SWE-rebench),
scaleswe-M2.5 (MiniMax-M2.5, Scale-SWE) under mini-swe-agent; oh-bench-M2.5 under OpenHands (dropped: another harness).
Resolved mini-swe-agent rows are all in shards 0-11 (62,436 rows, 12,449 tasks, ~4 samples per task); shards 12-18
hold only OpenHands and unresolved rows (checked 2026-10-03), so the default reads shards 0-11.
Format: one `bash` tool (byte-identical to the harness's), mini-swe-agent's benchmarks/swebench.yaml system + instance
(Scale-SWE pool: the instance's two /testbed lines reworded, see SCALE_EDITS), swebench.yaml's text observation
template, one call per turn in every row seen, the last turn the submit (unanswered). MiniMax-M2.5 turns carry their
reasoning inline in content as `<think>..</think>` + optional prose; Qwen3.5 turns carry NO reasoning (the release
has no reasoning field and ~92% of its turns have empty content: the turn is the bare tool call).
The card's "identifiers and paths are scrubbed" left no trace we could find (2026-10-03): 300/300 SWE-rebench PR texts
equal nebius/SWE-rebench's problem_statement, 80/80 first `cat`s of a file before any edit equal the file on GitHub at
the task's base_commit (so command paths are real too), 263/266 such reads on Scale-SWE tasks equal nvidia/Open-SWE-
Traces' (3 differ only by CRLF, which NVIDIA folds), and no placeholder marker recurs across repos. qa.json carries a
marker census over every screened record (`scrub_marker_rows`).

Kept: resolved; mini-swe-agent pool; no SWE-bench Verified instance (by instance id and by (repo, PR number), repo
names compared with '/' and '_' runs folded, since Orchard writes owner__repo / owner_repo); the bash tool; the
swebench.yaml prompt (re-rendered byte-identical from the extracted task); every assistant turn has >= 1 `bash` call
with {"command": str}, each answered in order by a tool message with its id (parallel calls kept), the last turn's
calls unanswered with COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT in a call (the source harness ends on an accepted submit,
the same rule as 2.4.6: an earlier sentinel call with returncode != 0 is answered and the run goes on); no observation
that 2.4.6 would have taken as the submit before the end (early_submit); a well-formed reasoning split (one leading
<think>..</think> or none); no upstream format-error message; every observation parses as the source template's render
(round trip byte for byte); no harness artefact: an <exception> other than the form our harness sends (`An error
occurred while executing the command: Command ..`, its timeouts), i.e. Orchard Env's `Agent connection error: ..`,
`No pod IP available for sandbox ..`, `An error occurred while executing the command: 500 Server Error ..` (its sandbox
API), and its own timeout wording `Command timed out after 180s`; no benchmark canary; <= 65,536 tokens; >= 1 trained
token.
One trajectory per task: candidates ordered by teacher (--prefer, default MiniMax-M2.5 first: the stronger SWE teacher,
and the one with reasoning), then by sha1('<instance_id>:<pool>:<sample_idx>'); a task's candidates are rendered in
that order until one passes.

Observations are re-rendered into mini.yaml's observation template (render_si2ca.parse_obs + load_templates: exact,
long outputs included). Rendering is render_msa.render through render_swe_traces's helpers, as render_si2ca does: every
assistant turn a relay TEACHER turn, reasoning as <|start_think|>..<|end_think|> (every turn but the last cut to
reasoning_cap.CAP_TOKENS, trained only when uncut and <= --think-limit tokens), then the prose, then the calls; loss
on content + calls + <|eot_id|>; system, user and tool turns masked. A Qwen3.5 turn renders as the bare call(s).

Prompt (--prompt):
  mini    (default) render_swe_traces.mini_prompt: system + task as harbor mini-swe-agent-host sends them under
          mini-swe-agent 2.4.6 mini.yaml for a harbor SWE-bench task (the PR description with CRLF / CR folded to LF as
          harbor's instruction.md read does, dedented and stripped, plus a newline; a Daytona uname by a hash of the sid).
  source  Orchard's own system + user verbatim.
The trajectories' own actions stay as they are (Scale-SWE runs work under /workspace/<repo>, SWE-rebench runs under
/testbed), including the submit.

    python render_orchard.py --out <dir>/rows.jsonl [--raw-dir <orchard download>] [--shards 0 1 ..] [--max-tasks N]
                             [--teachers MiniMax-M2.5 Qwen3.5-397B-A17B] [--prefer MiniMax-M2.5 ..] [--procs 4]
writes rows.jsonl (one row per kept task: arm 'orchard', task = instance id, teacher, source swe-rebench / scale-swe,
pool, sample_idx, sid, the QA tags, ids, loss) and qa.json beside it (records kept / dropped by first reason, per pool
and per teacher, selection counts, observation and marker census, token stats).
"""
import argparse
import collections
import hashlib
import json
import multiprocessing as mp
import os
import re
import sys

os.environ.setdefault('RAYON_NUM_THREADS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render_msa as rm  # noqa: E402
import render_si2ca as rs  # noqa: E402
import render_swe_traces as rst  # noqa: E402
from hz_pool_census import CANARY  # noqa: E402

REPO = 'microsoft/Orchard'
N_SHARDS = 19
SHARDS = tuple(range(12))
ARM = 'orchard'
RAW = '/scratch/11584/lukedhlee/experiments/sft_data/orchard_raw'
COLUMNS = ['tools', 'messages', 'metadata']
POOLS = {'rebench-M2.5': 'swe-rebench', 'rebench-Qwen3.5': 'swe-rebench', 'scaleswe-M2.5': 'scale-swe'}
TEACHERS = ('MiniMax-M2.5', 'Qwen3.5-397B-A17B')
SENTINEL = rs.SENTINEL
PR_RE = rs.PR_RE
# Scale-SWE pool's instance message = swebench.yaml's with these two lines reworded (its runs are not in /testbed)
SCALE_EDITS = (('- MODIFY: Regular source code files in /testbed (this is the working directory for all your subsequent '
                'commands)', '- MODIFY: Regular source code files in the current working directory'),
               ('- DO NOT MODIFY: Tests, configuration files (pyproject.toml, setup.cfg, etc.)',
                '- DO NOT MODIFY: Tests, configuration files'))
OWN_EXCEPTION = 'An error occurred while executing the command: Command '      # the harness's (timeouts) form
THINK_RE = re.compile(r'\A<think>(.*?)</think>(.*)\Z', re.S)
# placeholder / redaction markers an anonymiser would leave (Presidio entity tags, [REDACTED] ...): census only
MARKERS = re.compile(r'<(?:PERSON|EMAIL_ADDRESS|PHONE_NUMBER|IP_ADDRESS|LOCATION|DATE_TIME|US_SSN|CREDIT_CARD|NRP|'
                     r'ORGANIZATION|USERNAME|REDACTED|PATH|FILE_PATH|IDENTIFIER|ANON\w*)>|\[(?:REDACTED|ANONYMIZED|'
                     r'SCRUBBED|PATH|USER|NAME|EMAIL)\]|\bREDACTED\b|\bSCRUBBED\b')
REASONS = ('openhands', 'unresolved', 'teacher_excluded', 'swebench_verified', 'bad_tools', 'bad_structure',
           'prompt_unparsed', 'format_error_turn', 'no_call_turn', 'bad_call', 'bad_think', 'obs_unparsed',
           'harness_error', 'early_submit', 'no_submit', 'canary', 'over_64k', 'no_trained_token',
           'task_kept_other_candidate')


def nrepo(repo):
    """owner/repo, owner__repo and owner_repo in one form."""
    return re.sub(r'[/_]+', '_', (repo or '').lower())


def verified(iid, repo, vids, vnorm):
    if iid in vids:
        return True
    m = re.search(r'(?:_pr|-)(\d+)$', iid or '')
    return bool(m) and (nrepo(repo), int(m.group(1))) in vnorm


def load_templates(config_dir):
    T = rs.load_templates(config_dir)
    import yaml
    from jinja2 import StrictUndefined, Template
    src = yaml.safe_load(open(os.path.join(config_dir, 'benchmarks', 'swebench.yaml')))['agent']['instance_template']
    for a, b in SCALE_EDITS:
        if src.count(a) != 1:
            raise ValueError(f'swebench.yaml instance template lacks {a!r}')
        src = src.replace(a, b)
    T['scale_instance'] = Template(src, undefined=StrictUndefined)
    return T


def bash_args(tc):
    """A call's arguments json string if it is `bash` with exactly {"command": str}, else None."""
    f = tc.get('function') or {}
    try:
        args = json.loads(f.get('arguments'))
    except (TypeError, ValueError):
        return None
    if f.get('name') != 'bash' or not isinstance(args, dict) or set(args) != {'command'} \
            or not isinstance(args['command'], str):
        return None
    return f['arguments']


def split_think(content):
    """(reasoning or None, prose) of an Orchard assistant content, or None when the <think> markup is malformed."""
    c = content or ''
    if '<think>' not in c and '</think>' not in c:
        return None, c
    m = THINK_RE.match(c)
    if not m or '<think>' in m.group(1) or '</think>' in m.group(2) or '<think>' in m.group(2):
        return None
    return m.group(1), m.group(2)


def accepted_submit(p):
    """mini-swe-agent 2.4.6 _check_finished on an observation dict."""
    lines = (p['output'] or '').lstrip().splitlines(keepends=True)
    return bool(lines) and lines[0].strip() == SENTINEL and str(p['returncode']).strip() == '0'


def screen(r, md, T, st=None):
    """(drop reason or None, per-turn (reasoning, prose, [arguments json], [mini.yaml observation] or None), problem
    statement). `st` (a Counter) gets the observation / reasoning census."""
    st = st if st is not None else collections.Counter()
    pool = md.get('source')
    if pool not in POOLS:
        return ('openhands' if (pool or '').startswith('oh-') else 'bad_structure'), None, None
    if md.get('verify_status') != 'resolved':
        return 'unresolved', None, None
    if TEACHER_SET and md.get('model') not in TEACHER_SET:
        return 'teacher_excluded', None, None
    if verified(md.get('instance_id'), md.get('repo'), VIDS, VNORM):
        return 'swebench_verified', None, None
    try:
        tools = json.loads(r['tools'])
    except (TypeError, ValueError):
        tools = None
    if tools != [rm.BASH_TOOL]:
        return 'bad_tools', None, None
    ms = r['messages']
    if len(ms) < 3 or ms[0]['role'] != 'system' or ms[1]['role'] != 'user':
        return 'bad_structure', None, None
    mt = PR_RE.match(ms[1]['content'] or '')
    inst = T['scale_instance'] if POOLS[pool] == 'scale-swe' else T['src_instance']
    if not mt or T['src_system'].render() != ms[0]['content'] or inst.render(task=mt.group(1)) != ms[1]['content']:
        return 'prompt_unparsed', None, None
    turns, i = [], 2
    while i < len(ms):
        m = ms[i]
        if m['role'] == 'user':
            return 'format_error_turn', None, None
        if m['role'] != 'assistant':
            return 'bad_structure', None, None
        tcs = m.get('tool_calls') or []
        if isinstance(tcs, str):
            try:
                tcs = json.loads(tcs)
            except ValueError:
                return 'bad_call', None, None
        if not tcs:
            return 'no_call_turn', None, None
        args = [bash_args(tc) for tc in tcs]
        if any(a is None for a in args):
            return 'bad_call', None, None
        th = split_think(m.get('content'))
        if th is None:
            return 'bad_think', None, None
        obs, j = [], i + 1
        for tc in tcs:
            if j >= len(ms):
                break
            o = ms[j]
            if o['role'] == 'user':
                return 'format_error_turn', None, None
            if o['role'] != 'tool' or o.get('tool_call_id') != tc.get('id'):
                return 'bad_structure', None, None
            c = o['content'] or ''
            if c.startswith('Tool call error') or 'reached the output token limit' in c[:300]:
                return 'format_error_turn', None, None
            p = rs.parse_obs(c, T)
            if p is None:
                return 'obs_unparsed', None, None
            if (p['exception_info'] and not p['exception_info'].startswith(OWN_EXCEPTION)) or rs.harness_error(p['output']):
                return 'harness_error', None, None
            obs.append((p, T['mini_obs'].render(output=p)))
            j += 1
        last = j >= len(ms)
        if obs and len(obs) < len(tcs):                     # a turn's calls are answered all or none (the submit)
            return 'bad_structure', None, None
        if not obs and not last:
            return 'bad_structure', None, None
        if not last and any(accepted_submit(p) for p, _ in obs):
            return 'early_submit', None, None
        turns.append((th[0], th[1], args, [x for _, x in obs] or None))
        st['turns'] += 1
        st['turns_reasoning'] += bool((th[0] or '').strip())
        st['turns_prose'] += bool((th[1] or '').strip())
        st['turns_empty'] += not (th[0] or '').strip() and not (th[1] or '').strip()
        for p, _ in obs:
            st['obs'] += 1
            st['obs_long'] += len(p['output']) >= 10000
            st['obs_exception'] += bool(p['exception_info'])
        i = j
    if turns[-1][3] is not None or not any(SENTINEL in a for a in turns[-1][2]):
        return 'no_submit', None, None
    return None, turns, mt.group(1)


def record_census(r):
    """Per-record census, before any drop: parallel turns, exception kinds, anonymiser markers."""
    c = collections.Counter()
    ms = r['messages']
    c['has_parallel_turn'] = any(m['role'] == 'assistant' and len(m.get('tool_calls') or []) > 1 for m in ms)
    marker = False
    for m in ms:
        texts = [m['content'] or ''] + [tc['function']['arguments'] or '' for tc in (m.get('tool_calls') or [])]
        marker = marker or any(MARKERS.search(t) for t in texts)
        if m['role'] == 'tool':
            mt = re.match(r'<exception>(.*?)</exception>', m['content'] or '', re.S)
            if mt:
                kind = re.sub(r'\d+(?:\.\d+){3}(?::\d+)?', '<ip>', mt.group(1))
                kind = re.sub(r'sandbox \w+', 'sandbox <id>', kind)[:60]
                c['exception: ' + kind] += 1
    c['scrub_marker'] = marker
    return c


def init_worker(tokdir, mini_yaml, config_dir, prompt, think_limit, vids, vkeys, teachers):
    global T, VIDS, VNORM, TEACHER_SET
    import pyarrow as pa
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    rst.init_worker(tokdir, mini_yaml, prompt, think_limit)
    T, VIDS, TEACHER_SET = load_templates(config_dir), vids, set(teachers or ())
    VNORM = {(nrepo(r), n) for r, n in vkeys}


def iter_rows(path, rg):
    import pyarrow.parquet as pq
    k = 0
    for b in pq.ParquetFile(path).iter_batches(batch_size=32, row_groups=[rg], columns=COLUMNS):
        for r in b.to_pylist():
            yield k, r
            k += 1


def screen_unit(u):
    """Pass 1: one row group -> [(instance_id, teacher, pool, sample_idx, unit, row index, repo, drop reason, census)]."""
    path, rg = u
    out = []
    for k, r in iter_rows(path, rg):
        md = json.loads(r['metadata'])
        census = record_census(r)
        why, _, _ = screen(r, md, T)
        out.append((md['instance_id'], md.get('model'), md.get('source'), md.get('sample_idx'), u, k, md.get('repo'),
                    why, dict(census)))
    return out


def render_unit(job):
    """Pass 2: one row group, the given row indices -> [(drop reason or None, aligned, row meta, row json or None)]."""
    import pyarrow.parquet as pq
    (path, rg), ks = job
    return [render_one(r) for r in pq.ParquetFile(path).read_row_group(rg, columns=COLUMNS).take(ks).to_pylist()]


def render_one(r):
    md = json.loads(r['metadata'])
    census = collections.Counter()
    why, turns, ps = screen(r, md, T, census)
    pool, iid = md['source'], md['instance_id']
    sid = f"orchard:{pool}:{iid}:{md.get('sample_idx')}"
    meta = dict(arm=ARM, task=iid, sid=sid, teacher=md.get('model'), source=POOLS.get(pool), pool=pool,
                sample_idx=md.get('sample_idx'), obs=dict(census))
    if why:                                                  # pass 1 passed it; cannot happen on the same bytes
        return why, True, meta, None
    if rst.PROMPT == 'mini':
        system, user = rst.mini_prompt(ps.replace('\r\n', '\n').replace('\r', '\n'), sid)
    else:
        system, user = r['messages'][0]['content'], r['messages'][1]['content']
    msgs = [dict(role='system', content=system), dict(role='user', content=user)]
    recs = []
    for k, (reasoning, prose, arguments, obs) in enumerate(turns):
        cids = [f'call_{k}_{j}' for j in range(len(arguments))]
        m = dict(role='assistant', content=(prose or '').strip(),
                 tool_calls=[dict(id=c, type='function', function=dict(name='bash', arguments=a))
                             for c, a in zip(cids, arguments)])
        if (reasoning or '').strip():
            m['reasoning_content'] = reasoning
        msgs.append(m)
        for c, o in zip(cids, obs or []):
            msgs.append(dict(role='tool', content=o, tool_call_id=c))
        recs.append(dict(response=dict(choices=[dict(message=dict(tool_calls=[dict(id=cids[0])]))]), upstream_status=200,
                         owner='teacher', autofix=False, repair=False, turn=k, seq=k))
    canary = any(CANARY.search(m.get('content') or '') or CANARY.search(m.get('reasoning_content') or '') or
                 any(CANARY.search(tc['function']['arguments']) for tc in m.get('tool_calls') or []) for m in msgs)
    row = rm.render(msgs, recs, rst.TOK, rst.TPL, rst.BOS, think_limit=rst.THINK)
    aligned = rst.check_alignment(row)
    trained = sum(row['loss'])
    reason = 'canary' if canary else 'over_64k' if not row['fits'] else 'no_trained_token' if not trained else None
    t = row['turns']
    final = turns[-1][2]
    meta.update(trial=f"{pool}:{iid}:{md.get('sample_idx')}", reward=1.0, passed=True, cause=None,
                teacher_turns=len(t), student_turns=0, repairs=0, takeover=None, teacher_cut_turns=0,
                run=f'{REPO}/swe/{pool}', n_tokens=row['n_tokens'], fits=row['fits'], trained_tokens=trained,
                teacher_turns_rendered=len(t), cut_turns=sum(1 for x in t if x.get('cut_at') is not None),
                long_think_masked=sum(1 for x in t if x.get('cut_at') is None and x.get('reasoning_tokens', 0) > rst.THINK),
                think_trained_turns=sum(1 for x in t if x['think_trained']), autofix_student_turns=0,
                view_count_logged=None, view_count_rendered=None, verifier_ran=True, weak_timeout=False, noisy=False,
                leak=False, hunt=False, canary=canary, dataset=REPO, subset=pool, repo=md.get('repo'), language='python',
                hf_dataset_name=REPO, prompt=rst.PROMPT, parallel_turns=sum(len(x[2]) > 1 for x in turns),
                reasoning_turns=sum(bool((x[0] or '').strip()) for x in turns),
                empty_content_turns=sum(not (x[1] or '').strip() for x in turns),
                final_submit=SENTINEL in json.loads(final[0])['command'] if len(final) == 1 else False,
                final_submit_form=rs.submit_form(final))
    line = None
    if not reason:
        line = json.dumps(dict({k: v for k, v in meta.items() if k != 'obs'}, ids=row['ids'], loss=row['loss']))
    return reason, aligned, meta, line


def shard_units(raw_dir, shards):
    import pyarrow.parquet as pq
    units = []
    for s in shards:
        rel = f'swe/train/data-{s:05d}-of-{N_SHARDS:05d}.parquet'
        p = os.path.join(raw_dir, rel)
        if not os.path.exists(p):
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(REPO, rel, repo_type='dataset', local_dir=raw_dir)
        units += [(os.path.abspath(p), g) for g in range(pq.ParquetFile(p).metadata.num_row_groups)]
    return units


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True, help='rows.jsonl path; qa.json is written beside it')
    ap.add_argument('--raw-dir', default=RAW, help='Orchard download (swe/train/*.parquet; fetched when missing)')
    ap.add_argument('--shards', nargs='+', type=int, default=list(SHARDS), help='swe/train shard numbers to read')
    ap.add_argument('--teachers', nargs='+', default=list(TEACHERS), help='keep only these metadata.model teachers')
    ap.add_argument('--prefer', nargs='+', default=list(TEACHERS), help='teacher order when a task has several')
    ap.add_argument('--tokenizer-dir', default=rst.TOKDIR)
    ap.add_argument('--mini-yaml', default=os.path.join(rs.MSA_CONFIG, 'mini.yaml'), help='mini-swe-agent 2.4.6 mini.yaml')
    ap.add_argument('--msa-config', default=rs.MSA_CONFIG, help='mini-swe-agent 2.4.6 config dir (benchmarks/swebench'
                                                                '.yaml is the source prompt and observation template)')
    ap.add_argument('--prompt', choices=('mini', 'source'), default='mini')
    ap.add_argument('--verified-ids', help='SWE-bench Verified instance ids json (default <out dir>/swebench_verified_ids'
                                           '.json, fetched from HF when missing)')
    ap.add_argument('--think-limit', type=int, default=rm.THINK_LIMIT)
    ap.add_argument('--max-tasks', type=int, help='only the first N tasks in sha1(instance_id) order (a test subset)')
    ap.add_argument('--procs', type=int, default=4)
    a = ap.parse_args()
    out_dir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(out_dir, exist_ok=True)
    vids, vkeys = rst.verified_sets(a.verified_ids or os.path.join(out_dir, 'swebench_verified_ids.json'))
    vrepos = {nrepo(k[0]) for k in vkeys}
    units = shard_units(a.raw_dir, a.shards)
    init = (a.tokenizer_dir, a.mini_yaml, a.msa_config, a.prompt, a.think_limit, vids, vkeys, a.teachers)
    trank = {t: j for j, t in enumerate(a.prefer)}

    overall, dropped = collections.Counter(), collections.Counter()
    per_pool = collections.defaultdict(collections.Counter)
    per_teacher = collections.defaultdict(collections.Counter)
    obs = collections.Counter()
    cands, kept, misaligned, records = collections.defaultdict(list), [], 0, []
    task_teachers = collections.defaultdict(set)
    with mp.Pool(a.procs, init_worker, init) as pool:
        for res in pool.imap(screen_unit, units):                       # pass 1: screen every record
            records.extend(res)
        if a.max_tasks:
            ids = {x[0] for x in records if x[2] in POOLS and x[7] != 'unresolved'}
            keep = set(sorted(ids, key=lambda i: hashlib.sha1(i.encode()).hexdigest())[:a.max_tasks])
            records = [x for x in records if x[0] in keep]
        for iid, teacher, pl, sidx, u, k, repo, why, census in records:
            overall['records'] += 1
            per_pool[pl]['records'] += 1
            per_teacher[teacher]['records'] += 1
            for kk, v in census.items():
                overall[kk] += v
            if pl in POOLS and why != 'unresolved':
                overall['resolved_msa'] += 1
                per_pool[pl]['resolved'] += 1
                per_teacher[teacher]['resolved'] += 1
                overall['verified_repo_any'] += nrepo(repo) in vrepos
            if why:
                dropped[why] += 1
                per_pool[pl][why] += 1
                per_teacher[teacher][why] += 1
                continue
            per_pool[pl]['candidates'] += 1
            per_teacher[teacher]['candidates'] += 1
            task_teachers[iid].add(teacher)
            cands[iid].append((trank.get(teacher, len(trank)), hashlib.sha1(f'{iid}:{pl}:{sidx}'.encode()).hexdigest(),
                               teacher, pl, u, k))
        for c in cands.values():
            c.sort()
        done, rendered_upto, tmp, rank = set(), {}, a.out + '.tmp', 0
        with open(tmp, 'wb') as f:
            while True:                                                 # pass 2: render in preference order per task
                todo = collections.defaultdict(list)
                for iid in sorted(cands):
                    if iid not in done and rank < len(cands[iid]):
                        u, k = cands[iid][rank][4:]
                        todo[u].append(k)
                        rendered_upto[iid] = rank + 1
                if not todo:
                    break
                for res in pool.imap(render_unit, sorted(todo.items())):
                    for why, aligned, meta, line in res:
                        misaligned += not aligned
                        pl, tch = meta['pool'], meta['teacher']
                        if why:
                            dropped[why] += 1
                            per_pool[pl][why] += 1
                            per_teacher[tch][why] += 1
                            continue
                        done.add(meta['task'])
                        for kk, v in meta['obs'].items():
                            obs[kk] += v
                        per_pool[pl]['kept'] += 1
                        per_teacher[tch]['kept'] += 1
                        per_pool[pl][f'kept_rank{rank}'] += 1
                        kept.append(dict({k: v for k, v in meta.items() if k != 'obs'}, offset=f.tell()))
                        f.write(line.encode() + b'\n')
                rank += 1
    kept.sort(key=lambda r: hashlib.sha1(r['task'].encode()).hexdigest())    # rows in a pool-mixed, fixed order
    with open(tmp, 'rb') as src, open(tmp + '2', 'wb') as f:
        for r in kept:
            src.seek(r.pop('offset'))
            f.write(src.readline())
    os.replace(tmp + '2', a.out)
    os.remove(tmp)
    for iid, c in cands.items():         # a candidate never rendered: its task was kept from an earlier-ranked one
        for j, (_, _, tch, pl, _, _) in enumerate(c):
            if j >= rendered_upto.get(iid, 0):
                dropped['task_kept_other_candidate'] += 1
                per_pool[pl]['task_kept_other_candidate'] += 1
                per_teacher[tch]['task_kept_other_candidate'] += 1
    by_task = {r['task']: r for r in kept}
    selection = collections.Counter()
    for iid, ts in task_teachers.items():
        r = by_task.get(iid)
        key = '+'.join(sorted(ts))
        selection[f'tasks_with_candidates[{key}]'] += 1
        selection[f'kept[{key}] -> {r["teacher"] if r else "none"}'] += 1
    turns = [r['teacher_turns'] for r in kept]
    by = lambda key: {v: rs.tok_stats([r for r in kept if r[key] == v]) for v in sorted({r[key] for r in kept})}  # noqa
    qa = dict(source=REPO, config='swe', shards=a.shards, teachers=a.teachers, prefer=a.prefer, prompt=a.prompt,
              think_limit=a.think_limit, cap_tokens=rm.rcap.CAP_TOKENS, max_tokens=rm.MAX_TOKENS,
              selection_rule='per task: candidates by --prefer teacher, then sha1(instance_id:pool:sample_idx); the first '
              'that passes', swebench_verified_ids=len(vids), records=overall['records'],
              tasks_with_candidate=len(cands), kept=len(kept),
              dropped_first_reason={k: dropped[k] for k in REASONS if dropped[k]},
              all_records=dict(sorted(overall.items())), misaligned_rows=misaligned,
              observations_kept_rows=dict(obs), selection=dict(sorted(selection.items())),
              per_pool={s: dict(c) for s, c in per_pool.items()},
              per_teacher={s: dict(c) for s, c in per_teacher.items()},
              kept_stats=dict(rs.tok_stats(kept), tasks=len({r['task'] for r in kept}),
                              turns_p50=rst.pct(turns, .5), turns_max=max(turns, default=None), turns_total=sum(turns),
                              cut_turns=sum(r['cut_turns'] for r in kept),
                              think_trained_turns=sum(r['think_trained_turns'] for r in kept),
                              reasoning_turns=sum(r['reasoning_turns'] for r in kept),
                              long_think_masked=sum(r['long_think_masked'] for r in kept),
                              parallel_turns=sum(r['parallel_turns'] for r in kept),
                              empty_content_turns=sum(r['empty_content_turns'] for r in kept),
                              final_submit=sum(r['final_submit'] for r in kept),
                              final_submit_form=dict(collections.Counter(r['final_submit_form'] for r in kept)),
                              verified_repo_rows=sum(nrepo(r['repo']) in vrepos for r in kept),
                              by_teacher=by('teacher'), by_source=by('source'), by_pool=by('pool')))
    qa_path = os.path.join(out_dir, os.path.splitext(os.path.basename(a.out))[0].replace('rows', 'qa') + '.json')
    json.dump(qa, open(qa_path, 'w'), indent=1)
    print(json.dumps(qa, indent=1))


T = VIDS = VNORM = None
TEACHER_SET = set()

if __name__ == '__main__':
    main()
