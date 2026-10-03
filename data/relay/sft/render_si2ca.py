#!/usr/bin/env python3
"""Self-Improving-Coding-Agents/SI2CA-Training-Trajectories (mini-swe-agent v2 tool-mode trajectories) -> relay SFT rows
in render_swe_traces.py's schema and rendering, for mixing SWE replay into an MSA relay arm.

Source (CC-BY-4.0): Qwen3.5-122B-A10B, reasoning kept, one `bash` tool, on 10,780 Python tasks (6,146 SWE-rebench-V2
PRs, 4,634 SWE-smith bugs), each run once under three curation settings: `standard` (one sample per turn),
`full_self_judgement` (two candidate turns per step, a gold-patch-aware judge picks the executed one) and
`discovered_strategy` (the same, only inside the early-commitment window). The judge never enters the policy's
context: all three have the same system + user (mini-swe-agent's benchmarks/swebench.yaml, byte-identical across the
settings for every task), the same tool and the same observation template, so the settings are equivalent in form.

Kept: resolved; no SWE-bench Verified instance (by instance id, by (repo, PR number), and SWE-smith `.pr_<N>` PR mirrors
by (repo, N)); the swebench.yaml prompt (re-rendered byte-identical from the extracted task); every assistant turn has
at least one `bash` call with {"command": str} (parallel calls are kept: render_msa.render and the 09-21 template take
several calls in one turn, each answered by its own tool message, as mini-swe-agent 2.4.6 does); no upstream
format-error message (user-role, or tool-role 'Tool call error' / output-token-limit notices); every observation parses
as the source template's render (round trip checked byte for byte); no harness artefact observation (podman
`Error: unlinkat ...: directory not empty` appended with returncode 255, or `local docker command timed out after
60s` with returncode 124: our harness sends neither); the last turn is the submit (its call unanswered, the command
carries COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT); no benchmark canary; <= 65,536 tokens; >= 1 trained token.
One trajectory per task: among a task's settings that pass every check above, the one with the smallest
sha1('<instance_id>:<setting>') (a task's candidates are rendered in that order until one passes).

Observations are re-rendered into mini.yaml's own observation template, which is what the harness sends: SI2CA used
swebench.yaml's text template (<returncode>N</returncode><output>..</output>, or a <warning> with output_head /
output_tail past 10,000 chars). Both templates cut at the same 10,000 chars into the first and last 5,000, so the
re-render is exact for long outputs too (head, tail and the elided count are all mini.yaml needs). Rendering is jinja2
with the template text from the msa-2.4.6 configs, as mini-swe-agent renders it (checked 2026-10-03: 29,460 tool
messages of our mini-swe-agent-host SWE evals, 385 of them long, rendered through the source template and back come out
byte-identical to what the harness sent and to mini-swe-agent's format_toolcall_observation_messages).
SI2CA's assistant turns carry no visible content (Qwen3.5 writes everything in its reasoning), so rendered turns are
<|start_think|>..<|end_think|> + the call, with no prose between.

Rendering is render_msa.render itself, through render_swe_traces's helpers (09-21's chat_template.jinja, the bash tool
in the Tools block, calls as <tool_call>{"name": "bash", "arguments": {..}}</tool_call>, observations as
<tool_response name="bash">). Every assistant turn is a relay TEACHER turn: reasoning as <|start_think|>..<|end_think|>,
every turn but the last cut to reasoning_cap.CAP_TOKENS at a sentence / line boundary, the last kept whole; loss on
content + calls + <|eot_id|> of every turn and on its reasoning only when uncut and <= --think-limit tokens. System,
user and tool turns are masked.

Prompt (--prompt):
  mini    (default) render_swe_traces.mini_prompt: system + task as harbor mini-swe-agent-host sends them under
          mini-swe-agent 2.4.6 mini.yaml for a harbor SWE-bench task (the PR description, dedented and stripped, plus a
          newline; a Daytona uname picked by a hash of the trajectory id). SI2CA's task text already has CRLF folded to
          LF, as harbor's instruction.md read does.
  source  SI2CA's own system + user verbatim (swebench.yaml: /testbed, the git-patch submission steps).
The trajectories' own actions stay as they are, including the submit (`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT &&
cat patch.txt` in most rows), which mini-swe-agent 2.4.6 accepts: the first output line is the sentinel, returncode 0.

    python render_si2ca.py --out <dir>/rows.jsonl [--raw-dir <si2ca download>] [--settings ...] [--max-tasks N]
writes rows.jsonl (one row per kept task: arm 'si2ca', task = instance id, source rebench / swesmith, setting, sid,
the QA tags, ids, loss) and qa.json beside it (records kept / dropped by first reason, per setting and per source,
observation re-format counts, token stats).
"""
import argparse
import collections
import glob
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
import render_swe_traces as rst  # noqa: E402
from hz_pool_census import CANARY  # noqa: E402

REPO = 'Self-Improving-Coding-Agents/SI2CA-Training-Trajectories'
SETTINGS = ('standard', 'full_self_judgement', 'discovered_strategy')
ARM = 'si2ca'
RAW = '/scratch/11584/lukedhlee/experiments/sft_data/si2ca_raw'
MSA_CONFIG = os.path.expanduser('~/snowball/envs/msa-2.4.6/minisweagent/config')
COLUMNS = ['instance_id', 'curation_strategy', 'source_dataset', 'repo', 'messages', 'resolved', 'reward',
           'n_harness_error_msgs']
SOURCES = {'SWE-rebench-V2': 'rebench', 'SWE-smith': 'swesmith'}
SENTINEL = 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'
PR_RE = re.compile(r'^<pr_description>\nConsider the following PR description:\n(.*?)\n</pr_description>\n', re.S)
OBS_HEAD = re.compile(r'\A(?:<exception>(.*?)</exception>\n)?<returncode>(.*?)</returncode>\n', re.S)
ELIDED = re.compile(r'(\d+) characters elided\n</elided_chars>\n<output_tail>\n')
PODMAN = re.compile(r'Error: unlinkat [^\n]*: directory not empty')
TIMEOUT_TEXT = 'local docker command timed out after 60s'


def harness_error(text):
    """A source observation carrying a generation-harness artefact (the dataset's n_harness_error_msgs notices)."""
    return bool(PODMAN.search(text)) or TIMEOUT_TEXT in text
REASONS = ('unresolved', 'swebench_verified', 'bad_structure', 'prompt_unparsed', 'format_error_turn', 'no_call_turn',
           'bad_call', 'obs_unparsed', 'harness_error', 'no_submit', 'canary', 'over_64k', 'no_trained_token',
           'task_kept_other_setting')


def load_templates(config_dir):
    """(swebench.yaml system, instance and observation templates; mini.yaml observation template), jinja2 as
    mini-swe-agent uses it (StrictUndefined, default environment)."""
    import yaml
    from jinja2 import StrictUndefined, Template
    src = yaml.safe_load(open(os.path.join(config_dir, 'benchmarks', 'swebench.yaml')))
    mini = yaml.safe_load(open(os.path.join(config_dir, 'mini.yaml')))
    t = {k: Template(v, undefined=StrictUndefined) for k, v in dict(
        src_system=src['agent']['system_template'], src_instance=src['agent']['instance_template'],
        src_obs=src['model']['observation_template'], mini_obs=mini['model']['observation_template']).items()}
    long = t['src_obs'].render(output=dict(output='x' * 10000, returncode=0, exception_info=''))
    t['warning'] = long[long.index('<warning>'):long.index('<output_head>')] + '<output_head>\n'
    return t


def parse_obs(c, T):
    """The source observation's {output, returncode, exception_info} (for a long output: head + NULs + tail, which is
    all either template reads), or None unless re-rendering it with the source template gives `c` byte for byte."""
    m = OBS_HEAD.match(c)
    if not m:
        return None
    exc, rc, rest = m.group(1) or '', m.group(2), c[m.end():]
    if rest.startswith('<output>\n') and rest.endswith('</output>'):
        out = rest[len('<output>\n'):-len('</output>')]
    elif rest.startswith(T['warning']):
        r = rest[len(T['warning']):]
        head, r = r[:5000], r[5000:]
        if not r.startswith('\n</output_head>\n<elided_chars>\n'):
            return None
        mm = ELIDED.match(r, len('\n</output_head>\n<elided_chars>\n'))
        if not mm:
            return None
        tail, end = r[mm.end():mm.end() + 5000], r[mm.end() + 5000:]
        if end != '\n</output_tail>':
            return None
        out = head + '\0' * int(mm.group(1)) + tail
    else:
        return None
    o = dict(output=out, returncode=rc, exception_info=exc)
    return o if T['src_obs'].render(output=o) == c else None


def verified(r, vids, vkeys):
    iid, repo = r['instance_id'], r['repo']
    if iid in vids or rst.pr_key(iid, repo) in vkeys:
        return True
    m = re.search(r'\.pr_(\d+)$', iid or '')                # SWE-smith PR mirror of (repo, N)
    return bool(m) and ((repo or '').lower(), int(m.group(1))) in vkeys


def screen(r, vids, vkeys, T, stats=None):
    """(drop reason or None, per-turn (reasoning, content, [command arguments json], [mini.yaml observation]),
    problem statement). `stats` (a Counter) gets the observation census."""
    st = stats if stats is not None else collections.Counter()
    if not r['resolved']:
        return 'unresolved', None, None
    if verified(r, vids, vkeys):
        return 'swebench_verified', None, None
    ms = r['messages']
    if len(ms) < 3 or ms[0]['role'] != 'system' or ms[1]['role'] != 'user':
        return 'bad_structure', None, None
    mt = PR_RE.match(ms[1]['content'] or '')
    if not mt or T['src_system'].render() != ms[0]['content'] \
            or T['src_instance'].render(task=mt.group(1)) != ms[1]['content']:
        return 'prompt_unparsed', None, None
    turns, i = [], 2
    while i < len(ms):
        m = ms[i]
        if m['role'] == 'user':
            return 'format_error_turn', None, None
        if m['role'] != 'assistant':
            return 'bad_structure', None, None
        try:
            tcs = json.loads(m['tool_calls']) if m['tool_calls'] else []
        except ValueError:
            return 'bad_call', None, None
        if not tcs:
            return 'no_call_turn', None, None
        args = []
        for tc in tcs:
            f = tc.get('function') or {}
            a = f.get('arguments')
            if isinstance(a, str):
                try:
                    a = json.loads(a)
                except ValueError:
                    a = None
            if f.get('name') != 'bash' or not isinstance(a, dict) or set(a) != {'command'} \
                    or not isinstance(a['command'], str):
                return 'bad_call', None, None
            args.append(a)
        obs, j = [], i + 1
        for tc in tcs:
            if j >= len(ms):
                break
            o = ms[j]
            if o['role'] == 'user':
                return 'format_error_turn', None, None
            if o['role'] != 'tool' or o['tool_call_id'] != tc.get('id'):
                return 'bad_structure', None, None
            c = o['content'] or ''
            if c.startswith('Tool call error') or 'reached the output token limit' in c[:300]:
                return 'format_error_turn', None, None
            p = parse_obs(c, T)
            if p is None:
                return 'obs_unparsed', None, None
            if harness_error(p['output']):
                return 'harness_error', None, None
            obs.append(T['mini_obs'].render(output=p))
            st['obs'] += 1
            st['obs_long'] += len(p['output']) >= 10000
            st['obs_exception'] += bool(p['exception_info'])
            j += 1
        last = j >= len(ms)
        if obs and len(obs) < len(tcs):                     # a turn's calls are answered all or none (the submit)
            return 'bad_structure', None, None
        if not obs and not last:
            return 'bad_structure', None, None
        turns.append((m.get('reasoning_content'), m.get('content'),
                      [json.dumps(a, ensure_ascii=False) for a in args], obs or None))
        i = j
    if turns[-1][3] is not None or not any(SENTINEL in a for a in turns[-1][2]):
        return 'no_submit', None, None
    return None, turns, mt.group(1)


def init_worker(tokdir, mini_yaml, config_dir, prompt, think_limit, vids, vkeys):
    global T, VIDS, VKEYS
    rst.init_worker(tokdir, mini_yaml, prompt, think_limit)
    T, VIDS, VKEYS = load_templates(config_dir), vids, vkeys


def screen_unit(u):
    """Pass 1: one row group -> [(instance_id, setting, unit, row index, source, repo, drop reason or None, census)]."""
    import pyarrow.parquet as pq
    setting, path, rg = u
    out = []
    for k, r in enumerate(pq.ParquetFile(path).read_row_group(rg, columns=COLUMNS).to_pylist()):
        census = collections.Counter()
        ms = r['messages']
        census['has_parallel_turn'] = any(m['role'] == 'assistant' and m['tool_calls'] and
                                          len(json.loads(m['tool_calls'])) > 1 for m in ms)
        census['harness_error_col'] = bool(r['n_harness_error_msgs'])
        census['harness_msgs_col_mismatch'] = r['n_harness_error_msgs'] != sum(
            m['role'] == 'tool' and harness_error(m['content'] or '') for m in ms)
        why, _, _ = screen(r, VIDS, VKEYS, T)
        out.append((r['instance_id'], setting, u, k, SOURCES.get(r['source_dataset'], r['source_dataset']), r['repo'],
                    why, dict(census)))
    return out


def render_unit(job):
    """Pass 2: one row group, the given row indices -> [(drop reason or None, aligned, row meta, row json or None)]."""
    import pyarrow.parquet as pq
    (setting, path, rg), ks = job
    rows = pq.ParquetFile(path).read_row_group(rg, columns=COLUMNS).to_pylist()
    return [render_one(rows[k], setting) for k in ks]


def render_one(r, setting):
    census = collections.Counter()
    why, turns, ps = screen(r, VIDS, VKEYS, T, census)
    source = SOURCES.get(r['source_dataset'], r['source_dataset'])
    sid = f"si2ca:{setting}:{r['instance_id']}"
    meta = dict(arm=ARM, task=r['instance_id'], sid=sid, setting=setting, source=source, obs=dict(census))
    if why:                                                  # pass 1 passed it; cannot happen on the same bytes
        return why, True, meta, None
    if rst.PROMPT == 'mini':
        system, user = rst.mini_prompt(ps, sid)
    else:
        system, user = r['messages'][0]['content'], r['messages'][1]['content']
    msgs = [dict(role='system', content=system), dict(role='user', content=user)]
    recs = []
    for k, (reasoning, content, arguments, obs) in enumerate(turns):
        cids = [f'call_{k}_{j}' for j in range(len(arguments))]
        m = dict(role='assistant', content=content or '',
                 tool_calls=[dict(id=c, type='function', function=dict(name='bash', arguments=a))
                             for c, a in zip(cids, arguments)])
        if reasoning:
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
    meta.update(trial=f"{setting}:{r['instance_id']}", reward=r['reward'], passed=True, cause=None,
                teacher_turns=len(t), student_turns=0, repairs=0, takeover=None, teacher_cut_turns=0,
                run=f'{REPO}/{setting}', n_tokens=row['n_tokens'], fits=row['fits'], trained_tokens=trained,
                teacher_turns_rendered=len(t), cut_turns=sum(1 for x in t if x.get('cut_at') is not None),
                long_think_masked=sum(1 for x in t if x.get('cut_at') is None and x.get('reasoning_tokens', 0) > rst.THINK),
                think_trained_turns=sum(1 for x in t if x['think_trained']), autofix_student_turns=0,
                view_count_logged=None, view_count_rendered=None, verifier_ran=True, weak_timeout=False, noisy=False,
                leak=False, hunt=False, canary=canary, dataset=REPO, subset=setting, repo=r['repo'], language='python',
                hf_dataset_name=r['source_dataset'], prompt=rst.PROMPT, parallel_turns=sum(len(x[2]) > 1 for x in turns),
                empty_content_turns=sum(not (x[1] or '').strip() for x in turns),
                final_submit=SENTINEL in final[0] if len(final) == 1 else False,
                final_submit_form=submit_form(final))
    line = None
    if not reason:
        line = json.dumps(dict({k: v for k, v in meta.items() if k != 'obs'}, ids=row['ids'], loss=row['loss']))
    return reason, aligned, meta, line


def submit_form(args):
    """How the final call submits: 'echo' (`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`, alone), 'echo_and' (echo
    first, `&& ...` after it, e.g. `&& cat patch.txt`), 'prefixed' (other commands before the echo), 'parallel'."""
    if len(args) != 1:
        return 'parallel'
    c = json.loads(args[0])['command'].strip()
    if re.fullmatch(r"echo\s+'?COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'?", c):
        return 'echo'
    if re.match(r"echo\s+'?COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'?\s*&&", c):
        return 'echo_and'
    return 'prefixed'


def shard_units(raw_dir, settings):
    import pyarrow.parquet as pq
    units = []
    for s in settings:
        paths = sorted(glob.glob(os.path.join(raw_dir, 'data', s, '*.parquet')))
        if not paths:
            from huggingface_hub import snapshot_download
            snapshot_download(REPO, repo_type='dataset', local_dir=raw_dir, allow_patterns=[f'data/{s}/*', 'README.md'])
            paths = sorted(glob.glob(os.path.join(raw_dir, 'data', s, '*.parquet')))
        for p in paths:
            units += [(s, os.path.abspath(p), g) for g in range(pq.ParquetFile(p).metadata.num_row_groups)]
    return units


pct = rst.pct


def tok_stats(rows):
    n = [r['n_tokens'] for r in rows]
    tt = [r['trained_tokens'] for r in rows]
    return dict(rows=len(rows), tokens=sum(n), trained_tokens=sum(tt), n_tokens_p10=pct(n, .1), n_tokens_p50=pct(n, .5),
                n_tokens_p90=pct(n, .9), n_tokens_max=max(n, default=None), trained_p50=pct(tt, .5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True, help='rows.jsonl path; qa.json is written beside it')
    ap.add_argument('--raw-dir', default=RAW, help='SI2CA download (data/<setting>/*.parquet; fetched when missing)')
    ap.add_argument('--settings', nargs='+', default=list(SETTINGS), choices=SETTINGS)
    ap.add_argument('--tokenizer-dir', default=rst.TOKDIR)
    ap.add_argument('--mini-yaml', default=os.path.join(MSA_CONFIG, 'mini.yaml'), help='mini-swe-agent 2.4.6 mini.yaml')
    ap.add_argument('--msa-config', default=MSA_CONFIG, help='mini-swe-agent 2.4.6 config dir (benchmarks/swebench.yaml '
                                                             'is the source prompt and observation template)')
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
    vrepos = {k[0] for k in vkeys}
    units = shard_units(a.raw_dir, a.settings)
    init = (a.tokenizer_dir, a.mini_yaml, a.msa_config, a.prompt, a.think_limit, vids, vkeys)

    overall, dropped = collections.Counter(), collections.Counter()
    per_setting = collections.defaultdict(collections.Counter)
    per_source = collections.defaultdict(collections.Counter)
    obs = collections.Counter()
    cands, kept, misaligned, tasks_all, records = collections.defaultdict(list), [], 0, {}, []
    with mp.Pool(a.procs, init_worker, init) as pool:
        for res in pool.imap(screen_unit, units):                       # pass 1: screen every record
            records.extend(res)
        if a.max_tasks:
            keep = set(sorted({x[0] for x in records}, key=lambda i: hashlib.sha1(i.encode()).hexdigest())[:a.max_tasks])
            records = [x for x in records if x[0] in keep]
        for iid, setting, u, k, source, repo, why, census in records:
            tasks_all[iid] = source
            overall['records'] += 1
            per_setting[setting]['records'] += 1
            per_source[source]['records'] += 1
            for kk, v in census.items():
                overall[kk] += v
            if why != 'unresolved':
                overall['resolved'] += 1
                per_setting[setting]['resolved'] += 1
                per_source[source]['resolved'] += 1
            overall['verified_repo_any'] += (repo or '').lower() in vrepos
            if why:
                dropped[why] += 1
                per_setting[setting][why] += 1
                per_source[source][why] += 1
                continue
            per_setting[setting]['candidates'] += 1
            per_source[source]['candidates'] += 1
            cands[iid].append((hashlib.sha1(f'{iid}:{setting}'.encode()).hexdigest(), setting, u, k))
        for c in cands.values():
            c.sort()
        done, rendered_upto, tmp, rank = set(), {}, a.out + '.tmp', 0
        with open(tmp, 'wb') as f:
            while True:                                                 # pass 2: render in hash order per task
                todo = collections.defaultdict(list)
                for iid in sorted(cands):
                    if iid not in done and rank < len(cands[iid]):
                        _, _, u, k = cands[iid][rank]
                        todo[u].append(k)
                        rendered_upto[iid] = rank + 1
                if not todo:
                    break
                for res in pool.imap(render_unit, sorted(todo.items())):
                    for why, aligned, meta, line in res:
                        misaligned += not aligned
                        s, src = meta['setting'], meta['source']
                        if why:
                            dropped[why] += 1
                            per_setting[s][why] += 1
                            per_source[src][why] += 1
                            continue
                        done.add(meta['task'])
                        for kk, v in meta['obs'].items():
                            obs[kk] += v
                        per_setting[s]['kept'] += 1
                        per_source[src]['kept'] += 1
                        per_setting[s][f'kept_rank{rank}'] += 1
                        kept.append(dict({k: v for k, v in meta.items() if k != 'obs'}, offset=f.tell()))
                        f.write(line.encode() + b'\n')
                rank += 1
    kept.sort(key=lambda r: hashlib.sha1(r['task'].encode()).hexdigest())    # rows in a setting-mixed, fixed order
    with open(tmp, 'rb') as src, open(tmp + '2', 'wb') as f:
        for r in kept:
            src.seek(r.pop('offset'))
            f.write(src.readline())
    os.replace(tmp + '2', a.out)
    os.remove(tmp)
    # a candidate is kept, dropped at render, or never rendered: its task was kept from an earlier-ranked setting
    for iid, c in cands.items():
        for j, (_, s, _, _) in enumerate(c):
            if j >= rendered_upto.get(iid, 0):
                dropped['task_kept_other_setting'] += 1
                per_setting[s]['task_kept_other_setting'] += 1
                per_source[tasks_all[iid]]['task_kept_other_setting'] += 1
    turns = [r['teacher_turns'] for r in kept]
    by = lambda key: {v: tok_stats([r for r in kept if r[key] == v]) for v in sorted({r[key] for r in kept})}  # noqa
    qa = dict(source=REPO, settings=a.settings, prompt=a.prompt, think_limit=a.think_limit,
              cap_tokens=rm.rcap.CAP_TOKENS, max_tokens=rm.MAX_TOKENS, selection='per task: the passing setting with '
              'the smallest sha1(instance_id:setting)', swebench_verified_ids=len(vids), records=overall['records'],
              tasks=len(tasks_all), tasks_by_source=dict(collections.Counter(tasks_all.values())),
              tasks_with_candidate=len(cands), kept=len(kept),
              dropped_first_reason={k: dropped[k] for k in REASONS if dropped[k]},
              all_records=dict(overall), misaligned_rows=misaligned,
              observations_kept_rows=dict(obs),
              per_setting={s: dict(c) for s, c in per_setting.items()},
              per_source={s: dict(c) for s, c in per_source.items()},
              kept_stats=dict(tok_stats(kept), tasks=len({r['task'] for r in kept}),
                              turns_p50=pct(turns, .5), turns_max=max(turns, default=None), turns_total=sum(turns),
                              cut_turns=sum(r['cut_turns'] for r in kept),
                              think_trained_turns=sum(r['think_trained_turns'] for r in kept),
                              long_think_masked=sum(r['long_think_masked'] for r in kept),
                              parallel_turns=sum(r['parallel_turns'] for r in kept),
                              empty_content_turns=sum(r['empty_content_turns'] for r in kept),
                              final_submit=sum(r['final_submit'] for r in kept),
                              final_submit_form=dict(collections.Counter(r['final_submit_form'] for r in kept)),
                              verified_repo_rows=sum((r['repo'] or '').lower() in vrepos for r in kept),
                              by_source=by('source'), by_setting=by('setting')))
    qa_path = os.path.join(out_dir, os.path.splitext(os.path.basename(a.out))[0].replace('rows', 'qa') + '.json')
    json.dump(qa, open(qa_path, 'w'), indent=1)
    print(json.dumps(qa, indent=1))


T = VIDS = VKEYS = None

if __name__ == '__main__':
    main()
