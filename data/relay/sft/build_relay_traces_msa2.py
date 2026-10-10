#!/usr/bin/env python3
"""MSA2 relay episodes -> parquet tabs for the public laion/calibforge-relay-traces dataset, in the schema of its
Terminus-2 tab (one row per rendered relay candidate; `messages` exactly as the SFT row renders it, with each turn's
owner and whether it carries loss), plus the structure mini-swe-agent tool mode needs (reasoning, tool calls, the bash
tool).

Episodes tagged `canary` or carrying a benchmark canary GUID anywhere are left out entirely (counted as dropped_canary),
as in the Terminus-2 tab, so the dataset never trips canary-based contamination checks.

Candidates and their QA tags come from render_msa.py's rows.jsonl of each group; each episode is re-rendered from its
run (render_msa.render, the router's turns) to recover the messages. Splits, per tab:
  used             the episode is in at least one reported SFT arm's rows (`used_in`);
  unused_clean     eligible under msa_select.py's rule (verifier ran, no weak timeout, no leak / hunt / canary, fits
                   65,536 tokens, passed, ends on a trained submit) but left out by the at-most-2-per-task draw;
  unused_filtered  fails that rule (failures included: MSA2 arms train passes only).

    python build_relay_traces_msa2.py --out <dir> [--procs 4]
writes <out>/data/<tab>-<split>-NNNNN.parquet and <out>/msa2_splits.json.
"""
import argparse
import collections
import json
import multiprocessing as mp
import os
import sys

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'pilot'))
import render_msa as rm  # noqa: E402
from hz_pool_census import CANARY  # noqa: E402
import readout  # noqa: E402
import select_kept  # noqa: E402

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

E = '/scratch/11584/lukedhlee/experiments/relay_msa'
O = '/scratch/11584/lukedhlee/experiments/sft_data/msa_20261003'
TOKDIR = '/scratch/11584/lukedhlee/models/grug-datakit-sft-20260921'
SUBMIT = 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'
STUDENT, TEACHER = 'Grug Datakit 09-21', 'Qwen3.8-27B'

# tab -> [(variant, render_msa rows dir, description)]
GROUPS = {
    'msa2_calibforge': [
        ('relay_32k', ['rows_pilot1_relay_repair', 'rows_full_part1_relay_repair', 'rows_full_part2_relay_repair'],
         'relay_repair: one-turn repair of an unparseable student reply; sticky takeover on a done claim, a loop, '
         'no progress, or when the student view reaches 32k tokens'),
        ('guided_v2', ['rows_assist2_relay_repair'],
         'guided relay: as relay_32k plus non-sticky one-turn teacher assists (command timeout, 4 failing commands in '
         'a row, a repeated output, context check-ins at 20k / 35k; at most 6, 3 turns apart), sticky hand-off at 40k '
         'tokens or once the student has used 60 % of the task\'s time budget, teacher hint on'),
    ],
    'msa2_tmax': [
        ('relay_32k', ['rows_tmax_relay_repair'], 'as msa2_calibforge relay_32k, on TMax tasks'),
        ('guided_v2', ['rows_assist2tmax_relay_repair'], 'as msa2_calibforge guided_v2, on TMax tasks'),
    ],
}
# SFT row files (component files of the arms) -> the arms that trained on them (marin stages relay_<arm>)
COMPONENTS = {
    'msasub_rows.jsonl': ['msafin', 'msafins1', 'msafinnr', 'msafinnr2', 'msafine4', 'msafinmask', 'msafinnotmax'],
    'tmaxsub_rows.jsonl': ['msafin', 'msafins1', 'msafinnr', 'msafinnr2', 'msafine4', 'msafinmask', 'msafing1',
                           'msafing1s1', 'msafing2', 'msafing2s1', 'msafing2nr', 'msafing2nrcf', 'msafing2nrcfs1'],
    'assist2/assist2_rows.jsonl': ['msafing2', 'msafing2s1', 'msafing2nr', 'msafing2nrcf', 'msafing2nrcfs1',
                                   'msafing3', 'msafing3s1'],
    'assist2tmax/assist2tmax_rows.jsonl': ['msafing3', 'msafing3s1'],
}

MSG = pa.struct([('role', pa.string()), ('owner', pa.string()), ('content', pa.string()),
                 ('reasoning_content', pa.string()), ('tool_calls', pa.string()), ('tool_call_id', pa.string()),
                 ('trained', pa.bool_()), ('reasoning_trained', pa.bool_())])
SCHEMA = pa.schema([
    ('source', pa.string()), ('variant', pa.string()), ('run', pa.string()), ('student', pa.string()),
    ('teacher', pa.string()), ('task', pa.string()), ('trial', pa.string()), ('passed', pa.bool_()),
    ('reward', pa.float64()), ('failure_cause', pa.string()), ('takeover', pa.string()),
    ('student_turns', pa.int64()), ('teacher_turns', pa.int64()), ('repairs', pa.int64()), ('assists', pa.int64()),
    ('leak', pa.bool_()), ('hunt', pa.bool_()), ('canary', pa.bool_()), ('ends_submit', pa.bool_()),
    ('sid', pa.string()), ('n_tokens', pa.int64()), ('trained_tokens', pa.int64()),
    ('used_in', pa.list_(pa.string())), ('tools', pa.string()), ('messages', pa.list_(MSG)),
])


def sid_of(line):
    i = line.index('"sid": "') + 8
    return line[i:line.index('"', i)].split('#')[0]


def used_index():
    out = collections.defaultdict(set)
    for f, arms in COMPONENTS.items():
        with open(os.path.join(O, f)) as fh:
            for line in fh:
                out[sid_of(line[:4000])].update(arms)
    return out


def meta(path, tok):
    """sid -> candidate metadata and msa_select eligibility (pass + ends on a trained submit)."""
    out = {}
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            tail = tok.decode(r['ids'][-80:], skip_special_tokens=False)
            ends = (SUBMIT in tail[tail.rfind('<tool_call>'):] and tail.rstrip().endswith('<|eot_id|>')
                    and r['loss'][-1] == 1)
            ok = (r.get('verifier_ran') and not r.get('weak_timeout') and not (r.get('leak') or r.get('hunt') or
                  r.get('canary')) and r.get('fits') and r['n_tokens'] <= rm.MAX_TOKENS and r.get('trained_tokens', 0) > 0
                  and r.get('passed') and ends)
            r.pop('ids'), r.pop('loss')
            r.update(ends_submit=ends, eligible=bool(ok))
            out[r['sid']] = r
    return out


def episodes(run):
    """sid -> (student-view messages, per-assistant-turn info) for every candidate of one run (render_msa.one_run)."""
    tok = rm.rcap.load_tokenizer(os.path.join(TOKDIR, 'tokenizer.json'))
    tpl = rm.load_template(os.path.join(TOKDIR, 'chat_template.jinja'))
    name = os.path.basename(run)
    rv = readout.router_view(os.path.join(run, 'router_relay_repair'))
    start = next((e for e in rv['events'] if e.get('event') == 'start'), {})
    argv = start.get('argv') or []
    strip = ('strip' in argv[argv.index('--student-view-after-takeover') + 1:][:1]
             if '--student-view-after-takeover' in argv else False)
    recs = collections.defaultdict(list)
    for r in rv['recs']:
        recs[r['sid']].append(r)
    trials = {t['sid']: t for t in readout.arm_trials(run, name, 'relay_repair')}
    out = {}
    for c in select_kept.arm_rows(run, name, 'relay_repair'):
        t = trials.get(c['sid']) or {}
        up = os.path.join(os.path.dirname(t.get('traj_path') or ''), readout.MSA_TRAJECTORY)
        try:
            msgs = json.load(open(up))['messages']
            row = rm.render(msgs, recs[c['sid']], tok, tpl, BOS, strip_after_takeover=strip)
        except (OSError, ValueError, KeyError):
            continue
        assists = sum(1 for r in recs[c['sid']] if r.get('assist'))
        out[c['sid']] = (convert(row), assists)
    return name, out


def convert(row):
    """render_msa.render's student-view messages -> [{role, owner, content, reasoning_content, tool_calls, ...}]."""
    info = {m['i']: m for m in row['turns']}
    out = []
    for i, m in enumerate(row['messages']):
        role = m['role']
        d = dict(role=role, content=m.get('content') or '', reasoning_content=m.get('reasoning_content'),
                 tool_calls=json.dumps(m['tool_calls'], ensure_ascii=False) if m.get('tool_calls') else None,
                 tool_call_id=m.get('tool_call_id'), trained=False, reasoning_trained=False)
        if role == 'assistant':
            t = info[i]
            d['owner'] = t['owner']
            d['trained'] = t['owner'] == 'teacher'
            d['reasoning_trained'] = bool(d['trained'] and t.get('think_trained') and m.get('reasoning_content'))
        else:
            d['owner'] = 'system' if role == 'system' else 'environment'
        out.append(d)
    return out


def main():
    global BOS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--procs', type=int, default=4)
    a = ap.parse_args()
    bos = json.load(open(os.path.join(TOKDIR, 'tokenizer_config.json'))).get('bos_token')
    BOS = bos.get('content') if isinstance(bos, dict) else bos
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(TOKDIR, 'tokenizer.json'))
    used = used_index()
    os.makedirs(os.path.join(a.out, 'data'), exist_ok=True)
    stats = {}
    for tab, groups in GROUPS.items():
        cands = {}
        for variant, dirs, _ in groups:
            for d in dirs:
                for sid, r in meta(os.path.join(E, d, 'rows.jsonl'), tok).items():
                    cands[sid] = dict(r, variant=variant)
        runs = sorted({r['run'] for r in cands.values()})
        with mp.Pool(a.procs, initializer=_init, initargs=(BOS,)) as pool:
            rendered = dict(pool.imap_unordered(episodes, [os.path.join(E, 'runs', x) for x in runs]))
        rows = collections.defaultdict(list)
        missing = dropped = 0
        for sid, r in cands.items():
            ep = rendered.get(r['run'], {}).get(sid)
            if ep is None:
                missing += 1
                continue
            msgs, assists = ep
            if r['canary'] or CANARY.search(json.dumps(msgs, ensure_ascii=False)):
                dropped += 1
                continue
            arms = sorted(used.get(sid, ()))
            split = 'used' if arms else 'unused_clean' if r['eligible'] else 'unused_filtered'
            rows[split].append(dict(
                source='horizon_msa2', variant=r['variant'], run=r['run'], student=STUDENT, teacher=TEACHER,
                task=r['task'], trial=r['trial'], passed=bool(r['passed']), reward=r['reward'],
                failure_cause=r.get('cause'), takeover=r.get('takeover'), student_turns=r['student_turns'],
                teacher_turns=r['teacher_turns'], repairs=r['repairs'], assists=assists, leak=bool(r['leak']),
                hunt=bool(r['hunt']), canary=bool(r['canary']), ends_submit=r['ends_submit'], sid=sid,
                n_tokens=r['n_tokens'], trained_tokens=r['trained_tokens'], used_in=arms,
                tools=json.dumps([rm.BASH_TOOL]), messages=msgs))
        st = {'missing_render': missing, 'dropped_canary': dropped}
        for split, rs in rows.items():
            rs.sort(key=lambda x: (x['variant'], x['run'], x['sid']))
            n = max(1, (sum(x['n_tokens'] for x in rs) * 4) // (400 << 20) + 1)    # ~400 MB of text per shard
            per = (len(rs) + n - 1) // n
            for k in range(n):
                part = rs[k * per:(k + 1) * per]
                if part:
                    pq.write_table(pa.Table.from_pylist(part, schema=SCHEMA),
                                   os.path.join(a.out, 'data', f'{tab}-{split}-{k:05d}.parquet'), row_group_size=64)
            st[split] = dict(collections.Counter(f"{x['variant']}/{'pass' if x['passed'] else 'fail'}" for x in rs))
        stats[tab] = st
        print(tab, json.dumps(st), flush=True)
    json.dump(stats, open(os.path.join(a.out, 'msa2_splits.json'), 'w'), indent=1)


def _init(bos):
    global BOS
    BOS = bos


BOS = None

if __name__ == '__main__':
    main()
