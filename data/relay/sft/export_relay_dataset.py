#!/usr/bin/env python3
"""One dataset of every labelled CalibForge relay episode (Jupiter 09-26..28 and Horizon 09-29), as readable messages,
for sharing (Luke 2026-10-01). One row per episode:

  sid, source, run, student, teacher, task, trial, passed, reward, failure_cause, takeover, student_turns,
  teacher_turns, repairs, leak, hunt, canary, n_tokens, trained_tokens, used_in, messages

`messages` is the episode exactly as an SFT row renders it (render_think_limit.py, 09-28 settings), decoded with the
09-21 tokenizer and split at the chat headers: {role, owner, content, trained}. role is system / user / assistant; owner
is system, environment (task prompt and terminal output), student or teacher; trained says whether any token of the
message carries loss (only teacher turns do). The k-th assistant message is the row's k-th `turns` entry.

Inputs: Horizon = every rendered candidate (hz_render/*.rendered.jsonl) with its census line (labels and QA tags);
Jupiter = arm A's 1,616 rows with final_v2_manifest.jsonl (the other Jupiter candidates have no labels on Horizon),
tagged with build_hz_arms.tag_rendered. used_in lists the SFT arms whose rows contain the episode.

    python export_relay_dataset.py --hz-rendered <dir> --hz-census <census.jsonl> --jup-rows <arm A rows> \
        --jup-manifest <final_v2_manifest.jsonl> --tokenizer-dir <09-21> --arm-rows H1=<jsonl> H4=<jsonl> ... --out <dir>
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SH, EH, EOT = 128006, 128007, 128009   # <|start_header_id|>, <|end_header_id|>, <|eot_id|>
G = {}


def head(line):
    i = line.find('"ids"')
    return json.loads(line[:i].rstrip(', ') + '}')


def messages(r, tok):
    ids, loss, turns = r['ids'], r['loss'], r['turns']
    out, k, i, n = [], 0, 0, len(ids)
    while i < n:
        if ids[i] != SH:
            i += 1
            continue
        e = ids.index(EH, i)
        role = tok.decode(ids[i + 1:e], skip_special_tokens=False).strip()
        try:
            end = ids.index(EOT, e)
        except ValueError:
            end = n
        if role == 'assistant':
            owner = turns[k]['owner'] if k < len(turns) else 'unknown'
            k += 1
        else:
            owner = 'system' if role == 'system' else 'environment'
        out.append(dict(role=role, owner=owner, content=tok.decode(ids[e + 1:end], skip_special_tokens=False),
                        trained=any(loss[e + 1:min(end + 1, n)])))
        i = end + 1
    assert k == len(turns), (r['sid'], k, len(turns))
    return out


def work(job):
    path, source = job
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(G['tokdir'], 'tokenizer.json'))
    labels, used = G['labels'], G['used']
    if source == 'jupiter':
        from build_hz_arms import tag_rendered
    rows = []
    for line in open(path):
        h = head(line)
        lab = labels.get(h['sid'])
        if lab is None:
            continue
        r = json.loads(line)
        if source == 'jupiter':
            lab = dict(lab, **tag_rendered(r, tok))
        rows.append(dict(lab, sid=r['sid'], n_tokens=r['n_tokens'], trained_tokens=sum(r['loss']),
                         used_in=sorted(used.get(r['sid'], [])), messages=messages(r, tok)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--hz-rendered', required=True)
    ap.add_argument('--hz-census', required=True)
    ap.add_argument('--jup-rows', required=True)
    ap.add_argument('--jup-manifest', required=True)
    ap.add_argument('--tokenizer-dir', required=True)
    ap.add_argument('--arm-rows', nargs='*', default=[], help='ARM=<rows jsonl>: arms whose rows to mark in used_in')
    ap.add_argument('--out', required=True)
    ap.add_argument('--shard-rows', type=int, default=1000)
    ap.add_argument('--procs', type=int, default=16)
    a = ap.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq

    labels = {}
    for l in open(a.hz_census):
        c = json.loads(l)
        labels[c['sid']] = dict(source='horizon_0929', run=c['run'], student='relay SFT arm A (step 246)',
                                teacher='Qwen3.8-27B', task=c['task'], trial=c['trial'], passed=bool(c['passed']),
                                reward=c['reward'], failure_cause=c['cause'], takeover=c['takeover'],
                                student_turns=c['student_turns'], teacher_turns=c['teacher_turns'], repairs=c['repairs'],
                                leak=c['leak_turns'] > 0, hunt=c['hunt_turns'] > 0, canary=bool(c['canary']))
    for l in open(a.jup_manifest):
        c = json.loads(l)
        labels[c['sid']] = dict(source='jupiter_0928', run='relay_full_relaym6_20260928', student='Grug Datakit 09-21',
                                teacher='Qwen3.8-27B', task=c['task'], trial=c['trial'], passed=bool(c['passed']),
                                reward=c['reward'], failure_cause=c['cause'], takeover=c['takeover'],
                                student_turns=c['student_turns'], teacher_turns=c['teacher_turns'], repairs=c['repairs'])
    used = {}
    for s in [l['sid'] for l in map(json.loads, open(a.jup_manifest))]:
        used.setdefault(s, set()).update(('A', 'H2', 'H3'))   # H2 / H3 train arm A's rows (masked thinking / + Kimi)
    for spec in a.arm_rows:
        arm, path = spec.split('=', 1)
        for line in open(path):
            s = head(line)['sid']
            if not s.startswith('kimi:'):
                used.setdefault(s, set()).add(arm)
    G.update(tokdir=a.tokenizer_dir, labels=labels, used={k: sorted(v) for k, v in used.items()})
    jobs = [(p, 'horizon') for p in sorted(glob.glob(os.path.join(a.hz_rendered, '*.rendered.jsonl')))] + [(a.jup_rows, 'jupiter')]
    os.makedirs(os.path.join(a.out, 'data'), exist_ok=True)
    buf, shard, total, stats = [], 0, 0, {}

    def flush():
        nonlocal buf, shard
        if buf:
            pq.write_table(pa.Table.from_pylist(buf), os.path.join(a.out, 'data', f'train-{shard:05d}.parquet'), compression='zstd')
            shard += 1
            buf = []
    with mp.Pool(min(a.procs, len(jobs))) as pool:
        for rows in pool.imap_unordered(work, jobs):
            for r in rows:
                buf.append(r)
                total += 1
                key = (r['source'], 'pass' if r['passed'] else 'fail')
                stats[key] = stats.get(key, 0) + 1
                if len(buf) >= a.shard_rows:
                    flush()
    flush()
    rep = dict(rows=total, shards=shard, by_source_outcome={f'{s}/{o}': n for (s, o), n in sorted(stats.items())},
               labelled=len(labels))
    json.dump(rep, open(os.path.join(a.out, 'export.json'), 'w'), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
