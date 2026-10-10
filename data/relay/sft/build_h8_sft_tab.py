#!/usr/bin/env python3
"""The `terminus2_h8_sft` tab of laion/calibforge-relay-traces: exactly the 10,308 conversations our best Terminus-2
model H8 (laion/snowball-67b-a2b-relay-sft-allkimi-step1203) trained on, in one table and one schema:
  * 5,916 CalibForge relay episodes: the dataset's `used` split, copied as is (source jupiter_0928 / horizon_0929);
  * 4,392 Kimi-2.5 SWE-smith Terminus-2 traces: the trials in kimi_trials.txt, taken from our converted copy of
    open-athena/Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k (assistant turns already in the
    relay rows' form: <|start_think|>reasoning<|end_think|> + the Terminus-2 JSON), every assistant turn trained.

    python build_h8_sft_tab.py --relay-dir <download of laion/calibforge-relay-traces> --kimi <kimi_swesmith_v1 train
                               parquet> --out <dir>/data/terminus2_h8_sft
"""
import argparse
import glob
import os

import pyarrow as pa
import pyarrow.parquet as pq

KIMI_TEACHER = 'Kimi-2.5'
KIMI_SOURCE = 'kimi_swesmith'
KIMI_REPO = 'open-athena/Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--relay-dir', required=True)
    ap.add_argument('--kimi', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    used = pa.concat_tables(pq.read_table(f) for f in sorted(glob.glob(os.path.join(a.relay_dir, 'data/used-*.parquet'))))
    assert all('H8' in u for u in used.column('used_in').to_pylist()), 'a used row is not in H8'
    schema = used.schema
    trials = set(open(os.path.join(a.relay_dir, 'kimi_trials.txt')).read().split())
    kimi = pq.read_table(a.kimi).to_pylist()
    rows = []
    for r in kimi:
        if r['trial_name'] not in trials:
            continue
        msgs = []
        for m in r['conversations']:
            asst = m['role'] == 'assistant'
            msgs.append(dict(role=m['role'], owner='teacher' if asst else ('system' if m['role'] == 'system' else 'environment'),
                             content=m['content'], trained=asst))
        rows.append(dict(source=KIMI_SOURCE, run=KIMI_REPO, student=None, teacher=KIMI_TEACHER, task=r['task'],
                         trial=r['trial_name'], passed=float(r['result']) >= 1.0, reward=float(r['result']),
                         failure_cause=None, takeover=None, student_turns=0,
                         teacher_turns=sum(m['role'] == 'assistant' for m in r['conversations']), repairs=0,
                         leak=False, hunt=False, canary=False, sid=r['trial_name'], n_tokens=None, trained_tokens=None,
                         used_in=['H3', 'H7', 'H8', 'H9'], messages=msgs))
    assert len(rows) == len(trials), (len(rows), len(trials))
    out = pa.concat_tables([used, pa.Table.from_pylist(rows, schema=schema)])
    os.makedirs(a.out, exist_ok=True)
    n = 8
    per = (out.num_rows + n - 1) // n
    for k in range(n):
        pq.write_table(out.slice(k * per, per), os.path.join(a.out, f'train-{k:05d}-of-{n:05d}.parquet'), row_group_size=64)
    print(dict(relay=used.num_rows, kimi=len(rows), total=out.num_rows))


if __name__ == '__main__':
    main()
