#!/usr/bin/env python3
"""Whole-trace contamination check of an SFT set against our eval task trees (TB2.1, SWE-bench Verified random-100,
TB-lite), per trace and per eval task:
  * instruction overlap: word 14-grams shared between ANY message of the trace (prompt, actions, observations) and an
    eval task's instruction.md (a hit = >= --min-shared shared 14-grams with one task);
  * solution / test leakage: distinctive lines (>= 40 characters after whitespace normalisation, not boilerplate seen in
    >= 5 eval tasks) of an eval task's solution/ and tests/ files appearing verbatim in the trace; for SWE-bench these
    hold the gold patch and the fail-to-pass tests.
Also reports the instance / task names shared with the eval trees. Prints a summary and writes every hit with context.

    python contamination_check.py --parquet <sft parquet> --eval-dirs <tb21> <swe100> <tblite> --out hits.json
"""
import argparse
import collections
import glob
import json
import os
import re

WORD = re.compile(r'\w+')
N = 14


def shingles(text):
    w = WORD.findall(text.lower())
    return {' '.join(w[i:i + N]) for i in range(max(0, len(w) - N + 1))}


def norm(line):
    return ' '.join(line.split())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--parquet', required=True)
    ap.add_argument('--eval-dirs', nargs='+', required=True)
    ap.add_argument('--min-shared', type=int, default=3)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import pyarrow.parquet as pq

    ins_index = collections.defaultdict(set)
    line_index = collections.defaultdict(set)
    names = {}
    for d in a.eval_dirs:
        s = os.path.basename(os.path.normpath(d))
        for td in glob.glob(os.path.join(d, '*')):
            if not os.path.isdir(td):
                continue
            key = (s, os.path.basename(td))
            names[key[1].lower()] = key
            p = os.path.join(td, 'instruction.md')
            if os.path.exists(p):
                for g in shingles(open(p, errors='replace').read()):
                    ins_index[g].add(key)
            for sub in ('solution', 'tests'):
                for f in glob.glob(os.path.join(td, sub, '**', '*'), recursive=True):
                    if not os.path.isfile(f) or os.path.getsize(f) > 2_000_000:
                        continue
                    try:
                        txt = open(f, errors='strict').read()
                    except (UnicodeDecodeError, OSError):
                        continue
                    for ln in txt.splitlines():
                        n = norm(ln)
                        if len(n) >= 40:
                            line_index[n].add((key, sub))
    common = {l for l, ks in line_index.items() if len({k for k, _ in ks}) >= 5}   # boilerplate across many tasks
    for l in common:
        del line_index[l]

    rows = pq.read_table(a.parquet).to_pylist()
    ins_hits, line_hits, name_hits = [], [], []
    per_set = collections.defaultdict(lambda: dict(instruction=set(), solution=set(), tests=set(), name=set()))
    for i, r in enumerate(rows):
        conv = r.get('conversations') or []
        text = '\n'.join(m.get('content') or '' for m in conv)
        c = collections.Counter()
        for g in shingles(text):
            for k in ins_index.get(g, ()):
                c[k] += 1
        for k, n in c.items():
            if n >= a.min_shared:
                ins_hits.append(dict(row=i, id=r.get('instance_id') or r.get('trial_name'), set=k[0], task=k[1], shared_14grams=n))
                per_set[k[0]]['instruction'].add(k[1])
        seen = set()
        for ln in text.splitlines():
            n = norm(ln)
            if len(n) >= 40 and n in line_index and n not in seen:
                seen.add(n)
                for k, sub in line_index[n]:
                    line_hits.append(dict(row=i, id=r.get('instance_id') or r.get('trial_name'), set=k[0], task=k[1], file=sub, line=n[:200]))
                    per_set[k[0]][sub].add(k[1])
        rid = str(r.get('instance_id') or '').lower()
        for nm, k in names.items():
            if rid and (rid == nm or rid.startswith(nm + '.') or nm in rid.split('.')):
                name_hits.append(dict(row=i, id=rid, set=k[0], task=k[1]))
                per_set[k[0]]['name'].add(k[1])
    summary = dict(rows=len(rows), eval_tasks=len(names), distinctive_eval_lines=len(line_index),
                   dropped_boilerplate_lines=len(common),
                   instruction_hits=len(ins_hits), line_hits=len(line_hits), name_hits=len(name_hits),
                   per_set={s: {k: sorted(v) for k, v in d.items()} for s, d in per_set.items()})
    json.dump(dict(summary=summary, instruction_hits=ins_hits, line_hits=line_hits[:2000], name_hits=name_hits),
              open(a.out, 'w'), indent=1)
    print(json.dumps(summary, indent=1))
    by = collections.Counter((h['set'], h['file']) for h in line_hits)
    print('line hits by set/file:', dict(by))
    for h in line_hits[:15]:
        print('  ', h['set'], h['task'], h['file'], '|', h['line'][:140])


if __name__ == '__main__':
    main()
