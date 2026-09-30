#!/usr/bin/env python3
"""PedaGEPA stage 5: the recovery pool. Horizon relay remainder episodes (09-29, student = relay SFT arm A) whose handover
fired on a mistake signal (done_claim, loop, no_progress_wait), as training rows with the render-time leak filter, plus a
blind judge view per episode.

Selection: hz_pool_census.py rows with one of those triggers, the verifier's pytest ran, no weak timeout, no grader
hunting, no canary; the teacher wrote >= 2 turns; the rendered row (render_think_limit.py, 09-28 settings, autofix-loss
none) fits 65,536 tokens and has trained tokens.
Leak filter (pre-registered 09-30): every trained span (a run of loss=1 tokens = teacher text) whose decoded text matches
LEAK is loss-masked; a row that keeps < 50 % of its trained tokens after that is dropped. The masked text stays in the
context (it is never a training target). This replaces "remove the sentence" in the proposal: stripping sentences out of
token rows would break the rendered template; masking keeps the row coherent.
Judge view: the task, then every reply labelled [A] (first agent) or [B] (second agent), the handover line, no reward.

    python s5_pool.py --census census.jsonl --rendered <hz_render dir> --runs <relay runs dir> \
        --tokenizer-dir <09-21> --out-dir <dir>
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'relay', 'sft'))
import condense  # noqa: E402
from hz_pool_census import LEAK as LEAK0  # noqa: E402

TRIGGERS = ('done_claim', 'loop', 'no_progress_wait')
LEAK = re.compile(LEAK0.pattern + r'|\b(took|take|taking|takes) over\b|\bhand(ed|ing)?[ -]?over\b|\bthe system is asking\b'
                  r'|\breminder\b', re.I)


def spans(loss):
    i, n = 0, len(loss)
    while i < n:
        if loss[i]:
            j = i
            while j < n and loss[j]:
                j += 1
            yield i, j
            i = j
        else:
            i += 1


def view(trial, trigger, reason, max_chars=45000):
    traj, res, reward, exc, att = condense.load_trial(trial)
    instr, turns = condense.parse_turns(traj)
    owners = ['A' if (t['model'] or '').startswith('snowball') else 'B' for t in turns]
    first_b = next((i for i, o in enumerate(owners) if o == 'B' and all(x == 'B' for x in owners[i:])), None)
    body = condense.render(instr, turns, max_chars).split('\n')
    out = []
    for line in body:
        m = re.match(r'--- T(\d+)', line) or re.match(r'T(\d+)(-T\d+)?: \d+ repl', line)
        if m:
            k = int(m.group(1)) - 1
            if first_b is not None and k == first_b:
                out.append(f'=== HANDOVER before T{k + 1}: agent A stopped (router signal: {trigger}, "{reason}"); '
                           f'agent B continues from here with the same terminal ===')
            if k < len(owners):
                line = line.replace(f'T{k + 1}', f'T{k + 1} [{owners[k]}]', 1)
        out.append(line)
    return '\n'.join(out), first_b, len(turns)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--census', required=True)
    ap.add_argument('--rendered', required=True)
    ap.add_argument('--runs', required=True)
    ap.add_argument('--tokenizer-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(a.tokenizer_dir, 'tokenizer.json'))
    os.makedirs(os.path.join(a.out_dir, 'views'), exist_ok=True)
    st = collections.Counter()

    cand = {}
    for l in open(a.census):
        c = json.loads(l)
        if c['takeover'] not in TRIGGERS:
            continue
        st['census_trigger'] += 1
        if c['weak_timeout'] or not c['verifier_ran'] or c['hunt_turns'] > 0 or c['canary']:
            st['drop_census'] += 1
            continue
        if c['teacher_turns'] < 2:
            st['drop_teacher_lt2'] += 1
            continue
        cand[c['sid']] = c
    eps = {}
    for f in glob.glob(os.path.join(a.runs, 'relay_hz_a_s*_2026092*', 'router_relay_repair', 'episodes.json')):
        for e in json.load(open(f))['episodes']:
            eps[e['sid']] = e

    rows_out = open(os.path.join(a.out_dir, 'pool_rows.jsonl'), 'w')
    man_out = open(os.path.join(a.out_dir, 'pool_manifest.jsonl'), 'w')
    for f in sorted(glob.glob(os.path.join(a.rendered, 'relay_hz_a_s*.rendered.jsonl'))):
        for line in open(f):
            i = line.find('"ids"')
            h = json.loads(line[:i].rstrip(', ') + '}')
            c = cand.get(h['sid'])
            if c is None:
                continue
            st['rendered_candidate'] += 1
            if not h['fits']:
                st['drop_over_64k'] += 1
                continue
            r = json.loads(line)
            loss = list(r['loss'])
            n0 = sum(loss)
            if n0 == 0:
                st['drop_no_trained'] += 1
                continue
            masked = 0
            for s, e in list(spans(loss)):
                if LEAK.search(tok.decode(r['ids'][s:e], skip_special_tokens=False)):
                    for k in range(s, e):
                        loss[k] = 0
                    masked += 1
            n1 = sum(loss)
            if n1 < 0.5 * n0:
                st['drop_leak_heavy'] += 1
                continue
            st['masked_rows'] += bool(masked)
            ep = eps.get(h['sid'], {})
            tk = ep.get('takeover') or {}
            trial = glob.glob(os.path.join(a.runs, c['run'], 'jobs', c['run'] + '_relay_repair*', c['trial']))
            if not trial:
                st['drop_no_trial'] += 1
                continue
            try:
                v, first_b, nturns = view(trial[0], c['takeover'], tk.get('reason', ''))
            except Exception as ex:  # noqa: BLE001
                st['drop_view_error'] += 1
                print('view error', h['sid'], ex, file=sys.stderr)
                continue
            if first_b is None:
                st['drop_no_handover_in_traj'] += 1
                continue
            open(os.path.join(a.out_dir, 'views', h['sid'] + '.txt'), 'w').write(v)
            r['loss'] = loss
            rows_out.write(json.dumps(r) + '\n')
            man_out.write(json.dumps(dict(sid=h['sid'], task=c['task'], run=c['run'], trial=trial[0], passed=c['passed'],
                                          trigger=c['takeover'], reason=tk.get('reason'), handover_reply=first_b + 1,
                                          replies=nturns, trained_tokens=n1, trained_tokens_before=n0,
                                          masked_spans=masked, leak_turns_census=c['leak_turns'])) + '\n')
            st['pool'] += 1
            st['pool_pass'] += bool(c['passed'])
    rows_out.close()
    man_out.close()
    json.dump(dict(st), open(os.path.join(a.out_dir, 'pool_stats.json'), 'w'), indent=1)
    print(json.dumps(dict(st), indent=1))


if __name__ == '__main__':
    main()
