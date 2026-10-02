#!/usr/bin/env python3
"""MSA relay gate numbers for one or more run dirs (router --harness msa): what the smoke / pilot rules judge.

    python gate.py <run dir> [...] [--arm relay_repair]

Prints one JSON object:
  trials / scored / passes / pass_rate, harness_errors, overflow_rate (ContextLengthExceeded / scored),
  student_parse_rate      share of the student's replies harbor's own parser accepts as served (before autofix/repair),
  student_autofix / student_repairs,
  teacher_first_ok_rate   share of teacher agent turns whose first sample parsed (format guard outcome 'ok'),
  teacher_shapes          how Qwen wrote its calls (xml / json / server / none),
  takeovers               by trigger, takeover rate over episodes,
  done_claim              takeovers, with the note, teacher ran a non-submit command before its submit (S5 'worked'),
  note_in_trajectories    trajectories whose text holds the note (must be 0: the note lives only in the teacher view),
  leak_turns              teacher turns naming another / the previous agent or the note (hz_pool_census LEAK),
  sessions_joined         trials whose session id has router records, episodes_without_task.
"""
import argparse
import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'pilot'))
sys.path.insert(0, os.path.join(HERE, '..', 'router'))
sys.path.insert(0, os.path.join(HERE, '..', 'sft'))
import readout  # noqa: E402
import relay_router as rr  # noqa: E402
from hz_pool_census import LEAK  # noqa: E402
import select_kept  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--arm', default='relay_repair')
    a = ap.parse_args()
    c = collections.Counter()
    shapes, trig = collections.Counter(), collections.Counter()
    outs = collections.Counter()
    for run in a.runs:
        name = os.path.basename(os.path.normpath(run))
        rv = readout.router_view(os.path.join(run, f'router_{a.arm}'))
        rows = readout.arm_trials(run, name, a.arm)
        readout.mark_censored(rv, rows)
        sids = {r['sid'] for r in rv['recs']}
        c['trials'] += len(rows)
        c['sessions_joined'] += sum(1 for t in rows if t['sid'] in sids)
        c['episodes'] += len(rv['eps'])
        c['episodes_without_task'] += sum(1 for e in rv['eps'].values() if not e['task_id'])
        c['fatal'] += bool(rv['fatal'])
        for t in rows:
            c['harness_errors'] += t['harness_error']
            if readout.usable(t) or t['exc'] == 'ContextLengthExceededError':
                c['scored'] += 1
                c['overflow'] += t['exc'] == 'ContextLengthExceededError'
                c['passes'] += readout.usable(t) and readout.is_pass(t)
            outs[t['exc'] or ('pass' if readout.is_pass(t) else 'fail')] += 1
            if t.get('traj_path'):
                d = os.path.dirname(t['traj_path'])
                txt = ''.join(open(p).read() for p in glob.glob(os.path.join(d, '*.json')))
                c['note_in_trajectories'] += rr.VERIFY_NOTE_MSA in txt
        for r in rv['recs']:
            # every reply the student served: its own turns, and the ones the router discarded (repairs, takeovers)
            if (r.get('owner') == 'student' and r.get('upstream_status') == 200) or r.get('discarded_student_reply'):
                c['student_replies'] += 1
                c['student_parse_errors'] += bool(r.get('student_parse_error'))
            c['student_autofix'] += bool(r.get('autofix')) and r.get('owner') == 'student'
            c['student_repairs'] += r.get('repair_kind') == 'parse_error'
            g = r.get('teacher_guard')
            if g:
                c['teacher_turns'] += 1
                c['teacher_first_ok'] += g.get('outcome') == 'ok'
                c['teacher_unparseable_passed'] += g.get('outcome') == 'unparseable_passed'
                shapes.update(g.get('shapes') or [])
                m = (((r.get('response') or {}).get('choices') or [{}])[0].get('message')) or {}
                c['leak_turns'] += bool(LEAK.search((m.get('content') or '') + '\n' + (m.get('reasoning_content') or '')))
        for e in rv['eps'].values():
            if e['takeover']:
                trig[e['takeover']['trigger']] += 1
                if e['takeover']['trigger'] == 'done_claim':
                    c['done_claim'] += 1
                    c['done_claim_with_note'] += bool(e['takeover']['first_teacher_record'].get('verify_note'))
                    c['done_claim_teacher_worked'] += select_kept.teacher_worked(e)
    sc = c['scored'] or 1
    out = dict(runs=[os.path.basename(os.path.normpath(r)) for r in a.runs], **c,
               pass_rate=round(c['passes'] / sc, 4), overflow_rate=round(c['overflow'] / sc, 4),
               harness_error_rate=round(c['harness_errors'] / max(1, c['trials']), 4),
               student_parse_rate=round(1 - c['student_parse_errors'] / max(1, c['student_replies']), 4),
               teacher_first_ok_rate=round(c['teacher_first_ok'] / max(1, c['teacher_turns']), 4),
               takeover_rate=round(sum(trig.values()) / max(1, c['episodes']), 4), takeovers=dict(trig),
               teacher_shapes=dict(shapes), outcomes=dict(outs),
               done_claim_worked_rate=round(c['done_claim_teacher_worked'] / max(1, c['done_claim']), 4))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
