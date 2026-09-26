#!/usr/bin/env python3
"""extract.py — the done_claim takeover requests of the relay runs, for the verification-note replay.

At the student's first `task_complete: true` the router lets the reply through; Terminus-2 then sends its "Are you sure?"
confirmation request, and the router hands the episode to the teacher on THAT request (relay_router.py handle():
ep.pending + kind == 'confirm' -> take_over, rec['takeover'] set, owner teacher). So the target is exactly the
turns.jsonl line with takeover.trigger == 'done_claim' and request_kind == 'confirm'; its sent_body is the body the
teacher got (think-stripped student history). One per episode (take_over fires once).

Outcome: the episode's sid is the X-Harbor-Session-Id = trajectory.json session_id; the attempt's result.json holds
the verifier reward. Login-node safe (stdlib only).

    python3 extract.py --out requests.jsonl.gz [--runs-root ...] [run ...]
"""
import argparse
import glob
import gzip
import json
import os
import re

ROOT = '/e/fscratch/reformo/lee27/experiments/relay/pilot/runs'
RUNS = ['relay_run3b_20260925', 'relay_check_20260925', 'relay_recheck_20260925']
SID_RE = re.compile(r'"session_id":\s*"([^"]+)"')


def outcomes(run_dir):
    out = {}
    for traj in glob.glob(os.path.join(run_dir, 'jobs', '*relay_repair*', '*', 'attempts', '*', 'agent', 'trajectory.json')):
        with open(traj) as f:
            m = SID_RE.search(f.read(4096))
        if not m:
            continue
        att = os.path.dirname(os.path.dirname(traj))
        reward, exc = None, None
        try:
            r = json.load(open(os.path.join(att, 'result.json')))
            reward = ((r.get('verifier_result') or {}).get('rewards') or {}).get('reward')
            exc = (r.get('exception_info') or {}).get('exception_type') if isinstance(r.get('exception_info'), dict) else None
        except (OSError, ValueError):
            pass
        out[m.group(1)] = dict(reward=reward, exception=exc, trial=os.path.relpath(att, run_dir))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs-root', default=ROOT)
    p.add_argument('--out', required=True)
    p.add_argument('runs', nargs='*', default=RUNS)
    a = p.parse_args()
    n = 0
    with gzip.open(a.out, 'wt') as fo:
        for run in a.runs:
            rd = os.path.join(a.runs_root, run)
            ld = os.path.join(rd, 'router_relay_repair')
            oc = outcomes(rd)
            recs = [json.loads(l) for l in open(os.path.join(ld, 'turns.jsonl'))]
            tk = [r for r in recs if (r.get('takeover') or {}).get('trigger') == 'done_claim' and r.get('request_kind') == 'confirm']
            bodies = {}
            for r in tk:
                fn, seq = r['sent_body'].split('#')
                if fn not in bodies:
                    bodies[fn] = {d['seq']: d for d in map(json.loads, gzip.open(os.path.join(ld, fn), 'rt'))}
                d = bodies[fn][int(seq)]
                assert d['who'] == 'teacher', r['sent_body']
                o = oc.get(r['sid'], {})
                resp = r.get('response') or {}
                msg = (resp.get('choices') or [{}])[0].get('message') or {}
                fo.write(json.dumps(dict(
                    key=f"{run}/ep{r['episode']:05d}", run=run, episode=r['episode'], sid=r['sid'], task_id=r['task_id'],
                    turn=r['turn'], n_messages=r['n_messages'], prompt_tokens=(r.get('usage') or {}).get('prompt_tokens'),
                    reward=o.get('reward'), exception=o.get('exception'), trial=o.get('trial'),
                    orig_content=msg.get('content'), orig_reasoning=msg.get('reasoning_content') or msg.get('reasoning'),
                    orig_usage=r.get('usage'), orig_finish=r.get('finish_reason'),
                    body=d['body'])) + '\n')
                n += 1
            print(f'{run}: {len(tk)} done_claim takeovers, {sum(1 for r in tk if r["sid"] in oc)} joined to a result')
    print(f'wrote {n} -> {a.out}')


if __name__ == '__main__':
    main()
