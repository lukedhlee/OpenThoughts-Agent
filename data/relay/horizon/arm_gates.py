#!/usr/bin/env python3
"""In-run gates for a relay arm whose student is not 09-21 (the finetuned arm A): run_pilot.sh ARM_GATES=1 calls this
every ARM_GATES_EVERY s. Both gates were calibrated on 09-21 only, so by default they flag (log + <run>/FLAGS), they do
not cancel; --takeover-action stop / --accept-stop turn them into stops.

    python arm_gates.py <run dir> <run name> <arm> --student-urls http://c101-003:8000/v1[,...] \
        [--takeover-min 0.50 --takeover-max 0.95 --takeover-after 100 --takeover-action flag] [--accept-flag 1.8 --accept-stop 0]

Takeover rate = finished relay episodes with a sticky takeover / finished relay episodes. Finished = the trial wrote
result.json, it is not a harness error (a context overflow counts: it is the student's episode running out of
room) and the run's deadline did not end it (router "ending" event "deadline"). The takeover comes from the router's
events.jsonl ("takeover" events, by session id), the episode's session id from its trajectory.json, as readout.py joins
them. Judged once --takeover-after episodes have finished; until then "wait". 09-21 with the 32k context budget read
0.89 on this measure (attempts 6a / 6b, 09-27/28: 703 and 948 finished episodes); 0.28 without the budget (run 3,
done_claim only).

Draft acceptance = the student servers' mean acceptance length, 1 + accepted / drafts from vLLM's spec-decode counters
(vllm:spec_decode_num_drafts_total, vllm:spec_decode_num_accepted_tokens_total on /metrics, summed over engines and
servers), over the window since the previous call (state in <run dir>/arm_gates_state.json; needs >= 2,000 drafts, else
the cumulative value is used). 09-21 with its adapted EAGLE-3 draft (k=3): about 2.5-2.6. Below --accept-flag: flag;
below --accept-stop: stop (0 = off).

Prints one line "<ok|flag|stop|wait> <json>" and appends the JSON to <run dir>/arm_gates.log. Exit 0 unless it crashed.
Incremental: finished trials and the events file offset are cached in the state file, so each call reads only new data.
"""
import argparse
import glob
import json
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pilot'))
import readout  # noqa: E402

SID = re.compile(rb'"session_id":\s*"([^"\\]+)"')


def trajectory_sid(tdir):
    trajs = sorted(glob.glob(os.path.join(tdir, '**', 'agent', 'trajectory.json'), recursive=True), key=os.path.getmtime)
    if not trajs:
        return None
    with open(trajs[-1], 'rb') as f:
        m = SID.search(f.read(65536))
    if m:
        return m.group(1).decode()
    try:
        return json.load(open(trajs[-1])).get('session_id')
    except (OSError, ValueError):
        return None


def new_events(path, st):
    """Takeover events ({sid: trigger}) and deadline endings (sids) appended since the last call. Only whole lines; the
    offset is kept in st."""
    off = st['offsets'].get(path, 0)
    if not os.path.exists(path):
        return
    with open(path, 'rb') as f:
        f.seek(off)
        data = f.read()
    end = data.rfind(b'\n') + 1
    for line in data[:end].splitlines():
        if b'"takeover"' not in line and b'"deadline"' not in line:
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get('event') == 'takeover' and e.get('sid'):
            st['takeovers'].setdefault(e['sid'], e.get('trigger'))
        elif e.get('event') == 'ending' and e.get('ending') == 'deadline' and e.get('sid'):
            st.setdefault('deadline', {})[e['sid']] = 1
    st['offsets'][path] = off + end


def spec_counters(url):
    txt = urllib.request.urlopen(re.sub(r'/v1/?$', '', url) + '/metrics', timeout=10).read().decode()
    out = {}
    for key, name in (('drafts', 'vllm:spec_decode_num_drafts'), ('accepted', 'vllm:spec_decode_num_accepted_tokens')):
        vals = [float(m.group(1)) for m in re.finditer(r'^' + name + r'(?:_total)?(?:\{[^}]*\})?\s+([0-9.eE+-]+)', txt, re.M)]
        out[key] = sum(vals) if vals else None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir'); ap.add_argument('name'); ap.add_argument('arm')
    ap.add_argument('--student-urls', default='')
    ap.add_argument('--takeover-min', type=float); ap.add_argument('--takeover-max', type=float)
    ap.add_argument('--takeover-after', type=int, default=100)
    ap.add_argument('--takeover-action', choices=['flag', 'stop'], default='flag')
    ap.add_argument('--accept-flag', type=float, default=0.0); ap.add_argument('--accept-stop', type=float, default=0.0)
    ap.add_argument('--min-drafts', type=int, default=2000)
    a = ap.parse_args()
    sp = os.path.join(a.run_dir, f'arm_gates_state_{a.arm}.json')
    st = json.load(open(sp)) if os.path.exists(sp) else dict(offsets={}, takeovers={}, trials={}, metrics={})
    out = dict(ts=round(time.time()), arm=a.arm)
    verdict, why = 'ok', []

    # ---- takeover rate over finished episodes
    new_events(os.path.join(a.run_dir, f'router_{a.arm}', 'events.jsonl'), st)
    for d in readout.arm_job_dirs(a.run_dir, a.name, a.arm):
        for rj in glob.glob(os.path.join(d, '*', 'result.json')):
            key = f'{os.path.basename(d)}/{os.path.basename(os.path.dirname(rj))}'
            if key in st['trials']:
                continue
            try:
                task, reward, exc = readout.read_outcome(rj)
            except (OSError, ValueError, KeyError):
                continue
            vt = exc == readout.VERIFIER_TIMEOUT
            herr = not vt and ((reward is None and exc != 'ContextLengthExceededError') or (exc is not None and exc not in readout.AGENT_ENDS))
            st['trials'][key] = [trajectory_sid(os.path.dirname(rj)), not herr]
    dead = st.get('deadline') or {}   # episodes the run's deadline ended are censored, as in readout.py
    fin = [sid for sid, ok in st['trials'].values() if ok and sid and sid not in dead]
    tk = [st['takeovers'][s] for s in fin if s in st['takeovers']]
    rate = round(len(tk) / len(fin), 4) if fin else None
    trig = {}
    for t in tk:
        trig[t] = trig.get(t, 0) + 1
    out['takeover'] = dict(finished=len(fin), with_takeover=len(tk), rate=rate, by_trigger=trig,
                           band=[a.takeover_min, a.takeover_max], after=a.takeover_after,
                           no_sid=sum(1 for sid, ok in st['trials'].values() if ok and not sid))
    if a.takeover_min is not None or a.takeover_max is not None:
        if len(fin) < a.takeover_after:
            out['takeover']['state'] = 'wait'
        elif (a.takeover_min is not None and rate < a.takeover_min) or (a.takeover_max is not None and rate > a.takeover_max):
            out['takeover']['state'] = a.takeover_action
            why.append(f'takeover rate {rate} over {len(fin)} finished episodes outside [{a.takeover_min}, {a.takeover_max}]')
            verdict = a.takeover_action
        else:
            out['takeover']['state'] = 'ok'

    # ---- draft acceptance on the student servers
    urls = [u for u in a.student_urls.split(',') if u]
    if urls and (a.accept_flag > 0 or a.accept_stop > 0):
        now, per, tot = time.time(), {}, dict(drafts=0.0, accepted=0.0, d_drafts=0.0, d_accepted=0.0)
        for u in urls:
            try:
                c = spec_counters(u)
            except Exception as e:  # noqa: BLE001
                per[u] = dict(error=type(e).__name__)
                continue
            if c['drafts'] is None:
                per[u] = dict(error='no spec-decode counters')
                continue
            prev = st['metrics'].get(u)
            mono = prev and c['drafts'] >= prev[1] and (c['accepted'] or 0) >= prev[2]   # a restarted server resets its counters
            dd = c['drafts'] - prev[1] if mono else c['drafts']
            da = (c['accepted'] or 0) - prev[2] if mono else (c['accepted'] or 0)
            st['metrics'][u] = [now, c['drafts'], c['accepted'] or 0]
            per[u] = dict(drafts=c['drafts'], cum=round(1 + (c['accepted'] or 0) / c['drafts'], 3) if c['drafts'] else None,
                          window=round(1 + da / dd, 3) if dd else None, window_drafts=dd)
            tot['drafts'] += c['drafts']; tot['accepted'] += c['accepted'] or 0
            tot['d_drafts'] += dd; tot['d_accepted'] += da or 0
        cum = round(1 + tot['accepted'] / tot['drafts'], 3) if tot['drafts'] else None
        win = round(1 + tot['d_accepted'] / tot['d_drafts'], 3) if tot['d_drafts'] >= a.min_drafts else None
        judged = win if win is not None else (cum if tot['drafts'] >= a.min_drafts else None)
        out['acceptance'] = dict(window=win, cumulative=cum, judged=judged, drafts=tot['drafts'], window_drafts=tot['d_drafts'],
                                 flag_below=a.accept_flag, stop_below=a.accept_stop, servers=per)
        if judged is None:
            out['acceptance']['state'] = 'wait' if not any('error' in v for v in per.values()) else 'error'
            if out['acceptance']['state'] == 'error':
                why.append('acceptance: ' + '; '.join(f"{u} {v['error']}" for u, v in per.items() if 'error' in v))
                verdict = 'flag' if verdict == 'ok' else verdict
        elif a.accept_stop > 0 and judged < a.accept_stop:
            out['acceptance']['state'] = 'stop'; verdict = 'stop'
            why.append(f'draft acceptance {judged} < {a.accept_stop}')
        elif a.accept_flag > 0 and judged < a.accept_flag:
            out['acceptance']['state'] = 'flag'; verdict = 'flag' if verdict == 'ok' else verdict
            why.append(f'draft acceptance {judged} < {a.accept_flag}')
        else:
            out['acceptance']['state'] = 'ok'

    if verdict == 'ok' and all((out.get(k) or {}).get('state') in (None, 'wait') for k in ('takeover', 'acceptance')):
        verdict = 'wait'
    out['verdict'], out['why'] = verdict, why
    json.dump(st, open(sp + '.tmp', 'w'))
    os.replace(sp + '.tmp', sp)
    with open(os.path.join(a.run_dir, 'arm_gates.log'), 'a') as f:
        f.write(json.dumps(out) + '\n')
    tkv, acv = out['takeover'], out.get('acceptance') or {}
    print(f"{verdict} takeover {tkv['rate']} over {tkv['finished']} finished {tkv['by_trigger']}; "
          f"acceptance window {acv.get('window')} cumulative {acv.get('cumulative')}" + (f"; {' | '.join(why)}" if why else ''))


if __name__ == '__main__':
    main()
