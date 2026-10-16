#!/usr/bin/env python3
"""Replay recorded relay traffic through a relay router in-process, and diff two routers' replays.

Stage 2 of notes/ot-agent/relay-triggers moved the router's decisions into interventions.Controller with no change in
behavior; this is its proof: the router before and after, fed the same recorded requests, must answer and log the same.

A replay sample: manifest.json (the sampled episodes: run, arm, episode), router_flags.json (each run's router argv, by
"<run>/<arm>"), <run>/router_<arm>/{turns.jsonl, bodies/ep<N>.jsonl.gz} and _inputs/ (the task files, tokenizers and
Terminus-2 parser the flags name, under their cluster paths).

run   For each sampled episode, rebuild harbor's requests in order from the logs: the history harbor holds, i.e. its
      own messages (taken from the logged upstream bodies) with each earlier assistant turn as the router returned it
      (with or without the reasoning keys, whichever reproduces the logged request_sha). Each request goes to
      Router.h_chat with the run's own flags. A stub in place of Router.post answers every upstream call from the log
      (the student's raw reply, the teacher's attempts in order, /tokenize counts, upstream errors); time.time() is a
      fake clock at the logged request times, advanced by the logged latencies during model calls; uuid4 is a counter.
      The router writes its usual logs; the replay adds responses.jsonl (what harbor got back), calls.jsonl (every
      upstream call: who, path, body hash) and replay.json (fidelity: request_sha matches, stub misses, and who
      answered each request against the recording). --inject-retries N sends every Nth request twice (a transport
      retry), to exercise the retry paths the recordings do not hold.
diff  Two replays, request by request: who answered, the legacy decision fields, the whole turns.jsonl line, and the
      events, upstream calls and responses. --ignore drops record keys (the additive --trigger-config fields).

    python router_replay.py run --router <relay_router.py> --sample <dir> --out <dir> [--inject-retries N]
        [--trigger-config-from-flags]
    python router_replay.py diff <replay A> <replay B> [--ignore decisions,intervention]
"""
import argparse
import asyncio
import collections
import copy
import gzip
import hashlib
import importlib.util
import json
import os
import sys
import time
import types

LEGACY_FIELDS = ('owner', 'repair', 'repair_kind', 'assist', 'assist_kind', 'takeover', 'done_claim',
                 'done_claim_dropped', 'verify_note', 'autofix', 'autofix_kind', 'discarded_student_reply', 'ending')
TIMING = ('ts', 'elapsed_sec', 'paused_sec', 'student_clock_sec', 'latency_sec', 'student_latency_sec',
          'paused_this_turn_sec')
NOTES = ("Note: another agent did the previous work, and its claim that the task is complete may be wrong. Before "
         "confirming, run commands that check the task's key requirements (outputs, files, tests).",
         "Note: the claim that the task is complete may be wrong. Before confirming, run commands that check the "
         "task's key requirements (outputs, files, tests).",
         "Before submitting, run commands that check the task's key requirements (outputs, files, tests).")
VALUE_FLAGS = {'--takeover', '--context-budget-tokens', '--context-budget-min-turn', '--time-takeover-frac',
               '--assist-ctx', '--assist-error-streak', '--assist-gap', '--assist-max', '--verify-note-text'}


def sha(s):
    return hashlib.sha1((s or '').encode('utf-8', 'surrogatepass')).hexdigest()[:16]


def msg_sha(messages):
    return sha(json.dumps(messages, sort_keys=True, ensure_ascii=False))


# ==== the recording ==========================================================================================
class Recording:
    """One run/arm of the sample: its argv and every sampled episode's logged rows and upstream bodies."""

    def __init__(self, sample, run, arm, episodes, flags):
        self.sample, self.run, self.arm = sample, run, arm
        self.dir = os.path.join(sample, run, f'router_{arm}')
        self.argv = list(flags['argv'][1:])
        self.substituted = None
        self.episodes = episodes
        want = set(episodes)
        self.rows = collections.defaultdict(list)
        for line in open(os.path.join(self.dir, 'turns.jsonl')):
            r = json.loads(line)
            if r.get('episode') in want:
                self.rows[r['episode']].append(r)
        for v in self.rows.values():
            v.sort(key=lambda r: r['seq'])

    def bodies(self, ep):
        p = os.path.join(self.dir, 'bodies', f'ep{ep:05d}.jsonl.gz')
        out = collections.defaultdict(list)
        if os.path.exists(p):
            for line in gzip.open(p, 'rt'):
                d = json.loads(line)
                out[d['seq']].append((d['who'], d['body']))
        return out

    def local(self, path):
        if not path.startswith(('/scratch/', '/home1/')):
            return path
        p = os.path.join(self.sample, '_inputs', path.lstrip('/'))
        if not os.path.exists(p) and p.endswith('tokenizer.json'):
            # a tokenizer missing from the sample (a dangling hf_hub symlink): any of the run's family will do, it only
            # sizes the reasoning cut in the student's view, the same for both routers
            alt = sorted(os.path.join(d, f) for d, _, fs in os.walk(os.path.join(self.sample, '_inputs')) for f in fs
                         if f == 'tokenizer.json' and os.path.exists(os.path.join(d, f)))
            self.substituted = (path, alt[0])
            return alt[0]
        return p


def text_of(c):
    if isinstance(c, list):
        return '\n'.join(p.get('text', '') for p in c if isinstance(p, dict))
    return c or ''


def harbor_turn(msg, msa, reasoning):
    """The assistant message harbor keeps for a reply the router returned (msg); reasoning: the harness re-sends the
    reasoning (Terminus-2 interleaved thinking; mini-swe-agent-host builds that keep it)."""
    m = {'role': 'assistant', 'content': msg.get('content')}
    r = msg.get('reasoning_content') or msg.get('reasoning')
    if reasoning and r:
        m['reasoning_content'] = r
        m['reasoning'] = r
    if msa and msg.get('tool_calls'):
        m['tool_calls'] = msg['tool_calls']
    return m


def clean(m, notes, hint):
    """A non-assistant message of a logged upstream body as harbor sent it (the teacher's copy may carry the verify
    note on the last message and the teacher hint on the system message)."""
    c = m.get('content')
    if isinstance(c, str):
        for n in notes:
            if c.endswith('\n\n' + n):
                return dict(m, content=c[:-len(n) - 2])
        if hint and m.get('role') == 'system' and c.endswith('\n\n' + hint):
            return dict(m, content=c[:-len(hint) - 2])
    return m


def placeholder(row, prev_reply, msa):
    """Observation messages for a request with no logged body anywhere (an ending at the episode's end)."""
    if msa:
        ids = [tc['id'] for tc in (prev_reply or {}).get('tool_calls') or []] or ['call_x_0']
        return [{'role': 'tool', 'content': '(not recorded)', 'tool_call_id': i, 'name': 'bash'} for i in ids]
    if row.get('request_kind') == 'confirm':
        return [{'role': 'user', 'content': 'Current terminal state:\n(not recorded)\n\nAre you sure you want to mark '
                                            'the task as complete? (not recorded)'}]
    return [{'role': 'user', 'content': 'New Terminal Output:\n(not recorded)'}]


def rebuild(rec, ep, msa, notes, hint, reasoning):
    """harbor's requests of one episode: [(row, body)] in order; each body is what harbor posted to the router."""
    rows, bodies = rec.rows[ep], rec.bodies(ep)
    extra = None
    for s in sorted(bodies):
        for who, b in bodies[s]:
            if who == 'student':
                extra = {k: v for k, v in b.items() if k not in ('messages', 'model')}
                break
        if extra is not None:
            break
    if extra is None:
        b = next((b for s in sorted(bodies) for _, b in bodies[s]), {})
        extra = {k: v for k, v in b.items() if k not in ('messages', 'model', 'max_tokens')}
    out, hist, prev = [], None, None
    for i, row in enumerate(rows):
        n = row['n_messages']
        src = None
        for s in [row['seq']] + [r['seq'] for r in rows[i + 1:]]:
            if bodies.get(s):
                src = sorted(bodies[s], key=lambda wb: wb[0] != 'student')[0][1]['messages']
                break
        if row.get('retry') and hist is not None:
            msgs = list(hist)
        elif hist is None:
            msgs = [clean(m, notes, hint) for m in (src or [])[:n]]
        else:
            base = list(hist)
            # the reply harbor got last time, unless it kept none (an upstream error; mini-swe-agent drops a reply
            # its parser rejects and sends only the format error): the logged body says which
            kept = src[len(base)].get('role') == 'assistant' if src and len(src) > len(base) else True
            if kept and prev is not None and prev.get('response') and prev.get('upstream_status') in (None, 200):
                base.append(harbor_turn(prev['response']['choices'][0]['message'], msa, reasoning))
            new = [clean(m, notes, hint) for m in (src or [])[len(base):n]] if src and len(src) >= n else None
            if new is None or any(m.get('role') == 'assistant' for m in new):
                pr = prev['response']['choices'][0]['message'] if prev and prev.get('response') else None
                new = placeholder(row, pr, msa)
            msgs = base + new
        body = dict(extra, model='relay', messages=msgs)
        out.append((row, body))
        hist, prev = msgs, row
    return out


# ==== the stub upstream ======================================================================================
def _err_bytes(row, key='upstream_error'):
    return (row.get(key) or '').encode('utf-8', 'surrogatepass')


def raw_student(row, msa):
    """The student's reply as its server sent it, or None when the router never asked it."""
    if row.get('original_student_reply'):
        return row['original_student_reply']
    d = row.get('discarded_student_reply')
    if d and msa and not row.get('repair'):
        # an msa decision takeover logs the parsed reply; give it back with the tool calls inline, as 09-21 writes them
        d = copy.deepcopy(d)
        m = d['choices'][0]['message']
        think = m.get('reasoning_content') or m.get('reasoning')
        calls = ''.join('<tool_call>\n' + json.dumps({'name': tc['function']['name'],
                                                     'arguments': json.loads(tc['function']['arguments'])}) +
                        '\n</tool_call>' for tc in m.get('tool_calls') or [])
        d['choices'][0]['message'] = {'role': 'assistant', 'content': (
            f'<|start_think|>{think}<|end_think|>' if think else '') + (m.get('content') or '') + calls}
        return d
    if d:
        return d
    if row.get('owner') == 'student' and row.get('response'):
        resp = copy.deepcopy(row['response'])
        if msa and row.get('served_message'):
            resp['choices'][0]['message'] = row['served_message']
        return resp
    return None


def teacher_calls(row, msa):
    """[(status, bytes)] of the teacher's chat calls on this request, in order."""
    g = row.get('teacher_guard')
    out = [(400, b'{"error": {"message": "This model\'s maximum context length is 1 tokens (replay)"}}')] \
        if row.get('teacher_max_tokens_dropped') else []
    if g is not None:
        oc = g.get('outcome') or ''
        if oc.startswith('http_'):
            return out + [(row.get('upstream_status') or int(oc[5:]), _err_bytes(row))]
        out += [(200, json.dumps(r).encode()) for r in row.get('teacher_resampled_replies') or []]
        if oc.startswith('resample_http_'):
            return out + [(int(oc.split('_')[2]), _err_bytes(row, 'teacher_resample_error'))]
        final = copy.deepcopy(row['response'])
        if oc in ('autofix', 'resampled_autofix') and row.get('original_teacher_reply'):
            final = row['original_teacher_reply']
        elif msa and oc in ('ok', 'resampled_ok') and row.get('served_message'):
            final['choices'][0]['message'] = row['served_message']
        return out + [(200, json.dumps(final).encode())]
    out += [(200, json.dumps(r).encode()) for r in row.get('repair_rejected_replies') or []]
    if row.get('upstream_status') not in (None, 200):
        return out + [(row['upstream_status'], _err_bytes(row))]
    if row.get('response'):
        return out + [(200, json.dumps(row['response']).encode())]
    return out


class Clock:
    def __init__(self):
        self.T = 0.0

    def time(self):
        return self.T


class Stub:
    """Router.post's stand-in: each request's upstream calls answered from its logged row."""

    def __init__(self, clock, msa):
        self.clock, self.msa = clock, msa
        self.calls, self.misses = [], 0

    def load(self, row, tag):
        self.row, self.tag = row, tag
        s = raw_student(row, self.msa)
        if row.get('owner') == 'student' and row.get('upstream_status') not in (None, 200):
            self.student = [(row['upstream_status'], _err_bytes(row))]
        else:
            self.student = [] if s is None else [(200, json.dumps(s).encode())]
        self.teacher = teacher_calls(row, self.msa) if row.get('owner') == 'teacher' else []
        st = row.get('student_latency_sec') if row.get('student_sent_body') else row.get('latency_sec')
        self.lat = dict(student=st or 0.0, teacher=(row.get('latency_sec') or 0.0) / max(1, len(self.teacher)))

    async def post(self, who, path, body, ep=None):
        self.calls.append(dict(tag=self.tag, seq=self.row['seq'], who=who, path=path,
                               body=sha(json.dumps(body, sort_keys=True, ensure_ascii=False))))
        if path == '/tokenize':
            if 'student_view_tokens' not in self.row:
                self.misses += 1
                return 200, b'{"count": 0}'
            n = self.row['student_view_tokens']
            if n is not None:
                return 200, json.dumps({'count': n}).encode()
            e = self.row.get('student_view_tokens_error') or 'HTTP 500: '
            st, _, txt = e.partition(': ')
            return int(st.split()[1]), txt.encode('utf-8', 'surrogatepass')
        q = self.student if who == 'student' else self.teacher
        self.clock.T += self.lat[who]
        if q:
            return q.pop(0)
        self.misses += 1                  # the recording never asked this server here: give back what harbor got
        if self.row.get('response'):
            return 200, json.dumps(self.row['response']).encode()
        return 502, b'{"error": {"message": "replay: nothing recorded"}}'


class Req:
    def __init__(self, body, sid, header):
        self._body, self.headers = body, {header: sid}

    async def json(self):
        return copy.deepcopy(self._body)


# ==== run ====================================================================================================
def load_router(path):
    d = os.path.dirname(os.path.abspath(path))
    sys.path.insert(0, d)
    # its own module name: the replay patches its clock and uuids, which must not reach another relay_router import
    spec = importlib.util.spec_from_file_location('relay_router_replayed', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['relay_router_replayed'] = mod
    spec.loader.exec_module(mod)
    return mod


def strip_flags(argv, flags):
    out, skip = [], False
    for x in argv:
        if skip:
            skip = False
            continue
        if x in flags:
            skip = x in VALUE_FLAGS
            continue
        out.append(x)
    return out


def local_argv(rec, out_dir, extra=()):
    argv, out = list(rec.argv), []
    for i, x in enumerate(argv):
        prev = argv[i - 1] if i else None
        if prev == '--log-dir':
            x = out_dir
        elif prev in ('--tasks', '--student-tokenizer', '--terminus-parser', '--teacher-url-file'):
            x = rec.local(x)
        elif prev == '--port':
            x = '0'
        out.append(x)
    return out + list(extra)


async def replay_group(rr, rec, out_dir, inject, trigger_cfg):
    os.makedirs(out_dir, exist_ok=True)
    argv = local_argv(rec, out_dir)
    if trigger_cfg:
        import router_triggers as rtg
        a = rr.parse_args(argv)
        parser_loaded = bool(a.repair_on_parse_error or a.teacher_format_guard) and a.harness != 'msa'
        note = a.verify_note_text or (rr.VERIFY_NOTE_MSA if a.harness == 'msa' else rr.VERIFY_NOTE
                                      if a.mode == 'relay' else rr.VERIFY_NOTE_OWN)
        import yaml
        p = os.path.join(out_dir, 'trigger_config.yaml')
        with open(p, 'w') as f:
            yaml.safe_dump(rtg.legacy_config(a, parser_loaded, note), f, sort_keys=False, width=1000)
        drop = {o for _, o in rtg.LEGACY_FLAGS} | ({o for _, o in rtg.RELAY_NOTE_FLAGS} if a.mode == 'relay' else set())
        argv = strip_flags(argv, drop) + ['--trigger-config', p]
    a = rr.parse_args(argv)
    msa = a.harness == 'msa'
    hint = a.teacher_hint_text
    notes = NOTES + ((a.verify_note_text,) if a.verify_note_text else ())
    clock = Clock()
    rr.time = types.SimpleNamespace(time=clock.time, strftime=time.strftime)
    counter = iter(range(1, 10 ** 9))
    rr.uuid = types.SimpleNamespace(uuid4=lambda: types.SimpleNamespace(hex=f'{next(counter):032x}'))
    router = rr.Router(a)
    stub = Stub(clock, msa)
    router.post = stub.post
    fid = collections.Counter()
    owners = []
    with open(os.path.join(out_dir, 'responses.jsonl'), 'w') as fr:
        for ep in rec.episodes:
            best = None
            for reasoning in (True, False):
                reqs = rebuild(rec, ep, msa, notes, hint, reasoning)
                hits = sum(msg_sha(b['messages']) == row['request_sha'] for row, b in reqs)
                if best is None or hits > best[0]:
                    best = (hits, reqs)
            hits, reqs = best
            fid['requests'] += len(reqs)
            fid['request_sha_match'] += hits
            for k, (row, body) in enumerate(reqs):
                for rep in range(2 if inject and (k + 1) % inject == 0 else 1):
                    clock.T = row['ts'] + rep
                    stub.load(row, f'{ep}:{row["seq"]}:{rep}')
                    resp = await router.h_chat(Req(body, row['sid'], rr.SESSION_HEADER))
                    fr.write(json.dumps(dict(episode=ep, seq=row['seq'], rep=rep, status=resp.status,
                                             body=resp.body.decode('utf-8', 'surrogatepass') if resp.body else None))
                             + '\n')
                owners.append((ep, row['seq'], row.get('owner')))
    with open(os.path.join(out_dir, 'calls.jsonl'), 'w') as f:
        for c in stub.calls:
            f.write(json.dumps(c) + '\n')
    # fidelity: who answered each request in the replay vs the recording (the replayed router's version may differ
    # from the recording's; this measures how much of the recorded routing the replay exercises)
    got = {}
    for line in open(os.path.join(out_dir, 'turns.jsonl')):
        r = json.loads(line)
        got.setdefault((r['sid'], r['seq']), r)
    sids = {ep: rec.rows[ep][0]['sid'] for ep in rec.episodes}
    same = sum(1 for ep, seq, o in owners if (got.get((sids[ep], seq)) or {}).get('owner') == o)
    fid.update(owner_as_recorded=same, stub_misses=stub.misses)
    json.dump(dict(fid, run=rec.run, arm=rec.arm, episodes=len(rec.episodes), argv=argv,
                   substituted=rec.substituted), open(
        os.path.join(out_dir, 'replay.json'), 'w'), indent=1)
    await asyncio.sleep(0)
    return dict(fid)


def cmd_run(args):
    rr = load_router(args.router)
    man = json.load(open(os.path.join(args.sample, 'manifest.json')))
    flags = json.load(open(os.path.join(args.sample, 'router_flags.json')))
    groups = collections.OrderedDict()
    for e in man:
        groups.setdefault((e['run'], e['arm']), []).append(e['episode'])
    total = collections.Counter()
    for (run, arm), eps in groups.items():
        if args.runs and run not in args.runs.split(','):
            continue
        rec = Recording(args.sample, run, arm, eps, flags[f'{run}/{arm}'])
        if args.trigger_config_from_flags and rr.parse_args(local_argv(rec, '/tmp')).mode != 'relay' \
                and not args.all_modes:
            continue
        out = os.path.join(args.out, f'{run}__{arm}')
        fid = asyncio.run(replay_group(rr, rec, out, args.inject_retries, args.trigger_config_from_flags))
        total.update({k: v for k, v in fid.items() if isinstance(v, int)})
        print(f'{run}/{arm}: {fid}', flush=True)
    print('TOTAL', dict(total))


# ==== diff ===================================================================================================
def _lines(p):
    return [json.loads(x) for x in open(p)] if os.path.exists(p) else []


def _raw(p):
    return open(p, 'rb').read().splitlines() if os.path.exists(p) else []


def _bodies(d):
    p = os.path.join(d, 'bodies')
    return {f: gzip.open(os.path.join(p, f)).read() for f in sorted(os.listdir(p))} if os.path.isdir(p) else {}


def _drop(d, keys):
    return {k: v for k, v in d.items() if k not in keys}


def cmd_diff(args):
    ignore = set(k for k in (args.ignore or '').split(',') if k)
    groups = sorted(set(os.listdir(args.a)) & set(os.listdir(args.b)))
    tot = collections.Counter()
    bad = []
    for g in groups:
        A, B = os.path.join(args.a, g), os.path.join(args.b, g)
        ta, tb = _lines(os.path.join(A, 'turns.jsonl')), _lines(os.path.join(B, 'turns.jsonl'))
        tot['requests'] += max(len(ta), len(tb))
        if len(ta) != len(tb):
            bad.append((g, 'turns', f'{len(ta)} vs {len(tb)} lines'))
        if not ignore:   # without --trigger-config the records must be byte-identical (key order included)
            tot['line_bytes_same'] += sum(u == v for u, v in zip(_raw(os.path.join(A, 'turns.jsonl')),
                                                                  _raw(os.path.join(B, 'turns.jsonl'))))
        same = _bodies(A) == _bodies(B)       # the upstream bodies, uncompressed (gzip headers carry a wall time)
        tot['bodies_groups_same'] += same
        if not same:
            bad.append((g, 'bodies', 'upstream bodies differ'))
        for x, y in zip(ta, tb):
            x, y = _drop(x, ignore), _drop(y, ignore)
            tot['owner_same'] += x.get('owner') == y.get('owner')
            tot['legacy_same'] += all(x.get(k) == y.get(k) for k in LEGACY_FIELDS)
            tot['line_same_minus_timing'] += _drop(x, TIMING) == _drop(y, TIMING)
            tot['line_same'] += x == y
            if x != y:
                keys = sorted(k for k in set(x) | set(y) if x.get(k) != y.get(k))
                bad.append((g, f"ep{x.get('episode')} seq{x.get('seq')}", f'keys differ: {keys}'))
        for name in ('responses.jsonl', 'calls.jsonl', 'events.jsonl'):
            la, lb = _lines(os.path.join(A, name)), _lines(os.path.join(B, name))
            if name == 'events.jsonl':
                la = [_drop(e, ('trigger_config',)) for e in la]
                lb = [_drop(e, ('trigger_config',)) for e in lb]
            same = la == lb
            tot[f'{name}_groups_same'] += same
            if not same:
                i = next((i for i, (u, v) in enumerate(zip(la, lb)) if u != v), min(len(la), len(lb)))
                bad.append((g, name, f'first difference at line {i} ({len(la)} vs {len(lb)} lines)'))
    tot['groups'] = len(groups)
    print(json.dumps(dict(tot), indent=1))
    for b in bad[:args.show]:
        print('MISMATCH', *b)
    print(f'{len(bad)} mismatches')
    return 1 if bad else 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--router', required=True)
    r.add_argument('--sample', required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--runs', default=None, help='comma list of run names (default: all)')
    r.add_argument('--inject-retries', type=int, default=0, help='send every Nth request twice')
    r.add_argument('--trigger-config-from-flags', action='store_true',
                   help='replace the intervention flags with --trigger-config <the YAML they stand for> (relay runs)')
    r.add_argument('--all-modes', action='store_true', help='with --trigger-config-from-flags: control and '
                                                            'student-only runs too')
    d = sub.add_parser('diff')
    d.add_argument('a')
    d.add_argument('b')
    d.add_argument('--ignore', default='')
    d.add_argument('--show', type=int, default=50)
    args = p.parse_args(argv)
    if args.cmd == 'run':
        cmd_run(args)
    else:
        sys.exit(cmd_diff(args))


if __name__ == '__main__':
    main()
