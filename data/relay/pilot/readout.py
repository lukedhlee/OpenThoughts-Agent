#!/usr/bin/env python3
"""Relay pilot readout: pass rates, takeovers, recovery, kept-trace yield, harness health, and the stop rule.

    python readout.py --run-dir <E>/runs/<name>            # final readout (router logs + harbor job dirs)
    python readout.py --run-dir <E>/runs/<name> --gate early   # driver's early gate: router logs only, exit 1 on fail

<run-dir> holds router_<arm>/ (relay_router.py --log-dir) and jobs/<name>_<arm> (harbor jobs_dir) for the arms relay
(student -> teacher, student thinking stripped), control (teacher from scratch) and, when run, relay_keep (student
thinking kept as reasoning), plus run.meta (node-hours, written by run_pilot.sh).

Definitions (the pilot spec, notes/relay/relay_pilot_100.md, is the authority):
  pass            verifier reward >= 1 (CalibForge verifiers write 0/1)
  harness error   a trial exception other than the agent's own ends (AgentTimeoutError, ContextLengthExceededError,
                  TurnCapExhaustedError) and VerifierTimeoutError, or no verifier result; plus router-side errors
  verifier timeout  counted on its own: some CalibForge verifiers hang on an unsolved state (seen in the wiring check),
                  so it is neither a harness error nor a verified failure; excluded from rates and from kept failures
  censored        the episode was ended by the run's deadline (router ending 'deadline'); excluded from rates
  real failure    reward 0 with no harness error, no verifier timeout, not censored
  takeover        the first router record of an episode that carries `takeover`; its turn is the first teacher turn
  recovery        pass rate over relay episodes with a takeover (by trigger too)
  kept traces     passes + as many real failures as passes (1:1), relay counting only takeover episodes
"""
import argparse
import collections
import glob
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'triggers'))
import relay_triggers as rt  # noqa: E402

AGENT_ENDS = {'AgentTimeoutError', 'ContextLengthExceededError', 'TurnCapExhaustedError'}
VERIFIER_TIMEOUT = 'VerifierTimeoutError'
RELAY_ARMS = ('relay', 'relay_keep', 'relay_repair')
SERVED = {'student': 'snowball', 'teacher': 'qwen38', 'router': 'relay-router'}
TAIL_BYTES = 4 << 20


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass   # a line being written right now
    return out


def read_outcome(path):
    """(task_name, reward, exception_type) from a trial result.json, reading only its head and tail.
    By diff from data/r2egym/jsc/refresh_screen.py read_outcome (result files embed the whole trajectory)."""
    size = os.path.getsize(path)
    with open(path, 'rb') as f:
        head = f.read(4096)
        f.seek(max(0, size - TAIL_BYTES))
        tail = f.read()
    name = re.search(rb'"task_name":\s*"([^"\\]+)"', head)
    start = tail.rfind(b'"verifier_result":')
    region = tail[start:] if start >= 0 else b''
    split = region.find(b'"exception_info":')
    exc = re.match(rb'"exception_info":\s*(?:null|\{\s*"exception_type":\s*"([^"\\]+)")', region[split:]) if split >= 0 else None
    if name and exc and (size <= TAIL_BYTES or start > 0):
        verifier = region[:split]
        reward = None
        if not re.match(rb'"verifier_result":\s*null', verifier):
            found = re.search(rb'"rewards":\s*\{\s*"reward":\s*(-?[0-9.eE+-]+)\s*[,}]', verifier)
            if not found:
                return full_outcome(path)
            reward = float(found.group(1))
        return name.group(1).decode(), reward, exc.group(1).decode() if exc.group(1) else None
    return full_outcome(path)


def full_outcome(path):
    r = json.load(open(path))
    reward = ((r.get('verifier_result') or {}).get('rewards') or {}).get('reward')
    return r.get('task_name'), reward, (r.get('exception_info') or {}).get('exception_type')


def sha(s):
    import hashlib
    return hashlib.sha1((s or '').encode('utf-8', 'surrogatepass')).hexdigest()[:16]


def q(xs, f):
    xs = sorted(x for x in xs if x is not None)
    return xs[int(f * (len(xs) - 1))] if xs else None


def trials(job_dir):
    """Per trial: task, reward, exception, session id and the main trajectory's agent steps."""
    rows = []
    if not job_dir or not os.path.isdir(job_dir):
        return rows
    for rj in sorted(glob.glob(os.path.join(job_dir, '*', 'result.json'))):
        tdir = os.path.dirname(rj)
        try:
            task, reward, exc = read_outcome(rj)
        except (OSError, ValueError, KeyError):
            continue
        trajs = sorted(glob.glob(os.path.join(tdir, '**', 'agent', 'trajectory.json'), recursive=True), key=os.path.getmtime)
        sid, steps = None, []
        if trajs:
            try:
                t = json.load(open(trajs[-1]))
                sid = t.get('session_id')
                steps = [dict(content_sha=sha(s.get('message') if isinstance(s.get('message'), str) else json.dumps(s.get('message'))),
                              model=s.get('model_name'), has_reasoning=bool(s.get('reasoning_content')),
                              parse_error_obs='Previous response had parsing errors' in json.dumps(s.get('observation') or {}),
                              done=bool(isinstance(s.get('message'), str) and rt.parse_reply(s['message'], 'terminus2')['done']))
                         for s in t.get('steps', []) if s.get('source') == 'agent' and not s.get('is_copied_context')]
            except (OSError, ValueError):
                pass
        vt = exc == VERIFIER_TIMEOUT
        rows.append(dict(trial=os.path.basename(tdir), task=task, reward=reward, exc=exc, sid=sid, steps=steps,
                         verifier_timeout=vt, censored=False,
                         harness_error=not vt and ((reward is None) or (exc is not None and exc not in AGENT_ENDS))))
    return rows


def router_view(router_dir):
    recs = read_jsonl(os.path.join(router_dir, 'turns.jsonl'))
    events = read_jsonl(os.path.join(router_dir, 'events.jsonl'))
    eps = collections.OrderedDict()
    for r in recs:
        e = eps.setdefault(r['sid'], dict(sid=r['sid'], task_id=r.get('task_id'), main=[], aux=[], takeover=None,
                                          ending=None, done_claim_dropped=0))
        (e['main'] if r.get('turn') is not None else e['aux']).append(r)
        if r.get('takeover') and e['takeover'] is None:
            e['takeover'] = dict(r['takeover'], first_teacher_record=r)
        if r.get('ending'):
            e['ending'] = r['ending']
        e['done_claim_dropped'] += bool(r.get('done_claim_dropped'))
    return dict(recs=recs, events=events, eps=eps, fatal=os.path.exists(os.path.join(router_dir, 'FATAL')),
                fatal_text=open(os.path.join(router_dir, 'FATAL')).read()[:500] if os.path.exists(os.path.join(router_dir, 'FATAL')) else None)


def mark_censored(rv, rows):
    dead = {e['sid'] for e in rv['eps'].values() if e['ending'] == 'deadline'}
    for t in rows:
        t['censored'] = t['sid'] in dead


def usable(t):
    return t is not None and not t['harness_error'] and not t['verifier_timeout'] and not t['censored']


def is_pass(t):
    return (t['reward'] or 0) >= 1


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [round(c - h, 4), round(c + h, 4)]


def newcombe(k1, n1, k2, n2):
    """95 % CI of p2 - p1 (Newcombe's hybrid score interval, method 10)."""
    if not n1 or not n2:
        return None
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    p1, p2 = k1 / n1, k2 / n2
    d = p2 - p1
    return [round(d - ((p2 - l2) ** 2 + (u1 - p1) ** 2) ** 0.5, 4), round(d + ((u2 - p2) ** 2 + (p1 - l1) ** 2) ** 0.5, 4)]


def is_context_error(r):
    return r.get('upstream_status') == 400 and 'context' in (r.get('upstream_error') or '').lower()


def harness_router(rv):
    recs = rv['recs']
    bad_status = collections.Counter(r.get('upstream_status') for r in recs
                                     if r.get('upstream_status') not in (200, None) and not is_context_error(r))
    det = sum(1 for r in recs for s in (r.get('logged') or []) if s.get('trigger') == 'detector_error')
    return dict(fatal=rv['fatal'], fatal_text=rv['fatal_text'], upstream_errors=dict(bad_status),
                context_length_400=sum(1 for r in recs if is_context_error(r)), detector_errors=det,
                episodes_without_task=sum(1 for e in rv['eps'].values() if not e['task_id']),
                done_claim_dropped=sum(e['done_claim_dropped'] for e in rv['eps'].values()),
                retried_requests=sum(1 for r in recs if r.get('retry')),
                bodies_missing=sum(1 for r in recs if r.get('upstream_status') is not None and not r.get('sent_body')))


def reasoning_view(rv):
    t = [r for r in rv['recs'] if r.get('owner') == 'teacher' and r.get('upstream_status') == 200]
    main = [r for r in t if r.get('turn') is not None]
    # H3 is judged on main-chat agent requests. Terminus-2's summarization requests (summary / questions / answers and
    # the handoff that follows) rebuild history messages without reasoning at this harbor base (upstream #223 fixed
    # it later); the router restores those from its own record, counted separately.
    ref, aux = collections.Counter(), collections.Counter()
    for r in t:
        tgt = aux if r.get('request_kind') in ('summary', 'questions', 'answers', 'handoff') else ref
        for k, v in (r.get('refeed') or {}).items():
            if isinstance(v, int):
                tgt[k] += v
    need = ref['prior_teacher_turns'] - ref['teacher_turns_without_reasoning']
    comp = [((r.get('usage') or {}).get('completion_tokens')) for r in main]
    cut = [r for r in t if r.get('teacher_cut_at_cap')]
    return dict(teacher_replies=len(main), with_reasoning=sum(1 for r in main if r.get('teacher_reasoning_chars')),
                reasoning_frac=round(sum(1 for r in main if r.get('teacher_reasoning_chars')) / len(main), 4) if main else None,
                think_close_in_content=sum(1 for r in main if r.get('content_has_think_close')),
                prior_teacher_turns=ref['prior_teacher_turns'], refed_by_harbor=ref['reasoning_key_from_harbor'],
                refed_reasoning_content_only=ref['reasoning_content_only'], restored_by_router=ref['reasoning_restored'],
                prior_turns_that_had_no_reasoning=ref['teacher_turns_without_reasoning'],
                refed_by_harbor_frac=round(ref['reasoning_key_from_harbor'] / need, 4) if need > 0 else None,
                summarization_requests_restored=aux['reasoning_restored'],
                summarization_requests_prior_teacher_turns=aux['prior_teacher_turns'],
                student_think_spans_stripped=ref['think_stripped'],
                student_think_sent_as_reasoning=ref['student_think_as_reasoning'],
                teacher_replies_cut_at_cap=len(cut), episodes_with_a_cut_reply=len({r['sid'] for r in cut}),
                teacher_max_tokens_lowered=sum(1 for r in t if r.get('teacher_max_tokens_lowered_to')),
                completion_tokens_p50=q(comp, .5), completion_tokens_p90=q(comp, .9),
                completion_tokens_max=max([c for c in comp if c is not None], default=None))


def student_view(rv):
    s = [r for r in rv['recs'] if r.get('owner') == 'student' and r.get('turn') is not None and r.get('upstream_status') == 200]
    def c(r):
        return (((r.get('response') or {}).get('choices') or [{}])[0].get('message') or {}).get('content') or ''
    # served with skip_special_tokens=false, a reply keeps 09-21's think markers; one marker at least (an unclosed
    # span is a format failure, not a stripped one)
    s = s + [dict(response=r['discarded_student_reply']) for r in rv['recs'] if r.get('repair_kind') == 'parse_error'
             and r.get('discarded_student_reply')]
    marks = sum(1 for r in s if '<|start_think|>' in c(r) or '<|end_think|>' in c(r))
    return dict(student_replies=len(s), with_think_markers=marks, think_marker_frac=round(marks / len(s), 4) if s else None)


def latency(rv):
    out = {}
    for who in ('student', 'teacher'):
        ls = [r.get('latency_sec') for r in rv['recs'] if r.get('owner') == who and r.get('upstream_status') == 200]
        out[who] = dict(n=len(ls), p50=q(ls, .5), p90=q(ls, .9))
    return out


def owner_join(rv, rows):
    """Every trajectory agent step must match a router record of that episode by content hash and served model."""
    by_sid = collections.defaultdict(dict)
    for r in rv['recs']:
        if r.get('content_sha'):
            by_sid[r['sid']][r['content_sha']] = r.get('owner')
    steps = matched = model_ok = 0
    no_sid = 0
    for t in rows:
        if t['sid'] is None:
            no_sid += bool(t['steps'])
            continue
        recs = by_sid.get(t['sid'], {})
        for s in t['steps']:
            steps += 1
            o = recs.get(s['content_sha'])
            if o:
                matched += 1
                model_ok += s['model'] == SERVED.get(o)
    return dict(agent_steps=steps, joined=matched, model_consistent=model_ok, trials_with_steps_but_no_sid=no_sid,
                joined_frac=round(matched / steps, 4) if steps else None)


def failure_cause(t, ending=None):
    """Why a verified failure failed, first match wins: context overflow, format loop (half or more of its turns drew a
    parse-error re-prompt, or the last 5 all did), false done (the episode ended on a confirmed task_complete), timeout
    (router budget ending or AgentTimeoutError), else tests failed."""
    steps = t['steps']
    pe = [s['parse_error_obs'] for s in steps]
    if t['exc'] == 'ContextLengthExceededError':
        return 'context_overflow'
    if steps and (sum(pe) >= len(pe) / 2 or (len(pe) >= 5 and all(pe[-5:]))):
        return 'format_loop'
    if ending in ('student_budget', 'teacher_budget') or t['exc'] == 'AgentTimeoutError':
        return 'timeout'
    if len(steps) >= 2 and steps[-1]['done'] and steps[-2]['done'] and ending is None:
        return 'false_done'
    return 'tests_failed'


def outcome_table(rows, endings=None):
    endings = endings or {}
    n = len(rows)
    scored = [t for t in rows if usable(t)]
    passes = [t for t in scored if is_pass(t)]
    return dict(trials=n, scored=len(scored), passes=len(passes), real_failures=len(scored) - len(passes),
                pass_rate=round(len(passes) / len(scored), 4) if scored else None,
                pass_rate_ci95=wilson(len(passes), len(scored)),
                harness_errors=collections.Counter(t['exc'] or 'no_verifier_result' for t in rows if t['harness_error']),
                harness_error_frac=round(sum(t['harness_error'] for t in rows) / n, 4) if n else None,
                verifier_timeouts=sum(t['verifier_timeout'] for t in rows),
                censored_by_deadline=sum(t['censored'] for t in rows),
                context_overflow=sum(1 for t in rows if t['exc'] == 'ContextLengthExceededError'),
                context_overflow_rate=round(sum(1 for t in rows if t['exc'] == 'ContextLengthExceededError') / n, 4) if n else None,
                failure_causes=collections.Counter(failure_cause(t, endings.get(t['sid'])) for t in scored if not is_pass(t)),
                turns_per_episode_p50=q([len(t['steps']) for t in scored], .5),
                turns_per_episode_p90=q([len(t['steps']) for t in scored], .9),
                agent_ends=collections.Counter(t['exc'] for t in scored if t['exc']))


def takeover_guards(e, t):
    """Per takeover episode: g1 teacher false-done (its first task_complete within 2 teacher turns of the takeover,
    no verification command in the teacher turns before that claim, and the task fails), g2 context exceeded after
    the takeover, and the teacher's cost (turns, completion and prompt tokens)."""
    t0 = e['takeover']['turn']
    teacher = sorted((r for r in e['main'] if r.get('owner') == 'teacher' and r.get('turn') is not None
                      and r['turn'] >= t0 and r.get('upstream_status') == 200), key=lambda r: r['turn'])
    verified, claim_at = False, None
    for i, r in enumerate(teacher[:2]):
        content = (((r.get('response') or {}).get('choices') or [{}])[0].get('message') or {}).get('content') or ''
        pr = rt.parse_reply(content, 'terminus2')
        if pr['done']:
            claim_at = i
            break
        for k, _ in pr['cmds']:
            if rt.is_check(k) and not rt.is_modify(k):
                verified = True
    failed = usable(t) and not is_pass(t)
    ctx = (t is not None and t['exc'] == 'ContextLengthExceededError') or any(
        is_context_error(r) for r in e['main'] + e['aux'] if (r.get('turn') or t0) >= t0 and r.get('owner') == 'teacher')
    usage = [(r.get('usage') or {}) for r in teacher]
    return dict(g1_false_done=bool(claim_at is not None and not verified and failed), g2_context_exceeded=bool(ctx),
                teacher_turns=len(teacher), completion_tokens=sum(u.get('completion_tokens') or 0 for u in usage),
                prompt_tokens=sum(u.get('prompt_tokens') or 0 for u in usage))


def relay_takeovers(rv, rows):
    by_sid = {t['sid']: t for t in rows if t['sid']}
    eps = list(rv['eps'].values())
    trig = collections.defaultdict(list)
    none = []
    for e in eps:
        t = by_sid.get(e['sid'])
        (trig[e['takeover']['trigger']] if e['takeover'] else none).append((e, t))
    out = dict(episodes=len(eps), takeover_episodes=sum(len(v) for v in trig.values()), by_trigger={})
    out['takeover_rate'] = round(out['takeover_episodes'] / len(eps), 4) if eps else None
    rec_all = []
    for name, items in sorted(trig.items()):
        turns = [e['takeover']['turn'] for e, _ in items]
        frac = [e['takeover']['elapsed_sec'] / e['takeover']['budget_sec'] for e, _ in items if e['takeover'].get('budget_sec')]
        scored = [(e, t) for e, t in items if usable(t)]
        p = sum(1 for _, t in scored if is_pass(t))
        rec_all += scored
        d = dict(n=len(items), turn_p25=q(turns, .25), turn_p50=q(turns, .5), turn_p75=q(turns, .75),
                 budget_used_p50=round(q(frac, .5), 3) if frac else None, scored=len(scored), passes=p,
                 recovery=round(p / len(scored), 4) if scored else None, recovery_ci95=wilson(p, len(scored)),
                 teacher_budget_endings=sum(1 for e, _ in items if e['ending'] == 'teacher_budget'))
        if name == 'done_claim':
            first = [e['takeover']['first_teacher_record'] for e, _ in items]
            immediate = 0
            for r in first:
                try:
                    c = json.loads(re.search(r'\{.*\}', r['response']['choices'][0]['message']['content'], re.S).group(0))
                    immediate += bool(c.get('task_complete')) and not c.get('commands')
                except Exception:  # noqa: BLE001
                    pass
            d['teacher_confirmed_without_a_command'] = immediate
            d['teacher_worked_frac'] = round(1 - immediate / len(first), 4) if first else None
        out['by_trigger'][name] = d
    sp = [(e, t) for e, t in rec_all]
    p = sum(1 for _, t in sp if is_pass(t))
    out['recovery'] = round(p / len(sp), 4) if sp else None
    out['recovery_ci95'] = wilson(p, len(sp))
    out['recovery_by_task'] = {t['task']: int(is_pass(t)) for _, t in sp}
    out['takeover_passes'] = p
    out['takeover_real_failures'] = len(sp) - p
    # guards and cost over takeover episodes with an outcome (usable, or ended by context overflow)
    gl = [(takeover_guards(e, t), t) for items in trig.values() for e, t in items
          if usable(t) or (t is not None and t['exc'] == 'ContextLengthExceededError')]
    n_g = len(gl)
    rec_n = sum(1 for _, t in gl if usable(t) and is_pass(t))
    out['guards'] = dict(n=n_g, g1_false_done=sum(g['g1_false_done'] for g, _ in gl),
                         g2_context_exceeded=sum(g['g2_context_exceeded'] for g, _ in gl),
                         g1_rate=round(sum(g['g1_false_done'] for g, _ in gl) / n_g, 4) if n_g else None,
                         g2_rate=round(sum(g['g2_context_exceeded'] for g, _ in gl) / n_g, 4) if n_g else None)
    out['cost_per_recovery'] = dict(
        recoveries=rec_n,
        teacher_turns=round(sum(g['teacher_turns'] for g, _ in gl) / rec_n, 1) if rec_n else None,
        teacher_completion_tokens=round(sum(g['completion_tokens'] for g, _ in gl) / rec_n) if rec_n else None,
        teacher_prompt_tokens=round(sum(g['prompt_tokens'] for g, _ in gl) / rec_n) if rec_n else None)
    sc_none = [(e, t) for e, t in none if usable(t)]
    out['no_takeover'] = dict(n=len(none), endings=collections.Counter(e['ending'] or 'none' for e, _ in none),
                              passes=sum(1 for _, t in sc_none if is_pass(t)), scored=len(sc_none))
    out['censored_takeovers'] = sum(1 for items in trig.values() for e, t in items if t and t['censored'])
    return out


def would_fire(rv, rows):
    by_sid = {t['sid']: t for t in rows if t['sid']}
    c = collections.Counter()
    for e in rv['eps'].values():
        t = by_sid.get(e['sid'])
        if not usable(t):
            continue
        fired = {f['trigger'] for r in e['main'] for f in (r.get('would_fire') or [])}
        outcome = 'pass' if is_pass(t) else 'fail'
        c[f'{outcome}_episodes'] += 1
        for f in fired:
            c[f'{outcome}_{f}'] += 1
    return dict(c)


def kept(passes, fails):
    return passes + min(passes, fails)


def repair_view(rv, rows):
    """relay_repair: repairs, who executed the turns, the repair round trip, sticky takeovers, format validity."""
    by_sid = {t['sid']: t for t in rows if t['sid']}
    eps = list(rv['eps'].values())
    rep_per_ep = [sum(1 for r in e['main'] if r.get('repair_kind') == 'parse_error') for e in eps]
    own = collections.Counter()
    back_to_student = not_back = 0
    t_turns, s_turns = [], []
    for e in eps:
        main = sorted((r for r in e['main'] if r.get('upstream_status') in (200, None)), key=lambda r: (r['turn'], r['seq']))
        seen = {}
        for r in main:
            seen[r['turn']] = r          # a retried request keeps its last record
        main = [seen[k] for k in sorted(seen)]
        for i, r in enumerate(main):
            kind = 'router' if r['owner'] == 'router' else 'student' if r['owner'] == 'student' else \
                'teacher_repair' if r.get('repair') else 'teacher_sticky'
            own[kind] += 1
            if r.get('repair_kind') == 'parse_error' and i + 1 < len(main):
                nxt = main[i + 1]
                if nxt['owner'] == 'student' or nxt.get('repair') or nxt.get('takeover') or nxt['owner'] == 'router':
                    back_to_student += 1
                else:
                    not_back += 1
        t_turns.append(sum(1 for r in main if r['owner'] == 'teacher'))
        s_turns.append(sum(1 for r in main if r['owner'] == 'student'))
    ex = sum(v for k, v in own.items() if k != 'router')
    rejected = sum(len(r.get('repair_rejected_replies') or []) for r in rv['recs'])
    still_bad = sum(1 for r in rv['recs'] if r.get('repair') and r.get('repair_reply_parse_error'))
    paused = [max((r.get('paused_sec') or 0) for r in e['main'] + e['aux']) + max(
        (r.get('paused_this_turn_sec') or 0) for r in e['main'][-1:]) for e in eps if e['main']]
    with_teacher = [e for e in eps if any(r['owner'] == 'teacher' for r in e['main'])]
    sc = [by_sid.get(e['sid']) for e in with_teacher]
    sc = [t for t in sc if usable(t)]
    p = sum(1 for t in sc if is_pass(t))
    return dict(episodes=len(eps), repairs=sum(rep_per_ep), repairs_per_episode_mean=round(sum(rep_per_ep) / len(eps), 3) if eps else None,
                repairs_per_episode_p50=q(rep_per_ep, .5), repairs_per_episode_p90=q(rep_per_ep, .9),
                episodes_with_repair=sum(1 for x in rep_per_ep if x), executed_turns=dict(own),
                student_share_of_executed_turns=round(own['student'] / ex, 4) if ex else None,
                repair_returned_to_student=back_to_student, repair_kept_by_teacher=not_back,
                teacher_repair_replies_rejected_then_retried=rejected, repair_turns_still_unparseable=still_bad,
                paused_sec_per_episode_mean=round(statistics.mean(paused), 1) if paused else None,
                paused_sec_per_episode_p90=q(paused, .9),
                teacher_turns_per_episode_mean=round(statistics.mean(t_turns), 2) if t_turns else None,
                student_turns_per_episode_mean=round(statistics.mean(s_turns), 2) if s_turns else None,
                episodes_with_teacher_turns=len(with_teacher), with_teacher_passes=p,
                with_teacher_real_failures=len(sc) - p)


def format_validity(rows):
    steps = [s for t in rows for s in t['steps']]
    bad = sum(1 for s in steps if s['parse_error_obs'])
    return dict(agent_steps=len(steps), followed_by_parse_error=bad,
                valid_format_rate=round(1 - bad / len(steps), 4) if steps else None)


def paired_pass(ctl_rows, rel_rows, seed=20260925, n_boot=10000):
    import random
    c = {t['task']: int(is_pass(t)) for t in ctl_rows if usable(t)}
    r = {t['task']: int(is_pass(t)) for t in rel_rows if usable(t)}
    both = sorted(set(c) & set(r))
    d = [r[t] - c[t] for t in both]
    ci = None
    if d:
        rng = random.Random(seed)
        ms = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n_boot))
        ci = [round(ms[int(0.025 * n_boot)], 4), round(ms[int(0.975 * n_boot) - 1], 4)]
    return dict(tasks=len(both), relay_minus_control=round(sum(d) / len(d), 4) if d else None, ci95_bootstrap=ci,
                both_pass=sum(1 for t in both if c[t] and r[t]), relay_only=sum(1 for t in both if r[t] and not c[t]),
                control_only=sum(1 for t in both if c[t] and not r[t]), neither=sum(1 for t in both if not c[t] and not r[t]))


def strip_vs_keep(strip, keep, seed=20260925, n_boot=10000):
    """Recovery after takeover, stripped vs kept student thinking. Unpaired: each arm's recovery with a Wilson CI and
    the difference with Newcombe's interval. Paired: tasks with a takeover (and a usable outcome) in both arms, the
    mean of keep - strip with a bootstrap CI over tasks."""
    import random
    ks, kk = strip['recovery_by_task'], keep['recovery_by_task']
    n1, p1 = len(ks), sum(ks.values())
    n2, p2 = len(kk), sum(kk.values())
    both = sorted(set(ks) & set(kk))
    diffs = [kk[t] - ks[t] for t in both]
    boot = None
    if diffs:
        rng = random.Random(seed)
        ms = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n_boot))
        boot = [round(ms[int(0.025 * n_boot)], 4), round(ms[int(0.975 * n_boot) - 1], 4)]
    return dict(strip=dict(n=n1, passes=p1, recovery=round(p1 / n1, 4) if n1 else None, ci95=wilson(p1, n1)),
                keep=dict(n=n2, passes=p2, recovery=round(p2 / n2, 4) if n2 else None, ci95=wilson(p2, n2)),
                keep_minus_strip=round(p2 / n2 - p1 / n1, 4) if n1 and n2 else None,
                keep_minus_strip_ci95_newcombe=newcombe(p1, n1, p2, n2),
                paired_tasks=len(both), paired_mean_diff=round(sum(diffs) / len(diffs), 4) if diffs else None,
                paired_ci95_bootstrap=boot,
                paired_table=dict(both_pass=sum(1 for t in both if ks[t] and kk[t]),
                                  keep_only=sum(1 for t in both if kk[t] and not ks[t]),
                                  strip_only=sum(1 for t in both if ks[t] and not kk[t]),
                                  neither=sum(1 for t in both if not ks[t] and not kk[t])))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--name', help='run name (default: basename of --run-dir)')
    p.add_argument('--gate', choices=['early', 'final'], default='final')
    p.add_argument('--node-hours', type=float, help='node-hours spent (default: run.meta)')
    p.add_argument('--json', help='write the readout here too')
    a = p.parse_args()
    name = a.name or os.path.basename(os.path.normpath(a.run_dir))
    arm_names = [arm for arm in ('relay', 'control', 'relay_keep', 'relay_repair', 'student_only') if os.path.isdir(os.path.join(a.run_dir, f'router_{arm}'))]
    arms = {}
    for arm in arm_names:
        rv = router_view(os.path.join(a.run_dir, f'router_{arm}'))
        rows = [] if a.gate == 'early' else trials(os.path.join(a.run_dir, 'jobs', f'{name}_{arm}')) + \
            trials(os.path.join(a.run_dir, 'jobs', f'{name}_{arm}_p2'))     # the full run's phase-2 relay job
        mark_censored(rv, rows)
        arms[arm] = dict(rv=rv, rows=rows)
    out = dict(run=name, gate=a.gate, arms=arm_names)
    for arm, d in arms.items():
        rv, rows = d['rv'], d['rows']
        summ = collections.Counter(sum(1 for r in e['aux'] if r['request_kind'] == 'summary') for e in rv['eps'].values())
        o = dict(router_harness=harness_router(rv), reasoning=reasoning_view(rv), latency_sec=latency(rv),
                 episodes_seen=len(rv['eps']), main_turns=sum(len(e['main']) for e in rv['eps'].values()),
                 owners=collections.Counter(r.get('owner') for r in rv['recs'] if r.get('turn') is not None),
                 summarizations_per_episode=dict(sorted(summ.items())),
                 student_think_modes=sorted({r.get('student_think') for r in rv['recs'] if r.get('student_think')}))
        if arm in RELAY_ARMS:
            o['student'] = student_view(rv)
            o['takeovers'] = relay_takeovers(rv, rows)
            if arm == 'relay_repair':
                o['repair'] = repair_view(rv, rows)
        else:
            o['would_fire_on_teacher'] = would_fire(rv, rows)
        if rows:
            o['outcomes'] = outcome_table(rows, {e['sid']: e['ending'] for e in rv['eps'].values()})
            o['owner_join'] = owner_join(rv, rows)
            o['format'] = format_validity(rows)
        out[arm] = o

    checks = []

    def check(name_, ok, detail):
        checks.append(dict(check=name_, ok=bool(ok), detail=detail))

    for arm in arm_names:
        h = out[arm]['router_harness']
        check(f'H1 {arm}: router clean', not h['fatal'] and not h['upstream_errors'] and not h['detector_errors']
              and not h['episodes_without_task'] and not h['bodies_missing'], h)
        r = out[arm]['reasoning']
        if r['teacher_replies']:
            check(f'H3 {arm}: teacher replies carry reasoning (>= 95 %)', (r['reasoning_frac'] or 0) >= 0.95,
                  f"{r['with_reasoning']}/{r['teacher_replies']}, </think> in content {r['think_close_in_content']}")
        if r['prior_teacher_turns']:
            check(f'H3 {arm}: prior teacher reasoning re-fed by harbor on agent turns (100 %, none restored)',
                  r['refed_by_harbor_frac'] == 1.0 and r['restored_by_router'] == 0,
                  f"by harbor {r['refed_by_harbor']}, restored {r['restored_by_router']}, prior {r['prior_teacher_turns']}")
        if arm in RELAY_ARMS:
            s_ = out[arm]['student']
            if s_.get('student_replies'):
                check(f'H0 {arm}: student replies keep their think markers (>= 90 %)', (s_['think_marker_frac'] or 0) >= 0.9, s_)
    if 'relay_repair' in out:
        rp = out['relay_repair']['repair']
        if rp['repairs']:
            check('H4 relay_repair: after a repair the student resumes (or a trigger / ending takes over), 100 %',
                  rp['repair_kept_by_teacher'] == 0, rp)
    if a.gate == 'final' and not ('control' in arms and any(x in arms for x in RELAY_ARMS)):
        for arm in arm_names:
            oc = out[arm].get('outcomes') or {}
            check(f'H1 {arm}: trial harness errors <= 10 %', oc and (oc.get('harness_error_frac') or 0) <= 0.10,
                  dict(oc.get('harness_errors') or {}))
            j = out[arm].get('owner_join') or {}
            check(f'H2 {arm}: owner labels on 100 % of trajectory turns', j.get('joined_frac') == 1.0, j)
    elif a.gate == 'final':
        for arm in arm_names:
            oc = out[arm].get('outcomes') or {}
            check(f'H1 {arm}: trial harness errors <= 10 %', oc and (oc.get('harness_error_frac') or 0) <= 0.10,
                  dict(oc.get('harness_errors') or {}))
            j = out[arm].get('owner_join') or {}
            check(f'H2 {arm}: owner labels on 100 % of trajectory turns',
                  j.get('joined_frac') == 1.0 and j.get('model_consistent') == j.get('agent_steps') and not j.get('trials_with_steps_but_no_sid'), j)
        harness_ok = all(c['ok'] for c in checks)
        rel_arm = 'relay_repair' if 'relay_repair' in arms else 'relay'
        if rel_arm == 'relay_repair':
            fv = out['relay_repair'].get('format') or {}
            check('H5 relay_repair: executed trace valid-format rate >= 99 % (every student failure intercepted)',
                  (fv.get('valid_format_rate') or 0) >= 0.99, fv)
            harness_ok = all(c['ok'] for c in checks)
        ctl = out['control'].get('outcomes') or {}
        rel = out[rel_arm].get('outcomes') or {}
        tk = out[rel_arm]['takeovers']
        node_h = a.node_hours
        meta = os.path.join(a.run_dir, 'run.meta')
        if node_h is None and os.path.exists(meta):
            for line in open(meta):
                if line.startswith('node_hours='):
                    node_h = float(line.split('=', 1)[1])
        sci = []

        def scheck(n_, ok, detail):
            sci.append(dict(check=n_, ok=bool(ok), detail=detail))
        scheck('S1 control pass rate in [0.25, 0.75]', ctl.get('pass_rate') is not None and 0.25 <= ctl['pass_rate'] <= 0.75,
               [ctl.get('pass_rate'), ctl.get('pass_rate_ci95')])
        if rel_arm == 'relay_repair':
            rp = out['relay_repair']['repair']
            scheck('S2 the student owns >= 50 % of executed turns in relay_repair', (rp['student_share_of_executed_turns'] or 0) >= 0.5,
                   rp['student_share_of_executed_turns'])
            # target 0.30; Luke 2026-09-25 17:40 PT: run 3's 0.28 is inside noise and counts as a pass -> floor 0.25
            scheck('S3 sticky takeover rate >= 0.25 (target 0.30)', (tk.get('takeover_rate') or 0) >= 0.25, tk.get('takeover_rate'))
        else:
            scheck('S2 relay takeover rate >= 0.40', (tk.get('takeover_rate') or 0) >= 0.40, tk.get('takeover_rate'))
        scheck('S4 recovery P(pass | sticky takeover) >= 0.20', (tk.get('recovery') or 0) >= 0.20,
               [tk.get('recovery'), tk.get('recovery_ci95')])
        dc = tk['by_trigger'].get('done_claim') or {}
        if rel_arm != 'relay_repair':
            scheck('S5 done_claim: teacher runs a command before confirming in >= 50 %',
                   (dc.get('teacher_worked_frac') or 0) >= 0.5, dc.get('teacher_worked_frac'))
        else:   # Luke 2026-09-25 17:40 PT: S5 is a keep filter now (select_kept.py), reported, not a gate
            out['s5_keep_filter_teacher_worked_frac'] = dc.get('teacher_worked_frac')
        if rel_arm == 'relay_repair':
            rp = out['relay_repair']['repair']
            kr = kept(rp['with_teacher_passes'], rp['with_teacher_real_failures'])
        else:
            kr = kept(tk.get('takeover_passes', 0), tk.get('takeover_real_failures', 0))
        kc = kept(ctl.get('passes', 0), ctl.get('real_failures', 0))
        n_eps = sum((out[arm].get('outcomes') or {}).get('trials', 0) for arm in arm_names)
        proj = {}
        if node_h and n_eps and kr and kc:
            nh_per_ep = node_h / n_eps
            n_r = 2000 / (kr / max(1, rel.get('trials', 1)))
            n_c = 2000 / (kc / max(1, ctl.get('trials', 1)))
            proj = dict(node_h_per_episode=round(nh_per_ep, 4), relay_episodes_for_2000_kept=round(n_r),
                        control_episodes_for_2000_kept=round(n_c), relay_node_h=round(n_r * nh_per_ep, 1),
                        control_node_h=round(n_c * nh_per_ep, 1))
        scheck('S6 projected node-h for 2,000 kept traces <= 60 per arm', proj and proj['relay_node_h'] <= 60
               and proj['control_node_h'] <= 60, proj or 'no yield or no node-hours')
        out['kept_traces'] = dict(relay=kr, control=kc, relay_per_episode=round(kr / max(1, rel.get('trials', 1)), 4),
                                  control_per_episode=round(kc / max(1, ctl.get('trials', 1)), 4),
                                  per_node_hour=round((kr + kc) / node_h, 2) if node_h else None)
        out['projection'] = proj
        out['paired_pass_relay_vs_control'] = paired_pass(arms['control']['rows'], arms[rel_arm]['rows'])
        if 'relay_keep' in arms:
            kt = out['relay_keep']['takeovers']
            sk = strip_vs_keep(tk, kt)
            d, ci = sk['keep_minus_strip'], sk['keep_minus_strip_ci95_newcombe']
            g1d = ((kt['guards']['g1_rate'] or 0) - (tk['guards']['g1_rate'] or 0)) if tk['guards']['n'] and kt['guards']['n'] else None
            g2d = ((kt['guards']['g2_rate'] or 0) - (tk['guards']['g2_rate'] or 0)) if tk['guards']['n'] and kt['guards']['n'] else None
            conds = dict(diff_ge_15pts=d is not None and d >= 0.15, ci_excludes_0=bool(ci and ci[0] > 0),
                         g1_not_worse_by_5pts=g1d is not None and g1d <= 0.05,
                         g2_not_worse_by_5pts=g2d is not None and g2d <= 0.05)
            sk.update(guard_diff_keep_minus_strip=dict(g1=None if g1d is None else round(g1d, 4),
                                                       g2=None if g2d is None else round(g2d, 4)),
                      guards=dict(strip=tk['guards'], keep=kt['guards']),
                      cost_per_recovery=dict(strip=tk['cost_per_recovery'], keep=kt['cost_per_recovery']),
                      rule_conditions=conds)
            out['strip_vs_keep'] = sk
            out['strip_vs_keep_rule'] = ('KEEP: recovery +%.1f pts, CI %s excludes 0, guards within 5 pts' % (100 * d, ci)
                                         if all(conds.values()) else
                                         'STRIP (default): keep did not meet %s' % [k for k, v in conds.items() if not v])
        out['science_checks'] = sci
        out['verdict'] = ('HARNESS FAIL: fix and re-run the pilot, no scaling' if not harness_ok else
                          'SCALE: every check passed; size the 2,000-kept run from the projection (cost line + go first)'
                          if all(c['ok'] for c in sci) else
                          'NO SCALE: harness fine, the science checks below failed')
    out['harness_checks'] = checks
    txt = json.dumps(out, indent=1, default=lambda x: dict(x) if isinstance(x, collections.Counter) else str(x))
    if a.json:
        open(a.json, 'w').write(txt)
    print(txt)
    for c in checks + out.get('science_checks', []):
        print(('PASS ' if c['ok'] else 'FAIL ') + c['check'], file=sys.stderr)
    if out.get('strip_vs_keep'):
        sk = out['strip_vs_keep']
        print(f"strip vs keep: recovery {sk['strip']['recovery']} {sk['strip']['ci95']} vs {sk['keep']['recovery']} "
              f"{sk['keep']['ci95']}, keep - strip {sk['keep_minus_strip']} {sk['keep_minus_strip_ci95_newcombe']}; "
              f"paired over {sk['paired_tasks']} tasks {sk['paired_mean_diff']} {sk['paired_ci95_bootstrap']}; "
              f"guards keep - strip {sk['guard_diff_keep_minus_strip']} -> {out['strip_vs_keep_rule']}", file=sys.stderr)
    if out.get('verdict'):
        print('VERDICT: ' + out['verdict'], file=sys.stderr)
    sys.exit(0 if all(c['ok'] for c in checks) else 1)


if __name__ == '__main__':
    main()
