"""CPU tests for replay.py: tiny synthetic router logs (Terminus-2 and mini-swe-agent episodes written the way
relay_router.py writes them), replayed against triggers whose fires are known by construction. No network, no sandbox.

    python -m pytest -q data/relay/triggers/tests/test_replay.py
"""
import asyncio
import gzip
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import replay as rp  # noqa: E402
import interventions as iv  # noqa: E402   (replay put router/ on sys.path)

CONFIRM = ('Current terminal state:\nNew Terminal Output:\n\nroot@box:/app# \n\nAre you sure you want to mark the task '
           "as complete? This will trigger your solution to be graded and you won't be able to make any further "
           'corrections. If so, include "task_complete": true in your JSON response again.')
FORMAT_ERROR = ('Tool call error:\n\n<error>\nNo tool calls found in the response. Every response MUST include at '
                'least one tool call.\n</error>\n\nHere is general guidance on how to submit correct toolcalls.')
SUBMIT = 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'
HINT = 'Work economically.'
LOOP = ('loop', 'exact repeat x3: python run.py')
CLAIM = ('done_claim', 'task_complete: true')


def sha(s):
    return hashlib.sha1((s or '').encode('utf-8', 'surrogatepass')).hexdigest()[:16]


def t2(*cmds, done=False):
    """A Terminus-2 reply in 09-21's style: a think span, then the JSON."""
    return '<|start_think|>thinking<|end_think|>' + json.dumps(dict(
        analysis='a', plan='p', commands=[dict(keystrokes=c + '\n', duration=0.1) for c in cmds], task_complete=done))


def screen(cmd, out):
    return f'New Terminal Output:\n\nroot@box:/app# {cmd}\n{out}\nroot@box:/app# '


def resp(msg):
    return {'id': 'x', 'choices': [{'index': 0, 'finish_reason': 'stop', 'message': msg}], 'model': 'snowball'}


class Log:
    """One router log dir as relay_router.py writes it: events.jsonl (its start event), turns.jsonl, bodies/."""

    def __init__(self, root, run, arm, mode, harness='terminus2', argv=(), **start):
        self.root, self.run, self.arm, self.mode, self.harness = root, run, arm, mode, harness
        self.dir = root / run / f'router_{arm}'
        (self.dir / 'bodies').mkdir(parents=True)
        ev = dict(ts=0.0, event='start', arm=arm, mode=mode, harness=harness, takeover=['done_claim', 'loop',
                  'no_progress_wait'], verify_note=False, repair_on_parse_error=False, teacher_format_guard=False,
                  argv=['relay_router.py', '--mode', mode, '--arm', arm, '--harness', harness, *argv])
        ev.update(start)
        (self.dir / 'events.jsonl').write_text(json.dumps(ev) + '\n')
        self.manifest = []

    def kind(self, messages):
        if self.harness == 'msa':
            return 'main' if any(m['role'] == 'assistant' for m in messages) else 'initial'
        if 'Are you sure you want to mark the task as complete?' in messages[-1]['content']:
            return 'confirm'
        return 'initial' if len(messages) == 1 else 'main'

    def request(self, ep, sid, seq, turn, messages, owner='student', response=None, bodies=None, **extra):
        """One turns.jsonl line, and the bodies sent upstream (default: harbor's messages to the answering model)."""
        rec = dict(ts=float(seq), elapsed_sec=float(seq), arm=self.arm, episode=ep, sid=sid, task_id='task-' + sid,
                   request_kind=self.kind(messages), n_messages=len(messages),
                   request_sha=sha(json.dumps(messages, sort_keys=True, ensure_ascii=False)), seq=seq,
                   student_clock_sec=10.0 * seq, turn=turn, retry=False, owner=owner, upstream_status=200,
                   latency_sec=1.0, usage={'prompt_tokens': 100 + seq})
        if self.harness == 'msa':
            rec['harness'] = 'msa'
        for who, msgs in (bodies or {owner: messages}).items():
            with gzip.open(self.dir / 'bodies' / f'ep{ep:05d}.jsonl.gz', 'at') as f:
                f.write(json.dumps(dict(seq=seq, who=who, body=dict(model=who, messages=msgs))) + '\n')
            key = 'student_sent_body' if who == 'student' and owner != 'student' else 'sent_body'
            rec[key] = f'bodies/ep{ep:05d}.jsonl.gz#{seq}'
        if response is not None:
            rec['response'] = resp(response)
        rec.update(extra)
        with open(self.dir / 'turns.jsonl', 'a') as f:
            f.write(json.dumps(rec) + '\n')

    def add(self, ep, sid, group, outcome):
        self.manifest.append(dict(group=group, run=self.run, arm=self.arm, episode=ep, sid=sid, task='task-' + sid,
                                  outcome=outcome))


def write_manifest(root, *logs):
    p = root / 'manifest.json'
    p.write_text(json.dumps([r for lg in logs for r in lg.manifest]))
    return str(p)


def t2_episode(log, ep, sid, steps, outcome, group='student_only_t2'):
    """steps: [(reply, observation or None at the end, would_fire)]; a student-alone Terminus-2 episode."""
    msgs = [{'role': 'user', 'content': f'You are an AI assistant. Task {sid}: fix run.py'}]
    for i, (reply, obs, wf) in enumerate(steps, 1):
        log.request(ep, sid, i, i, list(msgs), response={'role': 'assistant', 'content': reply},
                    would_fire=[dict(trigger=t, reason=r) for t, r in wf])
        msgs += [{'role': 'assistant', 'content': reply}, {'role': 'user', 'content': obs}]
    log.add(ep, sid, group, outcome)


LOOP_STEPS = [
    (t2('ls -la'), screen('ls -la', 'run.py'), []),
    (t2('cat run.py'), screen('cat run.py', 'print(1)'), []),
    (t2('python run.py'), screen('python run.py', 'Traceback (most recent call last):\nValueError: bad'), []),
    (t2('python run.py'), screen('python run.py', 'Traceback (most recent call last):\nValueError: bad'), []),
    (t2('python run.py'), screen('python run.py', 'Traceback (most recent call last):\nValueError: bad'), []),
    (t2(done=True), CONFIRM, [LOOP, CLAIM]),        # the loop shows in this request; the claim in its reply
    (t2(done=True), None, []),                      # the confirmation: the scanner fires on the first claim only
]
PASS_STEPS = [
    (t2('ls -la'), screen('ls -la', 'run.py'), []),
    (t2('sed -i s/1/2/ run.py'), screen('sed -i s/1/2/ run.py', ''), []),
    (t2(done=True), CONFIRM, [CLAIM]),
    (t2(done=True), None, []),
]


def call(seq, cmd):
    return {'id': f'call_{seq}_0', 'type': 'function',
            'function': {'name': 'bash', 'arguments': json.dumps({'command': cmd})}}


def msa_reply(seq, cmd):
    """The parsed message the router returns to harbor (msa_tool.message): prose, the reasoning, the tool call."""
    return {'role': 'assistant', 'content': 'next step', 'reasoning_content': 'think', 'reasoning': 'think',
            'tool_calls': [call(seq, cmd)]}


def tool(seq, **result):
    return {'role': 'tool', 'content': json.dumps(result, indent=2), 'tool_call_id': f'call_{seq}_0', 'name': 'bash'}


def msa_episode(log, ep, sid, outcome, group='student_only_msa'):
    """A mini-swe-agent episode: ok, failing command, a reply with no tool call (harbor re-prompts, the turn repeats),
    a timed-out command, elided output, then the submit."""
    msgs = [{'role': 'system', 'content': 'You are a helpful assistant that can interact with a computer.'},
            {'role': 'user', 'content': f'Please solve this issue: task {sid}'}]
    plan = [('ls', dict(returncode=0, output='a.py\n')),
            ('python a.py', dict(returncode=1, output='Traceback (most recent call last):\nValueError: bad\n')),
            (None, None),
            ('python a.py', dict(returncode=-1, output='', exception_info="An error occurred while executing the "
                                 "command: Command 'python a.py' timed out after 30 seconds")),
            ('cat big.log', dict(returncode=0, output_head='HEAD', output_tail='TAIL', elided_chars=5000,
                                 warning='Output too long.')),
            (SUBMIT, None)]
    turns = [1, 2, 3, 3, 4, 5]          # the router's count: the re-prompt after a reply with no tool call repeats it
    for seq, ((cmd, result), turn) in enumerate(zip(plan, turns), 1):
        if cmd is None:     # no tool call: the router logs the parse error and no would_fire; harbor adds no turn
            raw = {'role': 'assistant', 'content': 'I will think about it more.'}
            log.request(ep, sid, seq, turn, list(msgs), response=raw, served_message=raw,
                        student_parse_error='No tool calls found in the response. Every response MUST include at '
                                            'least one tool call.')
            msgs.append({'role': 'user', 'content': FORMAT_ERROR})
            continue
        m = msa_reply(seq, cmd)
        wf = [CLAIM] if cmd == SUBMIT else []
        log.request(ep, sid, seq, turn, list(msgs), response=m, served_message=dict(m, content='<tool_call>...'),
                    would_fire=[dict(trigger=t, reason=r) for t, r in wf])
        msgs.append(m)
        if result is not None:
            msgs.append(tool(seq, **result))
    log.add(ep, sid, group, outcome)


def episodes(manifest, **kw):
    return list(rp.load_episodes([manifest], **kw))


def run(coro):
    return asyncio.run(coro)


# ==== Terminus-2 =============================================================================================
def test_t2_events_are_causal_in_order_and_normalized(tmp_path):
    log = Log(tmp_path, 'r1', 'student_only', 'student')
    t2_episode(log, 1, 's-loop', LOOP_STEPS, 'fail')
    ep, = episodes(write_manifest(tmp_path, log))
    steps = []
    for s in rp.iter_events(ep):             # causal: the history and the scanner end at this request
        assert len(s.ev.messages) == 2 * s.ev.turn - 1 and s.ev.scanner.turn == s.ev.turn - 1
        steps.append(s)
    assert [(s.ev.phase, s.ev.turn) for s in steps] == [(p, t) for t in range(1, 8) for p in ('request', 'reply')]
    assert steps[0].ev.last_result is None
    r2 = steps[2].ev.last_result             # turn 2's request: turn 1's action and its new output
    assert (r2.command, r2.exit_code, r2.timed_out, r2.truncated) == ('ls -la', None, False, False)
    assert 'run.py' in r2.output and 'root@box' in r2.output.splitlines()[0]
    assert 'Traceback' in steps[8].ev.last_result.output
    assert steps[12].ev.extra['request_kind'] == 'confirm'
    assert steps[1].ev.reply.text == LOOP_STEPS[0][0] and steps[1].ev.reply.action['cmds'][0][0] == 'ls -la\n'
    assert run(steps[4].ev.view_tokens()) == 103          # the student's own prompt_tokens for seq 3
    assert steps[4].ev.clock == iv.Clock(30.0, None) and steps[5].ev.clock == iv.Clock(31.0, None)


def test_t2_builtin_check_reproduces_the_logged_would_fire(tmp_path):
    log = Log(tmp_path, 'r1', 'student_only', 'student')
    t2_episode(log, 1, 's-loop', LOOP_STEPS, 'fail')
    t2_episode(log, 2, 's-pass', PASS_STEPS, 'pass')
    out = tmp_path / 'out'
    assert rp.main(['--runs', write_manifest(tmp_path, log), '--builtin-check', '--out', str(out)]) == 0
    res = json.loads((out / 'builtin_check.json').read_text())
    c = res['classes']['student_alone']
    assert c['requests'] == c['matched'] == 11 and c['logged_fires'] == c['replayed_fires'] == 3
    assert res['mismatches'] == [] and res['builtin_done_claim_extra_fires'] == {'student_alone': 2}


def test_a_wrong_log_is_a_mismatch(tmp_path):
    log = Log(tmp_path, 'r1', 'student_only', 'student')
    steps = list(LOOP_STEPS)
    steps[5] = (steps[5][0], steps[5][1], [CLAIM])                  # the router "forgot" the loop
    t2_episode(log, 1, 's-loop', steps, 'fail')
    out = tmp_path / 'out'
    assert rp.main(['--runs', write_manifest(tmp_path, log), '--builtin-check', '--out', str(out)]) == 1
    mm, = json.loads((out / 'builtin_check.json').read_text())['mismatches']
    assert mm['turn'] == 6 and [tuple(x) for x in mm['replayed']] == sorted([LOOP, CLAIM])


def test_counterfactual_cut_and_first_actionable(tmp_path):
    log = Log(tmp_path, 'r1', 'student_only', 'student')
    t2_episode(log, 1, 's-loop', LOOP_STEPS, 'fail')
    ep, = episodes(write_manifest(tmp_path, log))
    cfg = {'triggers': [{'use': 'scripted', 'params': {'turns': [3]}},
                        {'use': 'loop'},
                        {'use': 'done_claim', 'params': {'confirm_kind': 'confirm'}}]}
    *fires, summ = rp.replay_episode(ep, cfg)
    got = [(f['trigger'], f['phase'], f['turn'], f['valid']) for f in fires]
    assert got == [('scripted', 'request', 3, True), ('loop', 'request', 6, False), ('done_claim', 'reply', 6, False),
                   ('done_claim', 'reply', 7, False)]
    pol = summ['policy']
    assert (pol['trigger'], pol['decided_turn'], pol['start_turn'], pol['start_phase']) == ('scripted', 3, 3, 'request')
    fa = summ['first_actionable']
    assert (fa['loop']['decided_turn'], fa['loop']['start_turn']) == (6, 6)
    # a reply-time next: decided on the claim, it starts at Terminus-2's confirmation request
    assert (fa['done_claim']['decided_turn'], fa['done_claim']['start_turn'], fa['done_claim']['start_phase']) == \
        (6, 7, 'request')
    assert summ['first_fire']['loop'] == dict(turn=6, phase='request', reason=LOOP[1], score=None, valid=False)
    assert (summ['end_turn'], summ['stop'], summ['outcome']) == (7, 'end', 'fail')


# ==== mini-swe-agent =========================================================================================
def test_msa_events_results_and_action_error(tmp_path):
    log = Log(tmp_path, 'r2', 'student_only', 'student', harness='msa')
    msa_episode(log, 1, 's-msa', 'fail')
    ep, = episodes(write_manifest(tmp_path, log))
    req = [s.ev for s in rp.iter_events(ep) if s.ev.phase == 'request']
    assert [e.turn for e in req] == [1, 2, 3, 3, 4, 5]            # the re-prompt after no tool call repeats turn 3
    res = [e.last_result for e in req]
    assert res[0] is None and res[3] is None                        # the first request; the format-error re-prompt
    assert (res[1].command, res[1].exit_code, res[1].output) == ('ls', 0, 'a.py\n')
    assert res[2].exit_code == 1 and 'Traceback' in res[2].output
    assert (res[4].exit_code, res[4].timed_out) == (-1, True)
    assert (res[5].truncated, res[5].output) == (True, 'HEAD\nTAIL')
    rep = [s.ev.reply for s in rp.iter_events(ep) if s.ev.phase == 'reply']
    assert rep[2].action is None and rep[2].parse_error.startswith('No tool calls')
    assert rep[5].action['done'] is True
    cfg = {'triggers': [{'use': 'action_error', 'params': {'streak': 2}}]}
    *fires, summ = rp.replay_episode(ep, cfg)
    assert [(f['turn'], f['reason']) for f in fires] == [(4, '2 failed actions in a row, last: timed out')]
    assert summ['policy']['decided_turn'] == 4


def test_msa_builtin_check(tmp_path):
    log = Log(tmp_path, 'r2', 'student_only', 'student', harness='msa')
    msa_episode(log, 1, 's-msa', 'fail')
    out = tmp_path / 'out'
    assert rp.main(['--runs', write_manifest(tmp_path, log), '--builtin-check', '--out', str(out)]) == 0
    c = json.loads((out / 'builtin_check.json').read_text())['classes']['student_alone']
    assert c['requests'] == c['matched'] == 5 and c['logged_fires'] == 1     # the parse-error request logs none


# ==== relay episodes: replayed up to the first real intervention =============================================
def test_relay_t2_stops_after_a_reply_time_repair(tmp_path):
    log = Log(tmp_path, 'r3', 'relay_repair', 'relay', repair_on_parse_error=True)
    msgs = [{'role': 'user', 'content': 'You are an AI assistant. Task s-rel: fix run.py'}]
    for seq in (1, 2):
        log.request(1, 's-rel', seq, seq, list(msgs), response={'role': 'assistant', 'content': t2('ls')})
        msgs += [{'role': 'assistant', 'content': t2('ls')}, {'role': 'user', 'content': screen('ls', 'run.py')}]
    stripped = [dict(m, content=m['content'].split('<|end_think|>')[-1]) if m['role'] == 'assistant' else m
                for m in msgs]
    log.request(1, 's-rel', 3, 3, list(msgs), owner='teacher', bodies={'student': list(msgs), 'teacher': stripped},
                response={'role': 'assistant', 'content': t2('cat run.py')}, repair=True, repair_kind='parse_error',
                repair_reason='ERROR: No valid JSON found', discarded_student_reply=resp({'role': 'assistant',
                                                                                        'content': 'not json'}))
    log.request(1, 's-rel', 4, 4, list(msgs), response={'role': 'assistant', 'content': t2('ls')})
    log.add(1, 's-rel', 'relay_repair_t2', 'pass')
    ep, = episodes(write_manifest(tmp_path, log))
    info = {}
    steps = list(rp.iter_events(ep, info=info))
    assert [(s.ev.phase, s.ev.turn, s.real) for s in steps][-2:] == [('request', 3, None), ('reply', 3, 'reply')]
    assert steps[-1].ev.reply.text == 'not json' and steps[-1].ev.reply.parse_error
    assert info['stop'] == 'real_intervention' and info['real']['kind'] == 'repair' and info['real']['turn'] == 3
    *_, summ = rp.replay_episode(ep, {'triggers': [{'use': 'parse_repair'}]})
    assert summ['cls'] == 'relay' and summ['policy']['decided_turn'] == 3


def test_relay_msa_request_time_assist_rebuilds_harbors_history(tmp_path):
    log = Log(tmp_path, 'r4', 'relay_repair', 'relay', harness='msa', argv=('--teacher-hint-text', HINT))
    msgs = [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': 'task s-as'}]
    m1 = msa_reply(1, 'python slow.py')
    log.request(1, 's-as', 1, 1, list(msgs), response=m1, served_message=m1)
    msgs += [m1, tool(1, returncode=-1, output='', exception_info='Command timed out after 30 seconds')]
    teacher_view = [dict(msgs[0], content=msgs[0]['content'] + '\n\n' + HINT), msgs[1],
                    {k: v for k, v in m1.items() if k not in ('reasoning', 'reasoning_content')}, msgs[3]]
    log.request(1, 's-as', 2, 2, list(msgs), owner='teacher', bodies={'teacher': teacher_view},
                response=msa_reply(2, 'nohup python slow.py &'), assist=True, assist_kind='cmd_timeout')
    log.add(1, 's-as', 'msa_relay_assist', 'fail')
    ep, = episodes(write_manifest(tmp_path, log))
    stats = rp.collections.Counter()
    steps = list(rp.iter_events(ep, stats=stats))
    assert [(s.ev.phase, s.ev.turn, s.real) for s in steps] == [('request', 1, None), ('reply', 1, None),
                                                               ('request', 2, 'request')]
    assert steps[-1].ev.messages == msgs and steps[-1].ev.last_result.timed_out
    assert stats['history_exact'] == 2 and stats['history_teacher_view_undone'] == 1


# ==== the report =============================================================================================
def test_cli_report_labels_and_run_dir_mode(tmp_path):
    log = Log(tmp_path, 'r5', 'student_only', 'student')
    t2_episode(log, 1, 's-loop', LOOP_STEPS, 'fail')
    t2_episode(log, 2, 's-pass', PASS_STEPS, 'pass')
    labels = tmp_path / 'labels.jsonl'
    labels.write_text(json.dumps(dict(sid='s-loop', point_of_no_return_turn=8)) + '\n')
    out = tmp_path / 'out'
    cfg = str(HERE.parent / 'replay_example.yaml')
    assert rp.main(['--runs', write_manifest(tmp_path, log), '--config', cfg, '--labels', str(labels),
                    '--out', str(out)]) == 0
    s = json.loads((out / 'summary.json').read_text())['classes']['student_alone']
    loop = s['triggers']['loop']['first_fire']
    assert (loop['fire_rate_failing'], loop['fire_rate_passing'], loop['precision'], loop['recall']) == \
        (1.0, 0.0, 1.0, 1.0)
    assert (loop['first_turn_failing_mean'], loop['lead_failing_mean'], loop['lead_vs_pnr_mean']) == (6, 1, 2)
    dc = s['triggers']['done_claim']['first_fire']
    assert (dc['fire_rate_passing'], dc['fire_rate_failing']) == (1.0, 1.0)
    assert len((out / 'per_episode.jsonl').read_text().splitlines()) == 2
    fires = [json.loads(x) for x in (out / 'fires.jsonl').read_text().splitlines()]
    assert sorted((f['trigger'], f['turn'], f['valid']) for f in fires if f['sid'] == 's-loop') == [
        ('action_error', 6, True), ('done_claim', 6, False), ('done_claim', 7, False), ('loop', 6, True)]
    # the same run without a manifest: outcomes from the harbor trials (readout.py's rules), groups <arm>_<harness>
    for sid, reward in (('s-loop', 0.0), ('s-pass', 1.0)):
        t = tmp_path / 'r5' / 'jobs' / 'r5_student_only' / f'trial-{sid}'
        (t / 'agent').mkdir(parents=True)
        (t / 'result.json').write_text(json.dumps(dict(task_name='task-' + sid, verifier_result=dict(
            rewards=dict(reward=reward)), exception_info=None)))
        (t / 'agent' / 'trajectory.json').write_text(json.dumps(dict(session_id=sid, steps=[])))
    eps = list(rp.load_episodes([str(tmp_path / 'r5')]))
    assert sorted((e.sid, e.outcome, e.group) for e in eps) == [('s-loop', 'fail', 'student_only_t2'),
                                                                ('s-pass', 'pass', 'student_only_t2')]
    assert [e.sid for e in rp.load_episodes([str(tmp_path / 'r5')], harness='msa')] == []


def test_example_config_loads():
    policy, specs = rp.load_trigger_config(str(HERE.parent / 'replay_example.yaml'))
    assert [s.name for s in specs] == ['action_error', 'loop', 'no_progress_wait', 'done_claim']
    assert policy.earliest_turn == 2
