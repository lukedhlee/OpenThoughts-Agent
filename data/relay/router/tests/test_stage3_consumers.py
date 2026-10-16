"""CPU tests for the consumers of stage 3's records (notes/ot-agent/relay-triggers): router runs under --trigger-config
with a span of k = 2 that hands back render as SFT rows with every teacher turn trained and every student turn masked
(Terminus-2 through harbor's own agent, mini-swe-agent through the msa driver); a teacher turn follows its
intervention's student view; readout reads the run without a false H4 (a repair of two turns hands back after both);
select_kept, msa/gate.py and horizon/arm_gates.py count interventions. Legacy runs (no `intervention` field) read as
before: checked by diffing every consumer's output on recorded runs before and after (agent_logs 2026-10-16).

    PYTHONPATH=<harbor>/src:<harbor> python -m pytest -q data/relay/router/tests/test_stage3_consumers.py
"""
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
for d in ('pilot', 'sft', 'horizon'):
    sys.path.insert(0, str(HERE.parent.parent / d))
import readout  # noqa: E402
import select_kept  # noqa: E402
from test_relay_router import Stack as T2Stack, _word_tokenizer, needs_harbor, run_agent  # noqa: E402
from test_relay_router_msa import Driver, Stack, qwen_xml  # noqa: E402

BOS = '<|begin_of_text|>'
TEMPLATE = HERE / 'grug0921_chat_template.jinja'
LOOK = ['Look.\n\n' + qwen_xml('ls'), 'More.\n\n' + qwen_xml('pwd'), 'Again.\n\n' + qwen_xml('whoami')]


def config(tmp, d):
    p = Path(tmp) / 'trig.yaml'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(d))
    return ['--trigger-config', str(p)]


def masks(row, tok, owner_of_turn):
    """Per assistant turn of a rendered row: (owner, trained tokens, tokens, its <|eot_id|> trained)."""
    eot = tok.token_to_id('<|eot_id|>')
    out = []
    for m in row['turns']:
        s, e = m['span']
        toks = [k for k, (a, b) in enumerate(row['offsets']) if a >= s and b <= e]
        out.append((owner_of_turn(m), sum(row['loss'][k] for k in toks), len(toks),
                    bool(toks) and row['ids'][toks[-1]] == eot and row['loss'][toks[-1]] == 1))
    return out


@needs_harbor
def test_t2_span_of_two_renders_teacher_turns_trained_student_turns_masked(tmp_path):
    import render
    tokp = _word_tokenizer(tmp_path / 'tok.json')
    tok, tpl = render.rcap.load_tokenizer(tokp), render.load_template(TEMPLATE)
    for view in ('capped', 'stripped'):
        trig = {'use': 'scripted', 'params': {'turns': [3]}, 'span': 2, 'view': {'student_view_after_handback': view}}
        d = tmp_path / view
        with T2Stack(d, router_args=config(d, {'triggers': [trig]})) as st:
            r = run_agent(st, 'budget', d, max_turns=6)
            recs = st.turns(r.sid)
        main = [x for x in recs if x.get('turn') is not None]
        assert [x['owner'] for x in main][:5] == ['student', 'student', 'teacher', 'teacher', 'student']
        row = render.render_episode(r.traj, recs, tok, tpl, BOS)
        c = render.check_row(row, tok)
        got = masks(row, tok, lambda m: m['owner'])
        assert c['teacher_turns'] == 2 and [o for o, *_ in got][:5] == [o['owner'] for o in main][:5]
        assert all(n == 0 for o, n, _, _ in got if o == 'student')
        assert all(n > 0 and eot for o, n, _, eot in got if o == 'teacher')
        teacher_text = [row['text'][slice(*m['span'])] for m in row['turns'] if m['owner'] == 'teacher']
        assert all(('<|start_think|>' in t) == (view == 'capped') for t in teacher_text)


def test_msa_span_of_two_render_and_readout(tmp_path):
    """A parse repair of two turns (badjson's unclosed tool call), then a scripted turn at once: the row trains the
    three teacher turns and masks the student's; readout counts the repair as handed back after its two turns (no false
    H4: another intervention follows, as an assist may today) and lists both interventions."""
    import render_msa
    tokp = _word_tokenizer(tmp_path / 'tok.json')
    tok, tpl = render_msa.rcap.load_tokenizer(tokp), render_msa.load_template(TEMPLATE)
    cfg = {'triggers': [{'use': 'parse_repair', 'where': 'rewrite', 'span': 2, 'policy': {'earliest_turn': 1}},
                        {'use': 'scripted', 'params': {'turns': [4]}, 'span': 1}]}
    with Stack(tmp_path, router_args=config(tmp_path, cfg)) as st:
        st.teacher.script = LOOK * 3
        dr = Driver(st, 'badjson').run(9)
        recs = st.turns('sid-badjson')
        rv = readout.router_view(str(st.log_dir))
    main = [x for x in recs if x.get('turn') is not None]
    assert ''.join('T' if x['owner'] == 'teacher' else 'S' for x in main) == 'STTTS' and dr.submitted
    assert [(x.get('intervention') or {}).get('i') for x in main] == [None, 1, 2, 1, None]
    row = render_msa.render(dr.msgs, recs, tok, tpl, BOS)
    got = masks(row, tok, lambda m: m['owner'])
    assert [o for o, *_ in got] == ['student', 'teacher', 'teacher', 'teacher', 'student']
    assert all(n == 0 for o, n, _, _ in got if o == 'student') and all(n > 0 and e for o, n, _, e in got if o == 'teacher')
    rp = readout.repair_view(rv, [])
    assert (rp['repairs'], rp['repair_returned_to_student'], rp['repair_kept_by_teacher']) == (1, 1, 0)
    assert rp['executed_turns'] == {'student': 2, 'teacher_repair': 1, 'teacher_span': 2}
    iv = readout.interventions_view(rv, [])
    assert iv['interventions'] == 2 and iv['by_trigger']['parse_repair']['teacher_turns_p50'] == 2
    assert iv['by_trigger']['scripted']['handed_back'] == 1
    assert readout.interventions_view(dict(rv, eps={k: dict(e, interventions={}) for k, e in rv['eps'].items()}),
                                      []) is None


def test_released_takeover_rendered_without_the_strip(tmp_path):
    """--student-view-after-takeover strip: a takeover released before the end gives the student its thinking back,
    so the row (the student's view at the end) keeps the student's own reasoning; render_msa.sticky_at reads it."""
    import render_msa
    rec = [dict(seq=1, turn=1, owner='student'), dict(seq=2, turn=2, owner='teacher', takeover={'trigger': 'loop'}),
           dict(seq=3, turn=3, owner='teacher')]
    assert render_msa.sticky_at(rec) and render_msa.sticky_at(rec, 2)
    assert not render_msa.sticky_at(rec + [dict(seq=4, turn=4, owner='student')])
    assert render_msa.sticky_at(rec + [dict(seq=4, turn=4, owner='student')], 3)
    assert not render_msa.sticky_at(rec[:1])


def test_select_kept_reads_interventions():
    t = dict(first_teacher_record=dict(intervention={'id': 2}), turn=3, trigger='done_claim')
    after = [dict(owner='teacher', turn=3, intervention={'id': 2}, response={'choices': [{'message': {
                 'content': json.dumps({'analysis': 'a', 'plan': 'p', 'commands': [], 'task_complete': True})}}]}),
             dict(owner='student', turn=4),
             dict(owner='teacher', turn=5, intervention={'id': 3}, response={'choices': [{'message': {
                 'content': json.dumps({'analysis': 'a', 'plan': 'p', 'commands': [{'keystrokes': 'ls\n'}]})}}]})]
    e = dict(takeover=t, main=after, interventions={})
    assert not select_kept.teacher_worked(e)       # the command came under a later intervention, not the takeover's
    assert select_kept.teacher_worked(dict(e, takeover=dict(t, first_teacher_record={})))   # legacy: any later turn
    assert select_kept.stalled(dict(takeover=None, main=[], interventions={1: {'trigger': 'loop'}}), {}) == 'trigger'
    shadow = dict(takeover=None, interventions={}, main=[dict(decisions=[{'trigger': 'no_progress_wait',
                                                                         'outcome': 'shadow'}])])
    assert select_kept.stalled(shadow, {}) == 'trigger'


def test_arm_gates_counts_interventions(tmp_path):
    import arm_gates
    ev = tmp_path / 'events.jsonl'
    ev.write_text('\n'.join(json.dumps(e) for e in [
        dict(event='intervention', sid='a', trigger='action_error'), dict(event='intervention', sid='a', trigger='loop'),
        dict(event='takeover', sid='b', trigger='loop'), dict(event='ending', sid='c', ending='deadline')]) + '\n')
    st = dict(offsets={}, takeovers={}, trials={}, metrics={})
    arm_gates.new_events(str(ev), st)
    assert st['interventions'] == {'a': ['action_error', 'loop']} and st['takeovers'] == {'b': 'loop'}
