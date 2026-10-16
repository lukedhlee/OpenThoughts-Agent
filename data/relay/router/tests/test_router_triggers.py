"""CPU tests for stage 2 of notes/ot-agent/relay-triggers: the router's decisions through interventions.Controller.

The legacy config the router builds from its flags (router_triggers.legacy_config), --trigger-config (it replaces those
flags, routes the same, and adds the decisions / intervention record fields), the assist trigger moved out of the
router with today's retry order, and router_replay.py on a logged msa episode (the stage-2 differential replay's tool).
Router-level, no harbor needed.

    python -m pytest -q data/relay/router/tests/test_router_triggers.py
"""
import json
import sys
import time
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import interventions as iv  # noqa: E402
import relay_router as rr  # noqa: E402
import router_replay  # noqa: E402
import router_triggers as rtg  # noqa: E402
from test_relay_router_msa import Driver, Stack, post  # noqa: E402

BASE = ['--port', '0', '--log-dir', 'x', '--student-url', 'u', '--student-model', 's', '--teacher-url', 'u',
        '--teacher-model', 't']
T2_REPAIR = ['--mode', 'relay', '--repair-on-parse-error', '--autofix', '--context-budget-tokens', '32000',
             '--teacher-format-guard', '--verify-note']
MSA_ASSIST = ['--mode', 'relay', '--harness', 'msa', '--repair-on-parse-error', '--context-budget-tokens', '40000',
              '--assist', '--assist-ctx', '20000,35000', '--assist-gap', '3', '--assist-max', '6',
              '--time-takeover-frac', '0.6', '--verify-note']
# what each request's record must agree on between the legacy flags and their --trigger-config (timing and the
# additive fields aside)
FIELDS = ('turn', 'owner', 'repair', 'repair_kind', 'assist', 'assist_kind', 'done_claim', 'verify_note', 'autofix',
          'student_parse_error')


def specs_of(args, parser_loaded=True):
    a = rr.parse_args(args + BASE)
    return rtg.load_config(rtg.legacy_config(a, parser_loaded, 'NOTE'))[1]


def row(s):
    return (s.name, s.where, s.span, s.limits, s.observe, s.on_retry, s.teacher_note)


# ---- the legacy config -----------------------------------------------------------------------------------------
def test_legacy_config_terminus2_relay_repair():
    """Terminus-2: the repair confirmation first, the environment triggers, the context budget (asked from turn 1, its
    count is logged on every student request), then at reply time the parse repair and done_claim (next, last)."""
    got = [row(s) for s in specs_of(T2_REPAIR)]
    assert got == [
        ('repair_confirm', None, 1, {'earliest_turn': 1}, 'always', 'observe', None),
        ('loop', None, -1, {}, 'armed', 'memo', None),
        ('no_progress_wait', None, -1, {}, 'armed', 'memo', None),
        ('context_budget', None, -1, {'earliest_turn': 1}, 'armed', 'observe', None),
        ('parse_repair', 'rewrite', 1, {'earliest_turn': 1}, 'always', 'observe', None),
        ('done_claim', 'next', -1, {'earliest_turn': 1}, 'always', 'memo', 'NOTE')]
    # the parser is what turns the repair on: --teacher-format-guard alone loads it, and without it neither repair runs
    plain = specs_of(['--mode', 'relay'], parser_loaded=False)
    assert [s.name for s in plain] == ['loop', 'no_progress_wait', 'done_claim']
    assert plain[-1].teacher_note is None                           # no --verify-note


def test_legacy_config_msa_assist():
    """msa: environment, context, time, assist (one turn; cooldown = gap - 1 and its own cap, so repairs do not count),
    then the parse repair and the done claim (rewrite, sticky, the note) before the other decision triggers."""
    got = [row(s) for s in specs_of(MSA_ASSIST + ['--takeover', 'done_claim,loop,no_progress_wait,destructive'])]
    assert got == [
        ('loop', None, -1, {}, 'armed', 'memo', None),
        ('no_progress_wait', None, -1, {}, 'armed', 'memo', None),
        ('context_budget', None, -1, {'earliest_turn': 1}, 'armed', 'observe', None),
        ('time_budget', None, -1, {'earliest_turn': 2}, 'armed', 'observe', None),
        ('assist', None, 1, {'cooldown': 2, 'max_interventions': 6, 'earliest_turn': 2}, 'armed', 'observe', None),
        ('parse_repair', 'rewrite', 1, {'earliest_turn': 1}, 'always', 'observe', None),
        ('done_claim', 'rewrite', -1, {'earliest_turn': 1}, 'always', 'memo', 'NOTE'),
        ('destructive', 'rewrite', -1, {}, 'armed', 'memo', None)]
    a = specs_of(MSA_ASSIST)[4].trigger
    assert isinstance(a, rtg.Assist) and (a.ctx, a.error_streak) == ([20000, 35000], 4)


def test_control_and_student_only_log_every_fire(tmp_path):
    """--mode teacher / student never hand an episode over: the takeover triggers only log, on both owners' turns."""
    for mode in ('teacher', 'student'):
        a = rr.parse_args(['--mode', mode, '--port', '0', '--log-dir', str(tmp_path / mode), f'--{mode}-url', 'u',
                           f'--{mode}-model', 'm'])
        specs = rr.Router(a).trigger_specs
        assert [s.name for s in specs] == ['loop', 'no_progress_wait', 'done_claim']
        assert all(s.log_only and s.owners == {'student', 'teacher'} for s in specs)


def test_trigger_config_replaces_the_intervention_flags(tmp_path, capsys):
    p = tmp_path / 'c.yaml'
    p.write_text('triggers: []\n')
    for extra in (['--assist'], ['--context-budget-tokens', '9'], ['--takeover', 'loop'], ['--verify-note'],
                  ['--repair-on-parse-error']):
        with pytest.raises(SystemExit):
            rr.parse_args(['--mode', 'relay', '--trigger-config', str(p)] + extra + BASE)
        assert '--trigger-config replaces ' + extra[0] in capsys.readouterr().err
    # control's own verify note stays a router flag; flags that are not interventions stay too
    rr.parse_args(['--mode', 'teacher', '--trigger-config', str(p), '--verify-note', '--port', '0', '--log-dir', 'x',
                   '--teacher-url', 'u', '--teacher-model', 'm'])
    rr.parse_args(['--mode', 'relay', '--trigger-config', str(p), '--autofix', '--teacher-format-guard'] + BASE)


def test_print_trigger_config_round_trips(capsys):
    """`python router_triggers.py <flags>` prints the config the flags stand for; it loads to the same triggers."""
    rtg.main(MSA_ASSIST + BASE)
    cfg = yaml.safe_load(capsys.readouterr().out)
    want = [row(s) for s in rtg.load_config(rtg.legacy_config(rr.parse_args(MSA_ASSIST + BASE), False,
                                                              rr.VERIFY_NOTE_MSA))[1]]
    assert [row(s) for s in rtg.load_config(cfg)[1]] == want
    assert iv.BUILTINS['assist'] is iv.Assist         # stage 0's placeholder is back once a config has loaded
    assert cfg['triggers'][6]['teacher_note'] == rr.VERIFY_NOTE_MSA


# ---- the router on the Controller ------------------------------------------------------------------------------
OUT = {'echo step-1': 'The command timed out after 30 seconds and was killed.'}


def _episode(tmp, args, sc='work', n=8, outputs=OUT, budgets=None):
    with Stack(tmp, router_args=args, budgets=budgets) as st:
        st.student.count_fn = lambda msgs: 100
        st.teacher.count_fn = lambda msgs: 100
        Driver(st, sc, outputs=outputs).run(n)
        return st.turns(f'sid-{sc}'), st.router


@pytest.mark.parametrize('sc,args', [
    ('work', ['--assist', '--assist-ctx', '', '--context-budget-tokens', '40000']),
    ('badjson', ['--repair-on-parse-error', '--autofix', '--verify-note']),
    ('done', ['--verify-note', '--assist', '--assist-ctx', '']),
])
def test_trigger_config_routes_as_its_flags(tmp_path, capsys, sc, args):
    """The YAML a flag set stands for routes every request the same and writes the same legacy fields; only it adds
    decisions (every fire, what became of it) and intervention (the one the teacher answers under)."""
    legacy, _ = _episode(tmp_path / 'flags', args, sc)
    rtg.main(['--mode', 'relay', '--harness', 'msa'] + args + BASE)
    cfg = tmp_path / 'trig.yaml'
    cfg.write_text(capsys.readouterr().out)
    rest = router_replay.strip_flags(args, {o for _, o in rtg.LEGACY_FLAGS + rtg.RELAY_NOTE_FLAGS})
    viacfg, router = _episode(tmp_path / 'yaml', rest + ['--trigger-config', str(cfg)], sc)
    assert [{k: r.get(k) for k in FIELDS} for r in legacy] == [{k: r.get(k) for k in FIELDS} for r in viacfg]
    assert [(r.get('takeover') or {}).get('trigger') for r in legacy] == [(r.get('takeover') or {}).get('trigger')
                                                                          for r in viacfg]
    assert not any('decisions' in r or 'intervention' in r for r in legacy)
    assert all(isinstance(r.get('decisions'), list) for r in viacfg if r.get('turn'))
    acted = [d for r in viacfg for d in r['decisions'] if d['outcome'] == 'acted']
    assert len(acted) == sum(1 for r in viacfg if r.get('assist') and r['assist_kind'] != 'retry' or r.get('repair')
                             or r.get('takeover'))
    for r in viacfg:
        if r['owner'] == 'teacher' and (r.get('assist') or r.get('repair') or r.get('takeover')):
            assert r['intervention']['trigger'] in ('assist', 'parse_repair', 'done_claim', 'loop', 'context_budget')
    assert router.cfg['enabled'] == ['done_claim', 'loop', 'no_progress_wait']


def _assisted_then_retry(tmp, args, wait=0.0, budgets=None):
    """turn 3 follows a command timeout (an assist); the same request is then sent again (a transport retry)."""
    with Stack(tmp, router_args=['--assist', '--assist-ctx', ''] + args, budgets=budgets) as st:
        st.student.count_fn = lambda msgs: 100
        d = Driver(st, 'work', outputs=OUT)
        for _ in range(2):
            d.step()
        status, first = post(st, d.msgs, d.sid)
        time.sleep(wait)
        status2, again = post(st, d.msgs, d.sid)
        return st.turns(d.sid), st.router


def test_retry_of_an_assisted_request_is_the_teachers_again(tmp_path):
    """Today's order: a retry asks the budgets again (the context count is logged), then the teacher answers it again
    (assist_kind retry, not another assist)."""
    rows, router = _assisted_then_retry(tmp_path, ['--context-budget-tokens', '40000'])
    assert [r['owner'] for r in rows] == ['student', 'student', 'teacher', 'teacher']
    assert (rows[2]['assist_kind'], rows[3]['assist_kind'], rows[3]['retry']) == ('cmd_timeout', 'retry', True)
    assert rows[3]['student_view_tokens'] == 100 and router.counts['assists'] == 1
    assert list(rows[3])[list(rows[3]).index('retry') + 1] == 'student_view_tokens'      # counted before the assist


def test_retry_of_an_assisted_request_crossing_the_time_budget_takes_over(tmp_path):
    """The time budget crosses between the assisted request and its retry: the retry is the takeover (sticky)."""
    rows, router = _assisted_then_retry(tmp_path, ['--time-takeover-frac', '0.5'], wait=1.2, budgets={'work': 2.0})
    assert rows[2]['assist_kind'] == 'cmd_timeout' and rows[3]['retry']
    assert rows[3]['takeover']['trigger'] == 'time_budget' and not rows[3].get('assist')
    assert router.episodes['sid-work'].owner == 'teacher'


# ---- router_replay on a logged episode -------------------------------------------------------------------------
def test_replay_reproduces_logged_msa_episodes(tmp_path):
    """Logged msa relay episodes (an assist, a parse repair, a done claim) replayed through the router: every request
    rebuilt as harbor sent it, every routing as logged, no upstream call the log cannot answer; a second replay is
    identical byte for byte."""
    args = ['--repair-on-parse-error', '--autofix', '--assist', '--assist-ctx', '', '--verify-note']
    timeout = {'ls -la': 'The command timed out after 30 seconds and was killed.'}
    with Stack(tmp_path / 'live', router_args=args) as st:
        for sc in ('badjson', 'done'):
            Driver(st, sc, outputs=timeout).run(10)
        argv = ['relay_router.py', '--mode', 'relay', '--harness', 'msa', '--port', '0', '--log-dir', str(st.log_dir),
                '--tasks', str(st.tmp / 'tasks.json'), '--student-url', 'u', '--student-model', 'snowball',
                '--teacher-url', 'u', '--teacher-model', 'qwen38'] + args
        rows = st.turns()
    assert [k for k in ('assist', 'repair', 'done_claim') if any(r.get(k) for r in rows)] == ['assist', 'repair',
                                                                                              'done_claim']
    sample = tmp_path / 'sample'
    (sample / 'run1').mkdir(parents=True)
    (sample / 'run1' / 'router_relay').symlink_to(tmp_path / 'live' / 'router')
    eps = sorted({r['episode'] for r in rows})
    (sample / 'manifest.json').write_text(json.dumps([dict(run='run1', arm='relay', episode=e) for e in eps]))
    (sample / 'router_flags.json').write_text(json.dumps({'run1/relay': dict(argv=argv)}))
    router_here = str(HERE.parent / 'relay_router.py')
    for out in ('a', 'b'):
        router_replay.main(['run', '--router', router_here, '--sample', str(sample), '--out', str(tmp_path / out)])
        fid = json.loads((tmp_path / out / 'run1__relay' / 'replay.json').read_text())
        assert fid['requests'] == len(rows) == fid['request_sha_match'] == fid['owner_as_recorded']
        assert fid['stub_misses'] == 0
    with pytest.raises(SystemExit) as e:
        router_replay.main(['diff', str(tmp_path / 'a'), str(tmp_path / 'b')])
    assert e.value.code == 0
