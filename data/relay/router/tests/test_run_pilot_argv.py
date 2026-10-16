"""run_pilot.sh's TRIGGER_CONFIG on a Mac: the routers' flags (pilot/router_argv.sh, sourced in bash with the knobs
set) and the run's copy of the config (router_triggers.stage_config).

Without a config every arm's argv is today's: run_pilot_routers_8e4e515a.sh (the inline code router_argv.sh replaced,
verbatim) and router_argv.sh build the same argv on 120 random knob settings, AGENT=msa among them. With one, the arms
that run it get --trigger-config and lose exactly the flags the router refuses next to it (the router's own parser
accepts every argv built here); control stays the teacher-alone baseline. Plain bash 3.2 (a Mac's) runs it all.

    python -m pytest -q data/relay/router/tests/test_run_pilot_argv.py
"""
import os
import random
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import interventions as iv  # noqa: E402
import relay_router as rr  # noqa: E402
import router_triggers as rtg  # noqa: E402

ARGV_SH = HERE.parent.parent / 'pilot' / 'router_argv.sh'
ARMS = ('control', 'student_only', 'relay', 'relay_keep', 'relay_repair')
KNOBS = dict(AGENT='terminus2', HARBOR_SRC='/c/harbor/src', R='/runs/x', TREE='/tree', DEADLINE='1790000000',
             STUDENT_TOKENIZER='/m/tokenizer.json', TURL='http://t:8000/v1', SURL='http://s:8000/v1',
             TEACHER_MAX_TOKENS='32768', TEACHER_GUARD='1', TEACHER_RESAMPLES='2', VERIFY_NOTE='1', MAX_INPUT='131072',
             CTX_BUDGET='32000', ROW_MAX='65536', ROW_RESERVE='8192', ASSIST='1', TIME_TAKEOVER='0.6', CLOCK='paused',
             FAILOVER_5XX='1', BALANCE='active', TEACHER_HINT_TEXT='Keep it short.')
REFUSED = ('--repair-on-parse-error', '--context-budget-tokens', '--assist', '--assist-ctx', '--time-takeover-frac')


def argv(tc='', **knobs):
    """{arm: its router argv} from router_argv.sh under the knobs."""
    env = {**os.environ, **KNOBS, **knobs, 'TC': tc}
    script = (f'set -u; source {ARGV_SH}; abort() {{ echo "ABORT $*"; exit 3; }}; port=21000; '
              f'for arm in {" ".join(ARMS)}; do router_argv $arm $port; printf "%s\\n" "${{RARGV[@]}}"; echo ---; '
              f'port=$((port + 1)); done')
    out = subprocess.run(['bash', '-c', script], env=env, capture_output=True, text=True, check=True).stdout
    return dict(zip(ARMS, [blk.strip('\n').split('\n') for blk in out.split('---\n')[:len(ARMS)]]))


def test_without_a_config_the_intervention_flags_are_today_s():
    a = argv()
    rp = a['relay_repair']
    for flag in REFUSED + ('--verify-note', '--terminus-parser', '--autofix'):
        assert flag in rp, flag
    assert not any('--trigger-config' in x for x in a.values())
    assert rp[rp.index('--teacher-hint-text') + 1] == 'Keep it short.'      # an argument with spaces stays one
    for k, (arm, x) in enumerate(a.items()):
        assert rr.parse_args(x).port == 21000 + k


def test_with_a_config_the_router_gets_it_instead_of_the_flags(tmp_path):
    tc = tmp_path / 'trig.yaml'
    tc.write_text('triggers: [{use: loop}]\n')
    a = argv(str(tc), AGENT='msa')
    for arm in ('relay', 'relay_keep', 'relay_repair', 'student_only'):
        x = a[arm]
        assert x[x.index('--trigger-config') + 1] == str(tc), arm
        assert not set(REFUSED) & set(x), arm
        assert ('--verify-note' in x) == (arm == 'student_only'), arm   # a relay arm's note is its config's done_claim
        p = rr.parse_args(x)                      # the router's own conflict check passes
        assert p.trigger_config == str(tc) and p.mode == ('student' if arm == 'student_only' else 'relay')
    assert '--trigger-config' not in a['control'] and '--verify-note' in a['control']   # the baseline as before
    assert '--autofix' in a['relay_repair'] and '--student-row-max-tokens' in a['relay_repair']
    t2 = argv(str(tc))['relay_repair']            # Terminus-2: the parser stays for a parse_repair entry
    assert '--terminus-parser' in t2 and '--repair-on-parse-error' not in t2


OLD_SH = HERE / 'run_pilot_routers_8e4e515a.sh'
CHOICES = dict(AGENT=('terminus2', 'msa'), TURL=('', 'http://t:8000/v1'), SURL=('', 'http://s:8000/v1'),
               TEACHER_GUARD=('0', '1'), VERIFY_NOTE=('0', '1'), VERIFY_NOTE_TEXT=(None, '', 'Check it twice.'),
               TEACHER_HINT_TEXT=(None, '', 'Keep it short.'), MAX_INPUT=('65536', '131072'), CTX_BUDGET=('', '32000'),
               ROW_MAX=('', '65536'), STUDENT_VIEW_STRIP=(None, '0', '1'), ASSIST=(None, '0', '1'),
               ASSIST_CTX=(None, '10000,20000'), ASSIST_ERR=(None, '2'), ASSIST_GAP=(None, '5'), ASSIST_MAX=(None, '3'),
               TIME_TAKEOVER=(None, '', '0.6'), CLOCK=('wall', 'repair', 'paused'), FAILOVER_5XX=('0', '1'),
               BALANCE=('active', 'pinned'))       # None: unset
DROPPED = {'--repair-on-parse-error': 0, '--context-budget-tokens': 1, '--assist': 0, '--assist-ctx': 1,
           '--assist-error-streak': 1, '--assist-gap': 1, '--assist-max': 1, '--time-takeover-frac': 1}   # flag: n args
RELAY_DROPPED = {'--verify-note': 0, '--verify-note-text': 1}
TC = '/runs/x/trigger_config/trig.yaml'


def settings(n=120, seed=0):
    rnd = random.Random(seed)
    return [{k: rnd.choice(v) for k, v in CHOICES.items()} for _ in range(n)]


def old_and_new(cases):
    """Per knob setting, three {arm: argv}: today's inline code (the fixture, with RARGV set where it started the
    router), router_argv.sh, and router_argv.sh with TC. One bash for all; router_argv runs under set -u."""
    old = OLD_SH.read_text()
    for a, b in (('$PY $ROUTER ', 'RARGV=('), (' \\\n    > $R/router_$arm.log 2>&1 &', ')'), ('${PORT[$arm]}', '$port'),
                 ('  RPID[$arm]=$!', '  emit')):
        assert old.count(a) == 1, a
        old = old.replace(a, b)
    fixed = {k: KNOBS[k] for k in ('HARBOR_SRC', 'R', 'TREE', 'DEADLINE', 'STUDENT_TOKENIZER', 'TEACHER_MAX_TOKENS',
                                   'TEACHER_RESAMPLES', 'ROW_RESERVE')}
    script = [f'source {ARGV_SH}', 'abort() { echo "ABORT $*"; exit 3; }',
              'emit() { printf "%s\\n" "${RARGV[@]}"; echo ---; port=$((port + 1)); }',
              'old() {\nport=21000', old, '}',
              'new() { set -u; port=21000; for arm in $ARMS; do router_argv $arm $port; emit; done; set +u; }']
    for c in cases:
        knobs = [f'{k}={shlex.quote(v)}' if v is not None else f'unset {k}'
                 for k, v in {**fixed, **c, 'ARMS': ' '.join(ARMS)}.items()]
        script.append('( ' + '; '.join(knobs) + f'; TC=; old; echo ===; new; echo ===; TC={TC}; new ); echo "###"')
    out = subprocess.run(['bash'], input='\n'.join(script), capture_output=True, text=True, check=True).stdout
    blocks = out.split('###\n')[:-1]
    assert len(blocks) == len(cases)
    return [[dict(zip(ARMS, [a.strip('\n').split('\n') for a in part.split('---\n')[:len(ARMS)]]))
             for part in blk.split('===\n')] for blk in blocks]


def without(argv, flags):
    """argv less the flags and their arguments ({flag: number of arguments})."""
    out, skip = [], 0
    for x in argv:
        if skip:
            skip -= 1
        elif x in flags:
            skip = flags[x]
        else:
            out.append(x)
    return out


def test_without_a_config_every_arm_s_argv_is_today_s():
    cases = settings()
    assert any(c['AGENT'] == 'msa' and c['TURL'] and c['TEACHER_GUARD'] == '1' for c in cases)   # empty PARSER_ARGS
    for c, (old, new, _) in zip(cases, old_and_new(cases)):
        assert len(old) == len(ARMS) and all(old[arm] == new[arm] for arm in ARMS), c


def test_with_a_config_an_arm_loses_exactly_the_refused_flags():
    cases = settings(seed=1)
    for c, (old, _, tc) in zip(cases, old_and_new(cases)):
        assert tc['control'] == old['control'], c                       # the teacher-alone baseline as before
        for arm in ARMS[1:]:
            x = tc[arm]
            k = x.index('--trigger-config')
            assert x[k + 1] == TC and '--trigger-config' not in x[k + 2:], (c, arm)
            drop = {**DROPPED, **(RELAY_DROPPED if arm.startswith('relay') else {})}
            assert x[:k] + x[k + 2:] == without(old[arm], drop), (c, arm)
            if c['TURL'] and c['SURL']:
                assert rr.parse_args(x).trigger_config == TC              # the router's own conflict check passes


PLUGIN = '''
from interventions import Decision


class EveryOther:
    name, phases = 'every_other', frozenset({'request'})

    async def observe(self, ev):
        return Decision('', 'tick') if ev.turn % 2 == 0 else None
'''


def test_stage_copies_the_config_and_its_triggers(tmp_path):
    src = tmp_path / 'src'
    (src / 'plugins').mkdir(parents=True)
    (src / 'plugins' / 'mine.py').write_text(PLUGIN)
    (src / 'my.yaml').write_text('triggers:\n  - use: any_of\n    params:\n      of:\n'
                                 '        - {use: plugins/mine.py:EveryOther}\n        - {use: loop}\n    span: 1\n')
    copy = rtg.stage_config(str(src / 'my.yaml'), str(tmp_path / 'run' / 'trigger_config'))
    dest = tmp_path / 'run' / 'trigger_config'
    assert copy == str(dest / 'my.yaml') and (dest / 'plugins' / 'mine.py').read_text() == PLUGIN
    sums = (dest / 'SHA256SUMS').read_text().splitlines()
    assert [s.split()[1] for s in sums] == ['my.yaml', 'plugins/mine.py'] and all(len(s.split()[0]) == 64 for s in sums)
    out = subprocess.run([sys.executable, str(HERE.parent / 'router_triggers.py'), 'stage', str(src / 'my.yaml'),
                          str(tmp_path / 'run2')], capture_output=True, text=True, check=True).stdout
    assert out.strip() == str(tmp_path / 'run2' / 'my.yaml')
    (src / 'bad.yaml').write_text('triggers: [{use: plugins/missing.py:X}]\n')
    with pytest.raises(iv.ConfigError, match='missing.py does not exist'):
        rtg.stage_config(str(src / 'bad.yaml'), str(tmp_path / 'run3'))
    (src / 'typo.yaml').write_text('triggers: [{use: lop}]\n')
    with pytest.raises(iv.ConfigError, match='no built-in'):
        rtg.stage_config(str(src / 'typo.yaml'), str(tmp_path / 'run4'))
