"""Replay integration of the LLM judge (router/judge.py): replay.py runs judge configs against a local stub judge on
the synthetic Terminus-2 and mini-swe-agent episodes of test_replay.py. No network beyond 127.0.0.1.

    python -m pytest -q data/relay/triggers/tests/test_replay_judge.py
"""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROUTER_TESTS = HERE.parent.parent / 'router' / 'tests'
for _d in (HERE, HERE.parent, ROUTER_TESTS):
    sys.path.insert(0, str(_d))
import test_replay as tr  # noqa: E402   (the synthetic router-log writers)
import replay as rp  # noqa: E402
import judge as jg  # noqa: E402
from stub_judge import StubJudge, chat, prompt_of  # noqa: E402

PROMPT = 'TASK: $task\nHISTORY:\n$history\nLAST:\n$last_result\nEND\n$proposal'


def section(p, start, end):
    return p.split(start, 1)[1].split(end, 1)[0] if start in p else ''


def failing_last_result(body):
    """0.9 when the last action's result shows an error, else 0.1."""
    last = section(prompt_of(body), 'LAST:', 'END')
    bad = any(w in last for w in ('Traceback', 'timed out'))
    why = 'error in the last result' if bad else 'ok'
    return 200, chat(json.dumps({'score': 0.9 if bad else 0.1, 'rationale': why}))


@pytest.fixture(autouse=True)
def fresh_cache():
    jg.clear_cache()
    yield
    jg.clear_cache()


def test_judge_replay_t2(tmp_path):
    log = tr.Log(tmp_path, 'r1', 'student_only', 'student')
    tr.t2_episode(log, 1, 's-loop', tr.LOOP_STEPS, 'fail')
    tr.t2_episode(log, 2, 's-pass', tr.PASS_STEPS, 'pass')
    loop, ok = tr.episodes(tr.write_manifest(tmp_path, log))
    calls = tmp_path / 'calls.jsonl'
    with StubJudge(failing_last_result) as stub:
        cfg = {'triggers': [{'use': 'judge:LlmJudge', 'span': -1, 'params': {
            'endpoint': stub.url, 'model': 'judge', 'threshold': 0.5, 'prompt': PROMPT, 'log': str(calls)}}]}
        stats = rp.collections.Counter()
        r_loop = rp.replay_episode(loop, cfg, stats=stats)
        r_ok = rp.replay_episode(ok, cfg, stats=stats)
        bodies = list(stub.bodies)
    fires, summ = r_loop[:-1], r_loop[-1]
    # the loop's Traceback is the last result at requests 4, 5 and 6; the policy acts at 4, so 5 and 6 are invalid
    assert [(f['trigger'], f['turn'], f['phase'], f['valid']) for f in fires] == [
        ('llm_judge', 4, 'request', True), ('llm_judge', 5, 'request', False), ('llm_judge', 6, 'request', False)]
    assert fires[0]['info']['rationale'] == 'error in the last result' and fires[0]['score'] == 0.9
    assert (summ['policy']['decided_turn'], summ['policy']['start_phase']) == (4, 'request')
    assert r_ok[:-1] == [] and r_ok[-1]['policy'] is None
    # the student's view only: no think spans in any prompt
    assert all('<|start_think|>' not in prompt_of(b) and 'thinking' not in prompt_of(b) for b in bodies)
    # latency accounting: one line per ask (live and shadow controllers both ask); a prompt is sent once, the other
    # controller's ask is a cache hit
    lines = [json.loads(x) for x in calls.read_text().splitlines()]
    real = [x for x in lines if not x['cached']]
    assert len(bodies) == len({prompt_of(b) for b in bodies}) == len(real)
    assert len(lines) == (4 + 7) + (4 + 4)          # loop: live to its first intervention + shadow; pass: both, all
    assert all(x['latency_s'] > 0 and x['prompt_tokens'] == 123 for x in real)
    assert {x['outcome'] for x in lines} == {'quiet', 'fired'} and stats['request_events'] == 11


def test_llm_judge_by_its_builtin_name(tmp_path):
    """`use: llm_judge` replays as `judge:LlmJudge` does: replay registers the judge while it loads a config, as the
    router does, so one YAML works offline and online; the registry holds interventions' stub again afterwards."""
    stub_cls = rp.iv.BUILTINS['llm_judge']
    assert stub_cls is not jg.LlmJudge
    log = tr.Log(tmp_path, 'r4', 'student_only', 'student')
    tr.t2_episode(log, 1, 's-loop', tr.LOOP_STEPS, 'fail')
    loop, = tr.episodes(tr.write_manifest(tmp_path, log))
    got = {}
    with StubJudge(failing_last_result) as stub:
        for use in ('judge:LlmJudge', 'llm_judge'):
            jg.clear_cache()
            cfg = {'triggers': [{'use': use, 'span': -1, 'params': {
                'endpoint': stub.url, 'model': 'judge', 'threshold': 0.5, 'prompt': PROMPT}}]}
            got[use] = [(f['trigger'], f['turn'], f['phase'], f['valid'], f['score'])
                        for f in rp.replay_episode(loop, cfg)[:-1]]
    assert got['llm_judge'] == got['judge:LlmJudge'] == [
        ('llm_judge', 4, 'request', True, 0.9), ('llm_judge', 5, 'request', False, 0.9),
        ('llm_judge', 6, 'request', False, 0.9)]
    assert rp.iv.BUILTINS['llm_judge'] is stub_cls


def test_judge_replay_msa_reply_phase(tmp_path):
    log = tr.Log(tmp_path, 'r2', 'student_only', 'student', harness='msa')
    tr.msa_episode(log, 1, 's-msa', 'fail')
    ep, = tr.episodes(tr.write_manifest(tmp_path, log))

    def runs_python(body):
        prop = section(prompt_of(body), 'proposes this action, not yet run:', '\x00')
        return 200, chat(json.dumps({'score': 0.8 if 'python a.py' in prop else 0.0, 'rationale': 'reruns it'}))
    with StubJudge(runs_python) as stub:
        cfg = {'triggers': [{'use': 'judge:LlmJudge', 'phases': ['reply'], 'where': 'rewrite', 'span': 1,
                             'params': {'endpoint': stub.url, 'model': 'judge', 'prompt': PROMPT}}]}
        recs = rp.replay_episode(ep, cfg)
    fires, summ = recs[:-1], recs[-1]
    assert [(f['turn'], f['phase'], f['where']) for f in fires] == [(2, 'reply', 'rewrite'), (3, 'reply', 'rewrite')]
    assert (summ['policy']['decided_turn'], summ['policy']['where']) == (2, 'rewrite')


def test_example_judge_config_through_the_cli(tmp_path, monkeypatch):
    log = tr.Log(tmp_path, 'r3', 'student_only', 'student')
    tr.t2_episode(log, 1, 's-loop', tr.LOOP_STEPS, 'fail')
    tr.t2_episode(log, 2, 's-pass', tr.PASS_STEPS, 'pass')
    manifest = tr.write_manifest(tmp_path, log)

    def example_prompt(body):
        last = section(prompt_of(body), 'Result of its last action:', 'Reply with JSON')
        return 200, chat(json.dumps({'score': 0.9 if 'Traceback' in last else 0.2, 'rationale': 'r'}))
    monkeypatch.chdir(tmp_path)                      # the example logs to judge_calls.jsonl in the working dir
    with StubJudge(example_prompt) as stub:
        monkeypatch.setenv('JUDGE_URL', stub.url)
        out = tmp_path / 'out'
        example = HERE.parent / 'examples' / 'judge.yaml'
        assert rp.main(['--runs', manifest, '--config', str(example), '--out', str(out)]) == 0
    s = json.loads((out / 'summary.json').read_text())
    assert s['triggers'] == ['llm_judge']
    ff = s['classes']['student_alone']['triggers']['llm_judge']['first_fire']
    assert (ff['fire_rate_failing'], ff['fire_rate_passing']) == (1.0, 0.0)
    fires = [json.loads(x) for x in (out / 'fires.jsonl').read_text().splitlines()]
    assert [f['turn'] for f in fires] == [4, 6]      # every: 2 asks on even turns only; Traceback at 4, 5, 6
    lines = [json.loads(x) for x in (tmp_path / 'judge_calls.jsonl').read_text().splitlines()]
    assert sorted({x['turn'] for x in lines}) == [2, 4, 6] and all(x['turn'] % 2 == 0 for x in lines)
