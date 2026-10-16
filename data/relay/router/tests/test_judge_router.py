"""The LLM judge inside the live relay router: judge.LlmJudge calls a named endpoint (the trigger config's
`endpoints:`) through Event.extra['call_endpoint'], with stub_judge.StubJudge serving that endpoint. Router-level, on
the mini-swe-agent episodes of test_router_stage3.py.

    python -m pytest -q data/relay/router/tests/test_judge_router.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import judge as jg  # noqa: E402
import test_router_stage3 as s3  # noqa: E402
from stub_judge import StubJudge, chat, prompt_of  # noqa: E402

PROMPT = 'turn $turn\n$history'


def judge_cfg(stub, tmp, **params):
    """The judge calls the endpoint by name and gives no model: the router sets its endpoint's served name."""
    return {'endpoints': {'judge': {'url': stub.url, 'model': 'judge'}},
            'triggers': [{'use': 'judge:LlmJudge', 'span': 1, 'params': dict(
                endpoint='judge', prompt=PROMPT, threshold=0.5, log=str(tmp / 'calls.jsonl'), **params)}]}


def judge_bodies(stub):
    return [b for b in stub.bodies if prompt_of(b).startswith('turn ')]    # not the router's health check


def test_judge_through_the_routers_named_endpoint(tmp_path):
    jg.clear_cache()

    def answer(body):
        stuck = prompt_of(body).startswith('turn 3\n')
        return 200, chat(json.dumps({'score': 0.9 if stuck else 0.1, 'rationale': 'stuck at 3' if stuck else 'ok'}))
    with StubJudge(answer) as stub:
        rows, _, _, _ = s3.episode(tmp_path, judge_cfg(stub, tmp_path))
        bodies = judge_bodies(stub)
    assert s3.owners(rows) == 'SSTSSS' and rows[2]['intervention']['trigger'] == 'llm_judge'
    assert rows[2]['decisions'][0]['reason'] == 'judge 0.90 >= 0.5: stuck at 3'
    assert bodies and all(b['model'] == 'judge' for b in bodies)        # set by the router, not by the judge
    calls = [c for r in rows for c in r.get('endpoint_calls') or []]
    assert [(c['name'], c['status']) for c in calls] == [('judge', 200)] * len(bodies)
    lines = [json.loads(x) for x in (tmp_path / 'calls.jsonl').read_text().splitlines()]
    assert len(lines) == len(bodies) and {x['model'] for x in lines} == {'judge'}
    assert [x['turn'] for x in lines if x['outcome'] == 'fired'] == [3]
    assert all(x['latency_s'] > 0 and x['prompt_tokens'] == 123 for x in lines)


def test_endpoint_error_through_the_router_is_no_decision(tmp_path):
    jg.clear_cache()

    def answer(body):         # the health check's smoke completion succeeds; the judge's calls get a 503
        return (503, {'error': 'overloaded'}) if prompt_of(body).startswith('turn ') else (200, chat('hello'))
    with StubJudge(answer) as stub:
        rows, _, _, _ = s3.episode(tmp_path, judge_cfg(stub, tmp_path))
        n = len(judge_bodies(stub))
    assert s3.owners(rows) == 'SSSSSS' and n >= 4
    calls = [c for r in rows for c in r.get('endpoint_calls') or []]
    assert {c['status'] for c in calls} == {503} and len(calls) == n
    lines = [json.loads(x) for x in (tmp_path / 'calls.jsonl').read_text().splitlines()]
    assert {x['outcome'] for x in lines} == {'error'} and all(x['error'].startswith('HTTP 503') for x in lines)
