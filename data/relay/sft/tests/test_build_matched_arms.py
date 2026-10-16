"""build_matched_arms.py --relay-only: one arm's eligible rows, the matched arms' filters, into one rows file that
relay_rows_to_parquet.py accepts (plain concatenation of the rendered files does not).

    python -m pytest -q data/relay/sft/tests/test_build_matched_arms.py
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

SFT = Path(__file__).resolve().parent.parent
CENSUS = dict(weak_timeout=False, verifier_ran=True, leak_turns=0, hunt_turns=0, canary=False)
# sid: (census overrides, or None for no census row; rendered row overrides)
ROWS = {
    'ok1': ({'task': 'a', 'passed': True}, {}),
    'ok2': ({'task': 'a', 'passed': False}, {}),
    'ok3': ({'task': 'b', 'passed': True}, {}),
    'nocensus': (None, {}),
    'weak': ({'weak_timeout': True}, {}),
    'noverifier': ({'verifier_ran': False}, {}),
    'toolong': ({}, {'fits': False}),
    'untrained': ({}, {'trained': False}),
    'leak': ({'leak_turns': 2}, {}),
    'hunt': ({'hunt_turns': 1}, {}),
    'canary': ({'canary': True}, {}),
}
FILES = {'run1.rendered.jsonl': ['ok1', 'nocensus', 'weak', 'toolong', 'leak', 'ok2'],
         'run2.rendered.jsonl': ['noverifier', 'ok3', 'untrained', 'hunt', 'canary']}


def rendered(sid, fits=True, trained=True, n=12):
    return json.dumps(dict(sid=sid, n_tokens=n, fits=fits, ids=[128000] + [5] * (n - 1),
                           loss=[0] + [int(trained)] * (n - 1), turns=1)) + '\n'


@pytest.fixture
def arm(tmp_path):
    with open(tmp_path / 'census.jsonl', 'w') as f:
        for sid, (c, _) in ROWS.items():
            if c is not None:
                f.write(json.dumps({**CENSUS, 'sid': sid, 'task': 't', 'passed': False, **c}) + '\n')
    (tmp_path / 'rendered').mkdir()
    for name, sids in FILES.items():
        (tmp_path / 'rendered' / name).write_text(''.join(rendered(s, **ROWS[s][1]) for s in sids))
    return tmp_path


def run(*argv, check=True):
    return subprocess.run([sys.executable, *map(str, argv)], capture_output=True, text=True, check=check)


def test_relay_only_keeps_the_eligible_rows_and_converts(arm):
    out = arm / 'arms' / 'one_rows.jsonl'
    run(SFT / 'build_matched_arms.py', '--relay-only', '--relay-census', arm / 'census.jsonl',
        '--relay-rendered', arm / 'rendered', '--out', out)
    assert [json.loads(l)['sid'] for l in out.read_text().splitlines()] == ['ok1', 'ok2', 'ok3']   # file, line order
    rep = json.loads((arm / 'arms' / 'one_rows.json').read_text())
    assert rep['rows'] == 3 and rep['trained'] == 33 and rep['pool'] == dict(
        rendered=11, no_census=1, weak_timeout_or_no_verifier=2, over_64k=1, no_trained_token=1, tag_leak=1,
        tag_hunt=1, tag_canary=1, dropped_tagged=3, eligible=3)
    run(SFT / 'relay_rows_to_parquet.py', out, arm / 'pq')
    assert (arm / 'pq' / 'train-00000-of-00001.parquet').exists()
    cat = arm / 'cat.jsonl'                     # what the README warns against: the rendered files concatenated
    cat.write_text(''.join(p.read_text() for p in sorted((arm / 'rendered').glob('*.rendered.jsonl'))))
    assert run(SFT / 'relay_rows_to_parquet.py', cat, arm / 'pq_cat', check=False).returncode != 0


def test_the_matched_arms_still_need_their_inputs(arm):
    r = run(SFT / 'build_matched_arms.py', '--relay-census', arm / 'census.jsonl', '--relay-rendered',
            arm / 'rendered', '--out-dir', arm / 'arms', check=False)
    assert r.returncode == 2 and '--qwen-census, --qwen-rendered, --kimi-from' in r.stderr
    r = run(SFT / 'build_matched_arms.py', '--relay-only', '--relay-census', arm / 'census.jsonl', '--relay-rendered',
            arm / 'rendered', check=False)
    assert r.returncode == 2 and '--relay-only needs --out' in r.stderr
