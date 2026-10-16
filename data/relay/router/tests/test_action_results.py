"""CPU tests for action_results.py: the previous action's result per harness (Event.last_result), as the router and
the offline replay both build it.

    python -m pytest -q data/relay/router/tests/test_action_results.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import action_results as ar  # noqa: E402
import interventions as iv  # noqa: E402


def t2(cmds=(), done=False):
    return json.dumps(dict(analysis='a', plan='p', task_complete=done,
                           commands=[dict(keystrokes=k, duration=0.1) for k in cmds]))


TASK = {'role': 'user', 'content': 'Task Description:\nfix it\n\nCurrent terminal state:\nroot@h:/app# '}


def test_terminus2_new_output_and_command():
    H = [TASK, {'role': 'assistant', 'content': t2(['python run.py\n'])},
         {'role': 'user', 'content': 'New Terminal Output:\n\nroot@h:/app# python run.py\nTraceback (most recent call '
                                     'last):\nNameError: x\nroot@h:/app# '}]
    r = ar.terminus2(H, 'main', '')
    assert (r.command, r.exit_code, r.timed_out, r.truncated) == ('python run.py', None, False, False)
    assert 'Traceback' in r.output and r.output.startswith('root@h:/app# python run.py')


def test_terminus2_nothing_ran():
    assert ar.terminus2([TASK], 'initial', '') is None
    rejected = [TASK, {'role': 'assistant', 'content': '{"analysis": '},
                {'role': 'user', 'content': 'Previous response had parsing errors:\nERROR: Invalid JSON'}]
    assert ar.terminus2(rejected, 'main', '') is None
    after_summary = [TASK, {'role': 'assistant', 'content': t2(['ls\n'])}, {'role': 'user', 'content': 'x'}]
    assert ar.terminus2(after_summary, 'handoff', '') is None


def test_terminus2_screen_is_cut_against_the_previous_one():
    """A confirmation request shows the whole visible screen: only what came after the previous screen is new."""
    prev = 'root@h:/app# ls\nrun.py\nroot@h:/app# '
    H = [TASK, {'role': 'assistant', 'content': t2(['cat out.txt\n'], done=True)},
         {'role': 'user', 'content': 'Current terminal state:\nroot@h:/app# ls\nrun.py\nroot@h:/app# cat out.txt\n'
                                     'ok\nroot@h:/app#\n\nAre you sure you want to mark the task as complete?'}]
    assert ar.terminus2(H, 'confirm', prev).output == 'root@h:/app# cat out.txt\nok\nroot@h:/app#'
    limited = [TASK, {'role': 'assistant', 'content': t2(['yes\n'])},
               {'role': 'user', 'content': 'New Terminal Output:\n\ny\ny\n[... output limited to 10000 bytes ...]'}]
    assert ar.terminus2(limited, 'main', '').truncated


def tool(rc, out, **extra):
    return {'role': 'tool', 'tool_call_id': 'c', 'name': 'bash',
            'content': json.dumps(dict({'returncode': rc, 'output': out}, **extra))}


def call(*cmds):
    return {'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': f'c{j}', 'type': 'function', 'function': {'name': 'bash', 'arguments': json.dumps({'command': c})}}
        for j, c in enumerate(cmds)]}


SYS = [{'role': 'system', 'content': 'sys'}, {'role': 'user', 'content': 'task'}]


def test_msa_exit_code_output_and_commands():
    r = ar.msa(SYS + [call('ls', 'make'), tool(0, 'a b'), tool(2, 'make: *** [all] Error 2')])
    assert r == iv.ActionResult(command='ls\nmake', exit_code=2, output='a b\nmake: *** [all] Error 2')
    assert ar.msa(SYS + [call('ls'), tool(0, 'a')]).exit_code == 0


def test_msa_timeouts_elided_output_and_the_xml_form():
    t = ar.msa(SYS + [call('sleep 99'), tool(-1, '', exception_info='The command timed out after 30 seconds')])
    assert t.timed_out and t.exit_code == -1
    e = ar.msa(SYS + [call('cat big'), {'role': 'tool', 'content': json.dumps(
        {'returncode': 0, 'output_head': 'head', 'output_tail': 'tail', 'elided_chars': 9000})}])
    assert e.truncated and e.output == 'head\ntail'
    x = ar.msa(SYS + [call('false'), {'role': 'tool', 'content': '<returncode>1</returncode>\n<output>\n</output>'}])
    assert x.exit_code == 1


def test_msa_nothing_ran():
    assert ar.msa(SYS) is None
    assert ar.msa(SYS + [{'role': 'assistant', 'content': 'prose'},
                         {'role': 'user', 'content': 'Tool call error: no tool calls'}]) is None
