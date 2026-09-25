"""Unit tests for relay_triggers: real positive/negative fixtures, causality, and the new-output-only rule.

Run: python -m pytest data/relay/triggers/tests -q   (or python data/relay/triggers/tests/test_relay_triggers.py)
"""
import gzip
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import relay_triggers as rt  # noqa: E402

ALL = ['done_claim', 'gave_up', 'no_progress_wait', 'loop', 'edit_failed', 'error_streak', 'submit_check',
       'destructive', 'success_contradicted', 'success_claim_unchecked']


def _load(name):
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        return []
    with gzip.open(p, 'rt') as f:
        return [json.loads(l) for l in f]


def _reply(fx_reply):
    if fx_reply is None:
        return None
    return rt.tool_reply(fx_reply) if fx_reply.get('tool_calls') else fx_reply.get('content')


def T2(analysis='', plan='', cmds=(), done=False):
    return json.dumps(dict(analysis=analysis, plan=plan, task_complete=done,
                           commands=[dict(keystrokes=k, duration=d) for k, d in cmds]))


class RealFixtures(unittest.TestCase):
    """Real transcript windows: calibrated triggers fire on the positives and stay quiet on the negatives."""

    def test_fixtures(self):
        fx = _load('fixtures.jsonl.gz')
        self.assertTrue(fx, 'run tests/build_fixtures.py first')
        seen = set()
        for f in fx:
            cfg = dict(rt.CALIBRATED_CONFIG, enabled=[f['trigger']])
            fires = rt.detect(f['messages'], reply=_reply(f['reply']), config=cfg)
            fired = any(x['trigger'] == f['trigger'] for x in fires)
            with self.subTest(f['name']):
                self.assertEqual(fired, f['expect_fire'], f['note'])
                if fired:
                    self.assertTrue(fires[0]['reason'])
            seen.add((f['trigger'], f['expect_fire']))
        for trig in ('gave_up', 'no_progress_wait', 'loop'):
            self.assertIn((trig, True), seen)
            self.assertIn((trig, False), seen)


class Causality(unittest.TestCase):
    """The router's stateless call on each request prefix must equal the offline full-episode scan at that turn:
    no detector may use anything after the request."""

    def test_prefix_equals_offline(self):
        eps = _load('episodes.jsonl.gz')
        self.assertTrue(eps)
        cfg = dict(rt.CALIBRATED_CONFIG, enabled=ALL)
        for ep in eps:
            msgs = ep['messages']
            off = rt.fires_by_turn(rt.scan_messages(msgs), cfg)
            off_by_t = {}
            for f in off:
                off_by_t.setdefault(f['turn'], set()).add(f['trigger'])
            t = 0
            for i, m in enumerate(msgs):
                if m.get('role') != 'assistant':
                    continue
                t += 1
                live = rt.detect(msgs[:i], reply=_reply(m), config=cfg)
                with self.subTest(ep=ep['id'], turn=t):
                    self.assertEqual({f['trigger'] for f in live}, off_by_t.get(t, set()))


class NewOutputOnly(unittest.TestCase):
    """Old errors left in tmux scrollback must not re-fire (the 'Current Terminal Screen' fallback repeats them)."""

    def _ep(self, screens):
        msgs = [dict(role='user', content='Task Description:\nfix it\n\nCurrent terminal state:\nroot@h:/app# ')]
        for keys, screen in screens:
            msgs.append(dict(role='assistant', content=T2('a', 'p', [(keys, 1.0)])))
            msgs.append(dict(role='user', content=screen))
        return msgs

    def test_scrollback_error_does_not_count(self):
        err = 'Traceback (most recent call last):\nModuleNotFoundError: No module named foo'
        s1 = 'New Terminal Output:\nroot@h:/app# python run.py\n' + err + '\nroot@h:/app# '
        # full-screen fallback: the old traceback is still on screen, the new command printed nothing wrong
        s2 = ('Current Terminal Screen:\nroot@h:/app# python run.py\n' + err + '\nroot@h:/app# ls\nrun.py\n'
              'root@h:/app# ')
        s3 = ('Current Terminal Screen:\nroot@h:/app# python run.py\n' + err + '\nroot@h:/app# ls\nrun.py\n'
              'root@h:/app# echo ok\nok\nroot@h:/app# ')
        msgs = self._ep([('python run.py\n', s1), ('ls\n', s2), ('echo ok\n', s3)])
        cfg = dict(rt.CALIBRATED_CONFIG, enabled=['error_streak'], streak_n=2, streak_same_sig=False,
                   streak_no_edit_between=False)
        self.assertEqual(rt.detect(msgs, config=cfg), [])
        sc = rt.scan_messages(msgs)
        self.assertEqual([a['err'] for a in sc.actions], [True, False, False])

    def test_two_new_errors_do_fire(self):
        e1 = 'New Terminal Output:\nroot@h:/app# gcc a.c\na.c:1: error: expected ;\nroot@h:/app# '
        e2 = 'New Terminal Output:\nroot@h:/app# gcc a.c -o a\na.c:1: error: expected ;\nroot@h:/app# '
        msgs = self._ep([('gcc a.c\n', e1), ('gcc a.c -o a\n', e2)])
        cfg = dict(rt.CALIBRATED_CONFIG, enabled=['error_streak'], streak_n=2, streak_same_sig=False,
                   streak_no_edit_between=False)
        self.assertEqual([f['trigger'] for f in rt.detect(msgs, config=cfg)], ['error_streak'])


class Parsing(unittest.TestCase):
    def test_terminus2_with_think(self):
        r = rt.parse_reply('<think>long</think>\n' + T2('x', 'y', [('ls\n', 0.1)], done=True))
        self.assertTrue(r['ok'] and r['done'])
        self.assertEqual(r['cmds'][0][0], 'ls\n')

    def test_miniswe_text_block(self):
        r = rt.parse_reply('THOUGHT: done\n\n```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git diff\n```')
        self.assertEqual(r['harness'], 'miniswe')
        self.assertTrue(r['done'])

    def test_tool_calls(self):
        m = dict(role='assistant', content='Checking.', tool_calls=[
            dict(id='1', type='function', function=dict(name='bash', arguments='{"command": "ls /testbed"}'))])
        r = rt.tool_reply(m)
        self.assertEqual(r['cmds'][0][0].strip(), 'ls /testbed')
        self.assertFalse(r['done'])

    def test_gave_up_is_decision_and_needs_turn_2(self):
        reply = T2('pip is unavailable. We cannot install numpy, so instead we will write it manually.', 'x',
                   [('ls\n', 0.1)])
        first = [dict(role='user', content='Task Description:\nt\n\nCurrent terminal state:\n$ ')]
        cfg = dict(rt.CALIBRATED_CONFIG, enabled=['gave_up'])        # logged-only by default; enable to test it
        self.assertEqual(rt.detect(first, reply=reply, config=cfg), [])          # never before turn 2
        hist = first + [dict(role='assistant', content=T2('a', 'b', [('ls\n', 0.1)])),
                        dict(role='user', content='New Terminal Output:\nroot@h:/app# ls\nroot@h:/app# ')]
        self.assertEqual([f['trigger'] for f in rt.detect(hist, reply=reply, config=cfg)], ['gave_up'])
        self.assertEqual(rt.detect(hist, reply=reply), [])          # not a takeover trigger in DEFAULT_CONFIG
        logged = rt.detect(hist, reply=reply, include_logged=True)
        self.assertEqual([(f['trigger'], f['kind']) for f in logged], [('gave_up', 'logged')])

    def test_no_progress_wait(self):
        msgs = [dict(role='user', content='Task Description:\nt\n\nCurrent terminal state:\n$ ')]
        msgs += [dict(role='assistant', content=T2('a', 'b', [('make\n', 5)])),
                 dict(role='user', content='New Terminal Output:\nroot@h:/app# make\nbuilding...\n')]
        for _ in range(3):
            msgs += [dict(role='assistant', content=T2('wait', 'wait', [('', 30)])),
                     dict(role='user', content='New Terminal Output:\n\n')]
        self.assertEqual([f['trigger'] for f in rt.detect(msgs)], ['no_progress_wait'])
        self.assertEqual(rt.detect(msgs[:-2]), [])     # two empty waits are not enough (wait_k = 3)


if __name__ == '__main__':
    unittest.main()
