"""Load labelled transcripts from each source into one episode record:
    dict(id, source, population, bench, model, task, reward, exc, infra, outcome, harness, messages, timestamps,
         budget_sec, path)
messages are OpenAI-style role/content dicts, as the router would see them (assistant content = the Terminus-2 JSON
reply, user content = the observation text Harbor sent back).
"""
import datetime as _dt
import json
import os

INFRA_EXC = {'APIConnectionError', 'TmuxBatchProtocolError', 'TmuxCommandError', 'TmuxSessionEndedError',
             'VerifierTimeoutError', 'noresult', 'EnvironmentStartTimeoutError', 'DaytonaBadGatewayError',
             'DaytonaSandboxStopError', 'VerifierRuntimeError', 'AddTestsDirError'}


def _ts(s):
    try:
        return _dt.datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()
    except Exception:
        return None


def _obs_text(step):
    obs = step.get('observation') or {}
    return '\n'.join(r['content'] for r in obs.get('results', []) if isinstance(r.get('content'), str))


def atif_to_messages(traj):
    """Harbor ATIF trajectory -> (messages, timestamps as seconds since the first step)."""
    steps = traj['steps']
    msgs, tss = [], []
    t0 = None
    for s in steps:
        ts = _ts(s.get('timestamp') or '')
        if t0 is None and ts is not None:
            t0 = ts
        rel = (ts - t0) if (ts is not None and t0 is not None) else None
        src = s.get('source')
        if src == 'agent':
            m = s.get('message') or ''
            tcs = s.get('tool_calls') or []
            cmds = [dict(keystrokes=tc['arguments'].get('keystrokes', ''), duration=tc['arguments'].get('duration', 1.0))
                    for tc in tcs if tc.get('function_name') == 'bash_command' and isinstance(tc.get('arguments'), dict)]
            mark = any(tc.get('function_name') == 'mark_task_complete' for tc in tcs)
            if m.startswith('Analysis:'):
                a, _, p = m[len('Analysis:'):].partition('\nPlan:')
                content = json.dumps(dict(analysis=a.strip(), plan=p.strip(), commands=cmds, task_complete=mark))
            else:
                content = m   # unparsed raw reply (the harness answered with a parsing error)
            msgs.append(dict(role='assistant', content=content)); tss.append(rel)
            msgs.append(dict(role='user', content=_obs_text(s))); tss.append(rel)
        elif src in ('user', 'system'):
            msgs.append(dict(role='user' if src == 'user' else 'system', content=s.get('message') or ''))
            tss.append(rel)
    return msgs, tss


def outcome_of(reward, exc):
    infra = reward is None or (exc or '') in INFRA_EXC
    return 'infra' if infra else ('pass' if (reward or 0) > 0 else 'fail')


def load_atif(path, meta):
    try:
        traj = json.load(open(path))
    except Exception:
        return None
    msgs, tss = atif_to_messages(traj)
    rec = dict(meta)
    rec.setdefault('harness', 'terminus2')
    rec['messages'] = msgs
    rec['timestamps'] = tss
    rj = os.path.join(os.path.dirname(path), '..', '..', '..', 'result.json')
    if not os.path.exists(rj):
        rj = os.path.join(os.path.dirname(path), '..', 'result.json')
    budget = None
    try:
        r = json.load(open(rj))
        budget = ((r.get('config') or {}).get('resolved_timeouts') or {}).get('agent')
    except Exception:
        pass
    rec['budget_sec'] = budget
    rec['outcome'] = outcome_of(rec.get('reward'), rec.get('exc'))
    return rec


def load_messages_record(d):
    rec = dict(d)
    rec.setdefault('timestamps', None)
    rec.setdefault('budget_sec', None)
    rec['outcome'] = outcome_of(rec.get('reward'), rec.get('exc'))
    return rec
