#!/usr/bin/env python3
"""Replay Terminus-2's JSON parser on every agent turn of a TB2 job, with and without the inline think span.

Question (2026-09-25): a thinking model served with no reasoning parser (Grug 09-21, skip_special_tokens=false)
returns <|start_think|>...<|end_think|> inline in content, and Terminus-2 parses the whole reply. Its brace scanner
takes the first balanced {...} anywhere in the text, so a brace pair (or a stray double quote) inside the thinking can
make an otherwise valid action fail. This counts those turns.

    python brace_in_thinking.py <terminus_json_plain_parser.py> <tb2 job dir> [--dump out.jsonl]

The parser file must be the one the eval ran (harbor pin). Only parse-error turns keep the raw reply in the trajectory
(successful turns store "Analysis: ...\\nPlan: ..."), so successful turns are counted as "parses raw" and not replayed.

Two ways to remove the thinking:
  all   = drop everything up to and including the LAST <|end_think|> (what a reasoning parser hands back as content;
          09-21 sometimes closes a span it never opened). A <|start_think|> after the last close = ended in thinking.
  lead  = strip_leading_think_span from harbor branch lukedhlee/terminus2-think-parity (4146b5fc), verbatim: one
          leading span, cut at the FIRST close marker; anything else passes through unchanged.
"""
import argparse
import collections
import glob
import importlib.util
import json
import os

START, END = '<|start_think|>', '<|end_think|>'
THINK_SPAN_MARKERS = ((START, END), ('<think>', '</think>'))


def strip_leading_think_span(response):  # verbatim from harbor 4146b5fc terminus_2.py
    stripped = response.lstrip()
    for open_marker, close_marker in THINK_SPAN_MARKERS:
        if not stripped.startswith(open_marker):
            continue
        close_index = stripped.find(close_marker, len(open_marker))
        if close_index == -1:
            return response
        return stripped[close_index + len(close_marker):]
    return response


def strip_all(content):
    """(body, ended_in_thinking)."""
    ls, le = content.rfind(START), content.rfind(END)
    if ls != -1 and ls > le:
        return '', True
    if le == -1:
        return content, False
    return content[le + len(END):], False


def json_start(text):
    """Index where the pin's _extract_json_content starts its object (same scan), or -1."""
    depth, in_str, esc = 0, False, False
    start = -1
    for i, ch in enumerate(text):
        if esc:
            esc = False
            continue
        if ch == '\\':
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if not in_str:
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start != -1:
                    return start
    return -1


def err_kind(err):
    if not err:
        return 'ok'
    for key, name in (('No valid JSON found', 'no_json_object'), ('Missing required fields', 'missing_fields'),
                      ('Invalid JSON', 'invalid_json'), ("Field 'commands'", 'commands_not_list'),
                      ('Command ', 'bad_command'), ('must be a JSON object', 'not_object')):
        if key in err:
            return name
    return 'other'


def fail_cause(body, ended_in_thinking, err, completion_tokens):
    """Why a reply still fails with the thinking removed."""
    if ended_in_thinking:
        return 'ended_in_thinking'
    if completion_tokens is not None and completion_tokens >= 8000:
        return 'truncated_or_overflow'
    if '<tool_call>' in body or '<function=' in body:
        return 'tool_call_wrapper'
    if not body.strip():
        return 'empty_after_thinking'
    k = err_kind(err)
    if k == 'no_json_object':
        return 'unterminated_json' if '{' in body else 'prose_no_json'
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('parser')
    ap.add_argument('job')
    ap.add_argument('--dump')
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location('p', a.parser)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    P = m.TerminusJSONPlainParser()
    parse = lambda s: P.parse_response(s).error  # noqa: E731

    cls, causes, mech, lead_cls = (collections.Counter() for _ in range(4))
    repro = collections.Counter()
    episodes = []
    dump = open(a.dump, 'w') if a.dump else None
    for traj in sorted(glob.glob(os.path.join(a.job, '*/attempts/*/agent/trajectory.json'))):
        trial = traj.split(os.sep)[-5]
        res = json.load(open(os.path.join(os.path.dirname(traj), '..', '..', '..', 'result.json')))
        reward = ((res.get('verifier_result') or {}).get('rewards') or {}).get('reward')
        seq = []
        for s in json.load(open(traj))['steps']:
            if s.get('source') != 'agent' or s.get('message', '').startswith('Performed context summarization'):
                continue
            obs = json.dumps(s.get('observation'))
            if 'Previous response had parsing errors' not in obs:
                cls['parses_raw'] += 1
                seq.append('ok')
                continue
            raw = s.get('message') or ''
            ctok = (s.get('metrics') or {}).get('completion_tokens')
            e_raw = parse(raw)
            logged = ''.join(r.get('content') or '' for r in (s.get('observation') or {}).get('results', []))
            repro['same_error' if e_raw and f'ERROR: {e_raw}\n' in logged + '\n' else
                  ('different_error' if e_raw else 'raw_parses_now')] += 1
            body, eit = strip_all(raw)
            e_all = 'ended in thinking' if eit else parse(body)
            e_lead = parse(strip_leading_think_span(raw))
            has_think = START in raw or END in raw
            if not e_all:
                c = 'false_error_thinking'
                le = raw.rfind(END)
                js = json_start(raw)
                mech['brace_in_thinking' if 0 <= js < le else
                     'quote_parity_in_thinking' if raw[:le].count('"') % 2 else 'other'] += 1
            else:
                c = 'fails_both'
                causes[fail_cause(body, eit, e_all, ctok)] += 1
            cls[c] += 1
            if c == 'false_error_thinking':
                lead_cls['fixed_by_lead_strip' if not e_lead else 'not_fixed_by_lead_strip'] += 1
            seq.append('false' if c == 'false_error_thinking' else 'err')
            if dump:
                dump.write(json.dumps(dict(trial=trial, step=s.get('step_id'), cls=c, has_think=has_think,
                                           err_raw=e_raw[:120], err_all=(e_all or '')[:120], err_lead=(e_lead or '')[:120],
                                           completion_tokens=ctok, raw=raw)) + '\n')
        # streaks of consecutive parse errors (false or real)
        streaks, cur, fstreak = [], 0, []
        for i, x in enumerate(seq):
            if x == 'ok':
                if cur:
                    streaks.append(cur)
                cur = 0
            else:
                cur += 1
        if cur:
            streaks.append(cur)
        ended_in_error = bool(seq) and seq[-1] != 'ok'
        tail = 0
        for x in reversed(seq):
            if x == 'ok':
                break
            tail += 1
        # false errors that break a clean run (previous turn parsed) = would-be loop starters
        starters = sum(1 for i, x in enumerate(seq) if x == 'false' and (i == 0 or seq[i - 1] == 'ok'))
        episodes.append(dict(trial=trial, reward=reward, turns=len(seq), parse_err=sum(x != 'ok' for x in seq),
                             false=seq.count('false'), false_starters=starters, longest_err_streak=max(streaks or [0]),
                             ended_in_error_streak=ended_in_error, tail_err_streak=tail))
    n = sum(cls.values())
    nerr = cls['false_error_thinking'] + cls['fails_both']
    pct = lambda x, d: round(100 * x / d, 2) if d else None  # noqa: E731
    aff = [e for e in episodes if e['false']]
    out = dict(
        turns=n, parse_error_turns=nerr, parse_error_pct=pct(nerr, n),
        classes={k: dict(n=v, pct_turns=pct(v, n), pct_errors=pct(v, nerr) if k != 'parses_raw' else None)
                 for k, v in cls.items()},
        false_error_mechanism=dict(mech), think_parity_lead_strip=dict(lead_cls),
        fails_both_causes={k: dict(n=v, pct_errors=pct(v, nerr)) for k, v in causes.most_common()},
        logged_error_reproduced=dict(repro),
        episodes=dict(
            total=len(episodes), with_false_errors=len(aff),
            passed=sum(1 for e in episodes if e['reward'] == 1.0),
            with_false_errors_passed=sum(1 for e in aff if e['reward'] == 1.0),
            failing_with_false_errors=sum(1 for e in aff if e['reward'] != 1.0),
            false_in_passing=sum(e['false'] for e in aff if e['reward'] == 1.0),
            false_in_failing=sum(e['false'] for e in aff if e['reward'] != 1.0),
            loop_starters=sum(e['false_starters'] for e in episodes),
        ),
        per_episode=sorted(episodes, key=lambda e: -e['false']),
    )
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
