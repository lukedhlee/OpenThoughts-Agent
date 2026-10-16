#!/usr/bin/env python3
"""What the previous action did, normalized per harness: interventions.ActionResult, which a request-time trigger reads
as Event.last_result ("the teacher takes over when the action output has an error": action_error, action_ok).

One copy for the online router (relay_router.request_event) and the offline replay (data/relay/triggers/replay.py), so
a rule tested on recorded episodes judges the same thing live. Both call it on harbor's history for a request, once per
request: a retry is the same request, so it keeps the result of the first attempt.

    terminus2(messages, request_kind, prev_screen)   # the previous reply's keystrokes and the new terminal output
    msa(messages)                                     # the previous turn's bash commands and their tool results
"""
import json
import os
import re
import sys
from typing import cast

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'triggers'))
import relay_triggers as rt  # noqa: E402  # pyright: ignore[reportMissingImports] - on sys.path above
sys.path.insert(0, HERE)
import interventions as iv  # noqa: E402
import msa_tool as mt  # noqa: E402

T2_NOT_RUN = ('Previous response had parsing errors', 'NONE of the actions you just requested were performed')
T2_LIMIT_MARK = '[... output limited to'          # Terminus-2 cut the new terminal output
TIMED_OUT_RE = re.compile(r'timed out', re.I)


def text_of(content):
    if isinstance(content, list):
        return '\n'.join(p.get('text', '') for p in content if isinstance(p, dict))
    return content or ''


def terminus2(H, kind, prev_screen):
    """Terminus-2: the previous reply's keystrokes and the new terminal output they produced (what the scanner reads:
    the screen part after the last header, minus what the previous screen already showed; prev_screen is the scanner's
    before this request was fed). No exit code (the screen carries none); truncated when Terminus-2 cut the output.
    None when nothing ran (the first request, a parse-error re-prompt, a reply cut at the output limit, the request
    after summarization)."""
    if kind in ('initial', 'handoff') or len(H) < 2 or H[-2].get('role') != 'assistant':
        return None
    obs = text_of(H[-1].get('content'))
    if any(m in obs for m in T2_NOT_RUN):
        return None
    r = rt.parse_reply(text_of(H[-2].get('content')), 'terminus2')
    if not r['ok']:
        return None
    cmds = [c for c, _ in cast(list, r['cmds'])]
    k, scr = rt.screen_part(obs)
    out = rt.new_output(k, scr, cmds, prev_screen) if k else ''
    return iv.ActionResult(command='\n'.join(c.rstrip('\n') for c in cmds if c.strip()), output=out,
                           truncated=T2_LIMIT_MARK in out)


def msa(H):
    """mini-swe-agent: the previous turn's bash commands (one per line) and their tool results: exit_code the first
    non-zero returncode (else 0), output the outputs joined (head and tail when mini-swe-agent elided the middle:
    truncated), timed_out when the harness reports the command timed out. None when the request ends with a user
    message (the first request, a format-error re-prompt)."""
    if not H or H[-1].get('role') != 'tool':
        return None
    i = len(H)
    while i > 0 and H[i - 1].get('role') == 'tool':
        i -= 1
    a = H[i - 1] if i > 0 else {}
    cmds = mt.commands(a.get('tool_calls')) if a.get('role') == 'assistant' else []
    rcs, outs, timed_out, truncated = [], [], False, False
    for m in H[i:]:
        c = text_of(m.get('content'))
        try:
            j = json.loads(c)
        except ValueError:
            j = None
        if isinstance(j, dict):
            rc = j.get('returncode')
            if 'output' in j:
                o = str(j['output'])
            else:
                o = '\n'.join(str(j[k]) for k in ('output_head', 'output_tail') if k in j)
            truncated = truncated or 'elided_chars' in j or 'output_head' in j
            timed_out = timed_out or bool(TIMED_OUT_RE.search(str(j.get('exception_info') or '')))
        else:
            mm = re.search(r'<returncode>(-?\d+)</returncode>', c)
            rc, o = (int(mm.group(1)) if mm else None), c
        rcs.append(rc if isinstance(rc, int) else None)
        outs.append(o)
    code = next((x for x in rcs if x not in (None, 0)), 0 if any(x is not None for x in rcs) else None)
    return iv.ActionResult(command='\n'.join(cmds), exit_code=code, timed_out=timed_out, output='\n'.join(outs),
                           truncated=truncated)
