"""The claim judge for the router's --judge-gate: Qwen's "is the issue actually fixed?" on the student's first claim.

The prompt is opd_ref_judge.py's `claim` mode, byte for byte. The pure functions it needs are copied verbatim below
from data/opd_ref/opd_ref_judge.py (DONE_RE, SYSTEM, MAX_CHARS, issue_of, cut, render, session_steps, fit,
claim_item), data/opd_ref/opd_ref_branch.py (ISSUE_RE) and data/opd_ref/opd_ref_probe.py (START_THINK, END_THINK,
split_reply), so the router stays stdlib + aiohttp (those modules import numpy and each other through sys.path).
tests/test_relay_router.py checks that claim_item here and there give the same prompt.

The score is opd_ref_judge.cmd_score's: log P(Yes) - log P(No) over the first answer token's top-20 logprobs, each
side clipped to [-40, 0]; neither word in the top 20 -> nan (never "low").
"""
import hashlib
import math
import re

# ---- verbatim from data/opd_ref/opd_ref_probe.py ------------------------------------------------------------------
START_THINK, END_THINK = '<|start_think|>', '<|end_think|>'


def split_reply(c):
    """Assistant content (trimmed) -> [(kind, text)], markers excluded. kind = think | action."""
    if c.startswith(START_THINK) and END_THINK in c:
        body, _, rest = c[len(START_THINK):].partition(END_THINK)
        return [('marker', START_THINK), ('think', body), ('marker', END_THINK), ('action', rest)]
    return [('action', c)]


# ---- verbatim from data/opd_ref/opd_ref_branch.py -----------------------------------------------------------------
ISSUE_RE = re.compile(r'<issue_description>.*?</issue_description>', re.S)

# ---- verbatim from data/opd_ref/opd_ref_judge.py ------------------------------------------------------------------
DONE_RE = re.compile(r'"task_complete"\s*:\s*true')
SYSTEM = 'You are an expert software engineer who reviews the work of autonomous coding agents.'
MAX_CHARS = 180_000


def issue_of(prompt):
    m = ISSUE_RE.search(prompt)
    if m:
        return m.group(0)
    a, b = prompt.find('Task Description:'), prompt.find('Current terminal state:')
    return prompt[a:b].strip() if a >= 0 and b > a else prompt[:6000]


def cut(s, head=1500, tail=2500):
    return s if len(s) <= head + tail + 50 else s[:head] + '\n[... %d characters omitted ...]\n' % (
        len(s) - head - tail) + s[-tail:]


def render(user):
    return ('<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n'
            '<|im_start|>assistant\n<think>\n\n</think>\n\n' % (SYSTEM, user))


def session_steps(msgs, upto=None):
    """-> (step texts up to the first claim or the first `upto` turns, claimed?)"""
    steps = []
    for j in range(1, len(msgs), 2):
        act = ''.join(t for kd, t in split_reply(msgs[j]['content'].strip()) if kd == 'action').strip()
        claimed = bool(DONE_RE.search(act))
        s = '[step %d%s]\n%s' % (len(steps) + 1, ' - the agent declares the task complete' if claimed else '', act)
        if not claimed and j + 1 < len(msgs):
            s += '\n[terminal output]\n' + cut(msgs[j + 1]['content'])
        steps.append(s)
        if claimed or (upto is not None and len(steps) == upto):
            return steps, claimed
    return steps, False


def fit(steps):
    body = '\n\n'.join(steps)
    if len(body) > MAX_CHARS:            # keep the first step and the latest ones
        keep, n = [steps[0]], len(steps[0])
        tail = []
        for s in reversed(steps[1:]):
            if n + len(s) > MAX_CHARS:
                break
            tail.append(s)
            n += len(s)
        body = '\n\n'.join(keep + ['[... %d earlier steps omitted ...]' % (len(steps) - 1 - len(tail))] + tail[::-1])
    return body


def claim_item(tr):
    msgs = tr['messages']
    steps, claimed = session_steps(msgs)
    if not claimed:
        return None
    body = fit(steps)
    return render('A coding agent worked on the task below in a Linux terminal, in a checkout of the repository at '
                  '/testbed, and has now declared the task complete. Hidden tests will check whether the issue is '
                  'actually fixed.\n\n<task>\n%s\n</task>\n\n<session>\n%s\n</session>\n\nBased on the session, is '
                  'the issue actually fixed, so that the hidden tests will pass? Answer with one word, Yes or No.'
                  % (issue_of(msgs[0]['content']), body))


# ---- router side --------------------------------------------------------------------------------------------------
def _text(content):
    if isinstance(content, list):
        return '\n'.join(p.get('text', '') for p in content if isinstance(p, dict))
    return content or ''


def dump_messages(messages):
    """A Terminus-2 chat request's history (the confirmation message already dropped) -> the dump shape claim_item
    was written for: [user prompt, assistant, user, ..., assistant (the claim)], plain-string contents. Leading system
    messages are dropped when a user message follows them (else they become the user prompt); consecutive messages of
    one role are joined by a blank line so the turns alternate. Returns (messages, notes) or (None, reason)."""
    ms = [dict(role=m.get('role'), content=_text(m.get('content'))) for m in messages]
    notes = []
    while ms and ms[0]['role'] == 'system':
        if len(ms) > 1 and ms[1]['role'] == 'user':
            ms.pop(0)
            notes.append('dropped_system')
        else:
            ms[0] = dict(ms[0], role='user')
            notes.append('system_as_user')
    out = []
    for m in ms:
        if m['role'] not in ('user', 'assistant'):
            notes.append('dropped_%s' % m['role'])
            continue
        if out and out[-1]['role'] == m['role']:
            out[-1] = dict(out[-1], content=out[-1]['content'] + '\n\n' + m['content'])
            notes.append('merged_%s' % m['role'])
            continue
        out.append(m)
    if not out or out[0]['role'] != 'user' or out[-1]['role'] != 'assistant':
        return None, 'history does not run user ... assistant'
    return out, notes


def claim_prompt(messages):
    """-> (prompt, notes) or (None, reason). `messages` = the request up to and including the claim."""
    ms, notes = dump_messages(messages)
    if ms is None:
        return None, notes
    pr = claim_item({'messages': ms})
    if pr is None:
        return None, 'no task_complete: true in the history'
    return pr, notes


def _logaddexp(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def score(top):
    """top = {token: logprob} of the first generated token -> (logit, p_yes, yes, no) as in opd_ref_judge.cmd_score."""
    yes = no = -math.inf
    for tok, lp in (top or {}).items():
        w = (tok or '').strip().lower()
        if w == 'yes':
            yes = _logaddexp(yes, float(lp))
        elif w == 'no':
            no = _logaddexp(no, float(lp))
    if yes == -math.inf and no == -math.inf:
        return math.nan, math.nan, yes, no
    clip = lambda x: min(max(x, -40.0), 0.0)  # noqa: E731
    logit = clip(yes) - clip(no)
    return logit, 1.0 / (1.0 + math.exp(-logit)), yes, no


def arm_of(seed, sid, treat_frac):
    """'treat' | 'control', a fixed function of (seed, session id)."""
    u = int(hashlib.sha256((seed + sid).encode('utf-8', 'surrogatepass')).hexdigest(), 16) / 2 ** 256
    return 'treat' if u < treat_frac else 'control'
