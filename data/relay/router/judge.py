#!/usr/bin/env python3
"""LLM judge trigger: a model reads the student's recent turns and scores whether the teacher should step in.

Stage 4 of notes/ot-agent/relay-triggers (stage4_judge_scope.md), the offline half: the trigger, its view hygiene, its
cost bounds and latency accounting, usable in replay now and by the router once stage 3 wires the hooks below.

    triggers:
      - use: judge:LlmJudge          # this file (router/ is on sys.path in the router and in replay); after
                                     # judge.register(), also the built-in `llm_judge`
        phases: [request]            # request (default): judge the result of the last action; reply: judge the
                                     # proposed action before the harness sees it (where: rewrite discards it)
        span: 2
        params:
          endpoint: ${JUDGE_URL}     # a URL (http://host:8000/v1), or an endpoint name the router resolves
          model: judge
          prompt: |                  # string.Template: $task $history $last_result $reply $proposal $turn $phase
            ...                      # (or prompt_file: path; default DEFAULT_PROMPT)
          output: score              # score: JSON {"score": x, "rationale": ...}, fires at score >= threshold
          threshold: 0.7             # verdict: JSON {"verdict": ...} or one word, fires on a word in fire_on
          timeout_s: 20              # a timeout (or an HTTP error, or unparseable output) is no decision
          every: 2                   # ask on turns divisible by `every`
          max_calls: 15              # asks per episode (a retry of the same request is not a new ask)
          log: judge_calls.jsonl     # one line per ask: outcome, score, latency, tokens

What the judge sees (student_view): only what the student saw. Its input is built from Event.messages (the harness's
history as the student saw it), Event.last_result and Event.reply, never from Event.extra, files or the task's
metadata, so grader files and solutions cannot reach it; the template can only name the view's fields. From those:
reasoning fields are dropped (harbor re-sends the teacher's reasoning on its turns; a judge reads actions and
results), think spans are stripped unless think: keep, and teacher-only text is scrubbed wherever it shows up: the
router's verify notes, Event.extra['teacher_only'] (the router's own note / hint texts) and params scrub.

Cost and latency: every ask is counted in `stats` and, with `log`, written as one JSON line (outcome fired / quiet /
timeout / malformed / error / unresolved, score, latency_s = the call's wall time, waited_s = how long observe()
blocked on it, prompt and completion tokens, cached, prefetched). Identical prompts reuse one answer process-wide
(cache: true), so replay's live and shadow controllers, which see the same events, call the judge once; a timeout or
an error is never cached. A decision's info carries rationale, score or verdict, latency, tokens and the model.

Router hooks (stage 3; offline nothing is needed when endpoint is a URL):
  * endpoint names: Event.extra['post'] = async post(name, path, body) -> (status, body bytes | dict), the router's
    own post() with its pinning and failover (preferred); or Event.extra['endpoints'] = {name: url}; or
    set_resolver(fn(name) -> url) once at start.
  * overlap: prefetch(ev) starts the ask in the background; observe() at the same (phase, turn) awaits it instead of
    asking again. A request-time judge prefetched at finish() of turn t, with the history up to the reply just
    returned, runs alongside the tool call and is awaited at request t + 1 (it then judges the view of the prefetch,
    without that request's result). Judge time should pause the agent clock as model calls do.
  * teacher-only text: Event.extra['teacher_only'] = [the verify note text, the teacher hint text, ...].
  * `use: llm_judge` resolves here once register() has run (interventions.py keeps a stage-4 stub under that name,
    and importing this module changes nothing): the router (and replay) can call judge.register() at start, or
    interventions' stub can import this module lazily. Until then, `use: judge:LlmJudge`.
"""
import asyncio
import collections
import hashlib
import inspect
import json
import os
import re
import string
import sys
import time
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import interventions as iv  # noqa: E402

DEFAULT_PROMPT = """You are reviewing an AI agent that works in a Linux terminal to solve the task below. Decide \
whether the agent is stuck or about to fail, so that a stronger agent should take over now.

Task:
$task

The agent's most recent turns, oldest first:
$history

Result of its last action:
$last_result
$proposal
Reply with JSON only: {"score": <probability from 0 to 1 that the agent fails without help>, \
"rationale": "<one sentence>"}"""

THINK_RE = re.compile(r'<\|start_think\|>.*?<\|end_think\|>|<think>.*?</think>', re.S)
OPEN_THINK_RE = re.compile(r'(?:<\|start_think\|>|<think>).*\Z', re.S)   # an unterminated span runs to the end
FENCE_RE = re.compile(r'```(?:json)?\s*(\{.*?\})\s*```', re.S)
ROLES = ('system', 'user', 'assistant', 'tool')
OUTPUTS = ('score', 'verdict')
_CACHE: 'collections.OrderedDict[str, dict]' = collections.OrderedDict()
CACHE_MAX = 4096
_RESOLVER = None


def set_resolver(fn):
    """fn(name) -> URL for endpoint names (or None to clear): the router's name -> server map."""
    global _RESOLVER
    _RESOLVER = fn


def clear_cache():
    _CACHE.clear()


def _router_notes():
    """The router's teacher-only verify-note texts (relay_router imports aiohttp, as this module does)."""
    try:
        import relay_router as rr  # pyright: ignore[reportMissingImports] - router/ is on sys.path
    except Exception:  # noqa: BLE001 - a judge without the router (a notebook) scrubs what it is given
        return ()
    return tuple(getattr(rr, n) for n in ('VERIFY_NOTE', 'VERIFY_NOTE_OWN', 'VERIFY_NOTE_MSA') if hasattr(rr, n))


# ==== the student's view ======================================================================================
def _text(content):
    if isinstance(content, list):
        return '\n'.join(p.get('text', '') for p in content if isinstance(p, dict))
    return content if isinstance(content, str) else ''


def _cut(s, n):
    """At most n chars: the head and the tail, the middle replaced by a marker (errors are often at the end)."""
    if n is None or len(s) <= n:
        return s
    h = n * 2 // 3
    return f'{s[:h]}\n[... {len(s) - n} chars cut ...]\n{s[len(s) - (n - h):]}'


class _Scrub:
    def __init__(self, texts):
        # longest first, each also with the blank line the router puts before an appended note
        self.texts = sorted({t for t in texts if t and len(t) >= 8}, key=len, reverse=True)

    def __call__(self, s):
        for t in self.texts:
            s = s.replace('\n\n' + t, '').replace(t, '')
        return s


def _tool_calls(m):
    out = []
    for tc in m.get('tool_calls') or []:
        fn = tc.get('function') or {}
        args = fn.get('arguments')
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {'command': args}
        cmd = (args or {}).get('command') if isinstance(args, dict) else None
        out.append(f"[{fn.get('name') or 'tool'}] {cmd if cmd is not None else json.dumps(args)}")
    return out


def _message(m, think, scrub, max_chars):
    """One message as the judge reads it: role and visible content (no reasoning fields), tool calls as commands."""
    c = _text(m.get('content'))
    if m.get('role') == 'assistant' and think == 'strip':
        c = OPEN_THINK_RE.sub('', THINK_RE.sub('', c)).strip()
    parts = [c] + _tool_calls(m)
    return f"[{m.get('role')}]\n" + _cut(scrub('\n'.join(p for p in parts if p)), max_chars)


def _result(r, scrub, max_chars):
    if r is None:
        return '(none)'
    head = [f'$ {r.command.strip()}' if r.command else '(no command)']
    if r.exit_code is not None:
        head.append(f'exit code {r.exit_code}')
    if r.timed_out:
        head.append('timed out')
    if r.truncated:
        head.append('output truncated by the harness')
    return '\n'.join([', '.join(head), _cut(scrub(r.output or ''), max_chars)]).rstrip()


def _reply(reply, think, scrub, max_chars):
    if reply is None:
        return ''
    t = reply.text or ''
    if think == 'strip':
        t = OPEN_THINK_RE.sub('', THINK_RE.sub('', t)).strip()
    a = reply.action
    if isinstance(a, dict) and a.get('cmds'):
        cmds = [str(c[0] if isinstance(c, (list, tuple)) else c).strip() for c in a['cmds']]
        t = (t + '\n' if t else '') + '\n'.join(f'$ {c}' for c in cmds if c)
    if reply.parse_error:
        t += f'\n(the harness cannot parse this reply: {reply.parse_error[:200]})'
    return _cut(scrub(t), max_chars)


def student_view(ev, window=6, think='strip', max_chars=4000, task_chars=4000, scrub=()):
    """The judge's whole input as template fields, built from what the student saw only: Event.messages,
    last_result and reply (Event.extra contributes teacher-only strings to remove, nothing to read). history: the
    last `window` assistant turns with what followed them; task: the first user message (the task statement)."""
    sc = _Scrub(tuple(scrub) + _router_notes() + tuple(ev.extra.get('teacher_only') or ()))
    msgs = [m for m in (ev.messages or ()) if isinstance(m, dict) and m.get('role') in ROLES]
    first_user = next((i for i, m in enumerate(msgs) if m.get('role') == 'user'), None)
    task = _cut(sc(_text(msgs[first_user].get('content'))), task_chars) if first_user is not None else ''
    asst = [i for i, m in enumerate(msgs) if m.get('role') == 'assistant']
    start = asst[-window] if window and len(asst) >= window else (asst[0] if asst else len(msgs))
    hist = '\n\n'.join(_message(m, think, sc, max_chars) for m in msgs[start:]) or '(no turns yet)'
    reply = _reply(ev.reply, think, sc, max_chars) if ev.phase == 'reply' else ''
    return dict(task=task, history=hist, last_result=_result(ev.last_result, sc, max_chars), reply=reply,
                proposal=f'\nThe agent now proposes this action, not yet run:\n{reply}\n' if reply else '',
                turn=str(ev.turn), phase=ev.phase)


# ==== the judge's answer ======================================================================================
def _json_obj(text):
    """The first JSON object in a model's answer: the whole text, a ```json fence, or the first balanced {...}."""
    t = text.strip()
    for cand in [t] + FENCE_RE.findall(t):
        try:
            o = json.loads(cand)
            if isinstance(o, dict):
                return o
        except ValueError:
            pass
    i = t.find('{')
    while i != -1:
        depth, ins, esc = 0, False, False
        for j in range(i, len(t)):
            ch = t[j]
            if ins:
                esc, ins = (False, ins) if esc else (ch == '\\', ch != '"')
                continue
            if ch == '"':
                ins = True
            elif ch in '{}':
                depth += 1 if ch == '{' else -1
                if depth == 0:
                    try:
                        o = json.loads(t[i:j + 1])
                        if isinstance(o, dict):
                            return o
                    except ValueError:
                        pass
                    break
        i = t.find('{', i + 1)
    return None


def parse_answer(text, output='score', score_key='score', verdict_key='verdict', rationale_key='rationale'):
    """(value, rationale, error): value is a float (score) or a lower-case word (verdict); error when unparseable."""
    t = OPEN_THINK_RE.sub('', THINK_RE.sub('', text or '')).strip()
    o = _json_obj(t)
    rationale = str(o.get(rationale_key) or '') if o else ''
    if output == 'score':
        v = o.get(score_key) if o else None
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            try:
                v = float(str(v).strip())
            except ValueError:
                return None, rationale, f'no numeric {score_key!r}'
        return float(v), rationale, None
    v = o.get(verdict_key) if o else (t.split()[0].strip('.,:;!"\'') if t.split() else None)
    if v is None or isinstance(v, (dict, list)):
        return None, rationale, f'no {verdict_key!r}'
    return str(v).strip().lower(), rationale, None


# ==== the trigger =============================================================================================
class LlmJudge:
    """An LLM judge as a trigger (see the module docstring). A judgment that does not fire is no Decision but is still
    counted and logged; so are timeouts, HTTP errors and unparseable answers (no decision)."""
    name = 'llm_judge'
    phases = frozenset({'request'})
    allowed_phases = frozenset(iv.PHASES)

    def __init__(self, endpoint, model, prompt=None, prompt_file=None, system=None, window=6, think='strip',
                 max_chars=4000, task_chars=4000, output='score', threshold=0.5,
                 fire_on=('intervene', 'yes', 'true', 'stuck', 'fail'), score_key='score', verdict_key='verdict',
                 rationale_key='rationale', timeout_s=30.0, every=1, max_calls=None, max_tokens=512,
                 temperature=0.0, extra_body=None, api_key_env=None, scrub=(), log=None, cache=True):
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise iv.ConfigError('endpoint: a URL or an endpoint name')
        if not isinstance(model, str) or not model:
            raise iv.ConfigError('model: the served model name')
        if prompt is not None and prompt_file is not None:
            raise iv.ConfigError('give prompt or prompt_file, not both')
        if prompt_file is not None:
            try:
                prompt = open(os.path.expandvars(prompt_file)).read()
            except OSError as err:
                raise iv.ConfigError(f'prompt_file: {err}') from None
        if output not in OUTPUTS:
            raise iv.ConfigError(f'output must be one of {OUTPUTS}, not {output!r}')
        if think not in ('strip', 'keep'):
            raise iv.ConfigError(f'think must be strip or keep, not {think!r}')
        for k, v, lo in (('window', window, 1), ('every', every, 1), ('max_tokens', max_tokens, 1)):
            if type(v) is not int or v < lo:
                raise iv.ConfigError(f'{k} must be an int >= {lo}, not {v!r}')
        if max_calls is not None and (type(max_calls) is not int or max_calls < 0):
            raise iv.ConfigError(f'max_calls must be an int >= 0 or null, not {max_calls!r}')
        if not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
            raise iv.ConfigError(f'timeout_s must be a positive number, not {timeout_s!r}')
        self.endpoint, self.model = os.path.expandvars(endpoint.strip()), model
        self.prompt = string.Template(prompt if prompt is not None else DEFAULT_PROMPT)
        self.system, self.window, self.think = system, window, think
        self.max_chars, self.task_chars = max_chars, task_chars
        self.output, self.threshold = output, float(threshold)
        self.fire_on = frozenset(str(w).lower() for w in ([fire_on] if isinstance(fire_on, str) else fire_on))
        self.keys = dict(score_key=score_key, verdict_key=verdict_key, rationale_key=rationale_key)
        self.timeout_s, self.every, self.max_calls = float(timeout_s), every, max_calls
        self.sampling = dict(max_tokens=max_tokens, temperature=temperature, **dict(extra_body or {}))
        self.api_key_env, self.scrub, self.cache = api_key_env, tuple(scrub), bool(cache)
        self.log = os.path.expandvars(log) if log else None
        self.reset()

    def reset(self):
        self.stats = collections.Counter()     # asks, calls, cached, fired, quiet, timeout, malformed, error, skipped_*
        self.latencies = []                    # latency_s of every HTTP call made
        self.asks = 0                          # judged requests this episode (cache hits included): max_calls
        self._seen = {}                        # (phase, turn, prompt key) -> answer: a retry is the same ask
        self._inflight = {}                    # (phase, turn) -> (task, start time, key): prefetched asks

    # -- what the judge is asked --
    def render(self, ev):
        """The prompt for this event (the view's fields substituted into the template)."""
        return self.prompt.safe_substitute(student_view(ev, self.window, self.think, self.max_chars,
                                                        self.task_chars, self.scrub))

    def body(self, prompt):
        msgs = ([{'role': 'system', 'content': self.system}] if self.system else []) + \
            [{'role': 'user', 'content': prompt}]
        return dict(self.sampling, model=self.model, messages=msgs)

    def _key(self, body):
        return hashlib.sha1(json.dumps([self.endpoint, body], sort_keys=True).encode()).hexdigest()

    def _due(self, ev):
        """Asked on this event at all: a main request on a turn divisible by `every`."""
        return ev.main and ev.turn % self.every == 0

    # -- the router's early start --
    def prefetch(self, ev):
        """Start the ask for this event's (phase, turn) in the background; observe() at the same (phase, turn) awaits
        it. Needs a running event loop (the router's)."""
        k = (ev.phase, ev.turn)
        if not self._due(ev) or k in self._inflight:
            return
        body = self.body(self.render(ev))
        key = (ev.phase, ev.turn, self._key(body))
        if key in self._seen or (self.max_calls is not None and self.asks >= self.max_calls):
            return
        self.asks += 1
        self._inflight[k] = (asyncio.ensure_future(self._answer(ev, body, key[2])), time.monotonic(), key)

    # -- the trigger --
    async def observe(self, ev):
        if not self._due(ev):
            self.stats['skipped_every'] += 1
            return None
        pre = self._inflight.pop((ev.phase, ev.turn), None)
        ans: dict[str, Any]
        if pre is not None:
            task, _, key = pre
            t0 = time.monotonic()
            ans = dict(await task, prefetched=True)
            ans['waited_s'] = round(time.monotonic() - t0, 4)
            self._seen[key] = ans
        else:
            body = self.body(self.render(ev))
            key = (ev.phase, ev.turn, self._key(body))
            if key in self._seen:
                ans = dict(self._seen[key], repeat=True, waited_s=0.0)   # a retry: the same request, the same answer
                self.stats['repeats'] += 1
            elif self.max_calls is not None and self.asks >= self.max_calls:
                self.stats['skipped_max_calls'] += 1
                return None
            else:
                self.asks += 1
                t0 = time.monotonic()
                ans = await self._answer(ev, body, key[2])
                ans['waited_s'] = round(time.monotonic() - t0, 4)
                self._seen[key] = ans
        fire = ans['outcome'] == 'ok' and (ans['value'] >= self.threshold if self.output == 'score'
                                           else ans['value'] in self.fire_on)
        outcome = 'fired' if fire else ('quiet' if ans['outcome'] == 'ok' else ans['outcome'])
        if not ans.get('repeat'):
            self.stats[outcome] += 1
            self._write(ev, ans, outcome)
        if not fire:
            return None
        v = ans['value']
        info = {k: ans.get(k) for k in ('rationale', 'latency_s', 'waited_s', 'prompt_tokens', 'completion_tokens',
                                        'cached', 'prefetched')}
        info.update(model=self.model, raw=ans.get('raw', '')[:500], **({'score': v} if self.output == 'score'
                                                                      else {'verdict': v}))
        what = f'{v:.2f} >= {self.threshold:g}' if self.output == 'score' else repr(v)
        return iv.Decision(self.name, f"judge {what}: {ans.get('rationale') or 'no rationale'}"[:300],
                           score=float(v) if self.output == 'score' else None, info=info)

    async def _answer(self, ev, body, key) -> dict[str, Any]:
        """The judge's answer to one prompt: dict(outcome ok | timeout | malformed | error | unresolved, value,
        rationale, raw, latency_s, tokens, cached). Cached process-wide unless a timeout or an error."""
        self.stats['asks'] += 1
        if self.cache and key in _CACHE:
            _CACHE.move_to_end(key)
            self.stats['cached'] += 1
            return dict(_CACHE[key], cached=True, latency_s=0.0)
        t0 = time.monotonic()
        try:
            status, data = await asyncio.wait_for(self._post(ev, body), self.timeout_s)
        except asyncio.TimeoutError:
            return self._done(dict(outcome='timeout'), t0)
        except LookupError as err:
            return self._done(dict(outcome='unresolved', error=str(err)), t0)
        except Exception as err:  # noqa: BLE001 - a judge failure is no decision, never a router error
            return self._done(dict(outcome='error', error=repr(err)[:300]), t0)
        if status != 200:
            return self._done(dict(outcome='error', error=f'HTTP {status}: {str(data)[:300]}'), t0)
        try:
            msg = data['choices'][0]['message']
            raw = _text(msg.get('content'))
            usage = data.get('usage') or {}
        except (KeyError, IndexError, TypeError, AttributeError):
            return self._done(dict(outcome='malformed', error='not a chat completion', raw=str(data)[:500]), t0)
        value, rationale, err = parse_answer(raw, self.output, **self.keys)
        ans = self._done(dict(outcome='malformed' if err else 'ok', value=value, rationale=rationale, raw=raw,
                              error=err, prompt_tokens=usage.get('prompt_tokens'),
                              completion_tokens=usage.get('completion_tokens')), t0)
        if self.cache:
            _CACHE[key] = ans
            while len(_CACHE) > CACHE_MAX:
                _CACHE.popitem(last=False)
        return ans

    def _done(self, ans: dict[str, Any], t0) -> dict[str, Any]:
        ans.update(latency_s=round(time.monotonic() - t0, 4), cached=False)
        if ans['outcome'] != 'unresolved':
            self.stats['calls'] += 1
            self.latencies.append(ans['latency_s'])
        return ans

    async def _post(self, ev, body):
        """(status, parsed JSON) from the endpoint: a URL directly; a name through the router's post hook, its
        endpoint map, or the resolver. LookupError when a name resolves nowhere."""
        url = self.endpoint
        if not url.startswith(('http://', 'https://')):
            post = ev.extra.get('post')
            if callable(post):
                res: Any = post(url, '/chat/completions', body)
                status, data = await res if inspect.isawaitable(res) else res
                return status, json.loads(data) if isinstance(data, (bytes, str)) else data
            target = (ev.extra.get('endpoints') or {}).get(url) or (_RESOLVER(url) if _RESOLVER else None)
            if isinstance(target, (list, tuple)):
                target = target[0] if target else None
            if not target:
                raise LookupError(f'endpoint name {url!r} resolves nowhere (no post hook, endpoint map or resolver)')
            url = target
        import aiohttp
        url = url.rstrip('/')
        if not url.endswith('/chat/completions'):
            url += '/chat/completions'
        headers = {}
        if self.api_key_env and os.environ.get(self.api_key_env):
            headers['Authorization'] = f'Bearer {os.environ[self.api_key_env]}'
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout_s)) as s:
            async with s.post(url, json=body, headers=headers) as resp:
                return resp.status, await resp.json(content_type=None)

    def _write(self, ev, ans, outcome):
        if not self.log:
            return
        rec: dict[str, Any] = dict(ts=time.time(), turn=ev.turn, phase=ev.phase, owner=ev.owner, outcome=outcome,
                                   value=ans.get('value'), threshold=self.threshold if self.output == 'score' else None,
                                   rationale=(ans.get('rationale') or '')[:300], error=ans.get('error'),
                                   **{k: ans.get(k) for k in ('latency_s', 'waited_s', 'prompt_tokens',
                                                              'completion_tokens', 'cached')},
                                   prefetched=bool(ans.get('prefetched')), model=self.model,
                                   task_id=ev.extra.get('task_id'), sid=ev.extra.get('sid'), seq=ev.extra.get('seq'))
        fd = os.open(self.log, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, (json.dumps(rec) + '\n').encode())    # one write per line: concurrent episodes interleave
        finally:                                               # whole lines
            os.close(fd)


def register():
    """Make `use: llm_judge` mean LlmJudge (replacing interventions' stage-4 stub). Returns what it replaced, so a
    caller (a test) can put it back."""
    old = iv.BUILTINS.get('llm_judge')
    iv.BUILTINS['llm_judge'] = LlmJudge
    return old
