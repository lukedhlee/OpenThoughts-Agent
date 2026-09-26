#!/usr/bin/env python3
"""Relay router: one OpenAI-compatible endpoint in front of a student and a teacher vLLM server.

Terminus-2 (harbor, LiteLLM `api_base`) talks to this router as if it were one model. Per episode the router decides
who answers each request:

  * The student answers until a takeover trigger fires, then the teacher answers every later request of that episode
    (sticky, at most one hand-off, never before agent turn 2 except done_claim).
  * Environment triggers (loop, no_progress_wait) are read from the request itself, before the student is asked, so the
    teacher writes the turn the trigger fired at.
  * done_claim (the student's first `task_complete: true`) lets the student's reply through; Terminus-2 then sends its
    own "Are you sure?" confirmation request, and the teacher answers it. Nothing is discarded.
  * Any other enabled decision trigger discards the student's reply (it is saved in the log as preference data) and the
    teacher answers the same request.
  * parse_error (--repair-on-parse-error, the relay_repair arm): the student's reply is run through Terminus-2's own
    parser (harbor terminus_json_plain_parser.py, loaded from its file) before harbor sees it. On a hard parse error
    (warnings pass) the reply is discarded (logged), the teacher answers the same request for ONE turn, and the
    student keeps the episode: no "fix your JSON" re-prompt enters the trace. If the teacher's repair turn claims
    done, it also answers Terminus-2's confirmation. The student then sees the repair turn rendered as the SFT
    converter renders teacher turns for 09-21: <|start_think|>{teacher reasoning}<|end_think|>{content}. With
    --student-tokenizer, only the most recent teacher turn keeps its reasoning whole; older ones are cut to their first
    --reasoning-cap tokens at a sentence/line boundary (reasoning_cap.py), each cut's char offset logged per request.
    With --autofix, a reply whose single intended action is recoverable is first rewritten into valid Terminus-2 JSON
    (autofix.py; the student's thinking kept verbatim) and runs as the student's turn (autofix=True, the original raw
    reply and the rewrite kind logged); only unrecoverable replies go to the teacher.
  * --mode teacher (control arm): the teacher answers everything; triggers are computed on its own turns and logged as
    `would_fire` only. --mode student: the reverse, for debugging.

Episode identity: the X-Harbor-Session-Id header that Terminus-2 sends with `llm_session_header` set (harbor branch
lukedhlee/terminus2-relay). It is per trial attempt and equals trajectory.json's session_id. `/tokenize` calls carry no
header and are routed by the first message of the conversation.

What each model sees. The student gets harbor's request unchanged (its own replies with inline <|start_think|> spans;
it never sees a teacher turn). The teacher gets its own earlier turns with their reasoning under both `reasoning` (the
key vLLM's chat parser reads) and `reasoning_content`, and the student's earlier turns per --student-think:
  strip (default; design note: "Snowball's earlier thinking is stripped from the history the teacher sees"): the think
        spans are removed, the Terminus-2 JSON (analysis, plan, commands) stays; Qwen3.8's template renders each such
        turn with an empty <think></think>.
  keep: the spans are removed from content and sent as that turn's reasoning, so the template renders the student's
        thinking inside <think>. Every log line and episode record carries the mode.
Harbor re-sends the teacher's reasoning itself (interleaved_thinking + the reasoning-key fix); if a turn arrives
without it, the router restores it from its own record and counts that as `reasoning_restored`.

Deadline (--deadline-epoch): after it, every episode is ended at its next request the same way (ending 'deadline'), so
a run stops inside its node-hour cap with its verifiers run and its results written; the readout counts those
episodes as censored.

Teacher endpoints: --teacher-url may be a comma-separated list; each episode is pinned to one (round-robin by
episode index), for its chat and /tokenize requests alike.

Budgets (--budget-mode, relay arm). The student's clock is the wall time since the episode's first request minus the
time its repair turns were with the teacher (from the router sending the repair request, retries included, to the
teacher's answer): repair queueing is infrastructure, not the student's time. The student's own discarded reply still
counts as its time. Every log line carries paused_sec and student_clock_sec.
 harbor runs the relay arm with agent_timeout_multiplier ~2 and the router ends
the episode itself when the student has used 1x the task's agent timeout without a takeover (`student_budget`), or
the teacher has used 1x from its takeover (`teacher_budget`). It ends an episode by answering the pending request, and
Terminus-2's confirmation request after it, with `task_complete: true` and no commands (owner `router`, never trained
on). The verifier then scores the sandbox as a timeout would. The teacher so gets a full budget of its own.

Logs (--log-dir): turns.jsonl (one line per request: episode, turn, owner, trigger + evidence, logged signals, raw
upstream response incl. reasoning, usage, latency), bodies/ep<N>.jsonl.gz (the exact body sent upstream for every
request), events.jsonl (start, health, takeovers, fatal), health.json, episodes.json (state snapshot), FATAL on abort.

Guards: at start both endpoints must list the configured served model names and answer one real completion (student:
think markers survive skip_special_tokens=false; teacher: the reasoning parser splits reasoning off); any failure exits
3 before harbor starts. At run time an upstream 401/403/404, or a 400 that names a missing model, is FATAL: the router
answers 503 from then on and writes FATAL so the driver can stop the run. Context-length 400s pass through (Terminus-2
handles them).

    python relay_router.py --mode relay --arm relay --port 8100 --log-dir <dir> --tasks tasks.json \
        --student-url http://node0:8000/v1 --student-model snowball \
        --teacher-url http://node1:8000/v1 --teacher-model qwen38 --budget-mode on
"""
import argparse
import asyncio
import collections
import gzip
import hashlib
import json
import os
import re
import signal
import sys
import time
import uuid

from aiohttp import ClientConnectionError, ClientSession, ClientTimeout, TCPConnector, web

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'triggers'))
import relay_triggers as rt  # noqa: E402
sys.path.insert(0, HERE)
import autofix as af  # noqa: E402
import reasoning_cap as rcap  # noqa: E402

ROUTER_VERSION = 'relay-router/1 (2026-09-25)'
SESSION_HEADER = 'X-Harbor-Session-Id'

# Terminus-2 prompt openings (harbor terminus_2.py; stable across the v0.1 pin and lukedhlee/terminus2-relay)
SUMMARY_PREFIX = 'You are about to hand off your work to another AI agent.'
QUESTIONS_PREFIX = 'You are picking up work from a previous AI agent on this task:'
ANSWERS_PREFIX = 'The next agent has a few questions for you'
HANDOFF_PREFIX = 'Here are the answers the other agent provided.'
CONFIRM_MARK = 'Are you sure you want to mark the task as complete?'
AUX_KINDS = ('summary', 'questions', 'answers')

THINK_SPAN_RE = re.compile(r'<\|start_think\|>.*?<\|end_think\|>|<think>.*?</think>', re.S)
OPEN_THINK_RE = re.compile(r'(?:<\|start_think\|>|<think>).*\Z', re.S)   # an unterminated span runs to the end
SYNTHETIC_DONE = json.dumps({'analysis': '', 'plan': '', 'commands': [], 'task_complete': True})


def sha(s):
    return hashlib.sha1((s or '').encode('utf-8', 'surrogatepass')).hexdigest()[:16]


def text_of(content):
    if isinstance(content, list):
        return '\n'.join(p.get('text', '') for p in content if isinstance(p, dict))
    return content or ''


def strip_think(content):
    """Student reply -> the text the teacher sees: every think span removed, an unterminated one to the end."""
    out = THINK_SPAN_RE.sub('', content)
    out = OPEN_THINK_RE.sub('', out)
    return out.strip()


THINK_INNER_RE = re.compile(r'<\|start_think\|>(.*?)<\|end_think\|>|<think>(.*?)</think>', re.S)
OPEN_INNER_RE = re.compile(r'(?:<\|start_think\|>|<think>)(.*)\Z', re.S)


def think_text(content):
    """The text inside a student reply's think spans (an unterminated span to the end), joined by blank lines."""
    parts = [(a or b).strip() for a, b in THINK_INNER_RE.findall(content)]
    rest = OPEN_INNER_RE.search(THINK_SPAN_RE.sub('', content))
    if rest:
        parts.append(rest.group(1).strip())
    return '\n\n'.join(p for p in parts if p)


def request_kind(messages):
    last = text_of(messages[-1].get('content')) if messages else ''
    if messages and messages[-1].get('role') == 'user':
        if last.startswith(SUMMARY_PREFIX):
            return 'summary'
        if last.startswith(QUESTIONS_PREFIX) and len(messages) == 1:
            return 'questions'
        if last.startswith(ANSWERS_PREFIX):
            return 'answers'
        if last.startswith(HANDOFF_PREFIX):
            return 'handoff'
        if CONFIRM_MARK in last:
            return 'confirm'
    return 'initial' if len(messages) == 1 else 'main'


def load_terminus_parser(path=None):
    """Terminus-2's own JSON parser (harbor terminus_json_plain_parser.py, stdlib-only), loaded from its file so the
    router applies exactly the harness's accept/reject decision. Searches sys.path when no path is given."""
    import hashlib as _h
    import importlib.util
    rel = os.path.join('harbor', 'agents', 'terminus_2', 'terminus_json_plain_parser.py')
    cands = [path] if path else [os.path.join(p, rel) for p in sys.path if p]
    for c in cands:
        if c and os.path.isfile(c):
            spec = importlib.util.spec_from_file_location('terminus_json_plain_parser', c)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.TerminusJSONPlainParser(), c, _h.sha256(open(c, 'rb').read()).hexdigest()[:16]
    raise SystemExit(f'Terminus-2 parser not found ({path or "on sys.path"}); pass --terminus-parser')


def openai_error(status, message, etype='relay_router_error'):
    return web.json_response({'error': {'message': message, 'type': etype, 'code': status}}, status=status)


class Episode:
    def __init__(self, sid, idx, first_text, task, owner):
        self.sid, self.idx = sid, idx
        self.first_sha = sha(first_text)
        self.task = task                  # {'task_id', 'budget_sec'} or None
        self.t0 = time.time()
        self.last_t = self.t0
        self.owner = owner                # 'student' | 'teacher'
        self.pending = None               # done_claim fire waiting for the confirmation request
        self.takeover = None              # {'trigger', 'kind', 'reason', 'turn', 'elapsed_sec', ...}
        self.takeover_t = None
        self.sc = rt.EpisodeScanner('terminus2')
        self.pending_reply = None         # content of the last main-chat reply returned, fed to the scanner later
        self.last_main_key = None
        self.replies = {}                 # content sha -> {'owner', 'turn', 'reasoning'}
        self.ending = None                # 'student_budget' | 'teacher_budget'
        self.n_requests = 0
        self.owners = []                  # (turn, owner) per main-chat reply returned
        self.lock = asyncio.Lock()
        self.student_think = None
        self.n_repairs = 0
        self.pinned = {}                  # who -> the server this episode is pinned to
        self.paused_sec = 0.0             # wall time the student's clock was stopped (teacher repair turns in flight)
        self.repair_confirm = False       # the teacher claimed done in a repair turn: it answers that confirmation

    def summary(self):
        return dict(episode=self.idx, sid=self.sid, task_id=(self.task or {}).get('task_id'), owner=self.owner,
                    student_think=self.student_think,
                    takeover={k: v for k, v in (self.takeover or {}).items() if k != 'discarded_student_reply'} or None,
                    ending=self.ending, turns=self.sc.turn, n_requests=self.n_requests, repairs=self.n_repairs,
                    paused_sec=round(self.paused_sec, 1), pinned=dict(self.pinned),
                    started=self.t0, last=self.last_t, owners=self.owners)


class Router:
    def __init__(self, a):
        self.a = a
        self.mode = a.mode
        # --teacher-url may list several servers (comma-separated); each episode is pinned to one, round-robin by
        # episode, so its prefix cache stays on one server and load splits evenly
        self.urls = {w: [u.strip().rstrip('/') for u in (getattr(a, f'{w}_url') or '').split(',') if u.strip()]
                     for w in ('student', 'teacher')}
        self.inflight = collections.Counter()
        self.pinned_count = collections.Counter()
        self.down_until = {}
        self.models = {'student': a.student_model, 'teacher': a.teacher_model}
        self.need = {'relay': ('student', 'teacher'), 'teacher': ('teacher',), 'student': ('student',)}[a.mode]
        self.cfg = dict(rt.CALIBRATED_CONFIG)
        if a.takeover is not None:
            self.cfg['enabled'] = [t for t in a.takeover.split(',') if t]
        self.logged_cfg = dict(self.cfg, enabled=[t for t in rt.LOGGED if t not in self.cfg['enabled']])
        self.teacher_drop = set(k for k in a.teacher_drop.split(',') if k)
        self.teacher_extra = json.loads(a.teacher_extra) if a.teacher_extra else {}
        self.student_extra = json.loads(a.student_extra) if a.student_extra else {}
        self.tasks = self._load_tasks(a.tasks)
        self.tok = rcap.load_tokenizer(a.student_tokenizer) if a.student_tokenizer else None
        self.parser = self.parser_path = self.parser_sha = None
        if a.repair_on_parse_error:
            self.parser, self.parser_path, self.parser_sha = load_terminus_parser(a.terminus_parser)
        self.episodes = {}
        self.by_first = {}
        self.n_episodes = 0
        self.fatal = None
        self.max_model_len = None
        self.counts = dict(requests=0, upstream_errors=0, takeovers=0, synthetic=0, no_task_match=0, repairs=0,
                           repair_reply_rejected=0, repinned=0, autofixes=0, teacher_cut_at_cap=0)
        os.makedirs(a.log_dir, exist_ok=True)
        os.makedirs(os.path.join(a.log_dir, 'bodies'), exist_ok=True)
        # one os.write per line on an O_APPEND fd: readers (the driver's gates) never see a half-written line
        self._turns = os.open(os.path.join(a.log_dir, 'turns.jsonl'), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        self._events = os.open(os.path.join(a.log_dir, 'events.jsonl'), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        self.http = None

    # ---- setup -------------------------------------------------------------------------------------------------
    @staticmethod
    def _load_tasks(path):
        if not path:
            return []
        tasks = json.load(open(path))
        out = []
        for t in tasks:
            key = (t.get('instruction') or '').strip()[:400]
            if key:
                out.append(dict(task_id=t['task_id'], key=key, budget_sec=float(t.get('agent_timeout_sec') or 0)))
        return out

    def match_task(self, first_text):
        hits = [t for t in self.tasks if t['key'] in first_text]
        if len(hits) == 1:
            return dict(task_id=hits[0]['task_id'], budget_sec=hits[0]['budget_sec'])
        if hits:   # an instruction that is a prefix of another: take the longest match
            h = max(hits, key=lambda t: len(t['key']))
            return dict(task_id=h['task_id'], budget_sec=h['budget_sec'], ambiguous=len(hits))
        return None

    def event(self, _event, **kw):
        os.write(self._events, (json.dumps(dict(ts=time.time(), event=_event, arm=self.a.arm, **kw)) + '\n').encode())

    def set_fatal(self, reason):
        if self.fatal:
            return
        self.fatal = reason
        print(f'RELAY_ROUTER FATAL: {reason}', flush=True)
        self.event('fatal', reason=reason)
        with open(os.path.join(self.a.log_dir, 'FATAL'), 'w') as f:
            f.write(reason + '\n')

    async def start_http(self):
        self.http = ClientSession(connector=TCPConnector(limit=self.a.upstream_connections),
                                  timeout=ClientTimeout(total=None, sock_connect=30, sock_read=self.a.upstream_read_timeout))

    async def health_check(self):
        """Served names, max_model_len and one real completion per endpoint. Returns (ok, report)."""
        report = {'router_version': ROUTER_VERSION, 'mode': self.mode, 'arm': self.a.arm, 'endpoints': {}}
        ok = True
        for who, url, name in [(w, u, w if len(self.urls[w]) == 1 else f'{w}{k}')
                               for w in self.need for k, u in enumerate(self.urls[w])]:
            r = {'url': url, 'model': self.models[who]}
            report['endpoints'][name] = r
            ids, mml = None, None
            for attempt in range(self.a.health_retries):
                try:
                    async with self.http.get(url + '/models', timeout=ClientTimeout(total=20)) as resp:
                        data = await resp.json(content_type=None)
                    ids = [m.get('id') for m in data.get('data', [])]
                    mml = next((m.get('max_model_len') for m in data.get('data', []) if m.get('id') == self.models[who]), None)
                    break
                except Exception as e:  # noqa: BLE001
                    r['models_error'] = repr(e)
                    await asyncio.sleep(self.a.health_wait)
            r['served_ids'] = ids
            r['max_model_len'] = mml
            if not ids or self.models[who] not in ids:
                r['fail'] = f'served model name mismatch: want {self.models[who]!r}, server lists {ids!r}'
                ok = False
                continue
            body = {'model': self.models[who], 'messages': [{'role': 'user', 'content': 'Print hello in bash. Be brief.'}],
                    'max_tokens': self.a.health_max_tokens}
            body.update(self.student_extra if who == 'student' else self.teacher_extra)
            if who == 'student':
                body.setdefault('skip_special_tokens', False)
            try:
                async with self.http.post(url + '/chat/completions', json=body,
                                          timeout=ClientTimeout(total=600)) as resp:
                    status, data = resp.status, await resp.json(content_type=None)
            except Exception as e:  # noqa: BLE001
                r['fail'] = f'smoke completion failed: {e!r}'
                ok = False
                continue
            if status != 200:
                r['fail'] = f'smoke completion HTTP {status}: {str(data)[:300]}'
                ok = False
                continue
            msg = data['choices'][0]['message']
            content = msg.get('content') or ''
            reasoning = msg.get('reasoning_content') or msg.get('reasoning') or ''
            r['smoke'] = dict(finish=data['choices'][0].get('finish_reason'), usage=data.get('usage'),
                              content=content[:300], reasoning=reasoning[:300], model=data.get('model'))
            if who == 'student' and self.a.strict_smoke and '<|start_think|>' not in content and '<|end_think|>' not in content:
                r['fail'] = 'student smoke reply has no <|start_think|>/<|end_think|> marker (thinking stripped?)'
                ok = False
            if who == 'teacher' and self.a.strict_smoke and (not reasoning or '</think>' in content):
                r['fail'] = 'teacher smoke reply has no separate reasoning (reasoning parser off?)'
                ok = False
        lens = [e.get('max_model_len') for e in report['endpoints'].values() if e.get('max_model_len')]
        self.max_model_len = min(lens) if lens else None
        report['max_model_len'] = self.max_model_len
        report['ok'] = ok
        with open(os.path.join(self.a.log_dir, 'health.json'), 'w') as f:
            json.dump(report, f, indent=1)
        self.event('health', ok=ok, report=report)
        return ok, report

    # ---- episodes ----------------------------------------------------------------------------------------------
    def episode(self, sid, messages):
        ep = self.episodes.get(sid)
        first = text_of(messages[0].get('content')) if messages else ''
        new_attempt = (ep is not None and sid.startswith('h-') and len(messages) == 1
                       and request_kind(messages) == 'initial' and ep.sc.turn > 0)
        if ep is None or new_attempt:
            task = self.match_task(first) if first else None
            if task is None:
                self.counts['no_task_match'] += 1
            self.n_episodes += 1
            ep = Episode(sid, self.n_episodes, first, task, 'teacher' if self.mode == 'teacher' else 'student')
            ep.student_think = self.a.student_think
            self.episodes[sid] = ep
            self.by_first.setdefault(ep.first_sha, []).append(sid)
            self.event('episode_start', episode=ep.idx, sid=sid, task=task, student_think=self.a.student_think)
        ep.last_t = time.time()
        return ep

    def owner_for_tokenize(self, messages):
        first = text_of(messages[0].get('content')) if messages else ''
        sids = self.by_first.get(sha(first)) or []
        eps = [self.episodes[s] for s in sids if s in self.episodes]
        if not eps:
            return None, ('teacher' if self.mode == 'teacher' else 'student')
        ep = max(eps, key=lambda e: e.last_t)
        return ep, ep.owner

    # ---- history conversion ------------------------------------------------------------------------------------
    def for_teacher(self, ep, messages):
        stats = dict(student_think=self.a.student_think, student_think_as_reasoning=0,
                     prior_teacher_turns=0, reasoning_key_from_harbor=0, reasoning_content_only=0, reasoning_restored=0,
                     teacher_turns_without_reasoning=0, student_turns=0, think_stripped=0, other_assistant=0)
        out = []
        for m in messages:
            if m.get('role') != 'assistant':
                out.append(m)
                continue
            content = text_of(m.get('content'))
            rec = ep.replies.get(sha(content)) if ep else None
            owner = rec['owner'] if rec else None
            m2 = {k: v for k, v in m.items() if k not in ('reasoning', 'reasoning_content')}
            if owner == 'teacher':
                stats['prior_teacher_turns'] += 1
                r = m.get('reasoning') or m.get('reasoning_content')
                if m.get('reasoning'):
                    stats['reasoning_key_from_harbor'] += 1
                elif m.get('reasoning_content'):
                    stats['reasoning_content_only'] += 1
                elif rec.get('reasoning'):
                    r = rec['reasoning']
                    stats['reasoning_restored'] += 1
                else:
                    stats['teacher_turns_without_reasoning'] += 1
                if r:
                    m2['reasoning'] = r
                    m2['reasoning_content'] = r
            elif owner == 'student' or (owner is None and ('<|start_think|>' in content or '<think>' in content)):
                stats['student_turns'] += owner == 'student'
                s = strip_think(content)
                if s != content:
                    stats['think_stripped'] += 1
                    m2['content'] = s
                    if self.a.student_think == 'keep':
                        # the student's thinking becomes that turn's reasoning, which Qwen's template renders in <think>
                        th = think_text(content)
                        if th:
                            m2['reasoning'] = th
                            m2['reasoning_content'] = th
                            stats['student_think_as_reasoning'] += 1
            else:   # router's synthetic turns, summarization replies: keep any reasoning harbor sent
                stats['other_assistant'] += 1
                r = m.get('reasoning') or m.get('reasoning_content')
                if r:
                    m2['reasoning'] = r
                    m2['reasoning_content'] = r
            out.append(m2)
        return out, stats

    def for_student(self, ep, messages):
        """The student sees its own turns unchanged. A teacher turn in its history (a parse_error repair) is rendered
        the way the SFT converter renders teacher turns for 09-21: the teacher's reasoning inside 09-21's think
        markers, then the content, no newlines around the span, no separate reasoning field."""
        stats = dict(teacher_turns_inline=0, teacher_turns_without_reasoning=0, cuts=[])
        teacher_idx = [i for i, m in enumerate(messages) if m.get('role') == 'assistant' and ep
                       and (ep.replies.get(sha(text_of(m.get('content')))) or {}).get('owner') == 'teacher']
        last_teacher = teacher_idx[-1] if teacher_idx else None
        out = []
        for i, m in enumerate(messages):
            if m.get('role') != 'assistant' or not ep:
                out.append(m)
                continue
            content = text_of(m.get('content'))
            rec = ep.replies.get(sha(content))
            m2 = {k: v for k, v in m.items() if k not in ('reasoning', 'reasoning_content')}
            if rec and rec['owner'] == 'teacher':
                r = m.get('reasoning') or m.get('reasoning_content') or rec.get('reasoning') or ''
                cut = self.tok is not None and i != last_teacher     # older teacher turns only
                key = ('cut', sha(content))
                if cut and key in rec:
                    text, at = rec[key]
                else:
                    text, at, _ = rcap.teacher_turn_for_student(content, r, self.tok, cut, self.a.reasoning_cap)
                    if cut:
                        rec[key] = (text, at)
                m2['content'] = text
                stats['teacher_turns_inline'] += 1
                stats['teacher_turns_without_reasoning'] += not r.strip()
                if at is not None:
                    stats['cuts'].append(dict(content_sha=sha(content), reasoning_chars=len(r.strip()), cut_at=at))
            out.append(m2)
        return out, stats

    def upstream_body(self, who, body, messages):
        if who == 'teacher':
            b = {k: v for k, v in body.items() if k not in self.teacher_drop}
            b.update(self.teacher_extra)
            if self.a.teacher_max_tokens:
                # the cap on one teacher reply (Luke 18:10 PT); harbor's chat path sends no max_tokens of its own
                b['max_tokens'] = min(int(b.get('max_tokens') or self.a.teacher_max_tokens), self.a.teacher_max_tokens)
        else:
            b = dict(body)
            b.update(self.student_extra)
        b['model'] = self.models[who]
        b['messages'] = messages
        return b

    def active(self, who):
        now = time.time()
        up = [u for u in self.urls[who] if self.down_until.get(u, 0) <= now]
        return up or list(self.urls[who])

    def pick(self, who, ep):
        """The server for this request: the episode's pinned one while it is listed and up, else the listed server
        with the fewest pinned episodes (the episode is then re-pinned there)."""
        act = self.active(who)
        if ep is not None:
            cur = ep.pinned.get(who)
            if cur in act:
                return cur
            u = min(act, key=lambda x: (self.pinned_count[x], act.index(x)))
            if cur is not None:
                self.pinned_count[cur] -= 1
                self.counts['repinned'] += 1
            ep.pinned[who] = u
            self.pinned_count[u] += 1
            return u
        return min(act, key=lambda x: (self.inflight[x], act.index(x)))

    def reload_teacher_urls(self):
        """--teacher-url-file: the driver edits the list to drain servers before releasing their nodes."""
        f = self.a.teacher_url_file
        if not f or not os.path.exists(f):
            return
        urls = [u.strip().rstrip('/') for u in open(f).read().replace(',', '\n').split() if u.strip()]
        if urls and urls != self.urls['teacher']:
            self.event('teacher_urls', old=self.urls['teacher'], new=urls)
            self.urls['teacher'] = urls

    async def post(self, who, path, body, ep=None):
        last = None
        for attempt in range(self.a.connect_retries + 1):
            base = self.pick(who, ep)
            url = (base + path) if path.startswith('/chat') else (re.sub(r'/v1$', '', base) + path)
            self.inflight[base] += 1
            try:
                async with self.http.post(url, json=body) as resp:
                    return resp.status, await resp.read()
            except (ClientConnectionError, ConnectionResetError) as e:   # connect errors, resets, disconnects
                last = e
                if len(self.urls[who]) > 1:
                    self.down_until[base] = time.time() + 60   # fail over: the next attempt re-pins elsewhere
                await asyncio.sleep(min(30, 2 ** attempt))
            finally:
                self.inflight[base] -= 1
        self.counts['upstream_errors'] += 1
        return 502, json.dumps({'error': {'message': f'relay router: {who} unreachable: {last!r}'}}).encode()

    def check_fatal(self, who, status, data):
        if status in (401, 403, 404) or (status == 400 and b'does not exist' in data):
            self.set_fatal(f'{who} answered HTTP {status}: {data[:300].decode("utf-8", "replace")}')

    def write_body(self, ep, seq, who, body):
        if self.a.log_bodies == 'none' or (self.a.log_bodies == 'teacher' and who != 'teacher'):
            return None
        name = f'ep{ep.idx:05d}.jsonl.gz'
        with gzip.open(os.path.join(self.a.log_dir, 'bodies', name), 'at') as f:
            f.write(json.dumps(dict(seq=seq, who=who, body=body)) + '\n')
        return f'bodies/{name}#{seq}'

    # ---- HTTP handlers -----------------------------------------------------------------------------------------
    async def h_models(self, request):
        return web.json_response({'object': 'list', 'data': [{
            'id': self.a.public_model, 'object': 'model', 'owned_by': 'relay-router', 'max_model_len': self.max_model_len}]})

    async def h_endpoints(self, request):
        return web.json_response({w: {u: dict(inflight=self.inflight[u], pinned=self.pinned_count[u],
                                               down=self.down_until.get(u, 0) > time.time()) for u in self.urls[w]}
                                  for w in ('student', 'teacher')})

    async def h_health(self, request):
        return web.json_response({'ok': not self.fatal, 'fatal': self.fatal}, status=503 if self.fatal else 200)

    async def h_tokenize(self, request):
        body = await request.json()
        if body.get('model') not in (None, self.a.public_model):
            return openai_error(404, f"The model `{body.get('model')}` does not exist.", 'NotFoundError')
        messages = body.get('messages')
        if not messages:
            who = 'teacher' if self.mode == 'teacher' else 'student'
            status, data = await self.post(who, '/tokenize', dict(body, model=self.models[who]))
            return web.Response(status=status, body=data, content_type='application/json')
        ep, who = self.owner_for_tokenize(messages)
        msgs = self.for_teacher(ep, messages)[0] if who == 'teacher' else self.for_student(ep, messages)[0]
        b = {k: v for k, v in body.items() if not (who == 'teacher' and k in self.teacher_drop)}
        b.update(model=self.models[who], messages=msgs)
        status, data = await self.post(who, '/tokenize', b, ep)
        return web.Response(status=status, body=data, content_type='application/json')

    async def h_chat(self, request):
        self.counts['requests'] += 1
        if self.fatal:
            return openai_error(503, f'relay router stopped: {self.fatal}')
        body = await request.json()
        if body.get('stream'):
            return openai_error(400, 'relay router: streaming is not supported')
        if body.get('model') != self.a.public_model:
            return openai_error(404, f"The model `{body.get('model')}` does not exist.", 'NotFoundError')
        messages = body.get('messages') or []
        if not messages:
            return openai_error(400, 'relay router: empty messages')
        sid = request.headers.get(SESSION_HEADER)
        if not sid:
            if not self.a.allow_hash_fallback:
                self.set_fatal(f'request without the {SESSION_HEADER} header (harbor needs llm_session_header)')
                return openai_error(503, f'relay router stopped: {self.fatal}')
            sid = 'h-' + sha(text_of(messages[0].get('content')))
        ep = self.episode(sid, messages)
        async with ep.lock:
            try:
                return await self.handle(ep, body, messages)
            except Exception as e:  # noqa: BLE001 - a router bug stops the run loudly instead of mislabelling turns
                import traceback
                self.set_fatal(f'router error on episode {ep.idx}: {e!r}\n{traceback.format_exc()}')
                return openai_error(503, f'relay router stopped: {e!r}')

    # ---- the decision ------------------------------------------------------------------------------------------
    async def handle(self, ep, body, messages):
        ep.n_requests += 1
        now = time.time()
        kind = request_kind(messages)
        rec = dict(ts=now, elapsed_sec=round(now - ep.t0, 3), arm=self.a.arm, router_version=ROUTER_VERSION,
                   episode=ep.idx, sid=ep.sid, task_id=(ep.task or {}).get('task_id'), request_kind=kind,
                   n_messages=len(messages), request_sha=sha(json.dumps(messages, sort_keys=True, ensure_ascii=False)),
                   seq=ep.n_requests, student_think=self.a.student_think, paused_sec=round(ep.paused_sec, 3),
                   student_clock_sec=round(now - ep.t0 - ep.paused_sec, 3))
        if kind in AUX_KINDS:
            who = ep.owner
            rec.update(turn=None, owner=who)
            return await self.answer(ep, who, body, messages, rec, main=False)

        # feed what harbor holds into the scanner (lazily, so a retried request does not advance it)
        key = (len(messages), sha(text_of(messages[-1].get('content'))))
        retry = key == ep.last_main_key
        if not retry and kind != 'initial':
            if kind == 'handoff':
                # summarization replaced the history; the observation of the last reply never reached the model
                if ep.pending_reply is not None:
                    ep.sc.add_message({'role': 'assistant', 'content': ep.pending_reply})
                ep.sc.add_message({'role': 'user', 'content': ''})
            else:
                prev = messages[-2] if len(messages) >= 2 else None
                if prev is not None and prev.get('role') == 'assistant':
                    ep.sc.add_message({'role': 'assistant', 'content': text_of(prev.get('content'))})
                ep.sc.add_message({'role': 'user', 'content': text_of(messages[-1].get('content'))})
            ep.pending_reply = None
        ep.last_main_key = key
        t = ep.sc.turn + 1
        rec.update(turn=t, retry=retry)

        # budgets and endings
        if not ep.ending and self.a.deadline_epoch and now >= self.a.deadline_epoch:
            ep.ending = 'deadline'
            self.event('ending', episode=ep.idx, sid=ep.sid, ending='deadline', turn=t, elapsed_sec=now - ep.t0)
        if ep.ending:
            return self.synthetic(ep, rec)
        if ep.pending is not None and ep.owner == 'student':
            if kind == 'confirm':
                self.take_over(ep, ep.pending, t, now)
                rec['takeover'] = self.takeover_rec(ep)
            elif not retry:
                # Terminus-2 did not accept the claim (e.g. its parser rejected the reply): no confirmation turn
                self.event('done_claim_dropped', episode=ep.idx, sid=ep.sid, turn=t, request_kind=kind)
                rec['done_claim_dropped'] = True
            if not retry:
                ep.pending = None
        if self.a.budget_mode == 'on' and ep.task and ep.task.get('budget_sec'):
            b = ep.task['budget_sec']
            if ep.owner == 'student' and self.mode in ('relay', 'student') and now - ep.t0 - ep.paused_sec >= self.a.student_budget_frac * b:
                ep.ending = 'student_budget'
            elif ep.owner == 'teacher' and ep.takeover_t and now - ep.takeover_t >= self.a.teacher_budget_frac * b:
                ep.ending = 'teacher_budget'
            elif ep.owner == 'teacher' and self.mode == 'teacher' and now - ep.t0 >= self.a.teacher_budget_frac * b:
                ep.ending = 'teacher_budget'
            if ep.ending:
                self.event('ending', episode=ep.idx, sid=ep.sid, ending=ep.ending, turn=t, elapsed_sec=now - ep.t0)
                return self.synthetic(ep, rec)

        # the teacher claimed done in a repair turn: it answers Terminus-2's confirmation, then the student resumes
        if ep.repair_confirm and ep.owner == 'student':
            if not retry:
                ep.repair_confirm = False
            if kind == 'confirm':
                rec.update(owner='teacher', repair=True, repair_kind='confirm')
                return await self.answer(ep, 'teacher', body, messages, rec, main=True)
        # environment triggers act on the request, before the student is asked
        if ep.owner == 'student' and self.mode == 'relay' and not retry:
            env = [f for f in ep.sc.current_fires(None, self.cfg) if f['kind'] == 'environment']
            if env:
                self.take_over(ep, env[0], t, now)
                rec['takeover'] = self.takeover_rec(ep)
        if ep.owner == 'student':
            return await self.answer_student(ep, body, messages, rec, t, now)
        rec['owner'] = 'teacher'
        return await self.answer(ep, 'teacher', body, messages, rec, main=True)

    def take_over(self, ep, fire, t, now, discarded=None):
        ep.owner = 'teacher'
        ep.takeover_t = now
        ep.takeover = dict(trigger=fire['trigger'], kind=fire['kind'], reason=fire.get('reason'), fired_turn=fire.get('turn'),
                           turn=t, elapsed_sec=round(now - ep.t0, 3),
                           budget_sec=(ep.task or {}).get('budget_sec'), discarded_student_reply=discarded)
        self.counts['takeovers'] += 1
        self.event('takeover', episode=ep.idx, sid=ep.sid, task_id=(ep.task or {}).get('task_id'),
                   **{k: v for k, v in ep.takeover.items() if k != 'discarded_student_reply'},
                   discarded=discarded is not None)

    @staticmethod
    def takeover_rec(ep):
        return {k: v for k, v in ep.takeover.items() if k != 'discarded_student_reply'}

    def logged_signals(self, ep, reply):
        try:
            fs = ep.sc.current_fires(reply, self.logged_cfg)
        except Exception as e:  # noqa: BLE001 - a detector bug must not stop a rollout
            return [dict(trigger='detector_error', reason=repr(e)[:200])]
        return [dict(trigger=f['trigger'], kind=f['kind'], reason=f.get('reason')) for f in fs]

    async def answer_student(self, ep, body, messages, rec, t, now):
        rec['owner'] = 'student'
        msgs, sstats = self.for_student(ep, messages)
        rec['student_view'] = sstats
        b = self.upstream_body('student', body, msgs)
        rec['sent_body'] = self.write_body(ep, rec['seq'], 'student', b)
        t1 = time.time()
        status, data = await self.post('student', '/chat/completions', b)
        rec.update(latency_sec=round(time.time() - t1, 3), upstream_status=status)
        if status != 200:
            self.check_fatal('student', status, data)
            rec['upstream_error'] = data[:2000].decode('utf-8', 'replace')
            self.log(rec)
            return web.Response(status=status, body=data, content_type='application/json')
        resp = json.loads(data)
        content = text_of(resp['choices'][0]['message'].get('content'))
        if self.parser is not None and self.mode == 'relay':
            pr = self.parser.parse_response(content)
            if pr.error and self.a.autofix:
                fx = af.autofix(content, resp['choices'][0].get('finish_reason'), self.parser)
                if isinstance(fx, af.Fix):
                    # format autofix: the student's own action, rewritten into valid Terminus-2 JSON; it runs, and the
                    # triggers judge the rewrite
                    rec.update(autofix=True, autofix_kind=fx.kind, autofix_reason=pr.error[:300],
                               original_student_reply=json.loads(json.dumps(resp)))
                    resp['choices'][0]['message']['content'] = fx.content
                    content = fx.content
                    self.counts['autofixes'] += 1
                    pr = self.parser.parse_response(content)
                else:
                    rec['autofix_unfixable'] = fx.reason
            if pr.error:
                # parse_error (non-sticky repair): Terminus-2 would reject this reply; it never reaches harbor. The
                # teacher answers the same request, one turn, and the student keeps the episode.
                ep.n_repairs += 1
                self.counts['repairs'] += 1
                rec.update(repair=True, repair_kind='parse_error', repair_reason=pr.error[:500], owner='teacher',
                           discarded_student_reply=resp, discarded_usage=resp.get('usage'),
                           student_latency_sec=rec.pop('latency_sec'), student_sent_body=rec.pop('sent_body'))
                self.event('repair', episode=ep.idx, sid=ep.sid, turn=t, reason=pr.error[:200])
                return await self.answer(ep, 'teacher', body, messages, rec, main=True)
        rec['logged'] = self.logged_signals(ep, content)
        dec = [] if self.mode != 'relay' else [f for f in ep.sc.current_fires(content, self.cfg) if f['kind'] == 'decision']
        if self.mode == 'student':
            rec['would_fire'] = [dict(trigger=f['trigger'], reason=f.get('reason'))
                                 for f in ep.sc.current_fires(content, self.cfg)]
        if (self.mode == 'relay' and 'done_claim' in self.cfg['enabled'] and not any(f['trigger'] == 'done_claim' for f in dec)
                and rt.parse_reply(content, 'terminus2')['done']):
            # the scanner fires done_claim only on the episode's first task_complete; a teacher repair turn may have
            # claimed first, so judge the student's own claim directly
            dec.append(dict(trigger='done_claim', kind='decision', turn=t, reason='task_complete: true (student)'))
        done = [f for f in dec if f['trigger'] == 'done_claim']
        other = [f for f in dec if f['trigger'] != 'done_claim']
        if other:
            # decision trigger: the student's reply never runs; the teacher answers the same request
            self.take_over(ep, other[0], t, now, discarded=resp)
            rec.update(discarded_student_reply=resp, discarded_usage=resp.get('usage'),
                       takeover=self.takeover_rec(ep), owner='teacher')
            rec['student_latency_sec'] = rec.pop('latency_sec')
            rec['student_sent_body'] = rec.pop('sent_body')
            return await self.answer(ep, 'teacher', body, messages, rec, main=True)
        if done:
            ep.pending = done[0]
            rec['done_claim'] = True
            self.event('done_claim', episode=ep.idx, sid=ep.sid, turn=t)
        return self.finish(ep, 'student', resp, rec, t, main=True)

    async def answer(self, ep, who, body, messages, rec, main):
        if who == 'teacher':
            msgs, stats = self.for_teacher(ep, messages)
            rec['refeed'] = stats
        else:
            msgs, stats = self.for_student(ep, messages)
            rec['student_view'] = stats
        b = self.upstream_body(who, body, msgs)
        rec['sent_body'] = self.write_body(ep, rec['seq'], who, b)
        t1 = time.time()
        attempts = self.a.repair_attempts if (rec.get('repair') and self.parser is not None) else 1
        for attempt in range(attempts):
            status, data = await self.post(who, '/chat/completions', b, ep)
            if status == 400 and who == 'teacher' and b.get('max_tokens') and b'maximum context length' in data:
                # the cap asked for more than the context has left. vLLM's "at least N input tokens" is only a lower
                # bound, so ask again WITHOUT max_tokens: vLLM then generates into what is left, which is below the
                # cap anyway (the uncapped request's behaviour). A 400 on that retry is a real context overflow.
                b = {k: v for k, v in b.items() if k != 'max_tokens'}
                rec['teacher_max_tokens_dropped'] = True
                status, data = await self.post(who, '/chat/completions', b, ep)
            if status != 200 or attempt == attempts - 1:
                break
            c = text_of(json.loads(data)['choices'][0]['message'].get('content'))
            perr = self.parser.parse_response(c).error
            if not perr:
                break
            self.counts['repair_reply_rejected'] += 1
            rec.setdefault('repair_rejected_replies', []).append(json.loads(data))
        if rec.get('repair') and self.parser is not None and status == 200:
            c = text_of(json.loads(data)['choices'][0]['message'].get('content'))
            pr = self.parser.parse_response(c)
            rec['repair_reply_parse_error'] = pr.error[:300] if pr.error else None
            if not pr.error and pr.is_task_complete and rec.get('repair_kind') == 'parse_error':
                ep.repair_confirm = True
        rec.update(latency_sec=round(time.time() - t1, 3), upstream_status=status)
        if rec.get('repair') and ep.owner == 'student':
            # the student's clock stops while its repair turn is with the teacher (queueing is infrastructure)
            ep.paused_sec += time.time() - t1
            rec['paused_this_turn_sec'] = round(time.time() - t1, 3)
        if status != 200:
            self.check_fatal(who, status, data)
            rec['upstream_error'] = data[:2000].decode('utf-8', 'replace')
            self.log(rec)
            return web.Response(status=status, body=data, content_type='application/json')
        resp = json.loads(data)
        msg = resp['choices'][0]['message']
        if who == 'teacher':
            r = msg.get('reasoning_content') or msg.get('reasoning')
            if r:   # harbor reads reasoning_content; vLLM's chat parser reads reasoning on the way back
                msg['reasoning_content'] = r
                msg['reasoning'] = r
            rec['teacher_reasoning_chars'] = len(r or '')
            if resp['choices'][0].get('finish_reason') == 'length':
                # cut at the cap (or at the context's end): passed to harbor as served; Terminus-2 salvages it or
                # asks again ("NONE of the actions ... were performed")
                rec['teacher_cut_at_cap'] = True
                self.counts['teacher_cut_at_cap'] += 1
            rec['content_has_think_close'] = '</think>' in (msg.get('content') or '')
        t = rec.get('turn')
        if main and self.mode in ('teacher', 'student'):
            content = text_of(msg.get('content'))
            rec['logged'] = self.logged_signals(ep, content)
            rec['would_fire'] = [dict(trigger=f['trigger'], reason=f.get('reason'))
                                 for f in ep.sc.current_fires(content, self.cfg)]
        elif main:
            rec['logged'] = self.logged_signals(ep, text_of(msg.get('content')))
        return self.finish(ep, who, resp, rec, t, main=main)

    def finish(self, ep, who, resp, rec, t, main):
        msg = resp['choices'][0]['message']
        content = text_of(msg.get('content'))
        reasoning = msg.get('reasoning_content') or msg.get('reasoning')
        ep.replies[sha(content)] = dict(owner=who, turn=t, reasoning=reasoning)
        if main:
            ep.pending_reply = content
            ep.owners.append((t, who))
        rec.update(content_sha=sha(content), usage=resp.get('usage'), finish_reason=resp['choices'][0].get('finish_reason'),
                   served_model=resp.get('model'), response=resp)
        self.log(rec)
        return web.json_response(resp)

    def synthetic(self, ep, rec):
        """End the episode the way a timeout would: task_complete with no commands, twice (Terminus-2 confirms)."""
        self.counts['synthetic'] += 1
        resp = {'id': f'relay-{uuid.uuid4().hex[:12]}', 'object': 'chat.completion', 'created': int(time.time()),
                'model': 'relay-router',
                'choices': [{'index': 0, 'finish_reason': 'stop',
                             'message': {'role': 'assistant', 'content': SYNTHETIC_DONE}}],
                'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}}
        rec.update(owner='router', ending=ep.ending, upstream_status=None, latency_sec=0.0)
        return self.finish(ep, 'router', resp, rec, rec.get('turn'), main=True)

    def log(self, rec):
        os.write(self._turns, (json.dumps(rec, ensure_ascii=False) + '\n').encode())

    def snapshot(self):
        path = os.path.join(self.a.log_dir, 'episodes.json')
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(dict(ts=time.time(), counts=self.counts, fatal=self.fatal,
                           episodes=[e.summary() for e in self.episodes.values()]), f)
        os.replace(tmp, path)

    def status_line(self):
        owners = {}
        trig = {}
        for e in self.episodes.values():
            owners[e.owner] = owners.get(e.owner, 0) + 1
            if e.takeover:
                trig[e.takeover['trigger']] = trig.get(e.takeover['trigger'], 0) + 1
        return (f'[{time.strftime("%H:%M:%S")}] arm={self.a.arm} episodes={len(self.episodes)} owners={owners} '
                f'takeovers={trig} counts={self.counts} fatal={self.fatal}')


def build_app(router):
    app = web.Application(client_max_size=256 * 1024 * 1024)
    app.router.add_get('/v1/models', router.h_models)
    app.router.add_get('/health', router.h_health)
    app.router.add_get('/endpoints', router.h_endpoints)
    app.router.add_post('/tokenize', router.h_tokenize)
    app.router.add_post('/v1/chat/completions', router.h_chat)
    return app


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=['relay', 'teacher', 'student'], required=True)
    p.add_argument('--arm', default=None, help='label written on every log line (default: the mode)')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--public-model', default='relay', help='the model name harbor asks for (hosted_vllm/<this>)')
    p.add_argument('--student-url')
    p.add_argument('--student-model')
    p.add_argument('--teacher-url')
    p.add_argument('--teacher-model')
    p.add_argument('--log-dir', required=True)
    p.add_argument('--tasks', help='JSON list of {task_id, instruction, agent_timeout_sec} (task match + budgets)')
    p.add_argument('--takeover', default=None,
                   help='comma list of takeover triggers (default: the calibrated set done_claim,loop,no_progress_wait)')
    p.add_argument('--budget-mode', choices=['on', 'off'], default='off')
    p.add_argument('--student-budget-frac', type=float, default=1.0)
    p.add_argument('--teacher-budget-frac', type=float, default=1.0)
    p.add_argument('--teacher-drop', default='skip_special_tokens',
                   help='request keys not forwarded to the teacher (student-only serving settings)')
    p.add_argument('--teacher-extra', default='', help='JSON merged into every teacher request body')
    p.add_argument('--student-extra', default='', help='JSON merged into every student request body')
    p.add_argument('--student-think', choices=['strip', 'keep'], default='strip',
                   help="what the teacher sees of the student's thinking in the student's earlier turns: strip (the "
                        "think spans are removed; the Terminus-2 JSON with its analysis/plan stays) or keep (the spans "
                        "are removed from content and sent as that turn's reasoning, which Qwen3.8's template renders "
                        "inside <think>)")
    p.add_argument('--teacher-url-file', default=None,
                   help='file listing the teacher servers; re-read every --status-every s (drain before release)')
    p.add_argument('--repair-on-parse-error', action='store_true',
                   help='parse_error trigger (non-sticky): a student reply that Terminus-2\'s own parser rejects is '
                        'discarded and the teacher answers the same request for one turn; the student keeps the episode')
    p.add_argument('--teacher-max-tokens', type=int, default=None,
                   help='max_tokens on every teacher request (reasoning + content); a reply cut there is logged as '
                        'teacher_cut_at_cap. Dropped (context remainder, below the cap) when the cap would not fit.')
    p.add_argument('--student-tokenizer', default=None,
                   help="09-21's tokenizer.json: turns on the cap on older teacher reasoning in the student's view")
    p.add_argument('--reasoning-cap', type=int, default=rcap.CAP_TOKENS,
                   help='tokens of an older teacher turn\'s reasoning the student sees (the latest turn stays whole)')
    p.add_argument('--autofix', action='store_true',
                   help='before a parse_error repair, rewrite a recoverable student reply into valid Terminus-2 JSON '
                        '(autofix.py) and run the student\'s own action; the teacher repairs only what is unrecoverable')
    p.add_argument('--terminus-parser', default=None,
                   help='path of harbor terminus_json_plain_parser.py (default: found on sys.path / PYTHONPATH)')
    p.add_argument('--repair-attempts', type=int, default=2,
                   help='teacher attempts per repair turn when its own reply fails the parser')
    p.add_argument('--deadline-epoch', type=float, default=None,
                   help='wall-clock time (unix) after which every episode is ended at its next request (ending '
                        "'deadline'), so the run finishes with its results inside the node-hour cap")
    p.add_argument('--allow-hash-fallback', action='store_true',
                   help='without the session header, key episodes by a hash of the first message (not concurrency-safe '
                        'for duplicate instructions; retried attempts start a new episode on a 1-message request)')
    p.add_argument('--log-bodies', choices=['all', 'teacher', 'none'], default='all')
    p.add_argument('--no-strict-smoke', dest='strict_smoke', action='store_false')
    p.add_argument('--skip-health', action='store_true', help='tests only')
    p.add_argument('--health-retries', type=int, default=6)
    p.add_argument('--health-wait', type=float, default=10.0)
    p.add_argument('--health-max-tokens', type=int, default=2000)
    p.add_argument('--connect-retries', type=int, default=4)
    p.add_argument('--upstream-connections', type=int, default=1024)
    p.add_argument('--upstream-read-timeout', type=float, default=3600.0)
    p.add_argument('--status-every', type=float, default=60.0)
    a = p.parse_args(argv)
    a.arm = a.arm or a.mode
    for who in {'relay': ('student', 'teacher'), 'teacher': ('teacher',), 'student': ('student',)}[a.mode]:
        if not getattr(a, f'{who}_url') or not getattr(a, f'{who}_model'):
            p.error(f'--mode {a.mode} needs --{who}-url and --{who}-model')
    return a


async def serve(a, ready_event=None):
    router = Router(a)
    await router.start_http()
    router.event('start', argv=sys.argv, version=ROUTER_VERSION, mode=a.mode, takeover=router.cfg['enabled'],
                 budget_mode=a.budget_mode, tasks=len(router.tasks), student_think=a.student_think,
                 deadline_epoch=a.deadline_epoch, repair_on_parse_error=a.repair_on_parse_error,
                 terminus_parser=router.parser_path, terminus_parser_sha256=router.parser_sha)
    if not a.skip_health:
        ok, report = await router.health_check()
        print(json.dumps(report, indent=1), flush=True)
        if not ok:
            router.set_fatal('health check failed: ' + '; '.join(
                f"{w}: {e['fail']}" for w, e in report['endpoints'].items() if e.get('fail')))
            await router.http.close()
            return router, 3
    runner = web.AppRunner(build_app(router), access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, a.host, a.port)
    await site.start()
    print(f'RELAY_ROUTER_READY http://{a.host}:{a.port}/v1 mode={a.mode} arm={a.arm} takeover={router.cfg["enabled"]}',
          flush=True)
    router.event('ready', url=f'http://{a.host}:{a.port}/v1')
    if ready_event is not None:
        ready_event.set()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for s in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(s, stop.set)
        except (NotImplementedError, RuntimeError):
            pass
    router._stop = stop
    try:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=a.status_every)
            except asyncio.TimeoutError:
                pass
            router.snapshot()
            router.reload_teacher_urls()
            print(router.status_line(), flush=True)
    finally:
        router.snapshot()
        router.event('stop', counts=router.counts)
        await runner.cleanup()
        await router.http.close()
    return router, 0


def main():
    a = parse_args()
    _, rc = asyncio.run(serve(a))
    sys.exit(rc)


if __name__ == '__main__':
    main()
