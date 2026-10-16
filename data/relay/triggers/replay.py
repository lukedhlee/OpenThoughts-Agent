#!/usr/bin/env python3
"""Offline replay of recorded relay episodes: run any trigger config over the router's logs, causally, turn by turn,
with no sandbox and no GPU.

Plan: notes/ot-agent/relay-triggers/stage1_replay_scope.md. The router logs one turns.jsonl line per request and the
exact body it sent upstream (bodies/ep<N>.jsonl.gz). For each episode this rebuilds, in the order the router saw them,
the request-phase Event of every turn (the history as the student saw it, the previous action's result normalized per
harness) and then the reply-phase Event (the student's reply as the router judged it), feeds a fresh EpisodeScanner
exactly as relay_router.handle / handle_msa feed it, and runs the config through interventions.Controller twice over:
  * shadow: every trigger log_only, so every fire is logged (turn, phase, reason, score, info) and nothing changes
    the replayed history;
  * live: the config as written, once for the whole policy and once per trigger alone, each until its first
    intervention: what the Controller would have done first. After the policy's first intervention the recorded
    history is no longer what would have happened, so later fires are logged with valid=false (counterfactual-invalid).

Which episodes give clean labels: student-alone episodes (router --mode student) are scored against the verifier
outcome. Relay episodes are replayed only up to their first real intervention (the first main request the student did
not answer: a parse repair, an assist, a takeover) and reported separately. Control episodes (--mode teacher) are
teacher-alone runs, replayed with the triggers judging the teacher's own turns, as the router's would_fire did; also
reported separately.

    python data/relay/triggers/replay.py --runs <manifest.json | run dirs> --config my.yaml --out out/
    python data/relay/triggers/replay.py --runs <...> --builtin-check --out out/   # vs the router's logged would_fire

From Python (in a notebook, `await areplay_episode(...)`, since a loop is already running there):
    import replay
    for ep in replay.load_episodes(['replay_sample/manifest.json'], groups=['student_only_t2']):
        recs = replay.replay_episode(ep, 'my.yaml')    # a dict per fire (record='fire'), then one record='episode'
    for step in replay.iter_events(ep): ...           # the rebuilt Events themselves

Outputs (--out): per_episode.jsonl, fires.jsonl, summary.json (and builtin_check.json), plus a table per trigger.

Event.view_tokens is the router's logged count, else the student's own prompt_tokens for that request, else (lazily)
an offline count: 09-21's chat template and the run's tokenizer, equal to vLLM's count on the sample. What the replay
cannot know is marked, never guessed: a reply's parse error depends on whether the harness parser is applied
(--reply-parse); a teacher-only request's history is rebuilt by undoing the router's teacher view and checked against
the logged request_sha (summary: history_exact / history_inexact).
"""
import argparse
import asyncio
import collections
import copy
import dataclasses
import functools
import glob
import gzip
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RELAY = os.path.dirname(HERE)
for _d in (os.path.join(RELAY, 'pilot'), os.path.join(RELAY, 'router'), HERE):
    if _d not in sys.path:
        sys.path.insert(0, _d)
import relay_triggers as rt  # noqa: E402
import interventions as iv  # noqa: E402
import msa_tool as mt  # noqa: E402
import readout  # noqa: E402

TEMPLATE = os.path.join(RELAY, 'router', 'tests', 'grug0921_chat_template.jinja')   # 09-21's chat template
BOS = '<|begin_of_text|>'
PARSER_REL = os.path.join('harbor', 'agents', 'terminus_2', 'terminus_json_plain_parser.py')
CLASS_OF_MODE = {'student': 'student_alone', 'relay': 'relay', 'teacher': 'teacher_alone'}
CLASSES = ('student_alone', 'relay', 'teacher_alone')
ENV_TRIGGERS = ('edit_failed', 'loop', 'no_progress_wait', 'error_streak')   # relay_triggers.env_fires, in its order
# Terminus-2 observations after which nothing ran: its parser rejected the reply, or the reply hit the output limit
T2_NOT_RUN = ('Previous response had parsing errors', 'NONE of the actions you just requested were performed')
T2_LIMIT_MARK = '[... output limited to'          # Terminus-2 cut the new terminal output
TIMED_OUT_RE = re.compile(r'timed out', re.I)
EPISODE_RE = re.compile(rb'"episode": (\d+)')
TOKENIZER_LOG_RE = re.compile(r"student tokenizer from the serve job's meta: (\S+)")
POLICY = '__policy__'                             # the whole config's first intervention


def _rr():
    """relay_router (strip_think, load_terminus_parser), imported on first use: it needs aiohttp."""
    import relay_router
    return relay_router


def _text(content):
    """As relay_router.text_of: a message's text, joining OpenAI content parts."""
    if isinstance(content, list):
        return '\n'.join(p.get('text', '') for p in content if isinstance(p, dict))
    return content or ''


def _msg_sha(messages):
    """relay_router's request_sha of a message list."""
    return readout.sha(json.dumps(messages, sort_keys=True, ensure_ascii=False))


# ==== where the logs and their inputs are =====================================================================
class Inputs:
    """Cluster paths named in a run's logs (the router's argv, the driver's tokenizer line) -> local files: the path
    itself when it exists, else the same absolute path under an inputs root (a copied run tree keeps them in
    _inputs/)."""

    def __init__(self, roots=()):
        self.roots = [r for r in roots if r and os.path.isdir(r)]

    def find(self, path):
        if not path:
            return None
        if os.path.exists(path):
            return path
        for r in self.roots:
            p = os.path.join(r, path.lstrip('/'))
            if os.path.exists(p):
                return p
        return None

    def find_rel(self, rel):
        """The first file under a root whose path ends with `rel`."""
        for r in self.roots:
            hits = sorted(glob.glob(os.path.join(r, '**', rel), recursive=True))
            if hits:
                return hits[0]
        return None


class TokenCounter:
    """09-21's prompt length of a request offline: its chat template rendered as vLLM renders it (tool-call arguments
    as objects, HF's tojson) and encoded with the run's tokenizer. On the 2026-10-16 sample it equals the logged
    prompt_tokens on every request checked (Terminus-2 and mini-swe-agent)."""

    def __init__(self, tokenizer_path, template_path=TEMPLATE):
        import jinja2
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(tokenizer_path)
        # as sft/render.py load_template, plus HF's tojson (jinja's own escapes <, > and &)
        src = re.sub(r'\{%-?\s*(end)?generation\s*-?%\}', '', open(template_path).read())
        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=['jinja2.ext.loopcontrols'])
        env.globals['raise_exception'] = lambda m: (_ for _ in ()).throw(jinja2.TemplateError(m))
        env.filters['tojson'] = lambda x, indent=None, separators=None, sort_keys=False: json.dumps(
            x, ensure_ascii=False, indent=indent, separators=separators, sort_keys=sort_keys)
        self.tpl = env.from_string(src)

    def count(self, messages, tools=None, kwargs=None):
        msgs = copy.deepcopy(list(messages))
        for m in msgs:
            for tc in m.get('tool_calls') or []:
                fn = tc.get('function') or {}
                if isinstance(fn.get('arguments'), str):
                    try:
                        fn['arguments'] = json.loads(fn['arguments'])
                    except ValueError:
                        pass
        text = self.tpl.render(messages=msgs, tools=tools, bos_token=BOS, add_generation_prompt=True,
                               **dict(kwargs or {}))
        return len(self.tok.encode(text, add_special_tokens=False).ids)


@functools.lru_cache(maxsize=8)
def _counter(tokenizer_path, template_path):
    return TokenCounter(tokenizer_path, template_path)


class RunLog:
    """One router log dir (router_<arm>/ of a run): its start event (argv, mode, harness, enabled triggers, verify
    note), turns.jsonl indexed by episode (byte offsets, so an episode is read without loading the run), the bodies,
    the task budgets, the tokenizer and Terminus-2's parser."""

    def __init__(self, router_dir, inputs=None, tokenizer=None, chat_template=TEMPLATE, terminus_parser=None):
        self.dir = os.path.abspath(router_dir)
        self.run_dir = os.path.dirname(self.dir)
        self.run = os.path.basename(self.run_dir)
        base = os.path.basename(self.dir)
        self.arm = base[len('router_'):] if base.startswith('router_') else base
        self.inputs = inputs or Inputs([os.path.join(os.path.dirname(self.run_dir), '_inputs')])
        self.start = next((e for e in readout.read_jsonl(os.path.join(self.dir, 'events.jsonl'))
                           if e.get('event') == 'start'), {})
        self.argv = self.start.get('argv') or []
        self.mode = self.start.get('mode') or self.flag('--mode') or \
            {'student_only': 'student', 'control': 'teacher'}.get(self.arm, 'relay')
        self.harness = self.start.get('harness') or self.flag('--harness') or 'terminus2'
        self.enabled = list(self.start.get('takeover') or rt.CALIBRATED_CONFIG['enabled'])
        self.note = (self.start.get('verify_note_text') or None) if self.start.get('verify_note') else None
        self.hint = self.flag('--teacher-hint-text')
        # the router parsed Terminus-2 replies itself only in relay mode with the parser loaded (answer_student)
        self.parses = self.mode == 'relay' and bool(self.start.get('repair_on_parse_error')
                                                    or self.start.get('teacher_format_guard'))
        self.cls = CLASS_OF_MODE.get(self.mode, self.mode)
        self._tok_override, self._template, self._parser_override = tokenizer, chat_template, terminus_parser
        self._index = self._tasks = None
        self._parser = False

    def flag(self, name, default=None):
        a = self.argv
        return a[a.index(name) + 1] if name in a and a.index(name) + 1 < len(a) else default

    def index(self):
        """{episode: [byte offsets of its turns.jsonl lines]}, one pass, lines not parsed."""
        if self._index is None:
            idx = collections.defaultdict(list)
            off = 0
            with open(os.path.join(self.dir, 'turns.jsonl'), 'rb') as f:
                for line in f:
                    m = EPISODE_RE.search(line, 0, 4096)
                    if m:
                        idx[int(m.group(1))].append(off)
                    off += len(line)
            self._index = dict(idx)
        return self._index

    def records(self, episode, first_only=False):
        """The episode's turns.jsonl records in request order (seq)."""
        out = []
        with open(os.path.join(self.dir, 'turns.jsonl'), 'rb') as f:
            for off in self.index().get(episode, [])[:1 if first_only else None]:
                f.seek(off)
                out.append(json.loads(f.readline()))
        return sorted(out, key=lambda r: r.get('seq') or 0)

    def bodies(self, episode):
        """{(seq, who): body} of every request the router sent upstream for the episode."""
        p = os.path.join(self.dir, 'bodies', f'ep{episode:05d}.jsonl.gz')
        out = {}
        if os.path.exists(p):
            with gzip.open(p, 'rt') as f:
                for line in f:
                    d = json.loads(line)
                    out[(d['seq'], d['who'])] = d['body']
        return out

    def budget(self, task_id):
        """The task's agent budget in seconds (the router's --tasks file), or None."""
        if self._tasks is None:
            self._tasks = {}
            p = self.inputs.find(self.flag('--tasks'))
            if p:
                for t in json.load(open(p)):
                    self._tasks[t.get('task_id')] = float(t.get('agent_timeout_sec') or 0) or None
        return self._tasks.get(task_id)

    def tokenizer_path(self):
        """--tokenizer, else the router's --student-tokenizer, else the driver's 'student tokenizer from the serve
        job's meta' line, else any tokenizer.json under the inputs (09-21's family shares one vocabulary)."""
        cands = [self._tok_override, self.flag('--student-tokenizer')]
        drv = os.path.join(self.run_dir, 'driver.log')
        if os.path.exists(drv):
            with open(drv, errors='replace') as f:
                cands += [m.group(1) for m in TOKENIZER_LOG_RE.finditer(f.read())][-1:]
        for c in cands:
            p = self.inputs.find(c)
            if p:
                return p
        return self.inputs.find_rel('tokenizer.json')

    def count_tokens(self, messages, body=None):
        p = self.tokenizer_path()
        if not p or not os.path.exists(self._template):
            return None
        return _counter(p, self._template).count(messages, (body or {}).get('tools'),
                                                 (body or {}).get('chat_template_kwargs'))

    def parser(self):
        """Terminus-2's own parser (the run's file, else one under the inputs, else harbor on sys.path), or None."""
        if self._parser is False:
            path = (self.inputs.find(self._parser_override) or self.inputs.find(self.start.get('terminus_parser'))
                    or self.inputs.find(self.flag('--terminus-parser')) or self.inputs.find_rel(PARSER_REL))
            try:
                self._parser = _rr().load_terminus_parser(path)[0]
            except SystemExit:
                self._parser = None
        return self._parser


@dataclasses.dataclass
class Episode:
    """One recorded episode: the router's episode index in its log, the trajectory's session id, the group it is
    reported under, and its verifier outcome (pass | fail | None when unusable: harness error, verifier timeout,
    deadline-censored, no trial)."""
    log: RunLog
    episode: int
    sid: str
    task_id: str | None = None
    group: str | None = None
    outcome: str | None = None
    outcome_why: str | None = None
    label: dict = dataclasses.field(default_factory=dict)     # e.g. point_of_no_return_turn
    meta: dict = dataclasses.field(default_factory=dict)

    @property
    def id(self):
        return f'{self.log.run}/{self.log.arm}/{self.episode}'

    @property
    def harness(self):
        return self.log.harness

    @property
    def mode(self):
        return self.log.mode

    @property
    def cls(self):
        return self.log.cls

    def row(self):
        return dict(id=self.id, run=self.log.run, arm=self.log.arm, episode=self.episode, sid=self.sid,
                    task_id=self.task_id, group=self.group, cls=self.cls, harness=self.harness, mode=self.mode,
                    outcome=self.outcome)


def _load_labels(path):
    """{sid or episode id: label dict} from a JSON list or JSONL of rows with `sid` or `id` and label fields
    (point_of_no_return_turn)."""
    if not path:
        return {}
    txt = open(path).read().strip()
    rows = json.loads(txt) if txt.startswith('[') else [json.loads(l) for l in txt.splitlines() if l.strip()]
    out = {}
    for r in rows:
        lab = {k: v for k, v in r.items() if k not in ('sid', 'id')}
        for k in ('sid', 'id'):
            if r.get(k):
                out[r[k]] = lab
    return out


def load_episodes(paths, groups=None, harness=None, inputs=(), labels=None, tokenizer=None, chat_template=TEMPLATE,
                  terminus_parser=None):
    """The episodes to replay, lazily: from manifest.json files (one row per episode: run, arm, episode, sid, group,
    outcome) or from run dirs (router_<arm>/ under them, or a router dir itself: every episode of its turns.jsonl, the
    outcome from the harbor trials in jobs/<run>_<arm>[_p2]/ under readout.py's rules: pass = reward >= 1; harness
    errors, verifier timeouts and no trial = no outcome; deadline-censored episodes are found while replaying).
    groups: keep these groups only (manifest groups; <arm>_<t2|msa> for run dirs). harness: terminus2 | msa."""
    groups = set(groups) if groups else None
    labels = labels if isinstance(labels, dict) else _load_labels(labels)
    logs = {}

    def log_of(d, roots):
        d = os.path.abspath(d)
        if d not in logs:
            logs[d] = RunLog(d, Inputs(list(inputs) + roots), tokenizer, chat_template, terminus_parser)
        return logs[d]

    def keep(ep):
        return (groups is None or ep.group in groups) and (harness is None or ep.harness == harness)

    for path in paths:
        if os.path.isfile(path):
            base = os.path.dirname(os.path.abspath(path))
            for row in json.load(open(path)):
                log = log_of(os.path.join(base, row['run'], f"router_{row['arm']}"), [os.path.join(base, '_inputs')])
                ep = Episode(log, int(row['episode']), row['sid'], row.get('task'), row.get('group'),
                             row.get('outcome'), None if row.get('outcome') in ('pass', 'fail') else 'manifest',
                             labels.get(f"{log.run}/{log.arm}/{row['episode']}") or labels.get(row['sid']) or {}, row)
                if keep(ep):
                    yield ep
            continue
        dirs = [path] if os.path.exists(os.path.join(path, 'turns.jsonl')) else \
            sorted(os.path.dirname(p) for p in glob.glob(os.path.join(path, 'router_*', 'turns.jsonl')))
        for d in dirs:
            log = log_of(d, [os.path.join(os.path.dirname(os.path.abspath(d)), '_inputs'),
                             os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(d))), '_inputs')])
            trials = {t['sid']: t for t in readout.arm_trials(log.run_dir, log.run, log.arm) if t['sid']}
            for n in sorted(log.index()):
                first = log.records(n, first_only=True)
                if not first:
                    continue
                sid = first[0].get('sid')
                t = trials.get(sid)
                if t is None:
                    outcome, why = None, 'no trial'
                elif t['harness_error'] or t['verifier_timeout']:
                    outcome, why = None, 'verifier timeout' if t['verifier_timeout'] else f"harness error {t['exc']}"
                else:
                    outcome, why = ('pass' if readout.is_pass(t) else 'fail'), None
                ep = Episode(log, n, sid, first[0].get('task_id'),
                             f"{log.arm}_{'msa' if log.harness == 'msa' else 't2'}", outcome, why,
                             labels.get(f'{log.run}/{log.arm}/{n}') or labels.get(sid) or {},
                             dict(outcome_from='trials', trial=t['trial'] if t else None,
                                  reward=t['reward'] if t else None, exception=t['exc'] if t else None))
                if keep(ep):
                    yield ep


# ==== rebuilding the events ===================================================================================
@dataclasses.dataclass
class Step:
    """One rebuilt Event, the router record of its request, and `real`: the phase of the episode's first real
    intervention when it happens at this event (relay episodes; the replay stops after it)."""
    ev: object
    rec: dict
    real: str | None = None


class _Returned:
    """Every reply the router returned to harbor so far (relay_router.finish), to undo the teacher's view of harbor's
    history where only a teacher body was logged: Terminus-2 student turns had their thinking stripped (strip_think);
    mini-swe-agent turns had their reasoning dropped; harbor holds what the router returned."""

    def __init__(self, msa):
        self.msa = msa
        self.by_call = {}       # msa: first tool call id -> the message returned
        self.by_stripped = {}   # Terminus-2: strip_think(content) -> content, the student's replies
        self.contents = set()

    def add(self, r):
        resp = r.get('response')
        if not resp or not resp.get('choices'):
            return
        msg = resp['choices'][0].get('message') or {}
        if self.msa:
            k = mt.call_id_key(msg)
            if k:
                self.by_call[k] = msg
            return
        c = _text(msg.get('content'))
        self.contents.add(c)
        if r.get('owner') == 'student':
            self.by_stripped.setdefault(_rr().strip_think(c), c)

    def undo_teacher_view(self, msgs, r, log):
        out = []
        for m in msgs:
            if m.get('role') == 'assistant':
                if self.msa:
                    m = self.by_call.get(mt.call_id_key(m), m)
                else:
                    c = _text(m.get('content'))
                    if c not in self.contents and c in self.by_stripped:
                        m = dict({k: v for k, v in m.items() if k not in ('reasoning', 'reasoning_content')},
                                 content=self.by_stripped[c])
            out.append(m)
        if self.msa and log.hint and out and out[0].get('role') == 'system':
            out[0] = _cut_suffix(out[0], '\n\n' + log.hint)          # the teacher-only hint on the system message
        if r.get('verify_note') and log.note and out:
            out[-1] = _cut_suffix(out[-1], '\n\n' + log.note)         # the verify note on the teacher's request
        return out


def _cut_suffix(m, suffix):
    c = _text(m.get('content'))
    return dict(m, content=c[:-len(suffix)]) if c.endswith(suffix) else m


def _history(r, bodies, returned, log, stats):
    """(harbor's messages for this request, the body they came from). The student body is harbor's request as sent
    (relay_router.for_student changes only teacher turns, and the replay stops before any); without one, the teacher
    body with the router's changes undone. Checked against the logged request_sha."""
    seq = r.get('seq')
    sb = bodies.get((seq, 'student')) if (r.get('student_sent_body') or
                                          (r.get('owner') == 'student' and r.get('sent_body'))) else None
    if sb is not None:
        H, body = sb['messages'], sb
    else:
        body = bodies.get((seq, 'teacher'))
        if body is None:
            return None, None
        H = body['messages']
        stats['history_from_teacher_body'] += 1
        if _msg_sha(H) != r.get('request_sha'):
            H = returned.undo_teacher_view(H, r, log)
            stats['history_teacher_view_undone'] += 1
    stats['history_exact' if _msg_sha(H) == r.get('request_sha') else 'history_inexact'] += 1
    return H, body


def _t2_result(H, kind, prev_screen):
    """Terminus-2: the previous reply's keystrokes and the new terminal output they produced (what the scanner reads:
    the screen part after the last header, minus what the previous screen already showed). No exit code (the screen
    carries none); truncated when Terminus-2 cut the output. None when nothing ran (the first request, a parse-error
    re-prompt, a reply cut at the output limit, the request after summarization)."""
    if kind in ('initial', 'handoff') or len(H) < 2 or H[-2].get('role') != 'assistant':
        return None
    obs = _text(H[-1].get('content'))
    if any(m in obs for m in T2_NOT_RUN):
        return None
    r = rt.parse_reply(_text(H[-2].get('content')), 'terminus2')
    if not r['ok']:
        return None
    cmds = [c for c, _ in r['cmds']]
    k, scr = rt.screen_part(obs)
    out = rt.new_output(k, scr, cmds, prev_screen) if k else ''
    return iv.ActionResult(command='\n'.join(c.rstrip('\n') for c in cmds if c.strip()), output=out,
                           truncated=T2_LIMIT_MARK in out)


def _msa_result(H):
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
        c = _text(m.get('content'))
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


def _msa_parse_error(served):
    p = mt.parse(_text(served.get('content')), served.get('reasoning_content') or served.get('reasoning'),
                 lambda j: f'call_{j}')
    return p.error or mt.toolcall_error(p.tool_calls) or None


def _reply(r, log, owner, mode, stats):
    """The reply the triggers judge at this request (interventions.Reply), or None when no model answered it for the
    episode's owner. text and action as the router judged them (after its autofix). parse_error per `mode`: 'harness'
    applies the harness's own parser (Terminus-2's parser file; msa_tool's parse for mini-swe-agent, which the router
    always ran); 'router' keeps only what that run's router knew (it parsed Terminus-2 replies only in relay mode with
    the parser loaded, so a student-mode would_fire judged every reply). action is None when parse_error is set."""
    if owner == 'student':
        resp = r.get('discarded_student_reply') or (r.get('response') if r.get('owner') == 'student' else None)
    else:
        resp = r.get('response') if r.get('owner') == 'teacher' else None
    if not resp or not resp.get('choices'):
        return None
    msg = resp['choices'][0].get('message') or {}
    stats['replies'] += 1
    if log.harness == 'msa':
        if owner == 'student':
            served = r.get('served_message') or msg
            text = _text(served.get('content'))
            err = r.get('repair_reason') if r.get('repair_kind') == 'parse_error' else None
            if err is None and r.get('student_parse_error') and not r.get('autofix'):
                err = _msa_parse_error(served) or r['student_parse_error']
        else:    # the teacher's (control): the router judged its parsed tool calls, else its text
            text = _text(msg.get('content'))
            err = (mt.toolcall_error(msg.get('tool_calls')) or None) if mode == 'harness' else None
        action = rt.tool_reply(msg) if msg.get('tool_calls') and not err else None
    else:
        text = _text(msg.get('content'))
        err = None
        if mode == 'harness' or (log.parses and owner == 'student'):
            p = log.parser()
            if p is not None:
                err = p.parse_response(text).error or None
            else:
                stats['no_terminus_parser'] += 1
                err = r.get('repair_reason') if r.get('repair_kind') == 'parse_error' else None
        action = None if err else rt.parse_reply(text, 'terminus2')
    stats['reply_parse_errors'] += bool(err)
    return iv.Reply(text=text, action=action, parse_error=err)


def _known_view_tokens(r, owner):
    """The student view's token count when the log has it: the router's /tokenize count, else the prompt_tokens the
    student's own request reported (same server, same template: the readout checks they agree)."""
    if r.get('student_view_tokens') is not None:
        return int(r['student_view_tokens'])
    if owner == 'student':
        u = r.get('discarded_usage') if r.get('discarded_student_reply') else \
            (r.get('usage') if r.get('owner') == 'student' else None)
        if u and u.get('prompt_tokens') is not None:
            return int(u['prompt_tokens'])
    return None


def _view_tokens(known, log, messages, body, stats):
    cache = {}

    async def view_tokens():
        if 'n' not in cache:
            if known is None:
                stats['view_tokens_counted'] += 1
            cache['n'] = known if known is not None else log.count_tokens(messages, body)
        return cache['n']
    return view_tokens


def _real_intervention(r, phase):
    to = r.get('takeover') or {}
    kind = 'repair' if r.get('repair') else 'assist' if r.get('assist') else 'takeover' if to else 'teacher'
    return dict(turn=r.get('turn'), phase=phase, kind=kind,
                trigger=to.get('trigger') or r.get('repair_kind') or r.get('assist_kind'),
                fired_turn=to.get('fired_turn', r.get('turn')))


def iter_events(ep, reply_parse='harness', stats=None, info=None):
    """Yield the episode's Steps in the router's order: per main request a request-phase Event, then, when a model
    answered it for the episode's owner, a reply-phase Event. Built only from the log up to that request (causal): the
    scanner is fed when the request arrives, as relay_router.handle (Terminus-2) and handle_msa feed it; turn and
    is_retry are the router's own. Stops at the router's endings (deadline, budget, context hard end: no trigger is
    asked there) and, in relay episodes, after the first real intervention. Fills `info` with stop, real and
    end_turn (the episode's last turn a model answered, in the whole log). Event.scanner is live: the next request
    feeds it, so read an event's scanner before advancing the generator (the Controller does)."""
    stats = stats if stats is not None else collections.Counter()
    info = info if info is not None else {}
    log = ep.log
    msa = log.harness == 'msa'
    owner = 'teacher' if log.mode == 'teacher' else 'student'
    recs = log.records(ep.episode)
    bodies = log.bodies(ep.episode)
    info.update(stop='end', real=None,
                end_turn=max((r['turn'] for r in recs if r.get('turn') is not None and r.get('response')
                              and r.get('owner') in ('student', 'teacher')), default=None))
    sc = rt.EpisodeScanner(None if msa else 'terminus2')
    returned = _Returned(msa)
    n_fed, pending_reply, last_result = 0, None, None
    budget = log.budget(ep.task_id)
    for r in recs:
        if r.get('turn') is None:            # Terminus-2 summarization: no turn, not fed, no trigger asked
            returned.add(r)
            continue
        if r.get('owner') == 'router':       # an ending: the router answered itself, no trigger is asked
            info['stop'] = f"ending:{r.get('ending')}"
            stats['stop_ending'] += 1
            return
        H, body = _history(r, bodies, returned, log, stats)
        if H is None:
            info['stop'] = 'missing_body'
            stats['stop_missing_body'] += 1
            return
        kind, retry = r.get('request_kind'), bool(r.get('retry'))
        prev_screen = sc.prev_screen
        if msa:                              # handle_msa: every message not fed yet, then close the turn
            if len(H) > n_fed:
                for m in H[n_fed:]:
                    sc.add_message(m)
                sc.flush()
                n_fed = len(H)
        elif not retry and kind != 'initial':   # handle: the last reply and its observation, lazily
            if kind == 'handoff':
                if pending_reply is not None:
                    sc.add_message({'role': 'assistant', 'content': pending_reply})
                sc.add_message({'role': 'user', 'content': ''})
            else:
                prev = H[-2] if len(H) >= 2 else None
                if prev is not None and prev.get('role') == 'assistant':
                    sc.add_message({'role': 'assistant', 'content': _text(prev.get('content'))})
                sc.add_message({'role': 'user', 'content': _text(H[-1].get('content'))})
            pending_reply = None
        stats['turn_ok' if sc.turn + 1 == r['turn'] else 'turn_mismatch'] += 1
        if not retry:                        # a retry is the same request: the same result
            last_result = _msa_result(H) if msa else _t2_result(H, kind, prev_screen)
        real = None
        if owner == 'student' and r.get('owner') != 'student':
            real = 'reply' if r.get('discarded_student_reply') else 'request'
        clock = iv.Clock(float(r.get('student_clock_sec') or 0.0), budget)
        extra = dict(harness=log.harness, task_id=ep.task_id, arm=log.arm, request_kind=kind, run=log.run,
                     sid=ep.sid, seq=r.get('seq'))
        ev = iv.Event('request', r['turn'], is_retry=retry, messages=H, last_result=last_result, scanner=sc,
                      clock=clock, view_tokens=_view_tokens(_known_view_tokens(r, owner), log, H, body, stats),
                      extra=extra)
        stats['request_events'] += 1
        yield Step(ev, r, real if real == 'request' else None)
        reply = None if real == 'request' else _reply(r, log, owner, reply_parse, stats)
        if reply is not None:
            used = clock.used_s + (0.0 if 'paused_this_turn_sec' in r
                                   else float(r.get('student_latency_sec') or r.get('latency_sec') or 0.0))
            stats['reply_events'] += 1
            yield Step(dataclasses.replace(ev, phase='reply', reply=reply, clock=iv.Clock(used, budget)), r, real)
        if real:
            info.update(stop='real_intervention', real=_real_intervention(r, real))
            return
        returned.add(r)
        if r.get('response'):
            pending_reply = _text(((r['response'].get('choices') or [{}])[0].get('message') or {}).get('content'))


# ==== running a config over an episode ========================================================================
def load_trigger_config(config):
    """(Policy, [TriggerSpec]) from a YAML path, a dict (interventions.load_config) or an already-built pair."""
    if isinstance(config, tuple):
        return config
    return iv.load_config(config)


def _own(spec):
    """The spec with trigger state of its own: the loader's factory rebuilds it per Controller; a hand-built spec
    without one gets a deep copy."""
    if spec.make is not None:
        return spec
    try:
        return dataclasses.replace(spec, trigger=copy.deepcopy(spec.trigger))
    except Exception:  # noqa: BLE001 - an uncopyable trigger (a live client): shared, as the caller built it
        return spec


def _shadow(spec, owner):
    """log_only; on teacher-alone episodes also asked on the teacher's turns (the router's would_fire judged those)."""
    owners = spec.owners | {'teacher'} if owner == 'teacher' else spec.owners
    return dataclasses.replace(_own(spec), log_only=True, owners=owners)


def _act(rec, I, ev):
    return dict(trigger=I.trigger, decided_turn=I.fired_turn, start_turn=I.start_turn, start_phase=ev.phase,
                where=I.where, span=I.span, reason=I.reason, score=I.score)


async def areplay_episode(ep, config, reply_parse='harness', per_trigger=True, stats=None):
    """Replay one episode under a trigger config. Returns a dict per shadow fire (record='fire': trigger, turn, phase,
    reason, score, info, where, span, seq, valid) and, last, the episode's summary (record='episode': outcome, end_turn,
    first_fire per trigger, first_actionable per trigger alone, policy = the whole config's first intervention,
    real_intervention for relay episodes, stop). valid=false: after the policy's first intervention."""
    stats = stats if stats is not None else collections.Counter()
    policy, specs = load_trigger_config(config)
    owner = 'teacher' if ep.mode == 'teacher' else 'student'
    shadow = iv.Controller(policy, [_shadow(s, owner) for s in specs], initial_owner=owner)
    live = {}
    if owner == 'student' and specs:
        live[POLICY] = iv.Controller(policy, [_own(s) for s in specs])
        if per_trigger and len(specs) > 1:
            for s in specs:
                live[s.name] = iv.Controller(policy, [_own(s)])
    base = ep.row()
    fires, acts, cut, info = [], {}, None, {}
    for i, step in enumerate(iter_events(ep, reply_parse, stats, info)):
        ev = step.ev
        for name, ctl in live.items():   # before the shadow, so a fire at the cut event itself is still valid
            if name in acts:
                continue                 # stopped at its first intervention
            r = await (ctl.on_request(ev) if ev.phase == 'request' else ctl.on_reply(ev))
            rec = next((x for x in r.records if x.outcome == 'acted'), None)
            if rec is not None:
                acts[name] = _act(rec, ctl.interventions[rec.intervention - 1], ev)
                if name == POLICY:
                    cut = i
        r = await (shadow.on_request(ev) if ev.phase == 'request' else shadow.on_reply(ev))
        for rec in r.records:
            d = rec.decision
            fires.append(dict(base, record='fire', trigger=d.trigger, phase=rec.phase, turn=rec.turn,
                              seq=ev.extra['seq'], reason=d.reason, score=d.score, info=getattr(d, 'info', None),
                              where=d.where, span=d.span, owner=ev.owner, valid=cut is None or i <= cut))
    for name, ctl in live.items():       # a reply-time next still waiting when the episode ended
        if name not in acts and ctl.pending is not None:
            _, d, fired = ctl.pending
            acts[name] = dict(trigger=d.trigger, decided_turn=fired, start_turn=None, start_phase=None, where=d.where,
                              span=d.span, reason=d.reason, score=d.score, pending_at_end=True)
    if len(specs) == 1 and POLICY in acts:
        acts.setdefault(specs[0].name, acts[POLICY])
    outcome, why = ep.outcome, ep.outcome_why
    if ep.meta.get('outcome_from') == 'trials' and info.get('stop') == 'ending:deadline':
        outcome, why = None, 'censored'  # readout.mark_censored's rule (a manifest row carries its own outcome)
    first = {}
    for f in fires:
        f['outcome'] = outcome
        first.setdefault(f['trigger'], dict(turn=f['turn'], phase=f['phase'], reason=f['reason'], score=f['score'],
                                            valid=f['valid']))
    summary = dict(base, record='episode', outcome=outcome, outcome_why=why, end_turn=info.get('end_turn'),
                   stop=info.get('stop'), real_intervention=info.get('real'), first_fire=first,
                   first_actionable={k: v for k, v in acts.items() if k != POLICY}, policy=acts.get(POLICY),
                   n_fires=len(fires), point_of_no_return_turn=ep.label.get('point_of_no_return_turn'))
    return fires + [summary]


def replay_episode(ep, config, reply_parse='harness', per_trigger=True, stats=None):
    """areplay_episode for scripts (inside a running loop, e.g. a notebook, await areplay_episode instead)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(areplay_episode(ep, config, reply_parse, per_trigger, stats))
    raise RuntimeError('an event loop is running here (a notebook?): use `await areplay_episode(...)`')


# ==== the report ==============================================================================================
def _r(a, b):
    return round(a / b, 4) if b else None


def _mean(xs):
    return round(statistics.mean(xs), 2) if xs else None


def _median(xs):
    return statistics.median(xs) if xs else None


def _metrics(eps, turn_of):
    """Fire rate on passing vs failing episodes, first turn and lead (turns before the end) on failing ones,
    precision / recall of 'fired' for 'failed', lead vs the labelled point of no return and vs the real intervention."""
    P = [e for e in eps if e['outcome'] == 'pass']
    F = [e for e in eps if e['outcome'] == 'fail']
    fP = [e for e in P if turn_of(e) is not None]
    fF = [e for e in F if turn_of(e) is not None]
    tF = [turn_of(e) for e in fF]
    lead = [e['end_turn'] - turn_of(e) for e in fF if e.get('end_turn') is not None]
    m = dict(episodes=len(eps), fired=sum(1 for e in eps if turn_of(e) is not None), passing=len(P), failing=len(F),
             fired_passing=len(fP), fired_failing=len(fF), fire_rate_passing=_r(len(fP), len(P)),
             fire_rate_failing=_r(len(fF), len(F)), first_turn_failing_mean=_mean(tF),
             first_turn_failing_median=_median(tF), lead_failing_mean=_mean(lead), lead_failing_median=_median(lead),
             precision=_r(len(fF), len(fF) + len(fP)), recall=_r(len(fF), len(F)))
    lab = [e for e in eps if e.get('point_of_no_return_turn') is not None]
    if lab:
        pl = [e['point_of_no_return_turn'] - turn_of(e) for e in lab if turn_of(e) is not None]
        m.update(labelled=len(lab), fired_labelled=len(pl), lead_vs_pnr_mean=_mean(pl), lead_vs_pnr_median=_median(pl),
                 fired_by_pnr=_r(sum(1 for x in pl if x >= 0), len(lab)))
    real = [e['real_intervention']['turn'] - turn_of(e) for e in eps
            if e.get('real_intervention') and turn_of(e) is not None]
    if real:
        m.update(lead_vs_real_intervention_mean=_mean(real), lead_vs_real_intervention_median=_median(real))
    return m


def _block(eps, names):
    owned = [e for e in eps if e['cls'] != 'teacher_alone']    # the teacher owns those: nothing to take over
    out = dict(episodes=len(eps), triggers={},
               policy=_metrics(owned, lambda e: (e['policy'] or {}).get('decided_turn')) if owned else None)
    for n in names:
        out['triggers'][n] = dict(
            first_fire=_metrics(eps, lambda e, n=n: (e['first_fire'].get(n) or {}).get('turn')),
            first_actionable=_metrics(owned, lambda e, n=n: (e['first_actionable'].get(n) or {}).get('decided_turn'))
            if owned else None)
    return out


def summarize(rows, names):
    """summary.json: per class (student_alone: clean labels; relay and teacher_alone: reported separately) and per
    group, per trigger, the metrics of its first fire on the recorded history and of its first actionable fire alone,
    plus the whole policy's first intervention."""
    by_cls, by_group = collections.defaultdict(list), collections.defaultdict(list)
    for e in rows:
        by_cls[e['cls']].append(e)
        by_group[e['group']].append(e)
    return dict(classes={c: _block(by_cls[c], names) for c in CLASSES if by_cls[c]},
                groups={g: _block(v, names) for g, v in sorted(by_group.items())})


def _pct(x):
    return '-' if x is None else f'{100 * x:.0f}%'


def _num(x):
    return '-' if x is None else (f'{x:g}' if isinstance(x, (int, float)) else str(x))


def print_table(summary, file=sys.stdout):
    hdr = ('trigger', 'pass', 'fail', 'fire|pass', 'fire|fail', 'turn|fail', 'lead|fail', 'prec', 'recall')
    for cls, b in summary['classes'].items():
        clean = '' if cls == 'student_alone' else '   (not clean: reported separately)'
        print(f"\n{cls}: {b['episodes']} episodes{clean}", file=file)
        print('  %-24s %5s %5s %9s %9s %11s %11s %6s %6s' % hdr, file=file)
        rows = [(n, t['first_fire']) for n, t in b['triggers'].items()] + [('POLICY (first act)', b['policy'])]
        for n, m in rows:
            if m is None:
                print(f'  {n:<24} n/a: the teacher owns these episodes', file=file)
                continue
            print('  %-24s %5s %5s %9s %9s %11s %11s %6s %6s' % (
                n[:24], f"{m['fired_passing']}/{m['passing']}", f"{m['fired_failing']}/{m['failing']}",
                _pct(m['fire_rate_passing']), _pct(m['fire_rate_failing']),
                f"{_num(m['first_turn_failing_mean'])}/{_num(m['first_turn_failing_median'])}",
                f"{_num(m['lead_failing_mean'])}/{_num(m['lead_failing_median'])}",
                _pct(m['precision']), _pct(m['recall'])), file=file)
    print('\n  fire|pass, fire|fail: share of passing / failing episodes where it fired. turn|fail, lead|fail:'
          '\n  mean/median over failing episodes that fired, lead = turns before the episode\'s end. prec, recall: of'
          '\n  "fired" for "failed". Trigger rows: each trigger\'s first fire on the recorded history. POLICY: the'
          '\n  whole config\'s first intervention under its policy (earliest turn, priority, pending next).', file=file)


# ==== the router's own would_fire, as a config ================================================================
class RouterWouldFire:
    """One relay_triggers trigger exactly as the router logs it in would_fire (--mode student / teacher):
    EpisodeScanner.current_fires on CALIBRATED_CONFIG with only this trigger enabled; environment triggers on the
    request, decision triggers on the reply as the router judged it. For done_claim this differs from the done_claim
    built-in, whose relay semantics fire on any claim of the student's: the scanner fires on the episode's first claim
    only."""

    def __init__(self, name):
        self.name = name
        self.kind = 'environment' if name in ENV_TRIGGERS else 'decision'
        self.phases = frozenset({'request' if self.kind == 'environment' else 'reply'})
        self.cfg = dict(rt.CALIBRATED_CONFIG, enabled=[name])

    async def observe(self, ev):
        if ev.scanner is None:
            return None
        view = None
        if ev.phase == 'reply':
            r = ev.reply
            view = r.action if r.action is not None else (None if r.parse_error else r.text)
            if view is None:
                return None
        for f in ev.scanner.current_fires(view, self.cfg):
            if f['trigger'] == self.name and f['kind'] == self.kind:
                return iv.Decision(self.name, f.get('reason') or self.name)
        return None


def would_fire_config(log):
    """The run's would_fire as a trigger config: its enabled relay_triggers triggers in the scanner's order, the
    interventions built-ins for all but done_claim (RouterWouldFire, first claim only), plus the done_claim built-in
    as `builtin_done_claim`, logged for comparison only."""
    names = [n for n in ENV_TRIGGERS if n in log.enabled] + [n for n in log.enabled if n not in ENV_TRIGGERS]
    specs = []
    for n in names:
        if n != 'done_claim' and n in iv.BUILTINS:
            specs += iv.load_config({'triggers': [{'use': n}]})[1]
        else:
            specs.append(iv.TriggerSpec(RouterWouldFire(n), make=functools.partial(RouterWouldFire, n)))
    h = 'msa' if log.harness == 'msa' else 'terminus2'
    specs.append(iv.TriggerSpec(iv.BUILTINS['done_claim'](harness=h), name='builtin_done_claim',
                                make=functools.partial(iv.BUILTINS['done_claim'], harness=h)))
    return iv.Policy(), specs


def compare_would_fire(ep, fires, info):
    """The logged would_fire vs the replayed fires, request by request: (rows, counts). A request the router logged no
    would_fire for (a reply it could not parse, an upstream error) is 'unlogged' when the replay fires there."""
    by_seq = collections.defaultdict(list)
    for f in fires:
        if f['trigger'] != 'builtin_done_claim':
            by_seq[f['seq']].append((f['trigger'], f['reason']))
    c, rows = collections.Counter(), []
    turns = set()
    for r in ep.log.records(ep.episode):
        if r.get('turn') is None or r.get('owner') == 'router':
            continue
        if 'would_fire' not in r:
            if by_seq.get(r['seq']):
                c['unlogged_requests'] += 1
                rows.append(dict(kind='unlogged', id=ep.id, seq=r['seq'], turn=r['turn'], replayed=by_seq[r['seq']],
                                 why=('parse error' if r.get('student_parse_error') else
                                      f"upstream {r.get('upstream_status')}" if r.get('upstream_status') != 200
                                      else 'no would_fire logged')))
            continue
        logged = sorted((f['trigger'], f.get('reason')) for f in r['would_fire'])
        mine = sorted(by_seq.get(r['seq'], []))
        c['requests'] += 1
        turns.add(r['turn'])
        c['logged_fires'] += len(logged)
        c['replayed_fires'] += len(mine)
        if logged == mine:
            c['matched'] += 1
        else:
            c['mismatched'] += 1
            rows.append(dict(kind='mismatch', id=ep.id, seq=r['seq'], turn=r['turn'], logged=logged, replayed=mine,
                             request_kind=r.get('request_kind')))
    c['turns'] = len(turns)
    return rows, c


# ==== CLI =====================================================================================================
async def _run(a, episodes, out, stats):
    rows, names, check_rows, check = [], [], [], collections.defaultdict(collections.Counter)
    builtin_extra = collections.Counter()
    cfg = None if a.builtin_check else load_trigger_config(a.config)
    cfgs = {}
    with open(os.path.join(out, 'per_episode.jsonl'), 'w') as pe, open(os.path.join(out, 'fires.jsonl'), 'w') as fe:
        for k, ep in enumerate(episodes):
            if a.limit and k >= a.limit:
                break
            if a.builtin_check:
                if ep.mode not in ('student', 'teacher'):
                    continue             # would_fire is logged in --mode student / teacher only
                if ep.log.dir not in cfgs:
                    cfgs[ep.log.dir] = would_fire_config(ep.log)
                c = cfgs[ep.log.dir]
            else:
                c = cfg
            recs = await areplay_episode(ep, c, a.reply_parse, a.per_trigger, stats)
            fires, summ = recs[:-1], recs[-1]
            for f in fires:
                fe.write(json.dumps(f) + '\n')
            pe.write(json.dumps(summ) + '\n')
            rows.append(summ)
            for s in c[1]:
                if s.name not in names:
                    names.append(s.name)
            if a.builtin_check:
                cr, cc = compare_would_fire(ep, fires, None)
                check_rows += cr
                check[ep.cls].update(cc)
                check[ep.cls]['episodes'] += 1
                dc = sum(1 for f in fires if f['trigger'] == 'done_claim')
                bdc = sum(1 for f in fires if f['trigger'] == 'builtin_done_claim')
                builtin_extra[ep.cls] += bdc - dc
    return rows, names, check_rows, check, builtin_extra


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', nargs='+', required=True,
                   help='manifest.json files and/or run dirs (holding router_<arm>/) or router dirs')
    p.add_argument('--config', help='trigger config (YAML), as interventions.load_config reads it')
    p.add_argument('--builtin-check', action='store_true',
                   help="replay each run's own would_fire config on its --mode student / teacher episodes and compare "
                        'with the logged would_fire, request by request (exit 1 on a student-alone mismatch)')
    p.add_argument('--out', required=True, help='output dir: per_episode.jsonl, fires.jsonl, summary.json')
    p.add_argument('--groups', help='comma list of groups to keep (manifest groups; <arm>_<t2|msa> for run dirs)')
    p.add_argument('--harness', choices=['terminus2', 'msa'])
    p.add_argument('--inputs', action='append', default=[],
                   help="roots holding the logs' cluster paths (tasks, tokenizers, Terminus-2's parser) under their "
                        'absolute paths; default <manifest or run dir parent>/_inputs')
    p.add_argument('--labels', help='JSON list or JSONL of {sid or id, point_of_no_return_turn}')
    p.add_argument('--reply-parse', choices=['harness', 'router'], default='harness',
                   help="harness: a reply's parse_error from the harness's own parser; router: only what that run's "
                        'router knew (forced by --builtin-check)')
    p.add_argument('--tokenizer', help='tokenizer.json for view_tokens counts the log lacks (default: the run\'s)')
    p.add_argument('--chat-template', default=TEMPLATE)
    p.add_argument('--terminus-parser', help="Terminus-2's terminus_json_plain_parser.py (default: the run's)")
    p.add_argument('--no-per-trigger', dest='per_trigger', action='store_false',
                   help='skip the per-trigger-alone live controllers (first_actionable per trigger)')
    p.add_argument('--limit', type=int, help='replay at most this many episodes')
    a = p.parse_args(argv)
    if bool(a.config) == bool(a.builtin_check):
        p.error('give --config or --builtin-check')
    if a.builtin_check:
        a.reply_parse = 'router'
    os.makedirs(a.out, exist_ok=True)
    episodes = load_episodes(a.runs, groups=a.groups.split(',') if a.groups else None, harness=a.harness,
                             inputs=a.inputs, labels=a.labels, tokenizer=a.tokenizer, chat_template=a.chat_template,
                             terminus_parser=a.terminus_parser)
    stats = collections.Counter()
    rows, names, check_rows, check, builtin_extra = asyncio.run(_run(a, episodes, a.out, stats))
    summary = summarize(rows, names)
    summary.update(config=a.config or 'builtin would_fire (per run)', reply_parse=a.reply_parse, triggers=names,
                   replay=dict(sorted(stats.items())), episodes=len(rows))
    with open(os.path.join(a.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(f"replayed {len(rows)} episodes: {stats['request_events']} request + {stats['reply_events']} reply events; "
          f"history exact {stats['history_exact']}, inexact {stats['history_inexact']}; "
          f"turn check {stats['turn_ok']} ok, {stats['turn_mismatch']} off")
    if stats['no_terminus_parser']:
        print(f"WARNING: Terminus-2's parser not found (--terminus-parser): {stats['no_terminus_parser']} replies "
              'got no parse_error check')
    print_table(summary)
    print(f'\nwrote {a.out}/per_episode.jsonl, fires.jsonl, summary.json')
    if not a.builtin_check:
        return 0
    res = dict(classes={k: dict(v) for k, v in check.items()}, builtin_done_claim_extra_fires=dict(builtin_extra),
               mismatches=[x for x in check_rows if x['kind'] == 'mismatch'],
               unlogged=[x for x in check_rows if x['kind'] == 'unlogged'])
    with open(os.path.join(a.out, 'builtin_check.json'), 'w') as f:
        json.dump(res, f, indent=1)
    print('\nbuilt-in check vs the logged would_fire (request by request, trigger and reason):')
    for cls, c in res['classes'].items():
        print(f"  {cls}: {c.get('episodes', 0)} episodes, {c.get('requests', 0)} requests ({c.get('turns', 0)} turns), "
              f"logged fires {c.get('logged_fires', 0)}, replayed {c.get('replayed_fires', 0)}; "
              f"matched {c.get('matched', 0)}/{c.get('requests', 0)} = "
              f"{_pct(_r(c.get('matched', 0), c.get('requests', 0)))}"
              f"; requests with no would_fire logged but a replayed fire: {c.get('unlogged_requests', 0)}")
    for x in res['mismatches'][:20]:
        print(f"  MISMATCH {x['id']} seq {x['seq']} turn {x['turn']}: logged {x['logged']} replayed {x['replayed']}")
    print(f"  done_claim built-in (relay semantics) vs would_fire: {dict(builtin_extra)} extra fires "
          '(it fires on every claim; the scanner on the first only)')
    bad = res['classes'].get('student_alone', {}).get('mismatched', 0)
    print(f"\nGATE {'GO' if not bad else 'NO-GO'}: {bad} student-alone mismatches")
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
