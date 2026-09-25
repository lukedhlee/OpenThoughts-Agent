#!/usr/bin/env python3
"""Relay-rollout triggers: causal detectors a live router can run on an agent episode.

A router sits between the agent harness (Harbor Terminus-2, or mini-swe-agent) and the models. On every request it
sees the OpenAI-style message list so far; for decision triggers it also sees the student's candidate reply. When a
trigger fires, the teacher takes over for the rest of the episode.

Two kinds of trigger (see ai_memory/active/snowball-sft/research/2026-09-24_relay_rollouts_design.md):
  * environment triggers fire on what the sandbox already returned (the new output of the previous turn's commands);
    the teacher writes the turn about to be generated.
  * decision triggers fire on the student's candidate reply for this turn; the router discards it and the teacher
    answers the same request.
Either way a fire "at turn t" means the teacher writes agent turn t (1-based) and every turn after it.

Calibrated set (2026-09-25, ai_memory/active/snowball-sft/research/2026-09-25_trigger_calibration/README.md):
takeover on done_claim, loop (3 identical keystroke turns of the last 6) and no_progress_wait (3 passive waits in a
row with no new output); gave_up, edit_failed, error_streak, submit_check, success_* and destructive are LOGGED only.

Router use (stateless):
    fires = detect(messages, reply=candidate_text)                        # takeover triggers (DEFAULT_CONFIG)
    fires = detect(messages, reply=candidate_text, include_logged=True)   # plus logged signals, kind='logged'
Router use (incremental, survives Terminus-2 context summarization, which replaces the message list):
    sc = EpisodeScanner(); per new message sc.add_message(m); then sc.flush(); fires = sc.current_fires(reply, config)
Offline (calibration): scan_messages(messages) -> EpisodeScanner with per-turn evidence; fires_by_turn(sc, cfg).

Every detector reads only the NEW output since the previous command (never tmux scrollback) and only messages at or
before the request. Ported by diff from the forensics scanner
(ai_memory/active/snowball-sft/research/2026-09-25_relay_trigger_forensics/scanner/scan.py); changes are marked
"relay:" in comments.
"""
import difflib
import fnmatch
import json
import os
import re

# ==== regexes and command classes (verbatim from scan.py unless marked) ====================================
PROMPT_RE = re.compile(r'^(?:\([^)\n]*\)\s*)?[\w.-]+@[\w.-]+:[^\n#$]*[#$] ?(.*)$')
# relay: bare prompts (`$ cmd`, `# cmd`; e.g. the JSC apptainer R2E-Gym sandboxes). A bare line only counts as a
# prompt when its text is the echo of one of this turn's commands (or `$` alone), so `# comment` lines in file
# listings do not split segments.
BARE_PROMPT_RE = re.compile(r'^(?:\([^)\n]*\)\s*)?([#$])(?: (.*))?$')


def prompt_match(line, cmd_lines=()):
    """Return the echoed command text if `line` is a shell prompt line, else None."""
    mm = PROMPT_RE.match(line)
    if mm:
        return mm.group(1)
    mb = BARE_PROMPT_RE.match(line)
    if not mb:
        return None
    rest = (mb.group(2) or '').strip()
    if not rest:
        return '' if mb.group(1) == '$' else None
    return rest if any(_same_cmd(rest, c) for c in cmd_lines) else None
ERR_RE = re.compile(r'Traceback|error:|command not found|No such file or directory|make: \*\*\*|'
                    r'undefined reference|fatal:|Segmentation fault|npm ERR!|FAILED|'
                    r'E: Unable to locate package|ModuleNotFoundError')
# relay: a broader error signature, a tuning option for loop / error_streak
ERR_BROAD_RE = re.compile(ERR_RE.pattern + r'|\b\w*(?:Error|Exception)\b|\berror\b|\bfailed\b|'
                          r'\bcannot\b|\bnot found\b|Permission denied|Killed|Aborted')

_REDIR = r'>>?\s*(?!/dev/null|&)[\'"]?[\w./~$-]'
MODIFY_RES = [
    re.compile(r'\bsed\s+(?:-\w*\s+)*-\w*i'), re.compile(r'\bsed\s+.*--in-place'),
    re.compile(r'\bperl\s+-\w*i'),
    re.compile(r'\bcat\s*' + _REDIR), re.compile(r'\bcat\s+<<.*?' + _REDIR),
    re.compile(r'<<-?\s*[\'"]?\w+[\'"]?\s*' + _REDIR),
    re.compile(r'(?:<<|\b(?:echo|printf|cat)\b[^|\n]*\|)\s*[^\n]*\btee\s+(?:-a\s+)?(?!/dev/null)[\w./~$-]'),
    re.compile(r'(?:^|[;&|\s])patch\s'), re.compile(r'\bgit\s+apply\b'),
    re.compile(r'\b(?:echo|printf)\b[^\n|]*' + _REDIR),
    re.compile(r'\btruncate\s'),
]
PY_RE = re.compile(r'\bpython[\d.]*\b')
PY_WRITE_RE = re.compile(r'open\([^)]*[\'"][wa]b?\+?[\'"]|write_text\(|write_bytes\(|\.write\(')
CHECK_RE = re.compile(
    r'\b(?:pytest|py\.test|tox|nosetests|unittest|make|cmake|ninja|cargo|go\s+(?:test|build|run)|'
    r'npm\s+(?:test|run|start)|yarn|node|gcc|g\+\+|cc|clang\+?\+?|javac|java|mvn|gradle|rustc|Rscript|R|'
    r'julia|ruby|php|perl|runtests\.py|manage\.py|django-admin|curl|wget|diff|cmp|sqlite3|psql|'
    r'mypy|flake8|pylint|ruff|tsc|dotnet|ghc|ocaml|lua|qemu[\w-]*|bash|sh)\b|(?:^|[\s;&|])\./[\w.-]+|'
    r'(?:^|[;&|]\s*)/(?:app|testbed|usr/local/bin|opt|root|tmp)/[\w./-]+(?:\s|$)|'
    r'\bpython[\d.]*\s+(?:-m\s|-c\s|-\s|<<|[\w./-]+\.py)')


def is_modify(k, writer_scripts=()):
    if any(r.search(k) for r in MODIFY_RES):
        return True
    if PY_RE.search(k) and PY_WRITE_RE.search(k):
        return True
    m = re.search(r'\bpython[\d.]*\s+(?:-\w+\s+)*([\w./~-]+\.py)\b', k)
    return bool(m and os.path.basename(m.group(1)) in writer_scripts)


def written_scripts(k):
    if not PY_WRITE_RE.search(k):
        return set()
    if not re.search(r'[\'"](?!/tmp/)[\w./-]+\.(?:py|pyx|c|h|cc|cpp|js|ts|rs|go|java|rb|R|jl)[\'"]', k):
        return set()
    return {os.path.basename(x) for x in re.findall(r'(?:cat\s*>|>\s*|tee\s+)[\'"]?([\w./~-]+\.py)', k)}


def is_check(k):
    return bool(CHECK_RE.search(k))


DESTRUCT_RES = [
    ('rm -rf', re.compile(r'\brm\s+(?:-\w+\s+)*-\w*[rR]\w*\b[^\n;&|]*')),
    ('git reset --hard', re.compile(r'\bgit\s+reset\s+--hard')),
    ('git checkout -- .', re.compile(r'\bgit\s+(?:checkout\s+(?:HEAD\s+)?(?:--\s+)?|restore\s+(?:--\S+\s+)*)\.(?:\s|$|;|&)')),
    ('git clean', re.compile(r'\bgit\s+clean\b')),
    ('rm test file', re.compile(r'\brm\s+[^\n;&|]*(?:\btests?/|test_[\w]*\.py|_test\.\w+|\.test\.\w+)')),
    ('truncate', re.compile(r'(?:^|[;&|]\s*)(?::\s*)?>\s*[\w./-]+\s*(?:$|[;&|])|\btruncate\s+-s\s*0|cat\s+/dev/null\s*>')),
]
BENIGN_RM = re.compile(r'(?:/tmp/|/var/tmp|__pycache__|\.pytest_cache|\bbuild\b|\bdist\b|\.egg-info|'
                       r'node_modules|\.cache|\*\.pyc|\.o\b|\bvenv\b|\.venv|/root/\.|\btarget\b|CMakeFiles|'
                       r'\bout\b|\bobj\b|\btmp\b|\btemp\b|\.lock\b)')


def created_names(k):
    names = set()
    for m in re.finditer(r'\bmkdir\s+(?:-p\s+)?([^\n;&|]+)', k):
        names |= {os.path.basename(x.rstrip('/')) for x in m.group(1).split() if not x.startswith('-')}
    for m in re.finditer(r'(?:>>?|\btee\s+(?:-a\s+)?|\btouch\s+|-o\s*|--output[= ]|\bcp\s+(?:-\w+\s+)*\S+\s+|'
                         r'\bmv\s+\S+\s+|open\([\'"])[\'"]?([\w./~*-]+)', k):
        names.add(os.path.basename(m.group(1).rstrip('/')))
    for m in re.finditer(r'\bvenv\s+([\w./-]+)|--target[= ]([\w./-]+)|\bpip\d?\s+install\b[^\n;&|]*\s-t\s+([\w./-]+)', k):
        names.add(os.path.basename((m.group(1) or m.group(2) or m.group(3)).rstrip('/')))
    for m in re.finditer(r'\bgit\s+clone\s+(?:-\S+\s+)*(\S+)(?:\s+([^\s;&|]+))?', k):
        names.add(os.path.basename(m.group(2) or re.sub(r'\.git$', '', m.group(1).rstrip('/'))))
    for m in re.finditer(r'\b(?:wget|curl)\b[^\n;&|]*?(https?://\S+)', k):
        b = os.path.basename(m.group(1).strip('\'"').rstrip('/'))
        names |= {b, re.sub(r'(\.tar)?\.(gz|tgz|bz2|xz|zip)$', '', b)}
    for m in re.finditer(r'\b(?:tar|unzip)\b[^\n;&|]*?([\w.-]+\.(?:tar\.gz|tgz|tar\.bz2|tar\.xz|zip|tar))', k):
        names.add(re.sub(r'(\.tar)?\.(gz|tgz|bz2|xz|zip)$|\.tar$', '', m.group(1)))
    if re.search(r'makedirs|\.mkdir\(|open\([^)]*[\'"][wa]|to_csv|to_parquet|\.save\(|write_text', k):
        names |= {os.path.basename(x.rstrip('/')) for x in re.findall(r'[\'"]([\w./~-]{3,})[\'"]', k)}
    return {n for n in names if n and n not in ('.', '..', '/', 'app', 'testbed', 'src', 'tests', 'test')}


def destructive_hits(k, created=frozenset()):
    hits = []
    for name, r in DESTRUCT_RES:
        for m in r.finditer(k):
            s = m.group(0)
            if name == 'rm -rf':
                tgts = [t for t in s.split()[1:] if not t.startswith('-')]

                def benign(t):
                    b = os.path.basename(t.rstrip('/'))
                    return BENIGN_RM.search(t) or any(fnmatch.fnmatch(c, b) or c == b for c in created)
                if not tgts or all(benign(t) for t in tgts):
                    continue
            if name == 'truncate' and re.search(r'/dev/null|/tmp/', s):
                continue
            if name == 'rm test file':
                tgts = [t for t in s.split()[1:] if not t.startswith('-')]
                if all(os.path.basename(t.rstrip('/')) in created or t.startswith('/tmp/') or
                       any(fnmatch.fnmatch(c, os.path.basename(t)) for c in created) for t in tgts):
                    continue
            hits.append((name, s.strip()[:120]))
    return hits


# ---- text classes ----------------------------------------------------------------------------------------
SUCCESS_RE = re.compile(
    r'\btests?\s+(?:now\s+)?pass(?:es|ed)?\b|\ball\s+\d*\s*(?:the\s+)?tests?\b[^.]*\bpass|'
    r'\b(?:is|are|was|were|been|now|has|have|got|successfully)\s+(?:now\s+)?(?:been\s+)?(?:properly\s+|correctly\s+)?fixed\b|^\W*fixed\b|'
    r'\b(?:it|this|fix|solution|script|code|program|implementation|everything|now|patch|change|regex|output)\s+'
    r'(?:now\s+|still\s+)?works\b|\bworks\s+(?:correctly|properly|as\s+expected|now)\b|'
    r'\bsuccessfully\s+(?:\w+\s+){0,1}(?:applied|fixed|implemented|created|completed|written|updated|modified|'
    r'resolved|passed|patched|generated|saved|built|compiled|solved|added|replaced|changed|made|wrote|produced)\b|'
    r'\b(?:applied|fixed|implemented|created|completed|written|updated|modified|resolved|passed|patched|'
    r'generated|saved|built|compiled|solved|added|replaced|changed|made|produced)\s+(?:\w+\s+){0,1}successfully\b|'
    r'\b(?:task|fix|implementation|solution|work|changes?|everything|all|job|requirements?|issue)\s+'
    r'(?:is|are|has\s+been|have\s+been|was|were)\s+(?:now\s+)?(?:fully\s+|successfully\s+)?(?:complete|completed|done|resolved)\b|'
    r'\b(?:is|was|were|are)\s+(?:now\s+)?successful\b|'
    r'\btask\s+(?:is\s+)?complete\b', re.I)
HEDGE_RE = re.compile(r"\b(?:not|no|never|n't|still|yet|need|needs|should|will|would|could|might|may|must|"
                      r"if|whether|verify|check|ensure|confirm|let me|let's|once|until|expect|expected|"
                      r"hopefully|try|attempt|want|goal|to be|required|requires|make sure|then|next|"
                      r"remaining|to complete|to be fixed|before|but|however|fail\w*|error|wrong|broken|issue\s+is)\b", re.I)
AGENCY = r"(?:\b(?:I|we|I'll|we'll|I'm|we're|let\s+me|let's|will|going\s+to|just|our)\b)"
# relay: the scan.py GIVEUP_RES list, split into named phrase families so each can be tuned on/off
GIVEUP_FAMILIES = {
    'stand_in': re.compile(r'\b(?:placeholder|stub|dummy|fake)\s+(?:move|value|data|output|file|model|results?|answer|'
                           r'solution|implementation|version|binary|weights|image|content|values?|response|entries|text|count)\b|'
                           r'\bas\s+a\s+placeholder\b|\bplaceholder\s+for\s+now\b', re.I),
    'mock': re.compile(r'\bmock(?:ed|ing)?\s+(?:implementation|version|data|output|model|results?|server|'
                       r'response|binary|weights)\b', re.I),
    'simplify': re.compile(r'\bsimplif(?:y|ied|ying)\s+(?:the\s+)?(?:task|problem|requirements?|implementation|version|solution)\b|'
                           r'\bsimplified\s+(?:version|implementation|solution)\b', re.I),
    'workaround': re.compile(r'\bwork\s*around\b|\bworkaround\b', re.I),
    'skip': re.compile(AGENCY + r'[^.]{0,40}\bskip(?:ping)?\s+(?:the\s+|this\s+|that\s+)?(?:test|tests|step|requirement|'
                       r'verification|check|build|compil\w*|install\w*|download\w*|part|it)\b', re.I),
    'instead_of_real': re.compile(AGENCY + r'[^.]{0,80}\binstead\s+of\s+(?:the\s+|a\s+|an\s+)?(?:real|actual|full|original|required|'
                                  r'requested|specified|proper|complete|official|exact)\b', re.I),
    'impossible_req': re.compile(r'\b(?:cannot|can\'t|can\s+not|unable\s+to|not\s+possible|impossible|no\s+way\s+to)\b[^.]{0,80}'
                                 r'\b(?:task|requirement|required|spec|specified|as\s+requested|exactly|achieve|satisfy|'
                                 r'solve|internet|network|without\s+(?:internet|network|access))\b', re.I),
    'cannot_install_pivot': re.compile(r'\b(?:cannot|can\'t|can\s+not|unable\s+to|not\s+possible)\b[^.]{0,60}\b(?:install|download|internet|'
                                       r'network|access|package|pip|apt|library|dependency|compiler|toolchain)\b[^.]{0,80}\b(?:instead|manually|'
                                       r'alternative|fallback|workaround|so\s+(?:we|I)\s*(?:\'ll|will)|we\'ll\s+have\s+to)\b', re.I),
    # relay: new families, tuning candidates (off unless calibration keeps them)
    'hardcode': re.compile(r'\bhard[- ]?cod(?:e|ed|ing)\s+(?:the\s+)?(?:expected|output|answer|result|values?|response)\b', re.I),
    'give_up_words': re.compile(r'\b(?:give\s+up|giving\s+up|out\s+of\s+(?:options|ideas)|best\s+(?:we|I)\s+can\s+do|'
                                r'accept\s+(?:the\s+)?(?:current|partial)|good\s+enough)\b', re.I),
}


def sentences(t):
    return [s for s in re.split(r'(?<=[.!?])\s+|\n+', t or '') if s.strip()]


def strip_code(s):
    return re.sub(r'`[^`]*`', ' ', s)


def success_sentences(text):
    out = []
    for s in sentences(text):
        s2 = strip_code(s)
        if SUCCESS_RE.search(s2) and not HEDGE_RE.search(s2):
            out.append(s.strip())
    return out


def giveup_hits(text):
    """relay: {family: first matching sentence} (scan.py giveup_sentences, per family)."""
    out = {}
    for s in sentences(text):
        s2 = strip_code(s)
        for fam, r in GIVEUP_FAMILIES.items():
            if fam not in out and r.search(s2):
                out[fam] = s.strip()[:220]
    return out


# relay: scan.py EDITFAIL_RE split into named families
EDITFAIL_FAMILIES = {
    'sed_error': re.compile(r'sed: -e expression|sed: can\'t read|unterminated `s\' command|unknown option to `s\''),
    'patch_fail': re.compile(r'Hunk #\d+ FAILED|can\'t find file to patch|malformed patch|patch: \*\*\*\*|'
                             r'error: patch failed|corrupt patch|does not apply'),
    'heredoc_eof': re.compile(r'here-document at line \d+ delimited by end-of-file|unexpected EOF while looking for matching'),
    'shell_syntax': re.compile(r'syntax error near unexpected token'),
    'replace_notfound': re.compile(r'(?i:(?:old|pattern|target|search|original)\s+(?:code|string|text|pattern|block)?\s*'
                                   r'(?:was\s+)?not\s+found)|(?i:no\s+(?:changes|replacements)\s+made)'),
}
PYERR_RE = re.compile(r'(SyntaxError|IndentationError|TabError|ImportError|ModuleNotFoundError)')
NOFILE_RE = re.compile(r'No such file or directory')

# ==== message parsing =======================================================================================
THINK_RE = re.compile(r'<think>.*?</think>|<\|start_think\|>.*?<\|end_think\|>', re.S)
HDR_RE = re.compile(r'^(New Terminal Output:|Current Terminal Screen:|Current terminal state:)\s*$', re.M)
BASH_BLOCK_RE = re.compile(r'```(?:bash|sh|shell)?\s*\n(.*?)```', re.S)
MINI_DONE_RE = re.compile(r'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT|MINI_SWE_AGENT_FINAL_OUTPUT')
RC_RE = re.compile(r'<returncode>\s*(-?\d+)\s*</returncode>')
MINI_OUT_RE = re.compile(r'<output(?:_head|_tail)?>\n?(.*?)</output(?:_head|_tail)?>', re.S)


def _json_obj(body):
    """First balanced top-level JSON object in body that parses (Terminus-2 replies)."""
    i = body.find('{')
    while i != -1:
        depth, ins, esc = 0, False, False
        for j in range(i, len(body)):
            c = body[j]
            if ins:
                if esc:
                    esc = False
                elif c == '\\':
                    esc = True
                elif c == '"':
                    ins = False
                continue
            if c == '"':
                ins = True
            elif c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    try:
                        o = json.loads(body[i:j + 1])
                        if isinstance(o, dict):
                            return o
                    except Exception:
                        pass
                    break
        i = body.find('{', i + 1)
    return None


def parse_reply(content, harness=None):
    """Assistant reply -> dict(harness, ok, analysis, plan, cmds[(keystrokes, duration)], done)."""
    if isinstance(content, dict) and 'cmds' in content:
        return content
    content = content if isinstance(content, str) else json.dumps(content)
    raw = content
    if '<|start_think|>' in content and '<|end_think|>' not in content:
        content = ''   # unterminated thinking: no action (scan.py parse_message)
    body = THINK_RE.sub('', content).strip()
    if harness is None:
        harness = 'miniswe' if (BASH_BLOCK_RE.search(body) and '"keystrokes"' not in body) else 'terminus2'
    if harness == 'miniswe':
        blocks = BASH_BLOCK_RE.findall(body)
        prose = BASH_BLOCK_RE.sub(' ', body)
        cmds = [(b.strip() + '\n', None) for b in blocks[:1]]   # mini-swe-agent runs exactly one block
        done = bool(blocks) and bool(MINI_DONE_RE.search(blocks[0]))
        return dict(harness=harness, ok=len(blocks) == 1, analysis=prose.strip(), plan='', cmds=cmds, done=done, raw=raw)
    if body.startswith('Analysis:'):   # ATIF message form (Harbor stores parsed replies this way)
        a, _, p = body[len('Analysis:'):].partition('\nPlan:')
        return dict(harness=harness, ok=True, analysis=a.strip(), plan=p.strip(), cmds=[], done=False, raw=raw)
    o = _json_obj(body)
    if o is None:
        return dict(harness=harness, ok=False, analysis='', plan='', cmds=[], done=False, raw=raw)
    cmds = []
    for c in o.get('commands') or []:
        if isinstance(c, dict):
            cmds.append((str(c.get('keystrokes', '')), c.get('duration')))
    tc = o.get('task_complete', False)
    done = tc is True or (isinstance(tc, str) and tc.strip().lower() == 'true')
    return dict(harness=harness, ok=True, analysis=str(o.get('analysis', '')), plan=str(o.get('plan', '')),
                cmds=cmds, done=done, raw=raw)


def tool_reply(m):
    """OpenAI tool-calling assistant message (mini-swe-agent tool variant) -> parsed reply dict.
    The prose is `content` only; `reasoning_content` (thinking) is not judged."""
    tcs = m.get('tool_calls')
    if isinstance(tcs, str):
        try:
            import ast
            tcs = ast.literal_eval(tcs)
        except Exception:
            tcs = []
    cmds = []
    for tc in tcs or []:
        fn = tc.get('function') or tc
        args = fn.get('arguments')
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {'command': args}
        cmd = (args or {}).get('command') or (args or {}).get('cmd') or ''
        cmds.append((str(cmd) + '\n', None))
    prose = THINK_RE.sub('', m.get('content') or '')
    done = any(MINI_DONE_RE.search(c) for c, _ in cmds)
    return dict(harness='tools', ok=bool(cmds), analysis=prose.strip(), plan='', cmds=cmds, done=done,
                raw=m.get('content') or '')


def screen_part(o):
    ms = list(HDR_RE.finditer(o))
    if not ms:
        return None, ''
    m = ms[-1]
    txt = o[m.end():]
    txt = re.split(r'\n\s*Are you sure you want to mark the task as complete', txt)[0]
    kind = m.group(1)
    if kind == 'Current terminal state:':   # relay: the confirmation turn shows the visible screen
        kind = 'Current Terminal Screen:'
    return kind, txt.strip('\n').rstrip()


def new_output(kind, txt, cmds, prev_screen):
    if kind != 'Current Terminal Screen:':
        return txt
    lines = txt.split('\n')
    first = next((c.strip().split('\n')[0].strip() for c in cmds if c.strip()), '')
    if first:
        for i, l in enumerate(lines):
            echo = prompt_match(l, [first])
            if echo is not None and _same_cmd(echo, first):
                return '\n'.join(lines[i:])
    if prev_screen:
        pl = prev_screen.split('\n')
        best = 0
        for k in range(min(len(lines), len(pl)), 0, -1):
            if lines[:k] == pl[-k:]:
                best = k
                break
        return '\n'.join(lines[best:])
    return txt


def _same_cmd(echo, cmd):
    e, c = echo.strip(), cmd.strip().split('\n')[0].strip()
    n = min(len(e), len(c), 30)
    return n >= 2 and e[:n] == c[:n]


def split_segments(txt, cmds):
    kjoin = '\n'.join(cmds)
    kflat = re.sub(r'\s+', '', kjoin)
    lead, segs, cur = [], [], None
    cmd_lines = [x.strip() for c in cmds for x in c.split('\n') if x.strip()]
    for l in txt.split('\n'):
        echo = prompt_match(l, cmd_lines)
        if echo is not None:
            cur = {'cmd': echo.strip(), 'body': [], 'echo': True}
            segs.append(cur)
            continue
        if cur is None:
            lead.append(l)
            continue
        if cur['echo']:
            s = l[1:].strip() if l.startswith('>') else l.strip()
            if l.startswith('>') or (s and re.sub(r'\s+', '', s) in kflat):
                continue
            cur['echo'] = False
        cur['body'].append(l)
    segs = [dict(cmd=s['cmd'], body='\n'.join(s['body']).strip()) for s in segs
            if s['cmd'] or '\n'.join(s['body']).strip()]
    return '\n'.join(lead).strip(), segs


def norm_keys(cmds):
    ks = [re.sub(r'\s+', ' ', c).strip() for c in cmds]
    ks = [k for k in ks if k]
    return tuple(ks) if ks else ('<WAIT>',)


HOST_RE = re.compile(r'[\w.-]+@[\w.-]+:')


def norm_out(s):
    s = HOST_RE.sub('@:', s)
    s = re.sub(r'\d+\.\d+s|\d{2}:\d{2}:\d{2}', '#', s)
    return re.sub(r'\s+', ' ', s).strip()


IDLE_RE = re.compile(r'^(?:<WAIT>|sleep\s+[\d.]+|wait|C-c|C-d|Enter|q|)$')
# relay: a "wait" for the no-progress trigger is only passive waiting (not C-c / q, which intervene)
WAIT_RE = re.compile(r'^(?:<WAIT>|sleep\s+[\d.]+|wait|Enter|)$')


def _target(c):
    return set(re.findall(r'[\w./~-]+\.(?:py|pyx|c|h|cc|cpp|hpp|js|ts|rs|go|java|'
                          r'rb|sh|txt|json|ya?ml|toml|cfg|ini|R|jl|html|css|md|mk)\b', c))


def _pyerr_on_target(tgts, body):
    bases = {os.path.basename(t) for t in tgts}
    for m in PYERR_RE.finditer(body):
        kind = m.group(1)
        line = body[m.start():body.find('\n', m.start()) if '\n' in body[m.start():] else len(body)]
        if kind in ('SyntaxError', 'IndentationError', 'TabError'):
            files = re.findall(r'File "([^"]+)", line \d+', body[:m.start()])
            if files and os.path.basename(files[-1]) in bases:
                return '%s in %s' % (kind, os.path.basename(files[-1]))
        else:
            for t in tgts:
                if not t.endswith('.py'):
                    continue
                parts = [x for x in t[:-3].split('/') if x not in ('', '.', '~', 'testbed', 'app')]
                mod = '.'.join(parts[-2:]) if len(parts) >= 2 else (parts[-1] if parts else '')
                if mod and (mod in line or ("'%s'" % parts[-1]) in line) and parts[-1] != '__init__':
                    return '%s naming %s' % (kind, mod)
    return None


def edit_fail_hits(tgt, body, cmd, own_output=False):
    """relay: scan.py _edit_fail returning every matching family as {family: detail}."""
    hits = {}
    if not body:
        if re.match(r'git\s+diff\b', cmd) and not own_output:
            hits['git_diff_empty'] = 'git diff empty after edit'
        return hits
    for fam, r in EDITFAIL_FAMILIES.items():
        m = r.search(body)
        if m:
            hits[fam] = m.group(0)
    if own_output and NOFILE_RE.search(body):
        hits['nofile_edit'] = 'No such file (edit cmd)'
    p = _pyerr_on_target(tgt, body)
    if p:
        hits['pyerr_target'] = p
    return hits


# ---- submit-time check (relay: new) ---------------------------------------------------------------------
STUB_RE = re.compile(r'\b(?:TODO|FIXME|placeholder|NotImplementedError|not\s+implemented)\b', re.I)
STUB_PASS_RE = re.compile(r'^\s*def [^\n]*:\s*\n\s*pass\s*$', re.M)   # a function whose whole body is `pass`
TEST_PATH_RE = re.compile(r'(?:^|/)(?:tests?/|test_[\w]*\.py|[\w]*_test\.\w+|[\w]*\.test\.\w+|conftest\.py)')
SKIP_RE = re.compile(r'pytest\.mark\.skip|@unittest\.skip|\.skipTest\(|pytest\.skip\(|@pytest\.mark\.xfail')


def _written_content(k):
    """Text a file-writing command puts into files (heredoc bodies, python -c strings)."""
    m = re.search(r'<<-?\s*[\'"]?(\w+)[\'"]?[^\n]*\n(.*?)\n\s*\1\s*$', k, re.S | re.M)
    if m:
        return m.group(2)
    return k


SIM_BACK = 8

# ==== incremental episode scanner ===========================================================================
class EpisodeScanner:
    """Feeds on an episode message by message; keeps per-turn evidence that is causal by construction.

    turn t (1-based) = the t-th assistant reply.
      env[t]  : evidence visible in the request for turn t (output of turns < t)
      dec[t]  : evidence in the student's reply for turn t (plus history)
    """

    def __init__(self, harness=None):
        self.harness = harness
        self.turn = 0
        self.env = {}              # t -> dict
        self.dec = {}              # t -> dict
        self.actions = []          # per executed turn: dict(turn, keys, keystr, osig, err, err_broad, wait, empty, dur, ts)
        self.segs_all = []         # per-command output segments (error_streak)
        self.cmdlog = []           # (turn, keystrokes, is_modify, is_check)
        self.pending_edits = []
        self.writers, self.created = set(), set()
        self.prev_screen = ''
        self.done_turn = None
        self.n_modify = 0
        self.stub_writes = []      # (turn, snippet)
        self.test_tamper = []      # (turn, what)
        self._last = None          # parsed reply awaiting its observation
        self._last_ts = None
        self.turn_act = {}         # turn -> index of its executed action
        self._tool_buf = []
        self.turn_ts = {}
        self._env_for(1)

    # -- feeding --
    def add_message(self, m, ts=None):
        role, content = m.get('role'), m.get('content')
        if isinstance(content, list):   # OpenAI content parts
            content = '\n'.join(p.get('text', '') for p in content if isinstance(p, dict))
        content = content or ''
        if role == 'tool' or m.get('orig_role') == 'tool':
            self._tool_buf.append(content)      # tool-calling harness: one result per call
            return
        if self._tool_buf:
            self.flush()
        if role == 'assistant':
            if m.get('tool_calls'):
                self.add_reply(tool_reply(m), ts)
            else:
                self.add_reply(content, ts)
        elif role == 'user' and self.turn > 0:
            self.add_observation(content, ts)

    def flush(self):
        """Close the pending turn (tool results collected so far, or no observation at all)."""
        if self._tool_buf:
            buf, self._tool_buf = self._tool_buf, []
            self.add_observation(buf, None)
        elif self._last is not None:
            self.add_observation('', None)

    def add_reply(self, content, ts=None):
        if self._last is not None:        # two replies in a row: previous one got no observation
            self.add_observation('', None)
        self.turn += 1
        t = self.turn
        self.turn_ts[t] = ts
        r = parse_reply(content, self.harness)
        if self.harness is None and r['ok']:
            self.harness = r['harness']
        self.dec[t] = self._decision_evidence(t, r)
        self._last, self._last_ts = r, ts
        if r['done'] and self.done_turn is None:
            self.done_turn = t

    def add_observation(self, content, ts=None):
        r, t = self._last, self.turn
        self._last = None
        if r is None:
            return
        if isinstance(content, str) and 'Previous response had parsing errors' in content and r['ok']:
            # the harness rejected this reply (nothing ran); a router mirroring Terminus-2's parser rejects it too
            r = dict(r, ok=False, cmds=[], done=False)
            self.dec[t] = self._decision_evidence(t, dict(r, analysis='', plan=''))
            if self.done_turn == t:
                self.done_turn = None
        self._absorb(t, r, content, self._last_ts)
        self._env_for(t + 1)

    # -- decision evidence (reply t) --
    def _decision_evidence(self, t, r):
        ev = dict(ok=r['ok'], done=r['done'] and self.done_turn is None, gave_up={}, success=None,
                  success_unchecked=None, success_contradicted=None, destructive=[], submit={})
        cmds = [c for c, _ in r['cmds']]
        if r['ok']:
            text = r['analysis'] + '\n' + r['plan']
            ev['gave_up'] = giveup_hits(text)
            ss = success_sentences(text)
            if ss:
                ev['success'] = ss[0][:200]
                mod_idx = max((i for i, c in enumerate(self.cmdlog) if c[2]), default=None)
                if mod_idx is not None and not any(c[3] for c in self.cmdlog[mod_idx + 1:]):
                    ev['success_unchecked'] = (ss[0][:200], self.cmdlog[mod_idx][1][:120])
                last = self.actions[-1] if self.actions else None
                if last is not None and last['err']:
                    ev['success_contradicted'] = (ss[0][:200], last['err_sig'])
        ev['destructive'] = [h for c in cmds for h in destructive_hits(c, self.created)]
        if ev['done']:
            ev['submit'] = self._submit_flags(t)
        return ev

    def _submit_flags(self, t):
        f = {}
        if self.n_modify == 0:
            f['no_edit'] = 'no file-modifying command in the episode'
        for _, kind, what in self.stub_writes + self.test_tamper:
            f[kind] = what
        return f

    # -- absorb the observation of turn t --
    def _absorb(self, t, r, o, ts):
        cmds = [c for c, _ in r['cmds']]
        durs = [d for _, d in r['cmds']]
        for c in cmds:
            mod = is_modify(c, self.writers)
            self.cmdlog.append((t, c, mod, is_check(c) and not mod))
            if mod:
                self.n_modify += 1
                w = _written_content(c)
                if not TEST_PATH_RE.search(c):
                    sm = STUB_RE.search(w)
                    if sm:
                        self.stub_writes.append((t, 'stub_marker', sm.group(0).strip()[:60]))
                    if STUB_PASS_RE.search(w):
                        self.stub_writes.append((t, 'stub_pass', 'def ...: pass'))
                if TEST_PATH_RE.search(c) and SKIP_RE.search(c):
                    self.test_tamper.append((t, 'tests_skip', 'skip marker written into a test file'))
                if re.search(r'\bsed\s+[^\n]*\d+(?:,\d+)?d\b', c) and TEST_PATH_RE.search(c):
                    self.test_tamper.append((t, 'tests_skip', 'sed deleted lines in a test file'))
            for name, s in destructive_hits(c, self.created):
                if name == 'rm test file':
                    self.test_tamper.append((t, 'tests_rm', 'rm test file: ' + s[:80]))
            self.writers |= written_scripts(c)
            self.created |= created_names(c)
        parse_err = isinstance(o, str) and 'Previous response had parsing errors' in o
        if parse_err or (not cmds and not r['done'] and r['harness'] in ('miniswe', 'tools')):
            return
        if r['harness'] in ('miniswe', 'tools'):
            self._absorb_rc(t, r, o if isinstance(o, list) else [o], cmds, ts)
            return
        if not r['ok']:
            return
        kind, scr = screen_part(o)
        new = new_output(kind, scr, cmds, self.prev_screen) if kind else ''
        if kind:
            self.prev_screen = scr
        lead, segs = split_segments(new, cmds)
        touched = []
        if lead and self.segs_all:
            self.segs_all[-1]['body'] += '\n' + lead
            self.segs_all[-1]['err'] = bool(ERR_RE.search(self.segs_all[-1]['body']))
            touched.append(len(self.segs_all) - 1)
        for s in segs:
            s.update(turn=t, err=bool(ERR_RE.search(s['body'])))
            self.segs_all.append(s)
            touched.append(len(self.segs_all) - 1)
        self._touched = touched
        # edit_failed evidence (from this turn's output) -> becomes env evidence for turn t+1
        ef = {}
        body_by_cmd = [s['body'] for s in segs]
        if self.pending_edits and body_by_cmd:
            pt, pc, pb = self.pending_edits.pop()
            for fam, d in edit_fail_hits(pb, body_by_cmd[0], segs[0]['cmd']).items():
                ef.setdefault(fam, (pc[:100], d))
        self.pending_edits = []
        for c in cmds:
            if not is_modify(c, self.writers):
                continue
            idx = next((i for i, sg in enumerate(segs) if _same_cmd(sg['cmd'], c)), None)
            tgt = _target(c)
            if idx is None:
                continue
            own = segs[idx]['body']
            hits = edit_fail_hits(tgt, own, segs[idx]['cmd'], own_output=True)
            if not hits and idx + 1 < len(segs):
                hits = edit_fail_hits(tgt, segs[idx + 1]['body'], segs[idx + 1]['cmd'])
            for fam, d in hits.items():
                ef.setdefault(fam, (c[:100], d))
            if not hits and idx + 1 >= len(segs):
                self.pending_edits = [(t, c, tgt)]
        self._edit_ev = ef
        body = '\n'.join(s['body'] for s in segs) + ('\n' + lead if lead else '')
        self._add_action(t, cmds, durs, body, r['done'], ts)

    def _absorb_rc(self, t, r, outs, cmds, ts):
        """mini-swe-agent (text or tool-calling): one output per command, with an exit code."""
        segs, bodies, rcs = [], [], []
        for k, o in enumerate(outs):
            rc, body = None, o or ''
            st = body.strip()
            if st.startswith('{'):
                try:
                    j = json.loads(st)
                    rc, body = j.get('returncode'), str(j.get('output', ''))
                except Exception:
                    pass
            if rc is None:
                m = RC_RE.search(body)
                if m:
                    rc = int(m.group(1))
                    mo = MINI_OUT_RE.findall(body)
                    body = '\n'.join(mo) if mo else body
            cmd = cmds[k] if k < len(cmds) else ''
            # grep/find/diff/test exit 1 means "no match / differs", not an error
            benign1 = rc == 1 and re.match(r'\s*(?:cd\s+\S+\s*&&\s*)?(?:grep|egrep|rg|find|diff|cmp|test|\[)\b', cmd or '')
            err = bool(ERR_RE.search(body)) or (rc not in (None, 0) and not benign1)
            seg = dict(cmd=(cmd or '').strip()[:200], body=body.strip(), turn=t, err=err)
            self.segs_all.append(seg)
            segs.append(seg); bodies.append(body); rcs.append(rc if not benign1 else 0)
        self._touched = list(range(len(self.segs_all) - len(segs), len(self.segs_all)))
        ef = {}
        for k, c in enumerate(cmds):
            if k < len(segs) and is_modify(c, self.writers):
                for fam, d in edit_fail_hits(_target(c), segs[k]['body'], segs[k]['cmd'], own_output=True).items():
                    ef.setdefault(fam, (c[:100], d))
        self._edit_ev = ef
        rc_any = next((x for x in rcs if x not in (None, 0)), 0)
        self._add_action(t, cmds, [None] * len(cmds), '\n'.join(bodies), r['done'], ts, rc=rc_any)

    def _add_action(self, t, cmds, durs, body, done, ts, rc=None):
        nk = norm_keys(cmds) if cmds else (('<MARK>',) if done else ('<WAIT>',))
        osig = norm_out(body) or None
        m = ERR_RE.search(body)
        err = bool(m) or (rc not in (None, 0))
        mb = ERR_BROAD_RE.search(body)
        wait = nk != ('<MARK>',) and all(WAIT_RE.match(x) for x in nk)
        dur = 0.0
        for c, d in zip(cmds or [''], durs or [None]):
            try:
                dur += float(d) if d is not None else 0.0
            except (TypeError, ValueError):
                pass
            ms = re.match(r'^\s*sleep\s+([\d.]+)', c or '')
            if ms:
                dur = max(dur, float(ms.group(1)))
        # no new output: nothing beyond prompts/echo, or identical to the previous action's output
        prev = self.actions[-1]['osig'] if self.actions else None
        empty = (not osig) or len(osig) < 3 or (osig == prev)
        keystr = ' ; '.join(nk)[:400]
        # keystroke similarity to the previous SIM_BACK actions, computed now (causal, and cheap to reuse)
        sims = []
        for a in reversed(self.actions[-SIM_BACK:]):
            if a['keystr'] == keystr:
                sims.append(1.0)
                continue
            sm = difflib.SequenceMatcher(None, a['keystr'], keystr)
            q = sm.quick_ratio()          # upper bound; below 0.5 it is under every threshold we use
            sims.append(q if q < 0.5 else sm.ratio())
        self.turn_act[t] = len(self.actions)
        self.actions.append(dict(i=len(self.actions), turn=t, keys=nk, keystr=keystr, osig=osig, err=err, sims=sims,
                                 err_sig=(m.group(0) if m else ('rc=%s' % rc if err else None)),
                                 err_broad=bool(mb) or err, wait=wait, empty=empty, dur=dur, ts=ts,
                                 mark=nk == ('<MARK>',)))

    # -- env evidence for turn t (called when the observation of turn t-1 arrived) --
    def _env_for(self, t):
        touched = list(getattr(self, '_touched', []) or []) if t > 1 else []
        # snapshot the error chains now: a later turn can append still-running output to an old segment
        chains = []
        for j in touched:
            if not self.segs_all[j]['err']:
                continue
            ch, p = [], j
            while p >= 0 and len(ch) < 8:
                sg = self.segs_all[p]
                if sg['body'].strip():
                    if not sg['err']:
                        break
                    m = ERR_RE.search(sg['body'])
                    ch.append((m.group(0) if m else 'rc', sg['turn']))
                p -= 1
            chains.append(ch)
        ev = dict(edit_failed=dict(getattr(self, '_edit_ev', {}) or {}) if t > 1 else {},
                  n_actions=len(self.actions), streak_chains=chains)
        self._edit_ev, self._touched = {}, []
        self.env[t] = ev

    def sim(self, i, j):
        """Keystroke similarity of actions i and j (|i-j| <= SIM_BACK), precomputed when j arrived."""
        if i == j:
            return 1.0
        lo, hi = min(i, j), max(i, j)
        d = hi - lo
        s = self.actions[hi]['sims']
        return s[d - 1] if d <= len(s) else 0.0

    def compact(self):
        """Drop bulky text once an episode is fully scanned (offline calibration keeps many in memory)."""
        for a in self.actions:
            a['osig'] = (hash(a['osig']), len(a['osig'])) if a['osig'] else None
            a['keystr'] = a['keystr'][:120]
        self.segs_all = []
        self.cmdlog = [(c[0], '', c[2], c[3]) for c in self.cmdlog]
        self.prev_screen = ''
        self.writers, self.created = set(), set()
        return self

    # -- router entry point --
    def current_fires(self, reply=None, config=None, elapsed_sec=None, budget_sec=None):
        """Fires for the request about to be answered (turn self.turn+1), optionally judging `reply`."""
        cfg = config or DEFAULT_CONFIG
        t = self.turn + 1
        fires = env_fires(self, t, cfg, elapsed_sec=elapsed_sec, budget_sec=budget_sec)
        if reply is not None:
            probe = _Probe(self)
            ev = probe.decision(reply)
            fires += dec_fires(ev, t, cfg)
        return [f for f in fires if t >= cfg.get('min_turn', 2) or f['trigger'] == 'done_claim']


class _Probe:
    """Judge a candidate reply without mutating the scanner."""

    def __init__(self, sc):
        self.sc = sc

    def decision(self, reply):
        sc = self.sc
        r = parse_reply(reply, sc.harness)
        saved = sc.done_turn
        ev = sc._decision_evidence(sc.turn + 1, r)
        sc.done_turn = saved
        return ev


# ==== trigger evaluation (config -> fires) ===================================================================
def _olen(o):
    return o[1] if isinstance(o, tuple) else len(o)


def _loop_fire(sc, t, cfg):
    """Loop at the request for turn t, judged on the executed actions of turns < t (last one = turn t-1)."""
    base = sc.turn_act.get(t - 1)
    if base is None or sc.actions[base]['mark']:
        return None
    W, k = cfg['loop_window'], cfg['loop_k']
    win = []
    i = base
    while i >= 0 and len(win) < W and base - i <= SIM_BACK:
        if not sc.actions[i]['mark']:
            win.append(i)
        i -= 1
    errkey = 'err_broad' if cfg.get('loop_err') == 'broad' else 'err'
    modes = cfg['loop_modes']
    last = sc.actions[base]
    if 'varied' in modes and last[errkey] and not last['wait']:
        sims = [i for i in win if sc.actions[i][errkey] and not sc.actions[i]['wait'] and
                sc.sim(i, base) >= cfg['loop_sim']]
        if len(sims) >= k:
            return 'varied retry: %d of last %d actions errored with keystrokes >= %.2f similar: %s' % (
                len(sims), len(win), cfg['loop_sim'], last['keystr'][:100])
    if 'exact' in modes and not last['wait']:
        c = sum(1 for i in win if sc.actions[i]['keys'] == last['keys'])
        if c >= k:
            return 'exact repeat x%d: %s' % (c, last['keystr'][:100])
    if 'same_output' in modes and last['osig'] and _olen(last['osig']) >= cfg.get('loop_min_out', 30):
        same = [i for i in win if sc.actions[i]['osig'] == last['osig']]
        distinct_cmds = len({sc.actions[i]['keys'] for i in same})
        if len(same) >= k and (not cfg.get('loop_same_output_distinct') or distinct_cmds >= 2):
            return 'same output x%d from %d distinct commands' % (len(same), distinct_cmds)
    return None


def _wait_fire(sc, t, cfg, elapsed_sec=None, budget_sec=None):
    base = sc.turn_act.get(t - 1)
    if base is None:
        return None
    streak = []
    i = base
    while i >= 0:
        a = sc.actions[i]
        if a['mark'] or not a['wait']:
            break
        streak.append(a)
        i -= 1
    if not streak:
        return None
    n_empty = 0
    for a in streak:
        if not a['empty']:
            break
        n_empty += 1
    k = cfg.get('wait_k')
    if k and n_empty >= k:
        return 'no-progress wait: %d waits in a row with no new output (%.0f s)' % (
            n_empty, sum(a['dur'] for a in streak[:n_empty]))
    frac = cfg.get('wait_budget_frac')
    if frac and budget_sec and elapsed_sec is not None:
        waited = sum(a['dur'] for a in streak)
        start = elapsed_sec - waited
        remaining = max(1.0, budget_sec - start)
        if waited >= frac * remaining and len(streak) >= cfg.get('wait_budget_min', 2):
            return 'waiting used %.0f s of %.0f s remaining at the streak start' % (waited, remaining)
    return None


def _streak_fire(sc, t, cfg):
    """error_streak (scan.py semantics at streak_n=2) plus narrowed variants: N consecutive errored commands
    (silent commands skipped), optionally the same error signature, optionally no file edit inside the streak."""
    env = sc.env.get(t)
    if not env or not env.get('streak_chains'):
        return None
    n = cfg.get('streak_n', 2)
    for ch in env['streak_chains']:
        if len(ch) < n:
            continue
        ch = ch[:n]
        sigs = [c[0] for c in ch]
        if cfg.get('streak_same_sig') and len(set(sigs)) > 1:
            continue
        if cfg.get('streak_no_edit_between'):
            lo = min(c[1] for c in ch)
            if any(c[2] and lo <= c[0] < t for c in sc.cmdlog):
                continue
        return 'error streak x%d: %s' % (n, ' / '.join(sigs[:3]))
    return None


def env_fires(sc, t, cfg, elapsed_sec=None, budget_sec=None):
    out = []
    env = sc.env.get(t)
    if env is None:
        return out
    en = cfg['enabled']
    if 'edit_failed' in en:
        fams = [f for f in env['edit_failed'] if f in cfg['edit_families']]
        k = cfg.get('edit_k', 1)
        if fams and k > 1:
            # narrowed: k failed edits within the last edit_window turns (this one included)
            n = sum(1 for u in range(max(1, t - cfg.get('edit_window', 10) + 1), t + 1)
                    if any(f in cfg['edit_families'] for f in sc.env.get(u, {}).get('edit_failed', {})))
            if n < k:
                fams = []
        if fams:
            cmd, detail = env['edit_failed'][fams[0]]
            out.append(dict(trigger='edit_failed', kind='environment', turn=t,
                            reason='%s after edit `%s`: %s' % (fams[0], cmd.strip()[:80], detail)))
    if 'loop' in en:
        r = _loop_fire(sc, t, cfg)
        if r:
            out.append(dict(trigger='loop', kind='environment', turn=t, reason=r))
    if 'no_progress_wait' in en:
        r = _wait_fire(sc, t, cfg, elapsed_sec, budget_sec)
        if r:
            out.append(dict(trigger='no_progress_wait', kind='environment', turn=t, reason=r))
    if 'error_streak' in en:
        r = _streak_fire(sc, t, cfg)
        if r:
            out.append(dict(trigger='error_streak', kind='environment', turn=t, reason=r))
    return out


def dec_fires(ev, t, cfg):
    out = []
    en = cfg['enabled']
    if ev['done'] and 'done_claim' in en:
        out.append(dict(trigger='done_claim', kind='decision', turn=t, reason='task_complete: true'))
    if ev['done'] and 'submit_check' in en:
        fl = {k: v for k, v in ev['submit'].items() if k in cfg['submit_flags']}
        if fl:
            k0 = sorted(fl)[0]
            out.append(dict(trigger='submit_check', kind='decision', turn=t, reason='%s: %s' % (k0, fl[k0])))
    if 'gave_up' in en:
        fams = [f for f in ev['gave_up'] if f in cfg['giveup_families']]
        if fams:
            out.append(dict(trigger='gave_up', kind='decision', turn=t,
                            reason='%s: "%s"' % (fams[0], ev['gave_up'][fams[0]][:200])))
    if 'success_claim_unchecked' in en and ev['success_unchecked']:
        s, c = ev['success_unchecked']
        out.append(dict(trigger='success_claim_unchecked', kind='decision', turn=t,
                        reason='claims "%s" with no check since `%s`' % (s[:120], c[:60])))
    if 'success_contradicted' in en and ev['success_contradicted']:
        s, sig = ev['success_contradicted']
        out.append(dict(trigger='success_contradicted', kind='decision', turn=t,
                        reason='claims "%s" but the last output shows %s' % (s[:120], sig)))
    if 'destructive' in en and ev['destructive']:
        out.append(dict(trigger='destructive', kind='decision', turn=t, reason='%s: %s' % ev['destructive'][0]))
    return out


def fires_by_turn(sc, cfg, budget_sec=None):
    """Offline: every fire over a fully scanned episode, as a list of dicts ordered by turn."""
    out = []
    for t in range(1, sc.turn + 1):
        el = sc.turn_ts.get(t) if budget_sec else None
        fs = env_fires(sc, t, cfg, elapsed_sec=el, budget_sec=budget_sec) + dec_fires(sc.dec[t], t, cfg)
        out += [f for f in fs if t >= cfg.get('min_turn', 2) or f['trigger'] == 'done_claim']
    return out


def scan_messages(messages, harness=None, timestamps=None):
    """Offline: scan a whole message list (OpenAI roles). timestamps: optional per-message elapsed seconds."""
    sc = EpisodeScanner(harness)
    for i, m in enumerate(messages):
        sc.add_message(m, timestamps[i] if timestamps else None)
    sc.flush()
    return sc


def detect(messages, reply=None, config=None, harness=None, elapsed_sec=None, budget_sec=None,
           include_logged=False):
    """Router entry point. messages: the request's message list (ends with the latest observation).
    reply: the student's candidate reply for this request (decision triggers), or None.
    Returns [{trigger, kind, turn, reason}] for the triggers enabled in config (takeover triggers). With
    include_logged=True it also returns the LOGGED signals, with kind='logged', for the relay metadata."""
    cfg = config or DEFAULT_CONFIG
    sc = EpisodeScanner(harness)
    for m in messages:
        sc.add_message(m)
    sc.flush()                   # close the last turn (its observation, or none if nothing ran yet)
    if not include_logged:
        return sc.current_fires(reply, cfg, elapsed_sec=elapsed_sec, budget_sec=budget_sec)
    extra = [t for t in LOGGED if t not in cfg['enabled']]
    fires = sc.current_fires(reply, dict(cfg, enabled=list(cfg['enabled']) + extra),
                             elapsed_sec=elapsed_sec, budget_sec=budget_sec)
    for f in fires:
        if f['trigger'] in extra:
            f['kind'] = 'logged'
    return fires


# ==== configs ===============================================================================================
# The design as of 2026-09-25 morning (forensics), expressed in this module's terms.
PROPOSAL_CONFIG = dict(
    enabled=['done_claim', 'edit_failed', 'loop', 'gave_up'],
    min_turn=2,
    edit_families=['sed_error', 'patch_fail', 'heredoc_eof', 'shell_syntax', 'replace_notfound', 'nofile_edit',
                   'pyerr_target', 'git_diff_empty'], edit_k=1, edit_window=10,
    loop_modes=['exact', 'same_output'], loop_window=4, loop_k=3, loop_sim=0.6, loop_err='strict',
    loop_min_out=30, loop_same_output_distinct=False,
    giveup_families=['stand_in', 'mock', 'simplify', 'workaround', 'skip', 'instead_of_real', 'impossible_req',
                     'cannot_install_pivot'],
    wait_k=3, wait_budget_frac=None,
    streak_n=2, streak_same_sig=False, streak_no_edit_between=False,
    submit_flags=['no_edit', 'stub_marker', 'stub_pass', 'tests_rm', 'tests_skip'],
)
# Calibrated 2026-09-25 on the train split (ai_memory/active/snowball-sft/research/2026-09-25_trigger_calibration/):
# each trigger's parameters maximise the student-failure hit rate with <= 5 % false fires on student passes
# (TB2, SWE-bench Verified, R2E-Gym, each group separately). `enabled` lists the takeover triggers; the rest are
# computed for logging only (LOGGED).
# After the held-out check and the Opus judges, only loop and the no-progress wait stay takeover triggers (next to
# done_claim); wait_k was tightened from the train optimum 2 to 3 because k=2 went over the cap on held-out.
CALIBRATED_CONFIG = dict(
    PROPOSAL_CONFIG,
    enabled=['done_claim', 'loop', 'no_progress_wait'],
    edit_families=['sed_error', 'patch_fail', 'heredoc_eof', 'shell_syntax', 'replace_notfound', 'nofile_edit',
                   'pyerr_target', 'git_diff_empty'], edit_k=4, edit_window=5,
    giveup_families=['workaround', 'impossible_req', 'hardcode', 'cannot_install_pivot'],
    loop_modes=['exact'], loop_window=6, loop_k=3,
    wait_k=3, wait_budget_frac=None,
    streak_n=5, streak_same_sig=True, streak_no_edit_between=True,
    submit_flags=['stub_pass', 'tests_skip'],
)
LOGGED = ['gave_up', 'edit_failed', 'error_streak', 'submit_check', 'success_claim_unchecked',
          'success_contradicted', 'destructive']
DEFAULT_CONFIG = CALIBRATED_CONFIG
