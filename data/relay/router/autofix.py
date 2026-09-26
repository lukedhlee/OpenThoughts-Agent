#!/usr/bin/env python3
"""Format autofix: turn a student reply that Terminus-2's strict parser rejects into valid Terminus-2 JSON, when the
student's intended action is recoverable without guessing. Used by relay_router.py before a parse_error repair
(Luke 2026-09-25 17:40 PT): the student's own action runs; the teacher is called only when nothing is recoverable.

    fix = autofix(content, finish_reason, parser)   # -> Fix(content, kind, commands) or Unfixable(reason)

What is kept:
  * the student's thinking, verbatim: everything up to and including its last <|end_think|> (09-21 sometimes closes
    a span it never opened; that text is still its thinking);
  * analysis = the visible prose before the action (or the object's own analysis); plan = the object's own plan or "";
    task_complete when the student wrote it.
What is recovered (the action), in this order:
  1. <tool_call> blocks: {"keystrokes": k, "duration": d}; {"command": c} / {"name": .., "command": c};
     {"name": .., "arguments"|"args": {"command"|"cmd": c}} (keystrokes c + "\\n"; arguments may be a JSON string;
     "keystrokes" and a "commands" list also accepted); a Terminus-2 object with "commands"; {"name": "finish"} ->
     task_complete true, no commands. Several blocks or objects -> the commands list, in order. Empty blocks skipped.
  2. otherwise JSON objects in the visible text, parsed leniently (json strict=False: raw newlines in strings):
     a Terminus-2 object (any of analysis / plan / commands / task_complete), bare command objects, or tool-call
     shaped objects; several bare command objects -> the commands list.
  3. otherwise exactly one ```bash / ```sh block -> keystrokes = its text + "\\n".
Unfixable (the teacher repairs): the reply ended inside its thinking; it was truncated (finish_reason length); no
action at all; an action whose JSON does not parse even leniently; two or more bash blocks with no JSON action; the
kept thinking itself holds a brace pair that Terminus-2's parser would read first (thinking_holds_json).
Every rewrite is checked with the same strict parser, and its commands must equal the recovered ones.
"""
import json
import re
from dataclasses import dataclass, field

START, END = '<|start_think|>', '<|end_think|>'
TOOL_RE = re.compile(r'<tool_call>(.*?)(?=</tool_call>|<tool_call>|\Z)', re.S)
BASH_RE = re.compile(r'```(?:bash|sh|shell)[ \t]*\n(.*?)```', re.S)


@dataclass
class Fix:
    content: str
    kind: str
    commands: list = field(default_factory=list)


@dataclass
class Unfixable:
    reason: str


def split_think(content):
    """(thinking part incl. markers, visible body) or None when the reply ended inside its thinking."""
    last_start = content.rfind(START)
    last_end = content.rfind(END)
    if last_start != -1 and last_start > last_end:
        return None
    if last_end == -1:
        return '', content
    i = last_end + len(END)
    return content[:i], content[i:]


def json_objects(text):
    """Top-level JSON objects in text, in order, each parsed leniently; (start, end, obj or None)."""
    out, i = [], text.find('{')
    dec = json.JSONDecoder(strict=False)
    while i != -1:
        try:
            obj, end = dec.raw_decode(text, i)
            out.append((i, end, obj))
            i = text.find('{', end)
        except json.JSONDecodeError:
            out.append((i, None, None))
            i = text.find('{', i + 1)
    return out


def _args(a):
    if isinstance(a, str):
        try:
            a = json.loads(a, strict=False)
        except json.JSONDecodeError:
            return {'command': a}
    return a if isinstance(a, dict) else {}


def obj_commands(o):
    """(commands [(keystrokes, duration)], meta {analysis, plan, task_complete}, shape) or None."""
    if not isinstance(o, dict):
        return None
    meta = {k: o[k] for k in ('analysis', 'plan', 'task_complete') if k in o}
    if isinstance(o.get('commands'), list):
        cmds = []
        for c in o['commands']:
            if isinstance(c, dict) and isinstance(c.get('keystrokes'), str):
                cmds.append((c['keystrokes'], c.get('duration')))
            elif isinstance(c, str):
                cmds.append((c if c.endswith('\n') else c + '\n', None))
            else:
                return None
        return cmds, meta, 'terminus_object'
    if isinstance(o.get('keystrokes'), str):
        return [(o['keystrokes'], o.get('duration'))], meta, 'keystrokes_object'
    if isinstance(o.get('command'), str) and o['command'].strip():      # {"command": c} or {"name": "bash", "command": c}
        return [(o['command'].rstrip('\n') + '\n', o.get('duration'))], meta, 'command_object'
    if str(o.get('name', '')).lower() in ('finish', 'submit', 'task_complete', 'mark_task_complete'):
        return [], dict(meta, task_complete=True), 'tool_call_finish'
    if 'name' in o and ('arguments' in o or 'args' in o):
        a = _args(o.get('arguments', o.get('args')))
        if isinstance(a.get('keystrokes'), str):
            return [(a['keystrokes'], a.get('duration'))], meta, 'tool_call_keystrokes'
        c = a.get('command', a.get('cmd'))
        if isinstance(c, str) and c.strip():
            return [(c.rstrip('\n') + '\n', a.get('duration'))], meta, 'tool_call_command'
        cs = a.get('commands')
        if isinstance(cs, list) and cs:
            out = []
            for x in cs:
                if isinstance(x, str) and x.strip():
                    out.append((x.rstrip('\n') + '\n', None))
                elif isinstance(x, dict) and isinstance(x.get('keystrokes'), str):
                    out.append((x['keystrokes'], x.get('duration')))
                elif isinstance(x, dict) and isinstance(x.get('command'), str):
                    out.append((x['command'].rstrip('\n') + '\n', x.get('duration')))
                else:
                    return None
            return out, meta, 'tool_call_command'
        return None
    if meta:   # a Terminus-2 object without commands (e.g. only task_complete)
        return [], meta, 'terminus_object'
    return None


def recover(body):
    """(commands, meta, kind, prose_before) or an Unfixable reason string."""
    blocks = [b for b in TOOL_RE.findall(body) if b.strip()]
    first = body.find('<tool_call>')
    if first != -1:
        cmds, meta, shapes = [], {}, []
        for b in blocks:
            objs = json_objects(b)
            if not objs or objs[0][2] is None:
                return 'tool_call_json_unparseable'
            for _, _, o in objs:          # several objects in one block: each must be a command
                if o is None:
                    return 'tool_call_json_unparseable'
                r = obj_commands(o)
                if r is None:
                    return 'tool_call_unrecognised'
                cmds += r[0]
                meta = {**r[1], **meta}
                shapes.append(r[2])
        if not blocks:
            return 'tool_call_empty'
        kind = ('tool_calls_to_list' if len(shapes) > 1 else 'tool_call_' + shapes[0].replace('tool_call_', ''))
        return cmds, meta, kind, body[:first]
    objs = json_objects(body)
    parsed = [(s, o) for s, e, o in objs if o is not None and obj_commands(o) is not None]
    if parsed:
        cmds, meta, shapes = [], {}, []
        for s, o in parsed:
            r = obj_commands(o)
            cmds += r[0]
            meta = {**r[1], **meta}
            shapes.append(r[2])
        if len(parsed) > 1 and 'terminus_object' in shapes:
            return 'several_terminus_objects'
        kind = 'json_objects_to_list' if len(parsed) > 1 else 'json_' + shapes[0]
        return cmds, meta, kind, body[:parsed[0][0]]
    if any(o is None for _, _, o in objs) and ('"keystrokes"' in body or '"commands"' in body or '"analysis"' in body):
        return 'json_unparseable'
    bash = BASH_RE.findall(body)
    if len(bash) == 1 and bash[0].strip():
        k = bash[0] if bash[0].endswith('\n') else bash[0] + '\n'
        return [(k, None)], {}, 'bash_block', body[:body.find('```')]
    if len(bash) > 1:
        return 'several_bash_blocks'
    return 'no_action'


def autofix(content, finish_reason, parser):
    if finish_reason == 'length':
        return Unfixable('truncated')
    sp = split_think(content or '')
    if sp is None:
        return Unfixable('ended_inside_thinking')
    think, body = sp
    r = recover(body)
    if isinstance(r, str):
        return Unfixable(r)
    cmds, meta, kind, prose = r
    obj = {'analysis': str(meta['analysis']) if 'analysis' in meta else prose.strip(),
           'plan': str(meta.get('plan', '')),
           'commands': [dict(keystrokes=k, **({'duration': d} if isinstance(d, (int, float)) else {})) for k, d in cmds]}
    if 'task_complete' in meta:
        obj['task_complete'] = meta['task_complete']
    new = think + json.dumps(obj, ensure_ascii=False)
    res = parser.parse_response(new)
    if res.error:
        # Terminus-2's parser takes the first balanced {...} anywhere in the reply; a brace pair inside the kept
        # thinking comes first, and the thinking is not ours to edit
        if not parser.parse_response(json.dumps(obj, ensure_ascii=False)).error:
            return Unfixable('thinking_holds_json')
        return Unfixable('rewrite_rejected: ' + res.error[:80])
    if [c.keystrokes for c in res.commands] != [k for k, _ in cmds]:
        return Unfixable('rewrite_commands_mismatch')
    return Fix(new, kind, [k for k, _ in cmds])


def main():
    """Replay: python autofix.py <parser.py> <jsonl with content, finish[, reason]> -> share fixable by kind."""
    import collections
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location('p', sys.argv[1])
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    parser = m.TerminusJSONPlainParser()
    c, n = collections.Counter(), 0
    for line in open(sys.argv[2]):
        x = json.loads(line)
        f = autofix(x['content'], x.get('finish'), parser)
        n += 1
        c[('fix', f.kind) if isinstance(f, Fix) else ('teacher', f.reason.split(':')[0])] += 1
    fixed = sum(v for k, v in c.items() if k[0] == 'fix')
    print(json.dumps(dict(replies=n, autofixed=fixed, autofix_share=round(fixed / n, 4) if n else None,
                          by_kind={f'{a}:{b}': v for (a, b), v in c.most_common()}), indent=1))


if __name__ == '__main__':
    main()
