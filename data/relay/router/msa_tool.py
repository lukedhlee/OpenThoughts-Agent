"""mini-swe-agent tool mode (harbor mini-swe-agent-host, model_class litellm) for the relay router.

The harness sends the full structured history every turn (assistant turns with `tool_calls`, `tool` results named
"bash"), the `bash` tool schema and tool_choice "none" (no server tool parser), and reads the reply's tool calls from
`message.tool_calls`, or, when there are none, parses `<tool_call>{json}</tool_call>` blocks from the reply text itself.
The router answers every agent turn already parsed: `content` (the prose), `reasoning_content` (the thinking) and
`tool_calls` with ids the router chooses, so it knows who wrote each turn of the history it gets back. For a student
reply the split and parse are harbor's own (copied verbatim below), so the history the student sees later is what the
plain harness would have built from the same reply.

    parsed = student_reply(content, call_ids)      # harbor's client-side parse of a 09-21/H9 reply
    parsed = teacher_reply(content, reasoning, call_ids)   # Qwen3.8's XML (or JSON) tool calls, parsed here
    err = toolcall_error(parsed.tool_calls)         # upstream's FormatError rule (actions_toolcall.py), '' = accepted
"""
import json
import re
from dataclasses import dataclass, field

SUBMIT_SENTINEL = 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'
# upstream format_error_template's two openings (mini.yaml 2.4.6): a parse error, and a reply cut at the output limit
FORMAT_ERROR_MARKS = ('Tool call error:', 'Your previous response reached the output token limit')

# ---- verbatim from harbor src/harbor/llms/lite_llm.py and agents/mini_swe_agent_host/adapters.py (lukedhlee/mini-swe-relay)
_COMPLETION_REASONING_MARKERS = (
    ("<think>", "</think>"),
    ("<|start_think|>", "<|end_think|>"),
)
INLINE_TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
TOOL_CALL_OPEN = "<tool_call>"


def strip_think_spans(text):
    had_leading_span = False
    while True:
        stripped = text.lstrip()
        for open_marker, close_marker in _COMPLETION_REASONING_MARKERS:
            if stripped.startswith(open_marker):
                _, closed, rest = stripped[len(open_marker):].partition(close_marker)
                if not closed:
                    return ""
                text = rest
                had_leading_span = True
                break
        else:
            break
    if had_leading_span:
        return text
    for _, close_marker in _COMPLETION_REASONING_MARKERS:
        _, closed, rest = text.partition(close_marker)
        if closed:
            return rest
    return text


def split_inline_reasoning(text):
    answer = strip_think_spans(text)
    if answer == text:
        return None, text
    reasoning = text[: len(text) - len(answer)] if answer else text
    for open_marker, close_marker in _COMPLETION_REASONING_MARKERS:
        reasoning = reasoning.replace(open_marker, "").replace(close_marker, "")
    return reasoning, answer


@dataclass
class Parsed:
    reasoning: str = None
    content: str = ''
    tool_calls: list = field(default_factory=list)
    error: str = ''          # harbor's parse error ('' = parsed)
    shape: str = ''          # how the calls were found (teacher: server / xml / json / none)


def parse_inline_tool_calls(text, call_ids):
    """harbor's parse_inline_tool_calls, with the ids taken from `call_ids` (a function j -> id) instead of
    call_<call_index>_<j>."""
    reasoning, answer = split_inline_reasoning(text)
    tool_calls, errors = [], []
    for j, body in enumerate(INLINE_TOOL_CALL.findall(answer)):
        try:
            call = json.loads(body)
        except json.JSONDecodeError as exc:
            errors.append(f"Tool call {j + 1} is not valid JSON: {exc}.")
            continue
        if not isinstance(call, dict) or "name" not in call:
            errors.append(f'Tool call {j + 1} has no "name".')
            continue
        arguments = call.get("arguments", {})
        tool_calls.append({"id": call_ids(j), "type": "function",
                           "function": {"name": call["name"],
                                        "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments)}})
    content = INLINE_TOOL_CALL.sub("", answer)
    if TOOL_CALL_OPEN in content:
        errors.append("A tool call was opened but never closed.")
    return Parsed(reasoning=reasoning, content=content.strip(), tool_calls=tool_calls, error=" ".join(errors),
                  shape='inline')
# ---- end of the verbatim copies


def toolcall_error(tool_calls):
    """upstream parse_toolcall_actions' rule (minisweagent 2.4.6 models/utils/actions_toolcall.py): '' when every call
    is a `bash` call whose JSON arguments hold a `command`, else the error text upstream would put in its FormatError."""
    if not tool_calls:
        return 'No tool calls found in the response. Every response MUST include at least one tool call.'
    for tc in tool_calls:
        fn = tc.get('function') or {}
        msg, args = '', {}
        try:
            args = json.loads(fn.get('arguments'))
        except Exception as e:  # noqa: BLE001 - upstream catches everything here too
            msg = f'Error parsing tool call arguments: {e}.'
        if fn.get('name') != 'bash':
            msg += f"Unknown tool '{fn.get('name')}'."
        if not isinstance(args, dict) or 'command' not in args:
            msg += "Missing 'command' argument in bash tool call."
        if msg:
            return msg.strip()
    return ''


def parse(content, reasoning, call_ids):
    """A reply as harbor reads it: the reasoning from the server's reasoning field when it has one (the reply text is
    then the answer), else split off inline; tool calls parsed from the answer."""
    if reasoning:
        p = parse_inline_tool_calls(content or '', call_ids)
        p.reasoning = reasoning
        return p
    return parse_inline_tool_calls(content or '', call_ids)


def commands(tool_calls):
    """The bash commands of a parsed reply's tool calls ('' for a call without one)."""
    out = []
    for tc in tool_calls or []:
        try:
            a = json.loads((tc.get('function') or {}).get('arguments') or '{}')
        except Exception:  # noqa: BLE001
            a = {}
        out.append(str(a.get('command', '')) if isinstance(a, dict) else '')
    return out


def is_submit(tool_calls):
    """The done claim: a tool call whose command contains the submit sentinel (upstream submits when the command's
    first output line is the sentinel and it exits 0; the router judges the command, before it runs)."""
    return any(SUBMIT_SENTINEL in c for c in commands(tool_calls))


# ---- Qwen3.8's tool calls (served without a tool parser: tool_choice "none") ----------------------------------------
QWEN_FN_RE = re.compile(r'<function=([^>\s]+)>(.*?)(?:</function>|\Z)', re.S)
QWEN_PARAM_RE = re.compile(r'<parameter=([^>\s]+)>(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|\Z)', re.S)
TOOL_BLOCK_RE = re.compile(r'<tool_call>(.*?)(?:</tool_call>|(?=<tool_call>)|\Z)', re.S)


def _param_value(v):
    # the qwen3_coder parser's convention: one newline after the opening tag and one before the closing tag belong to
    # the markup, not to the value
    if v.startswith('\n'):
        v = v[1:]
    if v.endswith('\n'):
        v = v[:-1]
    return v


def qwen_calls(answer):
    """Qwen3.8's `<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>` blocks (or a JSON
    `{"name", "arguments"}` body, the hermes form) -> ([(name, arguments dict)], prose, error)."""
    blocks = TOOL_BLOCK_RE.findall(answer)
    prose = TOOL_BLOCK_RE.sub('', answer)
    prose = re.sub(r'</tool_call>', '', prose).strip()
    calls, errors = [], []
    for j, b in enumerate(blocks):
        m = QWEN_FN_RE.search(b)
        if m:
            args = {k: _param_value(v) for k, v in QWEN_PARAM_RE.findall(m.group(2))}
            calls.append((m.group(1).strip(), args))
            continue
        try:
            o = json.loads(b.strip())
        except json.JSONDecodeError as e:
            errors.append(f'Tool call {j + 1} has neither <function=...> nor valid JSON: {e}.')
            continue
        if not isinstance(o, dict) or 'name' not in o:
            errors.append(f'Tool call {j + 1} has no "name".')
            continue
        a = o.get('arguments', {})
        if isinstance(a, str):
            try:
                a = json.loads(a)
            except json.JSONDecodeError:
                a = {'command': a}
        calls.append((str(o['name']), a if isinstance(a, dict) else {}))
    return calls, prose, ' '.join(errors)


def teacher_reply(message, call_ids):
    """Qwen3.8's chat-completions message (reasoning split off by the qwen3 parser) -> Parsed. Tool calls the server
    parsed itself are kept as they are (ids replaced); else Qwen's XML / JSON tool calls are parsed out of the content."""
    reasoning = message.get('reasoning_content') or message.get('reasoning')
    content = message.get('content') or ''
    if message.get('tool_calls'):
        tcs = [{'id': call_ids(j), 'type': 'function',
                'function': {'name': (tc.get('function') or {}).get('name'),
                             'arguments': (tc.get('function') or {}).get('arguments') or '{}'}}
               for j, tc in enumerate(message['tool_calls'])]
        return Parsed(reasoning=reasoning, content=content.strip(), tool_calls=tcs, shape='server')
    if not reasoning:
        # no reasoning parser split (or an unterminated think span): harbor's own split
        reasoning, content = split_inline_reasoning(content)
    calls, prose, err = qwen_calls(content)
    tcs = [{'id': call_ids(j), 'type': 'function',
            'function': {'name': n, 'arguments': json.dumps(a, ensure_ascii=False)}} for j, (n, a) in enumerate(calls)]
    return Parsed(reasoning=reasoning, content=prose, tool_calls=tcs, error=err,
                  shape='xml' if '<function=' in content else ('json' if calls else 'none'))


def student_autofix(answer, call_ids, af):
    """A student answer whose tool call harbor cannot parse, recovered with autofix.py's recover() when its action is
    unambiguous (invalid JSON escapes, raw control characters, a missing closing tag, Qwen-style XML, one bash fence):
    -> (tool_calls, kind, prose before the action) or (None, reason, '')."""
    r = af.recover(answer)
    if isinstance(r, str):
        return None, r, ''
    cmds, meta, kind, prose = r
    cmds = [k.rstrip('\n') for k, _ in cmds if k and k.strip()]
    if not cmds:
        return None, 'no_command', ''
    return [{'id': call_ids(j), 'type': 'function',
             'function': {'name': 'bash', 'arguments': json.dumps({'command': c}, ensure_ascii=False)}}
            for j, c in enumerate(cmds)], kind, prose.strip()


def message(p):
    """The assistant message the router returns to harbor for a parsed reply."""
    m = {'role': 'assistant', 'content': p.content}
    if p.reasoning:
        m['reasoning_content'] = p.reasoning
        m['reasoning'] = p.reasoning
    if p.tool_calls:
        m['tool_calls'] = p.tool_calls
    return m


def call_id_key(m):
    """The key of an assistant turn in the router's records: its first tool call id (the router chose it), else None."""
    tcs = m.get('tool_calls') or []
    return tcs[0].get('id') if tcs and isinstance(tcs[0], dict) else None
