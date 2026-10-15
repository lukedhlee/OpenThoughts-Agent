#!/usr/bin/env python3
"""A teacher-only MSA2 episode (mini-swe-agent 2.4.6 tool mode) -> the student's token sequence for teacher-forced
scoring, with every target token's class and the per-turn checks that make one forward pass equal per-turn scoring.

History = what the teacher conditioned on when it wrote each turn: mini-swe-agent's system and task messages, every
earlier assistant turn WITH its reasoning (the collection ran `interleaved_thinking: true`, and Qwen3.8's template keeps
earlier thinking: `preserve_thinking` defaults to true), its tool calls, the tool results harbor sent back and any
format-error user messages. It is rendered with the student's own chat template (09-21 and RLVR1 share
tokenizer.json and chat_template.jinja byte for byte), the bash schema in the tools block and the tool calls in the
student's `<tool_call>{json}</tool_call>` form, exactly as sft/render_msa.py renders SFT rows (its view / as_rendered /
load_template), except that no reasoning is cut.

No message renders differently depending on what follows it, so one causal forward pass over the episode scores every
turn: the tokens before turn k's body are, token for token, the generation prompt of messages[:k] (check_prefixes).
Targets are the assistant bodies, <|start_think|>reasoning<|end_think|> + prose + tool calls + <|eot_id|>; the
assistant header and every system / user / tool token are context only.

Target classes, by the characters a token covers (a token spanning two classes takes the majority and is flagged mixed):
  reasoning  the thought                      think_tag  <|start_think|> / <|end_think|>
  prose      content before the tool calls    call_fmt   tool-call scaffolding: <tool_call>, {"name": "bash", ...}
  command    a tool-call argument's value     eot        <|eot_id|>
  (inside its JSON quotes, JSON-escaped)
reasoning / prose / command are content; think_tag / call_fmt / eot are format.

    python render_full.py --manifest sample.json --tokenizer-dir <student dir> --out-dir <dir>
writes <out>/<traj_id>.json per sampled episode and <out>/render_report.json.
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'sft'))
from render_msa import BASH_TOOL, END, EOT, START, as_rendered, load_template, tojson, view  # noqa: E402

CLASSES = ('reasoning', 'prose', 'command', 'think_tag', 'call_fmt', 'eot')
CONTENT = ('reasoning', 'prose', 'command')


def load_student(tokdir):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(tokdir, 'tokenizer.json'))
    tpl = load_template(os.path.join(tokdir, 'chat_template.jinja'))
    bos = json.load(open(os.path.join(tokdir, 'tokenizer_config.json'))).get('bos_token', '<|begin_of_text|>')
    return tok, tpl, (bos.get('content') if isinstance(bos, dict) else bos)


def body_pieces(m):
    """[(text, class)] of an assistant message's body, piece by piece as chat_template.jinja renders it."""
    out = []
    if m.get('reasoning_content'):
        out += [(START, 'think_tag'), (m['reasoning_content'], 'reasoning'), (END, 'think_tag')]
    content = (m.get('content') or '').strip()          # the template's `text(content) | trim`
    if content:
        out.append((content, 'prose'))
    for c in m.get('tool_calls') or []:
        f = c['function']
        out.append(('<tool_call>\n{"name": ' + tojson(f['name']) + ', "arguments": ', 'call_fmt'))
        a = f['arguments']
        if isinstance(a, str):                           # arguments that are not JSON render verbatim
            out.append((a, 'command'))
        else:
            out.append(('{', 'call_fmt'))
            for j, (k, v) in enumerate(a.items()):
                out.append(((', ' if j else '') + tojson(k) + ': ', 'call_fmt'))
                s = tojson(v)
                if isinstance(v, str):
                    out += [('"', 'call_fmt'), (s[1:-1], 'command'), ('"', 'call_fmt')]
                else:
                    out.append((s, 'command'))
            out.append(('}', 'call_fmt'))
        out.append(('}\n</tool_call>', 'call_fmt'))
    out.append((EOT, 'eot'))
    return out


def render(msgs, tok, tpl, bos):
    """The episode as one token sequence: text, ids, per-token display pieces, message and turn spans, target classes."""
    full = [as_rendered(view(m)) for m in msgs if m.get('role') != 'exit']
    while full and full[-1]['role'] != 'assistant':     # observations after the last reply are never conditioned on
        full.pop()

    def tpl_render(ms, gen):
        return tpl.render(messages=ms, tools=[BASH_TOOL], bos_token=bos, add_generation_prompt=gen)

    text = tpl_render(full, False)
    cls = [None] * len(text)
    pos = len(tpl_render([], False))                     # bos + the template's own system block (Reasoning, tools)
    turns, messages = [], [dict(i=-1, role='template', start=0, end=pos, tool_call_id=None, call_ids=[])]
    for i, m in enumerate(full):
        if m['role'] == 'assistant':
            prefix = tpl_render(full[:i], True)          # exactly messages[:i] + the generation prompt
            assert text.startswith(prefix), f'message {i}: history is not a prefix of the episode'
            start = len(prefix)
            pieces = body_pieces(m)
            body = ''.join(p for p, _ in pieces)
            assert text[start:start + len(body)] == body, f'message {i}: body pieces != template render'
            c = start
            for p, k in pieces:
                cls[c:c + len(p)] = [k] * len(p)
                c += len(p)
            turns.append(dict(msg=i, start=start, end=c, prefix_chars=len(prefix),
                              n_calls=len(m.get('tool_calls') or []),
                              reasoning_chars=len(m.get('reasoning_content') or ''),
                              has_reasoning=bool(m.get('reasoning_content'))))
            end = c
        else:
            end = len(tpl_render(full[:i + 1], False))
        messages.append(dict(i=i, role=m['role'], start=pos, end=end, tool_call_id=m.get('tool_call_id'),
                             call_ids=[c['id'] for c in m.get('tool_calls') or []]))
        pos = end
    assert pos == len(text)
    enc = tok.encode(text, add_special_tokens=False)
    ids, offs = enc.ids, enc.offsets
    # Display pieces: each token's characters, a byte-split character given to its first token only, so the pieces
    # concatenate to the text exactly.
    pieces, prev = [], 0
    for s, e in offs:
        s = max(s, prev)
        pieces.append(text[s:e] if e > s else '')
        prev = max(prev, e)
    assert ''.join(pieces) == text, 'token pieces do not concatenate to the text'
    tok_start = [None] * len(text)
    for j, (s, e) in enumerate(offs):
        if s < len(text) and tok_start[s] is None:
            tok_start[s] = j
    for t in turns:
        # a body starts and ends on token boundaries (its first token is the first after the header's newline)
        t['tok_start'] = tok_start[t['start']]
        assert t['tok_start'] is not None and offs[t['tok_start']][0] == t['start'], 'turn does not start on a token'
        j = t['tok_start']
        while j < len(ids) and offs[j][1] <= t['end']:
            j += 1
        t['tok_end'] = j
        assert offs[j - 1][1] == t['end'], 'turn does not end on a token'
        assert ids[j - 1] == tok.token_to_id(EOT)
    for m in messages:
        m['tok_start'] = tok_start[m['start']]
        assert m['tok_start'] is not None, f"message {m['i']} does not start on a token"
    for a, b in zip(messages, messages[1:]):
        a['tok_end'] = b['tok_start']
    messages[-1]['tok_end'] = len(ids)
    # Per target token: majority class over its characters, mixed when it covers two.
    target_cls, mixed = {}, {}
    for t in turns:
        for j in range(t['tok_start'], t['tok_end']):
            s, e = offs[j]
            ks = [cls[c] for c in range(s, max(e, s + 1))]
            assert None not in ks, f'token {j} leaves the body'
            best = max(set(ks), key=ks.count)
            target_cls[j] = best
            mixed[j] = len(set(ks)) > 1
    return dict(text=text, ids=ids, offsets=offs, pieces=pieces, messages=messages, turns=turns, full=full,
                target_cls=target_cls, mixed=mixed, tpl_render=tpl_render)


def check_prefixes(r, tok):
    """Per turn: the generation prompt of messages[:k], tokenized on its own, is the episode's token prefix before the
    turn's body. Scoring the whole episode in one pass is then the same as scoring each turn with its own history."""
    for t in r['turns']:
        prefix = r['tpl_render'](r['full'][:t['msg']], True)
        p_ids = tok.encode(prefix, add_special_tokens=False).ids
        assert p_ids == r['ids'][:t['tok_start']], f"turn at message {t['msg']}: prefix tokens differ"
    return len(r['turns'])


def check_targets(r, tok):
    """Target tokens decode to the assistant bodies and nothing else: no system / user / tool token is a target."""
    roles = {}
    for m in r['messages']:
        for j in range(m['tok_start'], m['tok_end']):
            roles[j] = m['role']
    n = 0
    for t in r['turns']:
        body = r['text'][t['start']:t['end']]
        assert ''.join(r['pieces'][t['tok_start']:t['tok_end']]) == body
        assert tok.decode(r['ids'][t['tok_start']:t['tok_end']], skip_special_tokens=False) == body
        n += t['tok_end'] - t['tok_start']
    assert all(roles[j] == 'assistant' for j in r['target_cls'])
    assert n == len(r['target_cls'])
    return n


def episode_record(r, tok, meta):
    """The JSON the scorer and the site read (checkpoint independent: scores are keyed by token position)."""
    vocab_piece = [tok.id_to_token(i) for i in r['ids']]
    turns = []
    for k, t in enumerate(r['turns']):
        cl = [r['target_cls'][j] for j in range(t['tok_start'], t['tok_end'])]
        turns.append(dict(k=k, msg=t['msg'], start=t['start'], end=t['end'], tok_start=t['tok_start'],
                          tok_end=t['tok_end'], n_targets=t['tok_end'] - t['tok_start'], n_calls=t['n_calls'],
                          has_reasoning=t['has_reasoning'], history_tokens=t['tok_start'],
                          cls=cl, mixed=[int(r['mixed'][j]) for j in range(t['tok_start'], t['tok_end'])],
                          n_by_class={c: cl.count(c) for c in CLASSES}))
    return dict(meta, n_tokens=len(r['ids']), n_targets=len(r['target_cls']), n_turns=len(turns),
                text_sha256=hashlib.sha256(r['text'].encode()).hexdigest(),
                ids=r['ids'], offsets=r['offsets'], pieces=r['pieces'], raw_pieces=vocab_piece,
                messages=[{k: v for k, v in m.items()} for m in r['messages']], turns=turns, text=r['text'])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--tokenizer-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--max-tokens', type=int, default=131072, help='serving context; longer episodes fail the render')
    a = ap.parse_args()
    tok, tpl, bos = load_student(a.tokenizer_dir)
    man = json.load(open(a.manifest))
    os.makedirs(a.out_dir, exist_ok=True)
    report = dict(tokenizer_dir=a.tokenizer_dir,
                  tokenizer_sha256=hashlib.sha256(open(os.path.join(a.tokenizer_dir, 'tokenizer.json'), 'rb').read()).hexdigest(),
                  template_sha256=hashlib.sha256(open(os.path.join(a.tokenizer_dir, 'chat_template.jinja'), 'rb').read()).hexdigest(),
                  episodes=[])
    for ep in man['episodes']:
        msgs = json.load(open(ep['msa_trajectory']))['messages']
        r = render(msgs, tok, tpl, bos)
        assert len(r['ids']) + 1 <= a.max_tokens, f"{ep['traj_id']}: {len(r['ids'])} tokens over {a.max_tokens}"
        n_turns = check_prefixes(r, tok)
        n_targets = check_targets(r, tok)
        rec = episode_record(r, tok, ep)
        json.dump(rec, open(os.path.join(a.out_dir, ep['traj_id'] + '.json'), 'w'))
        report['episodes'].append(dict(traj_id=ep['traj_id'], n_tokens=rec['n_tokens'], n_targets=n_targets,
                                       n_turns=n_turns, prefix_checks=n_turns, text_sha256=rec['text_sha256']))
        print(json.dumps(report['episodes'][-1]), flush=True)
    json.dump(report, open(os.path.join(a.out_dir, 'render_report.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
