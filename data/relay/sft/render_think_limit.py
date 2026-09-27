#!/usr/bin/env python3
"""Re-render a final manifest with a different thinking-loss limit, using an unchanged render.py.

    python render_think_limit.py --render-dir <worktree@d0ddd237>/data/relay/sft --run-dir <run> --arm relay_repair \
        --manifest final_manifest.jsonl --tokenizer-dir <09-21 model dir> --think-limit 16384 \
        --out final_rendered_think16k.jsonl

render.py is imported from --render-dir, so the rendering code (template, capped history, mask rules, 65,536 row
limit) is exactly that checkout's; only render_episode's think_limit argument changes (default 8,192, 09-21's old
per-turn output limit; 16,384 under the 65k/16k TB policy, coordinator 2026-09-27). The limit only decides whether an
uncut Qwen reasoning span is trained; it never changes the text, so row lengths are unchanged.

Prints per-run stats: rows, rows over 65,536, trained tokens, Qwen turns whose uncut reasoning is over 8,192 and over
16,384 tokens (masked at the old / new limit), and the max reasoning tokens in one Qwen turn.

--strip-copied-markers (coordinator 2026-09-27, both arms alike): in a relay episode Qwen sees 09-21's turns and copies
its format quirks into its own reply (`<tool_call>` or `<|end_think|>` before the JSON, `</tool_call>` or
`</assistant>` after it). With the option, every teacher turn's content loses the COPIED_MARKERS below, but only in the
text before and after the JSON object that Terminus-2's parser executes (its own _extract_json_content bounds), never
inside it; a `<|start_header_id|>role<|end_header_id|>` header goes as a whole. Prose around the object stays. The
teacher's reasoning, student turns (masked context), observations and turns without a marker are untouched. The strip
runs after the router-record join (by content hash), inside render.episode_turns, so render.py itself is unchanged.
Checks per stripped turn (needs --terminus-parser): the stripped reply parses with no error and gives the same commands
and task_complete as the original (a turn that fails this is kept as served and listed); and each episode is also rendered unstripped, so every turn's owner, reasoning span
text and cut, and every student turn's text are compared unchanged.
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

# 09-21 chat/tool markers Qwen copies from the student's turns (census of the final relay arm, 2026-09-27: every one
# also appears in 09-21's own turns); literal strings, removed only outside the executed JSON object.
COPIED_MARKERS = ('<|start_think|>', '<|end_think|>', '<tool_call>', '</tool_call>', '<|eot_id|>', '<|eom_id|>',
                  '<|begin_of_text|>', '<|end_of_text|>', '<|python_tag|>', '<|end|>', '<think>', '</think>',
                  '<assistant>', '</assistant>', '<user>', '</user>', '</parameter>', '</function>', '</invoke>',
                  '</result>', '</tool_response>', '</final>', '</final_response>', '</final_output>',
                  '</user_response>', '</answer>', '</analysis>', '</output>', '</s>')
HEADER = re.compile(r'<\|start_header_id\|>[^<>\n]{0,24}<\|end_header_id\|>')
MARKER_RE = re.compile('|'.join([HEADER.pattern] + [re.escape(m) for m in
                                                    sorted(COPIED_MARKERS + ('<|start_header_id|>', '<|end_header_id|>'),
                                                           key=len, reverse=True)]))


def _tidy(before, after, obj):
    before = re.sub(r'\n[ \t]*\n(?:[ \t]*\n)+', '\n\n', before).rstrip()
    after = re.sub(r'\n[ \t]*\n(?:[ \t]*\n)+', '\n\n', after).strip()
    return (before + '\n\n' if before.strip() else '') + obj + ('\n' + after if after else '')


def strip_copied(content, parser):
    """(new content, [markers removed]) — markers removed before/after the JSON object the parser executes. When the
    parser's brace scan finds no object (prose with an unbalanced brace; the harness then runs its auto-fixes), only
    markers that directly open a JSON object (`<tool_call>\\n{`) or directly follow one (`}\\n</tool_call>`) go."""
    js, _ = parser._extract_json_content(content)
    if js:
        a = content.find(js)
        before, after = content[:a], content[a + len(js):]
        found = MARKER_RE.findall(before) + MARKER_RE.findall(after)
        if not found:
            return content, []
        new = _tidy(MARKER_RE.sub('', before), MARKER_RE.sub('', after), js)
    else:
        opener = re.compile(r'[ \t]*(?:' + MARKER_RE.pattern + r')\s*(?=\{)')
        closer = re.compile(r'(?<=\})\s*(?:' + MARKER_RE.pattern + r')')
        found = [MARKER_RE.search(m.group(0)).group(0) for m in opener.finditer(content)] + \
            [MARKER_RE.search(m.group(0)).group(0) for m in closer.finditer(content)]
        if not found:
            return content, []
        new = closer.sub('', opener.sub(lambda m: '\n\n' if m.start() else '', content)).strip()
        new = re.sub(r'\n[ \t]*\n(?:[ \t]*\n)+', '\n\n', new)
    return new, ['header' if m.startswith('<|start_header_id|>') and m.endswith('<|end_header_id|>') else m
                 for m in found]


def load_parser(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location('terminus_json_plain_parser', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TerminusJSONPlainParser()


def same_action(p, old, new):
    """The stripped reply parses with no error and executes exactly what the original did."""
    a, b = p.parse_response(old), p.parse_response(new)
    key = lambda r: ([(c.keystrokes, c.duration) for c in r.commands], r.is_task_complete)  # noqa: E731
    return dict(new_ok=not b.error, old_ok=not a.error, same=(not b.error) and (bool(a.error) or key(a) == key(b)),
                warn_before=('before JSON' in (b.warning or '')), warn_after=('after JSON' in (b.warning or '')))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--render-dir', required=True)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--arm', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--tokenizer-dir', required=True)
    ap.add_argument('--think-limit', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--strip-copied-markers', action='store_true',
                    help="remove 09-21's chat/tool markers from teacher content outside the executed JSON (see above)")
    ap.add_argument('--terminus-parser', help="harbor terminus_json_plain_parser.py (required with the strip)")
    a = ap.parse_args()
    if a.strip_copied_markers and not a.terminus_parser:
        ap.error('--strip-copied-markers needs --terminus-parser')
    sys.path.insert(0, os.path.abspath(a.render_dir))
    import render
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(a.render_dir)), 'pilot'))
    import readout
    name = os.path.basename(os.path.normpath(a.run_dir))
    want = {json.loads(l)['sid'] for l in open(a.manifest) if l.strip()}
    tok = render.rcap.load_tokenizer(os.path.join(a.tokenizer_dir, 'tokenizer.json'))
    tpl = render.load_template(os.path.join(a.tokenizer_dir, 'chat_template.jinja'))
    bos = json.load(open(os.path.join(a.tokenizer_dir, 'tokenizer_config.json'))).get('bos_token', '<|begin_of_text|>')
    bos = bos.get('content') if isinstance(bos, dict) else bos
    rv = readout.router_view(os.path.join(a.run_dir, f'router_{a.arm}'))
    recs = collections.defaultdict(list)
    for r in rv['recs']:
        if r['sid'] in want:
            recs[r['sid']].append(r)
    st = collections.Counter()
    max_think, done, over = 0, set(), []
    strip = dict(on=False, markers=collections.Counter(), turns=0, checks=collections.Counter(), bad=[])
    if a.strip_copied_markers:
        parser = load_parser(a.terminus_parser)
        orig_turns = render.episode_turns

        def episode_turns(traj, records):
            turns = orig_turns(traj, records)
            if not strip['on']:
                return turns
            for i, (role, text, m) in enumerate(turns):
                if role == 'assistant' and m['owner'] == 'teacher':
                    new, found = strip_copied(text, parser)
                    if found:
                        strip['markers'].update(found)
                        strip['turns'] += 1
                        strip['row_hit'] = True
                        c = same_action(parser, text, new)
                        strip['checks'].update(k for k, v in c.items() if v)
                        if not c['same']:   # never change what the harness executed: keep the turn as served
                            strip['bad'].append(dict(sid=traj.get('session_id'), turn=m.get('turn'), **c))
                            strip['markers'].subtract(found)
                            strip['turns'] -= 1
                            continue
                        turns[i] = (role, new, m)
            return turns
        render.episode_turns = episode_turns
    with open(a.out, 'w') as out:
        for d in readout.arm_job_dirs(a.run_dir, name, a.arm):
            for tj in sorted(glob.glob(os.path.join(d, '*', '**', 'agent', 'trajectory.json'), recursive=True)):
                traj = json.load(open(tj))
                sid = traj.get('session_id')
                if sid not in want or sid in done:
                    continue
                if a.strip_copied_markers:
                    strip['on'] = False
                    ref = render.render_episode(traj, recs[sid], tok, tpl, bos, think_limit=a.think_limit)
                    strip['on'], strip['row_hit'] = True, False
                row = render.render_episode(traj, recs[sid], tok, tpl, bos, think_limit=a.think_limit)
                render.check_row(row, tok)
                if a.strip_copied_markers:
                    st['rows_stripped'] += strip['row_hit']
                    st['split_checked_rows'] += 1
                    if not split_same(ref, row):
                        st['split_changed_rows'] += 1
                        strip['bad'].append(dict(sid=sid, split_changed=True))
                done.add(sid)
                st['rows'] += 1
                st['over_64k'] += not row['fits']
                st['trained_tokens'] += sum(row['loss'])
                for m in row['turns']:
                    if m['owner'] != 'teacher':
                        continue
                    st['qwen_turns'] += 1
                    n = m.get('reasoning_tokens') or 0
                    max_think = max(max_think, n)
                    uncut = m['cut_at'] is None and m.get('has_reasoning') and not m.get('autofix')
                    st['uncut_over_8192'] += bool(uncut and n > 8192)
                    st['uncut_over_16384'] += bool(uncut and n > 16384)
                    st['think_trained_turns'] += bool(m.get('think_trained'))
                if not row['fits']:
                    over.append(dict(sid=sid, n_tokens=row['n_tokens']))
                out.write(json.dumps(dict(sid=sid, n_tokens=row['n_tokens'], fits=row['fits'], ids=row['ids'],
                                          loss=row['loss'], turns=row['turns'])) + '\n')
    if a.strip_copied_markers:
        st['strip'] = dict(turns=strip['turns'], markers=dict(strip['markers'].most_common()),
                           checks=dict(strip['checks']), not_same=strip['bad'][:50], n_not_same=len(strip['bad']))
    res = dict(st, over_64k_rows=over, max_reasoning_tokens_per_turn=max_think, missing=len(want - done),
               think_limit=a.think_limit,
               render_dir=os.path.abspath(a.render_dir))
    print(json.dumps(res, indent=1, default=str))


def split_same(ref, row):
    """Same turns and owners, the same reasoning span text and cut per teacher turn, the same student turn text."""
    if len(ref['turns']) != len(row['turns']):
        return False
    for x, y in zip(ref['turns'], row['turns']):
        if x['owner'] != y['owner'] or x.get('cut_at') != y.get('cut_at'):
            return False
        if x['owner'] == 'teacher':
            if (x.get('reasoning_tokens'), x.get('think_trained')) != (y.get('reasoning_tokens'), y.get('think_trained')):
                return False
            if ref['text'][x['span'][0]:x['think_end']] != row['text'][y['span'][0]:y['think_end']]:
                return False
        elif ref['text'][x['span'][0]:x['span'][1]] != row['text'][y['span'][0]:y['span'][1]]:
            return False
    return True


if __name__ == '__main__':
    main()
