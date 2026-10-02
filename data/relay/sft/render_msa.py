#!/usr/bin/env python3
"""Relay episodes under mini-swe-agent tool mode (router --harness msa) -> 09-21 SFT rows, with the QA tags.

One row per episode, rendered with 09-21's own chat template exactly as the student saw the conversation at its end:
  * system + task: mini-swe-agent's messages, with the `bash` tool schema in the template's tools block (harbor sends
    tools=[BASH_TOOL] on every request);
  * student turns: as harbor re-sent them (reasoning_content -> <|start_think|>..<|end_think|>, prose, the tool call
    re-rendered from its structured form as 09-21's `<tool_call>\\n{"name": .., "arguments": {..}}\\n</tool_call>`);
  * teacher turns: the same form, with Qwen's reasoning stripped; every teacher turn but the last has its reasoning cut
    to reasoning_cap.CAP_TOKENS at a sentence boundary (the router's for_student view, same function);
  * tool results: as harbor sent them (name "bash"); upstream's format-error user messages verbatim.
Loss as sft/render.py (Terminus-2): teacher turn = content + tool call + <|eot_id|> always, its reasoning span only if
never cut and <= THINK_LIMIT tokens; student turns, router turns, system/user/tool turns masked (autofix loss none:
09-28's arm C showed training autofixed student actions adds nothing).
The row is checked against the router's own count: the student-view token count the router logged for the episode's
last student-view request must equal the rendered prefix up to that request (+ the generation prompt).

Candidates (select_kept.arm_rows: scored, the teacher wrote a turn, S5 = after a done-claim takeover the teacher ran a
command other than the submit) are tagged with the five Terminus-2 QA filters, re-measured here:
  leak     a trained teacher text (content or reasoning) names another / the previous agent or the note;
  hunt     any command reads the grader's files (/tests, /logs/verifier, /setup_files, daytona);
  canary   a benchmark canary GUID anywhere in the episode;
  noisy    a failure whose verifier never ran its tests (final_v2's rule) or a weak timeout (did not stall);
  autofix  rows render with student autofix turns masked (counted, never dropped).

    python render_msa.py --runs <run dir> ... --tokenizer-dir <09-21 or H9 dir> --out-dir <dir> [--procs 8]
writes <out>/rows.jsonl (ids, loss, tags, outcome per candidate) and <out>/qa.json.
"""
import argparse
import collections
import json
import multiprocessing as mp
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'router'))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'pilot'))
import reasoning_cap as rcap  # noqa: E402
import readout  # noqa: E402
import select_kept  # noqa: E402
from hz_pool_census import CANARY, HUNT, LEAK, RAN  # noqa: E402

ARM = 'relay_repair'
A_HDR = '<|start_header_id|>assistant<|end_header_id|>\n'
START, END, EOT = '<|start_think|>', '<|end_think|>', '<|eot_id|>'
MAX_TOKENS = 65536
THINK_LIMIT = 8192
BASH_TOOL = {'type': 'function', 'function': {'name': 'bash', 'description': 'Execute a bash command', 'parameters': {
    'type': 'object', 'properties': {'command': {'type': 'string', 'description': 'The bash command to execute'}},
    'required': ['command']}}}


def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    """transformers' chat-template tojson (vLLM renders with it), not jinja2's HTML-safe one."""
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def load_template(path):
    import jinja2
    src = re.sub(r'\{%-?\s*(end)?generation\s*-?%\}', '', open(path).read())
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=['jinja2.ext.loopcontrols'])
    env.filters['tojson'] = tojson
    env.globals['raise_exception'] = lambda m: (_ for _ in ()).throw(jinja2.TemplateError(m))
    return env.from_string(src)


def owners(records):
    """first tool-call id of each reply the router returned -> its record."""
    out = {}
    for r in records:
        m = (((r.get('response') or {}).get('choices') or [{}])[0].get('message')) or {}
        tcs = m.get('tool_calls') or []
        if tcs and r.get('upstream_status') in (200, None):
            out[tcs[0]['id']] = r
    return out


def view(m):
    """An upstream mini-swe-agent message as the chat request carried it (harbor tool_mode_message_view + names)."""
    v = {'role': m['role'], 'content': m.get('content') or ''}
    if m.get('tool_calls'):
        v['tool_calls'] = [{'id': c['id'], 'type': 'function',
                            'function': {'name': c['function']['name'], 'arguments': c['function']['arguments']}}
                           for c in m['tool_calls']]
    if m['role'] == 'tool':
        v['tool_call_id'] = m['tool_call_id']
        v['name'] = 'bash'
    if m.get('reasoning_content'):
        v['reasoning_content'] = m['reasoning_content']
    return v


def as_rendered(m):
    """vLLM json.loads tool-call arguments before the template renders them."""
    if not m.get('tool_calls'):
        return m
    tcs = []
    for c in m['tool_calls']:
        try:
            a = json.loads(c['function']['arguments'])
        except (TypeError, ValueError):
            a = c['function']['arguments']
        tcs.append(dict(c, function=dict(c['function'], arguments=a)))
    return dict(m, tool_calls=tcs)


def student_view(msgs, by_id, tok, cap=rcap.CAP_TOKENS, think_limit=THINK_LIMIT, strip_student=False):
    """(messages as the router's for_student showed them, per-assistant-turn info): teacher reasoning stripped, every
    teacher turn but the last of THIS list cut to `cap` tokens; strip_student (router --student-view-after-takeover
    strip, after a sticky takeover): the student's turns without their thinking."""
    info, teacher_idx = [], []
    for i, m in enumerate(msgs):
        if m['role'] == 'assistant':
            r = by_id.get((m.get('tool_calls') or [{}])[0].get('id'))
            if r is None:
                raise ValueError(f'assistant message {i} has no router record')
            info.append(dict(i=i, owner=r['owner'], autofix=bool(r.get('autofix')), repair=bool(r.get('repair')),
                             turn=r.get('turn')))
            if r['owner'] == 'teacher':
                teacher_idx.append(i)
    last_t = teacher_idx[-1] if teacher_idx else None
    out, k = [], 0
    for i, m in enumerate(msgs):
        if m['role'] != 'assistant':
            out.append(m)
            continue
        meta = info[k]
        k += 1
        if meta['owner'] == 'student' and strip_student:
            m = {kk: v for kk, v in m.items() if kk != 'reasoning_content'}
        if meta['owner'] == 'teacher':
            r_full = (m.get('reasoning_content') or '').strip()
            r, at = (rcap.cut_reasoning(r_full, tok, cap) if (i != last_t and r_full) else (r_full, None))
            m = {kk: v for kk, v in m.items() if kk != 'reasoning_content'}
            if r:
                m['reasoning_content'] = r
            n_think = len(tok.encode(r_full, add_special_tokens=False).ids) if r_full else 0
            meta.update(cut_at=at, reasoning_tokens=n_think, think_trained=bool(r_full) and at is None and n_think <= think_limit)
        out.append(m)
    return out, info


def takeover_seq(recs):
    return min((r['seq'] for r in recs if r.get('takeover')), default=None)


def render(msgs, recs, tok, tpl, bos, cap=rcap.CAP_TOKENS, think_limit=THINK_LIMIT, strip_after_takeover=False):
    by_id = owners(recs)
    full = [view(m) for m in msgs if m.get('role') != 'exit']
    msgs = list(full)
    while msgs and msgs[-1]['role'] != 'assistant':      # observations after the last reply train nothing
        msgs.pop()
    out, info = student_view(msgs, by_id, tok, cap, think_limit,
                             strip_student=strip_after_takeover and takeover_seq(recs) is not None)
    text = tpl.render(messages=[as_rendered(m) for m in out], tools=[BASH_TOOL], bos_token=bos, add_generation_prompt=False)
    loss_ranges, pos = [], 0
    for meta in info:
        h = text.index(A_HDR, pos)
        start = h + len(A_HDR)
        end = text.index(EOT, start) + len(EOT)
        body = text[start:end]
        think_end = start + body.index(END) + len(END) if (body.startswith(START) and END in body) else start
        if meta['owner'] == 'teacher':
            loss_ranges.append((start, end) if (meta['think_trained'] or think_end == start) else (think_end, end))
        meta.update(span=(start, end), think_end=think_end)
        pos = end
    enc = tok.encode(text, add_special_tokens=False)
    loss = [1 if any(s >= a and e <= b for a, b in loss_ranges) else 0 for s, e in enc.offsets]
    return dict(ids=enc.ids, offsets=enc.offsets, loss=loss, text=text, turns=info, n_tokens=len(enc.ids),
                fits=len(enc.ids) <= MAX_TOKENS, messages=out, full=full, by_id=by_id, strip=strip_after_takeover)


def view_count(row, recs, tok, tpl, bos):
    """(router's logged student-view count of the episode's last counted request, the same request rendered here)."""
    last = max((r for r in recs if r.get('student_view_tokens') is not None), key=lambda r: r['seq'], default=None)
    if last is None:
        return None, None
    n_prefix = last['n_messages']
    if len(row['full']) < n_prefix:
        return last['student_view_tokens'], None
    ts = takeover_seq(recs)
    msgs, _ = student_view(row['full'][:n_prefix], row['by_id'], tok,
                           strip_student=row['strip'] and ts is not None and ts < last['seq'])
    text = tpl.render(messages=[as_rendered(m) for m in msgs], tools=[BASH_TOOL], bos_token=bos, add_generation_prompt=True)
    return last['student_view_tokens'], len(tok.encode(text, add_special_tokens=False).ids)


def tag(row, msgs, recs, traj_text):
    trained, i, loss, ids = [], 0, row['loss'], row['ids']
    teacher_text = []
    for meta in row['turns']:
        if meta['owner'] == 'teacher':
            m = row['messages'][meta['i']]
            teacher_text.append((m.get('content') or '') + '\n' + (m.get('reasoning_content') or ''))
    cmds = [json.dumps(c['function'].get('arguments')) for m in msgs if m.get('role') == 'assistant'
            for c in (m.get('tool_calls') or [])]
    return dict(leak=any(LEAK.search(t) for t in teacher_text),
                hunt=any(HUNT.search(c) for c in cmds),
                canary=bool(CANARY.search(traj_text)))


def one_run(run):
    name = os.path.basename(os.path.normpath(run))
    tok = rcap.load_tokenizer(os.path.join(TOKDIR, 'tokenizer.json'))
    tpl = load_template(os.path.join(TOKDIR, 'chat_template.jinja'))
    rv = readout.router_view(os.path.join(run, f'router_{ARM}'))
    start = next((e for e in rv['events'] if e.get('event') == 'start'), {})
    argv = start.get('argv') or []
    strip = 'strip' in argv[argv.index('--student-view-after-takeover') + 1:][:1] if '--student-view-after-takeover' in argv else False
    recs = collections.defaultdict(list)
    for r in rv['recs']:
        recs[r['sid']].append(r)
    trials = {t['sid']: t for t in readout.arm_trials(run, name, ARM)}
    out, stats = [], collections.Counter()
    for c in select_kept.arm_rows(run, name, ARM):
        t = trials.get(c['sid']) or {}
        up = os.path.join(os.path.dirname(t.get('traj_path') or ''), readout.MSA_TRAJECTORY)
        try:
            msgs = json.load(open(up))['messages']
            traj_text = open(up).read()
            row = render(msgs, recs[c['sid']], tok, tpl, BOS, strip_after_takeover=strip)
        except (OSError, ValueError, KeyError) as e:
            stats['render_error: ' + str(e)[:60]] += 1
            continue
        logged, mine = view_count(row, recs[c['sid']], tok, tpl, BOS)
        tags = tag(row, msgs, recs[c['sid']], traj_text)
        so = ''
        rj = t['traj_path']
        while rj != '/' and not os.path.exists(os.path.join(rj, 'result.json')):
            rj = os.path.dirname(rj)
        rj = os.path.join(rj, 'result.json')
        if not c['passed'] and os.path.exists(rj):
            so = (json.load(open(rj)).get('verifier_result') or {}).get('stdout') or ''
        ran = c['passed'] or bool(RAN.search(so))
        weak = c['cause'] == 'timeout' and not c.get('stalled')
        out.append(dict(c, run=name, n_tokens=row['n_tokens'], fits=row['fits'], trained_tokens=sum(row['loss']),
                        teacher_turns_rendered=sum(1 for m in row['turns'] if m['owner'] == 'teacher'),
                        cut_turns=sum(1 for m in row['turns'] if m.get('cut_at') is not None),
                        long_think_masked=sum(1 for m in row['turns'] if m['owner'] == 'teacher' and m.get('cut_at') is None
                                              and m.get('reasoning_tokens', 0) > THINK_LIMIT),
                        autofix_student_turns=sum(1 for m in row['turns'] if m['owner'] == 'student' and m['autofix']),
                        view_count_logged=logged, view_count_rendered=mine, verifier_ran=ran, weak_timeout=weak,
                        noisy=(not c['passed']) and (not ran or weak), **tags,
                        ids=row['ids'], loss=row['loss']))
    return out, stats


def main():
    global TOKDIR, BOS, ARM
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--tokenizer-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--arm', default=ARM)
    ap.add_argument('--procs', type=int, default=4)
    a = ap.parse_args()
    TOKDIR, ARM = a.tokenizer_dir, a.arm
    bos = json.load(open(os.path.join(TOKDIR, 'tokenizer_config.json'))).get('bos_token', '<|begin_of_text|>')
    BOS = bos.get('content') if isinstance(bos, dict) else bos
    os.makedirs(a.out_dir, exist_ok=True)
    with mp.Pool(min(a.procs, len(a.runs))) as pool:
        res = pool.map(one_run, a.runs)
    rows = [r for rs, _ in res for r in rs]
    stats = sum((s for _, s in res), collections.Counter())
    with open(os.path.join(a.out_dir, 'rows.jsonl'), 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    n = len(rows) or 1
    cmp_ = [(r['view_count_logged'], r['view_count_rendered']) for r in rows if r['view_count_rendered'] is not None]
    clean = [r for r in rows if r['fits'] and not (r['leak'] or r['hunt'] or r['canary'] or r['noisy'])]
    qa = dict(candidates=len(rows), passes=sum(r['passed'] for r in rows), render_errors=dict(stats),
              tag_rates={k: round(sum(bool(r[k]) for r in rows) / n, 4) for k in ('leak', 'hunt', 'canary', 'noisy')},
              leak_rate_done_claim=round(sum(r['leak'] for r in rows if r['takeover'] == 'done_claim')
                                         / max(1, sum(r['takeover'] == 'done_claim' for r in rows)), 4),
              over_64k=sum(not r['fits'] for r in rows),
              view_count_check=dict(compared=len(cmp_), equal=sum(1 for x, y in cmp_ if x == y),
                                    max_abs_diff=max((abs(x - y) for x, y in cmp_), default=None)),
              rows_with_autofixed_student_turns=sum(1 for r in rows if r['autofix_student_turns']),
              clean=dict(rows=len(clean), passes=sum(r['passed'] for r in clean), tasks=len({r['task'] for r in clean}),
                         trained_tokens=sum(r['trained_tokens'] for r in clean),
                         tokens_p50=sorted(r['n_tokens'] for r in clean)[len(clean) // 2] if clean else None),
              takeovers=dict(collections.Counter(r['takeover'] or 'repair_only' for r in rows)),
              failure_causes=dict(collections.Counter(r['cause'] for r in rows if not r['passed'])))
    json.dump(qa, open(os.path.join(a.out_dir, 'qa.json'), 'w'), indent=1)
    print(json.dumps(qa, indent=1))


TOKDIR = BOS = None

if __name__ == '__main__':
    main()
