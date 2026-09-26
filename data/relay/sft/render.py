#!/usr/bin/env python3
"""Render a relay episode as one SFT row for 09-21, with its loss mask (Luke 2026-09-25 17:40 PT, training layout 1).

One sequence per episode, rendered with exactly the capped history the student saw at the end of the episode:
  * student turns: its own replies as served (inline <|start_think|>...<|end_think|> + Terminus-2 JSON);
  * teacher turns (repairs and takeover turns): <|start_think|>{reasoning}<|end_think|>{content} inline, the way the
    router showed them to the student; the final teacher turn keeps its reasoning whole, every older teacher turn's
    reasoning is cut to its first 1,000 tokens at a sentence/line boundary (router/reasoning_cap.py, same function);
  * observations: the user messages harbor sent, verbatim.
Loss (per token, 1 = trained):
  * teacher turn: visible content + <|eot_id|> always; its reasoning span (<|start_think|> .. <|end_think|>) only if the
    reasoning was never cut (<= cap, or the final teacher turn). A cut turn's whole span is masked, end marker
    included: a truncated prefix followed by <|end_think|> would teach abrupt stops;
  * student turn: masked, except an autofixed one (label autofix=True): with autofix_loss='content' (default) its
    rewritten action/format (content + <|eot_id|>) is trained, never its reasoning; 'none' masks it;
  * router turns (budget endings), system header and user turns: masked.

    row = render_episode(trajectory, router_records, tok, template, bos)       # dict(ids, loss, turns, n_tokens)

The chat template is 09-21's own chat_template.jinja, rendered with jinja2 (its {% generation %} tags removed; the
spans are located in the rendered text instead). Input: harbor's trajectory.json (trajectory_config.raw_content) and
the router's turns.jsonl records of that episode (joined by content hash, as readout.py does).
"""
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'router'))
import reasoning_cap as rcap  # noqa: E402

A_HDR = '<|start_header_id|>assistant<|end_header_id|>\n'
START, END, EOT = '<|start_think|>', '<|end_think|>', '<|eot_id|>'
MAX_TOKENS = 65536


def sha(s):
    return hashlib.sha1((s or '').encode('utf-8', 'surrogatepass')).hexdigest()[:16]


def load_template(path):
    import jinja2
    src = open(path).read()
    src = re.sub(r'\{%-?\s*(end)?generation\s*-?%\}', '', src)
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=['jinja2.ext.loopcontrols'])   # as HF's apply_chat_template
    env.globals['raise_exception'] = lambda m: (_ for _ in ()).throw(jinja2.TemplateError(m))
    return env.from_string(src)


def _obs_text(step):
    obs = step.get('observation') or {}
    return '\n'.join(r.get('content') or '' for r in obs.get('results', []) if isinstance(r.get('content'), str))


def episode_turns(traj, records):
    """[(role, text, meta)] from harbor's trajectory, owners joined from the router's records."""
    by_sha = {}
    for r in records:
        if r.get('content_sha') and r.get('turn') is not None:
            by_sha.setdefault(r['content_sha'], r)
    steps = [s for s in traj['steps'] if not s.get('is_copied_context')]
    first = next(s for s in steps if s.get('source') == 'user')
    turns = [('user', first['message'], {})]
    agent = [s for s in steps if s.get('source') == 'agent']
    for k, s in enumerate(agent):
        msg = s['message'] if isinstance(s.get('message'), str) else json.dumps(s.get('message'))
        rec = by_sha.get(sha(msg))
        if rec is None:
            raise ValueError(f'agent step {k} has no router record (content sha {sha(msg)})')
        turns.append(('assistant', msg, dict(owner=rec['owner'], reasoning=s.get('reasoning_content'),
                                             autofix=bool(rec.get('autofix')), repair=bool(rec.get('repair')),
                                             turn=rec.get('turn'))))
        if k < len(agent) - 1:
            turns.append(('user', _obs_text(s), {}))
    return turns


def render_episode(traj, records, tok, template, bos, autofix_loss='content', cap=rcap.CAP_TOKENS):
    turns = episode_turns(traj, records)
    teacher = [i for i, (role, _, m) in enumerate(turns) if role == 'assistant' and m['owner'] == 'teacher']
    final_teacher = teacher[-1] if teacher else None
    messages, info = [], []
    for i, (role, text, m) in enumerate(turns):
        if role == 'assistant' and m['owner'] == 'teacher':
            full, _, r_full = rcap.teacher_turn_for_student(text, m['reasoning'], tok, cut=False, cap=cap)
            shown, at, _ = rcap.teacher_turn_for_student(text, m['reasoning'], tok, cut=(i != final_teacher), cap=cap)
            messages.append(dict(role='assistant', content=shown))
            info.append(dict(i=i, owner='teacher', repair=m['repair'], cut_at=at, reasoning_chars=len(r_full),
                             has_reasoning=bool(r_full)))
        else:
            messages.append(dict(role=role, content=text))
            if role == 'assistant':
                info.append(dict(i=i, owner=m['owner'], autofix=m['autofix'], cut_at=None))
    text = template.render(messages=messages, bos_token=bos, add_generation_prompt=False)
    # char-level loss over the rendered text, assistant turn by assistant turn (headers are never trained)
    loss_ranges, pos, a = [], 0, 0
    for m in messages:
        if m['role'] != 'assistant':
            continue
        h = text.index(A_HDR, pos)
        start = h + len(A_HDR)
        end = text.index(EOT, start) + len(EOT)
        body = text[start:end]
        meta = info[a]
        a += 1
        think_end = start + body.index(END) + len(END) if body.startswith(START) else start
        if meta['owner'] == 'teacher':
            if meta['cut_at'] is None:
                loss_ranges.append((start, end))               # reasoning (uncut) + content + eot
            else:
                loss_ranges.append((think_end, end))           # content + eot; the cut span is masked, markers too
        elif meta['owner'] == 'student' and meta.get('autofix') and autofix_loss == 'content':
            loss_ranges.append((think_end, end))               # the rewritten action/format, not the reasoning
        meta.update(span=(start, end), think_end=think_end)
        pos = end
    enc = tok.encode(text, add_special_tokens=False)
    loss = [0] * len(enc.ids)
    for t, (s, e) in enumerate(enc.offsets):
        if any(s >= a0 and e <= b0 for a0, b0 in loss_ranges):
            loss[t] = 1
    return dict(ids=enc.ids, offsets=enc.offsets, loss=loss, text=text, turns=info, n_tokens=len(enc.ids),
                fits=len(enc.ids) <= MAX_TOKENS, loss_ranges=loss_ranges)


def check_row(row, tok):
    """Assert the mask rules on a rendered row; returns a dict of counts."""
    text, ids, loss = row['text'], row['ids'], row['loss']
    start_id, end_id, eot_id = (tok.token_to_id(x) for x in (START, END, EOT))
    counts = dict(teacher_turns=0, cut_turns=0, autofix_turns=0, trained_tokens=sum(loss))
    for m in row['turns']:
        s, e = m['span']
        toks = [k for k, (a, b) in enumerate(row['offsets']) if a >= s and b <= e]
        trained = [k for k in toks if loss[k]]
        if m['owner'] == 'teacher':
            counts['teacher_turns'] += 1
            assert toks and loss[toks[-1]] == 1 and ids[toks[-1]] == eot_id, 'teacher <|eot_id|> must be trained'
            think = [k for k in toks if row['offsets'][k][1] <= m['think_end']]
            if m['cut_at'] is not None:
                counts['cut_turns'] += 1
                assert think and not any(loss[k] for k in think), 'a cut reasoning span (markers included) is masked'
                assert any(ids[k] == end_id for k in think)
            elif think:
                assert all(loss[k] for k in think), 'an uncut teacher reasoning span is trained'
        elif m['owner'] == 'student' and m.get('autofix'):
            counts['autofix_turns'] += 1
            think = [k for k in toks if row['offsets'][k][1] <= m['think_end']]
            assert not any(loss[k] for k in think), "an autofixed turn's reasoning is never trained"
        else:
            assert not trained, f"{m['owner']} turn must be masked"
    return counts


def main():
    """Render every episode of a relay run: python render.py <run dir> <arm> <tokenizer dir> [--out rows.jsonl]."""
    import argparse
    import glob
    import collections
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'pilot'))
    import readout
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('arm')
    ap.add_argument('tokenizer_dir')
    ap.add_argument('--name')
    ap.add_argument('--out')
    a = ap.parse_args()
    name = a.name or os.path.basename(os.path.normpath(a.run_dir))
    tok = rcap.load_tokenizer(os.path.join(a.tokenizer_dir, 'tokenizer.json'))
    tpl = load_template(os.path.join(a.tokenizer_dir, 'chat_template.jinja'))
    bos = json.load(open(os.path.join(a.tokenizer_dir, 'tokenizer_config.json'))).get('bos_token', '<|begin_of_text|>')
    bos = bos.get('content') if isinstance(bos, dict) else bos
    rv = readout.router_view(os.path.join(a.run_dir, f'router_{a.arm}'))
    recs = collections.defaultdict(list)
    for r in rv['recs']:
        recs[r['sid']].append(r)
    out = open(a.out, 'w') if a.out else None
    stats = collections.Counter()
    lens = []
    for tj in sorted(glob.glob(os.path.join(a.run_dir, 'jobs', f'{name}_{a.arm}', '*', '**', 'agent', 'trajectory.json'), recursive=True)):
        traj = json.load(open(tj))
        sid = traj.get('session_id')
        if sid not in recs:
            stats['no_router_records'] += 1
            continue
        try:
            row = render_episode(traj, recs[sid], tok, tpl, bos)
            c = check_row(row, tok)
        except (ValueError, AssertionError, StopIteration) as e:
            stats['error: ' + str(e)[:60]] += 1
            continue
        stats['episodes'] += 1
        stats['fits_64k'] += row['fits']
        stats['with_cut'] += c['cut_turns'] > 0
        stats['cut_turns'] += c['cut_turns']
        stats['teacher_turns'] += c['teacher_turns']
        stats['autofix_turns'] += c['autofix_turns']
        lens.append(row['n_tokens'])
        if out:
            out.write(json.dumps(dict(sid=sid, n_tokens=row['n_tokens'], fits=row['fits'], ids=row['ids'],
                                      loss=row['loss'], turns=row['turns'])) + '\n')
    lens.sort()
    q = lambda f: lens[int(f * (len(lens) - 1))] if lens else None  # noqa: E731
    print(json.dumps(dict(stats, tokens_p50=q(.5), tokens_p90=q(.9), tokens_max=lens[-1] if lens else None), indent=1))


if __name__ == '__main__':
    main()
