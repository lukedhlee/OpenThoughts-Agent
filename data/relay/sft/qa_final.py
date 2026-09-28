#!/usr/bin/env python3
"""QA of the two final SFT arms (coordinator 2026-09-28, "really curated, good data"), both arms alike.

    python qa_final.py --relay <run dir> --baseline <run dir> --manifest final_v2_manifest.jsonl \
        --rendered final_v2_rendered_think16k_clean.jsonl --terminus-parser <harbor>/.../terminus_json_plain_parser.py \
        --tokenizer <09-21 tokenizer.json> [--new-tasks <attempt-6 task list>] [--min-trained 200] [--apply]

Per arm, over the rendered rows (decoded with 09-21's tokenizer; the per-turn char spans come from render.py):
  fit       rows over 65,536 tokens; ids/loss length mismatches; rows with fewer than --min-trained trained tokens
  markers   trained Qwen turns whose TRAINED tokens contain a copied 09-21 marker (render_think_limit.MARKER_RE), as a
            share of trained Qwen turns
  parse     trained Qwen turns whose content (after the think span) Terminus-2's own parser rejects (error), and those
            it accepts with a warning
  student   trained tokens inside 09-21 turns: in non-autofixed turns (must be 0) and in autofixed ones (the rewritten
            action, trained by design; render.py autofix_loss)
  fairness  unique tasks, overlap between arms, rows per task, passes/failures, failure mix, and the share of failures
            on --new-tasks
--apply: rows under --min-trained are dropped from their arm, and both arms are trimmed back to the same N passes and
N failures (seeded uniform draw of the extra rows; the rows kept are a subset, so the rendered file is subset, not
re-rendered). The pre-QA files are kept as <name>.pre_qa. Writes qa_final.json to both run dirs.
"""
import argparse
import bisect
import collections
import json
import os
import random
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from render_think_limit import MARKER_RE, load_parser  # noqa: E402

EOT = '<|eot_id|>'


def read(p):
    return [json.loads(line) for line in open(p) if line.strip()]


def row_qa(r, tok, parser):
    ids, loss = r['ids'], r['loss']
    text = tok.decode(ids, skip_special_tokens=False)
    enc = tok.encode(text, add_special_tokens=False)
    starts = [s for s, _ in enc.offsets]
    at = lambda ch: bisect.bisect_left(starts, ch)  # noqa: E731
    q = dict(n_tokens=len(ids), len_ok=len(ids) == len(loss) == r['n_tokens'], ids_ok=enc.ids == ids,
             trained=sum(loss), q_turns=0, q_marker=0, q_parse_error=0, q_parse_warning=0, student_trained=0,
             student_autofix_trained=0, parse_errors=[])
    for t in r['turns']:
        s, e = t['span']
        a, b = at(s), at(e)
        tr = sum(loss[a:b])
        if t['owner'] == 'student':
            q['student_autofix_trained' if t.get('autofix') else 'student_trained'] += tr
            continue
        if t['owner'] != 'teacher' or tr == 0:
            continue
        q['q_turns'] += 1
        # trained CONTENT of the turn (after its think span; the template's own think markers are not copies): the
        # tokens with loss 1, decoded, without the turn's closing <|eot_id|>
        c0 = at(t.get('think_end', s))
        trained_text = tok.decode([i for i, w in zip(ids[c0:b], loss[c0:b]) if w], skip_special_tokens=False)
        if trained_text.endswith(EOT):
            trained_text = trained_text[:-len(EOT)]
        if MARKER_RE.search(trained_text):
            q['q_marker'] += 1
        content = text[t.get('think_end', s):e]
        if content.endswith(EOT):
            content = content[:-len(EOT)]
        res = parser.parse_response(content)
        if res.error:
            q['q_parse_error'] += 1
            q['parse_errors'].append(res.error[:120])
        elif res.warning:
            q['q_parse_warning'] += 1
    return q


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--relay', required=True)
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--manifest', default='final_v2_manifest.jsonl')
    ap.add_argument('--rendered', default='final_v2_rendered_think16k_clean.jsonl')
    ap.add_argument('--terminus-parser', required=True)
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--new-tasks')
    ap.add_argument('--min-trained', type=int, default=200)
    ap.add_argument('--max-tokens', type=int, default=65536)
    ap.add_argument('--seed', type=int, default=20260928)
    ap.add_argument('--apply', action='store_true')
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    parser = load_parser(a.terminus_parser)
    new = {t.strip() for t in open(a.new_tasks) if t.strip()} if a.new_tasks else set()
    arms = {'relay': a.relay, 'baseline': a.baseline}
    man, qa, out = {}, {}, dict(min_trained=a.min_trained, max_tokens=a.max_tokens)
    for k, d in arms.items():
        man[k] = {m['sid']: m for m in read(os.path.join(d, a.manifest))}
        qa[k] = {}
        for line in open(os.path.join(d, a.rendered)):
            r = json.loads(line)
            qa[k][r['sid']] = row_qa(r, tok, parser)
        rows = list(qa[k].values())
        errs = collections.Counter(e for x in rows for e in x['parse_errors'])
        qt = sum(x['q_turns'] for x in rows)
        out[k] = dict(
            rows=len(rows), manifest_rows=len(man[k]), missing_rendered=len(set(man[k]) - set(qa[k])),
            over_max=sum(x['n_tokens'] > a.max_tokens for x in rows), len_mismatch=sum(not x['len_ok'] for x in rows),
            ids_roundtrip_mismatch=sum(not x['ids_ok'] for x in rows),
            under_min_trained=sorted(s for s, x in qa[k].items() if x['trained'] < a.min_trained),
            trained_tokens=sum(x['trained'] for x in rows), trained_min=min(x['trained'] for x in rows),
            q_turns=qt, q_marker_turns=sum(x['q_marker'] for x in rows),
            q_marker_rate=round(sum(x['q_marker'] for x in rows) / max(1, qt), 5),
            q_parse_error_turns=sum(x['q_parse_error'] for x in rows),
            q_parse_warning_turns=sum(x['q_parse_warning'] for x in rows), parse_error_kinds=errs.most_common(5),
            student_trained_tokens=sum(x['student_trained'] for x in rows),
            student_autofix_trained_tokens=sum(x['student_autofix_trained'] for x in rows))
    drop = {k: set(out[k]['under_min_trained']) for k in arms}
    if a.apply and any(drop.values()):
        rng = random.Random(a.seed)
        keep = {k: [m for s, m in man[k].items() if s not in drop[k]] for k in arms}
        n = min(min(sum(1 for m in v if m['passed']), sum(1 for m in v if not m['passed'])) for v in keep.values())
        for k, d in arms.items():
            p = sorted((m for m in keep[k] if m['passed']), key=lambda m: m['trial'])
            f = sorted((m for m in keep[k] if not m['passed']), key=lambda m: m['trial'])
            final = rng.sample(p, n) + rng.sample(f, n)
            sids = {m['sid'] for m in final}
            for name in (a.manifest, a.rendered):
                shutil.copy(os.path.join(d, name), os.path.join(d, name + '.pre_qa'))
            with open(os.path.join(d, a.manifest), 'w') as fh:
                for m in final:
                    fh.write(json.dumps(m) + '\n')
            with open(os.path.join(d, a.rendered + '.pre_qa')) as src, open(os.path.join(d, a.rendered), 'w') as dst:
                for line in src:
                    if json.loads(line[:line.find('"ids"')].rstrip(', ') + '}')['sid'] in sids:
                        dst.write(line)
            man[k] = {m['sid']: m for m in final}
        out['applied'] = dict(dropped={k: len(v) for k, v in drop.items()}, N=n)
    # fairness, on the (possibly trimmed) manifests
    tasks = {}
    for k in arms:
        ms = list(man[k].values())
        per = collections.Counter(m['task'] for m in ms)
        fails = [m for m in ms if not m['passed']]
        tasks[k] = set(per)
        out[k].update(final_rows=len(ms), passes=len(ms) - len(fails), failures=len(fails), unique_tasks=len(per),
                      rows_per_task=dict(sorted(collections.Counter(per.values()).items())),
                      failure_mix=dict(collections.Counter(m['cause'] for m in fails)),
                      failures_on_new_tasks=sum(1 for m in fails if m['task'] in new),
                      passes_on_new_tasks=sum(1 for m in ms if m['passed'] and m['task'] in new),
                      failure_share_new=round(sum(1 for m in fails if m['task'] in new) / max(1, len(fails)), 4))
    tr, tb = tasks['relay'], tasks['baseline']
    out['task_overlap'] = dict(both=len(tr & tb), relay_only=len(tr - tb), baseline_only=len(tb - tr),
                               jaccard=round(len(tr & tb) / max(1, len(tr | tb)), 4))
    for d in arms.values():
        json.dump(out, open(os.path.join(d, 'qa_final.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
