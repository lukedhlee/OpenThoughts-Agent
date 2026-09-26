#!/usr/bin/env python3
"""analyze.py — classify the replayed confirmation replies with Terminus-2's own parser and compare A vs B.

Classes (one per reply, from harbor 89098635 terminus_json_plain_parser.py; warnings pass, as in the harness):
  parse_error     the parser returns an error (Terminus-2 would re-prompt)
  confirm         task_complete true; sub-count confirm_with_cmds (Terminus-2 runs them but ends the episode unseen)
  runs_first      task_complete false/absent with >= 1 command; split into
                    check  every non-trivial command only reads or runs (cat/ls/grep/diff/test/python/pytest/...)
                    other  at least one command writes, edits, installs, deletes or starts something in the background
  no_action       task_complete false, no commands

Rates are per takeover first (mean over its samples), then averaged; 95% CIs are a bootstrap over takeovers
(cluster = takeover), B-A paired within takeover. Login-node safe (stdlib only).

    python3 analyze.py --requests requests.jsonl.gz --replies replies.jsonl --parser <harbor>/.../terminus_json_plain_parser.py
"""
import argparse
import collections
import gzip
import importlib.util
import json
import random
import re
import statistics

TRIVIAL = re.compile(r'^\s*(C-[a-z]|clear|Enter|q|)\s*$')
# "other" = the command changes something outside /tmp (edits, writes, deletes, installs, restarts, background
# services); reading, running tests/programs and builds count as checking. Heredoc bodies are not scanned as shell;
# inline python (python -c / python - <<) counts as writing only if it opens a file for writing or removes/moves one.
WRITE_CMDS = {'rm', 'rmdir', 'mv', 'cp', 'mkdir', 'touch', 'chmod', 'chown', 'ln', 'truncate', 'dd', 'install'}
ALWAYS_MOD = re.compile(
    r'^(pip3?|uv|apt(-get)?|yum|dnf|conda|npm|yarn|kill|pkill|killall|nohup|systemctl|service|vi|vim|nano|crontab|'
    r'useradd|passwd|patch)\b|^git (add|commit|checkout|reset|apply|stash|merge|rebase|restore|rm|mv)\b|'
    r'^python3? -m pip\b|^(sed|perl) (-\S*\s+)*-i|^sed -[a-zA-Z]*i')
PY_CALL = re.compile(r'\bpython3?(\.\d+)? (-u )?(-c\b|-\s*<<)')
PY_WRITE = re.compile(r"open\([^)]*['\"](w|a|wb|ab|w\+|r\+|x)['\"]|\.write_text\(|\.write_bytes\(|os\.(remove|unlink|rename|replace|"
                      r"makedirs|mkdir)\(|shutil\.|\.to_csv\(|\.save\(|torch\.save\(")
REDIR = re.compile(r'(?<![0-9&<])(>>?|\btee( -a)?)\s*([^\s;&|]+)')


def _scratch(path):
    return path.strip('\'"').startswith(('/tmp', '/dev/null', '/dev/stdout', '/dev/stderr', '&'))


def shell_modifies(s):
    s = s.split('<<', 1)[0]                       # heredoc body is data, not shell
    for seg in re.split(r'&&|\|\||[;|\n]', s):
        seg = seg.strip()
        seg = re.sub(r'^(sudo\s+|env\s+|([A-Z_][A-Z0-9_]*=\S*\s+))+', '', seg)
        if not seg:
            continue
        if ALWAYS_MOD.search(seg):
            return True
        toks = seg.split()
        if toks[0] in WRITE_CMDS:
            paths = [t for t in toks[1:] if not t.startswith('-')]
            if not paths or not all(_scratch(p) for p in paths):
                return True
        for m in REDIR.finditer(seg):
            if not _scratch(m.group(3)):
                return True
    return bool(re.search(r'(?<![&>])&\s*$', s.strip()))   # started in the background


def is_modifying(ks):
    m = PY_CALL.search(ks)
    if m:
        return shell_modifies(ks[:m.start()]) or bool(PY_WRITE.search(ks[m.start():]))
    return shell_modifies(ks)


def load_parser(path):
    spec = importlib.util.spec_from_file_location('terminus_json_plain_parser', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TerminusJSONPlainParser()


def classify(parser, content):
    pr = parser.parse_response(content or '')
    if pr.error:
        return 'parse_error', []
    cmds = [c.keystrokes for c in pr.commands if not TRIVIAL.match(c.keystrokes.strip().replace('\\n', ''))]
    if pr.is_task_complete:
        return ('confirm_with_cmds' if cmds else 'confirm'), cmds
    if not cmds:
        return 'no_action', cmds
    return ('runs_first_other' if any(is_modifying(k) for k in cmds) else 'runs_first_check'), cmds


def boot(vals, reps=10000, seed=0):
    rnd = random.Random(seed)
    n = len(vals)
    ms = sorted(sum(rnd.choice(vals) for _ in range(n)) / n for _ in range(reps))
    return sum(vals) / n, ms[int(0.025 * reps)], ms[int(0.975 * reps) - 1]


def fmt(t):
    m, lo, hi = t
    return f'{m:.2f} [{lo:.2f}, {hi:.2f}]'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--requests', required=True)
    p.add_argument('--replies', required=True)
    p.add_argument('--parser', required=True)
    p.add_argument('--out', default=None, help='per-reply classification jsonl')
    a = p.parse_args()
    parser = load_parser(a.parser)
    reqs = {r['key']: r for r in map(json.loads, gzip.open(a.requests, 'rt'))}
    rows = [json.loads(l) for l in open(a.replies)]
    rows = [r for r in rows if r.get('status') == 200]
    seen = {}
    for r in rows:   # a resumed run may have written a sample twice: keep the first
        seen.setdefault((r['key'], r['variant'], r['sample']), r)
    rows = list(seen.values())
    for r in rows:
        r['cls'], r['cmds'] = classify(parser, r.get('content'))
    # the replies Qwen actually gave in the runs (one per takeover, for reference)
    orig = collections.Counter(classify(parser, q['orig_content'])[0] for q in reqs.values() if q.get('orig_content'))
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by[r['key']][r['variant']].append(r)
    keys = [k for k in reqs if by[k]['A'] and by[k]['B']]

    def rate(k, v, pred):
        xs = by[k][v]
        return sum(pred(x) for x in xs) / len(xs)
    preds = {
        'runs commands first': lambda x: x['cls'].startswith('runs_first'),
        '  of which check-only': lambda x: x['cls'] == 'runs_first_check',
        'confirms immediately (no cmds)': lambda x: x['cls'] == 'confirm',
        'confirms with cmds (unseen)': lambda x: x['cls'] == 'confirm_with_cmds',
        'no action, not confirmed': lambda x: x['cls'] == 'no_action',
        'parse error': lambda x: x['cls'] == 'parse_error',
        'cut at cap (finish=length)': lambda x: x.get('finish_reason') == 'length',
    }
    out = []
    out.append(f'takeovers with both variants: {len(keys)}; replies: {len(rows)} '
               f'(A {sum(len(by[k]["A"]) for k in keys)}, B {sum(len(by[k]["B"]) for k in keys)})')
    out.append(f'original run replies (1 per takeover): {dict(orig)}')
    out.append(f'{"":34s} {"A (unchanged)":20s} {"B (+note)":20s} {"B - A (paired)":20s}')
    res = {}
    for name, f in preds.items():
        A = [rate(k, 'A', f) for k in keys]
        B = [rate(k, 'B', f) for k in keys]
        D = [b - x for b, x in zip(B, A)]
        res[name] = dict(A=boot(A), B=boot(B), D=boot(D))
        out.append(f'{name:34s} {fmt(res[name]["A"]):20s} {fmt(res[name]["B"]):20s} {fmt(res[name]["D"]):20s}')
    for v in 'AB':
        rt = [x.get('reasoning_tokens') or 0 for k in keys for x in by[k][v]]
        ct = [(x.get('usage') or {}).get('completion_tokens') or 0 for k in keys for x in by[k][v]]
        out.append(f'thinking tokens {v}: mean {statistics.mean(rt):.0f}, median {statistics.median(rt):.0f}, '
                   f'p90 {sorted(rt)[int(0.9 * len(rt))]}; completion tokens mean {statistics.mean(ct):.0f}')
    f = preds['runs commands first']
    fc = preds['  of which check-only']
    for label, sel in (('episode FAILED (reward 0)', lambda q: q['reward'] == 0), ('episode PASSED (reward 1)', lambda q: q['reward'] == 1)):
        ks = [k for k in keys if sel(reqs[k])]
        if not ks:
            continue
        A = [rate(k, 'A', f) for k in ks]
        B = [rate(k, 'B', f) for k in ks]
        D = [b - x for b, x in zip(B, A)]
        Dc = [rate(k, 'B', fc) - rate(k, 'A', fc) for k in ks]
        out.append(f'{label} n={len(ks)}: runs-first A {fmt(boot(A))}, B {fmt(boot(B))}, B-A {fmt(boot(D))}; '
                   f'check-only B-A {fmt(boot(Dc))}')
    print('\n'.join(out))
    if a.out:
        with open(a.out, 'w') as fo:
            for r in rows:
                fo.write(json.dumps(dict(key=r['key'], variant=r['variant'], sample=r['sample'], cls=r['cls'], cmds=r['cmds'],
                                         reasoning_tokens=r.get('reasoning_tokens'), reward=reqs[r['key']]['reward'])) + '\n')


if __name__ == '__main__':
    main()
