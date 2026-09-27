#!/usr/bin/env python3
"""arm_quality.py — per-row quality facts for the two final SFT arms (relay vs Qwen alone), for the status gist.

Runs on the Jupiter login node with the snowball env's python (needs `tokenizers`; no torch). Read-only: it reads each
arm's final_manifest.jsonl and final_rendered_think16k.jsonl (the training set) and 09-21's tokenizer, and writes one
JSON line per row to stdout. status_figs.py aggregates them.

    OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 $PY arm_quality.py > arm_quality.jsonl
    $PY arm_quality.py --dump <sid> [<sid> ...]     # the decoded row, turn by turn, for hand-reading

A rendered row has ids, loss and per-assistant-turn char spans into the rendered text, not the text itself. The text is
recovered with tok.decode(ids) and re-encoded for token offsets; the re-encoded ids must equal the stored ones (checked
per row, counted as ids_mismatch).
"""
import argparse
import bisect
import json
import os
import re
import sys

RUNS = '/e/fscratch/reformo/lee27/experiments/relay/pilot/runs'
ARMS = {'relay': 'relay_full_relaym_20260926', 'baseline': 'relay_full_baseline6m_20260926'}
TOKENIZER = '/e/data1/mmlaion/lee27/models/grug-datakit-sft-20260921/tokenizer.json'
END, EOT = '<|end_think|>', '<|eot_id|>'
MARKERS = ('<|start_think|>', '<|end_think|>', '<tool_call>', '<|eot_id|>')   # chat markers that never belong in a reply's content
TASK_RE = re.compile(r'calibforge__([a-z_]+?)__([a-z0-9-]+?)_\d{8}_')
# what a command does, first match wins (keystrokes of one command)
TEST_RE = re.compile(r'\b(pytest|unittest|tox|nosetests|make\s+(test|check)|npm\s+(run\s+)?test|cargo\s+test|go\s+test|'
                     r'ctest|run_tests?|test_\w*\.(py|sh)|\w+_test\.(py|sh|go)|tests?/)', re.I)
RUN_RE = re.compile(r'(^|[;&|]\s*|\s)(python[0-9.]*|node|ruby|perl|php|java|go\s+run|cargo\s+run|bash|sh|Rscript|julia|'
                    r'\./[\w./-]+|make|gcc|g\+\+|cc|javac|sqlite3|psql|mysql|redis-cli|curl|wget|nc|dig|openssl)\b')
LOOK_RE = re.compile(r'(^|[;&|]\s*|\s)(cat|head|tail|less|diff|cmp|grep|rg|wc|ls|find|jq|md5sum|sha256sum|stat|file|tree|'
                     r'awk|sed\s+-n|od|xxd|du|ps|systemctl\s+status|git\s+(diff|status|log))\b')
GIVEUP_RE = re.compile(r"(cannot be (completed|done|solved)|unable to (complete|solve|finish)|give up|giving up|"
                       r"not possible to|impossible to|beyond (the )?scope|best (I|we) can do|as far as (I|we) can)", re.I)


def family(task):
    m = TASK_RE.search(task)
    return (m.group(1), m.group(2)) if m else ('?', '?')


def parse_reply(content):
    """(strict, lenient, obj): strict = the whole content is one JSON object; lenient = the outermost {...} parses."""
    c = content.strip()
    try:
        return True, True, json.loads(c)
    except ValueError:
        pass
    a, b = c.find('{'), c.rfind('}')
    if a >= 0 and b > a:
        try:
            return False, True, json.loads(c[a:b + 1])
        except ValueError:
            pass
    return False, False, None


def preamble_facts(content, strict):
    """Text before the JSON object, duplicated preamble paragraphs, and chat-marker residue in a reply's content."""
    c = content.strip()
    pre = '' if strict else c[:max(c.find('{'), 0)]
    paras = [x.strip() for x in re.split(r'\n\s*\n', pre) if x.strip()]
    return dict(preamble=len(pre), dup_preamble=len(paras) != len(set(paras)),
                residue=any(x in content for x in MARKERS), tool_call='<tool_call>' in content,
                think_marker='<|end_think|>' in content or '<|start_think|>' in content)


def cmd_kind(ks):
    ks = ks.strip()
    if not ks or ks in ('C-c', 'C-d', 'clear', 'q'):
        return 'wait'
    if TEST_RE.search(ks):
        return 'test'
    if RUN_RE.search(ks):
        return 'run'
    if LOOK_RE.search(ks):
        return 'look'
    return 'other'


def load_tok():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(TOKENIZER)


def row_facts(arm, m, r, tok):
    text = tok.decode(r['ids'], skip_special_tokens=False)
    enc = tok.encode(text, add_special_tokens=False)
    same = enc.ids == r['ids']
    starts = [s for s, _ in enc.offsets]
    loss = r['loss']

    def tok_at(ch):
        return bisect.bisect_left(starts, ch)

    turns = []
    for t in r['turns']:
        s, e = t['span']
        te = t.get('think_end', s)
        a, b = tok_at(s), tok_at(e)
        content = text[te:e]
        if content.endswith(EOT):
            content = content[:-len(EOT)]
        strict, lenient, obj = parse_reply(content)
        cmds = []
        if isinstance(obj, dict) and isinstance(obj.get('commands'), list):
            cmds = [c.get('keystrokes', '') if isinstance(c, dict) else str(c) for c in obj['commands']]
        kind = 'student' if t['owner'] == 'student' else ('repair' if t.get('repair') else 'sticky')
        if t['owner'] not in ('student', 'teacher'):
            kind = t['owner']
        turns.append(dict(
            kind=kind, tok0=a, ntok=b - a, trained=sum(loss[a:b]), think_tok=t.get('reasoning_tokens'),
            think_trained=t.get('think_trained'), autofix=bool(t.get('autofix')), strict=strict, lenient=lenient,
            done=bool(isinstance(obj, dict) and obj.get('task_complete') is True),
            cmds=[cmd_kind(c) for c in cmds], keys=[c.strip() for c in cmds][:8],
            giveup=bool(GIVEUP_RE.search(content)), **preamble_facts(content, strict)))
    q = [t for t in turns if t['kind'] in ('repair', 'sticky')]
    sticky = [i for i, t in enumerate(turns) if t['kind'] == 'sticky']
    first_sticky = sticky[0] if sticky else None
    # Qwen's own first done claim (arm-agnostic): the first trained Qwen turn with task_complete true
    qdone = next((i for i, t in enumerate(turns) if t['kind'] in ('sticky', 'repair') and t['done']), None)
    # verification before that claim: the claim turn and the two Qwen turns before it
    win = [t for t in turns[:qdone + 1] if t['kind'] in ('sticky', 'repair')][-3:] if qdone is not None else []
    wk = [k for t in win for k in t['cmds']]
    # the same over the last three executed turns of either owner (a repair turn's claim follows the student's work)
    wa = [k for t in turns[:qdone + 1][-3:] for k in t['cmds']] if qdone is not None else []
    fmt = {}
    for k in ('repair', 'sticky'):
        ts = [t for t in q if t['kind'] == k]
        fmt[k] = dict(n=len(ts), strict=sum(t['strict'] for t in ts), lenient=sum(t['lenient'] for t in ts),
                      preamble=sum(t['preamble'] > 0 for t in ts), dup_preamble=sum(t['dup_preamble'] for t in ts),
                      residue=sum(t['residue'] for t in ts), autofix=sum(t['autofix'] for t in ts),
                      tool_call=sum(t['tool_call'] for t in ts), think_marker=sum(t['think_marker'] for t in ts),
                      trained=sum(t['trained'] for t in ts),
                      trained_residue=sum(t['trained'] for t in ts if t['residue'] or t['dup_preamble']))
    # exact repeats: consecutive Qwen turns with the same non-empty keystrokes
    rep, longest, run_ = 0, 0, 0
    for x, y in zip(q, q[1:]):
        if x['keys'] and x['keys'] == y['keys']:
            rep += 1
            run_ += 1
            longest = max(longest, run_)
        else:
            run_ = 0
    fam = family(m['task'])
    return dict(
        arm=arm, sid=m['sid'], task=m['task'], solver=fam[0], family=fam[1], passed=m['passed'], cause=m['cause'],
        takeover=m.get('takeover'), repairs=m.get('repairs'), ids_ok=same, n_tokens=r['n_tokens'],
        trained=sum(loss),
        trained_by={k: sum(t['trained'] for t in turns if t['kind'] == k) for k in ('student', 'repair', 'sticky')},
        tokens_by={k: sum(t['ntok'] for t in turns if t['kind'] == k) for k in ('student', 'repair', 'sticky')},
        n_turns=len(turns), n_student=sum(t['kind'] == 'student' for t in turns),
        n_repair=sum(t['kind'] == 'repair' for t in turns), n_sticky=len(sticky),
        student_before_takeover=sum(t['kind'] == 'student' for t in turns[:first_sticky]) if sticky else None,
        ctx_at_takeover=turns[first_sticky]['tok0'] if sticky else None,
        student_autofix_trained=sum(1 for t in turns if t['kind'] == 'student' and t['autofix']),
        q_think=[t['think_tok'] or 0 for t in q], q_think_trained=[bool(t['think_trained']) for t in q],
        q_strict=sum(t['strict'] for t in q), q_lenient=sum(t['lenient'] for t in q),
        q_autofix=sum(t['autofix'] for t in q), q_done_turn=qdone, q_turns_to_done=(
            sum(1 for t in turns[:qdone + 1] if t['kind'] in ('sticky', 'repair')) if qdone is not None else None),
        n_done_claims=sum(t['done'] for t in q), verify_window=wk,
        q_cmd_kinds=[k for t in q for k in t['cmds']], q_wait_turns=sum(1 for t in q if t['cmds'] and
                                                                          set(t['cmds']) == {'wait'}),
        q_repeat_pairs=rep, q_longest_repeat=longest, q_giveup=sum(t['giveup'] for t in q), fmt=fmt,
        verify_window_any=wa, claim_by=turns[qdone]['kind'] if qdone is not None else None)


def iter_arm(arm):
    d = os.path.join(RUNS, ARMS[arm])
    man = {}
    for line in open(os.path.join(d, 'final_manifest.jsonl')):
        m = json.loads(line)
        man[m['sid']] = m
    for line in open(os.path.join(d, 'final_rendered_think16k.jsonl')):
        r = json.loads(line)
        yield man[r['sid']], r


def dump(sids, tok, width=1500):
    """Print the decoded rows, one block per assistant turn (owner, trained tokens) with the observation after it."""
    want = set(sids)
    for arm in ARMS:
        for m, r in iter_arm(arm):
            if m['sid'] not in want:
                continue
            text = tok.decode(r['ids'], skip_special_tokens=False)
            print(f"\n######## {arm} {m['sid']} {m['task']} passed={m['passed']} cause={m['cause']} "
                  f"takeover={m.get('takeover')} tokens={r['n_tokens']} trained={sum(r['loss'])}")
            first = r['turns'][0]['span'][0]
            print('--- task (first user turn, tail) ---\n' + text[max(0, first - 2500):first][-2500:])
            for k, t in enumerate(r['turns']):
                s, e = t['span']
                nxt = r['turns'][k + 1]['span'][0] if k + 1 < len(r['turns']) else len(text)
                who = t['owner'] if t['owner'] != 'teacher' else ('QWEN-repair' if t.get('repair') else 'QWEN')
                body = text[s:e]
                if t['owner'] == 'teacher':
                    te = t.get('think_end', s)
                    body = f"[think {t.get('reasoning_tokens')} tok, trained={t.get('think_trained')}] " + \
                        text[s:te][-600:].replace('\n', ' ') + '\n' + text[te:e]
                print(f'--- [{k}] {who} ---\n' + (body[:width] if t['owner'] == 'student' else body[:2 * width]))
                print('    obs: ' + text[e:nxt][:700].replace('\n', '\n    '))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', nargs='*')
    ap.add_argument('--procs', type=int, default=6, help='worker processes (single-threaded each; login-node pid limit)')
    a = ap.parse_args()
    if a.dump:
        dump(a.dump, load_tok())
        return
    import multiprocessing as mp
    with mp.get_context('fork').Pool(a.procs, initializer=_init) as pool:
        for arm in ARMS:
            for line in pool.imap(_facts, ((arm, m, r) for m, r in iter_arm(arm)), chunksize=4):
                sys.stdout.write(line)


_TOK = None


def _init():
    global _TOK
    _TOK = load_tok()


def _facts(job):
    arm, m, r = job
    return json.dumps(row_facts(arm, m, r, _TOK)) + '\n'


if __name__ == '__main__':
    main()
