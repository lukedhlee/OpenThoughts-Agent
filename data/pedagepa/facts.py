"""facts.py: the hard facts of one harbor Terminus-2 trial, for the judge and for anchor checks (no judgement needed).

  reward, exception, per-test results (verifier/ctrf.json, else pytest lines in test-stdout.txt), whether the grader ran
  claims: agent reply numbers with task_complete=true (executed replies only)
  A1 valid_rate, A2 (token proxy) share of generated tokens in replies that executed nothing
  A3 context use: limit, peak prompt tokens and share, overflow death, first reply past half the limit, composition of
     the final context by source (terminal output, executed replies, rejected replies + their error messages, reasoning
     re-fed by the harness, instruction) and the largest single output's share
  last_edit_reply / commands_between_last_edit_and_first_claim: non-empty commands after the last file-writing command,
     up to and including the first claim reply (command level, so a check in the same reply as the edit counts)
  error_before_claim: whether the terminal output of the reply before the first claim contains an error signature

  python facts.py <trial dir> [--refeed-reasoning] [--limit 65536]
"""
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from condense import load_trial, parse_turns, features, _new_output

# a command that writes a file: a redirect to a real path (not a numbered or &-fd redirect, not /dev/null), sed -i,
# tee to a file, patch / git apply, cp / mv, or a python open(..., 'w')
WRITE_RE = re.compile(r"((?<![0-9&])>>?\s*(?!/dev/null)(?!&)[\w./~$-]+|\bsed\s+-i|\btee\s+(?!/dev/null)[\w./~-]+|\bpatch\b|"
                      r"\bgit\s+apply\b|\bcp\s|\bmv\s|open\([^)]*['\"][wa]b?['\"])")
# v1.3.2 (2026-09-29): the anchor works on shell lines, not whole keystroke blocks. Heredoc bodies are not shell lines
# (`python3 - <<EOF ... assert x > 0` is a check, not a redirect); a body counts as a write only when it writes a file.
# Quoted text is dropped before the redirect test (`python3 -c "print(a > b)"`, `awk '$1 > 5'`).
HEREDOC_RE = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?")
BODY_WRITE_RE = re.compile(r"open\([^)]*['\"][wa]b?\+?['\"]|\.to_csv\(|\.to_json\(|\.to_parquet\(|\.savefig\(|\.write_text\(|"
                           r"\.write_bytes\(|json\.dump\(|np\.save|\.save\(|shutil\.(copy|move)")
QUOTED_RE = re.compile(r"'[^']*'|\"(?:[^\"\\]|\\.)*\"")


def shell_lines(ks):
    """(line, writes) per shell line of one keystroke block; heredoc bodies fold into their opening line."""
    out, lines, k = [], ks.split('\n'), 0
    while k < len(lines):
        line = lines[k]; k += 1
        if not line.strip():
            continue
        m = HEREDOC_RE.search(line); body = []
        if m:
            while k < len(lines) and lines[k].strip() != m.group(1):
                body.append(lines[k]); k += 1
            k += 1
        # a quoted string left open (multi-line `python3 -c "..."`) folds the following lines in until it closes
        while k < len(lines) and (line.count('"') - line.count('\\"')) % 2 == 1:
            line += '\n' + lines[k]; k += 1
        bare = QUOTED_RE.sub("''", line)
        segs = [s for s in re.split(r"&&|\|\||;", bare) if s.strip()] or [bare]
        # a write hidden in a heredoc body or a quoted python -c program is credited to the line's first command
        hidden = bool((body and BODY_WRITE_RE.search('\n'.join(body))) or re.search(r"open\([^)]*['\"][wa]b?\+?['\"]", line))
        for j, s in enumerate(segs):
            out.append((s.strip(), bool(WRITE_RE.search(s)) or (hidden and j == 0)))
    return out


ERR_RE = re.compile(r"(Traceback \(most recent call last\)|\bError\b|\berror:|FAILED|No such file|command not found|Segmentation fault|"
                    r"exit code [1-9]|AssertionError|cannot |failed)", re.I)


def tests(att):
    out = {'grader_ran': None, 'passed': [], 'failed': []}
    cj = os.path.join(att, 'verifier', 'ctrf.json')
    if os.path.exists(cj):
        try:
            d = json.load(open(cj))
            for t in d.get('results', {}).get('tests', []):
                (out['passed'] if t.get('status') == 'passed' else out['failed']).append(t.get('name'))
            out['grader_ran'] = True
            return out
        except Exception:
            pass
    ts = os.path.join(att, 'verifier', 'test-stdout.txt')
    if os.path.exists(ts):
        txt = open(ts, errors='replace').read()
        for m in re.finditer(r'^(\S+::\S+)\s+(PASSED|FAILED|ERROR)', txt, re.M):
            (out['passed'] if m.group(2) == 'PASSED' else out['failed']).append(m.group(1))
        out['grader_ran'] = (bool(re.search(r'=+ .*(passed|failed|error).* in [\d.]+s', txt)) or bool(out['passed'] or out['failed'])
                             or 'SWEBench results starts here' in txt or bool(re.search(r'^Ran \d+ tests? in', txt, re.M)))
    rj = os.path.join(att, 'verifier', 'report.json')
    if os.path.exists(rj):
        out['grader_ran'] = True
        try:
            rep = json.load(open(rj)); rep = next(iter(rep.values())) if len(rep) == 1 else rep
            ts_ = rep.get('tests_status') or {}
            for k in ('FAIL_TO_PASS', 'PASS_TO_PASS'):
                out.setdefault('swe', {})[k] = {kk: len(v) for kk, v in (ts_.get(k) or {}).items()}
        except Exception:
            pass
    return out


def facts(trial, refeed=False, limit=65536):
    traj, res, reward, exc, att = load_trial(trial)
    instr, turns = parse_turns(traj)
    f = features(turns)
    gen = sum(t['completion_tokens'] for t in turns) or 1
    a2 = sum(t['completion_tokens'] for t in turns if t['rejected']) / gen
    peak = max((t['prompt_tokens'] for t in turns), default=0)
    half = next((i + 1 for i, t in enumerate(turns) if t['prompt_tokens'] > limit / 2), None)
    comp = dict(terminal_output=sum(len(t['obs']) for t in turns if not t['rejected']),
                executed_replies=sum(len(t['raw']) for t in turns if not t['rejected']),
                rejected_replies_and_errors=sum(len(t['raw']) + len(t['obs']) for t in turns if t['rejected']),
                refed_reasoning=sum(len(t['think']) for t in turns) if refeed else 0, instruction=len(instr))
    tot = sum(comp.values()) or 1
    claims = [i + 1 for i, t in enumerate(turns) if t['task_complete'] and not t['rejected']]
    # command-level: the last file-writing command before the first claim, and how many commands ran after it up to and
    # including the claim reply (a test in the same reply as the edit counts)
    first_claim = claims[0] if claims else None
    seq = [(i + 1, line, w) for i, t in enumerate(turns) if not t['rejected']
           for c in t['cmds'] if isinstance(c, dict) for line, w in shell_lines(str(c.get('keystrokes', '')))]
    upto = [x for x in seq if first_claim is None or x[0] <= first_claim]
    wi = max((k for k, (_, _, w) in enumerate(upto) if w), default=None)
    last_edit = upto[wi][0] if wi is not None else None
    checks_after = (sum(1 for _ in upto[wi + 1:]) if (first_claim and wi is not None) else 0)
    err_before = bool(first_claim and first_claim >= 2 and ERR_RE.search(_new_output(turns[first_claim - 2]['obs'])[-3000:]))
    return dict(trial=trial, task=res.get('task_name'), reward=reward, exception=exc, tests=tests(att),
                n_replies=f['n_replies'], n_executed=f['n_executed'], claims=claims,
                A1_valid_rate=f['valid_rate'], A2_nothing_executed_token_share=round(a2, 3),
                A3=dict(limit=limit, peak_prompt=peak, peak_share=round(peak / limit, 3),
                        overflow_death='ContextLength' in (exc or ''), first_reply_past_half=half,
                        composition={k: round(v / tot, 3) for k, v in comp.items()},
                        largest_output_share=round(max((len(t['obs']) for t in turns), default=0) / tot, 3),
                        # the same output as a share of the WINDOW (context share x peak prompt / limit)
                        largest_output_window_share=round(max((len(t['obs']) for t in turns), default=0) / tot * peak / limit, 3)),
                runaway_replies=f['runaway_replies'], loop_fires=f['loop_fires'], wait_fires=f['wait_fires'],
                last_edit_reply=last_edit, commands_between_last_edit_and_first_claim=checks_after,
                error_signature_right_before_first_claim=err_before)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('trial'); ap.add_argument('--refeed-reasoning', action='store_true')
    ap.add_argument('--limit', type=int, default=65536); a = ap.parse_args()
    print(json.dumps(facts(a.trial, a.refeed_reasoning, a.limit), indent=1))
