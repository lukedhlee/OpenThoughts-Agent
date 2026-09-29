"""condense.py: one harbor Terminus-2 trial -> a compact text view for the behaviour judge, plus deterministic features.

The view shows the task instruction and every turn's analysis, plan, keystrokes and new terminal output (truncated to fit
a character budget). Replies the parser rejected executed nothing; consecutive ones are collapsed into one line with
their count. The reward is never in the view (the judge must not see the outcome).

Deterministic features (behaviours B1 / B2 / B7 of the PedaGEPA plan):
  valid_rate          share of agent replies the harness accepted (not followed by "Previous response had parsing errors")
  runaway_replies     replies with >= 8,192 completion tokens
  think_loop_replies  replies whose reasoning repeats one 200-char span 3+ times
  loop_fires          executed turns whose normalised keystrokes equal those of >= 2 other turns among the last 6 executed
  wait_fires          runs of 3 executed passive waits (no keystrokes / sleep / wait / Enter) with no new output

  python condense.py <trial dir> [--max-chars 60000]         # prints the view
"""
import argparse, ast, json, os, re, sys

PARSE_ERR = 'Previous response had parsing errors'
THINK_RE = re.compile(r'<\|start_think\|>(.*?)(<\|end_think\|>|$)', re.S)


def _obs_text(obs):
    if obs is None:
        return ''
    if isinstance(obs, dict):
        d = obs
    else:
        try:
            d = ast.literal_eval(obs)
        except Exception:
            return str(obs)
    try:
        return '\n'.join(str(r.get('content', '')) for r in d.get('results', []))
    except Exception:
        return str(obs)


def _metrics(m):
    if isinstance(m, dict):
        return m
    try:
        return ast.literal_eval(m) if m else {}
    except Exception:
        return {}


def _json_obj(text):
    """First JSON object in text that has 'analysis' or 'commands' (tolerant: skips wrappers, think spans)."""
    dec = json.JSONDecoder()
    for m in re.finditer(r'\{', text):
        try:
            obj, _ = dec.raw_decode(text[m.start():])
        except Exception:
            continue
        if isinstance(obj, dict) and ('analysis' in obj or 'commands' in obj or 'task_complete' in obj):
            return obj
    return None


def _clip(s, n, tail=0):
    s = s or ''
    if len(s) <= n + tail + 20:
        return s
    if tail:
        return s[:n] + f'\n[... {len(s) - n - tail} chars cut ...]\n' + s[-tail:]
    return s[:n] + f' [... {len(s) - n} chars cut]'


def _new_output(txt):
    """Strip harbor's 'New Terminal Output:' header; keep the rest."""
    t = txt.strip('\n')
    for h in ('New Terminal Output:', 'Current Terminal Screen:'):
        if t.startswith(h):
            t = t[len(h):].lstrip('\n')
    return t


def _norm_keys(cmds):
    return ' '.join(' '.join(str(c.get('keystrokes', '')).split()) for c in cmds if isinstance(c, dict))


def _is_wait(cmds):
    ks = [str(c.get('keystrokes', '')).strip() for c in cmds if isinstance(c, dict)]
    return all(k == '' or re.fullmatch(r'(sleep\s+[\d.]+|wait)?', k) for k in ks)


def load_trial(trial_dir):
    att = os.path.join(trial_dir, 'attempts', '000')
    if not os.path.isdir(att):
        att = trial_dir
    tj = os.path.join(att, 'agent', 'trajectory.json')
    traj = json.load(open(tj))
    res = {}
    for p in (os.path.join(att, 'result.json'), os.path.join(trial_dir, 'result.json')):
        if os.path.exists(p):
            try:
                res = json.load(open(p)); break
            except Exception:
                pass
    reward = None
    rp = os.path.join(att, 'verifier', 'reward.txt')
    if os.path.exists(rp):
        try:
            reward = float(open(rp).read().strip())
        except Exception:
            pass
    exc = None
    ep = os.path.join(att, 'exception.txt')
    if os.path.exists(ep):
        exc = open(ep).read().strip().splitlines()[-1][:200] if open(ep).read().strip() else None
    return traj, res, reward, exc, att


def parse_turns(traj):
    steps = traj['steps']
    instr = ''
    turns = []
    for s in steps:
        if s.get('source') == 'user' and not instr:
            msg = s.get('message', '')
            i = msg.find('Task Description:')
            instr = msg[i + len('Task Description:'):] if i >= 0 else msg
            j = instr.find('Current terminal state:')
            instr = instr[:j] if j >= 0 else instr
            continue
        if s.get('source') != 'agent':
            continue
        msg = s.get('message') or ''
        reasoning = s.get('reasoning_content') or ''
        think = ''.join(m.group(1) for m in THINK_RE.finditer(msg))
        body = THINK_RE.sub('', msg)
        obs = _obs_text(s.get('observation'))
        met = _metrics(s.get('metrics'))
        rejected = obs.lstrip().startswith(PARSE_ERR) or ('parsing errors' in obs[:200] and 'ERROR' in obs[:300])
        obj = _json_obj(body)
        cmds = obj.get('commands', []) if isinstance(obj, dict) and isinstance(obj.get('commands'), list) else []
        tc_done = False
        tcs = s.get('tool_calls')
        if isinstance(tcs, str):
            try:
                tcs = ast.literal_eval(tcs)
            except Exception:
                tcs = None
        if isinstance(tcs, list) and tcs:   # harbor's native tool-call path (v0.1 pin): bash_command / mark_task_complete
            for tc in tcs:
                fn = (tc or {}).get('function_name') or ((tc or {}).get('function') or {}).get('name')
                args = (tc or {}).get('arguments') or ((tc or {}).get('function') or {}).get('arguments') or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                if fn in ('bash_command', 'bash') and isinstance(args, dict):
                    cmds.append(dict(keystrokes=args.get('keystrokes', ''), duration=args.get('duration')))
                elif fn == 'mark_task_complete':
                    tc_done = True
            rejected = rejected and not cmds and not tc_done
        turns.append(dict(
            model=s.get('model_name'), rejected=rejected,
            analysis=str(obj.get('analysis', '')) if obj else ('' if not tcs else body[:2000]),
            plan=str(obj.get('plan', '')) if obj else '',
            cmds=cmds, task_complete=(bool(obj.get('task_complete')) if obj else False) or tc_done,
            think=(reasoning or think), completion_tokens=met.get('completion_tokens') or 0,
            prompt_tokens=met.get('prompt_tokens') or 0, obs=obs, raw=body))
    return instr.strip(), turns


def features(turns):
    n = len(turns)
    valid = sum(1 for t in turns if not t['rejected'])
    runaway = sum(1 for t in turns if (t['completion_tokens'] or 0) >= 8192)
    tloop = 0
    for t in turns:
        th = t['think']
        if len(th) >= 600:
            spans = {}
            for i in range(0, len(th) - 200, 100):
                k = th[i:i + 200]
                spans[k] = spans.get(k, 0) + 1
            if spans and max(spans.values()) >= 3:
                tloop += 1
    ex = [t for t in turns if not t['rejected']]
    loop_fires = 0
    for i in range(len(ex)):
        k = _norm_keys(ex[i]['cmds'])
        if not k or _is_wait(ex[i]['cmds']):
            continue
        win = ex[max(0, i - 5):i]
        if sum(1 for w in win if _norm_keys(w['cmds']) == k) >= 2:
            loop_fires += 1
    wait_fires, run, prev = 0, 0, None
    for t in ex:
        out = _new_output(t['obs']).strip()
        if _is_wait(t['cmds']) and (not out or out == prev):
            run += 1
            if run >= 3:
                wait_fires += 1
        else:
            run = 0
        prev = out
    gen = sum(t['completion_tokens'] for t in turns)
    gen_runaway = sum(t['completion_tokens'] for t in turns if t['completion_tokens'] >= 8192)
    claims = sum(1 for t in ex if t['task_complete'])
    return dict(n_replies=n, n_executed=valid, valid_rate=(valid / n) if n else None, runaway_replies=runaway,
                runaway_token_share=(gen_runaway / gen) if gen else 0.0, think_loop_replies=tloop,
                loop_fires=loop_fires, wait_fires=wait_fires, gen_tokens=gen, claims=claims)


def render(instr, turns, max_chars=60000):
    ex_n = sum(1 for t in turns if not t['rejected']) or 1
    per = max(500, min(4000, int((max_chars - min(len(instr), 6000)) / ex_n)))
    a_n, p_n, k_n, o_h, o_t = int(per * .18), int(per * .10), int(per * .25), int(per * .27), int(per * .20)
    out = ['## TASK INSTRUCTION', _clip(instr, 6000), '', '## TRAJECTORY (T = agent reply number; replies the parser '
           'rejected executed nothing and are collapsed)']
    i = 0
    while i < len(turns):
        t = turns[i]
        if t['rejected']:
            j = i
            while j < len(turns) and turns[j]['rejected']:
                j += 1
            k = j - i
            ex = turns[i]['raw'][:300].replace('\n', ' ')
            claimed = any(tt['task_complete'] for tt in turns[i:j])
            out.append(f'T{i + 1}{"-T%d" % j if k > 1 else ""}: {k} repl{"ies" if k > 1 else "y"} REJECTED by the parser '
                       f'(nothing executed{"; one of them said task_complete" if claimed else ""}). First one began: '
                       f'{ex!r}')
            i = j
            continue
        cmds = '\n'.join(f'  $ {c.get("keystrokes", "")!r}' for c in t['cmds'] if isinstance(c, dict)) or '  (no commands)'
        tok = f' [{t["completion_tokens"]} tokens generated]' if t['completion_tokens'] >= 4000 else ''
        out.append(f'--- T{i + 1}{tok}{"  task_complete=true" if t["task_complete"] else ""}')
        if t['analysis']:
            out.append('analysis: ' + _clip(t['analysis'], a_n))
        if t['plan']:
            out.append('plan: ' + _clip(t['plan'], p_n))
        out.append('commands:\n' + _clip(cmds, k_n, tail=200))
        out.append('output:\n' + _clip(_new_output(t['obs']), o_h, tail=o_t))
        i += 1
    return '\n'.join(out)


def condense(trial_dir, max_chars=60000):
    traj, res, reward, exc, att = load_trial(trial_dir)
    instr, turns = parse_turns(traj)
    f = features(turns)
    return dict(view=render(instr, turns, max_chars), features=f, reward=reward, exception=exc, attempt_dir=att,
                task=(res.get('task_name') or os.path.basename(trial_dir.rstrip('/')).split('__')[0]))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('trial')
    ap.add_argument('--max-chars', type=int, default=60000)
    a = ap.parse_args()
    r = condense(a.trial, a.max_chars)
    print(json.dumps(r['features']), r['reward'], r['exception'], file=sys.stderr)
    print(r['view'])
