"""evaldb.py: the PedaGEPA eval database (SQLite) and the per-benchmark eval logs made from it.

Tables
  models      one row per model: its serving config (registry.yaml)
  runs        one row per harbor job: benchmark, model, harness + commit, policy, routing, serve job, the full harbor
              job config.json, where it came from (registry.yaml + the job dir)
  trials      one row per trial: task, trial dir, reward, exception, hard facts (facts.py; Terminus-2 trials get the full
              set, other harnesses the generic subset), per-test results
  judgments   one row per (trial, judge run, checklist item): score, replies, note
  knowledge   one row per knowledge gap a judge listed: fact, kind, reply, blocked, confirmed_by
  task_flags  task-level defects (grader never ran, universal failing tests, a reader's defect verdict)

  python evaldb.py build  [--registry registry.yaml] [--db DB]     # rebuilds every table from disk (idempotent)
  python evaldb.py report [--db DB] --out DIR                        # one markdown eval log per benchmark
  python evaldb.py record <trial_key or substring> [--db DB]        # one trajectory's full eval record (markdown)
Default DB: /e/data1/mmlaion/lee27/experiments/pedagepa/evaldb/pedagepa.sqlite
"""
import argparse, glob, json, os, re, sqlite3, statistics as st, sys
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from facts import facts as t2_facts, tests as t_tests  # noqa: E402
from condense import load_trial  # noqa: E402

DB = '/e/data1/mmlaion/lee27/experiments/pedagepa/evaldb/pedagepa.sqlite'
ITEMS = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P5a', 'P5b', 'P6', 'P7', 'P8', 'P9', 'P10', 'T1', 'S1', 'S2', 'S3', 'S4']

SCHEMA = """
CREATE TABLE models (model_id TEXT PRIMARY KEY, role TEXT, hf TEXT, path TEXT, config TEXT);
CREATE TABLE runs (run_id TEXT PRIMARY KEY, benchmark TEXT, model_id TEXT, harness TEXT, harness_commit TEXT,
  policy TEXT, routing TEXT, serve TEXT, jobs_dir TEXT, run_dir TEXT, collected_by TEXT, job_config TEXT,
  n_trials INTEGER, registry_entry TEXT);
CREATE TABLE trials (trial_key TEXT PRIMARY KEY, run_id TEXT, benchmark TEXT, model_id TEXT, task TEXT, trial_dir TEXT,
  reward REAL, exception TEXT, grader_ran INTEGER, tests_passed INTEGER, tests_failed INTEGER, overflow INTEGER,
  a1_valid_rate REAL, a2_nothing_executed REAL, a3_peak_share REAL, n_replies INTEGER, n_executed INTEGER,
  facts TEXT, facts_error TEXT);
CREATE TABLE judgments (trial_key TEXT, judgment TEXT, judge TEXT, rubric TEXT, item TEXT, score TEXT, replies TEXT,
  note TEXT, source TEXT);
CREATE TABLE knowledge (trial_key TEXT, judgment TEXT, judge TEXT, fact TEXT, kind TEXT, reply TEXT, blocked TEXT,
  confirmed_by TEXT);
CREATE TABLE task_flags (benchmark TEXT, task TEXT, flag TEXT, note TEXT, source TEXT);
CREATE INDEX trials_task ON trials(benchmark, task);
CREATE INDEX judg_trial ON judgments(trial_key);
"""


def sub(s, X):
    return s.replace('${X}', X) if isinstance(s, str) else s


def generic_facts(trial):
    """Harness-independent facts: reward, exception, tests, peak prompt tokens, overflow (for mini-swe-agent trials)."""
    traj, res, reward, exc, att = load_trial(trial)
    peak = 0
    for s in traj.get('steps', []):
        m = s.get('metrics') or {}
        if isinstance(m, str):
            try:
                import ast; m = ast.literal_eval(m)
            except Exception:
                m = {}
        peak = max(peak, (m or {}).get('prompt_tokens') or 0)
    return dict(task=res.get('task_name'), reward=reward, exception=exc, tests=t_tests(att),
                n_replies=sum(1 for s in traj.get('steps', []) if s.get('source') == 'agent'),
                A3=dict(limit=65536, peak_prompt=peak, peak_share=round(peak / 65536, 3),
                        overflow_death='ContextLength' in (exc or '')))


def trial_dirs(jobs_dir):
    return sorted(d.rstrip('/') for d in glob.glob(jobs_dir + '/*/') if os.path.exists(d + 'result.json')
                  or os.path.isdir(d + 'attempts'))


def build(reg_path, db):
    reg = yaml.safe_load(open(reg_path)); X = reg['X']
    os.makedirs(os.path.dirname(db), exist_ok=True)
    tmp = db + '.tmp'
    if os.path.exists(tmp):
        os.remove(tmp)
    con = sqlite3.connect(tmp); con.executescript(SCHEMA)
    for mid, m in reg['models'].items():
        con.execute('INSERT INTO models VALUES (?,?,?,?,?)', (mid, m.get('role'), m.get('hf'), m.get('path'), json.dumps(m)))
    trial_key_by_dir = {}
    for r in reg['runs']:
        jd = sub(r['jobs_dir'], X)
        cfg = json.load(open(jd + '/config.json')) if os.path.exists(jd + '/config.json') else None
        tds = trial_dirs(jd) if os.path.isdir(jd) else []
        con.execute('INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)', (
            r['id'], r['benchmark'], r['model'], r.get('harness'), r.get('harness_commit'), r.get('policy'),
            r.get('routing'), json.dumps(r.get('serve')), jd, r.get('run_dir'), r.get('collected_by'),
            json.dumps(cfg), len(tds), json.dumps(r)))
        refeed = r['refeed_reasoning'] if 'refeed_reasoning' in r else bool(cfg and cfg['agents'][0].get('kwargs', {}).get('interleaved_thinking'))
        limit = (cfg or {}).get('agents', [{}])[0].get('kwargs', {}).get('model_info', {}).get('max_input_tokens') or 65536
        for td in tds:
            key = f"{r['id']}/{os.path.basename(td)}"; err = None
            try:
                f = t2_facts(td, refeed=refeed, limit=limit) if r.get('harness') == 'terminus-2' else generic_facts(td)
                f.pop('trial', None)
            except Exception as e:  # unfinished or unreadable trial
                err = f'{type(e).__name__}: {e}'[:300]
                try:
                    _, res, reward, exc, att = load_trial(td)
                    f = dict(task=res.get('task_name'), reward=reward, exception=exc, tests=t_tests(att))
                except Exception:
                    f = {}
            tests = f.get('tests') or {}
            a3 = f.get('A3') or {}
            con.execute('INSERT INTO trials VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', (
                key, r['id'], r['benchmark'], r['model'], f.get('task'), td, f.get('reward'), f.get('exception'),
                None if tests.get('grader_ran') is None else int(bool(tests.get('grader_ran'))),
                len(tests.get('passed') or []), len(tests.get('failed') or []), int(bool(a3.get('overflow_death'))),
                f.get('A1_valid_rate'), f.get('A2_nothing_executed_token_share'), a3.get('peak_share'),
                f.get('n_replies'), f.get('n_executed'), json.dumps(f), err))
            trial_key_by_dir[os.path.realpath(td)] = key
    for tf in reg.get('task_flags', []):
        src = sub(tf['source'], X)
        if not os.path.exists(src):
            continue
        for task, v in json.load(open(src)).items():
            if v.get('grader_never_ran'):
                con.execute('INSERT INTO task_flags VALUES (?,?,?,?,?)', (tf['benchmark'], task, 'grader_never_ran',
                            json.dumps(v['grader_never_ran'])[:500], src))
            if v.get('universal_failing_tests'):
                con.execute('INSERT INTO task_flags VALUES (?,?,?,?,?)', (tf['benchmark'], task, 'universal_failing_tests',
                            ', '.join(t.get('test', '') for t in v['universal_failing_tests'])[:500], src))
            if v.get('verdict') not in (None, 'ok'):
                con.execute('INSERT INTO task_flags VALUES (?,?,?,?,?)', (tf['benchmark'], task, 'verdict:' + str(v['verdict']),
                            (v.get('note') or '')[:500], src))
    for j in reg.get('judgments', []):
        load_judgment(con, j, X, trial_key_by_dir)
    con.commit(); con.close(); os.replace(tmp, db)
    print(f'built {db}')


def _score(s):
    return (s.get('score') if isinstance(s, dict) else s), (s.get('replies') or s.get('turns') if isinstance(s, dict) else None), \
           (s.get('note') if isinstance(s, dict) else None)


def load_judgment(con, j, X, keymap):
    files = sorted(glob.glob(sub(j['files'], X)))
    if j['kind'] == 'judge_jsonl':
        where = {}
        for b in glob.glob(sub(j['batches'], X)):
            for it in json.load(open(b)):
                where[it['id']] = os.path.realpath(it['trial'])
        for fn in files:
            for line in open(fn):
                if not line.strip():
                    continue
                o = json.loads(line); key = keymap.get(where.get(o.get('id')))
                if not key:
                    continue
                add_one(con, key, j, o, fn)
    elif j['kind'] == 'pair_reads':
        man = {e['task']: e for e in json.load(open(sub(j['manifest'], X)))}
        for fn in files:
            for o in json.load(open(fn)):
                e = man.get(o.get('task'))
                if not e:
                    continue
                for m in ('qwen', 'student'):
                    key = keymap.get(os.path.realpath(e[m]['trial']))
                    if key and isinstance(o.get(m), dict):
                        add_one(con, key, j, o[m], fn)


def add_one(con, key, j, o, fn):
    for item, s in (o.get('scores') or {}).items():
        sc, rp, note = _score(s)
        con.execute('INSERT INTO judgments VALUES (?,?,?,?,?,?,?,?,?)', (key, j['name'], j['judge'], j['rubric'], item,
                    None if sc is None else str(sc), json.dumps(rp), note, fn))
    for g in o.get('knowledge_gaps') or []:
        if isinstance(g, dict):
            con.execute('INSERT INTO knowledge VALUES (?,?,?,?,?,?,?,?)', (key, j['name'], j['judge'], g.get('fact'), g.get('kind'),
                        json.dumps(g.get('reply') or g.get('turn')), str(g.get('blocked')), g.get('confirmed_by') or g.get('confirmed')))
        else:
            con.execute('INSERT INTO knowledge VALUES (?,?,?,?,?,?,?,?)', (key, j['name'], j['judge'], str(g), None, None, None, None))


def pct(x):
    return '–' if x is None else f'{100 * x:.0f} %'


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def report(db, out):
    con = sqlite3.connect(db); con.row_factory = sqlite3.Row; os.makedirs(out, exist_ok=True)
    for (bench,) in con.execute('SELECT DISTINCT benchmark FROM runs ORDER BY benchmark').fetchall():
        L = [f'# Eval log: {bench}', '', f'Generated by `evaldb.py report` from `{db}`. One row per run; every number is '
             'recomputed from the trial directories.', '']
        runs = con.execute('SELECT * FROM runs WHERE benchmark=? ORDER BY model_id, run_id', (bench,)).fetchall()
        L += ['## Runs', '', '| run | model | harness @ commit | policy | routing / serving | trials | scored | pass | overflow | A1 valid (median) | A2 nothing-executed (median) | peak context (median) |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|']
        for r in runs:
            T = con.execute('SELECT * FROM trials WHERE run_id=?', (r['run_id'],)).fetchall()
            sc = [t for t in T if t['reward'] is not None]
            p = sum(1 for t in sc if t['reward'] > 0)
            serve = json.loads(r['serve'] or 'null') or {}
            L.append(f"| {r['run_id']} | {r['model_id']} | {r['harness']} @ {r['harness_commit']} | {r['policy']} | "
                     f"{r['routing'] or ''}; serve job {serve.get('job')} ({serve.get('layout', '')}) | {len(T)} | {len(sc)} | "
                     f"{p} ({pct(p / len(sc) if sc else None)}) | {pct(sum(t['overflow'] for t in sc) / len(sc) if sc else None)} | "
                     f"{pct(med([t['a1_valid_rate'] for t in sc]))} | {pct(med([t['a2_nothing_executed'] for t in sc]))} | "
                     f"{pct(med([t['a3_peak_share'] for t in sc]))} |")
        # paired pass table between models on shared tasks (first run per model)
        by_model = {}
        for r in runs:
            by_model.setdefault(r['model_id'], r['run_id'])
        if len(by_model) >= 2:
            (ma, ra), (mb, rb) = list(by_model.items())[:2]
            A = {t['task']: t['reward'] > 0 for t in con.execute('SELECT task, reward FROM trials WHERE run_id=? AND reward IS NOT NULL', (ra,))}
            B = {t['task']: t['reward'] > 0 for t in con.execute('SELECT task, reward FROM trials WHERE run_id=? AND reward IS NOT NULL', (rb,))}
            c = set(A) & set(B)
            L += ['', f'## Same tasks: {ra} vs {rb} ({len(c)} tasks scored in both)', '',
                  f'- both pass {sum(A[t] and B[t] for t in c)}, only {ma} {sum(A[t] and not B[t] for t in c)}, '
                  f'only {mb} {sum(B[t] and not A[t] for t in c)}, neither {sum(not A[t] and not B[t] for t in c)}']
        # judged items
        J = con.execute('SELECT j.judgment, j.judge, j.rubric, t.model_id, j.item, j.score FROM judgments j JOIN trials t '
                        'ON j.trial_key=t.trial_key WHERE t.benchmark=?', (bench,)).fetchall()
        if J:
            L += ['', '## Judged checklist (mean of numeric scores 0-2; n in brackets; NA/LD counted separately)', '']
            for jn in sorted({x['judgment'] for x in J}):
                rows = [x for x in J if x['judgment'] == jn]
                models = sorted({x['model_id'] for x in rows})
                items = [i for i in ITEMS if any(x['item'] == i for x in rows)]
                L += [f'**{jn}** (judge {rows[0]["judge"]}, rubric {rows[0]["rubric"]})', '',
                      '| item | ' + ' | '.join(models) + ' |', '|---|' + '---|' * len(models)]
                for i in items:
                    cells = []
                    for m in models:
                        v = [float(x['score']) for x in rows if x['model_id'] == m and x['item'] == i and re.fullmatch(r'[0-2](\.0)?', str(x['score']))]
                        o = sum(1 for x in rows if x['model_id'] == m and x['item'] == i and not re.fullmatch(r'[0-2](\.0)?', str(x['score'])))
                        cells.append(f'{st.mean(v):.2f} ({len(v)}; {o} NA/LD)' if v else f'– ({o} NA/LD)')
                    L.append(f'| {i} | ' + ' | '.join(cells) + ' |')
                L.append('')
        K = con.execute('SELECT k.*, t.model_id, t.task FROM knowledge k JOIN trials t ON k.trial_key=t.trial_key WHERE t.benchmark=? '
                        'ORDER BY t.model_id, k.kind', (bench,)).fetchall()
        if K:
            L += ['## Knowledge ledger (unmerged; one row per gap a judge listed)', '', '| model | kind | blocked | confirmed by | fact | task | judgment |', '|---|---|---|---|---|---|---|']
            for k in K:
                L.append(f"| {k['model_id']} | {k['kind']} | {k['blocked']} | {k['confirmed_by']} | {(k['fact'] or '').replace('|', '/')[:220]} | {(k['task'] or '')[-28:]} | {k['judgment']} |")
            L.append('')
        F = con.execute('SELECT flag, COUNT(*) n FROM task_flags WHERE benchmark=? GROUP BY flag', (bench,)).fetchall()
        if F:
            L += ['## Task flags', ''] + [f"- {f['flag']}: {f['n']} tasks" for f in F] + ['']
        open(os.path.join(out, f'{bench}.md'), 'w').write('\n'.join(L) + '\n')
        print('wrote', os.path.join(out, f'{bench}.md'))


NAMES = {'P0': 'harness feedback', 'P1': 'read the contract', 'P2': "look it up", 'P3': 'build incrementally',
         'P4': 'checkable steps', 'P5': 'manage context', 'P5a': 'bounds tool output', 'P5b': 'short own replies', 'P6': 'grounded state', 'P7': 'error recovery',
         'P8': 'progress control', 'P9': 'verify before done', 'P10': 'honest completion', 'T1': 'long jobs/services',
         'S1': 'repro fails first', 'S2': 'fix at the origin', 'S3': 'minimal checked edit', 'S4': "repo's own tests"}
KIND = {'K1': 'tool/package', 'K2': 'format/internals', 'K3': 'where things live', 'K4': 'domain method', 'K5': 'repo fact'}


def _short(x, n):
    x = ' '.join((x or '').split())
    return x if len(x) <= n else x[:n - 1].rstrip() + '…'


def record_compact(con, t, r, judgment):
    f = json.loads(t['facts'] or '{}'); a3 = f.get('A3') or {}; comp = a3.get('composition') or {}
    top = sorted(((v, k) for k, v in comp.items() if v), reverse=True)[:2]
    L = [f"task: {t['task']}   model: {r['model_id']}   passed: {'yes' if (t['reward'] or 0) > 0 else 'no'}",
         f"from: {r['run_id']} ({r['benchmark']}) · {r['harness']} @ {(r['harness_commit'] or '').split('@')[-1].strip()} · "
         f"trial {os.path.basename(t['trial_dir'])}",
         'policy (0 = absent, 1 = partial, 2 = present, NA = no chance to show it, LD = loop died first;',
         '        T6 = the agent\'s 6th reply, the turn the evidence points at)']
    J = {j['item']: j for j in con.execute('SELECT * FROM judgments WHERE trial_key=? AND judgment=?', (t['trial_key'], judgment))}
    for i in ITEMS:
        if i not in J:
            continue
        j = J[i]; rp = json.loads(j['replies'] or 'null') or []
        sc = str(j['score']).replace('.0', '')
        where = (', '.join(f'T{x}' for x in rp[:4]) + (', …' if len(rp) > 4 else '') + ': ') if rp else ''
        note = _short(j['note'], 78)
        L.append(f"  {NAMES.get(i, i):20} {sc:3}  " + (f"({note})" if sc in ('NA', 'LD') else where + note))
    L += [f"  {'valid action (auto)':20} {f.get('A1_valid_rate') or 0:.2f}",
          f"  {'wasted budget (auto)':20} {f.get('A2_nothing_executed_token_share') or 0:.2f} of generated tokens executed nothing",
          f"  {'context (auto)':20} peak {pct(a3.get('peak_share'))} of {a3.get('limit')}{', overflowed' if a3.get('overflow_death') else ''}; "
          + ', '.join(f"{k.replace('_', ' ')} {pct(v)}" for v, k in top),
          'knowledge gaps']
    K = con.execute('SELECT * FROM knowledge WHERE trial_key=? AND judgment=?', (t['trial_key'], judgment)).fetchall()
    for k in K:
        rp = json.loads(k['reply']) if k['reply'] else None
        rp = rp[0] if isinstance(rp, list) and rp else rp
        L += [f'  - "{_short(k["fact"], 150)}"',
              f"    kind: {KIND.get((k['kind'] or '')[:2], k['kind'])}   turn: T{rp}   blocked the task: {k['blocked']}"]
    if not K:
        L.append('  - none')
    return '\n'.join(L)


def record(db, q, judgment=None, compact=False):
    con = sqlite3.connect(db); con.row_factory = sqlite3.Row
    t = con.execute('SELECT * FROM trials WHERE trial_key=?', (q,)).fetchone() or \
        con.execute('SELECT * FROM trials WHERE trial_key LIKE ? OR task LIKE ? ORDER BY trial_key LIMIT 1', (f'%{q}%', f'%{q}%')).fetchone()
    if not t:
        return f'no trial matches {q!r}'
    r = con.execute('SELECT * FROM runs WHERE run_id=?', (t['run_id'],)).fetchone()
    if compact:
        jn = judgment or (con.execute('SELECT judgment FROM judgments WHERE trial_key=? LIMIT 1', (t['trial_key'],)).fetchone() or [None])[0]
        return record_compact(con, t, r, jn)
    m = json.loads(con.execute('SELECT config FROM models WHERE model_id=?', (r['model_id'],)).fetchone()['config'])
    f = json.loads(t['facts'] or '{}'); a3 = f.get('A3') or {}; tests = f.get('tests') or {}
    serve = json.loads(r['serve'] or 'null') or {}
    L = [f"### Eval record: {t['task']} · {r['model_id']}", '',
         '**Where it came from**', '',
         f"- run `{r['run_id']}` ({r['benchmark']}), trial `{os.path.basename(t['trial_dir'])}`, collected by {r['collected_by']}",
         f"- harness {r['harness']} @ {r['harness_commit']}; policy: {r['policy']}",
         f"- routing: {r['routing'] or 'direct'}; serve job {serve.get('job')} {('(' + serve['layout'] + ')') if serve.get('layout') else ''}",
         f"- model: {m.get('hf')} ({m.get('arch')}); {m.get('engine')}; spec decode {m.get('spec_decode')}; context {m.get('max_model_len')}; "
         f"sampling {m.get('sampling')}; thinking {m.get('thinking')}", '',
         '**Hard facts (from the logs)**', '',
         f"- reward {t['reward']}; exception {t['exception'] or 'none'}; grader ran {tests.get('grader_ran')}; tests passed "
         f"{len(tests.get('passed') or [])}, failed {len(tests.get('failed') or [])}",
         f"- turns {f.get('n_replies')}, of which executed {f.get('n_executed')}; claimed done at turns {f.get('claims')}; last file write at turn "
         f"{f.get('last_edit_reply')}; commands between it and the claim {f.get('commands_between_last_edit_and_first_claim', f.get('executed_replies_between_last_edit_and_first_claim'))}",
         f"- A1 replies accepted {pct(f.get('A1_valid_rate'))}; A2 tokens in replies that executed nothing {pct(f.get('A2_nothing_executed_token_share'))}",
         f"- A3 context: peak {a3.get('peak_prompt')} of {a3.get('limit')} ({pct(a3.get('peak_share'))}), overflow {a3.get('overflow_death')}; "
         f"filled by " + ', '.join(f"{k.replace('_', ' ')} {pct(v)}" for k, v in (a3.get('composition') or {}).items() if v) , '']
    J = con.execute('SELECT * FROM judgments WHERE trial_key=?' + (' AND judgment=?' if judgment else ''),
                    (t['trial_key'],) + ((judgment,) if judgment else ())).fetchall()
    for jn in sorted({j['judgment'] for j in J}):
        rows = {j['item']: j for j in J if j['judgment'] == jn}
        j0 = rows[next(iter(rows))]
        L += [f"**Policy scores** (judged by {j0['judge'].capitalize()}, rubric {j0['rubric']}; T6 = the agent's 6th reply, "
              "the turn the evidence points at)", '', '| behaviour | score | turns | evidence |', '|---|---|---|---|']
        for i in ITEMS:
            if i in rows:
                j = rows[i]
                L.append(f"| {NAMES.get(i, i)} ({i}) | {j['score']} | {', '.join('T' + str(x) for x in (json.loads(j['replies'] or 'null') or []))} | {(j['note'] or '').replace('|', '/')} |")
        K = con.execute('SELECT * FROM knowledge WHERE trial_key=? AND judgment=?', (t['trial_key'], jn)).fetchall()
        L += ['', '**Missing facts**', ''] + ([f"- {k['fact']} ({KIND.get((k['kind'] or '')[:2], k['kind'])}; turn T{(json.loads(k['reply']) if k['reply'] else '?')}; blocked the task: {k['blocked']}; confirmed by {k['confirmed_by']})" for k in K] or ['- none listed'])
        L.append('')
    return '\n'.join(L)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['build', 'report', 'record'])
    ap.add_argument('query', nargs='?'); ap.add_argument('--judgment', default=None); ap.add_argument('--compact', action='store_true')
    ap.add_argument('--registry', default=os.path.join(HERE, 'registry.yaml')); ap.add_argument('--db', default=DB)
    ap.add_argument('--out', default='/e/data1/mmlaion/lee27/experiments/pedagepa/evaldb/logs')
    a = ap.parse_args()
    if a.cmd == 'build':
        build(a.registry, a.db)
    elif a.cmd == 'report':
        report(a.db, a.out)
    else:
        print(record(a.db, a.query, a.judgment, a.compact))
