"""m1_table.py: the stage-2 (M1) same-task table from the eval database.

Per benchmark: per checklist item, teacher and student mean score (0-2), share of 2s, n; paired teacher-minus-student
difference over tasks with a 10,000-draw bootstrap CI; the automatic A1/A2 on every scored trial; the stage-2 rule; the
"both lack" list; and the student's failure-mode table (how often each item scores 0, how often it is the named failure
cause, and where in the episode it happens).
"""
import json, re, sqlite3, random, statistics as st, sys, glob, collections
DB = '/e/data1/mmlaion/lee27/experiments/pedagepa/evaldb/pedagepa.sqlite'
J = {'calibforge-heldout300': 'cf40_opusB_v1.2', 'tb2.1': 'r2_opus1_v1.3', 'swe100': 'r2_opus1_v1.3'}
ITEMS = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P5a', 'P5b', 'P6', 'P7', 'P8', 'P9', 'P10', 'T1', 'S1', 'S2', 'S3', 'S4']
NUM = re.compile(r'^[0-2](\.0)?$')
con = sqlite3.connect(DB); con.row_factory = sqlite3.Row
rnd = random.Random(0)


def boot(diffs, B=10000):
    if len(diffs) < 3:
        return None, None
    m = [st.mean(rnd.choices(diffs, k=len(diffs))) for _ in range(B)]
    m.sort(); return m[int(.025 * B)], m[int(.975 * B)]


out = {}
for bench, jn in J.items():
    rows = con.execute('SELECT t.task, t.model_id, t.trial_key, t.reward, j.item, j.score, j.replies FROM judgments j JOIN trials t '
                       'ON j.trial_key=t.trial_key WHERE t.benchmark=? AND j.judgment=?', (bench, jn)).fetchall()
    S = collections.defaultdict(dict)          # (task, model) -> item -> score
    for r in rows:
        S[(r['task'], r['model_id'])][r['item']] = r['score']
    tasks = sorted({t for (t, m) in S if (t, 'qwen38-27b') in S and (t, 'snowball-0921') in S})
    res = {'n_tasks': len(tasks), 'items': {}}
    for it in ITEMS:
        tv, sv, d = [], [], []
        for t in tasks:
            a, b = S[(t, 'qwen38-27b')].get(it), S[(t, 'snowball-0921')].get(it)
            if a is not None and NUM.match(str(a)): tv.append(float(a))
            if b is not None and NUM.match(str(b)): sv.append(float(b))
            if a is not None and b is not None and NUM.match(str(a)) and NUM.match(str(b)): d.append(float(a) - float(b))
        if not tv and not sv:
            continue
        lo, hi = boot(d)
        res['items'][it] = dict(teacher=round(st.mean(tv), 2) if tv else None, teacher_n=len(tv),
                                teacher_share2=round(sum(x == 2 for x in tv) / len(tv), 2) if tv else None,
                                student=round(st.mean(sv), 2) if sv else None, student_n=len(sv),
                                diff=round(st.mean(d), 2) if d else None, n_paired=len(d),
                                ci=[round(lo, 2), round(hi, 2)] if lo is not None else None,
                                student_ld=sum(1 for t in tasks if str(S[(t, 'snowball-0921')].get(it)) == 'LD'),
                                teacher_ld=sum(1 for t in tasks if str(S[(t, 'qwen38-27b')].get(it)) == 'LD'))
    # automatic A1 / A2 on every scored trial of the benchmark's runs, paired by task
    A = collections.defaultdict(dict)
    for r in con.execute('SELECT task, model_id, a1_valid_rate, a2_nothing_executed FROM trials WHERE benchmark=? AND reward IS NOT NULL', (bench,)):
        A[r['task']].setdefault(r['model_id'], []).append((r['a1_valid_rate'], r['a2_nothing_executed']))
    for k, idx in (('A1_valid', 0), ('A2_nothing_executed', 1)):
        d = []; tv = []; sv = []
        for t, mm in A.items():
            if 'qwen38-27b' in mm and 'snowball-0921' in mm:
                a = [x[idx] for x in mm['qwen38-27b'] if x[idx] is not None]; b = [x[idx] for x in mm['snowball-0921'] if x[idx] is not None]
                if a and b:
                    tv.append(st.mean(a)); sv.append(st.mean(b)); d.append(st.mean(a) - st.mean(b))
        lo, hi = boot(d)
        res['items'][k] = dict(teacher=round(st.mean(tv), 2), student=round(st.mean(sv), 2), diff=round(st.mean(d), 2),
                               n_paired=len(d), ci=[round(lo, 2), round(hi, 2)])
    # stage-2 rule
    def clear(it, thr):
        x = res['items'].get(it); return bool(x and x.get('diff') is not None and x['diff'] >= thr and x['ci'] and x['ci'][0] > 0)
    core = [it for it in ('P9', 'P10', 'P7', 'P8') if clear(it, 0.30)]
    res['rule'] = dict(A1_clear=clear('A1_valid', 0.20), core_clear=core, PASS=clear('A1_valid', 0.20) and len(core) >= 2)
    res['both_lack'] = [it for it, x in res['items'].items() if it.startswith(('P', 'T', 'S')) and x.get('teacher') is not None
                        and x['teacher_n'] >= 5 and (x['teacher'] < 1.5 or (x['teacher_share2'] or 0) < 0.6)]
    out[bench] = res
json.dump(out, open('/e/data1/mmlaion/lee27/experiments/pedagepa/m1/m1_table.json', 'w'), indent=1)
for b, r in out.items():
    print(f"\n== {b}: {r['n_tasks']} paired tasks; rule {r['rule']}; both lack {r['both_lack']}")
    for it, x in r['items'].items():
        print(f"  {it:20} teacher {x.get('teacher')} (n{x.get('teacher_n','')}, 2s {x.get('teacher_share2','')})  student {x.get('student')} (n{x.get('student_n','')}, LD {x.get('student_ld','')})  diff {x.get('diff')} {x.get('ci')}")
