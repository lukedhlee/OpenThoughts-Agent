"""agree.py: judge agreement per checklist item from the eval database.

For each pair of judgments (e.g. two Opus readers, Opus vs Sonnet) and each item: n trajectories both scored
numerically, exact agreement, quadratic-weighted kappa on 0/1/2, and how often they disagree on NA/LD vs a number.
Also anchor checks: P10 = 2 on a claim right after an error signature; P9 = 2 with no command after the last edit.

  python agree.py --db DB --pairs cf40_opusA_v1:cf40_opusB_v1.2 cf40_opusB_v1.2:cf40_sonnet_v1.2 ...
"""
import argparse, json, re, sqlite3
from collections import defaultdict

ITEMS = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P5a', 'P5b', 'P6', 'P7', 'P8', 'P9', 'P10', 'T1', 'S1', 'S2', 'S3', 'S4']
NUM = re.compile(r'^[0-2](\.0)?$')


def qwk(a, b, k=3):
    n = len(a)
    if n == 0:
        return None
    O = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        O[x][y] += 1
    ha = [sum(O[i]) for i in range(k)]; hb = [sum(O[i][j] for i in range(k)) for j in range(k)]
    num = sum(((i - j) ** 2) / (k - 1) ** 2 * O[i][j] for i in range(k) for j in range(k))
    den = sum(((i - j) ** 2) / (k - 1) ** 2 * ha[i] * hb[j] / n for i in range(k) for j in range(k))
    return None if den == 0 else 1 - num / den


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--db', default='/e/data1/mmlaion/lee27/experiments/pedagepa/evaldb/pedagepa.sqlite')
    ap.add_argument('--pairs', nargs='+', required=True); a = ap.parse_args()
    con = sqlite3.connect(a.db)
    S = defaultdict(dict)
    for key, jn, item, sc in con.execute('SELECT trial_key, judgment, item, score FROM judgments'):
        S[(jn, key)][item] = sc
    for pair in a.pairs:
        A, B = pair.split(':')
        keys = sorted({k for (j, k) in S if j == A} & {k for (j, k) in S if j == B})
        print(f'\n{A} vs {B}: {len(keys)} trajectories')
        print(f'{"item":5} {"n":>3} {"exact":>6} {"QWK":>6} {"num/NA mismatch":>16}')
        for it in ITEMS:
            xa, xb, mm = [], [], 0
            for k in keys:
                sa, sb = S[(A, k)].get(it), S[(B, k)].get(it)
                if sa is None or sb is None:
                    continue
                if NUM.match(str(sa)) and NUM.match(str(sb)):
                    xa.append(int(float(sa))); xb.append(int(float(sb)))
                elif bool(NUM.match(str(sa))) != bool(NUM.match(str(sb))):
                    mm += 1
            if not xa and not mm:
                continue
            q = qwk(xa, xb)
            ex = sum(x == y for x, y in zip(xa, xb)) / len(xa) if xa else None
            print(f'{it:5} {len(xa):>3} {"" if ex is None else f"{ex:.2f}":>6} {"" if q is None else f"{q:.2f}":>6} {mm:>16}')
    # anchors (current facts)
    print('\nanchor violations (current facts):')
    F = {k: json.loads(f or '{}') for k, f in con.execute('SELECT trial_key, facts FROM trials')}
    for jn in sorted({j for (j, _) in S}):
        v9 = v10 = n9 = n10 = 0
        for (j, k), sc in S.items():
            if j != jn:
                continue
            f = F.get(k, {})
            if f.get('claims'):
                if str(sc.get('P9')) in ('2', '2.0'):
                    n9 += 1; v9 += f.get('commands_between_last_edit_and_first_claim', 1) == 0 and f.get('last_edit_reply') is not None
                if str(sc.get('P10')) in ('2', '2.0'):
                    n10 += 1; v10 += bool(f.get('error_signature_right_before_first_claim'))
        print(f'  {jn}: P9=2 with no command after the last edit {v9}/{n9}; P10=2 right after an error signature {v10}/{n10}')


if __name__ == '__main__':
    main()
