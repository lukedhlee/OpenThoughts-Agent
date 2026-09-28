#!/usr/bin/env python3
"""Arm A (autofix_loss='none') against arm C (the same manifest rendered with autofix_loss='content'), row for row.

    python check_autofix_none.py <A rendered jsonl> <C rendered jsonl> <09-21 tokenizer.json> [--expect-drop N]

Passes (exit 0) only if: the same session ids in the same order; identical token ids; A's loss is C's loss with some
tokens switched off and none switched on; every switched-off token lies inside an autofixed 09-21 (student) turn of C;
A trains no token inside any 09-21 turn; and, with --expect-drop, the number switched off equals N (qa_final.py's
student_autofix_trained_tokens for C). Token -> character mapping: decode the ids and re-encode with offsets (the
render's own tokenizer; the round trip is checked per row). Login-node safe: one process, one thread.
"""
import argparse
import json
import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('RAYON_NUM_THREADS', '1')
from tokenizers import Tokenizer  # noqa: E402


def within(offsets, spans):
    """Per token: does it lie wholly inside one of the (sorted, disjoint) char spans? Two-pointer, linear."""
    spans = sorted(tuple(x) for x in spans)
    out, j = [False] * len(offsets), 0
    for k, (a, b) in enumerate(offsets):
        while j < len(spans) and spans[j][1] < b:
            j += 1
        out[k] = j < len(spans) and spans[j][0] <= a and b <= spans[j][1]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('a')
    ap.add_argument('c')
    ap.add_argument('tokenizer')
    ap.add_argument('--expect-drop', type=int)
    a = ap.parse_args()
    tok = Tokenizer.from_file(a.tokenizer)
    st = dict(rows=0, sid_or_order_mismatch=0, ids_mismatch=0, loss_switched_on=0, dropped=0,
              dropped_outside_student_autofix=0, a_trained_in_student_turns=0, roundtrip_mismatch=0,
              a_trained=0, c_trained=0, rows_changed=0)
    with open(a.a) as fa, open(a.c) as fc:
        for la, lc in zip(fa, fc):
            ra, rc = json.loads(la), json.loads(lc)
            st['rows'] += 1
            if ra['sid'] != rc['sid']:
                st['sid_or_order_mismatch'] += 1
                continue
            if ra['ids'] != rc['ids']:
                st['ids_mismatch'] += 1
                continue
            la_, lc_ = ra['loss'], rc['loss']
            st['a_trained'] += sum(la_)
            st['c_trained'] += sum(lc_)
            on = sum(1 for x, y in zip(la_, lc_) if x and not y)
            off = [k for k, (x, y) in enumerate(zip(la_, lc_)) if y and not x]
            st['loss_switched_on'] += on
            st['dropped'] += len(off)
            st['rows_changed'] += bool(off)
            text = tok.decode(ra['ids'], skip_special_tokens=False)
            enc = tok.encode(text, add_special_tokens=False)
            if enc.ids != ra['ids']:
                st['roundtrip_mismatch'] += 1
                continue
            in_stud = within(enc.offsets, [t['span'] for t in rc['turns'] if t['owner'] == 'student'])
            in_af = within(enc.offsets, [t['span'] for t in rc['turns'] if t['owner'] == 'student' and t.get('autofix')])
            st['dropped_outside_student_autofix'] += sum(1 for k in off if not in_af[k])
            st['a_trained_in_student_turns'] += sum(1 for k, x in enumerate(la_) if x and in_stud[k])
        st['extra_rows_a'] = sum(1 for _ in fa)
        st['extra_rows_c'] = sum(1 for _ in fc)
    ok = (st['rows'] > 0 and not st['sid_or_order_mismatch'] and not st['ids_mismatch'] and not st['loss_switched_on']
          and not st['dropped_outside_student_autofix'] and not st['a_trained_in_student_turns']
          and not st['roundtrip_mismatch'] and not st['extra_rows_a'] and not st['extra_rows_c']
          and (a.expect_drop is None or st['dropped'] == a.expect_drop))
    print(json.dumps(dict(st, expect_drop=a.expect_drop, verdict='OK' if ok else 'FAIL')))
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
