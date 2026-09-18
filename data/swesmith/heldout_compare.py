#!/usr/bin/env python3
"""Paired held-out comparison of scored exports (heldout_nll JSONs): overall, think vs rest, pass vs fail, and by
quartile of the BASE model's per-trial NLL (easy -> hard), so a coverage-style method can be read on the tail.
    python heldout_compare.py <logs dir> base full_lr5e-5 full_lr5e-5_tail25
"""
import json, statistics as st, sys
L = sys.argv[1]; names = sys.argv[2:]
D = {n: json.load(open(f"{L}/heldout_nll_{n}.json")) for n in names}
base = names[0]
print(f"{'model':22s} {'nll':>7s} {'think':>7s} {'rest':>7s} {'pass':>7s} {'fail':>7s} {'median':>7s}")
for n, d in D.items():
    print(f"{n:22s} {d['heldout_nll']:7.4f} {d['think_nll']:7.4f} {d['rest_nll']:7.4f} {d['pass_nll']:7.4f} {d['fail_nll']:7.4f} {d['per_sequence_nll_median']:7.4f}")
b = {x["instance_id"]: x for x in D[base]["per_sequence"]}
ids = [i for i in b]
qs = st.quantiles([b[i]["nll"] for i in ids], n=4)
def quart(v): return sum(v > q for q in qs)
groups = {q: [i for i in ids if quart(b[i]["nll"]) == q] for q in range(4)}
print("\nby quartile of BASE per-trial NLL (Q0 easiest .. Q3 hardest), mean per-trial NLL and paired delta vs base:")
for n in names[1:]:
    m = {x["instance_id"]: x for x in D[n]["per_sequence"]}
    row = []
    for q in range(4):
        g = groups[q]
        d = st.mean(m[i]["nll"] - b[i]["nll"] for i in g)
        w = sum(1 for i in g if m[i]["nll"] > b[i]["nll"])
        row.append(f"Q{q}: {st.mean(m[i]['nll'] for i in g):.4f} ({d:+.4f}, worse {w}/{len(g)})")
    print(f"  {n:22s} " + " | ".join(row))
if len(names) >= 3:
    a, c = names[1], names[2]
    ma = {x["instance_id"]: x for x in D[a]["per_sequence"]}; mc = {x["instance_id"]: x for x in D[c]["per_sequence"]}
    print(f"\npaired {c} minus {a} per trial:")
    for q in range(4):
        g = groups[q]
        d = [mc[i]["nll"] - ma[i]["nll"] for i in g]
        dt = [mc[i]["think_nll"] - ma[i]["think_nll"] for i in g]
        print(f"  Q{q}: nll {st.mean(d):+.4f} (better {sum(1 for x in d if x < 0)}/{len(g)}), think {st.mean(dt):+.4f}")
    d = [mc[i]["nll"] - ma[i]["nll"] for i in ids]
    print(f"  all: nll {st.mean(d):+.4f} median {st.median(d):+.4f} (better {sum(1 for x in d if x < 0)}/{len(ids)})")
