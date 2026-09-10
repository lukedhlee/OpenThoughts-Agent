#!/usr/bin/env python
"""Diff two loss_equiv.py JSON reports.  Usage: compare.py A.json B.json [labelA labelB]"""

from __future__ import annotations

import json
import sys


def load(path):
    with open(path) as fh:
        return json.load(fh)


def main() -> None:
    a, b = load(sys.argv[1]), load(sys.argv[2])
    la = sys.argv[3] if len(sys.argv) > 3 else "A"
    lb = sys.argv[4] if len(sys.argv) > 4 else "B"

    print(f"{la} = {a['git']['sha'][:8]}  {a['git']['subject'][:70]}")
    print(f"{lb} = {b['git']['sha'][:8]}  {b['git']['subject'][:70]}")
    print()

    print("== API ==")
    for key in ("kl_fn_name", "has_PRESCALED_SUM_LOSS_REDUCTIONS", "has_ROLLOUT_LOGPROB_POLICY_LOSSES"):
        if a["api"][key] != b["api"][key]:
            print(f"  {key}: {la}={a['api'][key]!r}  {lb}={b['api'][key]!r}")
    for key in ("policy_loss_types", "supported_reductions"):
        sa, sb = set(a["api"][key]), set(b["api"][key])
        if sa != sb:
            print(f"  {key}: {la}-only={sorted(sa - sb)}  {lb}-only={sorted(sb - sa)}")
    for name in sorted(set(a["api"]["signatures"]) | set(b["api"]["signatures"])):
        sa = a["api"]["signatures"].get(name)
        sb = b["api"]["signatures"].get(name)
        if sa != sb:
            print(f"  signature {name}:\n    {la}: {sa}\n    {lb}: {sb}")
    print()

    ra = {r["variant"]: r for r in a["results"]}
    rb = {r["variant"]: r for r in b["results"]}

    hdr = f"{'variant':<32} {'status':<18} {la+' loss':>14} {lb+' loss':>14} {'|dloss|':>10} {la+' |g|':>12} {lb+' |g|':>12} {'max|dg|':>10}"
    print("== variants ==")
    print(hdr)
    print("-" * len(hdr))
    for name in [r["variant"] for r in a["results"]]:
        x, y = ra.get(name), rb.get(name)
        if x["status"] != "ok" or y["status"] != "ok":
            sa = x["status"] if x["status"] == "ok" else f"{x['status']}:{x.get('error_type', '')}"
            sb = y["status"] if y["status"] == "ok" else f"{y['status']}:{y.get('error_type', '')}"
            print(f"{name:<32} {la}={sa} / {lb}={sb}")
            continue
        dloss = abs(x["optimization_loss"] - y["optimization_loss"])
        dg = max(abs(p - q) for p, q in zip(x["grad_first10"], y["grad_first10"]))
        dgn = abs(x["grad_norm"] - y["grad_norm"])
        same = "IDENTICAL" if (dloss == 0 and dg == 0 and dgn == 0) else "DIFFERS"
        print(
            f"{name:<32} {same:<18} {x['optimization_loss']:>14.10f} {y['optimization_loss']:>14.10f} "
            f"{dloss:>10.2e} {x['grad_norm']:>12.8f} {y['grad_norm']:>12.8f} {max(dg, dgn):>10.2e}"
        )
    print()

    print("== metric keys ==")
    for name in [r["variant"] for r in a["results"]]:
        x, y = ra.get(name), rb.get(name)
        if x["status"] != "ok" or y["status"] != "ok":
            continue
        ka, kb = set(x["metrics"]), set(y["metrics"])
        if ka != kb:
            print(f"  {name}: {la}-only={sorted(ka - kb)}  {lb}-only={sorted(kb - ka)}")
        diffs = {k: (x["metrics"][k], y["metrics"][k]) for k in ka & kb if x["metrics"][k] != y["metrics"][k]}
        if diffs:
            print(f"  {name}: differing values {diffs}")
    print("  (nothing listed above = identical metric keys and values on every ok variant)")


if __name__ == "__main__":
    main()
