#!/usr/bin/env python3
"""Verdict of the Horizon <-> Jupiter SFT port pair (marin HORIZON_JUPITER_PAIR.md) from the W&B curves.
Reads train/loss of snowball-kimi0921-pair-{horizon-r1,horizon-r2,jupiter-r1} in lukedhlee-marin/horizon-jupiter-sft-pair
and applies the pre-registered rule. Exit 0 = verdict printed, 3 = the Jupiter curve is not (fully) on W&B yet."""
import os, sys, wandb, statistics as st
api = wandb.Api(api_key=os.environ["WANDB_API_KEY"], timeout=120)
P = "lukedhlee-marin/horizon-jupiter-sft-pair"
def curve(rid):
    try: r = api.run(f"{P}/{rid}")
    except Exception as e: return None
    # history(), not scan_history(): the synced offline runs have no _step column for the scan API ("Step column
    # '_step' not found in schema")
    return {int(h["_step"]): h["train/loss"] for h in r.history(keys=["train/loss"], samples=10000, pandas=False)}
h1, h2, j1 = (curve(f"snowball-kimi0921-pair-{x}") for x in ("horizon-r1", "horizon-r2", "jupiter-r1"))
print("steps", len(h1 or {}), len(h2 or {}), len(j1 or {}))
sig = st.mean(abs(h1[t] - h2[t]) for t in range(1, 30)); print(f"horizon r1 vs r2: step0 {h1[0]:.5f} vs {h2[0]:.5f}; sigma (mean|d| t=1..29) {sig:.5f}; max {max(abs(h1[t]-h2[t]) for t in range(1,30)):.5f}")
print(" ".join(f"{t}:{h1[t]:.4f}" for t in range(30)))
if not j1 or len(j1) < 30:
    print("jupiter-r1 not complete on W&B"); sys.exit(3)
if j1:
    d = {t: h1[t] - j1[t] for t in range(30)}
    thr = max(3 * sig, 0.003); m = st.mean(abs(d[t]) for t in range(1, 30)); mx = max(abs(d[t]) for t in range(1, 30))
    print(f"D0 {d[0]:+.5f} (<=0.005: {abs(d[0])<=0.005}); mean|D| {m:.5f} (<= {thr:.5f}: {m<=thr}); max|D| {mx:.5f} (<=0.02: {mx<=0.02}); signed mean {st.mean(d[t] for t in range(1,30)):+.5f}")
    print("VERDICT", "PASS" if abs(d[0]) <= 0.005 and m <= thr and mx <= 0.02 else "FAIL")
