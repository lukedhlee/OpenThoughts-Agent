#!/usr/bin/env python3
"""Print the OTA SFT held-out table (base + every scored export) from the heldout_nll_ota-*.json files.

    python ota_readout.py [--logs /e/data1/mmlaion/lee27/snowball-sft/logs] [--filter full]
"""
import argparse, glob, json, os, re

ap = argparse.ArgumentParser()
ap.add_argument("--logs", default="/e/data1/mmlaion/lee27/snowball-sft/logs")
ap.add_argument("--filter", default="", help="substring the file name must contain (e.g. 'full' or 'std-')")
a = ap.parse_args()
rows = []
for f in sorted(glob.glob(os.path.join(a.logs, "heldout_nll_ota-*.json"))):
    name = os.path.basename(f)[len("heldout_nll_ota-"):-len(".json")]
    if a.filter and a.filter not in name:
        continue
    d = json.load(open(f))
    sl = d.get("by_slice", {})
    rows.append((name, d["sequences"], d["heldout_nll"], d["think_nll"], d["rest_nll"],
                 *(sl.get(k, {}).get("heldout_nll", float("nan")) for k in ("issue", "swesmith", "tezos", "superuser"))))
print(f"{'export':44s} {'n':>5s} {'nll':>7s} {'think':>7s} {'rest':>7s} | {'issue':>6s} {'swesm':>6s} {'tezos':>6s} {'super':>6s}")
for r in rows:
    print(f"{r[0]:44s} {r[1]:5d} {r[2]:7.4f} {r[3]:7.4f} {r[4]:7.4f} | " + " ".join(f"{v:6.3f}" for v in r[5:]))
