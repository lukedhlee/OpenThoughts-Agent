#!/usr/bin/env python3
"""Cross-arm comparison table from bench results/. Usage: collect.py [results_dir] [--jobs id,id,...]"""
import glob, json, os, sys
R = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "/e/project1/transfernetx/lee27/code/snowball/bench/results"
only = None
if "--jobs" in sys.argv: only = set(sys.argv[sys.argv.index("--jobs") + 1].split(","))
def g(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or k not in d: return default
        d = d[k]
    return d
def f(v, nd=1): return "na" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))
rows = []
for d in sorted(glob.glob(os.path.join(R, "*_*"))):
    arm, jid = os.path.basename(d).rsplit("_", 1)
    if only and jid not in only: continue
    row = {"arm": arm, "job": jid}
    for fn in glob.glob(os.path.join(d, "sweep_p*.json")):
        j = json.load(open(fn)); ctx = j.get("prefix_prompt_tokens"); pts = j.get("points", {})
        if not ctx: continue
        tag = "s" if ctx < 6000 else ("m" if ctx < 20000 else "l")
        for c in ("16", "44", "128"):
            if c in pts: row[f"sw{tag}_c{c}"] = g(pts[c], "decode_tps_p50")
    for fn in glob.glob(os.path.join(d, "replay_forced_c*.json")):
        j = json.load(open(fn)); st = j.get("steady", {}); c = j.get("conc")
        row.update({f"rf{c}_lat50": st.get("latency_p50"), f"rf{c}_lat90": st.get("latency_p90"), f"rf{c}_dec50": st.get("decode_tps_p50"),
                    f"rf{c}_ttft50": st.get("ttft_p50"), f"rf{c}_agg": st.get("agg_tps"), f"rf{c}_tpm": st.get("turns_per_min"), f"rf{c}_run": st.get("mean_running"), f"rf{c}_err": st.get("errors")})
    for fn in glob.glob(os.path.join(d, "replay_natural_c*.json")):
        j = json.load(open(fn)); st = j.get("steady", {}); c = j.get("conc")
        row.update({f"rn{c}_lat50": st.get("latency_p50"), f"rn{c}_dec50": st.get("decode_tps_p50"), f"rn{c}_comp50": st.get("completion_tokens_p50"), f"rn{c}_agg": st.get("agg_tps"), f"rn{c}_tpm": st.get("turns_per_min"), f"rn{c}_err": st.get("errors")})
    fn = os.path.join(d, "prefill.json")
    if os.path.exists(fn):
        j = json.load(open(fn)); pts = j.get("points", {})
        for c in ("16", "64"):
            if c in pts: row[f"pf_c{c}"] = g(pts[c], "prefill_tps")
    rows.append(row)
cols = ["arm", "job", "sws_c16", "sws_c44", "sws_c128", "swm_c44", "swm_c128", "rf64_lat50", "rf64_lat90", "rf64_ttft50", "rf64_dec50", "rf64_agg", "rf64_tpm", "rf64_run", "rf64_err",
        "rn64_lat50", "rn64_dec50", "rn64_comp50", "rn64_agg", "rn64_tpm", "rn64_err", "pf_c16", "pf_c64"]
extra = sorted({k for r in rows for k in r} - set(cols))
cols += extra
print("\t".join(cols))
for r in rows:
    print("\t".join(f(r.get(c)) for c in cols))
