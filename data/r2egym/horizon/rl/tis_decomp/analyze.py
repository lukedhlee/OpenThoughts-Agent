#!/usr/bin/env python3
"""tis_decomp analysis: mean |delta log p| per comparison (token-weighted, the tis/log_ratio_abs_mean analogue) with
95 % bootstrap CIs over turns, router-flip rates, and the gap binned by flips, router margin, entropy and position.

  python analyze.py --vdir VDIR --tout TOUT --s3-set S --h9-set H [--proxy 'P.part*.jsonl'] [--gate2 G.jsonl] [--json OUT]
numpy only (no torch); light enough for the login node.
"""
import argparse
import glob
import json
import math
import os

import numpy as np

S3_LABELS = {"vA": "s3_dp4ep_c4.r0", "vB": "s3_dp4ep_c4.r1", "vC": "s3_dp4ep_c1.r0", "vD": "s3_dp4ep_c1.r1",
             "vE": "s3_tp1_c1.r0"}
H9_LABELS = {"vA": "h9_dp4ep_c4.r0", "vB": "h9_dp4ep_c4.r1", "vC": "h9_dp4ep_c1.r0"}
NAMES = {
    "vA": "vLLM DP4-EP, 4 in flight (gate-2 setting), run 1",
    "vB": "vLLM DP4-EP, 4 in flight, run 2",
    "vC": "vLLM DP4-EP, one request at a time, run 1",
    "vD": "vLLM DP4-EP, one request at a time, run 2",
    "vE": "vLLM single GPU (no EP), one at a time",
    "orig": "vLLM sampling-time logprobs (rollout)",
    "g0": "gate-2 job 35865 run 0", "g1": "gate-2 job 35865 run 1",
    "proxy": "proxy trainer job 36712",
    "T0": "trainer (FA2, grouped_mm, bf16 logits)", "T0b": "trainer, same forward again",
    "eager_moe": "trainer, eager per-expert MoE loop", "ref_attn": "trainer, eager fp32-score attention",
    "fp32logits": "trainer, fp32 LM head", "bf16_rederived": "trainer, bf16 LM head re-derived",
    "replay_vC": "trainer with vLLM(C)'s experts forced", "replay_vA": "trainer with vLLM(A)'s experts forced",
    "replay_vE": "trainer with vLLM(E)'s experts forced", "replay_self": "trainer with its own experts forced",
    "replay_vB": "trainer with vLLM(B)'s experts forced", "replay_vD": "trainer with vLLM(D)'s experts forced",
    "replay_refattn_vC": "trainer, vLLM(C)'s experts + fp32-score attn",
}
MARGIN_EDGES = [1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]
ENT_EDGES = [0.01, 0.1, 0.3, 1.0, 2.0]
POS_EDGES = [8192, 16384, 32768]


def load_vllm(path):
    out = {}
    if not os.path.exists(path):
        return out
    for l in open(path):
        d = json.loads(l)
        if "id" in d:
            out[d["id"]] = d["lp"]
    return out


def boot(sums, counts, n=2000, seed=0):
    sums, counts = np.asarray(sums, float), np.asarray(counts, float)
    m = sums.sum() / max(counts.sum(), 1)
    if len(sums) < 2 or counts.sum() == 0:
        return m, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n, len(sums)))
    s, c = sums[idx].sum(1), counts[idx].sum(1)
    r = np.sort(s / np.maximum(c, 1))
    return m, r[int(0.025 * n)], r[int(0.975 * n) - 1]


class Model:
    def __init__(self, tag, a, set_path, labels):
        self.tag = tag
        parts = sorted(glob.glob(f"{a.tout}.{tag}.part*.npz"))
        if not parts:
            raise FileNotFoundError(f"no trainer parts {a.tout}.{tag}.part*.npz")
        rows = {}
        for l in open(set_path):
            r = json.loads(l)
            rows[r["id"]] = r
        tok, turn, ids = {}, {}, []
        for p in parts:
            z = np.load(p)
            base = len(ids)
            ids += list(z["ids"])
            for k in z.files:
                if k.startswith("tok_"):
                    v = z[k] + base if k == "tok_turn" else z[k]
                    tok.setdefault(k[4:], []).append(v)
                elif k.startswith("turn_"):
                    turn.setdefault(k[5:], []).append(z[k])
        self.ids = ids
        self.tok = {k: np.concatenate(v) for k, v in tok.items()}
        self.turn = {k: np.concatenate(v) for k, v in turn.items()}
        self.T = self.tok["turn"]
        n = len(self.T)
        for k in [k for k, v in self.tok.items() if len(v) != n]:
            print(f"[{tag}] dropping misaligned {k} ({len(self.tok[k])} vs {n} tokens)")
            del self.tok[k]
        self.lp = {k[3:]: v.astype(np.float64) for k, v in self.tok.items() if k.startswith("lp_")}
        for short, lab in labels.items():
            for pre in ("replay_", "replay_refattn_"):
                if f"{pre}{lab}" in self.lp:
                    self.lp[f"{pre}{short}"] = self.lp.pop(f"{pre}{lab}")
        # vLLM sides, aligned to the trainer's token order
        srcs = {s: load_vllm(os.path.join(a.vdir, f"{lab}.jsonl")) for s, lab in labels.items()}
        srcs["orig"] = {i: rows[i].get("orig_logprobs") for i in ids if rows[i].get("orig_logprobs")}
        if tag == "s3" and a.gate2:
            srcs["g0"], srcs["g1"] = {}, {}
            for l in open(a.gate2):
                d = json.loads(l)
                if "id" in d:
                    srcs[f"g{d['rep']}"][d["id"]] = d["lp"]
        if tag == "s3" and a.proxy:
            srcs["proxy"] = {}
            for p in sorted(glob.glob(a.proxy)):
                for l in open(p):
                    d = json.loads(l)
                    srcs["proxy"][d["id"]] = d["lp"]
        for s, d in srcs.items():
            if all(i in d for i in ids):
                v = np.concatenate([np.asarray(d[i], float) for i in ids])
                assert len(v) == n, (s, len(v), n)
                self.lp[s] = v
            elif d:
                print(f"[{tag}] {s}: {sum(i in d for i in ids)} of {len(ids)} turns, skipped")
        self.ent = self.tok["ent"].astype(np.float64)
        self.nturn = len(ids)

    def pair(self, x, y, sel=None):
        if x not in self.lp or y not in self.lp:
            return None
        d = np.abs(self.lp[x] - self.lp[y])
        ok = np.isfinite(d)
        w = ok.astype(float) if sel is None else (sel & ok).astype(float)
        d = np.where(ok, d, 0.0)
        if w.sum() == 0:
            return None
        sums = np.bincount(self.T, weights=d * w, minlength=self.nturn)
        cnt = np.bincount(self.T, weights=w, minlength=self.nturn)
        m, lo, hi = boot(sums, cnt)
        big = float(((d > 0.69) * w).sum() / max(w.sum(), 1))
        return {"mean": m, "lo": lo, "hi": hi, "tokens": int(w.sum()), "frac_gt_0.69": big}


def fmt(r):
    return "n/a" if r is None else f"{r['mean']:.5f} [{r['lo']:.5f}, {r['hi']:.5f}]"


def table(M, comps, title):
    print(f"\n## {M.tag}: {title}")
    res = {}
    for x, y in comps:
        r = M.pair(x, y)
        if r is None:
            continue
        res[f"{x}|{y}"] = r
        print(f"  {NAMES.get(x, x):46s} vs {NAMES.get(y, y):46s} {fmt(r)}  |d|>0.69 {r['frac_gt_0.69']:.1e}")
    return res


def binned(M, x, y, key, edges, label):
    v = {"ent": M.ent, "pos": M.tok["pos"], "minmargin": M.tok["minmargin"]}.get(key)
    if v is None:
        v = M.tok[key]
    if x not in M.lp or y not in M.lp:
        return None
    b = np.digitize(v, edges)
    d = np.abs(M.lp[x] - M.lp[y])
    ok = np.isfinite(d)
    d = np.where(ok, d, 0.0)
    tot = d.sum()
    out = []
    names = [f"<{edges[0]}"] + [f"{edges[i]}-{edges[i + 1]}" for i in range(len(edges) - 1)] + [f">={edges[-1]}"]
    print(f"  by {label} ({NAMES.get(x, x)} vs {NAMES.get(y, y)}):")
    for k, nm in enumerate(names):
        sel = (b == k) & ok
        if not sel.any():
            continue
        r = M.pair(x, y, sel)
        r.update({"bin": nm, "token_share": float(sel.sum() / ok.sum()), "mass_share": float(d[sel].sum() / tot)})
        out.append(r)
        print(f"    {nm:>12s}  tokens {r['token_share']:6.1%}  mean|d| {fmt(r)}  share of |d| mass {r['mass_share']:6.1%}")
    return out


def flips(M, pairs):
    print(f"\n## {M.tag}: router flips (top-4 set differs), per (position, layer) and per completion token")
    res = {}
    L = M.turn[f"layerflips_{pairs[0][0]}"].shape[1] if f"layerflips_{pairs[0][0]}" in M.turn else 26
    for name, desc in pairs:
        if f"nflip_{name}" not in M.tok:
            continue
        has = M.turn.get(f"has_{name}", np.ones(len(M.turn["npos"]), int)).astype(bool)
        if not has.any():
            continue
        np_ = M.turn["npos"][has].sum()
        lf = M.turn[f"layerflips_{name}"][has]
        nf = M.tok[f"nflip_{name}"]
        nf = nf[nf != 255]
        mh = M.turn[f"marginhist_{name}"][has].sum(0)
        r = {"pos_layer_rate": float(lf.sum() / (np_ * L)), "per_layer_rate": (lf.sum(0) / np_).tolist(),
             "tok_any_flip": float((nf > 0).mean()), "tok_mean_layers_flipped": float(nf.mean()),
             "flip_rate_by_margin": [float(f / t) if t else None for t, f in mh],
             "margin_bin_share": [float(t / mh[:, 0].sum()) for t, _ in mh]}
        res[name] = r
        print(f"  {desc:58s} (pos,layer) {r['pos_layer_rate']:.2e}  tokens with any flip {r['tok_any_flip']:6.2%}  "
              f"mean layers flipped {r['tok_mean_layers_flipped']:.3f}")
    names = [f"<{MARGIN_EDGES[0]}"] + [f"{MARGIN_EDGES[i]}-{MARGIN_EDGES[i + 1]}"
                                      for i in range(len(MARGIN_EDGES) - 1)] + [f">={MARGIN_EDGES[-1]}"]
    if res:
        print("  P(flip | trainer's 4th-5th margin), all positions and layers:")
        print("    " + " ".join(f"{n:>10s}" for n in ["margin"] + names))
        k0 = next(iter(res))
        print("    " + " ".join(f"{x:>10s}" for x in ["share"] + [f"{s:.2%}" for s in res[k0]["margin_bin_share"]]))
        for name, r in res.items():
            print("    " + " ".join(f"{x:>10s}" for x in [name] + ["-" if f is None else f"{f:.1e}"
                                                                for f in r["flip_rate_by_margin"]]))
    return res


def flip_split(M, x, y, fname):
    if f"nflip_{fname}" not in M.tok or x not in M.lp or y not in M.lp:
        return None
    nf = M.tok[f"nflip_{fname}"]
    d = np.abs(M.lp[x] - M.lp[y])
    ok = np.isfinite(d) & (nf != 255)
    f, g = ok & (nf > 0), ok & (nf == 0)
    a, b = M.pair(x, y, f), M.pair(x, y, g)
    r = {"flipped": a, "unflipped": b, "token_share_flipped": float(f.sum() / ok.sum()),
         "mass_share_flipped": float(d[f].sum() / d[ok].sum())}
    print(f"  {NAMES.get(x, x)} vs {NAMES.get(y, y)} split by routing flip at the predicting position ({fname}):")
    print(f"    flipped   tokens {r['token_share_flipped']:6.2%}  mean|d| {fmt(a)}  share of |d| mass {r['mass_share_flipped']:6.1%}")
    print(f"    unflipped tokens {1 - r['token_share_flipped']:6.2%}  mean|d| {fmt(b)}")
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vdir", required=True)
    ap.add_argument("--tout", required=True, help="trainer_decomp --out prefix")
    ap.add_argument("--s3-set", required=True)
    ap.add_argument("--h9-set", default="")
    ap.add_argument("--proxy", default="")
    ap.add_argument("--gate2", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    summary = {}
    models = {}
    for tag, sp, labels in (("s3", a.s3_set, S3_LABELS), ("h9", a.h9_set, H9_LABELS)):
        if not sp:
            continue
        try:
            M = Model(tag, a, sp, labels)
        except FileNotFoundError as e:
            print(e)
            continue
        models[tag] = M
        S = summary[tag] = {"turns": M.nturn, "tokens": int(len(M.T)), "mean_entropy": float(M.ent.mean()),
                            "median_entropy": float(np.median(M.ent))}
        print(f"\n# {tag}: {M.nturn} turns, {len(M.T)} completion tokens, trainer entropy mean {M.ent.mean():.3f} "
              f"median {np.median(M.ent):.4f}; lp sources: {sorted(M.lp)}")
        S["vllm_noise"] = table(M, [("vA", "vB"), ("vC", "vD"), ("vA", "vC"), ("vC", "vE"), ("g0", "g1"), ("vA", "g0")],
                                "(A) vLLM against itself")
        S["trainer_vs_vllm"] = table(M, [("T0", "vA"), ("T0", "vB"), ("T0", "vC"), ("T0", "vE"), ("T0", "orig"),
                                         ("T0", "g0"), ("vA", "orig"), ("vC", "orig")],
                                     "trainer vs vLLM (the tis/log_ratio_abs_mean analogue)")
        S["trainer_kernels"] = table(M, [("T0", "T0b"), ("T0", "proxy"), ("T0", "bf16_rederived"), ("T0", "replay_self"),
                                         ("T0", "eager_moe"), ("T0", "ref_attn"), ("T0", "fp32logits"),
                                         ("eager_moe", "vC"), ("ref_attn", "vC"), ("fp32logits", "vC")],
                                     "(B) trainer against itself and kernel variants")
        S["replay"] = table(M, [("replay_vC", "vC"), ("replay_vA", "vA"), ("replay_vB", "vB"), ("replay_vE", "vE"),
                                ("replay_refattn_vC", "vC"), ("replay_refattn_vC", "replay_vC"),
                                ("replay_vC", "vD"), ("replay_vA", "vB"), ("replay_vC", "orig"),
                                ("replay_vA", "replay_vB"), ("replay_vC", "replay_vD"), ("replay_vA", "replay_vC"),
                                ("replay_vC", "replay_vE"), ("replay_vC", "T0"), ("replay_vA", "T0")],
                            "(C) router replay: trainer forced onto vLLM's experts (residual = non-routing numerics)")
        S["flips"] = flips(M, [("T0_vC", "trainer vs vLLM C"), ("T0_vA", "trainer vs vLLM A"),
                               ("T0_vE", "trainer vs vLLM E (single GPU)"), ("vA_vB", "vLLM A vs B (rerun, 4 in flight)"),
                               ("vC_vD", "vLLM C vs D (rerun, one at a time)"), ("vA_vC", "vLLM A vs C (batching)"),
                               ("vC_vE", "vLLM C vs E (EP vs single GPU)"), ("T0_eager", "trainer vs eager MoE"),
                               ("T0_refattn", "trainer vs fp32-score attention")])
        print(f"\n## {tag}: |delta| split by routing flips")
        S["flip_split"] = {k: flip_split(M, x, y, f) for k, x, y, f in (
            ("T0_vC", "T0", "vC", "T0_vC"), ("T0_vA", "T0", "vA", "T0_vA"), ("vA_vB", "vA", "vB", "vA_vB"),
            ("vA_vC", "vA", "vC", "vA_vC"), ("T0_refattn", "T0", "ref_attn", "T0_refattn"),
            ("T0_eager", "T0", "eager_moe", "T0_eager"), ("T0_orig_byC", "T0", "orig", "T0_vC"))}
        print(f"\n## {tag}: |delta| binned")
        S["bins"] = {}
        for x, y in (("T0", "vA"), ("T0", "vC"), ("T0", "orig"), ("vA", "vB"), ("replay_vC", "vC")):
            S["bins"][f"{x}|{y}"] = {
                "minmargin": binned(M, x, y, "minmargin", MARGIN_EDGES, "min router margin over layers at the predicting position"),
                "entropy": binned(M, x, y, "ent", ENT_EDGES, "trainer entropy (nats)"),
                "position": binned(M, x, y, "pos", POS_EDGES, "token position in the sequence")}
    if "s3" in models and "h9" in models:
        s3, h9 = models["s3"], models["h9"]
        print("\n# (D) Stage-3 vs H9: does the gap track entropy? Per entropy bin mean|d|, and H9's gap predicted by "
              "reweighting Stage-3's per-bin gap with H9's entropy mix")
        D = summary["s3_vs_h9"] = {}
        for x, y in (("T0", "vA"), ("T0", "vC"), ("T0", "orig"), ("vA", "vB")):
            if x not in s3.lp or y not in s3.lp or x not in h9.lp or y not in h9.lp:
                continue
            b3, b9 = np.digitize(s3.ent, ENT_EDGES), np.digitize(h9.ent, ENT_EDGES)
            d3, d9 = np.abs(s3.lp[x] - s3.lp[y]), np.abs(h9.lp[x] - h9.lp[y])
            pred, rows = 0.0, []
            for k in range(len(ENT_EDGES) + 1):
                m3 = d3[b3 == k].mean() if (b3 == k).any() else float("nan")
                m9 = d9[b9 == k].mean() if (b9 == k).any() else float("nan")
                w9 = (b9 == k).mean()
                if w9 and not math.isnan(m3):
                    pred += w9 * m3
                rows.append({"bin": k, "s3_share": float((b3 == k).mean()), "h9_share": float(w9),
                             "s3_mean": float(m3), "h9_mean": float(m9)})
            D[f"{x}|{y}"] = {"s3": float(d3.mean()), "h9": float(d9.mean()), "h9_predicted_from_s3_by_entropy": pred,
                             "bins": rows}
            print(f"  {NAMES.get(x, x)} vs {NAMES.get(y, y)}: s3 {d3.mean():.5f}, h9 {d9.mean():.5f}, "
                  f"h9 predicted from s3 per-entropy-bin gap {pred:.5f}")
            for r in rows:
                print(f"    ent bin {r['bin']}: share s3 {r['s3_share']:6.1%} h9 {r['h9_share']:6.1%}   "
                      f"mean|d| s3 {r['s3_mean']:.5f} h9 {r['h9_mean']:.5f}")
    if a.json:
        json.dump(summary, open(a.json, "w"), indent=1)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
