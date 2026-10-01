#!/usr/bin/env python3
"""tis_decomp, trainer side: where does `policy/tis/log_ratio_abs_mean` come from?

Same forward as rl_logprob_parity.py (MarinSkyRL HFModelWrapper, GrugMoeForCausalLM, flash_attention_2, bf16, native
grouped_mm, no packing, one GPU per process, the whole model per GPU), on the same fixed turns, plus:

  router patch   GrugMoeRouter.forward replaced by the same math (fp32 logits + fp32 bias, stable top-(k+1), sigmoid
                 combine renormalized to 2.5) that also (a) records each token's top-4 set and the margin between the
                 4th and 5th biased router logit per layer, and (b) can force given expert ids per layer (router
                 replay; combine weights still come from the live trainer logits, as vLLM computes its own).
                 MarinSkyRL refuses its own R3 replay for Grug, hence the patch.
  variants       T0 the proxy forward (records routing, keeps the final hidden state of the completion positions);
                 T0b the same forward again (trainer run-to-run); eager_moe (the model's per-expert loop instead of
                 grouped_mm); ref_attn (the model's eager attention: fp32 scores, query-chunked so 62k tokens fit, in
                 place of FA2); fp32logits (LM head in fp32 on T0's hidden state; also gives the entropy);
                 bf16_rederived (bf16 LM head on the same hidden state: checks the re-derivation); replay_<label>
                 (vLLM's routed experts for that vLLM run forced at every position); replay_self (T0's own routing
                 forced: checks the patch).
  routing pairs  per completion token: the number of layers (0-26) whose top-4 set differs between two runs at the
                 position that predicts it; per turn: flips per layer over all positions, and flips by T0's margin.

Writes OUT.<tag>.part<rank>.npz (per-token arrays, concatenated over turns) and a progress log; analyze.py reads them.
"""
import argparse
import gc
import json
import math
import os
import sys
import time

HORIZON = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # .../data/r2egym/horizon
sys.path.insert(0, HORIZON)
from rl_logprob_parity import load_set  # noqa: E402

MARGIN_EDGES = [1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]  # biased router logit units; 8 bins


class Ctx:
    record = False
    sel = None
    margin = None
    replay = None
    ref_attn = False
    want_hidden = None
    hidden = None


CTX = Ctx()


def install_patches(hf):
    import torch
    import torch.nn.functional as F
    from skyrl_train.models import grug_moe as gm

    for i, layer in enumerate(hf.model.layers):
        layer.mlp.router._tisd_layer = i
    if getattr(gm, "_tisd_patched", False):
        return
    gm._tisd_patched = True

    def router_forward(self, hidden_states):
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            router_logits = F.linear(hidden_states.float(), self.weight.float())
            bias = self.bias.detach().to(device=router_logits.device, dtype=torch.float32)
            biased_logits = router_logits + bias
            topk_logits, topk_indices = gm.jax_top_k(biased_logits, self.top_k + 1)
            selected_experts = topk_indices[:, :-1]
            i = self._tisd_layer
            if CTX.record:
                CTX.sel[i] = selected_experts.to(torch.uint8)
                CTX.margin[i] = topk_logits[:, -2] - topk_logits[:, -1]
            if CTX.replay is not None:
                selected_experts = CTX.replay[i]
            selected_logits = torch.gather(router_logits, dim=-1, index=selected_experts)
            combine_weights = torch.sigmoid(selected_logits)
            combine_weights = combine_weights * (
                gm.GRUG_ROUTING_RENORM_SUM / (combine_weights.sum(dim=-1, keepdim=True) + gm.GRUG_ROUTER_RENORM_EPS))
        return gm.GrugMoeRouterOutput(router_logits, selected_experts, combine_weights)

    gm.GrugMoeRouter.forward = router_forward
    flash = gm.GrugMoeAttention._flash_attention

    def attention(self, query, key, value, attention_mask, *, is_long):
        if not CTX.ref_attn:
            return flash(self, query, key, value, attention_mask, is_long=is_long)
        if attention_mask is not None and not bool(attention_mask.all()):
            raise RuntimeError("ref_attn assumes an all-ones mask (one unpadded sequence)")
        return ref_attention(self, query, key, value, is_long)

    gm.GrugMoeAttention._flash_attention = attention


def ref_attention(self, query, key, value, is_long, qc=1024):
    """GrugMoeAttention._eager_attention (fp32 scores, -1e9 mask, softmax, cast to bf16, bf16 PV), query-chunked and
    with sliding layers reading only their window, so it runs at 62k tokens."""
    import torch
    _, S, _, _ = query.shape
    kr = self._repeat_kv_heads(key)
    vr = self._repeat_kv_heads(value)
    kf = kr.float()
    win = self.config.sliding_window
    out = torch.empty_like(query)
    for qs in range(0, S, qc):
        qe = min(S, qs + qc)
        ks = 0 if is_long else max(0, qs - (win - 1))
        sc = torch.einsum("bqhd,bkhd->bhqk", query[:, qs:qe].float() / math.sqrt(self.config.head_dim), kf[:, ks:qe])
        qpos = torch.arange(qs, qe, device=query.device).view(-1, 1)
        kpos = torch.arange(ks, qe, device=query.device).view(1, -1)
        allowed = kpos <= qpos
        if not is_long:
            allowed = allowed & (kpos >= qpos - (win - 1))
        sc.masked_fill_(~allowed.view(1, 1, qe - qs, qe - ks), -1e9)
        w = torch.softmax(sc, dim=-1).to(value.dtype)
        del sc
        out[:, qs:qe] = torch.einsum("bhqk,bkhd->bqhd", w, vr[:, ks:qe])
        del w
    return out, vr


def set_moe_eager(hf, eager):
    from skyrl_train.models import grug_moe as gm
    ex = gm._GRUG_EAGER_EXPERT_EXECUTION if eager else gm._GRUG_GROUPED_EXPERT_EXECUTION
    for layer in hf.model.layers:
        layer.mlp._expert_execution = ex


def logits_variants(h, W, W32, labels, chunk=2048):
    import torch
    import torch.nn.functional as F
    lb, lf, ent = [], [], []
    for s in range(0, h.shape[0], chunk):
        hs, ls = h[s:s + chunk], labels[s:s + chunk, None]
        x = F.linear(hs, W).float().log_softmax(-1)
        lb.append(x.gather(-1, ls)[:, 0])
        x = F.linear(hs.float(), W32).log_softmax(-1)
        lf.append(x.gather(-1, ls)[:, 0])
        ent.append(-(x.exp() * x).sum(-1))
    return torch.cat(lb), torch.cat(lf), torch.cat(ent)


def run_model(a, tag, model_path, set_path, vlabels, replay_labels, variants, pairs):
    import numpy as np
    import torch
    import skyrl_train.models  # noqa: F401  registers GrugMoeForCausalLM
    from skyrl_train.model_wrapper import HFModelWrapper

    items = sorted(load_set(set_path), key=lambda r: -(len(r["prompt_ids"]) + len(r["completion_ids"])))
    items = items[a.rank::a.world][: a.limit or None]
    t0 = time.time()
    model = HFModelWrapper(model_path, use_flash_attention_2=True, bf16=True, use_sample_packing=False,
                           use_grouped_mm=True, attn_backend="flash_attention_2", training_strategy="fsdp2")
    model = model.to("cuda").eval()
    hf = model.model
    install_patches(hf)
    nl = len(hf.model.layers)
    hook = hf.model.register_forward_hook(
        lambda m, i, o: setattr(CTX, "hidden", o.last_hidden_state[0, CTX.want_hidden[0]:CTX.want_hidden[1]].clone())
        if CTX.want_hidden else None)
    W = hf.lm_head.weight
    W32 = W.float()
    edges = torch.tensor(MARGIN_EDGES, device="cuda")
    print(f"[{tag}] rank {a.rank}: loaded in {time.time() - t0:.0f}s, {len(items)} turns, {nl} layers", flush=True)

    tok = {}            # per-token arrays, concatenated over turns
    per_turn = {}       # per-turn arrays
    ids = []

    def put(d, k, v):
        d.setdefault(k, []).append(v)

    with torch.no_grad():
        for ti, r in enumerate(items):
            ts = time.time()
            P, C = len(r["prompt_ids"]), len(r["completion_ids"])
            N = P + C
            seq = torch.tensor([r["prompt_ids"] + r["completion_ids"]], dtype=torch.long, device="cuda")
            mask = torch.ones_like(seq)
            labels = seq[0, P:]
            q0, q1 = P - 1, N - 1      # positions whose next-token logprob is a completion token

            def fwd():
                return model(seq, num_actions=C, attention_mask=mask)[0].float()

            def recorded(fn):
                CTX.record, CTX.sel, CTX.margin = True, [None] * nl, [None] * nl
                lp = fn()
                CTX.record = False
                return lp, torch.stack(CTX.sel, 1), torch.stack(CTX.margin, 1)   # [N, L, 4] uint8, [N, L] fp32

            lp, tv = {}, {}

            def timed(name, fn):
                torch.cuda.synchronize()
                t = time.time()
                v = fn()
                torch.cuda.synchronize()
                tv[name] = time.time() - t
                return v

            CTX.want_hidden = (q0, q1)
            lp["T0"], sel_t0, mar_t0 = timed("T0", lambda: recorded(fwd))
            CTX.want_hidden = None
            lp["bf16_rederived"], lp["fp32logits"], ent = logits_variants(CTX.hidden, W, W32, labels)
            CTX.hidden = None
            lp["T0b"] = timed("T0b", fwd)
            sets = {"T0": sel_t0}
            if "eager_moe" in variants:
                set_moe_eager(hf, True)
                lp["eager_moe"], sets["eager_moe"], _ = timed("eager_moe", lambda: recorded(fwd))
                set_moe_eager(hf, False)
            if "ref_attn" in variants:
                CTX.ref_attn = True
                lp["ref_attn"], sets["ref_attn"], _ = timed("ref_attn", lambda: recorded(fwd))
                CTX.ref_attn = False
            for lab in vlabels:
                p = os.path.join(a.vdir, f"route_{lab}", r["id"] + ".npy")
                if os.path.exists(p):
                    arr = np.load(p)
                    if arr.shape != (N, nl, 4):
                        raise ValueError(f"{p}: shape {arr.shape}, expected {(N, nl, 4)}")
                    sets[lab] = torch.from_numpy(arr).cuda()
            for lab in replay_labels + ["self"]:
                src = sel_t0 if lab == "self" else sets.get(lab)
                if src is None:
                    continue
                CTX.replay = [src[:, i, :].long().contiguous() for i in range(nl)]
                lp[f"replay_{lab}"] = timed(f"replay_{lab}", fwd)
                if lab == replay_labels[0] and "ref_attn" in variants:
                    CTX.ref_attn = True        # same forced routing, fp32-score attention: is the residual attention?
                    lp[f"replay_refattn_{lab}"] = timed("replay_refattn", fwd)
                    CTX.ref_attn = False
                CTX.replay = None
            # every variant gets an entry for every turn (NaN when it could not run) so the arrays stay aligned
            for k in variants + [f"replay_{lab}" for lab in replay_labels + ["self"]] + (
                    [f"replay_refattn_{replay_labels[0]}"] if "ref_attn" in variants else []):
                if k not in lp:
                    lp[k] = torch.full((C,), float("nan"), device="cuda")
            # routing pairs (missing: nflip 255, zero counts, has_<pair> = 0)
            srt = {k: v.sort(-1).values for k, v in sets.items()}
            mb = torch.bucketize(mar_t0.contiguous(), edges)   # [N, L] bin of T0's margin
            nb = len(MARGIN_EDGES) + 1
            for name, x, y in pairs:
                if x not in srt or y not in srt:
                    put(tok, f"nflip_{name}", np.full(C, 255, dtype=np.uint8))
                    put(per_turn, f"layerflips_{name}", np.zeros(nl, dtype=np.int64))
                    put(per_turn, f"marginhist_{name}", np.zeros((nb, 2), dtype=np.float32))
                    put(per_turn, f"has_{name}", np.array(0))
                    continue
                d = (srt[x] != srt[y]).any(-1)                 # [N, L]
                put(tok, f"nflip_{name}", d[q0:q1].sum(-1).to(torch.uint8).cpu().numpy())
                put(per_turn, f"layerflips_{name}", d.sum(0).cpu().numpy())
                tot = torch.bincount(mb.flatten(), minlength=nb)
                fl = torch.bincount(mb.flatten(), weights=d.flatten().float(), minlength=nb)
                put(per_turn, f"marginhist_{name}", torch.stack([tot.float(), fl.float()], 1).cpu().numpy())
                put(per_turn, f"has_{name}", np.array(1))
            put(tok, "minmargin", mar_t0[q0:q1].min(-1).values.cpu().numpy())
            put(tok, "ent", ent.cpu().numpy())
            put(tok, "pos", np.arange(P, N, dtype=np.int32))
            put(tok, "turn", np.full(C, ti, dtype=np.int32))
            for k, v in lp.items():
                assert v.shape[0] == C, (k, v.shape, C)
                put(tok, f"lp_{k}", v.cpu().numpy().astype(np.float32))
            put(per_turn, "npos", np.array(N))
            ids.append(r["id"])
            dd = (lp["T0"] - lp["T0b"]).abs().max().item()
            print(f"[{tag}] rank {a.rank}: {r['id']} len {N} comp {C} {time.time() - ts:.1f}s |T0-T0b|max {dd:.2e} "
                  f"|T0-self|max {(lp['T0'] - lp['replay_self']).abs().max().item():.2e} "
                  f"|T0-bf16re|max {(lp['T0'] - lp['bf16_rederived']).abs().max().item():.2e} "
                  f"secs " + " ".join(f"{k}={v:.1f}" for k, v in tv.items()), flush=True)
            del seq, mask, sets, srt, sel_t0, mar_t0, lp, mb
            torch.cuda.empty_cache()
    hook.remove()
    out = {"ids": np.array(ids)}
    for k, v in tok.items():
        out["tok_" + k] = np.concatenate(v)
    for k, v in per_turn.items():
        out["turn_" + k] = np.stack(v)
    np.savez_compressed(f"{a.out}.{tag}.part{a.rank}.npz", **out)
    print(f"[{tag}] rank {a.rank}: wrote {a.out}.{tag}.part{a.rank}.npz", flush=True)
    del model, hf, W, W32
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vdir", required=True, help="vllm_decomp output dir (route_<label>.r<rep>/<id>.npy)")
    ap.add_argument("--s3-model", required=True)
    ap.add_argument("--s3-set", required=True)
    ap.add_argument("--h9-model", default="")
    ap.add_argument("--h9-set", default="")
    ap.add_argument("--limit", type=int, default=0, help="turns per rank (0 = all)")
    ap.add_argument("--only", default="", help="s3 or h9")
    a = ap.parse_args()
    A, B, Cc, D, E = "s3_dp4ep_c4.r0", "s3_dp4ep_c4.r1", "s3_dp4ep_c1.r0", "s3_dp4ep_c1.r1", "s3_tp1_c1.r0"
    if a.only in ("", "s3"):
        run_model(a, "s3", a.s3_model, a.s3_set, vlabels=[A, B, Cc, D, E], replay_labels=[Cc, A, B, D, E],
                  variants=["eager_moe", "ref_attn"],
                  pairs=[("T0_vC", "T0", Cc), ("T0_vA", "T0", A), ("T0_vE", "T0", E), ("vA_vB", A, B),
                         ("vC_vD", Cc, D), ("vA_vC", A, Cc), ("vC_vE", Cc, E),
                         ("T0_eager", "T0", "eager_moe"), ("T0_refattn", "T0", "ref_attn")])
    if a.h9_model and a.only in ("", "h9"):
        hA, hB, hC = "h9_dp4ep_c4.r0", "h9_dp4ep_c4.r1", "h9_dp4ep_c1.r0"
        run_model(a, "h9", a.h9_model, a.h9_set, vlabels=[hA, hB, hC], replay_labels=[hC, hA, hB], variants=[],
                  pairs=[("T0_vC", "T0", hC), ("T0_vA", "T0", hA), ("vA_vB", hA, hB), ("vA_vC", hA, hC)])
    print("TRAINER_DECOMP_RANK_DONE", a.rank, flush=True)


if __name__ == "__main__":
    sys.exit(main())
