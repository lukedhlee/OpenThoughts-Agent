#!/usr/bin/env python3
"""Proxy for Horizon port gate 5: how far the RL trainer's log-probs sit from the rollout engine's, on GB200.

Gate 5 (ai_memory/active/horizon-port/objective.md) reads the trainer's step-1 `policy/tis/log_ratio_abs_mean` =
mask-weighted mean over response tokens of |log p_trainer - log p_rollout|. A real step 1 needs the R2E-Gym Daytona
pool. This script measures the same quantity without sandboxes, on the gate-2 set (96 R2E-Gym turns of the Stage-3
step-1888 policy with fixed completions, 142,817 completion tokens):

  trainer side  MarinSkyRL's own HFModelWrapper.forward (GrugMoeForCausalLM, flash_attention_2, bf16, native
                grouped_mm, no sample packing), i.e. the arm config's trainer.policy settings, one GPU per process;
  rollout side  (a) Horizon vLLM prompt logprobs from gate 2 (job 35865, DP4-EP, two repeats), and
                (b) the orig_logprobs Jupiter's vLLM returned when it sampled these tokens (the probe's rollouts).

Prints token-weighted mean |delta| per rollout side (the tis/log_ratio_abs_mean analogue), the per-turn bootstrap CI,
and vLLM's own repeat-to-repeat noise. Run inside a 1-node job:  torchrun-free, one process per GPU:
    for r in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$r python rl_logprob_parity.py score --rank $r --world 4 ... & done; wait
    python rl_logprob_parity.py compare --set S --vllm V --parts 'OUT.part*.jsonl'
"""
import argparse
import glob
import json
import math
import random
import sys
import time


def load_set(path):
    return [json.loads(l) for l in open(path)]


def score(a):
    import torch
    import skyrl_train.models  # noqa: F401  registers GrugMoeForCausalLM with AutoModelForCausalLM
    from skyrl_train.model_wrapper import HFModelWrapper

    items = load_set(a.set)
    items = sorted(items, key=lambda r: -(len(r["prompt_ids"]) + len(r["completion_ids"])))[a.rank::a.world]
    t0 = time.time()
    model = HFModelWrapper(a.model, use_flash_attention_2=True, bf16=True, use_sample_packing=False,
                           use_grouped_mm=True, attn_backend="flash_attention_2")
    model = model.to("cuda").eval()
    print(f"rank {a.rank}: loaded in {time.time() - t0:.0f}s, {len(items)} turns", flush=True)
    with open(f"{a.out}.part{a.rank}.jsonl", "w") as f, torch.no_grad():
        for r in items:
            ids = r["prompt_ids"] + r["completion_ids"]
            seq = torch.tensor([ids], dtype=torch.long, device="cuda")
            mask = torch.ones_like(seq)
            t = time.time()
            lp = model(seq, num_actions=len(r["completion_ids"]), attention_mask=mask)
            f.write(json.dumps({"id": r["id"], "lp": lp[0].float().tolist(), "secs": round(time.time() - t, 2)}) + "\n")
            f.flush()
            print(f"rank {a.rank}: {r['id']} len {len(ids)} {time.time() - t:.1f}s", flush=True)
            del seq, mask, lp
            torch.cuda.empty_cache()


def boot(per, n=2000, seed=0):
    """per = [(sum_abs, count)] per turn; token-weighted mean and its percentile CI over turns."""
    rng = random.Random(seed)
    stat = lambda xs: sum(s for s, _ in xs) / sum(c for _, c in xs)
    reps = sorted(stat([per[rng.randrange(len(per))] for _ in per]) for _ in range(n))
    return stat(per), reps[int(0.025 * n)], reps[int(0.975 * n) - 1]


def compare(a):
    items = {r["id"]: r for r in load_set(a.set)}
    trainer = {}
    for p in sorted(glob.glob(a.parts)):
        for l in open(p):
            r = json.loads(l)
            trainer[r["id"]] = r["lp"]
    vllm = {}
    for l in open(a.vllm):
        r = json.loads(l)
        if "id" in r:
            vllm.setdefault(r["id"], {})[r["rep"]] = r["lp"]
    ids = sorted(i for i in items if i in trainer)
    print(f"turns scored by the trainer: {len(ids)} of {len(items)}; tokens {sum(len(items[i]['completion_ids']) for i in ids)}")

    def side(name, get):
        per, big = [], 0
        for i in ids:
            ref = get(i)
            if ref is None:
                continue
            t = trainer[i]
            assert len(ref) == len(t), (i, len(ref), len(t))
            d = [abs(x - y) for x, y in zip(t, ref) if x is not None and y is not None]
            big += sum(x > 0.69 for x in d)
            per.append((sum(d), len(d)))
        m, lo, hi = boot(per)
        tok = sum(c for _, c in per)
        print(f"  {name:44s} mean|dlogp| {m:.5f}  95% CI [{lo:.5f}, {hi:.5f}]  turns {len(per)}  tokens {tok}  |d|>0.69: {big / tok:.2e}")
        return m

    print("trainer vs rollout (the tis/log_ratio_abs_mean analogue):")
    side("Horizon trainer vs Horizon vLLM rep 0", lambda i: vllm.get(i, {}).get(0))
    side("Horizon trainer vs Horizon vLLM rep 1", lambda i: vllm.get(i, {}).get(1))
    side("Horizon trainer vs Jupiter vLLM (sampling-time)", lambda i: items[i].get("orig_logprobs"))
    per = []
    for i in ids:
        v = vllm.get(i, {})
        if 0 in v and 1 in v:
            d = [abs(x - y) for x, y in zip(v[0], v[1]) if x is not None and y is not None]
            per.append((sum(d), len(d)))
    if per:
        m, lo, hi = boot(per)
        print(f"  {'context: Horizon vLLM rep 0 vs rep 1':44s} mean|dlogp| {m:.5f}  95% CI [{lo:.5f}, {hi:.5f}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--model", required=True)
    s.add_argument("--set", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--rank", type=int, default=0)
    s.add_argument("--world", type=int, default=1)
    c = sub.add_parser("compare")
    c.add_argument("--set", required=True)
    c.add_argument("--vllm", required=True)
    c.add_argument("--parts", required=True)
    a = ap.parse_args()
    {"score": score, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
