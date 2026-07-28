#!/usr/bin/env python3
"""Paired probe: is our GSM8K eval score a generation-budget truncation artifact?

Background
----------
The 6-node Jupiter GRPO run (`jupiter_moe30b_gsm8k_grpo_6n_fast_50step_eval5`,
job 1042857) measured step-0 `pass_at_1 = 0.4511` on the 1319-example GSM8K test
split, against 78.2% zero-shot / 91.8% 4-shot-CoT reported for Qwen3-30B-A3B.
That run generated with `max_generate_length: 1024` and thinking left ON, and its
own metrics show the failures pinned at the ceiling:

    generate/max_num_tokens            = 1024 on EVERY step
    generate/avg_tokens_zero_rewards   ~ 950   (failures, at the cap)
    generate/avg_tokens_non_zero_rewards ~ 640 (successes, well under)

Hypothesis: responses are truncated mid-`<think>` before emitting the required
`#### <answer>`, so correct reasoning scores 0 and the metric mostly measures
"did you finish in budget", not math.

Design
------
ONE generation pass at a generous budget, then score the SAME rollouts under
several conditions. Because every condition reads the same samples, the
comparison is exactly paired -- no sampling noise between arms:

  strict @ full     -- the fixed-budget number
  strict @ 1024     -- same rollouts truncated to 1024 TOKENS, reproducing the
                       old condition (vLLM simply stopped at 1024 there, so
                       slicing token ids is a faithful emulation)
  flexible @ full   -- last-number extraction; separates formatting from math

Scoring uses `skyrl_gym.envs.gsm8k.utils.compute_score` so the scorer is
bit-identical to the one the RL run used -- see `load_compute_score`, which
falls back to executing that same file directly when the package import trips
over an unrelated missing dependency. No reimplemented copy on purpose: a
divergent scorer would silently invalidate the comparison.

Inference only. Loads no optimizer, writes no checkpoint, touches no Daytona.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet", required=True, help="GSM8K validation parquet (prompt + reward_spec.ground_truth)")
    p.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    p.add_argument("--output", required=True, help="Where to write the results JSON")
    p.add_argument("--max-generate-length", type=int, default=4096, help="Generous budget for the single pass")
    p.add_argument(
        "--truncate-at",
        type=int,
        nargs="+",
        default=[1024, 2048],
        help="Token budgets to re-score the same rollouts at (emulates a smaller max_generate_length)",
    )
    # Sampling params mirror the RL run's gsm8k config exactly (temperature 0.7,
    # top_p 0.95, no top_k, eval_n_samples_per_prompt=1) so the arms differ ONLY
    # in generation budget.
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tensor-parallel-size", type=int, default=4)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument(
        "--moe-backend",
        default="triton",
        help="vLLM MoE kernel backend. Default 'triton' on purpose: with 'auto', vLLM picks a "
        "FlashInfer CUTLASS path that JIT-builds 183 nvcc units at startup. That build failed "
        "(exit 9) and hung the engine until walltime on job 1085876. Triton needs no runtime "
        "CUTLASS build. Set 'auto' to restore stock behaviour.",
    )
    p.add_argument("--limit", type=int, default=0, help="Only the first N rows (0 = all); smoke-test knob")
    p.add_argument(
        "--save-samples",
        type=int,
        default=25,
        help="How many full (prompt, response, verdict) records to keep for eyeballing",
    )
    p.add_argument(
        "--scorer-path",
        default=os.environ.get("GSM8K_SCORER_PATH", ""),
        help="Explicit path to skyrl_gym/envs/gsm8k/utils.py. Only needed if the package "
        "import fails and the file isn't findable via PYTHONPATH.",
    )
    return p.parse_args()


def load_compute_score(explicit_path: str = "") -> tuple[Callable[..., float], str]:
    """Return skyrl_gym's `compute_score`, however we can reach it.

    Prefer the normal package import. It fails in our RL venv because
    `skyrl_gym/__init__` eventually imports `skyrl_gym.tools.sql`, which needs
    `func_timeout` -- a dependency of the SQL tool group that has nothing to do
    with GSM8K and isn't installed. So fall back to executing the SAME
    `utils.py` directly, which sidesteps the package `__init__` chain.

    This is deliberately NOT a reimplementation: it is byte-for-byte the file
    the RL run scored with (it imports only `re`), so the comparison stays
    valid. Hard-fail if neither route works.
    """
    try:
        from skyrl_gym.envs.gsm8k.utils import compute_score  # noqa: PLC0415

        return compute_score, "package import"
    except Exception as exc:  # ImportError, or a transitive dep blowing up
        first_error = exc

    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if entry:
            candidates.append(Path(entry) / "skyrl_gym" / "envs" / "gsm8k" / "utils.py")

    for cand in candidates:
        if not cand.is_file():
            continue
        spec = importlib.util.spec_from_file_location("skyrl_gsm8k_utils", cand)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fn = getattr(module, "compute_score", None)
        if fn is not None:
            return fn, f"direct file load: {cand}"

    print(
        "FATAL: cannot obtain skyrl_gym's gsm8k compute_score.\n"
        f"       package import failed with: {type(first_error).__name__}: {first_error}\n"
        f"       and no usable utils.py among: {[str(c) for c in candidates] or '(none)'}\n"
        "       Pass --scorer-path /path/to/skyrl_gym/envs/gsm8k/utils.py, or add\n"
        "       .../MarinSkyRL/skyrl-gym to PYTHONPATH. Refusing to substitute a local\n"
        "       copy of the scorer -- that would invalidate the comparison.",
        file=sys.stderr,
    )
    raise SystemExit(2)


def load_rows(parquet: str, limit: int) -> list[dict]:
    import pandas as pd

    df = pd.read_parquet(parquet)
    if limit:
        df = df.iloc[:limit]
    rows = []
    for _, r in df.iterrows():
        # `prompt` is a list of chat messages; `reward_spec.ground_truth` is the
        # bare number string that compute_score compares against.
        prompt = r["prompt"]
        prompt = [dict(m) for m in prompt]
        rows.append({"prompt": prompt, "ground_truth": r["reward_spec"]["ground_truth"]})
    return rows


def summarize(name: str, scores: list[float]) -> dict:
    n = len(scores)
    correct = int(sum(1 for s in scores if s > 0))
    acc = correct / n if n else 0.0
    # Binomial SE on the pass rate; the paired arms share samples so this is a
    # per-arm figure, not the SE of a between-arm difference.
    se = (acc * (1 - acc) / n) ** 0.5 if n else 0.0
    return {"arm": name, "n": n, "correct": correct, "accuracy": acc, "se": se, "ci95": 1.96 * se}


def main() -> int:
    args = parse_args()

    # The EXACT scorer the RL run used. Hard-fails rather than reimplement.
    compute_score, scorer_origin = load_compute_score(args.scorer_path)
    print(f"[probe] scorer loaded via {scorer_origin}", flush=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = load_rows(args.parquet, args.limit)
    print(f"[probe] loaded {len(rows)} rows from {args.parquet}", flush=True)

    # Default Qwen3 chat template => thinking ON, matching the RL run (its gsm8k
    # config sets no custom template and no enable_thinking kwarg).
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [
        tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True) for r in rows
    ]
    print(f"[probe] example templated prompt:\n{prompts[0][:600]}\n", flush=True)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        seed=args.seed,
        enable_prefix_caching=True,
        trust_remote_code=True,
        moe_backend=args.moe_backend,
    )
    sp = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_generate_length,
        seed=args.seed,
    )

    t0 = time.time()
    outs = llm.generate(prompts, sp)
    gen_secs = time.time() - t0
    print(f"[probe] generation done in {gen_secs:.1f}s", flush=True)

    budgets = sorted(set(args.truncate_at))
    arms: dict[str, list[float]] = {"strict@full": [], "flexible@full": []}
    for b in budgets:
        arms[f"strict@{b}"] = []

    lengths: list[int] = []
    samples: list[dict] = []

    for i, out in enumerate(outs):
        comp = out.outputs[0]
        tok_ids = list(comp.token_ids)
        text = comp.text
        gt = rows[i]["ground_truth"]
        lengths.append(len(tok_ids))

        s_full = compute_score(text, gt, method="strict")
        f_full = compute_score(text, gt, method="flexible")
        arms["strict@full"].append(s_full)
        arms["flexible@full"].append(f_full)

        per_budget = {}
        for b in budgets:
            # Faithful emulation: the old run's engine stopped emitting at `b`
            # tokens, so slice token ids and decode what survived.
            trunc_text = tok.decode(tok_ids[:b], skip_special_tokens=True) if len(tok_ids) > b else text
            s_b = compute_score(trunc_text, gt, method="strict")
            arms[f"strict@{b}"].append(s_b)
            per_budget[b] = s_b

        if len(samples) < args.save_samples:
            samples.append(
                {
                    "index": i,
                    "ground_truth": gt,
                    "n_tokens": len(tok_ids),
                    "strict_full": s_full,
                    "flexible_full": f_full,
                    "strict_truncated": per_budget,
                    "finish_reason": comp.finish_reason,
                    "response": text,
                }
            )

    n = len(lengths)
    at_cap = sum(1 for L in lengths if L >= args.max_generate_length)
    over = {b: sum(1 for L in lengths if L > b) for b in budgets}

    # Length split by correctness under the FULL budget -- mirrors the
    # generate/avg_tokens_{zero,non_zero}_rewards metrics from the RL run.
    ok = [L for L, s in zip(lengths, arms["strict@full"]) if s > 0]
    bad = [L for L, s in zip(lengths, arms["strict@full"]) if s <= 0]
    mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0  # noqa: E731

    results = {
        "model": args.model,
        "parquet": args.parquet,
        "n_examples": n,
        "generation_seconds": gen_secs,
        "config": {
            "max_generate_length": args.max_generate_length,
            "max_model_len": args.max_model_len,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "tensor_parallel_size": args.tensor_parallel_size,
            "thinking": "default chat template (ON for Qwen3)",
            "moe_backend": args.moe_backend,
            "scorer_origin": scorer_origin,
        },
        "arms": [summarize(k, v) for k, v in arms.items()],
        "tokens": {
            "mean": mean(lengths),
            "max": max(lengths) if lengths else 0,
            "min": min(lengths) if lengths else 0,
            "at_full_cap": at_cap,
            "at_full_cap_frac": at_cap / n if n else 0.0,
            "over_budget": {str(b): over[b] for b in budgets},
            "over_budget_frac": {str(b): (over[b] / n if n else 0.0) for b in budgets},
            "mean_tokens_correct": mean(ok),
            "mean_tokens_incorrect": mean(bad),
        },
        "samples": samples,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(results, fh, indent=2)

    print("\n" + "=" * 72)
    print(f"GSM8K BUDGET PROBE -- {args.model}  (n={n})")
    print("=" * 72)
    for a in results["arms"]:
        print(f"  {a['arm']:>18}  {a['accuracy']*100:6.2f}%  ({a['correct']}/{a['n']})  +/-{a['ci95']*100:.2f}")
    print("-" * 72)
    t = results["tokens"]
    print(f"  tokens: mean={t['mean']:.0f}  max={t['max']}  at {args.max_generate_length} cap: "
          f"{t['at_full_cap']} ({t['at_full_cap_frac']*100:.1f}%)")
    for b in budgets:
        print(f"  would exceed {b:>5}: {over[b]} ({over[b]/n*100:.1f}%)")
    print(f"  mean tokens  correct={t['mean_tokens_correct']:.0f}  incorrect={t['mean_tokens_incorrect']:.0f}")
    print("=" * 72)
    print("\nInterpretation: if strict@1024 lands near 0.451 while strict@full is far")
    print("higher, the old curve was a truncation artifact. If flexible@full is much")
    print("higher than strict@full, the residual gap is answer FORMATTING, not math.")
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
