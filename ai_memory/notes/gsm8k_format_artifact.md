# GSM8K `pass_at_1 ≈ 0.45` is a FORMAT artifact, not math ability

Paired evidence that the whole GSM8K plateau was answer formatting, not reasoning: `strict@1024 = 41.6%`
vs `flexible@4096 = 90.67%` on identical rollouts, with the decomposition and root cause.
Read before citing any GSM8K number from our runs, and before interpreting an r2egym reward curve —
the transferable lesson at the bottom applies there too.

**Measured 2026-07-29, Jupiter job `1086698`** (n=1319, full GSM8K val, paired design).
Probe: `scripts/analysis/gsm8k_budget_probe.py` + `eval/jupiter/gsm8k_budget_probe.sbatch`.
Raw results: `/e/scratch/reformo/lee27/experiments/jupiter_gsm8k_budget_probe/results/probe_1086698.json`.
Parent: [[handoff]]. Related env landmines: [[jupiter_cluster]].

## The numbers

One generation pass at 4096, then the SAME rollouts re-scored at smaller budgets, so the arms are
exactly paired (no sampling noise between them). Scorer is skyrl_gym's own
`envs/gsm8k/utils.compute_score` — byte-identical to the RL run's.

| arm | score | ±95% |
|---|---|---|
| `strict@1024` | **41.62%** (549/1319) | 2.66 |
| `strict@2048` | 51.18% (675/1319) | 2.70 |
| `strict@4096` | 52.92% (698/1319) | 2.69 |
| **`flexible@4096`** | **90.67%** (1196/1319) | 1.57 |

```
tokens: mean=1183   at 4096 cap: 30 (2.3%)
exceed 1024: 554 (42.0%)    exceed 2048: 161 (12.2%)
mean tokens  correct=829   incorrect=1581
```

**`strict@1024 = 41.6%` reproduces the 6-node run's step-0 `pass_at_1 = 0.4511`** (that run used
`max_generate_length: 1024`). Residual ~3.5 pts = prompt-template/sampling differences.

## Decomposition

| effect | size | what it is |
|---|---|---|
| answer formatting | **+37.7** | right answer, not in the scored format |
| truncation (1024→4096) | +11.3 | 42% of responses don't fit in 1024 |
| **true ability** | **90.67%** | vs Qwen's published **91.81%** 4-shot ⇒ consistent |

Format dominates truncation by >3×.

## Root cause — the model ignores the requested format

The prompt DOES ask for it:
> `... Let's think step by step and output the final answer after "####".`

The strict extractor is:
```python
re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)   # first match wins; needs exactly one space
```

But inspecting `strict=0, flexible=1` samples (8 of 25 saved): **`has ####: False` in every case.**
The model answers in `\boxed{...}` with markdown `###` headers, and `finish_reason=stop` — it completed
normally, got the right answer, and formatted it its own way:
```
gt=20   TAIL: $$\boxed{20}$$
gt=260  TAIL: ### Final Answer:  $$\boxed{260}$$
gt=160  TAIL: ### ✅ Final Answer: $$\boxed{160}$$
```
⇒ NOT scorer brittleness (no `#### $18`-style near-misses), NOT truncation for this subset.
**A single user-turn instruction does not override Qwen3's post-trained `\boxed{}` math habit.**

## What this means

- **The 0.4511 → 0.5057 lift over 45 steps was most plausibly the model learning FORMAT COMPLIANCE**
  (override `\boxed{}`, emit `####`), not arithmetic. It had the arithmetic at ~91% from step 0.
  Do not report that curve as evidence that MoE GRPO improves reasoning.
- **The strict ceiling at 1024 is ~52.9%, not ~90%** — that's the headroom the run was climbing toward.
- **GSM8K is a poor vehicle for demonstrating MoE RL.** Fix the format and step-0 jumps to ~85–90%,
  leaving almost no headroom. Reinforces moving to **r2egym**, whose binary test-pass reward cannot be
  satisfied by formatting.

## If we ever rerun GSM8K GRPO

| knob | change | why |
|---|---|---|
| `max_generate_length` | 1024 → **2048** | 42% exceed 1024; only 12% exceed 2048; 4096 buys +1.7 pts for 2× cost |
| prompt | **4-shot CoT with `####` exemplars** | demonstration overrides habit; it's also what Qwen's 91.81% uses |
| scorer | **keep `strict`** | `flexible` is last-number extraction ⇒ gameable as a reward, and rewards luck |
| `enable_thinking` | consider `false` | thinking is what blows the budget (incorrect: 1581 tok vs correct: 829) |

⚠ **`flexible@4096 = 90.67%` is an UPPER BOUND.** It takes the last number in the response, so some
credit is coincidence. False-positive rate NOT measured.

## Transferable lesson

Before trusting any reward curve, check what fraction of failures are **non-substantive** (format,
truncation, malformed output) rather than capability. For **r2egym** the structurally identical failure
is a rollout that fixes the bug but emits a malformed patch, or runs out of budget mid-task. **Measure
this before interpreting r2egym's `avg_raw_reward`.**
