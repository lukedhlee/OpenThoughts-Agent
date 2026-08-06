# Marianna-parity sheet (authoritative, from her actual launch script)

Source: `ai_memory/reference/marianna_rl_launch.sh` — verbatim (partial) copy of
her r2egym RL launch script, pasted by the operator 2026-08-07 after a prior
session failed to persist it. Her cluster checkout is NOT readable by lee27;
this file is the durable record. **Check here BEFORE claiming any of her
settings are unknown.**

## The numbers that keep getting lost

| knob | her value | our band/smoke value | note |
|---|---|---|---|
| **Trial timeout** | **`TIMEOUT_SEC=1800`** (30 min) | 3600 (target; broken at 900 as of 08-07, see gotchas) | Her comment: DeepSWE reference = 5400s; "raise for long-context runs, e.g. 3600". Our agent thinks (slower per turn), so 3600 is inside her own recommended range, NOT an anti-parity choice. The binding parity constraint is the STEP cap, not wall-clock. |
| Agent step cap | `MAX_EPISODES=50` | 50 (matched) | mirrors DeepSWE agent.max_steps=50 |
| Agent | `terminus-structured` | OpenCode (operator's choice) | absolute band % may differ; criterion = within-group variance exists |
| Model | `DCAgent--g1_diverse_tezos_100k_8b` @ a5c4fa2b | same model | |
| TP / engines | TP=4, 8 engines (1 engine/node) | TP=1 × 4 engines | same GPUs, different sharding; deliberate |
| Train batch | 64 | — | |
| LR | 8e-6 (DeepSWE-seqmean ref is 1e-6) | — | |
| MAX_STEPS | 61 (= 60 updates; trainer stops one early) | — | |
| Dataset | `r2egym_swebenchfmt_pool` (full 4,578, swebench-fmt) | our allowlist 4,469 of the same pool | she trains on the RAW pool while g1-pass-rate filtering narrows it |
| Learnable band | **358 tasks** (@ bs=64 ≈ 5.6 unique steps/epoch) | to be reproduced | ≈8% of 4,578 at 0<pass@8<1 for g1 — NOT 36%-solvable; see note below |
| Eval | swebench-verified-random-100 (DCAgent2 split) + r2egym_learnable_heldout + format-control swebench_r2egymfmt | — | in-loop eval disabled; decoupled watch_and_eval per HF ckpt every 5 steps |
| KL loss | script overrides DeepSWE ref to use_kl_loss=true (ref = false) | — | her comment: alignment tool, capability RL drops it |
| Dynamic sampling | off by default (needs sync trainer / colocate_all=true) | — | |

## Band-number disambiguation (two different figures circulate)

- **358 learnable** = tasks with 0 < pass@8 < 1 for g1 on her runs (≈8% of pool)
  — the RLOO/GRPO-gradient-bearing set. This is the number her script banks on.
- **~36%** = the earlier-quoted per-trial solve rate figure. Do not conflate:
  reproducing "her band" means reproducing a non-degenerate learnable set in
  the hundreds, not a 36% pass rate.

## Why our setup was "lossy" here (operator's criticism, accepted)

The prior session's parity block in `gen_band_yaml.py` captured steps/sampling
but not TIMEOUT_SEC, and the script itself lived only in an unreadable dir +
a chat transcript. Rule going forward: any external config we depend on gets a
verbatim copy under `ai_memory/reference/` the moment we see it.
