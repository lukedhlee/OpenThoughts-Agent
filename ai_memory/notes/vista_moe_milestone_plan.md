# Vista MoE — milestone plan (Luke)

The original five-step ladder from gsm8k smoke to full agentic RL, plus the TaskTrove-vs-AgentTrove-vs-
held-out-evals distinction (which artifact plays which role).
Read for the framing of what trains on what and what measures success; the ladder itself is superseded
by [[r2egym_grpo_plan]].

**Major milestone:** RL `Qwen/Qwen3-30B-A3B` on Vista with Harbor + Daytona on coding/agentic tasks → lift on **SWE-Bench Verified / OT-TB Lite / TB2**.

**Immediate goal:** land gsm8k GRPO smoke (job `842670`) — prove MoE train loop only.

## TaskTrove vs evals

| Artifact | Role |
|----------|------|
| **TaskTrove** (`open-thoughts/TaskTrove`) | **Train / datagen task pool** — Harbor task binaries (gzip tars) with verifiers. Source for agentic RL rollouts. Not the PI scoreboard. |
| **AgentTrove** | Traces from running models *on* TaskTrove (SFT / analysis). |
| **SWE-Bench Verified, OT-TB Lite, TB2** | **Held-out evals** — measure success after RL. |

## Ladder

1. gsm8k STANDARD GRPO (in flight) → MoE EP/FSDP green  
2. Tiny agentic Harbor+Daytona smoke (≤8 nodes) → verifier reward moves  
3. Scale to ~24-node disagg on `gh`  
4. Train on TaskTrove coding subsets (e.g. SWE-ish / pymethods / swegym)  
5. Eval on PI benches; iterate data/hparams if flat  
