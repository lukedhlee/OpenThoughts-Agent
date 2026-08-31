# Snowball RL — first-release goal proposals (DRAFT)

Status: draft for Luke's edit → Ben review, week of 2026-08-25.
Basis: 5-agent literature sweep of Feb–Aug 2026 work (agentic RL · RLVR data recipes · MoE-RL/weak-to-strong/collateral · reception taxonomy · prompt-opt × RL), run 2026-08-21. Key references grouped at the bottom.

---

## 1. Framing

**We agree with Ben's premise.** At ≤3B-active scale, distillation is the dominant tool for *injecting* capability. RL's comparative-advantage region, where every proposal below lives:

1. verifiable-reward domains where no teacher traces exist;
2. multi-turn interactive settings where SFT on static traces suffers compounding error;
3. behavior shaping with controlled collateral;
4. closing a *measured* reward gap to a stronger teacher.

**The 2026 evidence refines (not refutes) the bearish view.** RL on top of *strong* distilled SFT adds double-digit pass@1 on competition math (AceReason-Nemotron 1.1, ICLR'26: +10.6 AIME24 / +16.4 AIME25), and stronger SFT bases finish higher after RL — complements, not substitutes. Sequential distill-then-RL beats interleaved SFT+RL methods decisively. But **vanilla RLVR sharpens rather than expands** (pass@1 up, pass@k flat); genuine capability expansion requires specific machinery (teacher hints at the 0/k boundary, saturation gating, prolonged negative-focused training). On held-out *agentic* benchmarks, historical RL deltas are ~1–2 points across the published literature; the levers that convert dev-set gains into benchmark gains are environment realism/diversity and difficulty calibration. (SETA externally benchmarks OpenThoughts-Agent-v1-RL at 4.9% TB2.0 and attributes the small delta to environment quality — but our own paper's RL section has internal known issues, so the v1 datapoint should not be treated as a clean baseline nor SETA's attribution as the established explanation for it.)

**Shared evaluation contract** (applies to every proposal; this is what makes the release credible):

- pass@1 **and** pass@64/256 deltas vs the post-SFT checkpoint (defuses the "RLVR just sharpens" reviewer);
- a **GEPA-optimized-prompt baseline in our exact harness** next to every headline RL number (GEPA is an ICLR'26 oral; report rollouts *and* tokens — imported prompts don't transfer across scaffolds);
- **fixed agent harness across arms** (Qwen3-Coder-Next: cross-scaffold transfer is weak and will swamp effects);
- **ID/OOD split** on our existing eval infra (KARL: distillation wins ID, RL wins OOD — the split is where RL's contribution is legible);
- the **collateral report card** (P1 below) applied pre/post on every released checkpoint.

## 2. Blog-worthiness rubric (extracted from what actually landed, Feb–Aug 2026)

1. One memorable headline number on a benchmark people track (small-beats-big / cheap-beats-expensive framings).
2. A dose-response curve, not just an endpoint — sweep one knob.
3. Boring algorithm, stated as a finding; spend novelty on data/environments/measurement.
4. At least one corrective/surprising sub-result (the quotable thing).
5. Full-stack release: environments + data + code + weights + pitfalls log.
6. Cost honesty (GPU-hours, $-per-task).
7. A transfer section (harness/domain swap table).
8. Isolate one variable cleanly.

Reference exemplar: Tmax (Ai2, arXiv:2606.23321) — 14.6K difficulty-controlled envs + outcome-only RL, 9B → 27% TB2.0, full release, cost-honest. 16 models / 12 datasets derived from it within two months.

---

## 3. Proposals

### P0 — Stable MoE-GRPO foundation (enabler; weeks 1–3; mandatory)

**What:** implement rollout routing replay (R3) + per-token rollout-vs-trainer KL telemetry + the GLM-5 consistency checklist (token-in-token-out, deterministic top-k, frozen router during RL, stale-rollout dropping) in MarinSkyRL; run an instrumented ablation — current TIS vs routing replay vs router-aware IS (RSPO) — on the running Qwen3-Coder-30B-A3B agentic arm.
**Why:** three independent report threads converge on this prerequisite. TIS alone is insufficient for MoE (R3 beats it; VCPO documents seq-level TIS collapsing at 2-step policy lag in multi-turn tool RL). Every published MoE-RL stability result is single-turn math; **no agentic MoE-RL stability study exists**, and Snowball's ~3% active ratio is untested territory. Without this, every recipe effect downstream is confounded with instability. Ring-mini-2.0 (16B/1.4B-active, RLVR'd successfully) is the existence proof at our activation scale.
**Deliverable:** first agentic MoE-RL stability study + pitfalls log (checklist items 3, 4, 5).
**Gate:** stable reward curves without collapse over a full agentic run on the 30B arm; telemetry shipped as standing infra.
**Risk:** wiring replay through vLLM↔Megatron is the engineering unknown; fallback is router-aware IS (engine-agnostic, cheaper to integrate).

### P1 — Behavior recipe cards + the collateral report card (anchor; Ben's example 2)

**What:** 2–3 well-scoped behaviors, each shipped as a recipe card (reward spec + data + config + effect size + collateral profile), reproduced on two bases (Qwen3-Coder-30B now; Snowball post-SFT when it lands). Candidate behaviors: (a) write-a-meaningful-failing-test-first, (b) self-verify/reflect before submit, (c) grounded tool/search use. Reward pattern: ReflexiCoder's format-gated composite (binary structural gate × outcome reward × efficiency decay) — every gate checkable from the Harbor trajectory, no LLM judge.
**The collateral report card** (the second half of the deliverable, and the part with no prior art as a bundle): KL(post‖pre) on the task distribution (RL's Razor, ICLR'26 — the headline forgetting predictor), the pass@k fan, prior-task behavioral KL, automated generation-diffing with statistical validation (arXiv:2605.05090) bolted onto our existing `analyze-rl-behavior` pipeline, and a fixed multi-benchmark regression panel.
**Why feasible:** it is our existing stack end-to-end; the report card can start **this week on checkpoints we already have**, before Snowball arrives. Why exciting: ReflexiCoder is single-turn code-gen — the multi-turn agentic version with real collateral accounting is the publishable delta, and Nemotron-Cascade 2 shows collateral regression from scoped RL is unsolved even at NVIDIA scale.
**Headline shape:** "behavior X: 10→90% adoption, zero regression across N benchmarks" + a dose-response sweep of gate strength or data dose (rubric items 1, 2, 4, 8).
**Gate:** ≥1 behavior at high adoption with flat collateral panel by week 4; report card v0 on 2 existing checkpoints by week 2.
**Risk:** a behavior may be adopted but never move any score anywhere — mitigate by picking behaviors with documented score payoff (self-verification has published resolve-rate gains) and reporting score movement as the secondary endpoint.

### P2 — Weak-to-strong: the 3-arm gap-closing study (Ben's example 3)

**What:** given a stronger teacher's reward on a fixed agentic task distribution, how much of the gap does the student close via (i) selective/gated on-policy distillation (SOD/SAGE-OPD-style), (ii) GRPO on verifiable rewards, (iii) TGPO-style hybrid (teacher-guided exploration + verifiable reward)? Fixed harness; report fraction-of-gap-closed + the ID/OOD transfer matrix.
**Teacher constraint (load-bearing):** logit-level arms require a shared tokenizer → **Qwen3.5-122B-A10B**, not GLM-5.2 (GLM usable only for reward-level matching or trace SFT). Student: Qwen3-Coder-30B-A3B first, Snowball second.
**Why:** DeepSeek-V4's whole post-training is "RL creates capability in domain teachers, on-policy distillation transfers it" — Ben's example 3 validated at frontier scale — but **nobody has published the controlled 3-arm comparison with an ID/OOD split**. That slot is open and 6–8-week-shaped. Cheap extra arm: Direct-OPD shows a pre/post-RL checkpoint *pair* is itself a transferable teacher — we already own such pairs.
**Gate:** measurable gap-closure trend by week 4 in at least one arm.
**Risk:** multi-turn OPD is the least mature ingredient (uniform dense OPD breaks in multi-turn; supervision must be gated per-step); cross-thinking-pattern mismatch can make a stronger teacher fail (Rethinking-OPD) — SFT cold-start on teacher traces first. Agentic RL at scale is also our flakiest pipeline: this is the stretch track, not the anchor.

### P3 — Non-agentic: boundary-aware teacher-hint RLVR (Ben's example 1, reframed)

**What:** on post-SFT Snowball, standard GRPO with three evidence-backed data levers: (1) **teacher hints injected only on problems where the student is 0/k, then GRPO consolidation** — the one recipe with pass@256 receipts (+10.3 over vanilla RLVR); (2) online difficulty control steering rollout pass rates toward ~50% via prefix sampling (critical for a 2B-active student facing many 0/k problems on AIME-class data); (3) curated ~20% subset + rollout depth over breadth (~2x compute efficiency).
**Reframe of the target:** "match Qwen3-Next-80B-A3B-Thinking (AIME25 87.8 / GPQA-D 75.9)" has failure baked in — RL elicits, rarely injects, and the SFT checkpoint quality is outside our control. Instead: **close the gap on AIME/math** (parity with Qwen3-30B-A3B-Thinking's mid-80s is aggressive but defensible — math is where all the double-digit RL results live), with **measured-not-promised movement on GPQA** from an explicit verifiable-science data slice (math RL demonstrably does not transfer to knowledge domains; the science slice is the main data-engineering lift; no ≤3B-active model has published GPQA ~76).
**Narrative:** "distillation and RL compose at the example level" — engages the bearish premise head-on instead of dodging it.
**Gate:** dose-response curve (hint fraction / difficulty targeting) with pass@1 *and* pass@k both moving by week 4.
**Risk:** wholly dependent on Snowball post-SFT arrival and quality → sequence behind P0/P1, or dry-run the recipe on a Qwen base.

### P4 — Prompt-opt × RL: mandatory baseline + optional flagship

**Baseline (mandatory, cheap):** GEPA-optimized prompt in our exact harness beside every headline RL number; report rollout *and* token accounting.
**Screening (one day):** the OPSD linear law — the prompt-conditioned vs bare pass-rate gap predicts internalization payoff — run before any distillation GPU spend.
**Optional flagship (only if P0+P1 land by week 4):** P²O-agentic — detect dead (all-zero-reward) task groups in agentic GRPO → GEPA-evolve scaffolds until pass rates enter the learnable band → context-distill into weights with No-Context Anchoring regularization → continue GRPO. Every component is separately 2026-validated; **unpublished on terminal-agent tasks with an MoE student.**
**Instrumented failure modes:** hint-reliance (HiLL metric: no-hint success on previously-hinted tasks), context-induced degradation post-distillation (eval with and without the old scaffold), judge-reward hacking onset (CHERRL) if rubric rewards are ever used.

---

## 4. Recommended portfolio (6–8 weeks)

- **Weeks 1–3:** P0 (mandatory, unblocks everything) ∥ P1 collateral report card on existing checkpoints ∥ P4 baseline + OPSD screening (cheap).
- **Weeks 2–8:** P1 behavior-training arms as the **anchor**. Choose **one** of P2/P3 as the second track at week 2, based on Snowball post-SFT arrival + quality and compute: Snowball late or weak → P2 on Qwen3-Coder; Snowball on time and math-capable → P3.
- **Stretch:** P²O-agentic flagship only if the week-4 gates pass.
- Compute is shared, not competing: P0's runs are P1/P2's substrate; the report card is the measurement layer over everything.
- Release = recipes + environments/data + code + weights + pitfalls log + cost accounting (rubric items 5–6), evaluated under the shared contract in §1.

## 5. Per-goal skeleton (to fill per proposal before the review)

Deliverable artifact · quantitative success gate · what-we-control vs what-we-assume · week-2 / week-4 milestones with kill criteria · compute ask (node-days, cluster).

Milestone spine: **wk 2** — P0 ablation readout; report card v0; screening numbers; second-track decision. **wk 4** — first behavior adoption curve; second-track go/no-go; P²O go/no-go. **wk 6–8** — reproduction on second base (Snowball if available); writeup + release packaging.

---

## 6. Key references (by theme)

**Terminal/agentic RL recipes:** Tmax 2606.23321 · Endless Terminals 2601.16443 · SETA 2607.10891 · Nemotron-Terminal 2602.21193 · Recursive Synthesis 2608.05466 · Agent World Model 2602.10090 · ReflexiCoder 2603.05863 · Qwen3-Coder-Next 2603.00729 · GLM-5 2602.15763 · DeepSeek-V4 2606.19348 · Nemotron-Cascade 2 2603.19220 · Demystifying long-horizon RL 2603.21972 · Generalization tax 2601.18217 · KARL 2603.05218.
**RLVR data + RL-vs-distillation:** AceReason 1.1 2506.13284 · RL vs distillation 2505.14216 · Pre/mid/RL interplay 2512.07783 · SFT-then-RL 2604.23747 · Two-stage boundary view 2510.04028 · Diversity collapse/BBG 2606.15455 · Curriculum-beyond-base 2606.22317 · DEPO 2509.01321 · IRDS 2605.28247 · Pass-rate control 2605.05112 · CoScale-RL 2601.14695 · One-sample 2601.03111 · Breaking Barriers 2506.19733 · TAC 2606.25178.
**MoE stability:** R3 2510.11370 · PR2 2606.00395 · RSPO 2510.23027 · Qwen stabilizing-practices 2512.01374 · VeXact 2605.14220 · AIS 2605.13907 · VCPO 2602.17616 · Ring-mini-2.0 (Ant blog).
**Weak-to-strong / OPD:** Direct-OPD 2607.05394 · W2S-OPD 2607.26246 · TGPO 2605.13230 · SOD 2605.07725 · SAGE-OPD 2606.19659 · Rethinking OPD 2604.13016 · REOPOLD 2603.11137.
**Collateral:** RL's Razor 2509.04259 · Side-effect audits 2605.05090 · CPO 2607.04364 · Circuit vulnerability 2605.28860 · DAIA 2608.04347 · Divergence choice 2509.07430 · Retaining by Doing 2510.18874 · BSA test 2410.19406.
**Prompt-opt × RL:** GEPA 2507.19457 · P²O 2603.21877 · SAGE self-hinting 2602.03143 · HiLL 2604.00698 · PATS 2607.21419 · Context-return/NCA 2606.11627 · OPSD law 2605.30070 · POW3R 2605.20164 · CHERRL 2606.04923 · MAS-PromptBench 2606.23664 · SPEAR 2605.26275.
