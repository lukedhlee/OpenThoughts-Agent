# Marianna's band-vs-raw experiment result (verbatim, relayed by operator 2026-08-08)

Operator note: Luke says he relayed this to multiple prior AI sessions but it never
got persisted properly. This is the durable verbatim record. Check here before
making claims about what her band experiment showed.

"""
Experiment: DCAgent/g1_diverse_tezos_100k_8b on R2E-gym (terminus-structured).
Goal: figure out whether genuinely learnable band (all tasks are neither too easy or too hard) changes transfer on OOD swebench.

Learnable band: R2E-gym filtered by the base's  p@4 (only tasks with headroom for RL left) - ~1.6k tasks out of 4.5k.

Result: filtered band converges faster (it reaches the same ~45 p@1 on swebench by step 60) but it doesn't give any performance boost. The reward seems to be more trending up then the raw pool and the filtered band model is more efficient then the raw (even less tokens are used for task solution).

Comparison of compute spent to reach ~45.5 on SWB (is it worth it to invest in pre-filtering of tasks from the compute point of view?):
RAW 120 steps - 60k rollouts
BAND 60 steps + p@4 filtering = 18k + 30k = 48k rollouts
So filtering has some compute advantage on top of raw training.
"""

## Key takeaways (interpretation, 2026-08-08)

- **Her band is ~1.6k of 4.5k (~35%)**, defined as "filtered by the base's p@4 —
  only tasks with headroom for RL left". This is the ~36% figure that circulates.
- **TENSION with the 358 figure** in `../marianna_parity.md` (358 ≈ 8% at strict
  0<pass@4<1). Both numbers are hers but the definitions differ: ~1.6k = "has
  headroom" (likely includes pass@4=0 tasks, i.e. excludes only the saturated),
  358 = strict mixed pass/fail. When quoting "her band", say WHICH definition.
  Our band512 probe measures the strict-mixed set (0<pass@4<1) AND can also
  report the headroom set (pass@4<1) from the same data — report both.
- **Band filtering buys compute efficiency, not final performance**: same ~45
  p@1 on OOD swebench, reached in half the training steps (60 vs 120); total
  compute incl. the filtering probe still favors band (48k vs 60k rollouts).
  Also: smoother/steadier reward curve and more token-efficient solutions.
- Implication for our probe: its value is (a) the task pool for efficient GRPO,
  (b) NOT a promise of a higher ceiling than raw-pool training.
