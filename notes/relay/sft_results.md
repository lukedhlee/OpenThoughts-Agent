# Relay SFT on 09-21: results (three arms + 09-21)

Results go here when the evals finish. The pre-registration below was written and committed on 2026-09-28 ~11:53 PT,
before any training job or eval of this comparison was submitted (only the two data-prep jobs had been).

## Pre-registration (fixed before any result)

**Models under test.** Four models, each served the same way.
- **09-21.** `grug-datakit-sft-20260921` itself, the before point.
- **A, relay, 09-21 turns masked (main arm).** The relay final_v2 manifest re-rendered with `--autofix-loss none`
  (think cap 16k, copied markers stripped, render.py at d0ddd237). Marin stage `relay_relay`.
- **B, Qwen alone.** The baseline final_v2 file as it is. Stage `relay_qwen`.
- **C, relay + autofix loss.** The current relay final_v2 file as it is (autofixed 09-21 actions trained). Stage
  `relay_relayaf`.
- Every arm has the same recipe: the 09-21 import init, router bias frozen at 09-21's non-zero value, LR 3e-4 with the
  Bespoke schedule (cosine to 10 %, 5 % warmup), seed 0, 3 real passes (steps = ceil(packs / 16) × 3), 16 × 65,536 on
  4 nodes, packing with cross-row attention blocked, loss taken from the rendered rows.
- The checkpoint under test is the **final pass-3 export** of each arm. Earlier passes are exported only to watch for a
  late regression; they do not enter the decision.

**Held-out 300 CalibForge.** The same driver as `heldout0921_20260925`.
- `run_pilot.sh` with `RUN_KIND=heldout`, tree `calibforge_heldout300`, 300 tasks, `ARMS=student_only`, 1 serve node
  (`serve_relay.sbatch`, `N_STUDENT=1`), `CONC=100`, `CAP_NODE_H=2.5`.
- 65,536 input and 16,384 output tokens. `CLOCK=wall`, which is harbor's own 1× agent budget per task.
- Terminus-2 strict. Harbor `harbor-terminus2-relay` at 89098635 for all four models.
- One trial per task.

**TB2.** The current Marin TB2 policy with only the token limits changed.
- `tb2_marin_policy_0924_65k16k.yaml`: harbor-p0924 at 761fb516, 65,536 input and 16,384 output tokens, 1,800 s agent
  budget, Daytona, 16 concurrent.
- 89 tasks, one trial per task. `serve_snowball.sbatch` on 1 node (EAGLE-3 draft, DP4), thinking at the model default.

**Analysis.** `data/relay/sft/paired_eval.py` as committed in f9aa523d, unchanged.
- Pass rates carry a Wilson 95 % CI. Paired differences use the tasks both models scored, with a bootstrap 95 % CI over
  tasks (10,000 resamples, seed 20260928).
- **Decision.** A wins if the held-out paired A − B CI lies above 0 and TB2 is not worse (the paired TB2 A − B CI does
  not lie entirely below 0). Then the relay is worth scaling. Otherwise, scale Qwen-alone traces.
- **Sanity.** An arm beats 09-21 if its held-out paired gain over 09-21 has a CI above 0. If neither A nor B beats 09-21,
  the SFT failed and A vs B says nothing.
- **Also reported.** C − A (does training the autofixed 09-21 actions help or hurt), C − B, and each arm vs 09-21.

**Runs and retries.**
- A failed step (job crash, serve death, harness errors over 10 % of trials) is retried once. A retry of an eval is merged
  per task, the later run winning, as `paired_eval.py` does. A second failure stops that step and is reported as FAILED.
- The A, B and C evals run at the same time, so the decision is not confounded by load drift. The 09-21 evals may run
  earlier.
- Budget ~43 node-hours, hard cap 48.
