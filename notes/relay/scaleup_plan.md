# Relay scale-up: more tasks, finetuned student (plan, not launched)

Written 2026-09-28 ~22:00 PT after Luke asked to run the relay recipe on the rest of CalibForge, TMax and SWE-like tasks
with a finetuned Snowball as the student. Nothing is submitted; the first batch waits for Luke's go and the node-hour line.

## Verdict

**Start with the rest of CalibForge, served with the finetuned student. It is the only pool that runs on Daytona today
without new snapshots.** About 2,660 unused tasks remain on the three CalibForge snapshots we already hold. TMax needs a
conversion and snapshot slots, and every SWE-like pool needs 8–15 snapshot slots in an eval org that is full (40/40).

## Pools

| pool | tasks usable | runs on | snapshot slots | SWE-bench Verified risk | ready? |
|---|---|---|---|---|---|
| CalibForge, unused | ~2,660 (of 3,088 unstaged; ≤1,800 s budgets, covered bases, no TB2 copies) | Daytona, existing 3 snapshots | 0 | none | yes, after `build_tree.py --all` and mirroring the new layers |
| TMax (`laion/TMax-15K-Harbor`) | 14,601 | needs the CalibForge pattern (one ubuntu 22.04 snapshot + setup replay) | 1–2 | TB-Hard is drawn from TMax (spends it as a held-out set) | no: validation never run, 600 s budgets, internet on, no reference solutions, GLM-5.2 solves ~92 % |
| R2E-Gym Daytona v3 | 3,020 gated; ~1,750 without sympy | Daytona + apptainer | 11 (none held now) | **sympy overlaps: 14 of our random-100 eval tasks are sympy; 3 R2E tasks share a base commit with eval tasks (06762/sympy-21847, 07797/23413, 06565/16597)** | after freeing 11 slots and dropping sympy |
| SWE-smith | ~2,500 on 15 repos | Daytona | ~15 | clean | no snapshots built |
| SWE-rebench V2 (untraced) | ~1,540 | Daytona via TaskTrove images | ~15 | clean | no snapshots built |

## Cost

From the 09-26/28 relay runs (8 nodes: 4 student, 4 teacher): **15–20 node-h per 1,000 episodes, 16–21 per 1,000 usable
rows** (a pass or a failure whose verifier ran). The final SFT set used fewer rows than generated because of the 1:1
match with the Qwen-alone arm and the 2-per-task cap, not because of generation.

## Finetuned student

- Swapping the student is one variable (`STUDENT_MODEL` in serve_relay.sbatch). The relay-SFT exports carry 09-21's
  tokenizer, chat template and config, so the router's reasoning cap and the template stay right.
- What is tuned to 09-21 and needs an early check: the EAGLE-3 draft (acceptance unmeasured on the finetuned model:
  speed only), the takeover triggers and run gates (calibrated on 09-21: a better student triggers fewer takeovers, so
  fewer trained teacher turns per episode), and autofix (09-21's format quirks; the SFT models reject 15–20 % of turns
  vs 89 %).
- Which student: the better of A (relay SFT) and MIX (relay + Qwen-alone) after MIX's evals on 09-29.

## First batch (proposed)

1. `build_tree.py --all` for the unused CalibForge tasks; `mirror_upload.py` for their layers (or accept the Docker Hub
   fallback).
2. **Pilot: 1,000 tasks × 2 tries ≈ 2,000 episodes, ~30–40 node-h**, with the existing early gates plus two new ones:
   takeover rate and draft acceptance on the finetuned student.
3. If the pilot holds: the remaining ~1,600 tasks × 2, ~50–65 node-h more.
4. Next pool: R2E-Gym v3 without sympy once 11 snapshot slots are free.

## Filtering pass (2026-09-28 ~23:50 PT): overlap with 09-21's SFT mix and TMax validity

**After removing what 09-21 already trained on and what is broken, the clean pools are CalibForge (all 5,431; ~2,660
unused and runnable now), a filtered TMax (~3.1k strict to ~8.1k lenient) and non-Python SWE-smith. R2E-Gym, Scale-SWE and
most of SWE-rebench are already in 09-21's mix.** Evidence and id lists: `/e/data1/mmlaion/lee27/tmp_overlap/`.

| pool | tasks | already in 09-21's mix | clean remaining | note |
|---|---|---|---|---|
| CalibForge | 5,431 | 0 (released after every 09-21 source was pinned) | 5,431 (~2,660 unused, runnable) | 3 snapshots held |
| TMax | 14,601 | 0 (no text overlap with Nemotron-Terminal; same 9 domains) | ~8,130 lenient / ~3,136 strict | an outside GPT-5.6 audit: 76 % weak verifier, 11 % answer leak, 8 % clean |
| R2E-Gym v3 | 3,035 | 1,749 (every non-sympy task, via CoderForge / Nemotron SWE) | 1,286 sympy | **sympy is 14 of our SWE-bench eval tasks, 3 share a base commit: not usable while we eval on SWE-bench random-100** |
| SWE-rebench V2 | 32,079 | 22,562 | 9,517 (1,507 gold-verified, mostly non-Python) | |
| Scale-SWE | 20,181 | 19,429 | 752 (39 gold-verified) | used up |
| SWE-smith | 88,130 | 36,911 Python (CoderForge) | ~37,222 non-Python | git history leaks the fix (earlier report) |

**TMax filter (no reference solutions needed).** (1) Drop the audit's 724 structural defects and 1,592 answer leaks
(→ 12,285); strict also drops weak verifiers that do not check the answer (→ 5,082). (2) Drop tasks Qwen3.5-9B solved on
every try (→ 8,130 lenient / 3,136 strict; a proxy, from a 70 %-complete run). (3) Paid, 0 GPU: one no-op sandbox per
task (setup must pass `test_initial_state.py`, `test_final_state.py` must fail untouched), ~210–540 sandbox-hours on one
shared ubuntu 22.04 snapshot with the setup replayed at start (1–2 slots).

**Snapshot slots.** 40/40 used; 16 idle for 39 h to 6 days (14 `cap-d14-*` / `cap-verifier-*` / `d14s9-toolchain-s1` from
09-22/23, owner unverified, and `harbor__28f81f69f5af`, `harbor__2fc093da8275`). Freeing them covers a TMax base.
Nothing deleted.

**Revised order.** CalibForge remainder (ready) → TMax strict set after the no-op gate → non-Python SWE-smith / SWE-rebench
only if we want non-Python coverage. R2E-Gym is out while SWE-bench random-100 is the eval.
