# Relay SFT on 09-21: launch plan for both arms (not submitted)

Written 2026-09-28 ~03:30 PT, read-only on Jupiter, no job submitted.

## Verdict

**The chain now trains on exactly the rendered rows.** Before this change it could not. The chain only took chat rows
and rendered them itself, and a chat template's loss mask trains every assistant turn. That includes 09-21's own turns
in the relay rows and every reasoning span our render masks. On the 150 audited rows it would have trained 2.7 M extra
tokens. The adapter makes the chain take each row's `ids` and `loss` as they are, with no template and no
re-tokenization.

The rows go through the chain's own prep and data loader, and what the trainer draws matches the render on every audited
row: 86 relay rows and 64 Qwen-alone rows, the same ids and the same per-token mask. Nothing is trained across a row
boundary, and no row is cut.

Two things are different from the Bespoke recipe on purpose. Both are decisions for the coordinator.
- **A "pass" counts packs, not tokens.** Relay rows fill only about 75 % of a 65,536 sequence. So the chain's
  token-based "3 epochs" would be about 2.2 real passes for the relay arm and 2.3 for the Qwen-alone arm. The Bespoke
  run had the same shortfall. This plan runs 3 real passes per arm.
- **Equal passes do not mean equal steps.** Relay rows carry 09-21's turns as masked context, so the relay arm needs
  about 1.6× the optimizer steps of the Qwen-alone arm, for about the same number of trained tokens. The plan keeps
  equal passes, the usual reading of "same recipe".

## Index

| item | status | where |
|---|---|---|
| adapter: rows in as ids + loss | committed, CPU-tested | marin `lukedhlee/vista-snowball-sft` 3a5ae99a88 (`grug_datakit_chat.SnowballPrerenderedChatFormat`, stages `relay_relay` / `relay_qwen`) |
| real-pass step count (`SNOWBALL_EPOCH_STEPS`) | committed | same marin commit (launcher + keep-per-epoch); `data/relay/sft/pack_epoch_steps.py` |
| rows → parquet | committed | `data/relay/sft/relay_rows_to_parquet.py` (OTA d10ea28b, as are the rows below) |
| full-set equality check | committed | `data/relay/sft/verify_relay_cache.py` |
| arm launcher | committed | `data/relay/sft/relay_sft_arm.sh` |
| held-out 300 at 16,384 output | committed knob | `data/relay/pilot/run_pilot.sh` `MAX_OUTPUT`, `relay_pilot.yaml` `__MAX_OUTPUT__` |
| TB2 at 65k/16k | committed policy file | `data/tb2/jupiter/tb2_marin_policy_65k16k.yaml` |

## The recipe

Both arms get the same recipe. Only the data differs.
- **Init.** 09-21 imported once, at `init-dk0921-step0`. It carries qk_mult 1.75 and `pending_qb_betas = −router_bias`.
  The router bias stays frozen at 09-21's value for the whole run.
- **Layout.** 16 × 65,536 on 4 nodes, one sequence per GPU, the XLA memory settings from 09-23, and seed 0.
- **LR 3e-4, kept from Bespoke.** It was the only setting on 09-21 that moved held-out loss (3e-5 barely moved it, and
  1e-3 spiked). Cosine to 10 %, with 5 % warmup.
- **Risk.** Bespoke's pick ran 45 steps, and these arms run about 180–280. So each pass is exported, and a late
  regression would be visible. The decision still reads pass 3 in both arms, fixed before any result.
- **3 real passes.** Each pass is ceil(packs / 16) steps. The trainer's loader is close to, but not exactly, one pass
  per 1/3 of the run. With pack-based steps, 96 % of the current relay packs are drawn exactly 3 times.

| per arm | relay (current 1,814 rows) | Qwen alone (current 1,814 rows) | at ~2,000 rows (final_v2, estimate) |
|---|---|---|---|
| tokens / trained tokens | 66.0 M / 14.96 M | 43.8 M / 15.77 M | ×1.10 |
| packs at 65,536 (fill) | 1,352 (0.745) | 860 (0.777) | relay ~1,490, Qwen ~950 |
| steps per pass / total (3 passes) | 85 / 255 | 54 / 162 | relay ~94 / ~282, Qwen ~60 / ~180 |
| token-based steps (for reference) | 63 / 189 (2.24 passes) | 42 / 126 (2.34 passes) | |
| warmup (5 %) | 13 | 8 | 14 / 9 |
| train wall, 4 nodes | ~58 min → **3.9 node-h** | ~40 min → **2.6 node-h** | **4.3** / **2.9 node-h** |

The step time comes from the Bespoke think_all 3e-4 run (job 1967664), measured at the same shapes. Steps 17 to 45 took
341 s, including two checkpoint saves, which is 12.2 s per step. There is also about 5 min of startup and about 1.5 min
of compile. The exact numbers for final_v2 are printed by `PREP_ONLY=1` (the `packs:` line) before any training job.

## Cost (stated before any submission)

| step | nodes | node-h per arm | both arms |
|---|---|---|---|
| prep (cache + probes) | 1 | ~0.25 | 0.5 |
| train, 3 passes | 4 | 4.3 / 2.9 | 7.2 |
| exports, one per pass | 4 | ~1.0 | 2.0 |
| common held-out NLL, one per pass | 1 | ~0.5 | 1.0 |
| **training side** | | | **~10.7** |
| held-out 300 eval (09-21, relay SFT, Qwen SFT) | 1 | ≤2.5 cap each | ≤7.5 (09-21 took 1.29 at 8k output) |
| TB2 at 65k/16k (same three) | 1 | ~4–5 each | ~12–15 |
| **all** | | | **~30–33** |

## Before launch (login node; nothing here spends node-hours)

1. **Check that no SFT job runs from `marin-sft`, then pull it.**
   ```
   squeue -u lee27 -o '%.10i %.30j %.8T' | grep -iE 'snowball|guarded|export|heldout' || echo none
   git -C /e/project1/transfernetx/lee27/code/marin-sft pull --ff-only    # -> 3a5ae99a88
   ```
2. **Make an OTA worktree at d10ea28b (the chain-side code), and copy the chain files** to the snowball code dir.
   `snowball_sft_chain.sh` there is older than the repo. Its 09-21 path is the same, but copy it anyway so the two
   match.
   ```
   O=/e/project1/transfernetx/lee27/code/OpenThoughts-Agent; W=/e/project1/transfernetx/lee27/code/ota-relay-sft
   git -C $O fetch -q origin && git -C $O worktree add --detach $W d10ea28b   # or a later commit of the branch
   C=/e/project1/transfernetx/lee27/code/snowball
   cp $W/data/r2egym/jsc/{snowball_sft_chain.sh,ota_lane.sh} $W/data/relay/sft/{relay_sft_arm.sh,relay_rows_to_parquet.py,pack_epoch_steps.py} $C/
   cp $W/data/tb2/jupiter/tb2_marin_policy_65k16k.yaml /e/project1/transfernetx/lee27/code/tb2/
   ```
3. **Point at the final_v2 files** once the other agent has written them. Use the same format as
   `final_rendered_think16k_clean.jsonl`.
   ```
   R=/e/fscratch/reformo/lee27/experiments/relay/pilot/runs
   ROWS_RELAY=$R/relay_full_relaym_20260926/final_v2_<...>.jsonl
   ROWS_QWEN=$R/relay_full_baseline6m_20260926/final_v2_<...>.jsonl
   ```

## Launch

These commands run on a Jupiter login node, inside tmux. The two arms run one after the other or side by side. If they
run side by side, start them a few minutes apart: two 16-rank compiles in the same minute have wedged before.

1. **Prep both arms (1 node each, about 15 min).** This converts the rows, builds the cache and prints the pack count
   and step plan. The converter refuses the whole file if any row is over 65,536 tokens, has a mismatched mask, or has a
   trained or missing BOS.
   ```
   cd /e/project1/transfernetx/lee27/code/snowball
   tmux new -d -s rsft_prep_relay "ARM=relay ROWS=$ROWS_RELAY PREP_ONLY=1 bash relay_sft_arm.sh"
   tmux new -d -s rsft_prep_qwen  "ARM=qwen  ROWS=$ROWS_QWEN  PREP_ONLY=1 bash relay_sft_arm.sh"
   # -> /e/data1/mmlaion/lee27/snowball-sft/logs/relay_{relay,qwen}/arm.log: "packs: {...} -> N steps per pass; 3N steps"
   ```
2. **Optional full-set gate, off the login node.** The sample proof covers 150 rows. To check every row of final_v2,
   copy the parquet and the cache the prep job built (about 0.5 GB per arm) to the Mac, then run:
   ```
   SNOWBALL_SEQ_LEN=65536 SNOWBALL_BATCH=16 SNOWBALL_DEVICES=16 JAX_PLATFORMS=cpu PYTHONPATH=<marin checkout @3a5ae99a88> \
     <marin CPU env python> data/relay/sft/verify_relay_cache.py relay_relay <rows.jsonl> <parquet dir> <cache dir> <09-21 tokenizer dir>
   # verdict EQUAL / exit 0 required; same for relay_qwen
   ```
3. **Train, export each pass and score the common held-out NLL.** Each arm is one lane: 4 nodes for training, then 3
   exports on 4 nodes and 3 NLL scores on 1 node.
   ```
   tmux new -d -s rsft_relay "ARM=relay bash relay_sft_arm.sh"
   sleep 300
   tmux new -d -s rsft_qwen  "ARM=qwen  bash relay_sft_arm.sh"
   ```
   - **What to check a few minutes after start.** Look for these lines in
     `/e/data1/mmlaion/lee27/snowball-sft/logs/snowball-relay_<arm>.<job>.log`:
     - `model_qk_mult=1.75 … freeze_router_bias=True batch=16 devices=16`
     - `optimizer lr=0.0003 warmup=<5 %>`
     - `checkpoint_keep_every=<steps per pass>`
     - "Router bias frozen at the init's pending_qb_betas"
     - the launcher's `epochs = 3 (<N> packed steps per epoch; SNOWBALL_EPOCH_STEPS=<N>)`
   - **Memory.** Bespoke's 64k step peaked at 90.00 of 90.25 GiB, at the same shapes and settings. An OOM would show at
     step 1.
   - **Outputs.** `/e/data1/mmlaion/lee27/snowball-sft/experiments/snowball-relay-sft/relay_<arm>/lr3e-4-sched3/export-step<S>-hf-bf16`,
     one per pass. The final pass is the one under test.

## Eval (after both final exports exist)

All three models are evaluated with the same serve and settings: 09-21 itself (the before point), the relay SFT and the
Qwen-alone SFT. **09-21 must be re-run.** Its held-out run (`heldout0921_20260925`) used 8,192 output tokens, and its TB2
runs used v0.1's 32k/8k, so neither is a 65k/16k number.

**Held-out 300 CalibForge.** This runs through the same driver as the `heldout0921` run: student-only pass-through
router, Terminus-2 strict, summarization off, 65,536 input and now 16,384 output. It uses 1 node with a 2.5 node-h cap.
```
P=$W/data/relay/pilot; cd $P
for M in 0921:/e/data1/mmlaion/lee27/models/grug-datakit-sft-20260921 \
         relaysft:<relay final export> qwensft:<qwen final export>; do
  n=${M%%:*}; m=${M#*:}
  J=$(STUDENT_MODEL=$m N_STUDENT=1 RELAY_PILOT_DIR=$P sbatch --parsable --nodes=1 --time=02:40:00 serve_relay.sbatch)
  tmux new -d -s ho_$n "cd $P && RUN_KIND=heldout TREE=/e/fscratch/reformo/lee27/tasks/calibforge_heldout300 NTASKS=300 \
    ARMS=student_only NODES=1 CAP_NODE_H=2.5 CLOCK=wall MAX_OUTPUT=16384 CONC=100 bash run_pilot.sh $J heldout6516_${n}_20260928"
done
```
- `CLOCK=wall` gives each task harbor's own 1× budget, as TB2 does and as `heldout0921` ran. The other choice is
  `paused`, which removes serving speed from the budget. The coordinator should pick one before the first run.
- Each run's `readout.json` carries per-task outcomes. The decision rule needs a paired relay-vs-Qwen CI on the same 300
  tasks, and that comparison script is not written yet.

**TB2 at 65k/16k.** This is the v0.1 policy with only the token budgets changed. The coordinator should confirm this is
the intended "65k/16k policy". It runs on 1 node per model, with the same pipeline the Bespoke evals used.
```
T=/e/project1/transfernetx/lee27/code/tb2; E=/e/fscratch/reformo/lee27/experiments/tb2
for M in grugdk0921think:/e/data1/mmlaion/lee27/models/grug-datakit-sft-20260921 relaysft:<relay final export> qwensft:<qwen final export>; do
  n=${M%%:*}_t1_6516_20260928; m=${M#*:}
  J=$(cd $T && MODEL=$m POLICY=trained sbatch --parsable serve_snowball.sbatch)
  tmux new -d -s pl_$n   "cd $T && POLICY_FILE=tb2_marin_policy_65k16k.yaml bash pipeline.sh $J $n"
  tmux new -d -s post_$n "cd $T && POLICY_FILE=tb2_marin_policy_65k16k.yaml bash post_run.sh $n"
  tmux new -d -s reap_$n "while [ ! -f $E/final_$n.DONE ] && squeue -h -j $J | grep -q .; do sleep 900; done; scancel $J"
done
```
- `serve_snowball.sbatch` already serves 65,536 with 09-21's template and the EAGLE-3 draft. Thinking is the model
  default, `Reasoning: /think`, the mode both arms trained in.

## Evidence

- **Equality, sample.** `verify_relay_cache.py --prepare` ran on the Mac CPU through the chain's own `prepare-data`,
  `write-provenance`, preflight validation and `snowball_chat_data_config(...).train_sets` at 65,536.
  - Relay: 86 rows in 83 packs. Qwen alone: 64 rows in 44 packs.
  - Every row's ids and shifted loss weights equal the render.
  - Trained tokens match exactly: 665,721 and 644,575.
  - No weight sits on a row's last position or on padding, and every row appears exactly once.
- **What the sample covers.** The 10 longest rows per arm (up to 65,433 tokens), every row with a Qwen turn whose uncut
  reasoning is over 16,384 tokens (34 relay rows, 19 Qwen-alone rows), 12 rows with autofixed 09-21 turns, 12 with
  autofixed Qwen turns, 12 stripped rows (5 in the Qwen-alone arm, all it has) and 20 random rows.
- **Whole-file census, both current final sets, 1,814 rows each.** 0 rows over 65,536 (max 65,433 relay, 58,152 Qwen
  alone). All rows start with an untrained BOS, have loss values of 0/1 only, and have ids ≤ 128,011.
- **Template.** The rows are 09-21's `chat_template.jinja` (b3f20bc2). That is also the `tokenizer_config.json`
  template and what vLLM serves with `--chat-template`.
  - All 150 rows start `<|start_header_id|>system<|end_header_id|>Reasoning: /think<|eot_id|>`.
  - HF `apply_chat_template` re-renders every row byte for byte.
  - At all 2,800 assistant turns, vLLM's generation prompt (history + `<|start_header_id|>assistant<|end_header_id|>\n`)
    is the row's exact id prefix.
  - The chain's vendored training template (6f55d2ce) with `enable_thinking=True` and `reasoning_content` gives the same
    ids for 149 of 150 rows. The one difference is a trimmed newline after `<|end_think|>` in a masked 09-21 turn.
  - The Stage-3 Marin template differs from the first token after BOS: it has no system header.
  - The adapter bypasses templates, so the trainer sees the serving render.
- **Bespoke.** Its caches were built with the training template, with `enable_thinking` True on every think row and
  False on every fold row. Its exports ship the serving template, and its TB2 runs used the matching mode (think arms by
  default, fold arms with `enable_thinking: false`). It was consistent.
- **Router bias.** Checked on artifacts on 2026-09-28.
  - 09-21's bias is non-zero, with per-layer std 0.26–0.60.
  - The init's `pending_qb_betas` equal −bias exactly.
  - All 36 Bespoke exports carry bias = bf16(09-21 bias − layer mean) exactly, while their weights moved.
  - Every Bespoke log says `freeze_router_bias=True`.
  - The `relay_*` stages freeze it by default and refuse an init without the sidecar.
- **Loader.** The trainer's sampling was replayed index for index (block shuffle + mixture). At 3 token-epochs, relay
  packs are drawn 1–4 times, 2.24 on average. At 3 pack-epochs, 1,298 of 1,352 are drawn exactly 3 times.
- **Scratch.** The audit scripts and outputs are in the session scratchpad `sft_audit/`: `template_compare.py`,
  `g_check.py`, `coverage_sim.py` and `verify_*.json`.
