# tito_match.py — does the trainer train on the ids vLLM sampled?

TITO holds when the trainer assembles the training sequence FROM harbor's served
token-id streams instead of re-tokenizing the conversation. `tito_match.py` decides
that from harbor artifacts alone, read-only, no GPU.

## Run it on the migration smoke

```bash
# TOK = the policy's tokenizer. Any local snapshot of the arm's model works, e.g.
#   TOK=$(huggingface-cli download laion/snowball-67b-a2b-sft-s3-nemotron-terminal-step1888)
# PY  = any python3 with `transformers` (no GPU, no MarinSkyRL deps needed).
"$PY" tito_match.py \
  --trials-dir <arm trials_dir> --tokenizer "$TOK" --json after.json
```

`trials_dir` comes from the arm config (`arm_rl_config.json`). Harbor writes one
`<trial>/result.json` per trial — **not** `trajectory.json`; the script walks either.
When the artifact store is an ext image, read it from a login node with
`debugfs -R "cat /trace_jobs/<trial>/result.json" <artifact_store.img>` and copy the
files down. Without `--tokenizer` the generation prompt is inferred from the data,
which can be a few tokens longer than the real one; the prefix result is unaffected.

Add `--marinskyrl <path to a MarinSkyRL checkout>` to run MarinSkyRL's
actual assembly (heavy deps stubbed, module loaded by file path) and compare the
loss_mask==1 ids to the sampled ids directly. Needs `--tokenizer`. It agrees with the
cheap path on every trial tested.

## What "TITO holds" looks like

100% exact match, 0 divergent turns, no decline reasons, and `tokens the trainer
would train on` equal to `sampled completion tokens`.

## The metrics you get for free

`use_tis=true` sets `rollout_logprobs_required`, which selects full TITO whether or
not `trainer.algorithm.tito_full` is set, so the arm already emits per step:

- `generate/tis/tito_full/attempts`
- `generate/tis/tito_full/success_fraction`  ← the headline; 1.0 means TITO holds
- `generate/tis/tito_full/decline_count`
- `generate/tis/tito_full/decline/<reason>`  ← same reason names the script prints

With `trainer.logger=console` they land in the trainer log as `Step N:` blocks; on
W&B they are those keys verbatim. The script exists to attribute a non-1.0 fraction
to specific trials and turns, and to check the claim offline.

## Baseline on the old stack (the "before")

Two finished Jupiter arms, both on harbor without #111/#117 and MarinSkyRL `be413fcb`:

| arm | rollouts | TITO exact | ≥20 turns exact | divergent turns | sampled tokens | trained |
|---|---|---|---|---|---|---|
| `snowball_ttband_lr5e7_fulldist_r36…` (job 1699607) | 15 | 2 (13%) | 0 of 11 | 28 / 371 | 495,933 | 69,904 |
| `snowball_v2rest_fulldist_s24` (job 1735900) | 14 | 8 (57%) | 7 of 13 | 9 / 348 | 334,654 | 218,911 |

Every decline is `prefix_mismatch`, and every divergence lands inside the model's own
sampled tokens (`recut`) — issue #520, typically 2 tokens wide. Samples are in
`artifacts_ttband_arm/` and `artifacts_v2rest_arm/`; results in `baseline_*.json`.

## Caveats found

- **A decline costs the whole trajectory.** With `use_tis=true` a declined rollout has
  its entire loss mask zeroed, so it contributes 0 training tokens. That is why the
  trained column is 0, not "re-tokenized but still trained".
- **The re-tokenizing fallback still gets the generated region right.** Under the
  pre-#521 splice path the sampled ids are spliced in exactly; what is wrong is the
  masked *context*. So the cost of a decline is the guarantee, not the sequence.
- **The streams only exist when `collect_rollout_details=true`.** Without it there are
  no `prompt_token_ids` and the script reports `missing_streams`.
- **`dump_train_rollouts` cannot settle this.** It writes decoded text
  (`skip_special_tokens=True`), no token ids. `--trainer-dump` only prints that fact.
- **Only `rollout_details[0]` is measured** — later entries are subagents, outside the
  linear trained sequence. The script flags trials that have more than one.
- **`HARBOR_TERMINUS2_HISTORY_THINK`** rewrites prior assistant spans and would break
  the prefix invariant by design. Do not baseline on an arm that sets it.
- Turn 0 has no prefix to check, so `divergent turns` is out of `total turns − rollouts`.
