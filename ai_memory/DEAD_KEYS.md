# DEAD KEYS — config that is accepted, echoed, and ignored

This is the project's single most repeated bug class: **seven times** a key was set, deep-merged, written to
the materialized config, confirmed present by inspection — and ignored by its consumer. Each instance cost at
least one job. Two of them produced a *wrong published headline*, which is worse than a crash.

**The rule this file exists to enforce:** a config key is not "set" until a **log line** proves the consumer
acted on it. Verifying a key arrived proves nothing. The gold-standard pattern is `SKYRL_CHUNKED_LOGPROBS=1`
→ log line `chunked gathered log-softmax ACTIVE`. If you cannot name the line that fires, you have not
verified anything — you have read your own input back.

`hpc/skyrl_standard/jupiter/band_preflight.sh` hard-fails on the three of these that can appear in a band
config. The rest are here so nobody re-derives them.

| # | key | consumer | what actually happened | real lever |
|---|---|---|---|---|
| 1 | `strict_json_parser` | OpenCode | accepted, never read | — |
| 2 | `compaction.reserved` | OpenCode | never read; compaction threshold unmoved | `context_budget.client_window_tokens` (OT-Agent `4fb4a158`) — advertise a **smaller** window to the client while the server keeps its own |
| 3 | `store_all_messages` | OpenCode | silently ignored; **severed the training data path for days**, hidden because every earlier run died sooner | harbor `179b31e9` — `_build_all_messages()` reconstructs history from stdout events |
| 4 | `agent.override_timeout_sec` | harbor `apptainer.py` | present as `1800.0` the whole time, but `opencode.py:871` calls `exec_as_agent()` with **no** `timeout_sec`, so `apptainer.py:397` fell back to `600`. The outer guard can only fire if it is *shorter* than the inner one, so the worker always won. **Truncated 19/25 trials on `1221005` and produced the wrong headline "raw r2egym has zero variance."** | harbor `f44a1170` `BRIDGE_EXEC_TIMEOUT`, set **above** the agent budget (2100 vs 1800) so `AgentTimeoutError` (passthrough, preserves the trajectory) wins over an exec kill |
| 5 | `hf_upload_mode` | `callbacks/builtin.py:977` | the consumer is **never constructed** — the whole `HFHubUploadCallback` is gated on an unrelated unset key, `hf_hub_repo_id`. Export written to disk, never uploaded, no warning. | set `hf_hub_repo_id` |
| 6 | `timeout: 900` on the harbor block | `lite_llm.py` | not honoured; trials still died `APITimeoutError`. Job `1236881` was spent **entirely** testing this inert key. | — |
| 7 | `extra_body` / `interleaved_thinking` | harbor | implemented for `terminus_2`, `openhands`, `mini_swe_agent` only — **there is no OpenCode path**. Inert but *misleading*: the config states a thinking mode it cannot deliver. | server-side `default_chat_template_kwargs` (MarinSkyRL `9904058`) — the only lever that reaches an external agent |

## Two generalizations worth more than the table

**1. Grep for the guard, not the key.** #5 is the nastiest variant: the key is not ignored by its consumer,
the consumer is never built, because a *different* unset key gates it. So search for
`if <other_key> and ...` around your key's use site, not just for your key.

**2. Same symptom, different mechanism — do not reuse a diagnosis.** `median_steps=1` was produced by
OpenCode (one text response, no tool call, because the tool-call dialect never matched) *and* by terminus-2
(the single "step" is the prompt itself; no model turn recorded, because the LLM call timed out). Identical
gate output, unrelated causes. A matching symptom is not a matching bug.

## Why this file is separate from `gotchas.md`

All seven were already documented — in 1,808 lines of `gotchas.md`, which no fresh session reads end to end,
and **none of them appeared in `SESSION_START.md`**, the doc that actually bootstraps a session. Documented
and unread is indistinguishable from undocumented. Keep this file short enough to read in one sitting; when
an entry can become a preflight assertion, move it into the script and leave the row here as history.
