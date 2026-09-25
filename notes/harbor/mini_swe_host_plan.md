# mini-swe-agent v2 on the host: Harbor agent spec

2026-09-24 · **status (09-25 12:30 PT): PARKED by Luke — we stick with Terminus-2. Stages 0–3 ✅ harness gates, tool mode built (`lukedhlee/mini-swe-host` @cdd5b008); no further MSA runs** · target: marin-community/harbor (remote `upstream` in
`/Users/lukedhlee/harbor`); for Stage 5 only, also MarinSkyRL · worktree `/Users/lukedhlee/harbor-wt-mini-swe-host`
cut from `upstream/main` (761fb516; executed on 07dd3ef3) · branch `lukedhlee/mini-swe-host` · evidence
`agent_logs/2026-09-24_mini_swe_host_scoping.md`, execution log `agent_logs/2026-09-25_mini_swe_host_exec.md`

## Parked (Luke, 2026-09-25 12:30 PT): we stick with Terminus-2

The harness works; the relay student does not work in it. Tool mode is the right mode for 09-21 (it was trained on
mini-swe-agent traces as tool calls, and its format errors drop from 32 % to 12 %), but in tool mode it still never
submits: all 8 episodes run long (median 44.5 calls) and 6 of 8 hit the 64k context. The deeper mismatch is the
environment: mini-swe-agent runs every action in a fresh subshell (no `cd`, no env vars, no background process carries
over), while the terminal tasks we care about (TB2-style, CalibForge) assume a persistent shell, which Terminus-2 gives.
No further MSA runs; the code stays on `lukedhlee/mini-swe-host` (no PR).

**Four-row readout (8 CalibForge tasks, Daytona, `mini_textbased.yaml` for text / `mini.yaml` for tool):**

| model · mode | format errors / calls | format-error causes | submit | pass | median calls | harness errors | other |
|---|---|---|---|---|---|---|---|
| 09-21 · text (job 2021265) | 40 / 125 (32 %) | 11 prose without a block, 9 cut off at the context end, 8 wrong fence (```bash/```json), 5 ended inside reasoning, 6 two blocks, 1 Terminus JSON | 2/8 | 1/8 | 5 | 0 | 6/8 end on 3 format errors in a row |
| 09-21 · tool (job 2023105) | 45 / 363 (12 %) | 21 cut off at the context end, 11 invalid tool-call JSON, 9 ended inside reasoning, 2 unclosed `<tool_call>`, 1 prose, 1 unknown tool | 0/8 | 0/8 | 44.5 | 0 | 6/8 ContextLengthExceeded (every episode reached ~56–65k prompt tokens), 2/8 RepeatedFormatError; 14 multi-call replies |
| Qwen3.8 · text (job 2011666) | 8 / 191 (4 %) | 4 cut off at the context end, 3 wrong fence, 1 run-on | 5/8 | 6/8 | 23.5 | 0 | 1 agent timeout, 1 context overflow |
| Qwen3.8 · tool (job 2024414) | 5 / 161 (3 %) | 5 cut off at the context end | 6/8 | 5/8 | 20.5 | 0 | 1 context overflow; 43 multi-call replies; 0 HTTP errors |

False format errors (harness mistakes) are 0 in all four runs by the readout's independent parser. Tool mode serves:
09-21 exactly as in its evals with `tools=[bash]`, `tool_choice: "none"` (no tool parser; the agent parses
`<tool_call>` from the text); Qwen3.8 with `--enable-auto-tool-choice --tool-call-parser qwen3_coder` and
`VLLM_ENFORCE_STRICT_TOOL_CALLING=0` (the strict structural-tag grammar rejects MTP draft tokens on this build: HTTP 500
on 28 of 64 calls in cancelled job 2024000). Both tool-mode renders were checked: on CPU against 09-21's template and
live via `/tokenize` on the 09-21 server (Tools block, re-rendered `<tool_call>`, `<tool_response name="bash">`, prior
reasoning re-fed).

**Spend:** 2.23 Jupiter node-hours in total: Stages 2–3 0.99, 09-21 text 0.29, 09-21 tool 0.33, Qwen tool 0.62 (0.09
failed start on `EADDRINUSE`, 0.18 cancelled on the grammar 500s, 0.36 the good run).

## Execution status (2026-09-25)

| Stage | Status | Commit | Gate result |
|---|---|---|---|
| 0 | ✅ DONE | harbor `38c6c548` | Replay of unmodified upstream 2.4.6 on 5 scripted episodes (submit ×2 configs, format error ×3 incl. a truncated reply, format-error recovery, step limit): 11/11, deterministic, every request = message list so far |
| 1 | ✅ DONE | harbor `7f042513`, fixes `05dd2161` + `543d611c` | Requests and messages byte-identical to upstream on all 5 episodes; full unit suite vs baseline: 3,389 existing test ids, 0 outcome changes; 40 new tests pass |
| 2 | ✅ DONE (run 2) | smoke `0f74fd8d` (= `543d611c` + hook), job 2011666 | Qwen3.8-27B, 8 CalibForge tasks: 0 harness errors; prior reasoning re-rendered inside `<think>` (live `/tokenize`); 8 format errors / 191 calls (4 %), 0 false; 6 command timeouts; submit 5/8, pass 6/8 (sanity); median 23.5 calls; sentinel ended every submit; 1 agent timeout + 1 context overflow, both scored |
| 3 (stand-in) | ✅ harness gate (run 2) | smoke `bb2994de`/`0f74fd8d`, job 2011566 | **SFT→RL step30, a stand-in, not the relay student.** 8 tasks: 0 harness errors, **0 false format errors** (independent readout); 26 format errors / 45 calls (58 %: 13 Terminus-2 JSON, 7 run-on replies with two blocks, 6 ```bash fences); submit 1/8, pass 0/8; median 4 calls; sentinel ended the submit |
| 3 (relay student, tool) | ✅ harness gate; model does not finish | harbor `84b443f3` + `cdd5b008`, smoke `343bc505`, job 2023105 | **09-21, thinking on, tool mode — the target mode for this student.** See the four-row table above: 12 % format errors, 0/8 submits, 6/8 hit the 64k context |
| 3 (relay student, text) | ✅ harness gate; model off-format | smoke `0f74fd8d`, job 2021265 | **09-21 Datakit SFT, thinking on, text mode (wrong mode for a tool-trained student).** 8 tasks: 0 harness errors, 0 false format errors; **40 format errors / 125 calls (32 %)**: 11 prose with no block, 9 cut off (finish_reason=length), 8 wrong fence (```bash/```json, some after `<tool_call>`), 5 ended inside reasoning, 6 two blocks (4 run-on), 1 Terminus-2 JSON; submit 2/8, pass 1/8; median 5 calls; 6/8 end on RepeatedFormatError; sentinel ended both submits; 0.29 node-h |

**Red runs, fixed and re-gated.** Stage 2 run 1 (job 2011216) re-fed no reasoning: Harbor's `Chat` re-sends it as
`reasoning_content`, and this vLLM reads an assistant turn's reasoning only from `reasoning`, so Qwen3.8's template
rendered an empty `<think></think>` for every prior turn (Terminus-2 shares the gap; `Chat` is untouched here). Stage 3
run 1 (job 2011217) had 3 false format errors: Snowball sometimes writes a second `<|end_think|>` after its answer and
the first parser took the text after that stray marker as the answer. Both fixed (`05dd2161`, `543d611c`), plus an
Opus review's findings (reasoning replies broke the history check when `interleaved_thinking` was off; TITO context
overflow escaped as an unscored error; non-transient errors went through upstream's 10-attempt backoff; no hard cap on
a hung exec).

**Student correction (Luke, 09-25).** The relay student is the **09-21 Datakit SFT**
(`open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21`), not SFT→RL step30. The step30 numbers below (39 % / 58 %
format errors) are a stand-in; the 09-21 rows are the ones that count for the relay student, and tool mode is the
target: text mode was a wrong choice for a tool-trained student (Datakit renders the Open-SWE-Traces mini-swe-agent rows
as structured tool calls with the bash schema in `tools`; `datakit/download/open_swe_traces.py::row_to_chat_doc`). On 09-21 the
harness is clean (0 harness errors, 0 false format errors) but the model is not on-format in text mode: a third of its
calls end without exactly one ```` ```mswea_bash_command ```` block, and 6 of 8 episodes die on three format errors in
a row. Unlike step30 it rarely answers in Terminus-2 JSON (1 of 40); its failures are dropped blocks, runaway
reasoning, and wrong fences. Which mode it learned from Datakit's Open-SWE-Traces rendering was not checked (Luke
dropped that step), and tool mode was not run.

**What the smokes say about the models (not the harness).** Snowball step30 (stand-in) is off-format in mini-swe-agent's text
mode: it answers in Terminus-2 JSON or a ```bash fence and dies on 3 consecutive format errors within ~4 calls in 7 of
8 episodes, and it sometimes runs on past its answer into a hallucinated next turn (`</assistant><user>…`). Relay
rollouts in this harness with this student would mostly relay on format errors. Qwen3.8 follows the format (4 %
format errors, most from replies that reasoned into the 64k context) and occasionally runs on too.

**Spend:** 0.99 Jupiter node-hours of the 2.5 cap (18.0 + 10.1 + 9.2 + 21.9 min, one node each, `transfernetx`),
plus 0.29 of a separate 0.5 cap for the 09-21 re-run (17.5 min): 1.28 node-h in total.
Not checked: TB2 through this agent (running TB2 tasks could make `auto_snapshot` create a snapshot, and the org has
one free slot), and a SWE-pool smoke with `swebench_backticks.yaml` (no SWE snapshots on Daytona yet; covered by the
Stage 0/1 replay only).

Branches: `upstream/lukedhlee/mini-swe-host` (the agent, one concern) and `upstream/lukedhlee/mini-swe-host-smoke`
(that plus a merge of `lukedhlee/setup-files-hook`, which CalibForge on Daytona needs; smokes run from it). Launch
scripts: `data/mini_swe_host/` in OT-Agent. A zero-GPU dry run (scripted model server, real Daytona, 8 CalibForge
tasks from the Jupiter login node) passed before any GPU was requested: 0 harness errors, `/app` cwd, a command past the
30 s timeout killed with its process gone, the sentinel ending every episode, and a think span that quotes a code block
not counted as an action.

## What this is

A new Harbor agent, `mini-swe-agent-host`, runs mini-swe-agent v2's own loop and prompts inside the Harbor process on
the Jupiter coordinator, the same place Terminus-2 runs. Each bash command goes to the sandbox through Harbor's
`environment.exec`, so Daytona needs nothing new. Every model call goes through Harbor's `LiteLLM`/`Chat`, the same
client Terminus-2 uses. That one choice buys three things: Snowball's inline think spans are parsed the way our
Terminus-2 evals parse them, Qwen3.8's reasoning is re-fed the same way, and turning on exact token capture for RL
(TITO) is the same flag as for Terminus-2.

We need it for three jobs, in order: Qwen3.8-27B teacher rollouts on Daytona, Snowball evaluation in the same harness,
and later SkyRL training of Snowball in this harness.

Why not the existing pieces:
- Harbor's `mini-swe-agent` agent installs itself **inside** the sandbox and calls the model from there, and a Daytona
  sandbox cannot reach vLLM on a Jupiter compute node.
- Kevin Li's `MiniSweAgentV1` (marin `kevin/repro-mini-coder`, `lib/harbor/.../mini_swe_agent_v1.py`, 336 lines) has
  exactly the right shape and ran on Daytona for #4898, but it targets **v1**. v2 changed the environment API to
  `execute(action: dict)`, moved the `Submitted` check into the environment, and changed the model interface. It also
  calls litellm directly, so it has no think parsing and no token capture. We copy its shape, not its code.
- MarinSkyRL's `mini_swe` runner is v1, runs on docker/singularity with no Daytona, and re-tokenizes after the fact,
  so it produces no behaviour logprobs.

## Goal (testable end state)

1. **Faithful.** Given the same scripted model replies, the messages our agent builds are identical to what upstream
   mini-swe-agent 2.4.6 builds with the same config: system prompt, instance prompt, observation and format-error
   templates, truncation, and submit handling.
2. **Works on Daytona** with Qwen3.8-27B and with Snowball, and finishes with zero harness errors on a smoke run.
3. **RL-ready (Stage 5).** With `collect_rollout_details: true`, every turn satisfies
   `prompt[t] = prompt[t-1] + completion[t-1] + observation` with no fallback, and MarinSkyRL accepts the agent with
   TIS on.

## Design

Three small pieces around the unmodified upstream `DefaultAgent` (pinned `mini-swe-agent==2.4.6`):

- **`HarborEnvironment`** (about 60 lines), implementing v2's environment protocol:
  - `execute(action, cwd, timeout)` calls `environment.exec` through `asyncio.run_coroutine_threadsafe` on Harbor's
    loop, then `.result()`.
  - It matches the reference Docker environment: `cwd=/testbed` for SWE configs; the config's env vars (`PAGER=cat`,
    `MANPAGER=cat`, `LESS=-R`, `PIP_PROGRESS_BAR=off`, `TQDM_DISABLE=1`); `bash -c`; stderr merged into stdout in
    order, by running `{ cmd; } 2>&1` rather than joining two streams.
  - Harbor reports a timeout as rc 124. The adapter maps it to the reference timeout result: returncode −1 and
    `exception_info` "An error occurred while executing the command: …".
  - `_check_finished` is copied verbatim from v2 `docker.py`: it raises `Submitted` on the
    `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` first line with rc 0.
- **`HarborModel`**, a subclass of v2's `LitellmTextbasedModel` (text mode, one `mswea_bash_command` block per turn):
  - It overrides only the query call, so upstream action parsing and observation formatting are kept.
  - Each call is forwarded to a per-episode Harbor `Chat` wrapping `LiteLLM`: `api_base`, `model_info`, `extra_body`,
    `llm_call_kwargs`, `interleaved_thinking`, `collect_rollout_details`, `session_id`.
  - It asserts that mini-swe's message list and `Chat`'s history agree after every turn.
  - Think spans are removed from the text before the action is parsed. An inline Snowball think span that quotes a
    code block must not trip the one-block rule.
  - Cost tracking is `ignore_errors`, since vLLM models have no price.
- **`MiniSweAgentHost(BaseAgent)`** (about 150 lines), shaped like Kevin's agent:
  - It loads the chosen upstream config yaml **verbatim** from the installed package; no local copy of the templates.
  - It runs `DefaultAgent.run` on a dedicated thread pool sized to concurrency.
  - In a `finally` block (so a timeout still keeps them), it fills `AgentContext`: tokens, `rollout_details`, and
    `metadata = {all_messages, stop_reason, exit_status, summarization_count: 0}`.
  - It writes the upstream trajectory json unchanged plus ATIF, built from `Chat` the way Terminus-2's
    `_build_trajectory` builds it, with token ids kept.
  - Registration: `AgentName.MINI_SWE_AGENT_HOST` and an `AgentFactory` entry. `import_path` works before the enum
    ships.

**Why text mode, not v2's default tool calls.** Snowball has no tool-call parser. A `tool` role message also breaks
the TITO continuation, which is why MarinSkyRL rates terminus-kira "completions only". Qwen3.8 is fine either way.
The cost is that swebench.com's v2 rows may use the tool-call config, so our numbers are not leaderboard-identical;
check which config swebench.com used before quoting any comparison.

**Why reuse the upstream loop instead of porting it.** Parity then holds by construction. Only the three adapters are
ours, and each one is checked against the reference in Stage 1.

## Stage map

| Stage | Title | What | Cost | Gate |
|---|---|---|---|---|
| 0 | Pin and replay harness | Pin 2.4.6; build a CPU test that runs upstream `DefaultAgent` with a scripted fake model and a fake env and records messages | CPU | Replay reproduces the upstream trajectory for 3 scripted episodes (submit, format error ×3, step limit) |
| 1 | Agent, CPU | The three pieces plus registration; fake Harbor env + fake `LiteLLM` | CPU | Messages identical to Stage 0 on all scripted episodes; existing agents' tests byte-identical |
| 2 | Qwen3.8 on Daytona | Serve Qwen3.8-27B (FP8, 1 Jupiter node, DP4) and run 5 R2E-Gym tasks via the Daytona path | ~1 node-h | 0 harness errors; the rendered prompt shows prior reasoning re-fed; `which python` in the sandbox is the repo env; a 70 s command times out and its process is gone |
| 3 | Snowball on Daytona | Same 5 tasks with the `trained` Snowball policy (`hosted_vllm/snowball`, `skip_special_tokens: false`, no temperature, EAGLE-3 server) | ~1 node-h | 0 harness errors; think spans stripped before parsing (no false format errors); format-error rate reported |
| 4 | Teacher generation | Real run: Qwen3.8 on the chosen pools | own cost line + go | Per the SFT plan's gates |
| 5 | RL enablement | `collect_rollout_details` on; MarinSkyRL: add the agent to `_HARBOR_EVIDENCE_PROFILES` and `harbor_agent_names.py`, bump the harbor-config wheel, teach `detect_termination_signals` the submit sentinel | 2-step RL smoke, own cost line | 100 % of turns TITO-exact, 0 fallbacks; `validate_trajectory_runner_capabilities` passes with TIS on |

Critical path: 0 → 1 → 2 → 4. Stage 3 can run beside 2. Stage 5 waits until Snowball RL in this harness is wanted.

Stage 5 fix list (from the 09-25 CPU TITO audit, `ai_memory/active/snowball-sft/research/2026-09-25_msa_tool_mode_tito.md`,
check script `data/mini_swe_host/tito_tool_mode_check.py` @b6d4748c). A tool turn does not by itself break TITO; the
chain breaks only because `Chat.append_tool_results` resets it on purpose and because today's tool mode sends the
re-rendered history. Needed before exact-token RL in this harness:
- **H1** `build_continuation_prompt_token_ids` takes a list of follow-up messages (`[*probe_base, *followups]`).
- **H2** a `Chat` method that continues the exact chain with tool (or tool + user) follow-ups; `append_tool_results`
  stops resetting when a token chain exists.
- **H3** a `Chat` rewind for a rejected reply (previous prompt minus the generation prompt + the rendered user turn +
  generation prompt), keeping upstream's history exactly; the rejected turn goes to `extra`, not the token lists.
- **A1** `HarborToolModel` uses that chain (turn 1 on the chat path, then H2; after a `FormatError`, H3).
- **A2** `name: "bash"` on every tool message — done (`cdd5b008`).
- **A3** parse the call from the sampled text only to drive upstream's action parsing; never feed a re-rendered
  assistant turn back into the prompt; keep the raw reply and raw arguments in `extra`.
- **A4** tools rendered without `tool_choice: auto` on parser-less serves — done (`tool_choice: "none"`).
- **M1** MarinSkyRL: `mini-swe-agent-host` in `harbor_agent_names.py` and `_HARBOR_EVIDENCE_PROFILES`, only after
  H1–H3 and A1–A3.

## Global invariants

- **Additive only.** No existing agent, environment or LLM code path changes. Terminus-2 and the installed
  `mini-swe-agent` behave identically; mini-swe-agent is an optional extra and is imported lazily inside the new agent.
- **Upstream templates are never edited.** Every behaviour change goes through config keys we pass, and each one is
  listed in the run's metadata.
- **One upstream version at a time.** 2.4.6 is recorded in the trajectory; any bump reruns Stage 0/1.
- **Import hygiene:** `MSWEA_SILENT_STARTUP=1`, `MSWEA_GLOBAL_CONFIG_DIR` set to a temp dir, no global cost or call
  limits, and a fresh `DefaultAgent` per episode (it does not reset `n_calls` or `cost`).
- **No blocking on Harbor's event loop.** Model and exec calls from a worker thread use `run_coroutine_threadsafe`,
  and nothing calls them from the loop thread (that deadlocks). The 09-11 coordinator stall was I/O on this loop.

## Decisions for Luke (before Stage 2)

1. **Config.** Recommend `benchmarks/swebench_backticks.yaml` (text-mode v2 SWE config: `/testbed`,
   `step_limit: 250`, 60 s command timeout) for R2E-Gym and SWE-smith pools, and `mini_textbased.yaml` later for
   terminal-style tasks. Its submit step cats a patch that Harbor's verifier ignores; that is harmless and kept
   verbatim.
2. **Context overflow.** mini-swe has no summarization. Recommend ending the episode as `context_exceeded` (the
   reference behaviour) rather than adding Terminus-style recovery. Qwen3.8 runs at 256k, so it rarely matters for
   the teacher; for Snowball at 64k it will.
3. **Step limit** for Snowball: keep 250 or match our Terminus-2 turn budget.

## Borrow map (anchors from 2026-09-24 reads; they drift, reconfirm at implementation)

- mini-swe-agent 2.4.6: `agents/default.py` (loop 88–124, `query` limits 130–152, save 159–190);
  `environments/docker.py` (execute 101–136, `_check_finished` 140–151); `models/litellm_textbased_model.py`
  (block regex :8); `models/litellm_model.py` (`_query` :64–71, stored message 99–105, cost 108–126);
  `config/benchmarks/swebench_backticks.yaml` (limits 174–179, env 115–128 in `swebench.yaml`).
- harbor upstream/main 761fb516: `agents/base.py:BaseAgent`, `TurnCapExhaustedError`;
  `agents/terminus_2/terminus_2.py` (`__init__` kwargs, `finally` context fill, `_build_trajectory`);
  `llms/lite_llm.py:LiteLLM` (`build_continuation_prompt_token_ids`, `_split_completion_reasoning`);
  `llms/chat.py:Chat`; `environments/daytona/environment.py:_sandbox_exec` (rc 124 timeout, fresh `bash -c` per call);
  `harbor_config/models/agent/name.py:AgentName`; `agents/factory.py:AgentFactory._AGENTS`;
  `agents/installed/mini_swe_agent.py:convert_mini_swe_agent_to_atif` (tool-call format, reference only).
- Snowball think handling: `lukedhlee/terminus2-think-parity*` (not on upstream main); read
  `.claude/projects/harbor/ops.md` § "Terminus-2 under Snowball" before Stage 3.
- MarinSkyRL origin/main 72cc492d: `trajectory_runners/harbor/runner.py` (`_process_trial_result`,
  `detect_termination_signals`), `config/trajectory_runner_capabilities.py:_HARBOR_EVIDENCE_PROFILES`,
  `marinskyrl/harbor_agent_names.py`, `pyproject.toml:265` (pinned harbor-config wheel).
- Shape reference: marin `kevin/repro-mini-coder` `lib/harbor/src/harbor/agents/installed/mini_swe_agent_v1.py`.

## Validation discipline

Each stage checks the default-off contract first (existing agents' tests unchanged), then the new behaviour. Stages 2
and 3 each need a one-sentence cost line and Luke's go before submission. Stage 2 and 3 readouts report the counts:
harness errors, format errors, timeouts, submit rate, and pass rate on the 5 tasks. The pass rate is a sanity check,
not a benchmark.

## 2026-09-25 revision: relay rollouts, both harnesses, and the smoke budget (Luke)

**Why now.** The relay experiment may run in both Terminus-2 and mini-swe-agent v2, so this agent is on the critical
path for the second harness. Build Stages 0–3 now; Stage 4 (real teacher generation) waits for its own cost line and go.

**What changes from the plan above**

- **Pools for the smokes.** R2E-Gym's Daytona snapshots were purged and the org has 1 free slot, so Stages 2–3 run on
  the live CalibForge pool (3 snapshots, `data/calibforge/daytona/`, rebuild with its `restore.sh`), 5–10 tasks. A
  SWE smoke waits until SWE-smith snapshots exist.
- **Config per pool.** CalibForge is terminal-style, not a repo in `/testbed`: use upstream `mini_textbased.yaml` with
  the task's own workdir (`/app`), not `swebench_backticks.yaml`. SWE pools keep `swebench_backticks.yaml`. Both are
  text mode, loaded verbatim from the pinned package.
- **Relay hook (new requirement).** mini-swe-agent has no "are you sure?" turn: the episode ends the moment a command's
  first output line is `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`. The relay router (an OpenAI-compatible proxy set as
  LiteLLM `api_base`) must therefore judge the student's reply **before it runs**: if its bash block contains the
  sentinel, discard it and let the teacher answer the same request. Nothing in this agent changes for that; the
  requirement on the agent is only that every model call goes through `api_base` with the full message list, and that
  `session_id` is stable per episode so the router can keep its sticky hand-off state.
  - **Checked 2026-09-25 (Stage 1 tests + the dry run): the agent meets it.** Every model call is one Harbor
    `LiteLLM.call` on the chat-completions path to `api_base`, carrying the whole message list (system, task, every
    reply and observation so far; after a format error, the two consecutive user turns upstream builds). `session_id`
    is set once per episode (a fresh uuid, or the agent kwarg) and sent in `extra_body` of every request, including
    retries. The reply is parsed and executed only after the call returns, so a proxy that sees
    `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` in the reply's `mswea_bash_command` block can substitute the teacher's reply
    before anything runs. Two details for the router: the sentinel only ends the episode when it is the **first
    output line with rc 0** (`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt` for SWE, a bare `echo` for
    mini_textbased), and a reply whose think span merely mentions the sentinel inside a quoted block is not an action,
    so the router should apply the same think-span stripping (`adapters.strip_think_spans`) before matching.
- **Trigger detectors.** `data/relay/triggers/relay_triggers.py` (being calibrated now) must accept this agent's
  message format too: text-mode bash blocks, observations with `returncode`, no tmux scrollback.
- **Decisions taken (defaults; Luke can override).** Text mode; context overflow ends the episode as
  `context_exceeded`; Snowball's step limit matches our Terminus-2 turn budget; the eval for models trained on
  mini-swe-agent data also runs in mini-swe-agent (TB2 through Harbor with this agent; unverified, Stage 2 checks it).

**Smoke budget (Stages 2–3):** one Jupiter node per stage, ≤1 h each, ≤2.5 node-hours total including start-up.
Qwen3.8 is served exactly as the bench found best (TP1 × DP4, MTP 2, 64k, the `relay-extra` torchvision side dir), not
FP8. Snowball is served with its eval policy (reasoning parser, EAGLE-3 draft). Each smoke reports: harness errors,
format errors, timeouts, submit rate, pass rate, median turns, and whether the submit sentinel ended the episode.
