"""CPU check: can mini-swe-agent-host in TOOL mode keep an exact token chain (TITO) with the 09-21 student?

No GPU, no vLLM. The 09-21 chat template is rendered with transformers' own jinja renderer (the one vLLM's
``apply_hf_chat_template`` calls) and tokenized with its ``tokenizer.json``. Real 09-21 replies come from the
text-mode smoke (job 2021265, mini-swe-agent trajectories); each accepted reply is re-expressed as the tool-mode
reply the model would sample (same reasoning, same pre-action text, the same command inside a ``<tool_call>`` JSON
block), and each observation becomes a ``tool`` message.

Per boundary (end of assistant turn t -> prompt of turn t+1) it checks three things:

  M1  observation suffix: Harbor's TITO continuation (``LiteLLM.build_continuation_prompt_token_ids``, ported
      verbatim and generalised from one user string to a list of follow-up messages) equals the tokens the
      template itself renders for that observation in the full conversation.
  M2  TITO vs re-render: prompt_t + sampled_t + suffix_t == tokenize(render(structured history up to t+1)).
      This is what the chain would look like if anything re-rendered the history (the chat path every turn, or
      ``Chat.append_tool_results``, which resets the chain). "local" checks one boundary on a rendered prefix,
      "cumulative" the whole chain so far. Differences are classified.
  M3  format-error rewind: prompt_t minus the generation prompt + the rendered format-error user turn is a clean
      token extension of prompt_t's history (upstream v2 drops the rejected reply).

Variants of the tool-mode reply: ``native`` writes ``content<tool_call>`` with nothing between (09-21's own habit
and the template's rendering), ``asis`` keeps the whitespace the reply had before its code fence, ``ascii`` writes
the arguments JSON with ``\\u`` escapes. ``--no-tool-name`` sends tool messages with only ``tool_call_id``, as
upstream v2 builds them.

Sampled IDs here are the canonical encoding of the sampled text, so the check cannot see a non-canonical token cut
from the sampler (the #520 effect); the exact-token path never re-tokenizes sampled IDs, so it is immune to it.

Usage:
  python tito_tool_mode_check.py --model-dir <dir with chat_template.jinja + tokenizer.json> \
      --runs <job dir with */attempts/000/agent/mini-swe-agent.trajectory.json> \
      [--variant native|asis|ascii] [--no-tool-name]
"""

import argparse
import glob
import json
import os
import re
from collections import Counter

from tokenizers import Tokenizer
from transformers.utils.chat_template_utils import render_jinja_template

BASH_TOOL = {  # minisweagent 2.4.6 models/utils/actions_toolcall.py
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The bash command to execute"}},
            "required": ["command"],
        },
    },
}
START_THINK, END_THINK, EOT = "<|start_think|>", "<|end_think|>", "<|eot_id|>"
ACTION_RE = re.compile(r"```mswea_bash_command\s*\n(.*?)\n```", re.DOTALL)  # upstream text-mode action_regex

# Harbor's probe conversation (harbor/llms/lite_llm.py, _TOKEN_IN_TOKEN_OUT_TEMPLATE_*), verbatim.
PROBE_BASE = [
    {"role": "user", "content": "I am a user."},
    {"role": "assistant", "content": "ok", "reasoning_content": "thinking"},
]
PROBE_ALT = [
    {"role": "user", "content": "I am a user."},
    {"role": "assistant", "content": "different", "reasoning_content": "thinking"},
]
PROBE_FOLLOWUP = {"role": "user", "content": "I am another user."}


class Renderer:
    def __init__(self, model_dir: str, tools):
        self.template = open(os.path.join(model_dir, "chat_template.jinja")).read()
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.tools = tools
        self.eot_id = self.tok.token_to_id(EOT)
        self.gen_ids = self.enc("<|start_header_id|>assistant<|end_header_id|>\n")

    def text(self, messages, gen: bool) -> str:
        rendered, _ = render_jinja_template(
            [messages],
            tools=self.tools,
            chat_template=self.template,
            add_generation_prompt=gen,
            bos_token="<|begin_of_text|>",
        )
        return rendered[0]

    def enc(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def ids(self, messages, gen: bool) -> list[int]:
        # vLLM /tokenize and chat serving: render, then encode without adding special tokens.
        return self.enc(self.text(messages, gen))


def common_suffix(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[-1 - n] == b[-1 - n]:
        n += 1
    return a[len(a) - n :] if n else []


def suffix_not_already_generated(template_suffix, completion):
    for k in range(min(len(template_suffix), len(completion)), 0, -1):
        if completion[-k:] == template_suffix[:k]:
            return template_suffix[k:]
    return list(template_suffix)


def harbor_continuation(r: Renderer, followup: list[dict], completion: list[int]) -> list[int]:
    """``build_continuation_prompt_token_ids`` minus the prefix, with ``[user prompt]`` generalised to a list."""
    base_ids = r.ids([*PROBE_BASE, PROBE_FOLLOWUP], True)
    alt_ids = r.ids([*PROBE_ALT, PROBE_FOLLOWUP], True)
    end = len(base_ids) - len(common_suffix(base_ids, alt_ids))
    cont = r.ids([*PROBE_BASE, *followup], True)
    if cont[:end] != base_ids[:end]:
        raise ValueError("no stable continuation prefix")
    return suffix_not_already_generated(cont[end:], completion)


def split_reply(raw: str):
    """Leading think span, the text before the action block, the command; None if not a one-action reply."""
    if not raw.startswith(START_THINK):
        return None
    reasoning, closed, rest = raw[len(START_THINK) :].partition(END_THINK)
    if not closed:
        return None
    blocks = ACTION_RE.findall(rest)
    if len(blocks) != 1:
        return None
    before = rest[: rest.find("```mswea_bash_command")]
    return reasoning, before, blocks[0]


def tool_mode_reply(reasoning, before, command, variant, call_id):
    """(sampled text, structured assistant message) for the tool-mode version of one reply."""
    if variant == "ascii":  # the model escapes non-ASCII in its JSON
        args_json = json.dumps({"command": command}, ensure_ascii=True)
    else:  # the template's own tojson: ensure_ascii=False, ", " / ": "
        args_json = json.dumps({"command": command}, ensure_ascii=False)
    content = before if variant == "asis" else before.rstrip()
    call_text = '<tool_call>\n{"name": "bash", "arguments": ' + args_json + "}\n</tool_call>"
    sampled = START_THINK + reasoning + END_THINK + content + call_text
    structured = {
        "role": "assistant",
        "content": content,
        "reasoning_content": reasoning,
        # What vLLM's _postprocess_messages feeds the template: JSON-string arguments parsed to a dict.
        "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "bash", "arguments": {"command": command}}}],
    }
    return sampled, structured


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else None


def classify(r: Renderer, tito, render, reply_text, structured):
    i = first_diff(tito, render)
    if i is None:
        return "exact", ""
    ctx_t = r.tok.decode(tito[max(0, i - 3) : i + 6], skip_special_tokens=False)
    ctx_r = r.tok.decode(render[max(0, i - 3) : i + 6], skip_special_tokens=False)
    content = structured["content"]
    if content != content.strip():
        why = "template trims content"
    elif r.tok.decode(tito, skip_special_tokens=False) == r.tok.decode(render, skip_special_tokens=False):
        why = "same text, different token cut"
    else:
        why = "template renders the turn differently"
    return why, f"tito={ctx_t!r} render={ctx_r!r}"


def episodes(runs_dir):
    for path in sorted(glob.glob(os.path.join(runs_dir, "*/attempts/000/agent/mini-swe-agent.trajectory.json"))):
        yield path.split(os.sep)[-5], json.load(open(path))["messages"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--variant", default="native", choices=["native", "asis", "ascii"])
    ap.add_argument("--no-tool-name", action="store_true", help="tool messages carry only tool_call_id (upstream v2)")
    args = ap.parse_args()
    r = Renderer(args.model_dir, tools=[BASH_TOOL])

    counts = Counter()
    reasons = Counter()
    examples = {}
    for name, msgs in episodes(args.runs):
        history = [{"role": m["role"], "content": m["content"]} for m in msgs[:2]]  # system, instance
        prompt = r.ids(history, True)  # turn 1 goes through chat completions
        i = 2
        turn = 0
        while i < len(msgs):
            m = msgs[i]
            if m["role"] == "exit":
                break
            if m["role"] == "user" and m.get("extra", {}).get("interrupt_type") == "FormatError":
                # M3: upstream drops the reply; the next prompt is history + format-error user turn.
                assert prompt[-len(r.gen_ids) :] == r.gen_ids
                fe = {"role": "user", "content": m["content"]}
                rewind = prompt[: -len(r.gen_ids)] + harbor_continuation(r, [fe], [])[1:]  # drop the probe's eot
                # Reference: the rendered upstream history, with this chain's own tokens as prefix.
                full_prev = r.text(history, False)
                full_new = r.text([*history, fe], True)
                assert full_new.startswith(full_prev)
                ref = prompt[: -len(r.gen_ids)] + r.enc(full_new[len(full_prev) :])
                counts["fe_boundaries"] += 1
                counts["fe_rewind_exact"] += rewind == ref
                history.append(fe)
                prompt = rewind
                i += 1
                continue
            if m["role"] != "assistant":
                i += 1
                continue
            parts = split_reply(m["content"])
            obs = msgs[i + 1] if i + 1 < len(msgs) else None
            if parts is None or obs is None or obs["role"] != "user":
                counts["skipped_turns"] += 1
                break
            turn += 1
            call_id = f"call_{turn}"
            sampled_text, structured = tool_mode_reply(*parts, args.variant, call_id)
            sampled = r.enc(sampled_text) + [r.eot_id]
            tool_msg = {"role": "tool", "tool_call_id": call_id, "content": obs["content"]}
            if not args.no_tool_name:
                tool_msg["name"] = "bash"

            # M1: Harbor's continuation vs the template's own rendering of this observation.
            cont = harbor_continuation(r, [tool_msg], sampled)
            full_prev = r.text([*history, structured], False)
            full_new = r.text([*history, structured, tool_msg], True)
            assert full_new.startswith(full_prev), "rendered prefix changed when the tool turn was added"
            truth = r.enc(full_new[len(full_prev) :])
            counts["boundaries"] += 1
            counts["m1_suffix_exact"] += cont == truth
            if cont != truth and "m1" not in examples:
                examples["m1"] = (r.tok.decode(cont[:12], skip_special_tokens=False), r.tok.decode(truth[:12], skip_special_tokens=False))

            # M2: exact chain vs a full re-render of the structured history.
            # Local: this boundary alone, on top of a rendered prefix. Cumulative: the whole chain so far.
            tito_next = prompt + sampled + cont
            render_next = r.ids([*history, structured, tool_msg], True)
            local_tito = r.ids(history, True) + sampled + cont
            why, ctx = classify(r, local_tito, render_next, sampled_text, structured)
            reasons[why] += 1
            counts["m2_local_exact"] += why == "exact"
            counts["m2_cumulative_exact"] += tito_next == render_next
            if why != "exact" and why not in examples:
                examples[why] = ctx
            # Is the sampled segment itself what the template would render for the assistant turn?
            asst_text = full_prev[len(r.text(history, True)) :]
            counts["assistant_render_equals_sampled"] += r.enc(asst_text) == sampled

            history += [structured, tool_msg]
            prompt = tito_next
            i += 2
        counts["episodes"] += 1

    # Two tool calls in one reply: two tool messages must be rendered as one suffix.
    two = [
        {"role": "tool", "tool_call_id": "a", "name": "bash", "content": "out a"},
        {"role": "tool", "tool_call_id": "b", "name": "bash", "content": "out b"},
    ]
    first = harbor_continuation(r, [two[0]], [r.eot_id])
    second = harbor_continuation(r, [two[1]], [r.eot_id])
    together = harbor_continuation(r, two, [r.eot_id])
    counts["two_calls_suffix_is_concat"] = int(together == first[: -len(r.gen_ids)] + second)
    # A user turn right after a tool turn (e.g. a format error after an observation) in one follow-up.
    user = {"role": "user", "content": "Format error: no tool call."}
    tool_then_user = harbor_continuation(r, [two[0], user], [r.eot_id])
    user_alone = harbor_continuation(r, [user], [r.eot_id])
    counts["tool_then_user_suffix_is_concat"] = int(tool_then_user == first[: -len(r.gen_ids)] + user_alone)

    print(json.dumps({"variant": args.variant, "tool_name": not args.no_tool_name, **counts}, indent=1))
    print("M2 classes:", dict(reasons))
    for k, v in examples.items():
        print(f"example [{k}]: {v}")


if __name__ == "__main__":
    main()
