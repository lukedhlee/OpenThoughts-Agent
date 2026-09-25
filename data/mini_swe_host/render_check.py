#!/usr/bin/env python3
"""CPU check: what 09-21's chat template renders for a mini-swe-agent-host tool-mode request.

Runs the host agent (tool mode, mini.yaml) on a two-turn scripted episode whose first reply is written the way the
09-21 Datakit SFT emits it without a server-side parser (inline think span, then an inline
``<tool_call>{"name": "bash", "arguments": {...}}</tool_call>``), captures the second request exactly as the agent
sends it (messages + tools), and renders it with the model's own chat_template.jinja through transformers, as vLLM
does. Prints the rendered prompt and checks: the Tools block with the bash schema, the prior reasoning inside
<|start_think|>…<|end_think|>, the prior call re-rendered as a <tool_call> block, and the observation as a
<tool_response name="bash">.

    PYTHONPATH=<harbor worktree>:<harbor worktree>/src:<mini-swe-agent 2.4.6 dir> \
      python render_check.py <chat_template.jinja> [--interleaved]
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")

from transformers.utils.chat_template_utils import render_jinja_template  # noqa: E402

from harbor.llms.lite_llm import LiteLLM  # noqa: E402
from tests.unit.agents.mini_swe_agent_host.host_replay import run_host  # noqa: E402
from tests.unit.agents.mini_swe_agent_host.replay import TOOL, TOOL_MODE, Episode, Reply  # noqa: E402

FIRST = (
    "<|start_think|>I should look at the directory first. I could run "
    "<tool_call>{\"name\": \"bash\", \"arguments\": {\"command\": \"rm -rf /\"}}</tool_call> but no.<|end_think|>"
    "Let me list the files.\n<tool_call>\n{\"name\": \"bash\", \"arguments\": {\"command\": \"ls -la\"}}\n</tool_call>"
)
SUBMIT = (
    "<|start_think|>Done.<|end_think|><tool_call>\n{\"name\": \"bash\", \"arguments\": "
    "{\"command\": \"echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\"}}\n</tool_call>"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("template")
    parser.add_argument("--interleaved", action="store_true", help="re-send reasoning as reasoning_content")
    args = parser.parse_args()
    template = Path(args.template).read_text()
    episode = Episode(config_file=TOOL, model_class=TOOL_MODE, replies=(Reply(FIRST), Reply(SUBMIT)),
                      expected_exit_status="Submitted")
    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch) / "work"
        work.mkdir()
        replay, context = asyncio.run(
            run_host(episode, work, Path(scratch) / "logs", tool_calls="none",
                     interleaved_thinking=args.interleaved)
        )
    request = replay.requests[1]
    tools = replay.request_tools[1]
    rendered = render_jinja_template(
        conversations=[request], tools=tools, chat_template=template, add_generation_prompt=True,
        bos_token="<|begin_of_text|>",
    )[0][0]
    print(rendered)
    print("=" * 80)
    checks = {
        "exit_status Submitted": replay.exit_status == "Submitted",
        "first reply executed ls -la, not the quoted rm": replay.messages[3]["extra"]["raw_output"].startswith("total"),
        "Tools block with bash schema": "### Tools" in rendered and '"name": "bash"' in rendered,
        "reasoning mode /think": "Reasoning: /think" in rendered,
        "prior reasoning inside think markers":
            "<|start_think|>I should look at the directory first." in rendered and "but no.<|end_think|>" in rendered,
        "prior call re-rendered as <tool_call>":
            '<tool_call>\n{"name": "bash", "arguments": {"command": "ls -la"}}\n</tool_call>' in rendered,
        "observation as tool_response name=bash": '<tool_response name="bash">' in rendered,
        "quoted call not re-rendered as a real call": rendered.count('"command": "rm -rf /"') <= 1,
        "ends with the assistant header": rendered.endswith("<|start_header_id|>assistant<|end_header_id|>\n"),
    }
    for name, ok in checks.items():
        print(("PASS " if ok else "FAIL ") + name)
    print("request messages:", json.dumps([{k: (v if k != "content" else v[:60]) for k, v in m.items()}
                                           for m in request[2:]], indent=1))
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
