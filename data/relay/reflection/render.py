"""Native Grug template rendering; offsets establish exact reflection-only labels."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .core import file_digest


def load_renderer(tokenizer_dir):
    import jinja2
    from tokenizers import Tokenizer

    root = Path(tokenizer_dir)
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    src = (root / "chat_template.jinja").read_text()
    src = re.sub(r"\{%-?\s*(end)?generation\s*-?%\}", "", src)
    env = jinja2.Environment(
        trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
    )
    env.filters["tojson"] = lambda x, **kw: json.dumps(
        x, ensure_ascii=kw.pop("ensure_ascii", False), **kw
    )

    def fail(message):
        raise jinja2.TemplateError(message)

    env.globals["raise_exception"] = fail
    template = env.from_string(src)
    tc = json.loads((root / "tokenizer_config.json").read_text())
    bos = tc.get("bos_token")
    if isinstance(bos, dict):
        bos = bos["content"]

    def format_messages(messages, add_generation_prompt):
        return template.render(
            messages=messages,
            bos_token=bos,
            add_generation_prompt=add_generation_prompt,
            tools=None,
        )

    return tokenizer, format_messages, file_digest(root / "chat_template.jinja")


# Turn and document boundaries of either author's template. A completion that writes one inside its reasoning or
# answer (an end-of-turn mid-text, a new header, a new document) has broken the turn it is supposed to be; rendering it
# would train that boundary mid-turn, so it is rejected and counted. Think markers are not in this list: the RLVR1 student
# often closes its thought two or more times, and the template reproduces exactly the tokens it sampled.
CONTROL_MARKERS = (
    "<|eot_id|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|im_start|>",
    "<|im_end|>",
)


def render_reflection(
    record, tokenizer, format_messages, sequence_length, template_digest
):
    message = record["message"]
    if any(
        marker in (message.get(field) or "")
        for marker in CONTROL_MARKERS
        for field in ("content", "reasoning_content")
    ):
        raise ValueError("control_token_in_completion")
    messages = [{"role": "user", "content": record["prompt"]}]
    prefix = format_messages(messages, True)
    text = format_messages(messages + [record["message"]], False)
    if not text.startswith(prefix):
        raise ValueError(
            "native template prefix is not identical; refusing guessed masks"
        )
    encoded = tokenizer.encode(text, add_special_tokens=False)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False).ids
    if encoded.ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError(
            "token boundary merges across prompt/target; refusing guessed masks"
        )
    if not encoded.ids or len(encoded.ids) > sequence_length:
        raise ValueError("sequence_overflow")
    boundary = len(prefix)
    loss = []
    for start, end in encoded.offsets:
        if start < boundary < end:
            raise ValueError("token crosses target boundary")
        loss.append(int(start >= boundary and end > start))
    if sum(loss) == 0 or loss[0]:
        raise ValueError("invalid or empty reflection target")
    if any(loss[: len(prefix_ids)]):
        raise ValueError("prompt receives loss")
    # An intact, native end-of-turn must be supervised; truncation is never accepted.
    if not loss[-1]:
        raise ValueError("native completion suffix missing loss")
    return {
        "sid": record["sid"],
        "source_id": record["source_id"],
        "task": record["task"],
        "ids": encoded.ids,
        "loss": loss,
        "n_tokens": len(loss),
        "trained_tokens": sum(loss),
        "prompt_tokens": len(prefix_ids),
        "fits": True,
        "template_sha256": template_digest,
        "trajectory_author": record["trajectory_author"],
        "reflection_author": record["reflection_author"],
        "policy_version": record["policy_version"],
    }


def render_records(records, tokenizer_dir, sequence_length):
    tok, fmt, sha = load_renderer(tokenizer_dir)
    rows = []
    rejected = {}
    for record in records:
        try:
            rows.append(render_reflection(record, tok, fmt, sequence_length, sha))
        except ValueError as error:
            why = str(error)
            rejected[why] = rejected.get(why, 0) + 1
    return rows, rejected
