"""What does declining actually cost? Rebuild SkyRL's fallback and diff it.

When full TITO declines, SkyRL re-tokenizes the conversation text and splices the
recorded completion IDs back over the assistant spans. The result is a valid
training sequence -- but it is not the sequence the model was served. This
measures the gap on a real trajectory: how many positions of the training
sequence differ from what inference actually conditioned on, and where they sit.

Needs only a recorded capture and the serving checkpoint's tokenizer + template,
so it runs anywhere.
"""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
from typing import Any, Dict, List


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--model", required=True, help="HF id or local path with tokenizer + chat template")
    ap.add_argument("--mode", default="text")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    capture = json.loads(Path(args.capture).read_text())
    if "rollout_details" in capture:
        cases = [(capture["messages"], capture["rollout_details"][0]["prompt_token_ids"],
                  capture["rollout_details"][0]["completion_token_ids"])]
    else:
        # A probe dump: every recorded text-transport trajectory is a case.
        cases = [
            (t["messages"], t["prompt_token_ids"], t["completion_token_ids"])
            for t in (capture[args.mode]["trajectories"] or [])
            if t.get("messages")
        ]
    results = [analyse(tok, m, p, c) for m, p, c in cases]
    exact = sum(1 for r in results if r["divergent_positions_served_side"] == 0)
    print(json.dumps({
        "cases": len(results),
        "context_reproduced_exactly": exact,
        "fraction": exact / max(len(results), 1),
        "examples": results[:2],
    }, indent=2)[:2500])
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
    return 0


def analyse(tok, messages, prompts, completions) -> Dict[str, Any]:

    import difflib
    # What inference actually ran on: the last turn's prompt plus its completion.
    served = list(prompts[-1]) + list(completions[-1])

    # What the fallback builds: the same conversation, tokenized from text.
    # (Assistant spans are then spliced back to the sampled ids, so any position
    # inside an assistant span is correct by construction; everything else keeps
    # whatever this tokenization produced.)
    n_used = len(completions)
    trimmed: List[Dict[str, Any]] = []
    seen_assistant = 0
    for m in messages:
        trimmed.append(m)
        if m.get("role") == "assistant":
            seen_assistant += 1
            if seen_assistant == n_used:
                break
    refetched = tok.apply_chat_template(trimmed, tokenize=True, add_generation_prompt=False)
    if hasattr(refetched, "input_ids"):
        refetched = refetched.input_ids
    fallback = [int(x) for x in refetched]

    sm = difflib.SequenceMatcher(a=served, b=fallback, autojunk=False)
    matched = sum(bl.size for bl in sm.get_matching_blocks())
    ops = [op for op in sm.get_opcodes() if op[0] != "equal"]

    # Where do the differences land -- inside what the model wrote, or in the
    # observation/template text around it?
    assistant_spans = []
    for t in range(len(completions)):
        start = len(prompts[t])
        assistant_spans.append((start, start + len(completions[t])))

    def in_assistant(i: int) -> bool:
        return any(lo <= i < hi for lo, hi in assistant_spans)

    changed_in_assistant = 0
    changed_elsewhere = 0
    for tag, i1, i2, _j1, _j2 in ops:
        for i in range(i1, i2):
            if in_assistant(i):
                changed_in_assistant += 1
            else:
                changed_elsewhere += 1

    result = {
        "served_length": len(served),
        "fallback_length": len(fallback),
        "length_delta": len(fallback) - len(served),
        "matched_positions": matched,
        "divergent_positions_served_side": len(served) - matched,
        "divergent_fraction": (len(served) - matched) / max(len(served), 1),
        "divergent_inside_assistant_spans": changed_in_assistant,
        "divergent_outside_assistant_spans": changed_elsewhere,
        "n_edit_blocks": len(ops),
        "first_edits": [
            {"tag": tag, "served_slice": [i1, i2], "fallback_slice": [j1, j2],
             "served_pieces": [tok.convert_ids_to_tokens(x) for x in served[i1:i2][:6]],
             "fallback_pieces": [tok.convert_ids_to_tokens(x) for x in fallback[j1:j2][:6]]}
            for tag, i1, i2, j1, j2 in ops[:6]
        ],
    }
    return result


if __name__ == "__main__":
    raise SystemExit(main())
