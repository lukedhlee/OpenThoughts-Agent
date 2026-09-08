"""How often does a model's own sampled token stream survive a text round trip?

MarinSkyRL#520's mechanism is a property of the tokenizer, not of the model: the
IDs the model sampled are decoded to text, and the next turn re-encodes that text
canonically. Wherever the sampled IDs were not the canonical encoding of their own
text, the re-encoding differs and the served-stream prefix invariant breaks.

This measures that directly on recorded completions -- no GPU, no server. Point it
at a retained rollout capture and its serving tokenizer:

    python hazard_census.py --tokenizer tokenizer.json --capture arm3_first_capture.json

Reported per turn: whether re-encoding the turn's own decoded text reproduces its
sampled IDs, and if not, how far into the turn the first divergence lands.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_tokenizer(path: str):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(path)


def roundtrip(tok, ids: List[int]) -> Dict[str, Any]:
    """Decode ``ids`` and re-encode; report whether the IDs came back unchanged."""
    text = tok.decode(ids, skip_special_tokens=False)
    re_ids = tok.encode(text, add_special_tokens=False).ids
    if re_ids == ids:
        return {"stable": True, "n_ids": len(ids), "n_re_ids": len(re_ids)}
    offset = next((i for i, (a, b) in enumerate(zip(ids, re_ids)) if a != b), min(len(ids), len(re_ids)))
    return {
        "stable": False,
        "n_ids": len(ids),
        "n_re_ids": len(re_ids),
        "first_divergence": offset,
        "fraction_into_turn": offset / max(len(ids), 1),
        "sampled_window": ids[max(0, offset - 2) : offset + 4],
        "reencoded_window": re_ids[max(0, offset - 2) : offset + 4],
        "sampled_pieces": [tok.id_to_token(i) for i in ids[max(0, offset - 2) : offset + 4]],
        "reencoded_pieces": [tok.id_to_token(i) for i in re_ids[max(0, offset - 2) : offset + 4]],
    }


def iter_completions(capture: Dict[str, Any]) -> Iterable[List[int]]:
    for detail in capture.get("rollout_details") or []:
        for ids in detail.get("completion_token_ids") or []:
            if ids:
                yield list(ids)


def span_sweep(tok, streams: List[List[int]], lengths: List[int], samples: int, rng: random.Random) -> List[Dict[str, Any]]:
    """Hazard rate as a function of span length, sampled from the same streams.

    A turn is a span of some length; the longer the span, the more chances it has
    to contain an unstable boundary. This says how the per-turn rate should scale
    with completion length.
    """
    flat = [ids for ids in streams if len(ids) >= 8]
    out = []
    for L in lengths:
        usable = [ids for ids in flat if len(ids) > L]
        if not usable:
            continue
        unstable = 0
        for _ in range(samples):
            ids = rng.choice(usable)
            start = rng.randrange(0, len(ids) - L)
            if not roundtrip(tok, ids[start : start + L])["stable"]:
                unstable += 1
        out.append({"span_length": L, "samples": samples, "unstable": unstable, "rate": unstable / samples})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="path to tokenizer.json")
    ap.add_argument("--capture", required=True, help="retained rollout capture (JSON)")
    ap.add_argument("--span-lengths", type=int, nargs="+", default=[16, 64, 256, 1024])
    ap.add_argument("--span-samples", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="hazard_census.json")
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer)
    capture = json.loads(Path(args.capture).read_text())
    streams = list(iter_completions(capture))

    per_turn = [roundtrip(tok, ids) for ids in streams]
    unstable = [r for r in per_turn if not r["stable"]]
    total_tokens = sum(r["n_ids"] for r in per_turn)

    rng = random.Random(args.seed)
    sweep = span_sweep(tok, streams, args.span_lengths, args.span_samples, rng)

    result = {
        "tokenizer": args.tokenizer,
        "capture": args.capture,
        "n_turns": len(per_turn),
        "n_unstable_turns": len(unstable),
        "unstable_turn_fraction": len(unstable) / max(len(per_turn), 1),
        "total_sampled_tokens": total_tokens,
        "unstable_per_1k_tokens": 1000 * len(unstable) / max(total_tokens, 1),
        "span_sweep": sweep,
        "examples": unstable[:10],
    }
    Path(args.out).write_text(json.dumps(result, indent=2))

    print(f"turns={result['n_turns']} unstable={result['n_unstable_turns']} "
          f"({result['unstable_turn_fraction']:.1%})  sampled_tokens={total_tokens}")
    for row in sweep:
        print(f"  span {row['span_length']:>5}: {row['rate']:.1%} unstable ({row['unstable']}/{row['samples']})")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
