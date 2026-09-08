"""Does the offline round-trip test predict what actually happened on the server?

The cheap census decodes a turn's sampled IDs and re-encodes them in isolation.
The expensive truth is whether the NEXT served prompt actually preserved them.
If the two agree, any recorded capture plus its tokenizer is enough to measure
the decline rate for a model -- no GPU, no live arm.

    python predictor_check.py --dump probe_dump.json --tokenizer Qwen/Qwen3-1.7B
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tito_checks import completion_survived  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="probe.py output written with --dump-trajectories")
    ap.add_argument("--tokenizer", required=True, help="HF id or path of the SERVING tokenizer")
    ap.add_argument("--mode", default="text", choices=["text", "tokens"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    data = json.loads(Path(args.dump).read_text())
    trajectories = data[args.mode]["trajectories"] or []

    tp = tn = fp = fn = 0
    for traj in trajectories:
        p, c = traj["prompt_token_ids"], traj["completion_token_ids"]
        for t in range(1, len(p)):
            if not p[t - 1] or not c[t - 1] or not p[t]:
                continue
            survived = completion_survived(p[t - 1], c[t - 1], p[t])
            text = tok.decode(c[t - 1], skip_special_tokens=False)
            stable = tok(text, add_special_tokens=False)["input_ids"] == list(c[t - 1])
            if stable and survived:
                tn += 1
            elif not stable and not survived:
                tp += 1
            elif stable and not survived:
                fn += 1  # broke on the server, round trip said it would not
            else:
                fp += 1  # round trip cried wolf

    n = tp + tn + fp + fn
    result = {
        "mode": args.mode,
        "boundaries": n,
        "predicted_and_broke": tp,
        "predicted_stable_and_held": tn,
        "missed": fn,
        "false_alarms": fp,
        "agreement": (tp + tn) / max(n, 1),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
