#!/usr/bin/env python3
"""Why do Daytona trajectories carry more tokens per turn than apptainer ones? Compare the terminal observations.

    python obs_size_compare.py --a-label apptainer --a run_1821359/apptainer_16 \
                               --b-label daytona   --b run_1822052/daytona_16 [--tokenizer <hf path>]

Reads every attempts/000/agent/trajectory.json (harbor ATIF) under each root. Per agent step it has the observation
text harbor fed back (New Terminal Output ...) and the exact prompt token count of the request that produced the step,
so the observation cost in tokens is the growth of the prompt from one step to the next minus the previous completion.

Prints, per backend: turns, observation chars and tokens per turn, and the share of observation characters taken by
shell prompts, trailing blanks, escape codes and full-width separator lines; then the prompt strings seen and, with a
tokenizer, how many tokens one prompt costs. Read-only, stdlib only (the tokenizer is optional).
"""

import argparse
import glob
import json
import os
import re
import statistics as st
from collections import Counter
from dataclasses import dataclass

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b[()][A-Za-z0-9]|\x1b[=>]|\r")
# a shell prompt at the start of a line: `root@<host>:<cwd># ` or `user@host:cwd$ ` or a bare `$ `
PROMPT = re.compile(r"^(?:[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+:[^\n#$]*[#$]|\$)(?= |$)", re.M)
SEP_LINE = re.compile(r"^[=_\-~*]{20,}\s*$", re.M)


@dataclass
class Summary:
    n: int
    mean_chars: float
    mean_tok: float
    pchars: int
    nprompts: int
    total: int


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def scan(root):
    per_turn = []  # (obs_chars, obs_tokens, prompt_chars, n_prompts, trailing, ansi, sep_chars, max_line, n_lines)
    prompts = Counter()
    max_line_hist = Counter()
    trajs = 0
    for f in sorted(glob.glob(os.path.join(root, "*", "attempts", "*", "agent", "trajectory.json"))):
        t = json.load(open(f))
        steps = t.get("steps", [])
        trajs += 1
        agent_steps = [s for s in steps if s.get("source") == "agent"]
        for i, s in enumerate(agent_steps):
            obs = "".join(r.get("content", "") for r in (s.get("observation") or {}).get("results", []))
            if not obs:
                continue
            # tokens this observation added = next prompt - this prompt - this completion (template overhead ~10)
            nxt = agent_steps[i + 1] if i + 1 < len(agent_steps) else None
            tok = None
            if nxt and s.get("metrics") and nxt.get("metrics"):
                tok = nxt["metrics"].get("prompt_tokens", 0) - s["metrics"].get("prompt_tokens", 0) - s["metrics"].get("completion_tokens", 0)
            ps = PROMPT.findall(obs)
            for p in ps:
                prompts[re.sub(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", "<uuid>", p)] += 1
            lines = obs.split("\n")
            trailing = sum(len(l) - len(l.rstrip(" ")) for l in lines)
            ansi = sum(len(m) for m in ANSI.findall(obs))
            sep_chars = sum(len(m) for m in SEP_LINE.findall(obs))
            ml = max((len(l) for l in lines), default=0)
            max_line_hist[ml] += 1
            per_turn.append((len(obs), tok, sum(len(p) for p in ps), len(ps), trailing, ansi, sep_chars, ml, len(lines)))
    return trajs, per_turn, prompts, max_line_hist


def summarize(label, trajs, per_turn, prompts, mlh, tokenizer=None):
    n = len(per_turn)
    chars = [r[0] for r in per_turn]
    toks = [r[1] for r in per_turn if r[1] is not None and r[1] >= 0]
    pchars = sum(r[2] for r in per_turn)
    nprompts = sum(r[3] for r in per_turn)
    trailing = sum(r[4] for r in per_turn)
    ansi = sum(r[5] for r in per_turn)
    sep = sum(r[6] for r in per_turn)
    total = sum(chars)
    print(f"\n== {label}: {trajs} trajectories, {n} observations ==")
    print(f"observation chars per turn: mean {st.mean(chars):,.0f}  p50 {st.median(chars):,.0f}  p90 {sorted(chars)[int(0.9*n)-1]:,.0f}  total {total:,}")
    if toks:
        print(f"observation tokens per turn (prompt growth minus completion): mean {st.mean(toks):,.0f}  p50 {st.median(toks):,.0f}  total {sum(toks):,}  chars/token {total/max(sum(toks),1):.2f}")
    print(f"shell prompts: {nprompts:,} ({nprompts/n:.1f} per turn), {pchars:,} chars = {pct(pchars,total):.1f} % of observation chars")
    print(f"trailing blanks {pct(trailing,total):.2f} %   escape/CR codes {pct(ansi,total):.2f} %   separator lines {pct(sep,total):.1f} %")
    lines = sum(r[8] for r in per_turn)
    print(f"lines per turn mean {lines/n:.0f}; longest line per turn p50 {st.median([r[7] for r in per_turn]):.0f}, max {max(r[7] for r in per_turn)}; turns whose longest line is exactly 80/120/160/200 cols: "
          + ", ".join(f"{w}:{mlh.get(w,0)}" for w in (80, 120, 160, 200)))
    print("prompt strings (uuid masked):", ", ".join(f"{repr(p)} x{c}" for p, c in prompts.most_common(4)))
    if tokenizer is not None:
        for p, _ in prompts.most_common(2):
            sample = p.replace("<uuid>", "9a4676a4-89ce-4919-a91f-9143147a5229")
            print(f"  tokens for one prompt {repr(sample)}: {len(tokenizer.encode(sample, add_special_tokens=False))}")
    return Summary(n=n, mean_chars=st.mean(chars), mean_tok=st.mean(toks) if toks else 0.0, pchars=pchars, nprompts=nprompts, total=total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--a-label", default="A")
    ap.add_argument("--b", required=True); ap.add_argument("--b-label", default="B")
    ap.add_argument("--tokenizer")
    args = ap.parse_args()
    tok = None
    if args.tokenizer:
        try:
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.tokenizer)
        except Exception as e:  # keep going without it
            print("tokenizer unavailable:", e)
    A = summarize(args.a_label, *scan(args.a), tokenizer=tok)
    B = summarize(args.b_label, *scan(args.b), tokenizer=tok)
    print("\n== difference ==")
    print(f"observation chars per turn: {args.b_label} / {args.a_label} = {B.mean_chars / A.mean_chars:.3f}")
    if A.mean_tok and B.mean_tok:
        print(f"observation tokens per turn: {args.b_label} / {args.a_label} = {B.mean_tok / A.mean_tok:.3f}")
    extra_prompt_chars = B.pchars / B.n - A.pchars / A.n
    print(f"prompt-string chars per turn: {A.pchars / A.n:.0f} vs {B.pchars / B.n:.0f} (+{extra_prompt_chars:.0f}); that is {pct(extra_prompt_chars, B.mean_chars - A.mean_chars):.0f} % of the per-turn char gap")


if __name__ == "__main__":
    main()
