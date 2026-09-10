#!/usr/bin/env python3
"""census_summary.py <census.jsonl>... — bucket TmuxSessionLostError attempts from tmux_census.py output: message kind, exec
return code, attempt index, and what the agent's last turn typed (exit / C-d / reset / stty / kill / other), with examples."""
import collections, json, re, sys

def kind(msg):
    if "no markers" in msg: return "no-markers (exec killed/timeout)"
    if "failed to send batched keys" in msg: return "send-failed (server gone mid-turn)"
    if "failed to send" in msg: return "send-failed (unbatched)"
    return "other"

PATTERNS = [("exit", r"^\s*exit\b|\bexit\s*$"), ("C-d", r"C-d"), ("logout", r"\blogout\b"), ("reset", r"^\s*reset\b|;\s*reset\b"),
            ("stty", r"\bstty\b"), ("kill/pkill", r"\b(kill|pkill|killall)\b"), ("tmux", r"\btmux\b"), ("exec", r"^\s*exec\b"),
            ("C-c", r"C-c"), ("clear", r"^\s*clear\b")]

def classify_keys(keys):
    hits = [name for name, pat in PATTERNS if any(re.search(pat, k or "") for k in keys)]
    return "+".join(hits) if hits else "other"

for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path)]
    summary = [r for r in rows if r.get("summary")]; lost = [r for r in rows if not r.get("summary")]
    print(f"=== {path}")
    for s in summary: print("  scanned", s["attempts_scanned"], "of", s["attempts_total"], "| exceptions:", {k: v for k, v in s["exceptions"].items() if v >= 3})
    print("  tmux-lost:", len(lost))
    kinds = collections.Counter(kind(r.get("message", "")) for r in lost); print("  kinds:", dict(kinds))
    rcs = collections.Counter((re.search(r"return_code=(-?\d+)", r.get("message", "")) or [None, "n/a"])[1] for r in lost); print("  return codes:", dict(rcs))
    att = collections.Counter(r["path"].split("/attempts/")[1][:3] for r in lost); print("  attempt idx:", dict(att))
    last = collections.Counter(); examples = collections.defaultdict(list)
    for r in lost:
        steps = r.get("last_steps") or []
        keys = [c.get("keystrokes") or "" for s in steps[-1:] for c in (s.get("commands") or [])] if steps else []
        c = classify_keys(keys) if keys else ("no-trajectory" if r.get("trajectory") == "missing" else "no-commands")
        last[(kind(r.get("message", "")), c)] += 1
        if len(examples[(kind(r.get("message", "")), c)]) < 2: examples[(kind(r.get("message", "")), c)].append([k[:70] for k in keys][-3:])
    print("  last-turn keystrokes by kind:")
    for (k, c), n in sorted(last.items(), key=lambda x: -x[1]): print(f"    {n:3d}  {k:40s} {c:12s} e.g. {examples[(k, c)]}")
