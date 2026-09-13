#!/usr/bin/env python3
"""distill_gates.py <run.out> <after_step>: one line per WANDB_MIRROR train step after <after_step>, with the context-
distillation gates. HARD_FAIL is appended when a step-1-class gate fails: edited < 480 of 512 (wrong tree), absent > 32
(marker mismatch), failed > 0 (span not isolable), shift_per_token exactly 0 with edited rows (no second forward), or
tis/log_ratio_abs_mean > 0.3 nats (the context shift leaked into the behaviour ratio; the no-block level is ~0.1)."""
import json, re, sys
L, after = sys.argv[1], int(sys.argv[2])
ansi = re.compile(r"\x1b\[[0-9;]*m")
KEYS = [("edited", "generate/context_distillation/edited"), ("absent", "generate/context_distillation/absent"),
        ("failed", "generate/context_distillation/failed"), ("removed", "generate/context_distillation/removed_tokens"),
        ("edfrac", "context_distillation/edited_fraction"), ("shift", "context_distillation/shift_per_token"),
        ("shift_head", "context_distillation/shift_per_token_head"), ("shift_tail", "context_distillation/shift_per_token_tail"),
        ("tis_abs", "tis/log_ratio_abs_mean"), ("tis_ratio", "tis/imp_ratio_mean"), ("tis_capped", "tis/imp_ratio_capped_fraction"),
        ("probdiff", "policy/rollout_train_prob_diff_mean"), ("exact", "generate/tis/exact_match_fraction"),
        ("reward", "avg_reward"), ("pass8", "avg_pass_at_8"), ("ent", "policy/entropy"), ("masked", "num_masked_trajectories")]


def find(d, suffix):
    for k, v in d.items():
        if k == suffix or k.endswith("/" + suffix) or k.endswith(suffix):
            return v
    return None


for line in open(L, errors="replace"):
    if "WANDB_MIRROR kind=train" not in line:
        continue
    line = ansi.sub("", line)
    m = re.search(r"kind=train step=(\d+)", line); step = int(m.group(1))
    if step <= after:
        continue
    try:
        d = json.loads(line[line.index("metrics=") + len("metrics="):].strip())
    except Exception as e:
        print("step %d PARSE FAIL %s" % (step, e)); continue
    vals = {name: find(d, key) for name, key in KEYS}
    fmt = lambda v: "-" if v is None else ("%.4g" % v if isinstance(v, float) else str(v))
    parts = ["step %d" % step] + ["%s=%s" % (n, fmt(vals[n])) for n, _ in KEYS]
    if vals["edited"] is not None and vals["removed"] is not None and vals["edited"]:
        parts.append("removed/edited=%.1f" % (vals["removed"] / vals["edited"]))
    hard = []
    if vals["edited"] is not None and vals["edited"] < 480: hard.append("edited<480")
    if vals["absent"] is not None and vals["absent"] > 32: hard.append("absent>32")
    if vals["failed"] is not None and vals["failed"] > 0: hard.append("failed>0")
    if vals["shift"] is not None and vals["shift"] == 0.0 and (vals["edited"] or 0) > 0: hard.append("shift==0")
    if vals["tis_abs"] is not None and vals["tis_abs"] > 0.3: hard.append("tis_abs>0.3")
    if hard: parts.append("HARD_FAIL(" + ",".join(hard) + ")")
    print(" ".join(parts))
