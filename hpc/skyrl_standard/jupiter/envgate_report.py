"""Score the model-free environment gate.

Reads envgate/swebench_gate JSON lines (one per task x mode) and answers three
DIFFERENT questions that are easy to conflate:

  1. HARNESS OK?   Did the run even execute? (offline_patch ok, not timed_out,
                   reward parsed to a float rather than null)
  2. GATE PASS?    On the SAME task, does the no-fix arm give 0.0 and the
                   gold-fix arm give 1.0? Only this proves the verifier measures
                   code state instead of returning a per-task constant.
  3. CEILING.      Fraction of tasks whose ORACLE arm reaches 1.0. A task whose
                   oracle cannot score 1.0 is unsolvable by ANY model, so it is
                   dead weight in the band and caps the achievable reward.

Deliberately does NOT report a single "pass rate" -- conflating a rate with a
band is an error this project has already made twice.
"""
import json
import sys
from collections import defaultdict

PRISTINE = {"pristine", "nop"}
ORACLE = {"oracle"}

rows = []
for path in sys.argv[1:]:
    for line in open(path, errors="replace"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass

by_task = defaultdict(dict)
for r in rows:
    if "mode" in r and "task" in r:
        by_task[r["task"]][r["mode"]] = r

print(f"records={len(rows)}  tasks={len(by_task)}\n")

# --- 1. harness health -------------------------------------------------------
errs = defaultdict(int)
timed = sum(1 for r in rows if r.get("timed_out"))
nulls = sum(1 for r in rows if r.get("reward") is None and "error" not in r)
badpatch = sum(1 for r in rows if str(r.get("offline_patch", "ok")) != "ok")
for r in rows:
    if "error" in r:
        errs[r["error"]] += 1
print("[1] HARNESS")
print(f"    hard errors     : {dict(errs) if errs else 'none'}")
print(f"    timed out       : {timed}/{len(rows)}")
print(f"    reward=null     : {nulls}/{len(rows)}")
print(f"    offline patch ok: {len(rows)-badpatch}/{len(rows)}")
if timed or nulls or badpatch or errs:
    print("    -> harness is NOT clean; numbers below are provisional")
print()

# --- 2. the gate -------------------------------------------------------------
gate_pass, gate_fail, incomplete = [], [], []
for t, m in sorted(by_task.items()):
    p = next((m[k] for k in m if k in PRISTINE), None)
    o = next((m[k] for k in m if k in ORACLE), None)
    if not p or not o or p.get("reward") is None or o.get("reward") is None:
        incomplete.append(t)
        continue
    if p["reward"] == 0.0 and o["reward"] == 1.0:
        gate_pass.append(t)
    else:
        gate_fail.append((t, p["reward"], o["reward"]))

print("[2] GATE  (no-fix 0.0 AND gold-fix 1.0 on the same task)")
print(f"    PASS       : {len(gate_pass)}")
print(f"    FAIL       : {len(gate_fail)}")
print(f"    incomplete : {len(incomplete)}")
for t, pr, orw in gate_fail[:15]:
    print(f"      {t:22} pristine={pr} oracle={orw}")
if gate_pass:
    print(f"    e.g. passing: {', '.join(gate_pass[:6])}")
print()

# --- 3. ceiling --------------------------------------------------------------
all_oracles = [m[k] for m in by_task.values() for k in m if k in ORACLE]
# Denominator counts only oracle runs that actually PRODUCED a reward. A run
# that timed out or errored tells us nothing about solvability, and counting it
# as "not solvable" silently understates the ceiling -- the same
# wrong-denominator mistake that produced bogus band numbers before.
oracles = [r for r in all_oracles if r.get("reward") is not None]
unusable = len(all_oracles) - len(oracles)
solved = [r for r in oracles if r.get("reward") == 1.0]
print("[3] CEILING  (oracle reaches 1.0 -> task is solvable at all)")
if oracles:
    print(f"    {len(solved)}/{len(oracles)} = {100*len(solved)/len(oracles):.1f}% of tasks with a"
          f" usable oracle result are solvable")
    print(f"    ({unusable} further oracle run(s) produced no reward and are EXCLUDED,")
    print("     not counted as unsolvable -- they are unknown, not zero.)")
    print("    Tasks whose oracle does NOT reach 1.0 can never be solved by any")
    print("    model and are dead weight in the band.")
else:
    print(f"    no usable oracle results yet ({unusable} run(s) produced no reward)")
print()

# --- stall forensics ---------------------------------------------------------
stalls = [r for r in rows if r.get("timed_out")]
if stalls:
    print("[4] STALL FORENSICS -- last output before the timeout")
    for r in stalls[:5]:
        tail = (r.get("tail") or "").replace("\n", " | ")[-240:]
        print(f"    {r['task']}/{r['mode']}: {tail}")
