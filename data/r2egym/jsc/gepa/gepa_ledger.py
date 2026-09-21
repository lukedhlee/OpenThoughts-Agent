#!/usr/bin/env python3
"""gepa_ledger.py — the GEPA population ledger. It is the source of truth for the loop: what every candidate is, who
its parent was, what it scored on dev, whether it was accepted, and which candidates are on the front right now.

experiments/gepa/ledger.json is a list of records:
  {id, wave, parent, block, created, split, tasks, pass, axes{...}, front_wins, front_sole, accepted, note}

Commands
  add    --wave W --cand C [--parent P] --candidates <wave>/cands.json --scores <wave>/scores.csv
         [--oodmini-delta D] [--accepted] [--note ...]
         (repeat --cand, or pass --all to add every non-control arm of the wave)
  show   [--wave W]                 the whole ledger, newest wave last
  front                             the candidates on the front (front_wins > 0), best first
  sample [--n 1] [--seed S]         parent picks for the next wave, sampled by front_wins (the GEPA selection rule)
  tree                              parent -> child lineage
  judge  --cand C --axis A --agree N [--of 8]
         Record a hand-grade of one axis against real traces. Below --freeze-below (6 of 8) the axis is FROZEN and
         gepa_score.py drops it from the Pareto front. A detector the session cannot reproduce by reading traces is
         not evidence, and an unchecked one would quietly decide selection.

Two flags on a candidate change what it can do:
  specialist   set from --oodmini-delta: dev went up while the held-out repos went down by more than
               --specialist-drop. It stays in the ledger for the record and is excluded from parent sampling, so a
               block that bought dev with the band's repos cannot breed.
  judge/frozen see above.

Python 3.9 / stdlib (Jupiter login node).
"""
import argparse, collections, csv, json, os, random, sys, time

E = "/e/fscratch/reformo/lee27/experiments"
ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["add", "show", "front", "sample", "tree", "judge"])
ap.add_argument("--ledger", default=E + "/gepa/ledger.json")
ap.add_argument("--wave")
ap.add_argument("--cand", action="append", default=[])
ap.add_argument("--all", action="store_true", help="add: every non-control arm in the wave manifest")
ap.add_argument("--parent", default=None)
ap.add_argument("--candidates", default=None, help="add: the wave's cands.json (block text + arms)")
ap.add_argument("--scores", default=None, help="add: the wave's scores.csv")
ap.add_argument("--split", default="dev")
ap.add_argument("--ctl", default="ctl")
ap.add_argument("--accepted", action="store_true")
ap.add_argument("--note", default="")
ap.add_argument("--n", type=int, default=1)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--axis", help="judge: the axis that was hand-graded")
ap.add_argument("--agree", type=int, help="judge: trials where the detector matched your reading")
ap.add_argument("--of", type=int, default=8, help="judge: trials graded")
ap.add_argument("--freeze-below", type=int, default=6, help="judge: agreement below this freezes the axis")
ap.add_argument("--oodmini-delta", type=float, help="add: this candidate's paired OODMINI delta vs ctl")
ap.add_argument("--specialist-drop", type=float, default=-0.05, help="add: OODMINI delta at or below this, with a positive dev delta, = specialist")
a = ap.parse_args()

AXES = ["self_check", "ran_test_after_edit", "in_place", "no_sed_patch",
        "no_repeat3", "no_json_reject", "no_input_delete", "no_ctx_death"]
led = json.load(open(a.ledger)) if os.path.exists(a.ledger) else []
byid = {r["id"]: r for r in led}


def save():
    os.makedirs(os.path.dirname(a.ledger), exist_ok=True)
    json.dump(led, open(a.ledger, "w"), indent=1)
    print("wrote %s (%d candidates)" % (a.ledger, len(led)))


if a.cmd == "add":
    if not (a.wave and a.candidates and a.scores):
        sys.exit("add needs --wave, --candidates and --scores")
    man = json.load(open(a.candidates))
    rowsr = list(csv.DictReader(open(a.scores)))
    cands = a.cand or ([c for c in man["arms"] if c != a.ctl] if a.all else [])
    if not cands:
        sys.exit("add needs --cand (repeatable) or --all")
    # tasks whose Pareto front has exactly one member: at small populations `sole` discriminates and `front_wins`
    # barely does (nine axes, few candidates -> almost nothing dominates).
    onfront = collections.defaultdict(list)
    for r in rowsr:
        if r.get("on_front") == "1":
            onfront[r["task"]].append(r["cand"])
    solecnt = collections.Counter(v[0] for v in onfront.values() if len(v) == 1)
    for c in cands:
        rs = [r for r in rowsr if r["cand"] == c]
        if not rs:
            sys.exit("no scores.csv rows for candidate %s" % c)

        def mean(k):
            v = [float(r[k]) for r in rs if r.get(k) not in (None, "", "None")]
            return round(sum(v) / len(v), 4) if v else None

        rec = {"id": c, "wave": a.wave, "parent": a.parent, "block": man["blocks"].get(c, ""),
               "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "split": a.split, "tasks": len(rs),
               "pass": mean("pass"), "axes": {k: mean(k) for k in AXES},
               "front_wins": sum(1 for r in rs if r.get("on_front") == "1"),
               "front_sole": solecnt.get(c, 0), "accepted": bool(a.accepted), "note": a.note,
               "oodmini_delta": a.oodmini_delta, "specialist": False}
        # SPECIALIST: bought dev with the band's own repos. Recorded, kept in the ledger for the record, and
        # excluded from parent sampling so the trick cannot breed. Needs the OODMINI leg scored first.
        prev = byid.get(c, {})
        dev_delta = None
        if prev.get("pass") is not None and rec["pass"] is not None:
            dev_delta = rec["pass"] - prev["pass"]
        if a.oodmini_delta is not None and a.oodmini_delta <= a.specialist_drop:
            rec["specialist"] = True
            print("  %s flagged SPECIALIST: OODMINI delta %+.3f <= %+.3f" % (c, a.oodmini_delta, a.specialist_drop))
        if a.oodmini_delta is None:
            print("  %s: no --oodmini-delta given; the specialist check is unset for this candidate" % c)
        if c in byid:
            rec["judge"] = byid[c].get("judge", {})
        if c in byid:
            byid[c].update(rec)
            print("updated %s" % c)
        else:
            led.append(rec)
            byid[c] = rec
            print("added %s (parent %s)" % (c, a.parent))
    save()
    sys.exit(0)

if not led:
    sys.exit("empty ledger at %s" % a.ledger)

if a.cmd == "show":
    rs = [r for r in led if not a.wave or r["wave"] == a.wave]
    print("| id | wave | parent | tasks | pass | front | sole | acc | %s |" % " | ".join(AXES))
    print("|---|---|---|---|---|---|---|---|%s" % ("---|" * len(AXES)))
    for r in rs:
        print("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["id"], r["wave"], r.get("parent") or "-", r["tasks"],
            "%.3f" % r["pass"] if r["pass"] is not None else "-", r.get("front_wins"), r.get("front_sole"),
            "y" if r.get("accepted") else "", " | ".join(
                "%.2f" % r["axes"][k] if r["axes"].get(k) is not None else "-" for k in AXES)))
    sys.exit(0)

if a.cmd == "tree":
    kids = collections.defaultdict(list)
    for r in led:
        kids[r.get("parent")].append(r["id"])

    def walk(p, d=0):
        for c in sorted(kids.get(p, [])):
            r = byid[c]
            print("%s%s  wave %s  pass %s  front %s%s" % (
                "  " * d, c, r["wave"], "%.3f" % r["pass"] if r["pass"] is not None else "-",
                r.get("front_wins"), "  ACCEPTED" if r.get("accepted") else ""))
            walk(c, d + 1)

    walk(None)
    sys.exit(0)

if a.cmd == "judge":
    if not (a.cand and a.axis and a.agree is not None):
        sys.exit("judge needs --cand, --axis and --agree (e.g. --cand c012 --axis self_check --agree 7 --of 8)")
    if a.axis not in AXES:
        sys.exit("%s is not an axis (%s)" % (a.axis, ", ".join(AXES)))
    for c in a.cand:
        if c not in byid:
            sys.exit("%s is not in the ledger" % c)
        j = byid[c].setdefault("judge", {})
        frozen = a.agree < a.freeze_below
        j[a.axis] = {"agree": a.agree, "of": a.of, "frozen": frozen,
                     "judged": time.strftime("%Y-%m-%dT%H:%M:%S")}
        print("%s %s: %d/%d agree -> %s" % (
            c, a.axis, a.agree, a.of,
            "FROZEN (the front will ignore this axis until a later judge clears it)" if frozen else "trusted"))
    save()
    sys.exit(0)

# A specialist stays in the ledger for the record but never breeds.
live = [r for r in led if r.get("accepted") and (r.get("front_wins") or 0) > 0 and not r.get("specialist")]
skipped_spec = [r["id"] for r in led if r.get("accepted") and r.get("specialist")]
if a.cmd == "front":
    if skipped_spec:
        print("excluded as SPECIALIST (dev up, OOD-repo down): %s\n" % ", ".join(skipped_spec))
    fr = sorted({ax for r in led for ax, j in (r.get("judge") or {}).items() if j.get("frozen")})
    if fr:
        print("frozen axes (ignored by the front): %s\n" % ", ".join(fr))
    print("Front (accepted candidates with front_wins > 0), best first.")
    print("With few candidates and nine axes almost nothing dominates, so front_wins saturates -- read `sole`"
          " (tasks this candidate wins alone) alongside it.\n")
    for r in sorted(live, key=lambda r: -(r.get("front_wins") or 0)):
        print("  %-8s wave %-4s parent %-8s front_wins %4d  sole %4s  pass %s" % (
            r["id"], r["wave"], r.get("parent") or "-", r["front_wins"], r.get("front_sole"),
            "%.3f" % r["pass"] if r["pass"] is not None else "-"))
    tot = sum(r["front_wins"] for r in live) or 1
    print("\nParent sampling weights (front_wins / total):")
    for r in sorted(live, key=lambda r: -r["front_wins"]):
        print("  %-8s %.3f" % (r["id"], r["front_wins"] / float(tot)))
    sys.exit(0)

if a.cmd == "sample":
    if not live:
        sys.exit("no accepted candidate has front_wins > 0 -- seed the population first")
    rng = random.Random(a.seed)
    ids = [r["id"] for r in live]
    w = [r["front_wins"] for r in live]
    picks = [ids[rng.choices(range(len(ids)), weights=w)[0]] for _ in range(a.n)]
    for p in picks:
        r = byid[p]
        print("%s  (wave %s, front_wins %d, pass %s)" % (
            p, r["wave"], r["front_wins"], "%.3f" % r["pass"] if r["pass"] is not None else "-"))
