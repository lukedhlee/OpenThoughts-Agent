#!/usr/bin/env python3
"""gepa_ledger.py — the GEPA population ledger. It is the source of truth for the loop: what every candidate is, who
its parent was, what it scored on dev, whether it was accepted, and which candidates are on the front right now.

experiments/gepa/ledger.json is a list of records:
  {id, wave, parent, block, created, split, tasks, pass, axes{...}, front_wins, front_sole, accepted, note}

Commands
  add    --wave W --cand C [--parent P] --candidates <wave>/cands.json --scores <wave>/scores.csv [--accepted] [--note ...]
         (repeat --cand, or pass --all to add every non-control arm of the wave)
  show   [--wave W]                 the whole ledger, newest wave last
  front                             the candidates on the front (front_wins > 0), best first
  sample [--n 1] [--seed S]         parent picks for the next wave, sampled by front_wins (the GEPA selection rule)
  tree                              parent -> child lineage

Python 3.9 / stdlib (Jupiter login node).
"""
import argparse, collections, csv, json, os, random, sys, time

E = "/e/fscratch/reformo/lee27/experiments"
ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["add", "show", "front", "sample", "tree"])
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
               "front_sole": solecnt.get(c, 0), "accepted": bool(a.accepted), "note": a.note}
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

live = [r for r in led if r.get("accepted") and (r.get("front_wins") or 0) > 0]
if a.cmd == "front":
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
