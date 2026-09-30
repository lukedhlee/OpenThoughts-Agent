#!/usr/bin/env python3
"""Cut the TMax relay pool: the documented quality filter, then every task that is (or copies) a TB-Hard task.

TMax (`laion/TMax-15K-Harbor`, 14,601 tasks) filter, from notes/relay/scaleup_plan.md "TMax filter":
  1. drop the outside audit's structural defects (INSTR-VERIFIER-MISMATCH, INSTR-ENV-MISMATCH, OTHER, INSTR-AMBIGUOUS)
  2. drop its answer leaks (ANSWER-LEAK)                                   -> lenient
     strict also drops weak verifiers that never check the answer          -> strict = the audit's keep_policy_b
  3. drop tasks Qwen3.5-9B solved on every try (at least 2 tries, all passed; the run is ~70 % complete, so a task
     with one try or none stays in)
  4. drop every TMax task that is a TB-Hard task. TB-Hard (Zhongzhi1228/Terminal-Bench-Hard, 100 tasks) is drawn from
     TMax with anonymized ids (tbh_task_<16 hex>, not derivable from the TMax id), so ids are matched through content:
     normalized instruction text (exact), and word 8-gram containment >= 0.3 either way or >= 25 shared 8-grams
     (shingles found in more than 50 TMax instructions are boilerplate and ignored).
  5. drop any task the relay router could not tell apart: its first 400 instruction characters (the router's key)
     appear inside another listed task's instruction.
  Report-only checks: instruction overlap with the references given by --ref (TB2.1, CalibForge; a hit is dropped),
  and TMax tasks that share a rare fixture file (used by at most 10 TMax tasks) with a TB-Hard task: siblings with a
  different instruction, kept.

  python select_tasks.py --audit <rubric parquet> --qwen qwen35_9b_k3.json --tree <TMax harbor dir> \
      --tbhard <Terminal-Bench-Hard tasks dir> [--ref tb21=<parquet>:desc --ref calibforge=<parquet>:desc] \
      [--ext-hosts tests_ext_hosts.json] --out <dir>

Writes tmax_lenient.txt, tmax_strict.txt (sorted ids), tmax_tbhard_map.tsv (TB-Hard id -> TMax id),
tmax_strata.tsv (every TMax task with its labels and the step that dropped it) and counts.json.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import zlib
from pathlib import Path

STRUCTURAL = {"INSTR-VERIFIER-MISMATCH", "INSTR-ENV-MISMATCH", "OTHER", "INSTR-AMBIGUOUS"}
N, BOILER_DF, CONT, SHARED = 8, 50, 0.3, 25
KEY_CHARS = 400  # relay_router._load_tasks: instruction.strip()[:400]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def shingles(s: str, n: int = N) -> set[int]:
    t = re.findall(r"[a-z0-9_./-]+", s.lower())
    return {zlib.crc32(" ".join(t[i:i + n]).encode()) for i in range(max(0, len(t) - n + 1))}


def read_dir_tasks(root: Path) -> dict[str, dict]:
    out = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        fx = d / "environment" / "_fixtures"
        out[d.name] = {
            "instr": (d / "instruction.md").read_text(errors="replace"),
            "fixtures": {hashlib.sha256(p.read_bytes()).hexdigest() for p in fx.rglob("*") if p.is_file()}
            if fx.exists() else set(),
        }
    return out


def overlap(cands: dict[str, str], refs: dict[str, str], corpus_df: collections.Counter) -> dict[str, tuple]:
    """For each candidate: (best ref, shared, containment in cand, containment in ref) if flagged."""
    boiler = {h for h, c in corpus_df.items() if c > BOILER_DF}
    rsh = {k: shingles(v) - boiler for k, v in refs.items()}
    inv = collections.defaultdict(list)
    for k, s in rsh.items():
        for h in s:
            inv[h].append(k)
    hits = {}
    for k, text in cands.items():
        q = shingles(text) - boiler
        cnt = collections.Counter(r for h in q for r in inv.get(h, ()))
        if not cnt:
            continue
        r, c = cnt.most_common(1)[0]
        cq, cr = c / max(1, len(q)), c / max(1, len(rsh[r]))
        if cq >= CONT or cr >= CONT or c >= SHARED:
            hits[k] = (r, c, round(cq, 3), round(cr, 3))
    return hits


def router_clashes(ids: list[str], instr: dict[str, str]) -> set[str]:
    """Tasks whose router key appears in another listed task's instruction (both sides of each clash)."""
    keys = {t: instr[t].strip()[:KEY_CHARS] for t in ids}
    by_head = collections.defaultdict(list)  # a key can only occur in a text that contains its first 32 chars
    for t, k in keys.items():
        by_head[k[:32]].append(t)
    bad = set()
    for a in ids:
        text = instr[a]
        heads = {text[i:i + 32] for i in range(max(0, len(text) - 31))} & by_head.keys()
        for head in heads:
            for b in by_head[head]:
                if b != a and keys[b] in text:
                    bad |= {a, b}
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", required=True, help="wAI-org/swerl-tmax-15k-rubric-gpt-5-6-sol train parquet")
    ap.add_argument("--qwen", required=True, help="qwen35_9b_k3.json ({'per': {task: [rewards]}, ...})")
    ap.add_argument("--tree", required=True, help="laion/TMax-15K-Harbor harbor/ dir")
    ap.add_argument("--tbhard", required=True, help="Zhongzhi1228/Terminal-Bench-Hard tasks/ dir")
    ap.add_argument("--ref", action="append", default=[], help="name=<parquet>:<text column> (report-only overlap)")
    ap.add_argument("--ext-hosts", help="tests_ext_hosts.json: tasks whose tests reach outside hosts (report-only)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import pandas as pd
    au = pd.read_parquet(a.audit, columns=["ground_truth", "verify_label", "verify_mechanism",
                                           "verify_secondary_labels", "keep_policy_b"])
    au = au.set_index("ground_truth")
    tm = read_dir_tasks(Path(a.tree))
    tb = read_dir_tasks(Path(a.tbhard))
    if set(au.index) != set(tm):
        raise SystemExit("audit ids differ from the tree's task ids")
    per = json.load(open(a.qwen))["per"]
    solved = {t for t, v in per.items() if len(v) >= 2 and all(x == 1.0 for x in v)}

    # TB-Hard -> TMax
    exact = collections.defaultdict(list)
    for t, r in tm.items():
        exact[hashlib.sha256(norm(r["instr"]).encode()).hexdigest()].append(t)
    df = collections.Counter(h for r in tm.values() for h in shingles(r["instr"]))
    tb_to_tm = overlap({k: r["instr"] for k, r in tb.items()}, {k: r["instr"] for k, r in tm.items()}, df)
    tm_to_tb = overlap({k: r["instr"] for k, r in tm.items()}, {k: r["instr"] for k, r in tb.items()}, df)
    rows = []
    for k, r in sorted(tb.items()):
        ex = exact.get(hashlib.sha256(norm(r["instr"]).encode()).hexdigest(), [])
        near = tb_to_tm.get(k)
        rows.append((k, ",".join(ex), near[0] if near else "", near[2] if near else 0.0))
    tbh = {t for _, ex, _, _ in rows for t in ex.split(",") if t} | {n for _, _, n, _ in rows if n} | set(tm_to_tb)
    # Fixtures are mostly generator assets shared by hundreds of tasks (the minicalc project is in 855); only a
    # fixture used by at most 10 TMax tasks marks a sibling of a TB-Hard task.
    fx_df = collections.Counter(h for r in tm.values() for h in r["fixtures"])
    rare_tb_fx = {h for r in tb.values() for h in r["fixtures"] if fx_df[h] <= 10}
    fixture_sib = {t for t, r in tm.items() if t not in tbh and r["fixtures"] & rare_tb_fx}

    refs = {}
    for spec in a.ref:
        name, rest = spec.split("=", 1)
        path, col = rest.rsplit(":", 1)
        d = pd.read_parquet(path)
        refs[name] = overlap({k: r["instr"] for k, r in tm.items()},
                             {f"{i}": str(x) for i, x in enumerate(d[col])}, df)
    ext = set(json.load(open(a.ext_hosts))) if a.ext_hosts else set()

    instr = {t: r["instr"] for t, r in tm.items()}
    label = au.verify_label.to_dict()
    struct = {t for t, l in label.items() if l in STRUCTURAL}
    leak = {t for t, l in label.items() if l == "ANSWER-LEAK"}
    keep_b = set(au.index[au.keep_policy_b])
    ref_hit = set().union(*[set(h) for h in refs.values()]) if refs else set()

    steps, lists = {}, {}
    for variant in ("lenient", "strict"):
        cur = set(tm)
        log = [("TMax, all tasks", len(cur))]
        cur -= struct; log.append((f"minus the audit's structural defects ({len(struct)})", len(cur)))
        cur -= leak; log.append((f"minus its answer leaks ({len(leak)})", len(cur)))
        if variant == "strict":
            cur &= keep_b; log.append(("minus weak verifiers that do not check the answer (keep_policy_b)", len(cur)))
        n0 = len(cur); cur -= solved
        log.append((f"minus tasks Qwen3.5-9B solved on every try (>= 2 tries; {n0 - len(cur)} here)", len(cur)))
        n0 = len(cur); cur -= tbh
        log.append((f"minus TB-Hard copies ({n0 - len(cur)} of the {len(tbh)} still here)", len(cur)))
        n0 = len(cur); cur -= ref_hit
        log.append((f"minus instruction overlap with {'/'.join(refs) or 'no reference'} ({n0 - len(cur)})", len(cur)))
        clash = router_clashes(sorted(cur), instr)
        cur -= clash
        log.append((f"minus tasks the router could not tell apart ({len(clash)})", len(cur)))
        steps[variant], lists[variant] = log, sorted(cur)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for v, ids in lists.items():
        (out / f"tmax_{v}.txt").write_text("\n".join(ids) + "\n")
    with open(out / "tmax_tbhard_map.tsv", "w") as f:
        f.write("tbhard_id\ttmax_exact_instruction\ttmax_near\tnear_containment\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
    strict_set, lenient_set = set(lists["strict"]), set(lists["lenient"])
    with open(out / "tmax_strata.tsv", "w") as f:
        f.write("task_id\tverify_label\tverify_mechanism\tsecondary_answer_leak\tkeep_policy_b\tsource\tdomain\t"
                "qwen_tries\tqwen_passes\ttbhard\tfixture_sibling_of_tbhard\ttests_ext_hosts\tin_lenient\tin_strict\n")
        for t in sorted(tm):
            toml = (Path(a.tree) / t / "task.toml").read_text()
            src = re.search(r'^source = "([^"]+)"', toml, re.M)
            dom = re.search(r'^domain = "([^"]+)"', toml, re.M)
            v = per.get(t, [])
            sec = au.verify_secondary_labels.get(t)
            f.write("\t".join(map(str, [
                t, label[t], str(au.verify_mechanism[t]).replace("\t", " "),
                "ANSWER-LEAK" in list(sec if sec is not None else []), bool(au.keep_policy_b[t]),
                src.group(1) if src else "", dom.group(1) if dom else "", len(v), int(sum(v)),
                t in tbh, t in fixture_sib, t in ext, t in lenient_set, t in strict_set])) + "\n")
    counts = {
        "steps": steps,
        "tbhard": {"tbhard_tasks": len(tb), "matched_exact_instruction": sum(bool(r[1]) for r in rows),
                   "tmax_ids_excluded": len(tbh), "tbhard_labels": dict(collections.Counter(label[t] for t in tbh)),
                   "tbhard_qwen_solved_every_try": len(tbh & solved)},
        "ref_overlap": {k: len(v) for k, v in refs.items()},
        "fixture_siblings_of_tbhard_kept": {"lenient": len(fixture_sib & lenient_set), "strict": len(fixture_sib & strict_set)},
        "tests_ext_hosts_kept": {"lenient": len(ext & lenient_set), "strict": len(ext & strict_set)},
        "source_kept": {v: dict(collections.Counter(
            re.search(r'^source = "([^"]+)"', (Path(a.tree) / t / "task.toml").read_text(), re.M).group(1)
            for t in ids)) for v, ids in lists.items()},
        "secondary_answer_leak_kept": {v: sum("ANSWER-LEAK" in list(au.verify_secondary_labels[t]) for t in ids)
                                       for v, ids in lists.items()},
    }
    (out / "counts.json").write_text(json.dumps(counts, indent=1) + "\n")
    print(json.dumps(counts, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
