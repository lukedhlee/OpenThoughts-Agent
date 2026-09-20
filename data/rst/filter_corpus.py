#!/usr/bin/env python3
"""Build the two filtered 2026-09-20 SFT corpora from the merged ota3_if_rst shard.

Both arms are defect-removal rules, not outcome rules: the evidence since early 2025 says
filtering agent traces on whether the agent succeeded makes terminal agents worse, while
removing specific defects from traces you keep helps. Survey and counts:
ai_memory/active/snowball-sft/research/2026-09-20_sft_filtering_survey.md

  ota3_if_rst_fmt_v1   arm 1, MALFORMED ACTION. Drop a row if any assistant turn, other than
                       the final turn of a `kind == "compaction"` row, has a body after
                       <|end_think|> that does not parse with json.loads. The exemption is
                       load-bearing: a compaction handoff summary is prose by design and
                       3,729 of swesmith's 3,732 compaction rows would otherwise go with it.
                       Expected 81,889 of 90,916 rows.

  ota3_if_rst_beh_v1   arm 2, DEGENERATE BEHAVIOUR. Drop a row if any of the five clauses fire:
                       loop      the identical non-empty command batch 3+ times consecutively
                       hammer    one keystroke string 20+ times in the trace
                       noop      3+ consecutive turns issuing only empty keystrokes
                       redundant 25%+ of adjacent turn pairs issue the identical batch
                       errors    50%+ of terminal observations contain a shell or Python error
                       Expected 83,851 of 90,916 rows.

One streaming pass writes both outputs. The schema is unchanged, the held-out file is copied
byte for byte, and each output dir gets parquet.list / SHA256SUMS / census.json exactly as
build_corpora.py leaves them, because the launcher and the marin stage expect that layout.

    OMP_NUM_THREADS=1 python filter_corpus.py \
        --in /e/data1/mmlaion/lee27/snowball-sft/data/ota3_if_rst_v1 \
        --out-root /e/data1/mmlaion/lee27/snowball-sft/data
"""
from __future__ import annotations
import argparse, collections, hashlib, json, re, shutil, sys, time
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq

END_THINK = "<|end_think|>"
TRAIN_NAME = "ota3-if-rst-train-00000-of-00001.parquet"
HELD_NAME = "ota3-heldout-00000-of-00001.parquet"
ROW_GROUP = 1000

# characters per Marin token, measured per slice against each slice's build census
# (ota_sft_100k_v2/census.json, ota_if_sft_v1/ifv2-census.json, rst_sft_v1/census.json).
# Token columns below are char counts divided by these, so they line up with the numbers the
# corpora were staged with without re-running the tokenizer over 1.3B tokens on a login node.
CHARS_PER_TOKEN = {"swesmith": 3.773, "superuser": 3.454, "tezos": 3.624,
                   "ifv2": 3.891, "if-v2": 3.891, "if": 3.891, "rst": 3.871}
FALLBACK_CPT = 3.70

ERR_RE = re.compile(r"command not found|No such file or directory|Traceback \(most recent call last\)|"
                    r"SyntaxError|ImportError|ModuleNotFoundError|Permission denied|"
                    r"bash: .*: command not found|E: Unable to locate|fatal: ", re.I)


def body(text: str) -> str:
    """The action JSON of an assistant turn: everything after the think span."""
    i = text.rfind(END_THINK)
    b = (text[i + len(END_THINK):] if i >= 0 else text).strip()
    if b.startswith("```"):
        b = b.split("\n", 1)[-1]
        if b.rstrip().endswith("```"):
            b = b.rstrip()[:-3]
    return b


def keystrokes(b: str):
    """(batch, parsed_ok). batch is the tuple of keystroke strings of a Terminus-2 action."""
    try:
        act = json.loads(b)
    except Exception:
        return None, False
    if not isinstance(act, dict):
        return None, False
    cmds = act.get("commands")
    if not isinstance(cmds, list):
        return (), True
    return tuple(c.get("keystrokes") if isinstance(c, dict) and isinstance(c.get("keystrokes"), str) else ""
                 for c in cmds), True


def judge(conv, kind: str):
    """Return (drop_fmt, clauses_fired) for one row, in a single pass over its turns."""
    asst = [m["content"] for m in conv if m["role"] == "assistant"]
    usr = [m["content"] for m in conv if m["role"] == "user"]
    last = len(asst) - 1
    drop_fmt = False
    batches = []
    for j, t in enumerate(asst):
        b = body(t)
        batch, ok = keystrokes(b)
        if not ok:
            # a compaction row's final assistant turn is the handoff summary: prose by design
            if not (kind == "compaction" and j == last):
                drop_fmt = True
            batches.append(None)
        else:
            batches.append(batch)

    run = best_run = 1
    for a, b in zip(batches, batches[1:]):
        if a is not None and a == b and any(s.strip() for s in a):
            run += 1
            best_run = max(best_run, run)
        else:
            run = 1
    nrun = best_noop = 0
    for b in batches:
        if b is not None and b and not any(s.strip() for s in b):
            nrun += 1
            best_noop = max(best_noop, nrun)
        else:
            nrun = 0
    kc = collections.Counter()
    for b in batches:
        for s in (b or ()):
            s2 = s.strip()
            if s2:
                kc[s2] += 1
    top_rep = kc.most_common(1)[0][1] if kc else 0
    pairs = sum(1 for a, b in zip(batches, batches[1:]) if a is not None and b is not None)
    redun = sum(1 for a, b in zip(batches, batches[1:])
                if a is not None and a == b and any(s.strip() for s in a))
    n_obs = max(1, len(usr) - 1)          # the first user turn is the task prompt, not an observation
    errs = sum(1 for u in usr[1:] if ERR_RE.search(u))

    clauses = []
    if best_run >= 3:
        clauses.append("loop")
    if top_rep >= 20:
        clauses.append("hammer")
    if best_noop >= 3:
        clauses.append("noop")
    if redun / max(1, pairs) >= 0.25:
        clauses.append("redundant")
    if errs / n_obs >= 0.5:
        clauses.append("errors")
    return drop_fmt, clauses


def finish(out: Path, train_name: str, src_held: Path, census: dict) -> dict:
    shutil.copy2(src_held, out / HELD_NAME)
    sums = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.glob("*.parquet"))}
    (out / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in sums.items()))
    (out / "parquet.list").write_text(f"{out}/{train_name}\n")
    chk = pq.ParquetFile(out / train_name)
    assert max(chk.metadata.row_group(i).num_rows for i in range(chk.num_row_groups)) <= ROW_GROUP
    census["train"] = {"rows": chk.metadata.num_rows, "row_groups": chk.num_row_groups}
    census["heldout"] = {"rows": pq.ParquetFile(out / HELD_NAME).metadata.num_rows,
                         "copied_from": str(src_held)}
    census["sha256"] = sums
    (out / "census.json").write_text(json.dumps(census, indent=1))
    return census["train"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, type=Path, help="ota3_if_rst_v1 dir")
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--expect-fmt", type=int, default=81889)
    ap.add_argument("--expect-beh", type=int, default=83851)
    a = ap.parse_args()

    src = a.inp / TRAIN_NAME
    held = a.inp / HELD_NAME
    for p in (src, held):
        if not p.exists():
            sys.exit(f"missing input: {p}")
    f = pq.ParquetFile(src)
    schema = f.schema_arrow
    total = f.metadata.num_rows
    print(f"input {src} rows={total} row_groups={f.num_row_groups}", flush=True)

    dirs = {"fmt": a.out_root / "ota3_if_rst_fmt_v1", "beh": a.out_root / "ota3_if_rst_beh_v1"}
    names = {"fmt": "ota3-if-rst-fmt-train-00000-of-00001.parquet",
             "beh": "ota3-if-rst-beh-train-00000-of-00001.parquet"}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    writers = {k: pq.ParquetWriter(dirs[k] / names[k], schema, compression="zstd") for k in dirs}

    stat = {k: collections.defaultdict(lambda: {"keep_rows": 0, "keep_chars": 0,
                                                "drop_rows": 0, "drop_chars": 0}) for k in dirs}
    clause_rows = collections.Counter()
    clause_sole = collections.Counter()
    slices_seen = collections.Counter()
    t0 = time.time()
    done = 0
    for rg in range(f.num_row_groups):
        tbl = f.read_row_group(rg)
        rows = tbl.to_pylist()
        keep_idx = {"fmt": [], "beh": []}
        for i, r in enumerate(rows):
            sl = r["slice"] or "?"
            slices_seen[sl] += 1
            chars = sum(len(m["content"]) for m in r["conversations"])
            drop_fmt, clauses = judge(r["conversations"], r["kind"])
            drop_beh = bool(clauses)
            for c in clauses:
                clause_rows[c] += 1
            if len(clauses) == 1:
                clause_sole[clauses[0]] += 1
            for k, dropped in (("fmt", drop_fmt), ("beh", drop_beh)):
                s = stat[k][sl]
                if dropped:
                    s["drop_rows"] += 1
                    s["drop_chars"] += chars
                else:
                    s["keep_rows"] += 1
                    s["keep_chars"] += chars
                    keep_idx[k].append(i)
        for k in dirs:
            if keep_idx[k]:
                writers[k].write_table(tbl.take(keep_idx[k]), row_group_size=ROW_GROUP)
        done += len(rows)
        if rg % 10 == 0 or done == total:
            print(f"  {done}/{total} rows  {time.time()-t0:.0f}s", flush=True)
    for w in writers.values():
        w.close()

    def tok(chars, sl):
        return int(chars / CHARS_PER_TOKEN.get(sl, FALLBACK_CPT))

    results = {}
    rules = {
        "fmt": ("arm 1, malformed action: drop a row if any assistant turn, other than the final "
                "turn of a kind=='compaction' row, has a body after <|end_think|> that does not "
                "parse with json.loads"),
        "beh": ("arm 2, degenerate behaviour: drop a row if any of loop (identical non-empty command "
                "batch 3+ times consecutively), hammer (one keystroke string 20+ times in the trace), "
                "noop (3+ consecutive turns issuing only empty keystrokes), redundant (25%+ of adjacent "
                "turn pairs identical), errors (50%+ of terminal observations contain a shell or "
                "Python error)"),
    }
    for k in dirs:
        per = {}
        for sl, s in sorted(stat[k].items()):
            per[sl] = {"keep_rows": s["keep_rows"], "keep_tokens": tok(s["keep_chars"], sl),
                       "drop_rows": s["drop_rows"], "drop_tokens": tok(s["drop_chars"], sl),
                       "chars_per_token": CHARS_PER_TOKEN.get(sl, FALLBACK_CPT),
                       "ratio_known": sl in CHARS_PER_TOKEN}
        keep_rows = sum(v["keep_rows"] for v in per.values())
        drop_rows = sum(v["drop_rows"] for v in per.values())
        census = {
            "corpus": dirs[k].name,
            "arm": "1 (malformed action)" if k == "fmt" else "2 (degenerate behaviour)",
            "rule": rules[k],
            "source": str(src),
            "source_rows": total,
            "kept_rows": keep_rows,
            "dropped_rows": drop_rows,
            "kept_tokens": sum(v["keep_tokens"] for v in per.values()),
            "dropped_tokens": sum(v["drop_tokens"] for v in per.values()),
            "drop_pct_rows": round(100 * drop_rows / total, 2),
            "per_slice": per,
            "note": "ai_memory/active/snowball-sft/research/2026-09-20_sft_filtering_survey.md",
            "token_method": ("char counts divided by each slice's measured chars-per-token, so the "
                             "token columns match the slice build censuses; not re-tokenized"),
        }
        if k == "beh":
            census["clause_rows_firing"] = dict(clause_rows.most_common())
            census["clause_only_rule_firing"] = dict(clause_sole.most_common())
        info = finish(dirs[k], names[k], held, census)
        results[k] = (keep_rows, info["rows"])
        print(f"{dirs[k].name}: kept {keep_rows} / {total} rows -> {info['rows']} written, "
              f"{census['kept_tokens']/1e6:.1f}M tokens", flush=True)

    print("slices seen:", dict(slices_seen), flush=True)
    print("arm 2 clauses firing:", dict(clause_rows.most_common()), flush=True)

    problems = []
    for k, exp in (("fmt", a.expect_fmt), ("beh", a.expect_beh)):
        counted, written = results[k]
        if counted != written:
            problems.append(f"{k}: counted {counted} but wrote {written}")
        if counted != exp:
            problems.append(f"{k}: expected {exp} from the survey note, got {counted} "
                            f"(delta {counted - exp})")
    if problems:
        print("DEVIATION FROM THE NOTE:\n  " + "\n  ".join(problems), flush=True)
        sys.exit(2)
    print("OK: both corpora match the survey note exactly.", flush=True)


if __name__ == "__main__":
    main()
