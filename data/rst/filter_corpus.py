#!/usr/bin/env python3
"""Build one filtered SFT corpus from a merged corpus shard, by applying one defect-removal rule.

Both rules are defect-removal, not outcome-removal: the evidence since early 2025 says filtering
agent traces on whether the agent succeeded makes terminal agents worse, while removing specific
defects from traces you keep helps. Survey, effect sizes and the full census:
ai_memory/active/snowball-sft/research/2026-09-20_sft_filtering_survey.md

  --rule fmt   MALFORMED ACTION. Drop a row if any assistant turn, other than the final turn of
               a `kind == "compaction"` row, has a body after <|end_think|> that does not parse
               with json.loads. The exemption is load-bearing: a compaction handoff summary is
               prose by design, and 3,729 of swesmith's 3,732 compaction rows would otherwise be
               deleted along with the one skill the eval harness asks for.
               On ota3_if_rst_v1 this keeps 81,889 of 90,916 rows.

  --rule beh   DEGENERATE BEHAVIOUR. Drop a row if any of five clauses fire:
                 loop       the identical non-empty command batch 3+ times consecutively
                 hammer     one keystroke string 20+ times in the trace
                 noop       3+ consecutive turns issuing only empty keystrokes
                 redundant  25%+ of adjacent turn pairs issue the identical batch
                 errors     50%+ of terminal observations contain a shell or Python error
               On ota3_if_rst_v1 this keeps 83,851 of 90,916 rows.

The output layout matches what build_corpora.py leaves behind, because the launcher and the
marin stage expect it: the train shard is named from the output dir (ota3_if_rst_fmt_v1 ->
ota3-if-rst-fmt-train-00000-of-00001.parquet), the held-out file is copied byte for byte, and
parquet.list / SHA256SUMS / census.json sit beside them. Schema is unchanged, output row groups
are 1,000 rows, compression is zstd, and the input is streamed a row group at a time so this is
safe on a login node.

    OMP_NUM_THREADS=1 python filter_corpus.py --rule fmt \
        --src /e/data1/mmlaion/lee27/snowball-sft/data/ota3_if_rst_v1 \
        --out /e/data1/mmlaion/lee27/snowball-sft/data/ota3_if_rst_fmt_v1
"""
from __future__ import annotations
import argparse, collections, hashlib, json, re, shutil, sys, time
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq

END_THINK = "<|end_think|>"
HELD_NAME = "ota3-heldout-00000-of-00001.parquet"
ROW_GROUP = 1000
CLAUSES = ("loop", "hammer", "noop", "redundant", "errors")

# Characters per Marin token, measured per slice against each slice's own build census
# (ota_sft_100k_v2/census.json, ota_if_sft_v1/ifv2-census.json, rst_sft_v1/census.json).
# The census token columns are char counts divided by these, so they line up with the numbers
# the corpora were staged with without re-tokenising 1.3B tokens on a login node.
CHARS_PER_TOKEN = {"swesmith": 3.773, "superuser": 3.454, "tezos": 3.624,
                   "ifv2": 3.891, "if-v2": 3.891, "if": 3.891, "rst": 3.871}
FALLBACK_CPT = 3.70

# kept-row counts from the survey note, asserted when the source is that corpus
EXPECTED = {"ota3_if_rst_v1": {"fmt": 81889, "beh": 83851}}

RULE_TEXT = {
    "fmt": ("arm 1, malformed action: drop a row if any assistant turn, other than the final turn "
            "of a kind=='compaction' row, has a body after <|end_think|> that does not parse with "
            "json.loads"),
    "beh": ("arm 2, degenerate behaviour: drop a row if any of loop (identical non-empty command "
            "batch 3+ times consecutively), hammer (one keystroke string 20+ times in the trace), "
            "noop (3+ consecutive turns issuing only empty keystrokes), redundant (25%+ of adjacent "
            "turn pairs issue the identical batch), errors (50%+ of terminal observations contain a "
            "shell or Python error)"),
}

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
    """(batch, parsed_ok); batch is the tuple of keystroke strings of a Terminus-2 action."""
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
    """Return (drop_fmt, clauses_fired) for one row, in a single pass over its turns.

    This is the rule code from the 2026-09-20 census scripts unchanged, so the counts reproduce."""
    asst = [m["content"] for m in conv if m["role"] == "assistant"]
    usr = [m["content"] for m in conv if m["role"] == "user"]
    last = len(asst) - 1
    drop_fmt = False
    batches = []
    for j, t in enumerate(asst):
        batch, ok = keystrokes(body(t))
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
    kc: collections.Counter = collections.Counter()
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

    fired = []
    if best_run >= 3:
        fired.append("loop")
    if top_rep >= 20:
        fired.append("hammer")
    if best_noop >= 3:
        fired.append("noop")
    if redun / max(1, pairs) >= 0.25:
        fired.append("redundant")
    if errs / n_obs >= 0.5:
        fired.append("errors")
    return drop_fmt, fired


def train_name_for(out: Path) -> str:
    """ota3_if_rst_fmt_v1 -> ota3-if-rst-fmt-train-00000-of-00001.parquet"""
    corpus = out.name
    if corpus.endswith("_v1"):
        corpus = corpus[: -len("_v1")]
    return f"{corpus.replace('_', '-')}-train-00000-of-00001.parquet"


def find_src_train(src: Path) -> Path:
    cands = [p for p in sorted(src.glob("*-train-*.parquet"))]
    if len(cands) != 1:
        sys.exit(f"expected exactly one *-train-*.parquet in {src}, found {len(cands)}: "
                 + ", ".join(p.name for p in cands))
    return cands[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rule", required=True, choices=("fmt", "beh"))
    ap.add_argument("--src", required=True, type=Path,
                    help="corpus dir holding the single *-train-*.parquet and " + HELD_NAME)
    ap.add_argument("--out", required=True, type=Path, help="output corpus dir; its name sets the shard name")
    ap.add_argument("--expect", type=int, default=None,
                    help="kept-row count to assert; defaults to the survey-note value when the "
                         "source corpus is one it covers")
    a = ap.parse_args()

    src_train = find_src_train(a.src)
    src_held = a.src / HELD_NAME
    if not src_held.exists():
        sys.exit(f"missing held-out file: {src_held}")
    expect = a.expect if a.expect is not None else EXPECTED.get(a.src.name, {}).get(a.rule)

    f = pq.ParquetFile(src_train)
    schema: pa.Schema = f.schema_arrow
    total = f.metadata.num_rows
    a.out.mkdir(parents=True, exist_ok=True)
    train_name = train_name_for(a.out)
    print(f"rule={a.rule}  src={src_train} rows={total} row_groups={f.num_row_groups}", flush=True)
    print(f"out={a.out / train_name}", flush=True)

    per = collections.defaultdict(lambda: {"keep_rows": 0, "keep_chars": 0, "drop_rows": 0, "drop_chars": 0})
    clause_rows: collections.Counter = collections.Counter()
    clause_sole: collections.Counter = collections.Counter()
    writer = pq.ParquetWriter(a.out / train_name, schema, compression="zstd")
    t0, done = time.time(), 0
    for rg in range(f.num_row_groups):
        tbl = f.read_row_group(rg)
        keep_idx = []
        for i, r in enumerate(tbl.to_pylist()):
            sl = r["slice"] or "?"
            chars = sum(len(m["content"]) for m in r["conversations"])
            drop_fmt, fired = judge(r["conversations"], r["kind"])
            for c in fired:
                clause_rows[c] += 1
            if len(fired) == 1:
                clause_sole[fired[0]] += 1
            dropped = drop_fmt if a.rule == "fmt" else bool(fired)
            s = per[sl]
            if dropped:
                s["drop_rows"] += 1
                s["drop_chars"] += chars
            else:
                s["keep_rows"] += 1
                s["keep_chars"] += chars
                keep_idx.append(i)
        if keep_idx:
            writer.write_table(tbl.take(keep_idx), row_group_size=ROW_GROUP)
        done += tbl.num_rows
        if rg % 10 == 0 or done == total:
            print(f"  {done}/{total} rows  {time.time()-t0:.0f}s", flush=True)
    writer.close()

    shutil.copy2(src_held, a.out / HELD_NAME)
    chk = pq.ParquetFile(a.out / train_name)
    assert max(chk.metadata.row_group(i).num_rows for i in range(chk.num_row_groups)) <= ROW_GROUP, \
        "output row groups exceed 1,000 rows"
    sums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(a.out.glob("*.parquet"))}
    (a.out / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in sums.items()))
    (a.out / "parquet.list").write_text(f"{(a.out / train_name).resolve()}\n")

    def tok(chars, sl):
        return int(chars / CHARS_PER_TOKEN.get(sl, FALLBACK_CPT))

    slices = {sl: {"keep_rows": s["keep_rows"], "keep_tokens": tok(s["keep_chars"], sl),
                   "drop_rows": s["drop_rows"], "drop_tokens": tok(s["drop_chars"], sl),
                   "chars_per_token": CHARS_PER_TOKEN.get(sl, FALLBACK_CPT),
                   "ratio_known": sl in CHARS_PER_TOKEN}
              for sl, s in sorted(per.items())}
    keep_rows = sum(v["keep_rows"] for v in slices.values())
    drop_rows = sum(v["drop_rows"] for v in slices.values())
    census = {
        "corpus": a.out.name,
        "rule": a.rule,
        "arm": "1 (malformed action)" if a.rule == "fmt" else "2 (degenerate behaviour)",
        "rule_text": RULE_TEXT[a.rule],
        "source": str(src_train.resolve()),
        "source_rows": total,
        "kept_rows": keep_rows,
        "dropped_rows": drop_rows,
        "kept_tokens": sum(v["keep_tokens"] for v in slices.values()),
        "dropped_tokens": sum(v["drop_tokens"] for v in slices.values()),
        "drop_pct_rows": round(100 * drop_rows / total, 2) if total else 0.0,
        "per_slice": slices,
        "clause_rows_firing": {c: clause_rows.get(c, 0) for c in CLAUSES},
        "clause_only_rule_firing": {c: clause_sole.get(c, 0) for c in CLAUSES},
        "train": {"rows": chk.metadata.num_rows, "row_groups": chk.num_row_groups},
        "heldout": {"rows": pq.ParquetFile(a.out / HELD_NAME).metadata.num_rows,
                    "copied_from": str(src_held.resolve())},
        "token_method": ("char counts divided by each slice's measured chars-per-token, so the token "
                         "columns match the slice build censuses; not re-tokenised"),
        "note": "ai_memory/active/snowball-sft/research/2026-09-20_sft_filtering_survey.md",
        "sha256": sums,
    }
    (a.out / "census.json").write_text(json.dumps(census, indent=1))

    print(f"{a.out.name}: kept {keep_rows} / {total} rows ({census['drop_pct_rows']}% dropped), "
          f"{census['kept_tokens']/1e6:.1f}M tokens, heldout {census['heldout']['rows']} rows", flush=True)
    if a.rule == "beh":
        print("clauses firing:", census["clause_rows_firing"], flush=True)
    if keep_rows != chk.metadata.num_rows:
        sys.exit(f"DEVIATION: counted {keep_rows} kept but wrote {chk.metadata.num_rows}")
    if expect is not None and keep_rows != expect:
        sys.exit(f"DEVIATION: expected {expect} kept rows, got {keep_rows} (delta {keep_rows - expect})")
    print("OK" + (f": matches the expected {expect} kept rows." if expect is not None
                  else ": no expected count for this source, nothing asserted."), flush=True)


if __name__ == "__main__":
    main()
