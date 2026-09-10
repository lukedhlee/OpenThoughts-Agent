#!/usr/bin/env python3
"""banner_spans.py <run> <step> [max_records] — per assistant span (trainable_spans), where does the harness banner sit?
Truncation artifact: one banner per span at ~5,000 bytes in a ~10,000-byte span. Generation: arbitrary offsets, several per span."""
import glob, gzip, json, sys, zipfile, collections
from tokenizers import Tokenizer
run, step = sys.argv[1], int(sys.argv[2]); maxn = int(sys.argv[3]) if len(sys.argv) > 3 else 200
E = "/e/fscratch/reformo/lee27/experiments"; B = "[... output limited to"
tok = Tokenizer.from_file("/e/fscratch/reformo/lee27/models/snowball-s3-nemotron-terminal-step1888/tokenizer.json")
zips = sorted(glob.glob(f"{E}/{run}/{run}/exports/training_trajectories/schema_v3/archives/phase=train/step={step:08d}/*.zip"))
n = 0; spans_total = 0; spans_with = 0; per_span = collections.Counter(); offsets = []; sizes = []; first_examples = []
obs_banner = 0; obs_spans = 0
for zp in zips:
    z = zipfile.ZipFile(zp)
    for name in z.namelist():
        if not name.endswith(".json.gz") or n >= maxn: continue
        rec = json.loads(gzip.decompress(z.read(name))); n += 1
        ids = rec["response"]["token_ids"]; m = rec["response"]["loss_mask"]; spans = rec["response"]["trainable_spans"]
        prev_end = 0
        for sp in spans:
            a, b = sp["start"], sp["end"]
            # observation region between spans
            if a > prev_end:
                ot = tok.decode(ids[prev_end:a], skip_special_tokens=False); obs_spans += 1; obs_banner += ot.count(B)
            prev_end = b
            t = tok.decode(ids[a:b], skip_special_tokens=False); spans_total += 1
            c = t.count(B)
            if c:
                spans_with += 1; per_span[min(c, 5)] += 1
                bb = t.encode("utf-8"); off = t[: t.find(B)].encode("utf-8"); offsets.append(len(off)); sizes.append(len(bb))
                if len(first_examples) < 3 and c == 1:
                    i = t.find(B); first_examples.append(f"{rec['trajectory']['instance_id']} span_bytes={len(bb)} banner_at={len(off)} ctx={t[max(0,i-80):i+70]!r}")
def q(v, p): v = sorted(v); return v[int(p * (len(v) - 1))] if v else None
print(f"run={run} step={step} records={n} assistant_spans={spans_total} spans_with_banner={spans_with} per-span count hist={dict(sorted(per_span.items()))} (5=5+)")
print(f"observation regions={obs_spans} banners in observations={obs_banner}")
if offsets:
    print(f"banner byte offset within span: p10={q(offsets,.1)} p50={q(offsets,.5)} p90={q(offsets,.9)} | span bytes: p10={q(sizes,.1)} p50={q(sizes,.5)} p90={q(sizes,.9)}")
    near5000 = sum(1 for o, s in zip(offsets, sizes) if 4900 <= o <= 5100 and 9900 <= s <= 10300)
    print(f"spans looking like a 10,000-byte truncation (offset≈5000, size≈10k): {near5000}/{spans_with}")
for e in first_examples: print(e[:330])
