#!/usr/bin/env python3
"""mask_audit.py <run> <step> [max_records] — decode dumped training trajectories (schema_v3 archives) and check what the loss mask
covers: fraction of tokens trainable, whether observation text (harness banner, user headers, tmux screen) lies under mask==1,
and, for length-stopped records, how much of the final turn is trainable."""
import glob, gzip, json, sys, zipfile, re
from tokenizers import Tokenizer
run, step = sys.argv[1], int(sys.argv[2]); maxn = int(sys.argv[3]) if len(sys.argv) > 3 else 200
E = "/e/fscratch/reformo/lee27/experiments"
tok = Tokenizer.from_file("/e/fscratch/reformo/lee27/models/snowball-s3-nemotron-terminal-step1888/tokenizer.json")
BANNER = "output limited to"; USER_HDR = "<|start_header_id|>user<|end_header_id|>"; ASST_HDR = "<|start_header_id|>assistant<|end_header_id|>"
zips = sorted(glob.glob(f"{E}/{run}/{run}/exports/training_trajectories/schema_v3/archives/phase=train/step={step:08d}/*.zip"))
n = 0; agg = dict(recs=0, tok=0, m1=0, banner_text=0, banner_masked=0, recs_banner_masked=0, user_hdr_masked=0, asst_hdr_masked=0,
                 length_stops=0, ls_final_turn_tok=0, ls_final_turn_m1=0, spans=0, spans_mismatch=0, eot_end=0)
examples = []
for zp in zips:
    z = zipfile.ZipFile(zp)
    for name in z.namelist():
        if not name.endswith(".json.gz") or n >= maxn: continue
        rec = json.loads(gzip.decompress(z.read(name))); n += 1
        r = rec["response"]; ids = r["token_ids"]; m = r["loss_mask"]; assert len(ids) == len(m)
        agg["recs"] += 1; agg["tok"] += len(ids); agg["m1"] += sum(m); agg["spans"] += len(r["trainable_spans"])
        masked_ids = [t for t, k in zip(ids, m) if k]
        mtext = tok.decode(masked_ids, skip_special_tokens=False)
        full = tok.decode(ids, skip_special_tokens=False)
        bt = full.count(BANNER); bm = mtext.count(BANNER)
        agg["banner_text"] += bt; agg["banner_masked"] += bm; agg["recs_banner_masked"] += int(bm > 0)
        agg["user_hdr_masked"] += mtext.count(USER_HDR); agg["asst_hdr_masked"] += mtext.count(ASST_HDR)
        # spans vs mask consistency: every trainable span should be all-ones
        for sp in r["trainable_spans"]:
            a, b = (sp[0], sp[1]) if isinstance(sp, (list, tuple)) else (sp.get("start"), sp.get("end"))
            if a is not None and b is not None and any(k == 0 for k in m[a:b]): agg["spans_mismatch"] += 1
        # last trainable token
        last1 = max((i for i, k in enumerate(m) if k), default=None)
        if last1 is not None and ids[last1] in (128009, 128001): agg["eot_end"] += 1
        if r["stop_reason"] == "length":
            agg["length_stops"] += 1
            sp = r["trainable_spans"][-1]; a, b = (sp[0], sp[1]) if isinstance(sp, (list, tuple)) else (sp.get("start"), sp.get("end"))
            agg["ls_final_turn_tok"] += (b - a); agg["ls_final_turn_m1"] += sum(m[a:b])
            if len(examples) < 2:
                tail = tok.decode(ids[b - 60:b], skip_special_tokens=False).replace("\n", "\\n")
                examples.append(f"length-stop {rec['trajectory']['instance_id']} final span {a}-{b} masked={sum(m[a:b])}/{b-a} tail: {tail[:200]}")
        if bm > 0 and len(examples) < 4:
            i = mtext.find(BANNER); examples.append(f"BANNER UNDER MASK {rec['trajectory']['instance_id']}: ...{mtext[max(0,i-120):i+80]!r}")
print(f"run={run} step={step} records={agg['recs']} tokens={agg['tok']} mask1_frac={agg['m1']/max(1,agg['tok']):.3f} spans/rec={agg['spans']/max(1,agg['recs']):.1f} span_mask_mismatch={agg['spans_mismatch']}")
print(f"banner: in text {agg['banner_text']} | under mask {agg['banner_masked']} (records {agg['recs_banner_masked']}) | user_hdr under mask {agg['user_hdr_masked']} | asst_hdr under mask {agg['asst_hdr_masked']} | records whose last trainable token is eot/eos {agg['eot_end']}")
if agg["length_stops"]:
    print(f"length_stops={agg['length_stops']} final-turn tokens {agg['ls_final_turn_tok']} trainable {agg['ls_final_turn_m1']}")
print("first record spans sample:", json.dumps(rec["response"]["trainable_spans"][:3])[:200], "step_boundaries:", str(rec["response"]["step_boundaries"])[:100])
for e in examples: print(e[:400])
